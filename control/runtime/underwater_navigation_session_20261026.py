"""单采集、单 PWM 线程的水下导航会话。阶段转换和滤波均使用采集时间。"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data_logger import JsonlLogger, RawSensorLoggers, write_metadata_yaml, prepare_raw_placeholders
from .fixed_dive_20260920_100122 import FixedDivePolicy, FixedDepthCalibration
from .pressure_drift_20261001 import PressureDrift, ObservedPressureBuffer, CorrectedSensors
from .rl_actions_20260914 import build_calibration
from .rl_safety_20260914 import RLSafetyMonitor
from .rl_sensor_calibration_20260914 import RLSensorCalibration
from .rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from .rl_servo_state_tracker_20260914 import RLServoStateTracker
from .sensor_manager import SensorManager
from .sensor_worker import BaseSensorWorker
from .surface_navigation_20261010 import CalibratedIMU, NavigationPolicy, NavigationTailAction, UWBStreamDecoder
from .underwater_navigation_20261026 import UnderwaterScheduler, InertialEstimator, StableUWB, UWBObservation, forward_tid


def calibrate_window(cfg,imus,depths,end_ns):
    """整个固定 t2 窗口逐批积累，检查数量、跨度、间隔和最后数据年龄。"""
    minima={}
    for name,rows,hz,age,gap in (('imu',imus,cfg.imu_rate_hz,cfg.imu_timeout_s,cfg.imu_gap_s),
                                ('depth',depths,cfg.pressure_rate_hz,cfg.pressure_timeout_s,cfg.pressure_timeout_s)):
        minimum=max(2,math.ceil(cfg.calibration_sample_ratio*cfg.t2*hz)); minima[name]=minimum
        if (len(rows)<minimum or rows[-1].t_ns-rows[0].t_ns<cfg.calibration_coverage_ratio*cfg.t2*1e9
                or end_ns-rows[-1].t_ns>age*1e9
                or any(b.t_ns-a.t_ns<=0 or b.t_ns-a.t_ns>gap*1e9 for a,b in zip(rows,rows[1:]))):
            raise ValueError(f'{name} calibration coverage/count/gap invalid ({len(rows)}/{minimum})')
    calibration=RLSensorCalibration(minimum_imu_samples=minima['imu'],minimum_depth_samples=minima['depth'],
        max_angle_std_deg=cfg.calibration_max_angle_std_deg,max_pressure_std_pa=cfg.calibration_max_pressure_std_pa,
        max_linear_acc_rms_mps2=cfg.calibration_max_linear_acc_rms_mps2,max_acc_std_mps2=cfg.calibration_max_acc_std_mps2,
        max_gyro_radps=cfg.calibration_max_gyro_radps,max_quat_euler_tilt_error_deg=cfg.calibration_max_quat_euler_tilt_error_deg,
        gravity_mps2=cfg.gravity_mps2,water_density_kg_m3=cfg.water_density_kg_m3)
    calibration.fit(imus,depths)
    if any('gyro_radps' not in s.data for s in imus): raise ValueError('calibration requires gyro_radps')
    bias=np.mean([calibration.linear_acceleration(s.data) for s in imus],axis=0)
    yaw=CalibratedIMU(cfg.yaw_tau_s,minima['imu'],cfg.calibration_max_angle_std_deg,cfg.navigation.yaw_sign)
    yaw.calibrate([s.data for s in imus])
    return calibration,bias,yaw


def check_camera_health(sensors,*,now_ns,started_ns,timeout_s=.3,ready_timeout_s=5.,shutdown=False):
    """真实存图任务必须有新图像；关闭后检查驱动异常及残留写图线程。"""
    fatal=sensors.fatal_errors()
    if fatal: raise RuntimeError(f'camera/sensor worker failure: {fatal}')
    if not shutdown:
        sample=sensors.buffers['vision'].latest()
        if sample is None:
            if now_ns-started_ns>ready_timeout_s*1e9: raise RuntimeError('camera has no first frame')
        elif not sample.ok or sample.data.get('image_save_error'):
            raise RuntimeError(f'camera invalid: {sample.error or sample.data.get("image_save_error")}')
        elif not 0<=now_ns-sample.t_ns<=timeout_s*1e9:
            raise RuntimeError('camera frame stale')
    for worker in sensors.workers:
        writer=getattr(worker,'_image_writer',None) or getattr(worker,'_closed_image_writer',None)
        if writer is None: continue
        if (writer.last_error or writer.dropped_frames or writer.index_logger.last_error
                or writer.index_logger.dropped_count): raise RuntimeError('camera image save failed / queue overflow')
        if shutdown and ((writer._thread is not None and writer._thread.is_alive())
                         or writer.index_logger.is_alive()):
            raise RuntimeError('camera writer thread did not stop')


class NavigationMission:
    """非阻塞控制阶段。只有 PWM 完成回执才能推进半程；每组最多一个在途请求。"""
    def __init__(self,cfg,calibration,executor,event):
        self.cfg,self.calibration,self.ex,self.event=cfg,calibration,executor,event
        self.phase='initial_fix'; self.stage_start=0; self.motion_start=None; self.segment_start=None
        self.depth=0.; self.depth_t=None; self.depth_policy=self._depth_policy(cfg.h0)
        self.nav_policy=NavigationPolicy(cfg.navigation,calibration)
        self.inflight={'tail':None,'fin':None}; self.epoch=0
        self.tail_side=0; self.tail_mode=None; self.tail_held=False; self.tail_ticket=None
        self.center_ticket=None; self.center_purpose=None; self.center_requested_ns=None
        self.pressure_paused=False; self.resume_count=0; self.reason=None
        self.signal_gate=StableUWB(cfg)

    def _depth_policy(self,target):
        last=self.ex.scheduler.fin_sequence.last_b2
        return FixedDivePolicy(self.calibration,h0=target,k=self.cfg.depth_kp,ki=self.cfg.depth_ki,
             initial_b2=-last if last else self.cfg.initial_b2,action_duration_s=self.cfg.fin_duration_s,
             error_scale_m=self.cfg.h0,max_integral_gap_s=self.cfg.pressure_timeout_s)

    def enter_navigation(self,now):
        self.phase='navigate'; self.stage_start=self.segment_start=now
        if self.motion_start is None: self.motion_start=now
        self.depth_policy=self._depth_policy(self.cfg.h0); self.depth_policy.start_control(now)
        self.nav_policy=NavigationPolicy(self.cfg.navigation,self.calibration)
        self.ex.resume_agents(('tail','fin')); self.epoch+=1
        self.inflight={'tail':None,'fin':None}; self.tail_side=0; self.tail_mode=None
        self.tail_held=False; self.pressure_paused=False; self.resume_count=0
        self.event('navigation_started',segment_start_t_ns=now,motion_start_t_ns=self.motion_start)

    def on_depth(self,t,depth):
        if self.depth_t is not None and t<=self.depth_t: return
        self.depth_t=t; self.depth=depth
        control_depth=max(depth,0.) if self.phase=='surface' else depth
        self.depth_policy.observe_depth(control_depth,t)
        if self.pressure_paused:
            self.resume_count=self.resume_count+1 if depth>self.cfg.resume_depth_m else 0

    def _center(self,now,purpose):
        if self.center_ticket is None:
            self.center_ticket=self.ex.request_hold(agents=('tail','fin'),center=True,block=True)
            self.center_requested_ns=now
        self.center_purpose=purpose

    def on_uwb(self,obs,now):
        if self.phase=='surface' and self.signal_gate.signal(obs,now): self._center(now,'acquire')

    def _completions(self):
        for item in self.ex.drain_completed():
            if item.request_id!=self.inflight[item.agent] or item.context!=self.epoch:
                self.event('late_completion_ignored',request_id=item.request_id,agent=item.agent)
                continue
            self.inflight[item.agent]=None
            if item.agent=='fin': self.depth_policy.acknowledge(item.action)
            else: self.tail_side=1-self.tail_side
            self.event('action_completed',agent=item.agent,request_id=item.request_id,
                       action=asdict(item.action),completion_t_ns=item.completion_t_ns)

    def _submit(self,agent,action):
        self.inflight[agent]=self.ex.submit(agent,action,context=self.epoch)
        self.event('action_submitted',agent=agent,request_id=self.inflight[agent],action=asdict(action))

    def tick(self,now,xyz,yaw):
        if self.reason: return
        if self.motion_start is not None and now-self.motion_start>=self.cfg.t3*1e9:
            self.reason='time_limit'; self.ex.request_center_stop(); return
        self._completions()
        if self.tail_ticket is not None:
            if self.tail_ticket.failure: raise RuntimeError(str(self.tail_ticket.failure))
            if self.tail_ticket.is_set():
                self.inflight['tail']=None; self.tail_ticket=None; self.tail_held=True
        if self.center_ticket is not None:
            ticket=self.center_ticket
            if ticket.failure: raise RuntimeError(str(ticket.failure))
            if not ticket.is_set():
                if now-self.center_requested_ns>self.cfg.center_ack_timeout_s*1e9:
                    raise TimeoutError('PWM center acknowledgement timeout')
                return
            self.inflight={'tail':None,'fin':None}; self.epoch+=1
            self.center_ticket=None; self.tail_ticket=None; self.tail_held=True
            if self.center_purpose=='acquire':
                self.phase='acquire'; self.stage_start=ticket.ack_t_ns
                self.event('uwb_acquire_started',center_ack_t_ns=ticket.ack_t_ns)
                return
            self.pressure_paused=True; self.resume_count=0
            self.depth_policy=self._depth_policy(0)
            self.event('surface_pressure_paused',center_ack_t_ns=ticket.ack_t_ns)
        if self.phase=='navigate' and now-self.segment_start>=self.cfg.t4*1e9:
            self.phase='surface'; self.stage_start=now; self.signal_gate.reset(now)
            self.tail_ticket=self.ex.request_hold(agents=('tail',),center=True,block=True)
            self.event('surface_started',depth_m=self.depth)
        if self.phase=='acquire':
            if now-self.stage_start>=self.cfg.fix_timeout_s*1e9: self.reason='fix_timeout'
            return
        if self.phase=='surface':
            if now-self.stage_start>=self.cfg.surface_timeout_s*1e9:
                self.reason='surface_timeout'; return
            if self.pressure_paused:
                if self.resume_count<self.cfg.resume_depth_samples: return
                self.pressure_paused=False; self.resume_count=0
                self.depth_policy=self._depth_policy(0); self.depth_policy.start_control(now)
                self.ex.resume_agents(('fin',)); self.event('surface_pressure_resumed')
            if not self.depth_policy.pair_active:
                if self.depth_policy.h0!=0:
                    self.depth_policy.set_target(0,error_scale_m=self.cfg.h0)
                    self.depth_policy.start_control(now)
                if self.depth<=0:
                    self._center(now,'pressure'); return
        if self.phase not in ('navigate','surface'): return
        if self.inflight['fin'] is None:
            action=self.depth_policy.next_action(max(self.depth,0) if self.phase=='surface' else self.depth)
            if action is not None: self._submit('fin',action)
        if self.phase!='navigate': return
        decision=self.nav_policy.decide(xyz[0],xyz[1],yaw,now)
        if decision.reached:
            if not self.tail_held and self.tail_ticket is None:
                self.tail_ticket=self.ex.request_hold(agents=('tail',),center=True,block=True)
                self.event('estimated_target_reached',distance_m=decision.distance_m)
            return
        if self.tail_ticket is not None: return
        if self.tail_held:
            self.ex.resume_agents(('tail',)); self.tail_held=False; self.tail_side=0
        if self.inflight['tail'] is None:
            if self.tail_mode!=decision.control_mode: self.tail_side=0
            self.tail_mode=decision.control_mode
            theta=decision.theta_left if self.tail_side==0 else decision.theta_right
            self._submit('tail',NavigationTailAction(theta,self.cfg.navigation.t_0))


class _UnderwaterMockWorker(BaseSensorWorker):
    """模拟传感器时序/失锁，仅用于软件流程验证，不模拟水动力或导航准确度。"""
    def __init__(self,name,manager):
        super().__init__(name,manager.buffers[name],stop_event=manager.stop_event,
                         loop_delay_s={'imu':1/manager.cfg.imu_rate_hz,'depth':1/manager.cfg.pressure_rate_hz,
                            'power':1/float(manager.config['sensors']['power']['rate_hz']),'uwb':.1}[name])
        self.manager=manager
        self.next_imu_ns=None

    def read_once(self):
        m=self.manager; now=time.monotonic_ns(); depth=m.sim_depth()
        if self.name=='imu':
            # Windows 等待精度可能为 15.6 ms；模拟 UART 一次读到多个 200 Hz 帧。
            if self.next_imu_ns is None: self.next_imu_ns=now
            samples=[]
            while self.next_imu_ns<=now:
                samples.append(self.make_sample(dict(acc_mps2=[0,0,m.cfg.gravity_mps2],quat=[1,0,0,0],gyro_radps=[0,0,0],
                    yaw_deg=0.,pitch_deg=0.,roll_deg=0.,tid=(self._seq+1)%65536),t_ns=self.next_imu_ns))
                self.next_imu_ns+=round(1e9/m.cfg.imu_rate_hz)
            return samples
        if self.name=='depth':
            sample=self.make_sample(dict(pressure_pa=101325.+depth*m.cfg.gravity_mps2*m.cfg.water_density_kg_m3),t_ns=now)
            if m.drift is not None:
                m.drift.observe([sample]); sample.data['pressure_pa']+=m.drift.drift_pa(m.drift.elapsed_s(now))
            return sample
        if self.name=='power': return self.make_sample(dict(voltage_v=12.,current_a=.5,power_w=6.),t_ns=now)
        if m.disable_uwb or depth>.02: return self.make_sample(dict(raw_hex=b'RANGE_ERROR\r\n'.hex()),t_ns=now)
        c=m.cfg.navigation; x,y=.25*c.L,.5*c.W
        ranges=[round(math.sqrt((x-a)**2+(y-b)**2+(c.H+depth)**2)*1000)
                for a,b in ((0,0),(c.L,0),(c.L,c.W),(0,c.W))]
        packet=('mc 00 '+' '.join(f'{r:08x}' for r in ranges)+
                f' 4705 {self._seq%256:02x} {self._seq:08x} {c.uwb_tag} 0000\r\n')
        return self.make_sample(dict(raw_hex=packet.encode().hex()),t_ns=now)


class UnderwaterMockSensors(SensorManager):
    def __init__(self,robot,cfg,stop,drift):
        super().__init__(robot,mock=True,stop_event=stop)
        self.cfg,self.drift,self.mission=cfg,drift,None
        self.disable_uwb=False

    def sim_depth(self):
        if self.mission is None: return 0.
        m=self.mission
        if m.phase=='navigate': return min(self.cfg.h0,(time.monotonic_ns()-m.segment_start)/1e9*.2)
        if m.phase=='surface': return max(0.,min(self.cfg.h0,self.cfg.t4*.2)-(time.monotonic_ns()-m.stage_start)/1e9*.3)
        return 0.

    def _create_mock_workers(self):
        self.workers=[_UnderwaterMockWorker(name,self) for name in ('imu','depth','power','uwb')]


def run_session(robot,cfg,session,*,dry_run=False,no_servo=False,stop_event=None,sensors=None,controller=None):
    if dry_run and no_servo: raise ValueError('dry-run 与 no-servo 不能同时使用')
    cal=build_calibration(robot)
    if cfg.h0>=float(robot['safety']['max_depth_m']): raise ValueError('h0 必须低于安全深度')
    stop=stop_event or threading.Event(); session=Path(session); session.mkdir(parents=True,exist_ok=True)
    prepare_raw_placeholders(session)
    drift=PressureDrift(Path(__file__).resolve().parents[2]/cfg.pressure_drift_file)
    if cfg.pressure_rate_hz!=float(drift.config['sample_rate_hz']):
        raise ValueError('压力采样率与 pressure_drift_file 的 sample_rate_hz 不一致')
    sensor_cfg=copy.deepcopy(robot)
    for name in ('imu','depth','power','uwb'):
        if sensor_cfg['sensors'][name]['enabled'] is not True:
            raise ValueError(f'水下导航要求 robot.yaml sensors.{name}.enabled=true')
    camera_enabled=sensor_cfg['sensors']['vision']['enabled']
    if type(camera_enabled) is not bool or type(sensor_cfg['sensors']['vision']['save_frames']) is not bool:
        raise ValueError('sensors.vision.enabled/save_frames 必须为布尔值')
    if sensors is None:
        sensors=(UnderwaterMockSensors(sensor_cfg,cfg,stop,drift) if dry_run else
                 SensorManager(sensor_cfg,mock=False,stop_event=stop,log_dir=session))
    if isinstance(sensors,UnderwaterMockSensors): sensors.drift=drift
    # 在启动 worker 前挂接，首压力样本不会因 t1 较长被环形缓冲覆盖。
    sensors.buffers['depth']=ObservedPressureBuffer(sensors.buffers['depth'],drift)
    corrected=CorrectedSensors(sensors,drift)
    if controller is None:
        if dry_run: controller=DryRunRLServoController(cal.centers)
        else:
            def controller():
                from drivers.pca9685_servo import PCA9685ServoController
                return PCA9685ServoController(robot)
    failures=queue.Queue()
    ex=RLServoExecutor(controller,UnderwaterScheduler(cal),RLServoStateTracker(cal),
        # 归中是阶段约定的直接写中位；运动时序/PWM频率读取本次配置。
        config={'command_hz':cfg.command_hz,'safe_recenter_s':0.,'initial_move_s':0.,'hold_at_center':True},
        shutdown_event=stop,failure_queue=failures,
        servo_id_to_channel={i:int(v['channel']) for i,v in cal.servos.items()})
    safety={'runtime':{'safety_poll_interval_s':cfg.safety_poll_s,'sensor_ready_timeout_s':cfg.ready_timeout_s},
        'state':{'enforce_data_freshness':True,'sensor_age':{
        'imu':{'fault_ms':cfg.imu_timeout_s*1000},'depth':{'fault_ms':cfg.pressure_timeout_s*1000},
        'power':{'fault_ms':cfg.power_timeout_s*1000}}}}
    guard=RLSafetyMonitor(corrected,ex,failures,stop,safety,robot)
    log_options={'flush_interval_s':cfg.log_flush_s,'queue_maxsize':cfg.log_queue_size}
    logs={n:JsonlLogger(session/(n+'.jsonl'),**log_options) for n in ('events','commands','navigation','imu_filtered')}
    raw=RawSensorLoggers(session,**log_options); phase='placement'; estimator=None; heading=None; depth_cal=None
    mission=None; executor_started=False; timer=None; deadline=threading.Event(); terminal_ticket=None
    reason='stopped'; fault=None; cleanup_errors=[]; start=time.monotonic_ns()
    calibration_start=start+round(cfg.t1*1e9); calibration_end=calibration_start+round(cfg.t2*1e9)
    initial_deadline=calibration_end+round(cfg.fix_timeout_s*1e9); initial_centered=False
    gate=StableUWB(cfg); decoder=UWBStreamDecoder(cfg.navigation.uwb_tag)
    last_seq={n:None for n in ('imu','depth','uwb')}; pending=[]; last_processed=-1
    calibration_rows={'imu':[],'depth':[]}; last_tid=None
    decoded_ids=deque(maxlen=512)
    after_calibration=[]
    recent_imus=deque(); recent_depths=deque(); last_status=0; gate_phase=None; diagnostic_start=None
    last_range_ns=None; valid_ranges=0; last_complete_uwb_ns=None

    def write(name,**data):
        current_phase=mission.phase if mission is not None else phase
        if not logs[name].write(dict(t_ns=time.monotonic_ns(),phase=current_phase,**data)):
            raise RuntimeError(f'{name} 日志队列满')

    def event(kind,**data):
        write('events',event=kind,**data)
        if kind.startswith('action_'): write('commands',event=kind,**data)

    def expire():
        nonlocal terminal_ticket
        deadline.set()
        if executor_started:
            guard.executor_started=False
            try: terminal_ticket=ex.request_center_stop()
            except RuntimeError:
                if not stop.is_set() and ex.failure is None: raise

    def arm_timer():
        nonlocal timer
        timer=threading.Timer(cfg.t3,expire); timer.daemon=True; timer.start()

    def log_health():
        for logger in [*logs.values(),*raw.loggers.values()]:
            if logger.last_error or logger.dropped_count:
                raise RuntimeError(f'日志失败 {logger.path}: {logger.last_error}, dropped={logger.dropped_count}')
        if camera_enabled and not isinstance(sensors,UnderwaterMockSensors):
            check_camera_health(sensors,now_ns=time.monotonic_ns(),started_ns=start,
                timeout_s=float(sensor_cfg['sensors']['vision']['timeout_ms'])/1000,
                ready_timeout_s=cfg.ready_timeout_s)

    def update_bias(fix,now):
        if (not cfg.auto_bias or no_servo or not initial_centered
                or phase not in ('initial_fix','acquire')
                or np.linalg.norm(fix.velocity)>=cfg.bias_speed_mps
                or fix.quality['max_half_speed_mps']>=cfg.bias_speed_mps
                or fix.quality['span_s']<.9*cfg.bias_window_s): return
        threshold=fix.t_ns-round(cfg.bias_window_s*1e9)
        if threshold<gate.boundary: return  # 零偏窗口必须完全位于归中静置之后。
        ims=[s for s in recent_imus if threshold<=s.t_ns<=fix.t_ns]
        ds=[d for t,d in recent_depths if threshold<=t<=fix.t_ns]
        if (len(ims)<cfg.calibration_sample_ratio*cfg.bias_window_s*cfg.imu_rate_hz
                or len(ds)<cfg.calibration_sample_ratio*cfg.bias_window_s*cfg.pressure_rate_hz
                or not ims or ims[-1].t_ns-ims[0].t_ns<cfg.calibration_coverage_ratio*cfg.bias_window_s*1e9): return
        residual=np.array([sensor_cal.linear_acceleration(s.data) for s in ims])
        if (max(np.linalg.norm(s.data['gyro_radps']) for s in ims)>=cfg.bias_gyro_radps
                or max(np.std(residual,axis=0))>=cfg.bias_acc_std_mps2 or np.std(ds)>=cfg.bias_depth_std_m): return
        delta=np.clip(cfg.bias_gain*(np.mean(residual,axis=0)-estimator.bias),-cfg.bias_step_mps2,cfg.bias_step_mps2)
        estimator.bias+=delta
        event('static_bias_updated',delta=delta.tolist(),b_body=estimator.bias.tolist())

    def apply_fix(fix,now):
        try: innovation=estimator.correct(fix.t_ns,fix.xy,fix.velocity)
        except ValueError as exc:
            event('fix_rejected',detail=str(exc)); return False
        event('position_corrected',fix_t_ns=fix.t_ns,xy=fix.xy,velocity=fix.velocity,
              innovation=innovation,quality=fix.quality,age_s=(now-fix.t_ns)/1e9)
        update_bias(fix,now)
        return True

    try:
        write_metadata_yaml(session/'metadata.yaml',{'task':'underwater_navigation_20261026','config':asdict(cfg),
            'robot':sensor_cfg,'dry_run':dry_run,'no_servo':no_servo,'z_positive':'down',
            'camera_note':'dry-run 不产生真实相机图像' if dry_run else '相机开关/存图按 sensors.vision 配置'})
        (session/'pressure_drift.json').write_text(json.dumps(drift.config,ensure_ascii=False),encoding='utf8')
        raw.start()
        # 配置/日志初始化不占人工放置与静止标定预算。
        start=time.monotonic_ns()
        calibration_start=start+round(cfg.t1*1e9); calibration_end=calibration_start+round(cfg.t2*1e9)
        initial_deadline=calibration_end+round(cfg.fix_timeout_s*1e9)
        sensors.start_all(); guard.start(); event('placement_started')
        while True:
            now=time.monotonic_ns()
            guard.raise_if_failed()
            if deadline.is_set(): reason='time_limit'; break
            if stop.is_set(): raise InterruptedError('收到外部停止请求')
            raw.write_from_buffers(sensors.buffers); log_health()
            if now>=calibration_start and phase=='placement': phase='calibration'; event('calibration_started')
            for name in last_seq:
                for sample in sensors.buffers[name].snapshot():
                    previous=last_seq[name]
                    if previous is not None and sample.seq<=previous: continue
                    if previous is not None and sample.seq!=previous+1:
                        event('buffer_sequence_gap',sensor=name,previous=previous,current=sample.seq)
                        if name=='uwb': decoder.reset_partial()
                    last_seq[name]=sample.seq
                    if name=='imu' and sample.ok:
                        tid=sample.data.get('tid')
                        if not forward_tid(last_tid,tid):
                            event('imu_duplicate_or_backward',previous=last_tid,current=tid)
                            continue
                        if tid is not None and last_tid is not None and (int(tid)-int(last_tid))%65536!=1:
                            event('imu_tid_gap',previous=last_tid,current=tid)
                        last_tid=tid
                    if name=='depth': sample=drift.correct(sample)
                    pending.append(sample)
            # 按配置留出合并余量，再按采集时间处理 UART/I²C worker 的批次。
            pending.sort(key=lambda s:(s.t_ns,0 if s.name=='depth' else 1))
            ready=[s for s in pending if s.t_ns<=now-round(cfg.merge_delay_s*1e9)]
            pending=[s for s in pending if s.t_ns>now-round(cfg.merge_delay_s*1e9)]
            fixes=[]
            for sample in ready:
                name,t,data=sample.name,sample.t_ns,sample.data
                if t<last_processed:
                    event('late_sample_ignored',sensor=name,sample_t_ns=t); continue
                last_processed=t
                if not sample.ok:
                    if name!='uwb': raise ValueError(f'{name}: {sample.error}')
                    continue
                if calibration_start<=t<=calibration_end and name in calibration_rows:
                    calibration_rows[name].append(sample)
                if estimator is None:
                    if t>calibration_end: after_calibration.append(sample)
                    continue
                if name=='imu':
                    if estimator.imu(t,data):
                        heading.update(data,t); recent_imus.append(sample)
                        write('imu_filtered',sample_t_ns=t,xyz=estimator.xyz,velocity=estimator.v.tolist(),
                              raw_linear_acc=estimator.raw_acc.tolist(),filtered_acc=estimator.filtered_acc,
                              b_body=estimator.bias.tolist(),yaw_deg=heading.values['yaw_deg'])
                elif name=='depth':
                    depth=depth_cal.depth_m(data['pressure_pa']); estimator.pressure(t,depth)
                    recent_depths.append((t,depth))
                    if mission: mission.on_depth(t,depth)
                elif name=='uwb':
                    raw_bytes=bytes.fromhex(data.get('raw_hex',''))
                    for frame in decoder.feed(raw_bytes,include_incomplete=True):
                        if frame.frame_id in decoded_ids: continue
                        decoded_ids.append(frame.frame_id)
                        last_range_ns=t; valid_ranges=sum(v is not None for v in frame.ranges_m)
                        if valid_ranges==4: last_complete_uwb_ns=t
                        obs=UWBObservation(t,frame,estimator.depth_at(t))
                        if mission and not no_servo: mission.on_uwb(obs,now)
                        if phase in ('initial_fix','acquire','diagnostic') and t>gate.boundary:
                            fix=gate.add(obs,now)
                            if fix is not None: fixes.append(fix)
                            write('navigation',event='uwb_quality',sample_t_ns=t,frame_id=frame.frame_id,
                                  valid_ranges=sum(v is not None for v in frame.ranges_m),reason=gate.reason,
                                  quality=gate.quality)
            while recent_imus and recent_imus[0].t_ns<now-round(cfg.history_s*1e9): recent_imus.popleft()
            while recent_depths and recent_depths[0][0]<now-round(cfg.history_s*1e9): recent_depths.popleft()
            if estimator is None and now>=calibration_end+round((cfg.merge_delay_s+cfg.control_poll_s)*1e9):
                sensor_cal,bias,heading=calibrate_window(cfg,calibration_rows['imu'],calibration_rows['depth'],calibration_end)
                depth_cal=FixedDepthCalibration(sensor_cal,sensor_cal.water_density_kg_m3*sensor_cal.gravity_mps2,drift)
                guard.calibration=depth_cal
                estimator=InertialEstimator(cfg,bias,heading.zeros['yaw_deg'],sensor_cal.gravity_mps2)
                # 标定末端给滤波器起点，后面的样本都逐条传播。
                last_i=calibration_rows['imu'][-1]; last_d=calibration_rows['depth'][-1]
                for s in sorted((last_i,last_d),key=lambda s:s.t_ns):
                    if s.name=='imu': estimator.imu(s.t_ns,s.data); heading.update(s.data,s.t_ns)
                    else: estimator.pressure(s.t_ns,depth_cal.depth_m(s.data['pressure_pa']))
                state={**depth_cal.state_dict(),'b_body':bias.tolist(),'heading':heading.diagnostics}
                (session/'calibration.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
                event('calibration_completed',imu_samples=len(calibration_rows['imu']),depth_samples=len(calibration_rows['depth']))
                # 在标定完成前已经排出的窗口后样本重新排队，避免漏积分。
                pending.extend(after_calibration); after_calibration.clear()
                last_processed=calibration_end
                phase='diagnostic' if no_servo else 'initial_fix'
                gate.reset(calibration_end); decoder.reset_partial()
                if no_servo: diagnostic_start=now; arm_timer()
            if estimator is None:
                stop.wait(cfg.control_poll_s); continue
            if now-estimator.last_imu>cfg.imu_timeout_s*1e9:
                raise ValueError('IMU accepted samples stale')
            if estimator.last_depth is None or now-estimator.last_depth>cfg.pressure_timeout_s*1e9:
                raise ValueError('pressure accepted samples stale')
            if phase=='initial_fix':
                if now>=initial_deadline: reason='initial_fix_timeout'; break
                if fixes and not executor_started:
                    ex.start(); executor_started=True
                    event('pwm_start_requested')
                if executor_started and not initial_centered and ex.wait_ready(0):
                    guard.executor_started=True; initial_centered=True
                    boundary=ex.last_write_ack_t_ns+round(cfg.settle_s*1e9)
                    gate.reset(boundary); decoder.reset_partial(); fixes=[]
                    event('initial_center_confirmed',center_ack_t_ns=ex.last_write_ack_t_ns)
                if initial_centered and fixes and apply_fix(fixes[-1],now):
                    mission=NavigationMission(cfg,cal,ex,event)
                    latest=corrected.buffers['depth'].latest()
                    mission.on_depth(latest.t_ns,depth_cal.depth_m(latest.data['pressure_pa']))
                    mission.enter_navigation(now); arm_timer(); phase=mission.phase
                    if isinstance(sensors,UnderwaterMockSensors): sensors.mission=mission
            elif phase=='diagnostic':
                if fixes and apply_fix(fixes[-1],now): gate.reset(fixes[-1].t_ns)
            elif mission:
                if phase=='acquire' and fixes and apply_fix(fixes[-1],now): mission.enter_navigation(now)
                if now-mission.motion_start>=cfg.t3*1e9:
                    # 与独立定时器使用同一正常停止入口，避免监测线程误报预期退出。
                    expire(); reason='time_limit'; break
                try: mission.tick(now,estimator.xyz,heading.values['yaw_deg'])
                except RuntimeError:
                    if deadline.is_set(): reason='time_limit'; break
                    raise
                phase=mission.phase
                if phase=='acquire' and gate_phase!=mission.stage_start:
                    gate_phase=mission.stage_start
                    gate.reset(mission.stage_start+round(cfg.settle_s*1e9)); decoder.reset_partial()
                    recent_imus.clear(); recent_depths.clear()
                if mission.reason: reason=mission.reason; break
            if now-last_status>=1e9/cfg.status_hz:
                last_status=now
                remaining=None if timer is None else max(0,cfg.t3-(now-(diagnostic_start if no_servo else mission.motion_start))/1e9)
                next_surface=None if not mission or phase!='navigate' else max(0,cfg.t4-(now-mission.segment_start)/1e9)
                distance=math.hypot(estimator.xyz[0]-cfg.navigation.x_d,estimator.xyz[1]-cfg.navigation.y_d)
                depth_target=mission.depth_policy.h0 if mission is not None else None
                fresh_ranges=valid_ranges if last_range_ns is not None and now-last_range_ns<=cfg.uwb_age_s*1e9 else 0
                uwb_reason=gate.reason if phase in ('initial_fix','acquire','diagnostic') else (
                    'waiting_surface_signal' if phase=='surface' else 'dead_reckoning')
                write('navigation',event='status',xyz=estimator.xyz,velocity=estimator.v.tolist(),
                      target=[cfg.navigation.x_d,cfg.navigation.y_d,cfg.h0],t3_remaining_s=remaining,
                      t4_remaining_s=next_surface,uwb_reason=uwb_reason,distance_m=distance,
                      depth_target_m=depth_target,valid_ranges=fresh_ranges,
                      depth_t_ns=estimator.last_depth,last_complete_uwb_t_ns=last_complete_uwb_ns)
                print(f'[{phase}] XYZ={tuple(round(v,3) for v in estimator.xyz)} '
                      f'd={distance:.3f}m h0={depth_target} t3={remaining} t4={next_surface} '
                      f'UWB={fresh_ranges}/4 {uwb_reason}',flush=True)
            stop.wait(cfg.control_poll_s)
    except (Exception,KeyboardInterrupt) as exc:
        reason='stopped' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'fault'
        fault=str(guard.failure or exc)
        ex.request_emergency_stop()
    finally:
        if timer: timer.cancel(); timer.join(timeout=cfg.shutdown_timeout_s)
        guard.executor_started=False
        try:
            if executor_started and fault is None and ex.is_alive():
                terminal_ticket=ex.request_center_stop()
                if not terminal_ticket.wait(cfg.center_ack_timeout_s) or terminal_ticket.failure:
                    raise RuntimeError(f'停止归中未完成: {terminal_ticket.failure}')
            if executor_started:
                ex.join(cfg.shutdown_timeout_s)
                if ex.is_alive(): ex.request_emergency_stop(); raise RuntimeError('PWM 线程未按时退出')
            if ex.failure and str(ex.failure)!='emergency stop: angle writes cancelled': raise RuntimeError(str(ex.failure))
        except Exception as exc: cleanup_errors.append(str(exc)); ex.request_emergency_stop()
        try: guard.stop(timeout=cfg.shutdown_timeout_s)
        except Exception as exc: cleanup_errors.append(str(exc))
        # 截止事件不能覆盖同时发生的低压/过流/采集故障；停止监测后统一收集。
        if guard.failure is not None:
            fault=str(guard.failure)
            if reason!='stopped': reason='fault'
        # 执行器只保存首个异常；关闭 PWM 的后续失败也必须逐项保留。
        while True:
            try: reported=failures.get_nowait()
            except queue.Empty: break
            if (reported.get('fault_code')=='SERVO_EXECUTOR_FAILED'
                    and reported.get('message')=='emergency stop: angle writes cancelled'): continue
            cleanup_errors.append(f"{reported.get('fault_code')}: {reported.get('message')}")
        try:
            sensors.stop_all(timeout=cfg.shutdown_timeout_s); raw.write_from_buffers(sensors.buffers)
            check_camera_health(sensors,now_ns=time.monotonic_ns(),started_ns=start,shutdown=True)
        except Exception as exc: cleanup_errors.append(str(exc))
        try: event('session_finished',reason=reason,fault=fault,cleanup_errors=cleanup_errors)
        except Exception as exc: cleanup_errors.append(str(exc))
        for logger in [*logs.values(),*raw.loggers.values()]:
            logger.stop(timeout=cfg.shutdown_timeout_s)
            if logger.is_alive() or logger.last_error: cleanup_errors.append(f'logger close: {logger.path}: {logger.last_error}')
    alive=sensors.alive_worker_names()
    if alive: cleanup_errors.append(f'workers still alive: {alive}')
    result=dict(reason=reason,fault=fault,cleanup_errors=cleanup_errors,session=str(session),
                executor_started=executor_started,alive_workers=alive,pressure_origin_t_ns=drift.origin_t_ns,
                xyz=None if estimator is None else estimator.xyz)
    (session/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
