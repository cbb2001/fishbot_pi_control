"""单采集、单 PWM 线程的水下导航会话。阶段转换和滤波均使用采集时间。"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, replace
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
from .underwater_navigation_20261026 import (
    UnderwaterScheduler, NavigationEstimator, AcceptedUWB, SurfaceFinPolicy, UWBObservation, forward_tid,
    validate_safety_depth,
)


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
        self.waypoints=cfg.route; self.waypoint_index=0; self.waypoint_start=None
        self.phase='initial_fix'; self.stage_start=0; self.motion_start=None; self.segment_start=None
        # 初始化时尚不运动；首段目标在 enter_navigation 中加载。
        # 保留此基类初始化接口，兼容拥有独立三元点表和 target 属性的 fixed-yaw 分支。
        self.depth=0.; self.depth_t=None; self.depth_policy=self._depth_policy(cfg.h0)
        self.nav_policy=NavigationPolicy(cfg.navigation,calibration)
        self.inflight={'tail':None,'fin':None}; self.epoch=0
        self.tail_side=0; self.tail_mode=None; self.tail_held=False; self.tail_ticket=None
        self.center_ticket=None; self.center_purpose=None; self.center_requested_ns=None
        self.surface_policy=None; self.surface_depth=0.; self.reason=None

    @property
    def target(self):
        return self.waypoints[self.waypoint_index]

    def _navigation_policy(self):
        return NavigationPolicy(replace(self.cfg.navigation,x_d=self.target.x,y_d=self.target.y),self.calibration)

    def _depth_policy(self,target):
        last=self.ex.scheduler.fin_sequence.last_b2
        return FixedDivePolicy(self.calibration,h0=target,k=self.cfg.depth_kp,ki=self.cfg.depth_ki,
             initial_b2=-last if last else self.cfg.initial_b2,action_duration_s=self.cfg.fin_duration_s,
             error_scale_m=target if target>0 else self.cfg.h0,max_integral_gap_s=self.cfg.pressure_timeout_s)

    def _reset_controls(self,now):
        """切目标只清控制器积分；位置/EKF、总时钟及上浮时钟均由外层保留。"""
        self.depth_policy=self._depth_policy(self.target.h); self.depth_policy.start_control(now)
        self.nav_policy=self._navigation_policy()
        self.tail_side=0; self.tail_mode=None; self.tail_held=False; self.tail_ticket=None

    def _waypoint_started(self,now):
        self.waypoint_start=now
        self.event('waypoint_started',waypoint_index=self.waypoint_index+1,
                   target=[self.target.x,self.target.y,self.target.h],timeout_s=self.target.t,start_t_ns=now)

    def _advance_waypoint(self,now,xyz):
        """点的时限含上浮/等UWB；暂停时不能拿冻结的惯性位置判断新到达。"""
        changed=False
        while self.waypoint_start is not None and self.target.t is not None:
            distance=math.hypot(xyz[0]-self.target.x,xyz[1]-self.target.y)
            reached=self.phase=='navigate' and distance<=self.cfg.navigation.delta_s
            expired=now-self.waypoint_start>=round(self.target.t*1e9)
            if not reached and not expired: break
            self.event('waypoint_completed',waypoint_index=self.waypoint_index+1,
                       target=[self.target.x,self.target.y,self.target.h],timeout_s=self.target.t,
                       start_t_ns=self.waypoint_start,completed_t_ns=now,
                       reason='reached' if reached else 'timeout',distance_m=distance)
            self.waypoint_index+=1; self._waypoint_started(now); changed=True
            # 同一有效位置可以覆盖连续的多个目标；遇到 null 立即停止切段。
        if changed and self.phase=='navigate':
            self._center(now,'waypoint')
        return changed

    def enter_navigation(self,now):
        self.phase='navigate'; self.stage_start=self.segment_start=now
        if self.motion_start is None: self.motion_start=now
        if self.waypoint_start is None: self._waypoint_started(now)
        self._reset_controls(now)
        self.ex.resume_agents(('tail','fin')); self.epoch+=1
        self.inflight={'tail':None,'fin':None}; self.tail_side=0; self.tail_mode=None
        self.tail_held=False; self.surface_policy=None
        self.event('navigation_started',segment_start_t_ns=now,motion_start_t_ns=self.motion_start)

    def on_depth(self,t,depth):
        if self.depth_t is not None and t<=self.depth_t: return
        self.depth_t=t; self.depth=depth
        if self.phase=='navigate': self.depth_policy.observe_depth(depth,t)

    def _center(self,now,purpose):
        if self.center_ticket is None:
            self.center_ticket=self.ex.request_hold(agents=('tail','fin'),center=True,block=True)
            self.center_requested_ns=now
        self.center_purpose=purpose

    def on_uwb(self,obs,now):
        if (self.phase=='surface' and obs.t_ns>self.stage_start
                and 0<=now-obs.t_ns<=self.cfg.uwb_age_s*1e9
                and any(v is not None and math.isfinite(v) and v>0 for v in obs.frame.ranges_m)):
            self._center(now,'acquire')

    def _completions(self):
        for item in self.ex.drain_completed():
            if item.request_id!=self.inflight[item.agent] or item.context!=self.epoch:
                self.event('late_completion_ignored',request_id=item.request_id,agent=item.agent)
                continue
            self.inflight[item.agent]=None
            if item.agent=='fin':
                policy=self.surface_policy if self.surface_policy is not None else self.depth_policy
                policy.acknowledge(item.action)
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
            if self.center_purpose=='waypoint':
                self._reset_controls(now)
                self.ex.resume_agents(('tail','fin'))
                self.event('waypoint_control_started',waypoint_index=self.waypoint_index+1,
                           center_ack_t_ns=ticket.ack_t_ns)
            else:
                self.surface_policy=SurfaceFinPolicy(self.calibration,self.cfg.fin_duration_s,
                    self.cfg.initial_b2,self.ex.scheduler.fin_sequence.last_b2)
                self.ex.resume_agents(('fin',))
                self.event('surface_sequence_started',center_ack_t_ns=ticket.ack_t_ns,
                           servo4_max_deg=self.calibration.limits[4][1],
                           servo4_min_deg=self.calibration.limits[4][0])
        if self.phase=='navigate' and now-self.segment_start>=self.cfg.t4*1e9:
            self.phase='surface'; self.stage_start=now
            self.surface_depth=self.depth
            # 取消旧定深动作并归中，收到回执后从b1=0开始固定上浮序列。
            self._center(now,'surface')
            self.event('surface_started',depth_m=self.depth)
            return
        # t4/上浮归中优先；切点绝不绕过上浮与 n 帧平均定位的阶段约束。
        if self._advance_waypoint(now,xyz) and self.phase=='navigate': return
        if self.phase=='acquire':
            # 一直等待合格四路帧，仅总任务t3/外部停止/安全故障可结束。
            return
        if self.phase=='surface':
            if self.inflight['fin'] is None:
                self._submit('fin',self.surface_policy.next_action())
            return
        if self.phase!='navigate': return
        if self.inflight['fin'] is None:
            action=self.depth_policy.next_action(self.depth)
            if action is not None: self._submit('fin',action)
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
        if m.phase=='navigate': return min(m.target.h,(time.monotonic_ns()-m.segment_start)/1e9*.2)
        if m.phase=='surface': return max(0.,m.surface_depth-(time.monotonic_ns()-m.stage_start)/1e9*.3)
        return 0.

    def _create_mock_workers(self):
        self.workers=[_UnderwaterMockWorker(name,self) for name in ('imu','depth','power','uwb')]


def run_session(robot,cfg,session,*,dry_run=False,no_servo=False,stop_event=None,sensors=None,controller=None):
    if dry_run and no_servo: raise ValueError('dry-run 与 no-servo 不能同时使用')
    cal=build_calibration(robot)
    SurfaceFinPolicy(cal,cfg.fin_duration_s,cfg.initial_b2)  # 采集/PWM前校验完整上浮行程。
    validate_safety_depth(cfg,robot['safety']['max_depth_m'])
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
        config={'command_hz':cfg.command_hz,'safe_recenter_s':0.,'initial_move_s':0.,'hold_at_center':True,
                'terminal_center_hold_s':cfg.center_settle_s},
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
    gate=AcceptedUWB(cfg); decoder=UWBStreamDecoder(cfg.navigation.uwb_tag)
    last_seq={n:None for n in ('imu','depth','uwb')}; pending=[]; last_processed=-1
    calibration_rows={'imu':[],'depth':[]}; last_tid=None
    decoded_ids=deque(maxlen=512)
    after_calibration=[]
    last_status=0; gate_phase=None; diagnostic_start=None
    last_range_ns=None; valid_ranges=0; last_complete_uwb_ns=None

    def write(name,**data):
        current_phase=mission.phase if mission is not None else phase
        target=mission.target if mission else cfg.route[0]
        row=dict(t_ns=time.monotonic_ns(),phase=current_phase,
                 waypoint_index=mission.waypoint_index+1 if mission else 1,
                 waypoint_count=len(cfg.route),target=[target.x,target.y,target.h])
        row.update(data)
        if not logs[name].write(row):
            raise RuntimeError(f'{name} 日志队列满')

    def event(kind,**data):
        # 阶段事件同步控制积分开关；截止时刻也按采样时间检查，不能跨越t4积分。
        if estimator is not None:
            if kind=='navigation_started':
                end=min(data['segment_start_t_ns']+round(cfg.t4*1e9),
                        data['motion_start_t_ns']+round(cfg.t3*1e9))
                estimator.start_navigation(data['segment_start_t_ns'],end)
            elif kind=='surface_started':
                estimator.pause_navigation(mission.stage_start)
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

    def accept_fix(fix,*,used_for_correction):
        gate.commit(fix)
        saved=asdict(fix); saved['fix_t_ns']=saved.pop('t_ns')
        event('uwb_accepted',used_for_correction=used_for_correction,**saved)

    def sync_acquire_gate():
        nonlocal phase,gate_phase
        phase=mission.phase
        if phase=='acquire' and gate_phase!=mission.stage_start:
            gate_phase=mission.stage_start
            gate.reset(mission.stage_start)  # 保留上一组平均参照；归中后重新累计n帧。

    def apply_fix(fix,now):
        try: innovation=estimator.apply_surface_fix(fix,now)
        except ValueError as exc:
            event('fix_rejected',detail=str(exc)); return False
        event('position_corrected',fix_t_ns=fix.t_ns,xy=fix.xy,velocity=fix.velocity,
              xyz=estimator.xyz,accel_filter_reset=True,integration_reset=True,
              relative_xy=estimator.relative_xy,relative_xyz=estimator.relative_xyz,ekf=innovation['ekf'],
              frame_id=fix.frame_id,ranges_m=fix.ranges_m,
              innovation=innovation,quality=fix.quality,age_s=(now-fix.t_ns)/1e9)
        accept_fix(fix,used_for_correction=True)
        return True

    try:
        write_metadata_yaml(session/'metadata.yaml',{'task':'underwater_navigation_20261026','config':asdict(cfg),
            'robot':sensor_cfg,'dry_run':dry_run,'no_servo':no_servo,'z_positive':'down',
            'waypoints':[asdict(point) for point in cfg.route],
            'waypoint_reached_mode':'horizontal_distance_only',
            'waypoint_time_includes_surface_and_acquire':True,
            'uwb_fix_mode':'mean_valid_ranges','uwb_average_frames':cfg.uwb_average_frames,
            'uwb_surface_depth_m':0.,'uwb_velocity_reset':'horizontal_zero_at_fix',
            'uwb_accel_filter_reset':'both_stages_zero_at_fix','imu_integration_mode':'navigate_only',
            'ekf_state_order':['x','y','z','vx','vy','vz','bx','by','bz'],
            'ekf_z_positive':'down','linear_acc_z_positive':'up',
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
                        heading.update(data,t)
                        write('imu_filtered',sample_t_ns=t,xyz=estimator.xyz,velocity=estimator.v.tolist(),
                              raw_linear_acc=estimator.raw_acc.tolist(),filtered_acc=estimator.filtered_acc,
                              b_body=estimator.bias.tolist(),yaw_deg=heading.values['yaw_deg'],
                              integrating=estimator.integrating,relative_xy=estimator.relative_xy,
                              relative_xyz=estimator.relative_xyz,ekf_bias_mps2=estimator.ekf.x[6:].tolist(),
                              ekf_variance=np.diag(estimator.ekf.P).tolist())
                elif name=='depth':
                    depth=depth_cal.depth_m(data['pressure_pa']); estimator.pressure(t,depth)
                    if estimator.integrating:
                        write('navigation',event='pressure_ekf_update',sample_t_ns=t,pressure_depth_m=depth,
                              xyz=estimator.xyz,quality=estimator.ekf.last_pressure_update)
                    if mission: mission.on_depth(t,depth)
                elif name=='uwb':
                    raw_bytes=bytes.fromhex(data.get('raw_hex',''))
                    for frame in decoder.feed(raw_bytes,include_incomplete=True):
                        if frame.frame_id in decoded_ids: continue
                        decoded_ids.append(frame.frame_id)
                        last_range_ns=t; valid_ranges=sum(v is not None for v in frame.ranges_m)
                        if valid_ranges==4: last_complete_uwb_ns=t
                        obs=UWBObservation(t,frame,estimator.depth_at(t))
                        if mission and not no_servo:
                            # 回执可能在本批传感器样本处理期间到达；首个归中后完整帧不能丢掉。
                            ticket=mission.center_ticket
                            if (ticket is not None and ticket.is_set()
                                    and mission.center_purpose=='acquire'):
                                mission.tick(now,estimator.xyz,heading.values['yaw_deg'])
                                sync_acquire_gate()
                            mission.on_uwb(obs,now)
                        awaiting_initial_center=phase=='initial_fix' and executor_started and not initial_centered
                        if (phase in ('initial_fix','acquire','diagnostic') and t>gate.boundary
                                and not fixes and not awaiting_initial_center):
                            fix=gate.add(obs,now)
                            if fix is not None: fixes.append(fix)
                            write('navigation',event='uwb_quality',sample_t_ns=t,frame_id=frame.frame_id,
                                  valid_ranges=sum(v is not None for v in frame.ranges_m),reason=gate.reason,
                                  quality=gate.quality)
            if estimator is None and now>=calibration_end+round((cfg.merge_delay_s+cfg.control_poll_s)*1e9):
                sensor_cal,bias,heading=calibrate_window(cfg,calibration_rows['imu'],calibration_rows['depth'],calibration_end)
                depth_cal=FixedDepthCalibration(sensor_cal,sensor_cal.water_density_kg_m3*sensor_cal.gravity_mps2,drift)
                guard.calibration=depth_cal
                estimator=NavigationEstimator(cfg,bias,heading.zeros['yaw_deg'],sensor_cal.gravity_mps2)
                # 保持采集/健康监测，只有 navigation_started 后才传播位置速度。
                last_i=calibration_rows['imu'][-1]; last_d=calibration_rows['depth'][-1]
                for s in sorted((last_i,last_d),key=lambda s:s.t_ns):
                    if s.name=='imu': estimator.imu(s.t_ns,s.data); heading.update(s.data,s.t_ns)
                    else: estimator.pressure(s.t_ns,depth_cal.depth_m(s.data['pressure_pa']))
                state={**depth_cal.state_dict(),'b_body':bias.tolist(),'heading':heading.diagnostics}
                (session/'calibration.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
                event('calibration_completed',imu_samples=len(calibration_rows['imu']),depth_samples=len(calibration_rows['depth']))
                # 标定后已排出的样本重新排队；初始定位期间仍不积分。
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
                    accept_fix(fixes[0],used_for_correction=False)
                    ex.start(); executor_started=True
                    event('pwm_start_requested')
                if executor_started and not initial_centered and ex.wait_ready(0):
                    guard.executor_started=True; initial_centered=True
                    boundary=ex.last_write_ack_t_ns
                    gate.reset(boundary); decoder.reset_partial(); fixes=[]
                    event('initial_center_confirmed',center_ack_t_ns=ex.last_write_ack_t_ns)
                if initial_centered and fixes and apply_fix(fixes[0],now):
                    mission=NavigationMission(cfg,cal,ex,event)
                    latest=corrected.buffers['depth'].latest()
                    mission.on_depth(latest.t_ns,depth_cal.depth_m(latest.data['pressure_pa']))
                    mission.enter_navigation(now); arm_timer(); phase=mission.phase
                    if isinstance(sensors,UnderwaterMockSensors): sensors.mission=mission
            elif phase=='diagnostic':
                if fixes and apply_fix(fixes[0],now): gate.reset(fixes[0].t_ns)
            elif mission:
                if now-mission.motion_start>=cfg.t3*1e9:
                    # 与独立定时器使用同一正常停止入口，避免监测线程误报预期退出。
                    expire(); reason='time_limit'; break
                try:
                    if phase=='acquire' and fixes and apply_fix(fixes[0],now): mission.enter_navigation(now)
                    mission.tick(now,estimator.xyz,heading.values['yaw_deg'])
                except RuntimeError:
                    if deadline.is_set(): reason='time_limit'; break
                    raise
                sync_acquire_gate()
                if mission.reason: reason=mission.reason; break
            if now-last_status>=1e9/cfg.status_hz:
                last_status=now
                remaining=None if timer is None else max(0,cfg.t3-(now-(diagnostic_start if no_servo else mission.motion_start))/1e9)
                next_surface=None if not mission or phase!='navigate' else max(0,cfg.t4-(now-mission.segment_start)/1e9)
                target=mission.target if mission else cfg.route[0]
                distance=math.hypot(estimator.xyz[0]-target.x,estimator.xyz[1]-target.y)
                depth_target=target.h if phase=='navigate' else None
                waypoint_elapsed=None if not mission or mission.waypoint_start is None else (now-mission.waypoint_start)/1e9
                waypoint_remaining=None if target.t is None or waypoint_elapsed is None else max(0.,target.t-waypoint_elapsed)
                fresh_ranges=valid_ranges if last_range_ns is not None and now-last_range_ns<=cfg.uwb_age_s*1e9 else 0
                uwb_reason=gate.reason if phase in ('initial_fix','acquire','diagnostic') else (
                    'waiting_surface_signal' if phase=='surface' else 'dead_reckoning')
                write('navigation',event='status',xyz=estimator.xyz,velocity=estimator.v.tolist(),
                      t3_remaining_s=remaining,waypoint_elapsed_s=waypoint_elapsed,
                      waypoint_remaining_s=waypoint_remaining,
                      t4_remaining_s=next_surface,uwb_reason=uwb_reason,distance_m=distance,
                      depth_target_m=depth_target,valid_ranges=fresh_ranges,
                      uwb_frames_collected=len(gate.rows),uwb_frames_required=cfg.uwb_average_frames,
                      integrating=estimator.integrating,relative_xy=estimator.relative_xy,
                      relative_xyz=estimator.relative_xyz,ekf_bias_mps2=estimator.ekf.x[6:].tolist(),
                      last_accepted_uwb_t_ns=None if gate.last_accepted is None else gate.last_accepted.t_ns,
                      depth_t_ns=estimator.last_depth,last_complete_uwb_t_ns=last_complete_uwb_ns)
                index=mission.waypoint_index+1 if mission else 1
                print(f'[{phase}] 目标{index}/{len(cfg.route)}=({target.x:g},{target.y:g},{target.h:g}) '
                      f'XYZ={tuple(round(v,3) for v in estimator.xyz)} '
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
        terminal_center=dict(
            hold_s=cfg.center_settle_s,
            command_ack_t_ns=ex.terminal_center_ack_t_ns,
            hold_completed_t_ns=ex.terminal_center_hold_completed_t_ns,
            pwm_disabled_t_ns=ex.pwm_disabled_t_ns,
            target_angles_deg=cal.centers,
        )
        try: event('terminal_center_status',**terminal_center)
        except Exception as exc: cleanup_errors.append(str(exc))
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
                waypoint_index=None if mission is None else mission.waypoint_index+1,
                waypoint_count=len(cfg.route),
                target=None if mission is None else [mission.target.x,mission.target.y,mission.target.h],
                terminal_center=terminal_center,
                executor_started=executor_started,alive_workers=alive,pressure_origin_t_ns=drift.origin_t_ns,
                last_accepted_uwb=None if gate.last_accepted is None else asdict(gate.last_accepted),
                xyz=None if estimator is None else estimator.xyz)
    (session/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
