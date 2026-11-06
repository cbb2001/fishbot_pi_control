"""水面调向后锁定 yaw 偏差的多目标水下导航；采集、PWM 与 UWB 规则沿用原版。"""
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
from .rl_sensor_calibration_20260914 import wrap180
from .rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from .rl_servo_state_tracker_20260914 import RLServoStateTracker
from .sensor_manager import SensorManager
from .surface_navigation_20261010 import NavigationPolicy, NavigationTailAction, UWBStreamDecoder
from .underwater_navigation_20261026 import (
    UnderwaterScheduler, NavigationEstimator, AcceptedUWB, SurfaceFinPolicy, UWBObservation, forward_tid,
)
from .underwater_navigation_20261026_only_one_yaw import FrozenYawPolicy, validate_safety_depth
from .underwater_navigation_session_20261026 import (
    calibrate_window, check_camera_health, NavigationMission as _BaseNavigationMission,
    UnderwaterMockSensors as _BaseMockSensors, _UnderwaterMockWorker as _BaseMockWorker,
)


class NavigationMission(_BaseNavigationMission):
    """合格水面定位或目标超时切点；只有水面调向能更新本航段 yaw 偏差。"""

    def __init__(self, cfg, calibration, executor, event):
        self.waypoint_index = 0
        self.waypoint_start = None
        self.surface_xy = None
        self.surface_distance_m = None
        self.surface_fix_t_ns = None
        self.desired_yaw_deg = None
        self.alignment_start_yaw = 0.
        self.alignment_start_ns = None
        self.frozen_yaw_error_deg = None
        self.frozen_policy = None
        self.target_reached = False
        self.final_reached = False
        self.last_decision = None
        super().__init__(cfg, calibration, executor, event)

    @property
    def target(self):
        return self.cfg.waypoints[self.waypoint_index]

    def _waypoint_started(self, now):
        self.waypoint_start = now
        self.event('waypoint_started', waypoint_index=self.waypoint_index,
                   target=self.target, timeout_s=self.target[3], start_t_ns=now)

    def _switch_waypoint(self, now, reason, *, fix_t_ns=None):
        # NULL 即持续守当前目标；即使它不是最后一点也不能跳过。
        if self.target[3] is None or self.waypoint_index >= len(self.cfg.waypoints)-1:
            return False
        previous = self.waypoint_index
        self.event('waypoint_completed', waypoint_index=previous, target=self.target,
                   reason=reason, start_t_ns=self.waypoint_start, completed_t_ns=now,
                   elapsed_s=(now-self.waypoint_start)/1e9, fix_t_ns=fix_t_ns)
        self.waypoint_index += 1
        self.target_reached = self.final_reached = False
        self.surface_xy = self.surface_distance_m = self.surface_fix_t_ns = None
        self.desired_yaw_deg = None
        self.frozen_yaw_error_deg = self.frozen_policy = self.last_decision = None
        self._waypoint_started(now)
        self.event('waypoint_changed', waypoint_index=self.waypoint_index,
                   previous_waypoint_index=previous, target=self.target, reason=reason,
                   fix_t_ns=fix_t_ns)
        return True

    def _start_surface(self, now, reason):
        self.surface_depth = max(0., self.depth)
        self.phase = 'surface'
        self.stage_start = now
        self._center(now, 'surface')
        self.event('surface_started', depth_m=self.depth, reason=reason)

    def update_waypoint_timeout(self, now):
        """切换目标不借用水下估计；新航向须等待合格水面 UWB。"""
        if (self.reason or self.motion_start is None or self.waypoint_start is None
                or now-self.motion_start >= self.cfg.t3*1e9):
            return False
        timeout = self.target[3]
        if timeout is None or now-self.waypoint_start < round(timeout*1e9):
            return False
        if not self._switch_waypoint(now, 'timeout'):
            return False
        if self.phase == 'navigate':
            # 即使 t4=t3，也不能拿旧位置/旧偏差给新目标导向。
            self._start_surface(now, 'waypoint_timeout')
        elif self.phase == 'align':
            # 调向期间也可能平移；先取消尾鳍并归中，再重新累计有效 UWB。
            self.phase = 'reacquire'
            self._center(now, 'acquire')
            self.event('waypoint_reacquire_requested', waypoint_index=self.waypoint_index)
        # 已在 surface/acquire 时继续原上浮/取数，不重置阶段边界或累计帧。
        return True

    def _depth_policy(self, target):
        last = self.ex.scheduler.fin_sequence.last_b2
        return FixedDivePolicy(
            self.calibration, h0=target, k=self.cfg.depth_kp, ki=self.cfg.depth_ki,
            initial_b2=-last if last else self.cfg.initial_b2,
            action_duration_s=self.cfg.fin_duration_s,
            error_scale_m=target, max_integral_gap_s=self.cfg.pressure_timeout_s)

    def enter_surface_alignment(self, fix, now, yaw):
        """仅接受 initial_fix/acquire 中经 AcceptedUWB 校验并提交的定位。"""
        if self.phase not in ('initial_fix', 'acquire'):
            raise RuntimeError('只能在水面定位阶段更新目标与 yaw')
        if self.motion_start is None:
            self.motion_start = now
            self._waypoint_started(now)
        elif now-self.motion_start >= self.cfg.t3*1e9:
            return
        # 取数等待也计入当前点的时限；新收到的有效定位可服务已切换的目标。
        self.update_waypoint_timeout(now)
        # 同一位置可能覆盖多个相邻目标，顺序跳过已到达点，不下潜绕回。
        while True:
            x, y, h, timeout = self.target
            distance = math.hypot(x-fix.xy[0], y-fix.xy[1])
            self.target_reached = distance < self.cfg.navigation.delta_s
            self.final_reached = self.target_reached and self.waypoint_index == len(self.cfg.waypoints)-1
            if distance >= self.cfg.navigation.delta_s:
                break
            self.event('waypoint_reached', waypoint_index=self.waypoint_index,
                       target=self.target, xy=fix.xy, distance_m=distance, fix_t_ns=fix.t_ns)
            if timeout is None:
                break
            self._switch_waypoint(now, 'reached', fix_t_ns=fix.t_ns)
        self.surface_xy = tuple(fix.xy)
        self.surface_fix_t_ns = fix.t_ns
        self.surface_distance_m = distance
        self.desired_yaw_deg = math.degrees(math.atan2(y-fix.xy[1], x-fix.xy[0])) if distance > 1e-9 else yaw
        nav = replace(self.cfg.navigation, x_d=x, y_d=y)
        self.nav_policy = NavigationPolicy(nav, self.calibration)
        self.frozen_policy = FrozenYawPolicy(nav, self.calibration)
        self.frozen_yaw_error_deg = None
        self.last_decision = None
        self.phase = 'align'
        self.stage_start = self.alignment_start_ns = now
        self.alignment_start_yaw = yaw
        self.surface_policy = None
        # 归中并阻塞胸鳍，调向阶段只恢复尾鳍。
        self._center(now, 'align')
        self.event('surface_alignment_started', waypoint_index=self.waypoint_index,
                   target=self.target, xy=self.surface_xy, desired_yaw_deg=self.desired_yaw_deg,
                   yaw_deg=yaw, yaw_error_deg=wrap180(self.desired_yaw_deg-yaw),
                   delta_yaw_deg=self.cfg.navigation.delta_yaw, target_reached=self.target_reached,
                   final_reached=self.final_reached,
                   motion_start_t_ns=self.motion_start)

    def enter_navigation(self, now):
        if not self.target_reached and self.frozen_yaw_error_deg is None:
            raise RuntimeError('下潜前必须完成水面调向并锁定 yaw 偏差')
        self.phase = 'navigate'
        self.stage_start = self.segment_start = now
        self.depth_policy = self._depth_policy(self.target[2])
        self.depth_policy.start_control(now)
        self.inflight = {'tail': None, 'fin': None}
        self.epoch += 1
        self.tail_side = 0
        self.tail_mode = None
        self.tail_held = self.target_reached
        self.surface_policy = None
        self.ex.resume_agents(('fin',) if self.target_reached else ('tail', 'fin'))
        self.event('navigation_started', segment_start_t_ns=now, motion_start_t_ns=self.motion_start,
                   waypoint_index=self.waypoint_index, target=self.target,
                   waypoint_start_t_ns=self.waypoint_start, waypoint_timeout_s=self.target[3],
                   frozen_yaw_error_deg=self.frozen_yaw_error_deg, target_reached=self.target_reached,
                   final_reached=self.final_reached)

    def _tail(self, decision):
        self.last_decision = decision
        if self.inflight['tail'] is None:
            if self.tail_mode != decision.control_mode:
                self.tail_side = 0
            self.tail_mode = decision.control_mode
            theta = decision.theta_left if self.tail_side == 0 else decision.theta_right
            self._submit('tail', NavigationTailAction(theta, self.cfg.navigation.t_0))
            self.event('tail_control', waypoint_index=self.waypoint_index,
                       yaw_error_source='surface_live' if self.phase == 'align' else 'surface_frozen',
                       decision=asdict(decision))

    def tick(self, now, xyz, yaw, *, yaw_t_ns=None):
        # xyz 仅保留原接口兼容；水下估计位置永远不能触发切点或改 yaw。
        if self.reason:
            return
        if self.motion_start is not None and now-self.motion_start >= self.cfg.t3*1e9:
            self.reason = 'time_limit'
            self.ex.request_center_stop()
            return
        self._completions()
        if self.update_waypoint_timeout(now) and self.phase in ('surface', 'reacquire'):
            # 下一轮消费归中回执，再恢复上浮动作或累计定位。
            return
        if self.center_ticket is not None:
            ticket = self.center_ticket
            if ticket.failure:
                raise RuntimeError(str(ticket.failure))
            if not ticket.is_set():
                if now-self.center_requested_ns > self.cfg.center_ack_timeout_s*1e9:
                    raise TimeoutError('PWM center acknowledgement timeout')
                return
            # 正式会话传入采集时间。合并队列中的旧 IMU 不能冒充归中后的角度复检。
            # 未传时间戳的纯计算调用约定 yaw 已是调用时刻的同步值。
            if (self.center_purpose == 'dive' and yaw_t_ns is not None
                    and yaw_t_ns <= ticket.ack_t_ns):
                return
            self.inflight = {'tail': None, 'fin': None}
            self.epoch += 1
            self.center_ticket = None
            self.tail_ticket = None
            self.tail_held = True
            self.tail_side = 0
            self.tail_mode = None
            if self.center_purpose == 'acquire':
                self.phase = 'acquire'
                self.stage_start = ticket.ack_t_ns
                self.event('uwb_acquire_started', center_ack_t_ns=ticket.ack_t_ns)
                return
            if self.center_purpose in ('align', 'dive'):
                error = wrap180(self.desired_yaw_deg-yaw)
                if self.target_reached:
                    # NULL 目标到点：尾鳍继续归中，保留压力定深和 t4 周期。
                    self.enter_navigation(now)
                elif self.center_purpose == 'dive' and abs(error) <= self.cfg.navigation.delta_yaw:
                    self.frozen_policy.freeze(*self.surface_xy, self.desired_yaw_deg, yaw,
                                              self.surface_distance_m)
                    self.frozen_yaw_error_deg = error
                    self.event('yaw_frozen', yaw_error_deg=error, yaw_deg=yaw,
                               desired_yaw_deg=self.desired_yaw_deg, xy=self.surface_xy,
                               fix_t_ns=self.surface_fix_t_ns, center_ack_t_ns=ticket.ack_t_ns,
                               yaw_sample_t_ns=yaw_t_ns)
                    self.enter_navigation(now)
                else:
                    # 归中等待期间若又转出容差，继续水面调向，不放行胸鳍。
                    self.ex.resume_agents(('tail',))
                    self.tail_held = False
            else:
                self.surface_policy = SurfaceFinPolicy(
                    self.calibration, self.cfg.fin_duration_s, self.cfg.initial_b2,
                    self.ex.scheduler.fin_sequence.last_b2)
                self.ex.resume_agents(('fin',))
                self.event('surface_sequence_started', center_ack_t_ns=ticket.ack_t_ns,
                           servo4_max_deg=self.calibration.limits[4][1],
                           servo4_min_deg=self.calibration.limits[4][0])
        if self.phase == 'navigate' and now-self.segment_start >= self.cfg.t4*1e9:
            self._start_surface(now, 't4')
            return
        if self.phase == 'acquire':
            return
        if self.phase == 'surface':
            if self.inflight['fin'] is None:
                self._submit('fin', self.surface_policy.next_action())
            return
        if self.phase == 'align':
            decision = self.nav_policy.decide(*self.surface_xy, yaw, now)
            self.last_decision = decision
            if abs(decision.yaw_error_deg) <= self.cfg.navigation.delta_yaw:
                self._center(now, 'dive')
            else:
                self._tail(decision)
            return
        if self.phase != 'navigate':
            return
        if self.inflight['fin'] is None:
            action = self.depth_policy.next_action(self.depth)
            if action is not None:
                self._submit('fin', action)
        if not self.target_reached:
            self._tail(self.frozen_policy.decide(now))


class _UnderwaterMockWorker(_BaseMockWorker):
    def read_once(self):
        samples = super().read_once()
        if self.name == 'imu':
            for sample in samples:
                yaw = self.manager.sim_yaw(sample.t_ns)
                a = math.radians(yaw)/2
                sample.data.update(yaw_deg=yaw, quat=[math.cos(a), 0., 0., math.sin(a)])
        return samples


class UnderwaterMockSensors(_BaseMockSensors):
    """仅模拟阶段/调向收敛，不模拟水动力，不用于评估导航精度。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_yaw = 0.

    def sim_yaw(self, t_ns):
        m = self.mission
        if m is not None and m.phase == 'align':
            error = wrap180(m.desired_yaw_deg-m.alignment_start_yaw)
            travelled = max(0., (t_ns-m.alignment_start_ns)/1e9)*60.
            self._mock_yaw = wrap180(m.alignment_start_yaw+math.copysign(min(abs(error), travelled), error))
        return self._mock_yaw

    def sim_depth(self):
        m = self.mission
        if m is None:
            return 0.
        if m.phase == 'navigate':
            return min(m.target[2], (time.monotonic_ns()-m.segment_start)/1e9*.2)
        if m.phase == 'surface':
            return max(0., m.surface_depth-(time.monotonic_ns()-m.stage_start)/1e9*.3)
        return 0.

    def _create_mock_workers(self):
        self.workers = [_UnderwaterMockWorker(name, self) for name in ('imu', 'depth', 'power', 'uwb')]


def run_session(robot,cfg,session,*,dry_run=False,no_servo=False,stop_event=None,sensors=None,controller=None):
    if dry_run and no_servo: raise ValueError('dry-run 与 no-servo 不能同时使用')
    cal=build_calibration(robot)
    SurfaceFinPolicy(cal,cfg.fin_duration_s,cfg.initial_b2)  # 采集/PWM前校验完整上浮行程。
    validate_safety_depth(cfg, float(robot['safety']['max_depth_m']))
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
        if not logs[name].write(dict(t_ns=time.monotonic_ns(),phase=current_phase,**data)):
            raise RuntimeError(f'{name} 日志队列满')

    def event(kind,**data):
        # 阶段事件同步控制积分开关；截止时刻也按采样时间检查，不能跨越t4积分。
        if estimator is not None:
            if kind=='navigation_started':
                end=min(data['segment_start_t_ns']+round(cfg.t4*1e9),
                        data['motion_start_t_ns']+round(cfg.t3*1e9))
                if data['waypoint_timeout_s'] is not None:
                    end=min(end,data['waypoint_start_t_ns']+round(data['waypoint_timeout_s']*1e9))
                estimator.start_navigation(data['segment_start_t_ns'],end)
            elif kind in ('surface_started', 'surface_alignment_started'):
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
        write_metadata_yaml(session/'metadata.yaml',{'task':'underwater_navigation_20261026_only_one_yaw','config':asdict(cfg),
            'robot':sensor_cfg,'dry_run':dry_run,'no_servo':no_servo,'z_positive':'down',
            'uwb_fix_mode':'mean_valid_ranges','uwb_average_frames':cfg.uwb_average_frames,
            'uwb_surface_depth_m':0.,'uwb_velocity_reset':'horizontal_zero_at_fix',
            'uwb_accel_filter_reset':'both_stages_zero_at_fix','imu_integration_mode':'navigate_only',
            'ekf_state_order':['x','y','z','vx','vy','vz','bx','by','bz'],
            'ekf_z_positive':'down','linear_acc_z_positive':'up',
            'yaw_control_mode':'surface_align_then_frozen_error',
            'waypoint_arrival_source':'accepted_surface_uwb_xy_only',
            'waypoint_format':['x','y','h','t'], 'waypoint_null_mode':'never_advance',
            'waypoint_timer_start':'target_activation_includes_alignment_surface_acquire',
            'waypoint_timeout_action':'advance_then_surface_fix_before_new_heading',
            't3_start':'first_surface_alignment', 't4_start':'each_navigation_start',
            'underwater_imu_usage':'health_and_ekf_logging_only_not_tail_or_arrival',
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
                        # 航向滤波只服务水面调向；水下保留锁定值，不计算实时 yaw 偏差。
                        if mission is None or mission.phase != 'navigate':
                            heading.update(data,t)
                        write('imu_filtered',sample_t_ns=t,xyz=estimator.xyz,velocity=estimator.v.tolist(),
                              raw_linear_acc=estimator.raw_acc.tolist(),filtered_acc=estimator.filtered_acc,
                              b_body=estimator.bias.tolist(),yaw_deg=heading.values['yaw_deg'],
                              yaw_updated=mission is None or mission.phase != 'navigate',
                              frozen_yaw_error_deg=None if mission is None else mission.frozen_yaw_error_deg,
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
                                mission.tick(now,estimator.xyz,heading.values['yaw_deg'],
                                             yaw_t_ns=heading.last_t_ns)
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
                    mission.enter_surface_alignment(fixes[0],now,heading.values['yaw_deg'])
                    arm_timer(); phase=mission.phase
                    if isinstance(sensors,UnderwaterMockSensors): sensors.mission=mission
            elif phase=='diagnostic':
                if fixes and apply_fix(fixes[0],now): gate.reset(fixes[0].t_ns)
            elif mission:
                if now-mission.motion_start>=cfg.t3*1e9:
                    # 与独立定时器使用同一正常停止入口，避免监测线程误报预期退出。
                    expire(); reason='time_limit'; break
                try:
                    if phase=='acquire' and fixes and apply_fix(fixes[0],now):
                        mission.enter_surface_alignment(fixes[0],now,heading.values['yaw_deg'])
                    mission.tick(now,estimator.xyz,heading.values['yaw_deg'],yaw_t_ns=heading.last_t_ns)
                except RuntimeError:
                    if deadline.is_set(): reason='time_limit'; break
                    raise
                sync_acquire_gate()
                if mission.reason: reason=mission.reason; break
            if now-last_status>=1e9/cfg.status_hz:
                last_status=now
                remaining=None if timer is None else max(0,cfg.t3-(now-(diagnostic_start if no_servo else mission.motion_start))/1e9)
                next_surface=None if not mission or phase!='navigate' else max(0,cfg.t4-(now-mission.segment_start)/1e9)
                target=cfg.waypoints[0] if mission is None else mission.target
                waypoint_elapsed=None if mission is None or mission.waypoint_start is None else (now-mission.waypoint_start)/1e9
                waypoint_remaining=None if target[3] is None or waypoint_elapsed is None else max(0.,target[3]-waypoint_elapsed)
                # 距离字段明确对应最后有效 UWB，不用水下估计值冒充到点依据。
                distance=None if mission is None else mission.surface_distance_m
                depth_target=target[2] if phase=='navigate' else None
                fresh_ranges=valid_ranges if last_range_ns is not None and now-last_range_ns<=cfg.uwb_age_s*1e9 else 0
                uwb_reason=gate.reason if phase in ('initial_fix','acquire','diagnostic') else (
                    'waiting_surface_signal' if phase=='surface' else
                    'waiting_center_for_new_target' if phase=='reacquire' else
                    'surface_alignment' if phase=='align' else 'frozen_yaw_navigation')
                write('navigation',event='status',xyz=estimator.xyz,velocity=estimator.v.tolist(),
                      target=target,t3_remaining_s=remaining,
                      waypoint_index=0 if mission is None else mission.waypoint_index,
                      waypoint_elapsed_s=waypoint_elapsed,waypoint_remaining_s=waypoint_remaining,
                      waypoint_count=len(cfg.waypoints),distance_source='last_accepted_surface_uwb',
                      frozen_yaw_error_deg=None if mission is None else mission.frozen_yaw_error_deg,
                      final_reached=False if mission is None else mission.final_reached,
                      target_reached=False if mission is None else mission.target_reached,
                      t4_remaining_s=next_surface,uwb_reason=uwb_reason,distance_m=distance,
                      depth_target_m=depth_target,valid_ranges=fresh_ranges,
                      uwb_frames_collected=len(gate.rows),uwb_frames_required=cfg.uwb_average_frames,
                      integrating=estimator.integrating,relative_xy=estimator.relative_xy,
                      relative_xyz=estimator.relative_xyz,ekf_bias_mps2=estimator.ekf.x[6:].tolist(),
                      last_accepted_uwb_t_ns=None if gate.last_accepted is None else gate.last_accepted.t_ns,
                      depth_t_ns=estimator.last_depth,last_complete_uwb_t_ns=last_complete_uwb_ns)
                print(f'[{phase}] XYZ={tuple(round(v,3) for v in estimator.xyz)} '
                      f'd_uwb={distance}m target={target} h0={depth_target} t3={remaining} t4={next_surface} '
                      f'point_t={waypoint_remaining} '
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
                terminal_center=terminal_center,
                executor_started=executor_started,alive_workers=alive,pressure_origin_t_ns=drift.origin_t_ns,
                last_accepted_uwb=None if gate.last_accepted is None else asdict(gate.last_accepted),
                waypoint_index=0 if mission is None else mission.waypoint_index,
                target=cfg.waypoints[0] if mission is None else mission.target,
                final_reached=False if mission is None else mission.final_reached,
                target_reached=False if mission is None else mission.target_reached,
                frozen_yaw_error_deg=None if mission is None else mission.frozen_yaw_error_deg,
                xyz=None if estimator is None else estimator.xyz)
    (session/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
