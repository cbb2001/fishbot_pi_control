"""固定深度双向 PI 策略；复用 PPO 的 action2 轨迹及执行器，不加载训练网络。

用户公式中的 theta1/theta2 是绝对角度；FinAction.theta 仍为相对中位偏移。
正常深度越界只在整组结束时决策，故障/停止信号仍可立即停车。
"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .rl_actions_20260914 import FinAction, build_calibration, build_rl_trajectory, validate_trajectory
from .rl_action_scheduler_20260914 import RLActionScheduler
from .rl_sensor_calibration_20260914 import finite_float, wait_for_calibration


class FixedDepthCalibration:
    """固定策略专用经验换算；IMU 标定仍复用原接口，PPO 不受影响。"""
    def __init__(self, calibration, pressure_per_meter, drift=None):
        self.calibration = calibration
        self.drift = drift
        self.pressure_per_meter = finite_float(pressure_per_meter, 'pressure_per_meter')
        if self.pressure_per_meter <= 0:
            raise ValueError('pressure_per_meter 必须大于0（Pa/m）')

    def depth_m(self, pressure_pa):
        return (finite_float(pressure_pa, 'pressure_pa')-self.calibration.p_surface_pa)/self.pressure_per_meter

    def state_dict(self):
        result = {**self.calibration.state_dict(), 'depth_conversion': 'linear_pressure',
                'pressure_per_meter': self.pressure_per_meter,
                'depth_formula': '(pressure_pa - p_surface_pa) / pressure_per_meter'}
        if self.drift is not None:
            result.update(depth_conversion='pressure_drift_lookup',
                          depth_formula='(raw_pressure_pa - p_surface_pa - f(elapsed_s)) / pressure_per_meter',
                          pressure_drift_file='pressure_drift.json',
                          pressure_drift_origin_t_ns=self.drift.origin_t_ns,
                          pressure_drift_calibration_id=self.drift.config.get('calibration_id'),
                          depth_sample_rate_hz=10.,
                          p0_estimator='manual' if self.calibration.fixed_surface_pressure_mbar is not None else 'mean(P_i - f(t_i))')
        return result


@dataclass(frozen=True)
class ContinuousFinAction(FinAction):
    """放开 PPO 离散角度和时长约束；action2 的参数含义、轨迹保持不变。"""

    def __post_init__(self):
        finite_float(self.theta, 'theta')
        duration = finite_float(self.t, 'action duration')
        if duration <= 0 or not math.isfinite(duration * 1e9) or round(duration * 1e9) < 1:
            raise ValueError('动作时长必须为可用纳秒表示的正有限秒数')
        object.__setattr__(self, 't', duration)
        if type(self.b1) is not int or self.b1 not in (0, 1):
            raise ValueError('b1 必须为 0 或 1')
        if type(self.b2) is not int or self.b2 not in (-1, 0, 1) or (self.b1 and not self.b2):
            raise ValueError('b1=1 时 b2 必须为 +1/-1')
        if self.b1 == 0:
            object.__setattr__(self, 'b2', 0)


@dataclass(frozen=True)
class FixedFinSequence:
    next_b1: int = 0
    last_b2: int = 0

    def validate(self, action):
        if action.b1 != self.next_b1:
            raise ValueError('必须依次执行 b1=0、b1=1')
        if action.b1 and self.last_b2 and action.b2 != -self.last_b2:
            raise ValueError('相邻 b1=1 动作的 b2 必须取反')

    def after(self, action):
        self.validate(action)
        return replace(self, next_b1=1-action.b1,
                       last_b2=action.b2 if action.b1 else self.last_b2)


class FixedDiveScheduler(RLActionScheduler):
    """用用户的 b2 交替规则替换 PPO 有效运动方向规则，其余调度原样复用。"""

    def __init__(self, calibration):
        super().__init__(calibration, limits=calibration.limits)
        self.fin_sequence = FixedFinSequence()


class FixedDivePolicy:
    def __init__(self, calibration, *, h0, k=1., ki=0., initial_b2=1, action_duration_s=.6,
                 max_integral_gap_s=.3, error_scale_m=None):
        self.k = finite_float(k, 'k')
        self.ki = finite_float(ki, 'ki')
        self.max_integral_gap_s = finite_float(max_integral_gap_s, 'max_integral_gap_s')
        # 复用动作参数校验，在传感器/PWM 初始化前拒绝非法时长。
        self.action_duration_s = ContinuousFinAction(0., action_duration_s, 0, 0).t
        if self.k < 0 or self.ki < 0 or self.max_integral_gap_s <= 0:
            raise ValueError('积分允许采样间隔必须大于0，k和ki必须非负')
        if type(initial_b2) is not int or initial_b2 not in (-1, 1):
            raise ValueError('initial_b2 必须为 +1/-1')
        self.calibration = calibration
        for sid in (4, 5, 6, 7):
            lo, hi = calibration.limits[sid]
            center = calibration.centers[sid]
            if not all(math.isfinite(v) for v in (lo, hi, center)) or not lo <= center <= hi:
                raise ValueError(f'舵机 {sid} 的中位/限位无效')
        self.next_b2 = initial_b2
        self.last_b2 = None
        self.completed_pairs = 0
        self._pair = None
        self._index = 0
        self.pair_info = None
        # 无量纲积分输出 ki*∫((h0-h)/error_scale_m)dt；默认归一化尺度仍为h0。
        self.i_term = 0.
        self._control_start_ns = None
        self._last_integral_t_ns = None
        self.set_target(h0, error_scale_m=error_scale_m)
        self.integral_status = 'not_started'

    def set_target(self, h0, *, error_scale_m=None):
        """仅在组边界切换目标；0米目标须显式提供正的误差尺度，避免除零。"""
        if self.pair_active:
            raise RuntimeError('须完成当前action2组合后才能切换目标')
        target = finite_float(h0, 'h0')
        scale = target if error_scale_m is None else finite_float(error_scale_m, 'error_scale_m')
        if target < 0 or scale <= 0:
            raise ValueError('h0必须非负，误差归一化尺度必须大于0（默认使用h0）')
        self.h0, self.error_scale_m = target, scale
        self.i_term = 0.
        # 新目标不继承上一阶段的采样间隔和积分，b2交替状态则继续保留。
        self._last_integral_t_ns = None
        self.integral_status = 'stage_reset'

    def _angle_cap(self, b2):
        """复用原根部/鳍尖限幅，积分也必须服从这些物理限制。"""
        c, limits = self.calibration.centers, self.calibration.limits
        root_cap = min(*(c[s]-limits[s][0] for s in (4, 6)),
                       *(limits[s][1]-c[s] for s in (4, 6)))
        tip_room = min(limits[5][1]-c[5] if b2 > 0 else c[5]-limits[5][0],
                       c[7]-limits[7][0] if b2 > 0 else limits[7][1]-c[7])
        coupling = self.calibration.coupling
        tip_cap = tip_room * coupling['root_reference_span_deg'] / coupling['tip_reference_span_deg'] / 2.
        return min(root_cap, tip_cap)

    def start_control(self, t_ns):
        """预热、标定和舵机初始化期间不积分；漂移模型的时间起点不受影响。"""
        self._control_start_ns = int(t_ns)
        self._last_integral_t_ns = None
        self.i_term = 0.
        self.integral_status = 'waiting_for_sample'

    def observe_depth(self, depth_m, sample_t_ns):
        """每条新压力样本最多积分一次，矩形积分使用实际采样间隔（秒）。"""
        depth = finite_float(depth_m, 'depth_m')
        t_ns = int(sample_t_ns)
        if self._control_start_ns is None or t_ns < self._control_start_ns:
            return
        previous = self._last_integral_t_ns
        if previous is not None and t_ns <= previous:
            return  # 重复/乱序样本不重复累计，也不回拨积分计时。
        self._last_integral_t_ns = t_ns
        if self.ki == 0:
            self.i_term = 0.
            self.integral_status = 'disabled'
            return
        if depth == self.h0 and not self.pair_active:
            self.i_term = 0.
            self.integral_status = 'reset_at_hold'
            return
        if previous is None:
            self.integral_status = 'first_sample'
            return
        dt = (t_ns-previous)/1e9
        if dt > self.max_integral_gap_s:
            self.integral_status = 'gap_skipped'
            return
        error = (self.h0-depth)/self.error_scale_m
        span = self.calibration.limits[4][1]-self.calibration.centers[4]
        if span <= 0:
            self.i_term = 0.
            self.integral_status = 'no_angle_room'
            return
        # 两个 b2 方向均须可用；输出正负方向都做 anti-windup。
        cap = min(self._angle_cap(1), self._angle_cap(-1))/span
        increment = self.ki*error*dt
        if not math.isfinite(increment):
            raise ValueError('积分计算溢出')
        candidate = self.i_term + increment
        headroom = max(0., cap-self.k*abs(error))
        unwinding = error*self.i_term < 0
        if unwinding:
            # 误差反向时先退积分，即使比例项已饱和也允许消除历史积分。
            # 若一次更新跨过零点，新方向积分仍须受输出余量限制。
            updated = (candidate if error*candidate <= 0 else
                       math.copysign(min(abs(candidate), headroom), error))
        else:
            magnitude = max(abs(self.i_term), min(abs(candidate), headroom))
            updated = math.copysign(magnitude, error) if error else self.i_term
        self.i_term = max(-cap, min(cap, updated))
        self.integral_status = ('anti_windup' if self.i_term != candidate else
                                'unwinding' if unwinding else 'integrating')

    def pi_state(self, depth_m):
        span = self.calibration.limits[4][1]-self.calibration.centers[4]
        return {'ki': self.ki, 'target_depth_m': self.h0, 'error_scale_m': self.error_scale_m,
                'p_term_deg': self.k*((depth_m-self.h0)/self.error_scale_m)*span,
                'i_term_deg': -self.i_term*span, 'integral_status': self.integral_status}

    def next_action(self, depth_m):
        depth = finite_float(depth_m, 'depth_m')
        if self._pair is None:
            if depth == self.h0:
                self.i_term = 0.
                self.integral_status = 'reset_at_hold' if self.ki else 'disabled'
                return None  # 达到目标时不提交动作，执行器保持已确认的最后姿态。
            # 仅在组边界清除与当前误差相反的旧积分，避免越过目标后继续朝旧方向拍动。
            # 组内仍按采样退积分，已冻结的两个动作不受影响。
            if self.i_term*(self.h0-depth) < 0:
                self.i_term = 0.
                self.integral_status = 'direction_reset'
            c = self.calibration.centers
            # 深度低于目标时 delta 为负（下潜），高于目标时 delta 为正（上浮）。
            terms = self.pi_state(depth)
            delta_raw = terms['p_term_deg'] + terms['i_term_deg']
            if not math.isfinite(delta_raw):
                raise ValueError('delta 计算溢出')
            # 同时约束两根部和左右鳍尖；缩小幅值，不裁剪轨迹，保持耦合定义。
            b2 = self.next_b2
            angle_cap = self._angle_cap(b2)
            delta = math.copysign(min(abs(delta_raw), angle_cap), delta_raw) if delta_raw else 0.
            if delta == 0:
                self.integral_status = 'zero_output'
                return None
            self._pair = (ContinuousFinAction(delta, self.action_duration_s, 0, 0),
                          ContinuousFinAction(-delta, self.action_duration_s, 1, b2))
            self._index = 0
            self.pair_info = {**terms, 'pair_index': self.completed_pairs, 'depth_at_start_m': depth,
                              'delta_raw_deg': delta_raw, 'delta_deg': delta,
                              'theta1_absolute_deg': c[4]+delta,
                              'theta2_absolute_deg': c[4]-delta, 'b2': b2,
                              'direction': 'diving' if delta < 0 else 'surfacing',
                              'angle_cap_deg': angle_cap, 'saturated': abs(delta_raw) >= angle_cap,
                              'limited': delta != delta_raw, 'action_duration_s': self.action_duration_s}
        # 本组中途的深度变化不重算角度、不取消第二动作。
        return self._pair[self._index]

    def acknowledge(self, action):
        """只在 PWM 端点成功写入的完成回执后推进，避免重复提交/提前反转。"""
        if self._pair is None or action != self._pair[self._index]:
            raise RuntimeError('完成回执与当前动作不匹配')
        if self._index == 0:
            self._index = 1
        else:
            self.last_b2 = action.b2
            self.next_b2 = -action.b2
            self.completed_pairs += 1
            self._pair = None
            self._index = 0

    @property
    def pair_active(self):
        return self._pair is not None


def prepare_fixed_config(raw, *, dry_run=False, pretrain_wait_s=None, calibration_window_s=None):
    """仅提取 PPO 的采集、标定和运行配置，不依赖 Torch/PPO 超参数。"""
    cfg = copy.deepcopy(raw)
    cfg.setdefault('startup', {}).setdefault('calibration_window_s', 5.)
    if dry_run:
        sim = cfg.get('simulation', {})
        for key in ('pretrain_wait_s', 'sensor_warmup_s', 'calibration_window_s', 'calibration_timeout_s'):
            if key in sim:
                cfg['startup'][key] = sim[key]
        for key in ('minimum_imu_samples', 'minimum_depth_samples'):
            if key in sim:
                cfg['calibration'][key] = sim[key]
    if pretrain_wait_s is not None:
        cfg['startup']['pretrain_wait_s'] = pretrain_wait_s
    if calibration_window_s is not None:
        cfg['startup']['calibration_window_s'] = calibration_window_s
        # 显式加长模拟窗口时，保留实机的重试预算，避免5秒窗口与5秒超时重合。
        if dry_run:
            cfg['startup']['calibration_timeout_s'] = raw['startup']['calibration_timeout_s']
    for section, keys, zero_ok in (
        ('startup', ('pretrain_wait_s', 'sensor_warmup_s'), True),
        ('startup', ('calibration_window_s', 'calibration_timeout_s'), False),
        ('runtime', ('poll_interval_s', 'shutdown_timeout_s', 'sensor_ready_timeout_s',
                     'action_timeout_margin_s', 'live_status_interval_s'), False),
    ):
        for key in keys:
            value = finite_float(cfg[section][key], f'{section}.{key}')
            if value < 0 or (not zero_ok and value == 0):
                raise ValueError(f'{section}.{key} 无效')
            cfg[section][key] = value
    if cfg['startup']['calibration_timeout_s'] <= cfg['startup']['calibration_window_s']:
        raise ValueError('标定超时必须大于标定窗口，以留出完成检查和重试时间')
    return cfg


def run_session(cfg, robot_cfg, session, *, h0, k=1., ki=0., initial_b2=1, dry_run=False,
                max_control_s=None, keep_pwm=False, stop_event=None, action_duration_s=.6,
                surface_pressure_pa=None, pressure_per_meter=None, pressure_drift_config=None):
    from .pressure_drift_20261001 import PressureDrift, CorrectedSensors
    from .data_logger import JsonlLogger, RawSensorLoggers, write_metadata_yaml
    from .rl_live_status_20260914 import LiveStatusPublisher
    from .rl_safety_20260914 import RLSafetyMonitor, RLSafetyFault
    from .rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
    from .rl_servo_state_tracker_20260914 import RLServoStateTracker
    from .sensor_manager import SensorManager

    calibration = build_calibration(robot_cfg)
    policy = FixedDivePolicy(calibration, h0=h0, k=k, ki=ki, initial_b2=initial_b2,
                             action_duration_s=action_duration_s,
                             max_integral_gap_s=cfg.get('state', {}).get('sensor_age', {}).get('depth', {}).get('fault_ms', 300.)/1000.)
    if max_control_s is not None:
        max_control_s = finite_float(max_control_s, 'max_control_s')
        if max_control_s <= 0:
            raise ValueError('max_control_s 必须大于 0')
    # 独立配置接口默认 null：不再隐式使用旧 robot.yaml 的固定驱动零点。
    depth_cfg = cfg.get('fixed_depth', {}) or {}
    drift_path = pressure_drift_config or depth_cfg.get('pressure_drift_config') or (
        Path(__file__).resolve().parents[2] / 'config/pressure_drift_10hz.json')
    drift = PressureDrift(drift_path)
    if surface_pressure_pa is None:
        surface_pressure_pa = depth_cfg.get('surface_pressure_pa')
    if pressure_per_meter is None:
        pressure_per_meter = depth_cfg.get('pressure_per_meter')
    if pressure_per_meter is None:
        pressure_per_meter = (finite_float(cfg['calibration'].get('water_density_kg_m3', 1000.), 'water density')
                              * finite_float(cfg['calibration'].get('gravity_mps2', 9.80665), 'gravity'))
    pressure_per_meter = finite_float(pressure_per_meter, 'pressure_per_meter')
    if pressure_per_meter <= 0:
        raise ValueError('pressure_per_meter 必须大于0（Pa/m）')
    if surface_pressure_pa is not None:
        surface_pressure_pa = finite_float(surface_pressure_pa, 'surface_pressure_pa')
        if surface_pressure_pa <= 0:
            raise ValueError('人工标定压力必须大于0（Pa）')
    stop = stop_event if stop_event is not None else threading.Event()
    failures = queue.Queue()
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    poll_s, shutdown_s = cfg['runtime']['poll_interval_s'], cfg['runtime']['shutdown_timeout_s']
    scheduler = FixedDiveScheduler(calibration)
    tracker = RLServoStateTracker(calibration)
    if dry_run:
        controller = DryRunRLServoController(calibration.centers)
    else:
        def controller():
            from drivers.pca9685_servo import PCA9685ServoController
            hardware = copy.deepcopy(robot_cfg)
            hardware['servo']['channels'] = [v for v in hardware['servo']['channels']
                                              if int(v['servo_id']) in (4, 5, 6, 7)]
            return PCA9685ServoController(hardware)
    ex = RLServoExecutor(controller, scheduler, tracker,
        config={'command_hz': 50., 'initial_move_s': 0. if dry_run else 2.,
                'safe_recenter_s': 0. if dry_run else 2.},
        active_servo_ids=(4, 5, 6, 7), shutdown_event=stop, failure_queue=failures,
        keep_pwm=keep_pwm, servo_id_to_channel={i: int(v['channel']) for i, v in calibration.servos.items()})
    sensor_cfg = copy.deepcopy(robot_cfg)
    # 仅本次固定下潜会话覆盖为模型要求的 10 Hz，不修改 robot.yaml/PPO。
    sensor_cfg.setdefault('sensors', {}).setdefault('depth', {})['rate_hz'] = 10.
    sensor_cfg['calibration'] = dict(cfg['calibration'])
    # 驱动自带 depth_m 不参与决策；兼容旧驱动不接受 null 的构造参数。
    if sensor_cfg.setdefault('depth_sensor', {}).get('surface_pressure_mbar') is None:
        sensor_cfg['depth_sensor']['surface_pressure_mbar'] = 1013.25
    if dry_run:
        # 只调整模拟压力轨迹斜率，让模拟深度与本次经验系数一致。
        sensor_cfg['calibration']['water_density_kg_m3'] = pressure_per_meter / float(cfg['calibration'].get('gravity_mps2', 9.80665))
    sensor_cfg['sensors'].setdefault('vision', {})['enabled'] = False
    # UWB沿用robot配置，仅由RawSensorLoggers保存原始样本，不参与下潜控制。
    if dry_run:
        from .rl_mock_sensors_20260914 import MockRLSensorManager
        sensors = MockRLSensorManager(sensor_cfg, stop_event=stop, log_dir=session)
        # 模拟原始压力包含同一漂移，确保标定/修正后仍还原模拟真实深度。
        sensors._create_workers()
        sensors._created = True
        for worker in sensors.workers:
            if worker.name == 'depth':
                original_read = worker.read_once
                def read_with_drift(read=original_read):
                    sample = read()
                    drift.observe([sample])
                    sample.data['pressure_pa'] += drift.drift_pa(drift.elapsed_s(sample.t_ns))
                    return sample
                worker.read_once = read_with_drift
    else:
        sensors = SensorManager(sensor_cfg, mock=False, stop_event=stop, log_dir=session)
    corrected_sensors = CorrectedSensors(sensors, drift)
    guard = RLSafetyMonitor(corrected_sensors, ex, failures, stop, cfg, robot_cfg)
    if policy.h0 >= guard.limits['max_depth_m']:
        raise ValueError('h0 必须小于配置的安全深度上限')
    logs = {name: JsonlLogger(session / name) for name in ('events.jsonl', 'commands.jsonl', 'depth_control.jsonl')}
    raw = RawSensorLoggers(session)
    status = LiveStatusPublisher(session / 'fixed_dive_status.json', cfg['runtime']['live_status_interval_s'])
    sensor_cal = None
    depth = None
    last_depth_t = -1
    phase = 'startup'
    fault = None
    cleanup_errors = []

    def write(name, row):
        if not logs[name].write({'t_ns': time.monotonic_ns(), **row}):
            raise RuntimeError(f'日志队列已满：{name}')

    def publish(force=False):
        status.publish({'phase': phase, 'depth_m': depth, 'target_depth_m': policy.h0,
                        'k': policy.k, 'ki': policy.ki, 'action_duration_s': policy.action_duration_s,
                        'pi': None if depth is None else policy.pi_state(depth),
                        'completed_pairs': policy.completed_pairs,
                        'last_b2': policy.last_b2, 'next_b2': policy.next_b2,
                        'controlled_servo_ids': [4, 5, 6, 7],
                        'pair': policy.pair_info, 'fault': fault}, force=force)

    def service():
        nonlocal depth, last_depth_t
        guard.raise_if_failed()
        if stop.is_set():
            raise RLSafetyFault('STOP_REQUESTED', '收到停止请求')
        if ex.failure is not None:
            raise ex.failure
        raw.write_from_buffers(sensors.buffers)
        sample = corrected_sensors.buffers['depth'].latest()
        if sensor_cal is not None and sample is not None and sample.t_ns > last_depth_t:
            if not sample.ok:
                raise RLSafetyFault('DEPTH_INVALID', str(sample.error))
            depth = sensor_cal.depth_m(sample.data['pressure_pa'])
            policy.observe_depth(depth, sample.t_ns)
            last_depth_t = sample.t_ns
            write('depth_control.jsonl', {**policy.pi_state(depth), 'sample_t_ns': sample.t_ns, 'depth_m': depth,
                                        'raw_pressure_pa': sample.data['raw_pressure_pa'],
                                        'corrected_pressure_pa': sample.data['pressure_pa'],
                                        'pressure_drift_pa': sample.data['pressure_drift_pa'],
                                        'drift_elapsed_s': sample.data['drift_elapsed_s'],
                                        'surface_pressure_pa': sensor_cal.calibration.p_surface_pa + sample.data['pressure_drift_pa'],
                                        'phase': phase, 'pair_active': policy.pair_active})
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.last_error or logger.dropped_count:
                raise RuntimeError(f'日志故障：{logger.path}: {logger.last_error}')
        publish()

    def wait(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            service()
            stop.wait(min(poll_s, max(0., end-time.monotonic())))
        service()

    try:
        for logger in logs.values():
            logger.start()
        raw.start()
        write_metadata_yaml(session / 'metadata.yaml', {
            'runtime_config': cfg, 'robot_config': sensor_cfg, 'dry_run': dry_run,
            'h0': h0, 'k': k, 'ki': ki, 'initial_b2': initial_b2, 'max_control_s': max_control_s,
            'max_integral_gap_s': policy.max_integral_gap_s,
            'surface_pressure_pa': surface_pressure_pa, 'pressure_per_meter': pressure_per_meter,
            'action_duration_s': policy.action_duration_s})
        # 会话保存查表快照，后续离线分析不依赖可能变化的配置文件。
        (session / 'pressure_drift.json').write_text(json.dumps(drift.config, ensure_ascii=False), encoding='utf-8')
        sensors.start_all()
        guard.start()
        wait(cfg['startup']['pretrain_wait_s'])
        wait(cfg['startup']['sensor_warmup_s'])
        phase = 'calibration'
        cal_cfg = dict(cfg['calibration'])
        cal_cfg.pop('surface_pressure_mbar', None)
        if surface_pressure_pa is not None:
            cal_cfg['surface_pressure_mbar'] = surface_pressure_pa / 100.
        else:
            # 10 Hz 的窗口边界可能少一条；保留原来的覆盖时长及稳定性检查。
            cal_cfg.setdefault('minimum_depth_samples', max(3, math.floor(cfg['startup']['calibration_window_s']*10*.9)))
        merged = {**cfg, 'sensors': sensor_cfg['sensors'], 'calibration': cal_cfg}
        sensor_cal = FixedDepthCalibration(
            wait_for_calibration(corrected_sensors, merged, stop_event=stop, on_check=service), pressure_per_meter, drift)
        guard.calibration = sensor_cal
        (session / 'calibration.json').write_text(json.dumps(sensor_cal.state_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        end = time.monotonic()+cfg['runtime']['sensor_ready_timeout_s']
        while any(sensors.buffers[n].latest() is None for n in ('imu', 'depth', 'power')):
            if time.monotonic() > end:
                raise TimeoutError('关键传感器未就绪')
            wait(poll_s)
        guard.check_once()
        phase = 'servo_initialization'
        ex.start()
        guard.executor_started = True
        end = time.monotonic()+10.
        while not ex.wait_ready(poll_s):
            service()
            if time.monotonic() > end:
                raise TimeoutError('侧鳍初始化超时')
        if dry_run:
            sensors.set_surface_mode(False)
        start = time.monotonic()
        policy.start_control(time.monotonic_ns())
        while True:
            service()
            # 常规运行时限也等整组结束；Ctrl+C/故障不受此限制。
            if not policy.pair_active and max_control_s is not None and time.monotonic()-start >= max_control_s:
                break
            action = policy.next_action(depth)
            if action is None:
                if phase != 'holding':
                    phase = 'holding'
                    write('events.jsonl', {'event': 'hold_endpoint', 'depth_m': depth})
                stop.wait(poll_s)  # 不用 action2 做定时保持，下一轮即可响应新深度。
                continue
            phase = policy.pair_info['direction']
            # 在进入执行器前再次验证完整轨迹；保留原轨迹函数及镜像关系。
            validate_trajectory(build_rl_trajectory(action, scheduler.previous_theta['fin'], calibration), calibration.limits)
            rid = ex.submit('fin', action, policy.pair_info)
            write('commands.jsonl', {'request_id': rid, 'action': action.to_dict(),
                                    'theta_absolute_deg': calibration.centers[4]+action.theta,
                                    'pair': policy.pair_info})
            deadline = time.monotonic()+action.t+cfg['runtime']['action_timeout_margin_s']
            completed = None
            while completed is None:
                service()  # 持续采集、监测；目标深度不触发取消当前组合。
                ex.drain_started()  # 消费回执，长时间运行不积累队列。
                for item in ex.drain_completed():
                    if item.request_id != rid:
                        raise RuntimeError('非预期动作完成回执')
                    completed = item
                if completed is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('action2 执行超时')
                    stop.wait(poll_s)
            if not completed.endpoint_written:
                raise RuntimeError('action2 端点 PWM 未确认')
            policy.acknowledge(completed.action)
            write('events.jsonl', {'event': 'action_completed', 'request_id': rid,
                                  'b1': action.b1, 'b2': action.b2,
                                  'completed_pairs': policy.completed_pairs})
    except BaseException as exc:
        ex.request_emergency_stop()
        fault = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        def cleanup(name, operation):
            try:
                operation()
            except BaseException as exc:
                cleanup_errors.append({'component': name, 'message': str(exc)})
                ex.request_emergency_stop()
        # 与 PPO 相同：正常退出回中，故障立即关 PWM；目标保持期间不回中。
        guard.executor_started = False
        cleanup('servo', lambda: ex.stop(timeout=shutdown_s))
        cleanup('safety', lambda: guard.stop(timeout=shutdown_s))
        if fault is None and (guard.failure is not None or ex.failure is not None):
            fault = {'message': str(guard.failure or ex.failure)}
        stop.set()
        cleanup('sensors', lambda: sensors.stop_all(timeout=shutdown_s))
        if sensors.alive_worker_names():
            cleanup_errors.append({'component': 'sensors', 'message': str(sensors.alive_worker_names())})
        # 正常退出回中期间仍在采样；停采后补写最后一批，避免遗漏收尾数据。
        cleanup('raw_drain', lambda: raw.write_from_buffers(sensors.buffers))
        cleanup('raw_logs', raw.stop)
        for name, logger in logs.items():
            cleanup(name, logger.stop)
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.is_alive() or logger.last_error or logger.dropped_count:
                cleanup_errors.append({'component': str(logger.path), 'message': logger.last_error or '日志丢失/未退出'})
        if cleanup_errors and fault is None:
            fault = {'message': str(cleanup_errors)}
        phase = 'fault' if fault else 'finished'
        cleanup('status', lambda: publish(force=True))
        (session / 'result.json').write_text(json.dumps({
            'completed_pairs': policy.completed_pairs, 'fault': fault,
            'cleanup_errors': cleanup_errors}, ensure_ascii=False, indent=2), encoding='utf-8')
    if fault or cleanup_errors:
        raise RuntimeError(f'固定下潜退出：{fault or cleanup_errors}；日志：{session}')
    return {'session': str(session), 'completed_pairs': policy.completed_pairs}
