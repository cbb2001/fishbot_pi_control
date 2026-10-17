"""侧鳍固定深度控制与尾鳍左右摆动的联合控制器（2026-10-09）。

侧鳍沿用 ``fixed_dive_20260920_100122`` 的 action2、压力漂移修正、标定和
安全监测逻辑；尾鳍沿用 20260725 三个尾鳍舵机同角度联动的绝对角度轨迹。
侧鳍浅于目标时下潜，深于目标时上浮；尾鳍由同一个执行器并行输出。
"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .rl_actions_20260914 import (
    TailAction as _DiscreteTailAction,
    build_calibration,
    build_rl_trajectory,
    validate_trajectory,
)
from .fixed_dive_20260920_100122 import (
    FixedDepthCalibration,
    FixedDivePolicy,
    FixedDiveScheduler,
    prepare_fixed_config,
)
from .rl_sensor_calibration_20260914 import finite_float, wait_for_calibration


@dataclass(frozen=True)
class VariableTailAction(_DiscreteTailAction):
    """允许连续角度和时长的尾鳍 action1，便于使用 -15/+15°、0.3 s 默认值。"""

    def __post_init__(self):
        theta = finite_float(self.theta, 'tail theta')
        duration = finite_float(self.t, 'tail action duration')
        if duration <= 0:
            raise ValueError('尾鳍摆动时间必须大于0')
        object.__setattr__(self, 'theta', theta)
        object.__setattr__(self, 't', duration)


@dataclass(frozen=True)
class TailStage:
    duration_s: float
    target_depth_m: float
    name: str


DEFAULT_STAGES = (
    TailStage(30., 1., 'dive_1m'),
    TailStage(30., 0., 'surface_1'),
    TailStage(30., 1.5, 'dive_1_5m'),
    TailStage(30., 0., 'surface_2'),
)


def _validate_stages(stages):
    result = tuple(stages)
    if not result:
        raise ValueError('必须配置至少1个阶段')
    for stage in result:
        if (not isinstance(stage, TailStage)
                or not math.isfinite(stage.duration_s) or stage.duration_s <= 0
                or not math.isfinite(stage.target_depth_m) or stage.target_depth_m < 0):
            raise ValueError('阶段时间必须大于0，目标深度不能为负')
    return result


def _stage_error_scale(stages, stage_index):
    """正目标按h0归一化；0米上浮阶段沿用最近下潜深度（默认1或1.5米）。"""
    for stage in reversed(stages[:stage_index+1]):
        if stage.target_depth_m > 0:
            return stage.target_depth_m
    return 1.  # 自定义流程从0米开始时，以1米作为误差尺度，不虚构目标深度。


def _make_tail_action(calibration, theta, duration_s):
    action = VariableTailAction(theta, duration_s)
    for sid in (1, 2, 3):
        target = calibration.centers[sid] + theta
        lo, hi = calibration.limits[sid]
        if not lo <= target <= hi:
            raise ValueError(f'尾鳍舵机 {sid} 目标角 {target:g} 超出范围 [{lo:g}, {hi:g}]')
    return action


def run_session(cfg, robot_cfg, session, *, k=1., ki=0., initial_b2=1,
                side_action_duration_s=.6, tail_action_duration_s=.3,
                tail_left_deg=-15., tail_right_deg=15., stages=DEFAULT_STAGES,
                dry_run=False, max_control_s=None, keep_pwm=False,
                stop_event=None, surface_pressure_pa=None,
                pressure_per_meter=None, pressure_drift_config=None):
    """运行任意数量阶段的联合控制；侧鳍组内动作完成后才允许目标阶段生效。"""
    from .pressure_drift_20261001 import PressureDrift, CorrectedSensors
    from .data_logger import JsonlLogger, RawSensorLoggers, write_metadata_yaml
    from .rl_live_status_20260914 import LiveStatusPublisher
    from .rl_safety_20260914 import RLSafetyMonitor, RLSafetyFault
    from .rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
    from .rl_servo_state_tracker_20260914 import RLServoStateTracker
    from .sensor_manager import SensorManager

    stages = _validate_stages(stages)
    tail_action_duration_s = finite_float(tail_action_duration_s, 'tail_action_duration_s')
    tail_left_deg = finite_float(tail_left_deg, 'tail_left_deg')
    tail_right_deg = finite_float(tail_right_deg, 'tail_right_deg')
    if tail_action_duration_s <= 0 or tail_left_deg >= tail_right_deg:
        raise ValueError('尾鳍时长必须大于0，左摆角必须小于右摆角')
    calibration = build_calibration(robot_cfg)
    policy = FixedDivePolicy(calibration, h0=stages[0].target_depth_m,
                             error_scale_m=_stage_error_scale(stages, 0),
                             k=k, ki=ki, initial_b2=initial_b2,
                             action_duration_s=side_action_duration_s,
                             max_integral_gap_s=cfg.get('state', {}).get('sensor_age', {})
                             .get('depth', {}).get('fault_ms', 300.) / 1000.)
    # 默认在所有阶段总时长后结束，正常结束仍等当前侧鳍组合完成。
    if max_control_s is None:
        max_control_s = sum(stage.duration_s for stage in stages)
    max_control_s = finite_float(max_control_s, 'max_control_s')
    if max_control_s <= 0:
        raise ValueError('max_control_s 必须大于0')
    depth_cfg = cfg.get('fixed_depth', {}) or {}
    drift_path = pressure_drift_config or depth_cfg.get('pressure_drift_config') or (
        Path(__file__).resolve().parents[2] / 'config/pressure_drift_10hz.json')
    drift = PressureDrift(drift_path)
    if surface_pressure_pa is None:
        surface_pressure_pa = depth_cfg.get('surface_pressure_pa')
    if pressure_per_meter is None:
        pressure_per_meter = depth_cfg.get('pressure_per_meter')
    if pressure_per_meter is None:
        pressure_per_meter = (float(cfg['calibration'].get('water_density_kg_m3', 1000.))
                              * float(cfg['calibration'].get('gravity_mps2', 9.80665)))
    pressure_per_meter = finite_float(pressure_per_meter, 'pressure_per_meter')
    if pressure_per_meter <= 0:
        raise ValueError('pressure_per_meter 必须大于0')
    if surface_pressure_pa is not None:
        surface_pressure_pa = finite_float(surface_pressure_pa, 'surface_pressure_pa')
        if surface_pressure_pa <= 0:
            raise ValueError('人工标定压力必须大于0')

    stop = stop_event or threading.Event()
    failures = queue.Queue()
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    scheduler = FixedDiveScheduler(calibration)
    tracker = RLServoStateTracker(calibration)
    if dry_run:
        controller = DryRunRLServoController(calibration.centers)
    else:
        def controller():
            from drivers.pca9685_servo import PCA9685ServoController
            hardware = copy.deepcopy(robot_cfg)
            hardware['servo']['channels'] = [v for v in hardware['servo']['channels']
                                              if int(v['servo_id']) in range(1, 8)]
            return PCA9685ServoController(hardware)
    ex = RLServoExecutor(controller, scheduler, tracker,
        config={'command_hz': 50., 'initial_move_s': 0. if dry_run else 2.,
                'safe_recenter_s': 0. if dry_run else 2.},
        active_servo_ids=(1, 2, 3, 4, 5, 6, 7), shutdown_event=stop,
        failure_queue=failures, keep_pwm=keep_pwm,
        servo_id_to_channel={i: int(v['channel']) for i, v in calibration.servos.items()})
    sensor_cfg = copy.deepcopy(robot_cfg)
    sensor_cfg.setdefault('sensors', {}).setdefault('depth', {})['rate_hz'] = 10.
    sensor_cfg['calibration'] = dict(cfg['calibration'])
    if sensor_cfg.setdefault('depth_sensor', {}).get('surface_pressure_mbar') is None:
        sensor_cfg['depth_sensor']['surface_pressure_mbar'] = 1013.25
    if dry_run:
        sensor_cfg['calibration']['water_density_kg_m3'] = pressure_per_meter / float(cfg['calibration'].get('gravity_mps2', 9.80665))
    for name in ('vision', 'uwb'):
        sensor_cfg['sensors'].setdefault(name, {})['enabled'] = False
    if dry_run:
        from .rl_mock_sensors_20260914 import MockRLSensorManager
        sensors = MockRLSensorManager(sensor_cfg, stop_event=stop, log_dir=session)
        sensors._create_workers(); sensors._created = True
        for worker in sensors.workers:
            if worker.name == 'depth':
                original_read = worker.read_once
                def read_with_drift(read=original_read):
                    sample = read(); drift.observe([sample])
                    sample.data['pressure_pa'] += drift.drift_pa(drift.elapsed_s(sample.t_ns))
                    return sample
                worker.read_once = read_with_drift
    else:
        sensors = SensorManager(sensor_cfg, mock=False, stop_event=stop, log_dir=session)
    corrected_sensors = CorrectedSensors(sensors, drift)
    guard = RLSafetyMonitor(corrected_sensors, ex, failures, stop, cfg, robot_cfg)
    if max(stage.target_depth_m for stage in stages) >= guard.limits['max_depth_m']:
        raise ValueError('阶段目标深度必须小于安全深度上限')
    logs = {name: JsonlLogger(session / name) for name in ('events.jsonl', 'commands.jsonl', 'depth_control.jsonl')}
    raw = RawSensorLoggers(session)
    status = LiveStatusPublisher(session / 'fixed_dive_tail_status.json', cfg['runtime']['live_status_interval_s'])
    sensor_cal = None; depth = None; last_depth_t = -1; phase = 'startup'; fault = None; cleanup_errors = []

    def write(name, row):
        if not logs[name].write({'t_ns': time.monotonic_ns(), **row}):
            raise RuntimeError(f'日志队列已满：{name}')
    def publish(force=False):
        status.publish({'phase': phase, 'depth_m': depth, 'stage_index': stage_index,
                        'stage_name': stages[stage_index].name, 'target_depth_m': target_depth,
                        'pi': None if depth is None else policy.pi_state(depth),
                        'k': policy.k, 'ki': policy.ki, 'side_action_duration_s': policy.action_duration_s,
                        'tail_action_duration_s': tail_action_duration_s,
                        'tail_left_deg': tail_left_deg, 'tail_right_deg': tail_right_deg,
                        'completed_pairs': policy.completed_pairs, 'pair': policy.pair_info,
                        'tail_direction': tail_direction, 'tail_inflight': tail_inflight,
                        'executor': ex.snapshot(), 'fault': fault}, force=force)
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
            if not sample.ok: raise RLSafetyFault('DEPTH_INVALID', str(sample.error))
            depth = sensor_cal.depth_m(sample.data['pressure_pa']); policy.observe_depth(depth, sample.t_ns)
            last_depth_t = sample.t_ns
            write('depth_control.jsonl', {**policy.pi_state(depth), 'sample_t_ns': sample.t_ns,
                'depth_m': depth, 'raw_pressure_pa': sample.data['raw_pressure_pa'],
                'corrected_pressure_pa': sample.data['pressure_pa'], 'pressure_drift_pa': sample.data['pressure_drift_pa'],
                'stage_index': stage_index, 'target_depth_m': target_depth, 'phase': phase,
                'pair_active': policy.pair_active})
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.last_error or logger.dropped_count: raise RuntimeError(f'日志故障：{logger.path}')
        publish()
    def wait(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            service(); stop.wait(min(cfg['runtime']['poll_interval_s'], max(0., end-time.monotonic())))
        service()

    stage_index = 0; target_depth = stages[0].target_depth_m; tail_direction = tail_left_deg
    tail_inflight = False; fin_inflight = False; control_start = None
    try:
        for logger in logs.values(): logger.start()
        raw.start()
        write_metadata_yaml(session / 'metadata.yaml', {'runtime_config': cfg, 'robot_config': sensor_cfg,
            'dry_run': dry_run, 'controller': 'fixed_dive_tail_20261009', 'stages': [s.__dict__ for s in stages],
            'k': k, 'ki': ki, 'initial_b2': initial_b2, 'tail_action_duration_s': tail_action_duration_s,
            'side_action_duration_s': policy.action_duration_s, 'max_control_s': max_control_s,
            'stage_error_scales_m': [_stage_error_scale(stages, i) for i in range(len(stages))],
            'tail_left_deg': tail_left_deg, 'tail_right_deg': tail_right_deg,
            'surface_pressure_pa': surface_pressure_pa, 'pressure_per_meter': pressure_per_meter})
        (session / 'pressure_drift.json').write_text(json.dumps(drift.config, ensure_ascii=False), encoding='utf-8')
        sensors.start_all(); guard.start()
        wait(cfg['startup']['pretrain_wait_s']); wait(cfg['startup']['sensor_warmup_s']); phase = 'calibration'
        cal_cfg = dict(cfg['calibration']); cal_cfg.pop('surface_pressure_mbar', None)
        if surface_pressure_pa is not None: cal_cfg['surface_pressure_mbar'] = surface_pressure_pa / 100.
        else: cal_cfg.setdefault('minimum_depth_samples', max(3, math.floor(cfg['startup']['calibration_window_s']*10*.9)))
        merged = {**cfg, 'sensors': sensor_cfg['sensors'], 'calibration': cal_cfg}
        sensor_cal = FixedDepthCalibration(wait_for_calibration(corrected_sensors, merged, stop_event=stop, on_check=service), pressure_per_meter, drift)
        guard.calibration = sensor_cal
        (session / 'calibration.json').write_text(json.dumps(sensor_cal.state_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        end = time.monotonic() + cfg['runtime']['sensor_ready_timeout_s']
        while any(sensors.buffers[n].latest() is None for n in ('imu', 'depth', 'power')):
            if time.monotonic() > end: raise TimeoutError('关键传感器未就绪')
            wait(cfg['runtime']['poll_interval_s'])
        guard.check_once(); phase = 'servo_initialization'; ex.start(); guard.executor_started = True
        end = time.monotonic() + 10.
        while not ex.wait_ready(cfg['runtime']['poll_interval_s']):
            service()
            if time.monotonic() > end: raise TimeoutError('舵机初始化超时')
        if dry_run: sensors.set_surface_mode(False)
        control_start = time.monotonic(); policy.start_control(time.monotonic_ns())
        while True:
            service(); elapsed = time.monotonic() - control_start
            # 先消费完成回执，再处理阶段边界，避免立即开新组导致阶段一直无法切换。
            ex.drain_started()
            for item in ex.drain_completed():
                if not item.endpoint_written:
                    raise RuntimeError(f'{item.agent} 端点PWM未确认')
                if item.agent == 'tail':
                    tail_inflight = False
                elif item.agent == 'fin':
                    fin_inflight = False
                    policy.acknowledge(item.action)
                else:
                    raise RuntimeError(f'非预期动作完成回执：{item.agent}')
                write('events.jsonl', {'event': 'action_completed', 'request_id': item.request_id,
                                      'agent': item.agent, 'stage_index': stage_index,
                                      'completed_pairs': policy.completed_pairs})
            control_expired = elapsed >= max_control_s
            if not policy.pair_active and control_expired:
                break
            # 使用累计阶段边界；正在执行的侧鳍组合结束后才切换目标。
            boundary = sum(s.duration_s for s in stages[:stage_index+1])
            while not control_expired and stage_index < len(stages)-1 and elapsed >= boundary:
                if policy.pair_active: break
                stage_index += 1; boundary = sum(s.duration_s for s in stages[:stage_index+1])
                target_depth = stages[stage_index].target_depth_m
                policy.set_target(target_depth, error_scale_m=_stage_error_scale(stages, stage_index))
                write('events.jsonl', {'event': 'stage_changed', 'stage_index': stage_index,
                                      'target_depth_m': target_depth,
                                      'error_scale_m': policy.error_scale_m,
                                      'integral_status': policy.integral_status})
            target_depth = stages[stage_index].target_depth_m

            # 尾鳍动作独立于侧鳍动作，每次完成后反向；执行器允许二者同时运行。
            if not tail_inflight and not control_expired:
                action = _make_tail_action(calibration, tail_direction, tail_action_duration_s)
                validate_trajectory(build_rl_trajectory(action, scheduler.previous_theta['tail'], scheduler), calibration.limits)
                rid = ex.submit('tail', action, {'stage_index': stage_index, 'direction': tail_direction})
                tail_inflight = True
                write('commands.jsonl', {'request_id': rid, 'agent': 'tail', 'action': action.to_dict(),
                                         'stage_index': stage_index})
                tail_direction = tail_right_deg if tail_direction == tail_left_deg else tail_left_deg
            # 侧鳍第一、第二个 action2 都要依次提交；组合中途不因深度变化取消。
            # 目标深度两侧都需要动作：浅于目标下潜，深于目标上浮；等于目标才保持。
            fin_idle = not fin_inflight
            should_start_pair = (not policy.pair_active and not control_expired
                                 and depth is not None)
            if fin_idle and (policy.pair_active or should_start_pair):
                # 等于目标时也交给共享策略判断，以同步执行保持时的积分清零。
                action = policy.next_action(depth if depth is not None else 0.)
                if action is not None:
                    phase = policy.pair_info['direction']
                    validate_trajectory(build_rl_trajectory(action, scheduler.previous_theta['fin'], scheduler), calibration.limits)
                    rid = ex.submit('fin', action, policy.pair_info)
                    fin_inflight = True
                    write('commands.jsonl', {'request_id': rid, 'agent': 'fin', 'action': action.to_dict(),
                                             'pair': policy.pair_info, 'stage_index': stage_index})
                else:
                    phase = 'holding'
            elif fin_idle and not policy.pair_active:
                phase = 'holding'
            stop.wait(cfg['runtime']['poll_interval_s'])
    except BaseException as exc:
        ex.request_emergency_stop(); fault = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        def cleanup(name, operation):
            try: operation()
            except BaseException as exc:
                cleanup_errors.append({'component': name, 'message': str(exc)}); ex.request_emergency_stop()
        guard.executor_started = False
        cleanup('servo', lambda: ex.stop(timeout=cfg['runtime']['shutdown_timeout_s']))
        cleanup('safety', lambda: guard.stop(timeout=cfg['runtime']['shutdown_timeout_s']))
        if fault is None and (guard.failure is not None or ex.failure is not None): fault = {'message': str(guard.failure or ex.failure)}
        stop.set(); cleanup('sensors', lambda: sensors.stop_all(timeout=cfg['runtime']['shutdown_timeout_s']))
        cleanup('raw_drain', lambda: raw.write_from_buffers(sensors.buffers)); cleanup('raw_logs', raw.stop)
        for name, logger in logs.items(): cleanup(name, logger.stop)
        phase = 'fault' if fault else 'finished'; cleanup('status', lambda: publish(force=True))
        (session / 'result.json').write_text(json.dumps({'completed_pairs': policy.completed_pairs, 'fault': fault,
            'cleanup_errors': cleanup_errors, 'stage_index': stage_index}, ensure_ascii=False, indent=2), encoding='utf-8')
    if fault or cleanup_errors: raise RuntimeError(f'联合控制退出：{fault or cleanup_errors}；日志：{session}')
    return {'session': str(session), 'completed_pairs': policy.completed_pairs, 'stage_index': stage_index}

