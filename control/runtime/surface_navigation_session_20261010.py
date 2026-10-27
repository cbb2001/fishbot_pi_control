"""水面导航：异步IMU/UWB采集，缺基站时复用完整定位，支持只测传感器。"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from .data_logger import JsonlLogger, RawSensorLoggers, write_metadata_yaml
from .rl_actions_20260914 import build_calibration
from .rl_sensor_calibration_20260914 import finite_float
from .rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
from .rl_servo_state_tracker_20260914 import RLServoStateTracker
from .sensor_manager import SensorManager
from .sensor_worker import BaseSensorWorker
from .surface_navigation_20261010 import (
    CalibratedIMU, NavigationPolicy, NavigationScheduler, NavigationTailAction,
    UWBStreamDecoder, position_from_ranges, validate_waypoints,
)


class _SurfaceMockWorker(BaseSensorWorker):
    """模拟真实mc字节帧及跨±180°的静止IMU；不声称模拟鱼的水动力学。"""
    def __init__(self, name, buffer, cfg, stop_event):
        super().__init__(name, buffer, stop_event=stop_event,
                         loop_delay_s=.005 if name == 'imu' else .1)
        self.cfg = cfg
        self.index = 0

    def read_once(self):
        self.index += 1
        if self.name == 'imu':
            yaw = 179.95 if self.index % 2 else -179.95
            return self.make_sample(dict(yaw_deg=yaw, pitch_deg=.02*math.sin(self.index), roll_deg=0.))
        c = self.cfg
        x, y = c.L*.25, c.W*.5
        height = c.H if c.H is not None else 1.
        distances = [round(math.sqrt((x-a)**2+(y-b)**2+height*height)*1000)
                     for a, b in ((0, 0), (c.L, 0), (c.L, c.W), (0, c.W))]
        packet = ('mc 00 ' + ' '.join(f'{r:08x}' for r in distances)
                  + f' 4705 {self.index % 256:02x} {self.index:08x} {c.uwb_tag} 0000\r\n')
        return self.make_sample({'raw_hex': packet.encode('ascii').hex(),
                                 'raw_text': packet.strip(), 'parsed': False})


class SurfaceMockSensors(SensorManager):
    def __init__(self, robot, cfg, stop_event):
        super().__init__(robot, mock=True, stop_event=stop_event)
        self.navigation_config = cfg

    def _create_mock_workers(self):
        self.workers = [_SurfaceMockWorker(name, self.buffers[name], self.navigation_config, self.stop_event)
                        for name in ('imu', 'uwb')]


def run_session(robot, cfg, session, *, dry_run=False, no_servo=False, max_control_s=None,
                stop_event=None, sensors=None, controller=None, waypoints=None):
    """缺基站时复用上一完整定位，断流/格式/几何错误回中等待。

可注入传感器管理器/舵机I/O后端，用于报文回放和不接硬件的集成验证。
no_servo进入真实传感器诊断，不初始化任何舵机后端。
运动模式主线程决策、独立执行线程写PWM；组内两动作串行。
"""
    waypoints = validate_waypoints(cfg, waypoints)
    waypoint_ends = []
    if waypoints is not None:
        for _, _, duration in waypoints:
            waypoint_ends.append((waypoint_ends[-1] if waypoint_ends else 0.) + duration)
        # 点表时长代替YAML单点t；API显式max_control_s仍可作为更短的总上限。
        if max_control_s is None:
            max_control_s = waypoint_ends[-1]
        else:
            max_control_s = min(finite_float(max_control_s, 'max_control_s'), waypoint_ends[-1])
        cfg = replace(cfg, x_d=waypoints[0][0], y_d=waypoints[0][1], t=max_control_s)
    elif max_control_s is None:
        max_control_s = cfg.t
    if max_control_s is not None:
        max_control_s = finite_float(max_control_s, 'max_control_s')
        if max_control_s <= 0:
            raise ValueError('max_control_s必须大于0')
    if no_servo and dry_run:
        raise ValueError('no_servo使用真实传感器，不能与dry_run同时启用')
    stop = stop_event or threading.Event()
    sensor_cfg = copy.deepcopy(robot)
    sensor_cfg.setdefault('runtime', {})['environment'] = 'surface'
    # 该脚本只需IMU与UWB；不打开相机/深度/电流采集，避免额外硬件依赖。
    for name in ('imu', 'uwb', 'vision', 'depth', 'power'):
        sensor_cfg.setdefault('sensors', {}).setdefault(name, {})['enabled'] = name in ('imu', 'uwb')
    sensor_cfg['sensors']['imu'].setdefault('sample_rate_hz', 200)
    if sensors is None:
        sensors = (SurfaceMockSensors(sensor_cfg, cfg, stop) if dry_run else
                   SensorManager(sensor_cfg, mock=False, stop_event=stop, log_dir=session))
    if no_servo:
        from .surface_navigation_sensor_test_20261014 import run_sensor_test
        return run_sensor_test(sensors, sensor_cfg, cfg, session, stop, duration_s=max_control_s)
    calibration = build_calibration(robot)
    policy = NavigationPolicy(cfg, calibration)
    if controller is None:
        if dry_run:
            controller = DryRunRLServoController(calibration.centers)
        else:
            def controller():
                from drivers.pca9685_servo import PCA9685ServoController
                hardware = copy.deepcopy(robot)
                hardware['servo']['channels'] = [row for row in hardware['servo']['channels']
                                                   if int(row['servo_id']) in (1, 2, 3)]
                return PCA9685ServoController(hardware)
    failures = queue.Queue()
    executor = RLServoExecutor(controller, NavigationScheduler(calibration), RLServoStateTracker(calibration),
        config={'command_hz': 50., 'initial_move_s': 0., 'safe_recenter_s': 0., 'hold_at_center': True},
        active_servo_ids=(1, 2, 3), shutdown_event=stop, keep_pwm=False,
        failure_queue=failures,
        servo_id_to_channel={i: int(calibration.servos[i]['channel']) for i in (1, 2, 3)})
    imu = CalibratedIMU(cfg.imu_filter_tau_s, cfg.calibration_min_samples,
                        cfg.calibration_max_std_deg, cfg.yaw_sign)
    decoder = UWBStreamDecoder(cfg.uwb_tag)
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    logs = {name: JsonlLogger(session / name) for name in
            ('events.jsonl', 'commands.jsonl', 'navigation.jsonl', 'imu_filtered.jsonl')}
    raw = RawSensorLoggers(session)
    phase = 'placement'
    calibration_start = calibration_end = None
    calibration_rows = []
    last_seen = {'imu': -1, 'uwb': -1}
    last_uwb_seq = None
    latest_fix = None
    last_measurement_ns = None
    reuse_last_fix = False
    uwb_fault_pending = False
    fault = None
    reason = 'stopped'
    cleanup_errors = []
    completed_pairs = 0
    sensors_started = False
    executor_started = False
    motion_timer = None
    motion_start = None
    waypoint_index = 0
    waypoint_timer = None
    waypoint_hold = None
    waypoint_timer_error = None
    waypoint_lock = threading.Lock()
    deadline_reached = threading.Event()

    class MotionTimeLimit(Exception):
        pass

    def expire_motion():
        # 独立定时线程只请求停止，PWM仍由执行器线程关闭；不依赖新UWB到来。
        deadline_reached.set()
        executor.request_emergency_stop()

    def check_time_limit():
        if deadline_reached.is_set() or (motion_start is not None and max_control_s is not None
                                        and time.monotonic()-motion_start >= max_control_s):
            expire_motion()
            raise MotionTimeLimit

    def hold_at_waypoint_deadline():
        nonlocal waypoint_hold, waypoint_timer_error
        # 主线程即使阻塞在终端输出，也由PWM线程按时取消旧段动作。
        # 与submit共用锁，禁止到期回中后又入队一个旧动作。
        with waypoint_lock:
            if executor_started and not deadline_reached.is_set() and not stop.is_set():
                try:
                    waypoint_hold = executor.request_hold()
                except RuntimeError as exc:
                    if not deadline_reached.is_set() and not stop.is_set():
                        waypoint_timer_error = exc

    def cancel_waypoint_timer():
        nonlocal waypoint_timer
        if waypoint_timer is not None:
            waypoint_timer.cancel()
            waypoint_timer.join()
            waypoint_timer = None

    def arm_waypoint_timer():
        nonlocal waypoint_timer
        if (waypoints is not None and waypoint_index+1 < len(waypoints)
                and waypoint_ends[waypoint_index] < max_control_s):
            delay = max(0., motion_start+waypoint_ends[waypoint_index]-time.monotonic())
            waypoint_timer = threading.Timer(delay, hold_at_waypoint_deadline)
            waypoint_timer.daemon = True
            waypoint_timer.start()

    def start_control_clock():
        nonlocal motion_start, motion_timer
        if motion_start is not None:
            return  # 到点暂停和漂移恢复都不重置总时间。
        motion_start = time.monotonic()
        if max_control_s is not None:
            motion_timer = threading.Timer(max_control_s, expire_motion)
            motion_timer.daemon = True
            motion_timer.start()
        arm_waypoint_timer()
        event('motion_started', time_limit_s=max_control_s)
        if waypoints is not None:
            announce_waypoint()

    def write(name, row):
        target = {'waypoint_index': waypoint_index+1, 'x_d': policy.config.x_d, 'y_d': policy.config.y_d}
        if not logs[name].write({'t_ns': time.monotonic_ns(), 'phase': phase, **target, **row}):
            raise RuntimeError(f'日志队列已满：{name}')

    def event(kind, **values):
        write('events.jsonl', {'event': kind, **values})

    def announce_waypoint():
        x, y, duration = waypoints[waypoint_index]
        event('waypoint_started', duration_s=duration,
              scheduled_start_s=waypoint_ends[waypoint_index]-duration)
        print(f'目标 {waypoint_index+1}/{len(waypoints)}：({x:g}, {y:g})m，{duration:g}s。', flush=True)

    def waypoint_due():
        return (waypoints is not None and motion_start is not None
                and waypoint_index+1 < len(waypoints)
                and time.monotonic()-motion_start >= waypoint_ends[waypoint_index])

    def advance_waypoint():
        nonlocal waypoint_index, policy, waypoint_hold
        if not waypoint_due():
            return
        cancel_waypoint_timer()  # 等回调完成，取同一个回中回执而非重复提交。
        # 先取消上一目标的整组动作并直接回中；不能续发旧目标右摆。
        if executor_started:
            center_tail(waypoint_hold)
        waypoint_hold = None
        while waypoint_due():
            event('waypoint_completed')
            waypoint_index += 1
            x, y, _ = waypoints[waypoint_index]
            # 每段重新建立PI，防止上一目标的积分和决策时间遗留。
            policy = NavigationPolicy(replace(cfg, x_d=x, y_d=y), calibration)
            announce_waypoint()
        arm_waypoint_timer()

    def service():
        nonlocal latest_fix, last_uwb_seq, uwb_fault_pending, last_measurement_ns, reuse_last_fix
        check_time_limit()
        if stop.is_set():
            raise InterruptedError('收到停止请求')
        if waypoint_timer_error is not None:
            raise RuntimeError(f'目标切换回中请求失败：{waypoint_timer_error}')
        if executor.failure is not None:
            check_time_limit()
            raise RuntimeError(f'尾鳍执行失败：{executor.failure}')
        if sensors.fatal_errors():
            raise RuntimeError(f'采集线程失败：{sensors.fatal_errors()}')
        raw.write_from_buffers(sensors.buffers)
        for sample in sensors.buffers['imu'].snapshot():
            if sample.t_ns <= last_seen['imu']:
                continue
            last_seen['imu'] = sample.t_ns
            if phase == 'placement':
                continue
            if not sample.ok:
                raise RuntimeError(f'IMU_INVALID：{sample.error}')
            try:
                angles = imu.read_angles(sample.data)
                if phase == 'calibration' and calibration_start <= sample.t_ns <= calibration_end:
                    calibration_rows.append((sample.t_ns, angles))
                elif imu.zeros is not None:
                    filtered = imu.update(angles, sample.t_ns)
                    write('imu_filtered.jsonl', {'sample_t_ns': sample.t_ns, **filtered})
            except (ValueError, KeyError) as exc:
                raise RuntimeError(f'IMU_INVALID：{exc}') from exc
        for sample in sensors.buffers['uwb'].snapshot():
            if sample.t_ns <= last_seen['uwb']:
                continue
            last_seen['uwb'] = sample.t_ns
            if last_uwb_seq is not None and sample.seq > last_uwb_seq+1:
                decoder.reset_partial()  # 环形缓冲覆盖后禁止把不相邻字节片段拼成帧。
                event('uwb_buffer_gap')
                latest_fix = None
                uwb_fault_pending = True
            last_uwb_seq = sample.seq
            if phase in ('waiting_for_uwb', 'navigation', 'holding') and sample.t_ns <= calibration_end:
                decoder.reset_partial()  # 也拒绝标定前采集、线程延迟后才入缓冲区的片段。
                continue
            if not sample.ok:
                # read_until可能在一行中间超时；普通空读保留片段，
                # 有效定位是否过期仍由uwb_timeout_s判断。
                if sample.error == 'no_signal_or_timeout':
                    continue
                decoder.reset_partial()
                latest_fix = None
                uwb_fault_pending = True
                continue
            try:
                # raw_text已经strip，会丢失片段边缘分隔符，因此必须使用原始字节。
                invalid_before = decoder.invalid_frames
                incomplete_before = decoder.incomplete_frames
                frames = decoder.feed(bytes.fromhex(sample.data['raw_hex']), include_incomplete=True)
            except (ValueError, KeyError, TypeError) as exc:
                decoder.reset_partial()
                latest_fix = None
                uwb_fault_pending = True
                event('uwb_invalid_bytes', message=str(exc))
                continue
            if decoder.invalid_frames-invalid_before > decoder.incomplete_frames-incomplete_before:
                # 格式错误仍使定位失效；缺基站报文在下方保留上一组完整距离。
                latest_fix = None
                uwb_fault_pending = True
                continue
            for frame in frames:
                if any(r is None for r in frame.ranges_m):
                    last_measurement_ns = sample.t_ns
                    reuse_last_fix = latest_fix is not None
                    event('uwb_incomplete', ranges_m=frame.ranges_m, frame_id=frame.frame_id,
                          position_reused=reuse_last_fix,
                          source_frame_id=latest_fix['uwb_frame_id'] if latest_fix else None)
                    continue
                try:
                    x, y = position_from_ranges(frame.ranges_m, cfg.L, cfg.W,
                                                cfg.boundary_tolerance_m, cfg.range_residual_limit_m, H=cfg.H)
                except ValueError as exc:
                    latest_fix = None
                    uwb_fault_pending = True
                    event('uwb_invalid_position', message=str(exc), ranges_m=frame.ranges_m)
                    continue
                latest_fix = {'sample_t_ns': sample.t_ns, 'uwb_frame_id': frame.frame_id,
                              'ranges_m': frame.ranges_m, 'x': x, 'y': y}
                last_measurement_ns = sample.t_ns
                reuse_last_fix = False
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.last_error or logger.dropped_count:
                raise RuntimeError(f'日志写入失败：{logger.path} {logger.last_error}')

    def pause_until(end_ns):
        while time.monotonic_ns() < end_ns:
            service()
            stop.wait(min(.005, max(0., (end_ns-time.monotonic_ns())/1e9)))
        service()

    def require_fresh_imu(now):
        if imu.last_t_ns is None or (now-imu.last_t_ns)/1e9 > cfg.imu_timeout_s:
            raise RuntimeError('IMU_TIMEOUT：没有新鲜的标定后IMU数据')

    def uwb_is_fresh():
        return (latest_fix is not None and
                (time.monotonic_ns()-(last_measurement_ns if reuse_last_fix else
                                     latest_fix['sample_t_ns']))/1e9 <= cfg.uwb_timeout_s)

    def ensure_executor():
        nonlocal executor_started
        if executor_started:
            return
        executor.start()
        executor_started = True
        ready_deadline = time.monotonic()+5.
        while not executor.wait_ready(.005):
            service()
            require_fresh_imu(time.monotonic_ns())
            if time.monotonic() >= ready_deadline:
                raise RuntimeError('尾鳍执行器启动超时')

    def center_tail(ticket=None):
        # PWM线程直接写中位；等待确认期间继续采集，不要求UWB有效。
        ensure_executor()
        try:
            if ticket is None:
                ticket = executor.request_hold()
        except RuntimeError:
            check_time_limit()
            if stop.is_set():
                raise InterruptedError('收到停止请求')
            raise
        hold_deadline = time.monotonic()+5.
        while not ticket.wait(.005):
            service()
            require_fresh_imu(time.monotonic_ns())
            if time.monotonic() >= hold_deadline:
                raise RuntimeError('尾鳍暂停确认超时')
        check_time_limit()
        if stop.is_set():
            raise InterruptedError('收到停止请求')
        if ticket.failure is not None:
            raise RuntimeError(f'尾鳍暂停失败：{ticket.failure}')
        executor.drain_started()
        executor.drain_completed()
        event('tail_held', angles_deg=ticket.angles_deg)

    def wait_for_valid_uwb():
        nonlocal phase, latest_fix, uwb_fault_pending
        if uwb_is_fresh() and not uwb_fault_pending:
            return False
        previous_phase = phase
        phase = 'waiting_for_uwb'
        if not uwb_is_fresh():
            latest_fix = None
        event('uwb_unavailable', message='四路UWB距离缺失、无效或超时，尾鳍回中等待')
        print('错误：四路UWB距离缺失、无效或超时；尾鳍保持中位，等待有效定位。', flush=True)
        center_tail()
        start_control_clock()
        policy.integral = 0.
        policy.last_t_ns = None
        while not uwb_is_fresh():
            service()
            require_fresh_imu(time.monotonic_ns())
            advance_waypoint()  # UWB异常等待也占用当前点时长，不能暂停路线时钟。
            stop.wait(.005)
        phase = previous_phase
        # 坏帧后紧接好帧也必须先完成一次回中，不能由好帧覆盖故障。
        uwb_fault_pending = False
        event('uwb_recovered', uwb_frame_id=latest_fix['uwb_frame_id'])
        print('四路UWB有效定位已恢复，继续定点控制。', flush=True)
        return True

    def submit(theta, pair, index):
        check_time_limit()
        action = NavigationTailAction(theta, cfg.t_0)
        context = {'uwb_frame_id': pair['uwb_frame_id'], 'pair_index': index,
                   'position_reused': pair['position_reused'],
                   'position_age_s': max(0., (time.monotonic_ns()-pair['sample_t_ns'])/1e9)}
        try:
            with waypoint_lock:
                if waypoint_due():
                    return None  # 下一轮切点，不提交刚刚到期的旧动作。
                rid = executor.submit('tail', action, context)
        except RuntimeError:
            check_time_limit()  # 处理定时线程恰在提交动作期间停止执行器的竞争。
            raise
        write('commands.jsonl', {'request_id': rid, 'agent': 'tail', 'action': action.to_dict(),
                                **context, 'decision': asdict(pair['decision'])})
        return rid

    try:
        for logger in logs.values():
            logger.start()
        raw.start()
        write_metadata_yaml(session / 'metadata.yaml', {'controller': 'surface_navigation_20261010',
            'navigation': asdict(cfg), 'robot_config': sensor_cfg, 'dry_run': dry_run, 'no_servo': False,
            'waypoints': waypoints, 'max_control_s': max_control_s, 'active_servo_ids': [1, 2, 3]})
        sensors_started = True
        sensors.start_all()
        event('placement_started', duration_s=cfg.t1)
        print(f'人工放置 {cfg.t1:g}s：鱼头朝+x，随后保持静止标定 {cfg.t2:g}s。', flush=True)
        pause_until(time.monotonic_ns()+round(cfg.t1*1e9))
        phase = 'calibration'
        calibration_start = time.monotonic_ns()
        calibration_end = calibration_start+round(cfg.t2*1e9)
        event('calibration_started', duration_s=cfg.t2)
        pause_until(calibration_end)
        if (len(calibration_rows) < cfg.calibration_min_samples or
                (calibration_rows[-1][0]-calibration_rows[0][0])/1e9 < .8*cfg.t2):
            raise RuntimeError('IMU标定窗口覆盖不足；请增大t2或检查采集')
        # 不允许用断流前后的少量样本拼成“稳定窗口”。
        if any((b[0]-a[0])/1e9 > cfg.imu_timeout_s for a, b in zip(calibration_rows, calibration_rows[1:])):
            raise RuntimeError('IMU标定窗口存在采样断流')
        imu.calibrate([row for _, row in calibration_rows])
        imu.update(calibration_rows[-1][1], calibration_rows[-1][0])
        write_metadata_yaml(session / 'calibration.yaml', imu.diagnostics)
        event('calibration_completed', **imu.diagnostics)
        decoder.reset_partial()
        latest_fix = None  # 连同半帧一起丢弃，必须从此后接收的新完整帧建立定位。
        last_measurement_ns = None
        reuse_last_fix = False
        uwb_fault_pending = False
        phase = 'waiting_for_uwb'
        # 标定期间收到的定位不能启动动作，必须等标定结束后的完整新帧。
        fix_deadline = time.monotonic_ns()+round(cfg.uwb_timeout_s*1e9)
        while True:
            service()
            now = time.monotonic_ns()
            require_fresh_imu(now)
            if uwb_fault_pending or now >= fix_deadline:
                wait_for_valid_uwb()
                break
            if latest_fix is not None and latest_fix['sample_t_ns'] > calibration_end:
                break
            stop.wait(.005)
        phase = 'navigation'
        event('navigation_started')
        print('标定完成，开始导航；缺基站时沿用上一完整定位。', flush=True)
        processed_id = consumed_id = None
        latest_decision = None
        active_pair = None
        inflight_id = None
        holding = False
        decision_waypoint_index = waypoint_index
        print_due = 0.
        while True:
            service()
            now = time.monotonic_ns()
            require_fresh_imu(now)
            advance_waypoint()
            if decision_waypoint_index != waypoint_index:
                decision_waypoint_index = waypoint_index
                active_pair = None
                inflight_id = None
                processed_id = consumed_id = None
                latest_decision = None
                holding = False
                phase = 'navigation'
            if wait_for_valid_uwb():
                # 取消整组，恢复后重新决策，不能继续旧组右摆。
                active_pair = None
                inflight_id = None
                processed_id = consumed_id = None
                continue
            now = time.monotonic_ns()
            completed_actions = executor.drain_completed()
            # 缺测期间每组动作结束后，以完整旧定位和最新IMU重算下一组。
            decision_updated = False
            if (latest_fix['uwb_frame_id'] != processed_id or completed_actions or
                    (reuse_last_fix and active_pair is None and not holding)):
                decision_updated = True
                if holding:
                    # 暂停时不积累航向积分；漂移恢复从当前误差重新计算。
                    policy.integral = 0.
                    policy.last_t_ns = None
                latest_decision = policy.decide(latest_fix['x'], latest_fix['y'], imu.values['yaw_deg'],
                                                now)
                processed_id = latest_fix['uwb_frame_id']
                write('navigation.jsonl', {**latest_fix, **asdict(latest_decision),
                                           'position_reused': reuse_last_fix,
                                           'position_age_s': max(0., (now-latest_fix['sample_t_ns'])/1e9),
                                           'integral': policy.integral, 'pair_active': active_pair is not None,
                                           'imu_sample_t_ns': imu.last_t_ns})
                if not holding and latest_decision.reached:
                    start_control_clock()  # 初始已在目标点，也要受总时长t约束。
                    holding = True
                    phase = 'holding'
                    consumed_id = processed_id
                    event('holding_started', **asdict(latest_decision))
                    if executor_started:
                        center_tail()
                    active_pair = None
                    inflight_id = None
                    completed_actions = []  # 已到点回中，丢弃本轮收取的旧动作回执。
                elif holding and latest_decision.distance_m >= cfg.delta_s:
                    # 保持中只依据位置漂出容差恢复；近点方位角变化不触发反复游动。
                    holding = False
                    phase = 'navigation'
                    event('navigation_resumed', **asdict(latest_decision))
            for item in executor.drain_started():
                event('action_started', request_id=item.request_id, context=item.context)
            restart_pair = False
            for item in completed_actions:
                if not item.endpoint_written or item.request_id != inflight_id:
                    raise RuntimeError('尾鳍动作回执不匹配或端点PWM未确认')
                event('action_completed', request_id=item.request_id, context=item.context)
                inflight_id = None
                if active_pair['decision'].control_mode != latest_decision.control_mode:
                    # 每个动作完成后用最新标定IMU判断，取消尚未开始的旧模式动作。
                    # 即使没有新UWB帧，也应立即恢复PI或从相反方向的30°开始。
                    if active_pair['index'] == 1:
                        completed_pairs += 1
                    active_pair = None
                    restart_pair = True
                elif active_pair['index'] == 0:
                    active_pair['index'] = 1
                    if latest_decision.control_mode != 'pi':
                        active_pair['decision'] = latest_decision
                    inflight_id = submit(active_pair['decision'].theta_right, active_pair, 1)
                else:
                    completed_pairs += 1
                    active_pair = None
            check_time_limit()
            if not holding and active_pair is None and (processed_id != consumed_id or
                                                       (reuse_last_fix and decision_updated) or
                                                       latest_decision.control_mode != 'pi' or restart_pair):
                if not executor_started:
                    ensure_executor()
                    continue  # 初始化期间仍收数；下一轮先检查新定位与到达状态。
                start_control_clock()
                active_pair = {**latest_fix, 'decision': latest_decision, 'index': 0,
                               'position_reused': reuse_last_fix}
                consumed_id = processed_id
                inflight_id = submit(latest_decision.theta_left, active_pair, 0)
            if time.monotonic() >= print_due and latest_decision is not None:
                d = latest_decision
                print(f'x={d.x:.3f} y={d.y:.3f} yaw={d.yaw_deg:.1f}° '
                      f'yaw_d={d.yaw_d_deg:.1f}° 距离={d.distance_m:.3f}m '
                      f'模式={d.control_mode} 动作1/2={d.theta_left:.1f}/{d.theta_right:.1f}° '
                      f'复用定位={reuse_last_fix} '
                      f"定位年龄={(time.monotonic_ns()-latest_fix['sample_t_ns'])/1e9:.2f}s", flush=True)
                print_due = time.monotonic()+1.
            stop.wait(.005)
    except MotionTimeLimit:
        reason = 'time_limit'
        if waypoints is not None and time.monotonic()-motion_start >= waypoint_ends[-1]:
            event('waypoint_completed')
            event('route_completed')
    except InterruptedError:
        reason = 'stopped'
    except BaseException as exc:
        reason = 'fault'
        fault = f'{type(exc).__name__}: {exc}'
    finally:
        # 先停PWM再停采集，所有路径（包括标定失败、Ctrl+C）执行清理。
        executor.request_emergency_stop()
        cancel_waypoint_timer()
        if motion_timer is not None:
            motion_timer.cancel()
            motion_timer.join()

        def cleanup(name, operation):
            try:
                operation()
            except BaseException as exc:
                cleanup_errors.append(f'{name}: {exc}')

        cleanup('servo', lambda: executor.stop(timeout=5.))
        # 共享执行器在一批三路PWM写入中收到停车请求时会用这个特定异常取消余下写入。
        # 这是成功响应停车，不应误报硬件故障；其他I/O异常仍完整报告。
        cancelled_write = (type(executor.failure) is RuntimeError and
                           str(executor.failure) == 'emergency stop: angle writes cancelled')
        if executor.failure is not None and fault is None and not cancelled_write:
            fault = str(executor.failure)
        # failure只保留首个异常；必须另外检查全部报告，防止“写入取消”掩盖随后关PWM失败。
        reported_failures = []
        while True:
            try:
                failure = failures.get_nowait()
            except queue.Empty:
                break
            if (failure.get('fault_code') == 'SERVO_EXECUTOR_FAILED' and
                    failure.get('message') == 'emergency stop: angle writes cancelled'):
                continue
            reported_failures.append(f"{failure.get('fault_code')}: {failure.get('message')}")
        if reported_failures:
            fault = '; '.join(([fault] if fault else []) + reported_failures)
        stop.set()
        if sensors_started:
            cleanup('sensors', lambda: sensors.stop_all(timeout=5.))
        cleanup('raw_drain', lambda: raw.write_from_buffers(sensors.buffers))
        cleanup('raw_logs', raw.stop)
        for logger in logs.values():
            cleanup(str(logger.path), logger.stop)
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.last_error or logger.dropped_count or logger.is_alive():
                cleanup_errors.append(f'日志未正常关闭：{logger.path} {logger.last_error}')
        result = {'session': str(session), 'reason': 'fault' if fault or cleanup_errors else reason,
                  'completed_pairs': completed_pairs, 'fault': fault, 'cleanup_errors': cleanup_errors,
                  'executor_alive': executor.is_alive(), 'last_position': latest_fix,
                  'waypoint_index': waypoint_index+1,
                  'invalid_uwb_frames': decoder.invalid_frames}
        (session / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if fault or cleanup_errors:
        raise RuntimeError(f'水面导航退出：{fault or cleanup_errors}；日志：{session}')
    return result
