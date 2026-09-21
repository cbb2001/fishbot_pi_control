"""侧鳍在线 PPO；独立健康监测覆盖标定、动作、推理和更新等待。"""
from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .rl_action_history_20260914 import FinActionHistory
from .rl_action_scheduler_20260914 import RLActionScheduler
from .rl_actions_20260914 import FinAction, build_calibration, validate_action_space
from .rl_episode_20260914 import EpisodeMonitor
from .rl_ppo_agent_20260914 import FinPPOAgent
from .rl_reward_20260914 import RLReward
from .rl_rollout_buffer_20260914 import Transition
from .rl_safety_20260914 import RLSafetyMonitor, RLSafetyFault
from .rl_sensor_calibration_20260914 import wait_for_calibration, finite_float
from .rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
from .rl_servo_state_tracker_20260914 import RLServoStateTracker
from .rl_state_builder_20260914 import RLStateBuilder
from .rl_state_buffer_20260914 import RLStateBuffer, RLStateSyncWorker
from .rl_training_coordinator_20260914 import FinTrainingCoordinator
from .data_logger import JsonlLogger, RawSensorLoggers, write_metadata_yaml
from .rl_live_status_20260914 import LiveStatusPublisher
from .sensor_manager import SensorManager


def _write(logs, name, row):
    if not logs[name].write(row):
        raise RuntimeError('日志队列已满: ' + name)


def run_session(cfg, robot_cfg, session, *, dry_run=False, seed=None, resume=None,
                keep_pwm=False, stop_event=None):
    # 独立驱动和 PPO 共用 robot.yaml 的压力参考；配置缺失时不得退回自动压力标定。
    pressure_value = robot_cfg.get('depth_sensor', {}).get('surface_pressure_mbar')
    fixed_surface_pressure_mbar = finite_float(pressure_value, 'depth_sensor.surface_pressure_mbar')
    if isinstance(pressure_value, bool) or fixed_surface_pressure_mbar <= 0 or not math.isfinite(fixed_surface_pressure_mbar*100.):
        raise ValueError('depth_sensor.surface_pressure_mbar 必须为正数')
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    stop = stop_event if stop_event is not None else threading.Event()
    failures = queue.Queue()
    poll_s = float(cfg['runtime']['poll_interval_s'])
    shutdown_s = float(cfg['runtime']['shutdown_timeout_s'])
    threads = int(cfg['runtime']['torch_num_threads'])
    if not all(math.isfinite(v) and v > 0 for v in (poll_s, shutdown_s, threads)):
        raise ValueError('轮询、退出超时和 Torch 线程数必须为正')
    torch.set_num_threads(threads)
    calibration = build_calibration(robot_cfg)
    validate_action_space(calibration, agents=('fin',))
    history = FinActionHistory(cfg['state']['fin_history_length'])
    ppo_cfg = {**cfg['fin_ppo'], 'time_aware_discount': cfg['training']['time_aware_discount'],
               'discount_reference_dt_s': cfg['training']['discount_reference_dt_s']}
    agent = FinPPOAgent(history_length=history.length, normalized_clip=cfg['state']['normalized_clip'],
                        config=ppo_cfg, seed=seed)
    tracker = RLServoStateTracker(calibration)
    scheduler = RLActionScheduler(calibration, limits=calibration.limits)
    if dry_run:
        controller = DryRunRLServoController(calibration.centers)
    else:
        # 实际构造延迟到传感器检查和标定通过之后的执行器线程中。
        def controller():
            from drivers.pca9685_servo import PCA9685ServoController
            hardware = copy.deepcopy(robot_cfg)
            hardware['servo']['channels'] = [c for c in hardware['servo']['channels']
                                              if int(c['servo_id']) in (4, 5, 6, 7)]
            return PCA9685ServoController(hardware)
    ex = RLServoExecutor(controller, scheduler, tracker,
        config={'command_hz': 50., 'safe_recenter_s': 0. if dry_run else 2.,
                'initial_move_s': 0. if dry_run else 2.},
        active_servo_ids=(4, 5, 6, 7), shutdown_event=stop, failure_queue=failures,
        keep_pwm=keep_pwm, servo_id_to_channel={i: int(v['channel']) for i, v in calibration.servos.items()})
    sensor_cfg = copy.deepcopy(robot_cfg)
    for name in ('vision', 'uwb'):
        sensor_cfg['sensors'].setdefault(name, {})['enabled'] = False
    if dry_run:
        from .rl_mock_sensors_20260914 import MockRLSensorManager
        sensors = MockRLSensorManager(sensor_cfg, stop_event=stop, log_dir=session)
    else:
        sensors = SensorManager(sensor_cfg, mock=False, stop_event=stop, log_dir=session)
    guard = RLSafetyMonitor(sensors, ex, failures, stop, cfg, robot_cfg)
    names = ('events.jsonl', 'commands.jsonl', 'rl_state_20260914.jsonl', 'synchronized_sensors.jsonl',
             'fin_transitions_20260914.jsonl', 'fin_ppo_updates_20260914.jsonl',
             'policy_publish_20260914.jsonl', 'episodes_20260914.jsonl',
             'update_holds_20260914.jsonl', 'faults_20260914.jsonl')
    # 日志独立停止，避免传感器先停后，故障与清理记录丢失。
    logs = {name: JsonlLogger(session / name) for name in names}
    raw = RawSensorLoggers(session)
    status = LiveStatusPublisher(session / 'rl_live_status_20260914.json', cfg['runtime']['live_status_interval_s'])
    sync = coord = statebuf = None
    completed = next_ep = 0
    fault = None
    cleanup_errors = []
    phase = 'startup'
    status_warning = None

    def health():
        guard.raise_if_failed()
        if sync is not None:
            sync.raise_if_failed()
        if ex.failure is not None:
            raise RLSafetyFault('SERVO_FAILED', str(ex.failure))
        if stop.is_set():
            raise RLSafetyFault('STOP_REQUESTED', '收到停止请求')
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.last_error or logger.dropped_count:
                raise RLSafetyFault('LOG_FAILED', f'{logger.path}: {logger.last_error or "日志丢失"}')

    def publish(force=False, running=True):
        nonlocal status_warning
        snapshot = coord.snapshot() if coord is not None else {}
        latest = statebuf.latest() if statebuf is not None else None
        try:
            status.publish({**snapshot, 'training_running': running, 'phase': phase,
                'control_mode': 'fin_only', 'controlled_servo_ids': [4, 5, 6, 7],
                'depth_m': None if latest is None else latest.depth_m,
                'yaw_deg': None if latest is None else latest.yaw_deg,
                'latest_depth_m': None if latest is None else latest.depth_m,
                'latest_yaw_deg': None if latest is None else latest.yaw_deg,
                'target_depth_m': cfg['targets']['depth_m'],
                'target_heading_deg': cfg['targets']['heading_deg'],
                'latest_reward': None if latest is None else latest.reward_total,
                'data_freshness_enforced': guard.enforce_data_freshness,
                'latest_data_age_ms': guard.latest_data_age_ms,
                'data_age_warnings': guard.data_age_warnings,
                'fin_sequence': scheduler.fin_sequence.to_dict(),
                'fault': fault, 'cleanup_errors': cleanup_errors, 'safety_limits': guard.limits}, force=force)
        except Exception as exc:
            status_warning = str(exc)  # 状态显示文件失败不能阻止停车。

    def service():
        health()
        if sync is None:
            raw.write_from_buffers(sensors.buffers)
        publish()

    def wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            service()
            stop.wait(min(poll_s, max(0., until - time.monotonic())))
        service()

    def record_fault(exc):
        return {'t_ns': time.monotonic_ns(), 'fault_source': type(exc).__name__,
                'fault_code': getattr(exc, 'code', getattr(exc, 'fault_code', 'TRAINING_FAILED')),
                'message': str(exc)}

    try:
        for logger in logs.values():
            logger.start()
        raw.start()
        coord = FinTrainingCoordinator(agent, cfg)
        if resume:
            next_ep = int(coord.load_checkpoint(resume)['next_episode_index'])
        write_metadata_yaml(session / 'metadata.yaml', {'rl_config': cfg, 'robot_config': robot_cfg,
                            'dry_run': dry_run, 'safety_limits': guard.limits, 'seed': seed,
                            'policy_architecture': agent.architecture(),
                            'initial_fin_sequence': scheduler.fin_sequence.to_dict()})
        sensors.start_all()
        guard.start()
        wait(float(cfg['startup']['pretrain_wait_s']))
        wait(float(cfg['startup']['sensor_warmup_s']))
        phase = 'calibration'
        merged = {**cfg, 'sensors': sensor_cfg['sensors'],
                  'calibration': {**cfg['calibration'], 'surface_pressure_mbar': fixed_surface_pressure_mbar}}
        sensor_cal = wait_for_calibration(sensors, merged, stop_event=stop, on_check=service)
        guard.calibration = sensor_cal
        (session / 'calibration_20260914.json').write_text(
            json.dumps(sensor_cal.state_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        service()
        # 所有关键样本就绪后，才允许初始化 PCA 并回中。
        ready_until = time.monotonic() + cfg['runtime']['sensor_ready_timeout_s']
        while any(sensors.buffers[n].latest() is None for n in ('imu', 'depth', 'power')):
            if time.monotonic() > ready_until:
                raise TimeoutError('关键传感器尚未就绪')
            wait(poll_s)
        phase = 'servo_initialization'
        guard.check_once()
        ex.start()
        guard.executor_started = True
        until = time.monotonic() + 10
        while not ex.wait_ready(poll_s):
            service()
            if time.monotonic() > until:
                raise TimeoutError('侧鳍执行器初始化超时')
        service()
        if dry_run:
            sensors.set_surface_mode(False)
        statebuf = RLStateBuffer()
        builder = RLStateBuilder(sensors, sensor_cal, tracker, merged, reward=RLReward(cfg))
        sync = RLStateSyncWorker(builder, statebuf, rate_hz=cfg['state']['sample_hz'], stop_event=stop,
            logger=logs['rl_state_20260914.jsonl'], sync_logger=logs['synchronized_sensors.jsonl'],
            raw_loggers=raw, failure_queue=failures)
        sync.start()
        guard.sync = sync
        until = time.monotonic() + cfg['runtime']['sensor_ready_timeout_s']
        while statebuf.latest() is None:
            if time.monotonic() > until:
                raise TimeoutError('等待 RL 状态超时')
            wait(poll_s)
        guard.statebuf = statebuf
        if not resume:
            agent.initialize_normalizer([np.r_[statebuf.latest().observation, history.vector(), scheduler.fin_sequence.vector()]])
        start_global = time.monotonic()
        limit = cfg['training']['max_training_s']
        while completed < cfg['training']['max_episodes']:
            service()
            if limit is not None and time.monotonic() - start_global >= limit:
                break
            ep = coord.begin_episode(next_ep)
            next_ep += 1
            ep_start = time.monotonic_ns()
            monitor = EpisodeMonitor(cfg)
            monitor.start(ep.episode_index, ep_start)
            action_n, last_t, rewards = 0, ep_start - 1, []
            phase = 'collecting'

            def observe():
                nonlocal last_t
                service()
                latest = statebuf.latest()
                if latest.t_ns > last_t:
                    for row in statebuf.between(last_t + 1, latest.t_ns):
                        monitor.observe(row, action_count=action_n)
                last_t = max(last_t, latest.t_ns)
                monitor.check_time(time.monotonic_ns(), action_count=action_n)
                return latest

            while monitor.decision is None or coord.ready:
                st = observe()
                if not coord.ready and (monitor.decision or (limit is not None and time.monotonic() - start_global >= limit)):
                    break
                if coord.ready:
                    phase = 'updating'
                    hold_start = time.monotonic_ns()
                    coord.start_update()
                    try:
                        while coord.updating:
                            observe()
                            metrics = coord.poll_update()
                            if metrics:
                                _write(logs, 'fin_ppo_updates_20260914.jsonl', metrics)
                                if metrics.get('published'):
                                    _write(logs, 'policy_publish_20260914.jsonl', metrics)
                                    every = cfg['training'].get('checkpoint_every_updates', 0)
                                    if cfg['training'].get('checkpoint_on_policy_publish') or (every and agent.update_count % every == 0):
                                        coord.save_checkpoint(session, next_episode_index=next_ep,
                                                              metadata={'targets': cfg['targets']})
                            wait(poll_s)
                    finally:
                        _write(logs, 'update_holds_20260914.jsonl', {'start_t_ns': hold_start,
                            'end_t_ns': time.monotonic_ns(), 'included_in_training': False})
                    phase = 'collecting'
                    continue
                # 上次完成回执之后读取不可变约束快照；跨更新/episode 不清零。
                sequence = scheduler.fin_sequence
                obs = np.r_[st.observation, history.vector(), sequence.vector()].astype(np.float32)
                dec = agent.act(obs, action_mask=sequence.action_mask())
                service()  # 推理期间若发生故障，禁止提交刚生成的动作。
                decision_age_ms = (time.monotonic_ns() - st.t_ns) / 1e6
                if decision_age_ms > cfg['state']['max_state_age_ms']:
                    # 此开关只控制旧决策的拦截；独立线程仍监测最新状态和传感器。
                    if cfg['state'].get('enforce_data_freshness', True) and cfg['state'].get('stop_on_stale_decision', True):
                        raise RLSafetyFault('DECISION_STALE', '推理所依据的状态已过期，取消动作')
                    _write(logs, 'events.jsonl', {
                        'event': 'stale_decision_allowed', 'level': 'warning',
                        't_ns': time.monotonic_ns(), 'state_t_ns': st.t_ns,
                        'decision_age_ms': decision_age_ms,
                        'limit_ms': cfg['state']['max_state_age_ms'],
                        'episode_index': ep.episode_index, 'policy_version': dec.policy_version,
                    })
                observe()
                if monitor.decision:
                    break
                action = agent.decode_action(dec.action)
                fa = FinAction(**action)
                sequence.validate(fa)
                rid = ex.submit('fin', fa, {'episode_index': ep.episode_index})
                _write(logs, 'commands.jsonl', {'t_ns': time.monotonic_ns(), 'agent': 'fin',
                    'request_id': rid, 'action': action, 'controlled_servo_ids': [4, 5, 6, 7],
                    'episode_index': ep.episode_index, 'policy_version': dec.policy_version,
                    'action_index': dec.action[0], 'action_mask': dec.action_mask,
                    'sequence_before': sequence.to_dict(), 'effective_direction': sequence.direction(fa)})
                end = time.monotonic() + fa.t + cfg['runtime']['action_timeout_margin_s']
                done = None
                while done is None:
                    observe()
                    for item in ex.drain_completed():
                        if item.request_id == rid:
                            done = item
                    if time.monotonic() > end:
                        raise TimeoutError('侧鳍动作未完成')
                    if done is None:
                        wait(poll_s)
                if not done.endpoint_written:
                    raise RuntimeError('PWM 端点未确认')
                if scheduler.fin_sequence != sequence.after(fa):
                    raise RuntimeError('执行器侧鳍序列状态与策略动作不一致')
                _write(logs, 'events.jsonl', {'event': 'fin_action_completed',
                    't_ns': done.completion_t_ns, 'request_id': rid,
                    'sequence_after': scheduler.fin_sequence.to_dict()})
                action_n += 1
                history.append(fa, completed=True, pwm_success=True,
                               action_start_t_ns=done.start_t_ns, completion_t_ns=done.completion_t_ns)
                # 使用端点之后的状态，不能用动作结束前的旧状态作 next_observation。
                while statebuf.latest().t_ns < done.completion_t_ns:
                    observe()
                    # 允许旧数据时，仍不能无限等待动作后的状态而越过训练总时限。
                    if monitor.decision or (limit is not None and time.monotonic() - start_global >= limit):
                        break
                    wait(poll_s)
                if statebuf.latest().t_ns < done.completion_t_ns:
                    _write(logs, 'events.jsonl', {
                        'event': 'transition_discarded_no_post_action_state',
                        't_ns': time.monotonic_ns(), 'request_id': rid,
                        'episode_index': ep.episode_index,
                    })
                    break  # 缺少动作后观测，不能伪造 transition；正常执行 episode 收尾。
                nxt = observe()
                stats = statebuf.reward_stats(done.start_t_ns, done.completion_t_ns)
                term = monitor.decision
                next_obs = np.r_[nxt.observation, history.vector(), scheduler.fin_sequence.vector()]
                tr = Transition(tuple(obs), tuple(dec.action), float(stats['reward_mean']), tuple(next_obs),
                    dec.log_prob, dec.value, 0. if term and term.terminated else agent.value(next_obs),
                    tuple(dec.normalized_observation), duration_s=(done.completion_t_ns - done.start_t_ns) / 1e9,
                    behavior_policy_version=dec.policy_version, episode_index=ep.episode_index,
                    action_mask=dec.action_mask,
                    terminated=bool(term and term.terminated), truncated=bool(term and term.truncated), **stats)
                service()
                coord.add_transition(tr)
                _write(logs, 'fin_transitions_20260914.jsonl', tr.to_dict())
                rewards.append(tr.reward)
            if coord.pending:
                coord.discard_pending('episode_end')
            result = coord.end_episode(terminated=bool(monitor.decision and monitor.decision.terminated))
            completed += 1
            _write(logs, 'episodes_20260914.jsonl', {'episode_index': ep.episode_index,
                'reason': monitor.decision.reason if monitor.decision else 'training_time_limit',
                'action_count': action_n, 'reward_mean': float(np.mean(rewards)) if rewards else None, 'result': result})
            if cfg['training']['checkpoint_on_episode_end']:
                coord.save_checkpoint(session, next_episode_index=next_ep, metadata={'targets': cfg['targets']})
            publish(force=True)
            if monitor.decision and monitor.decision.terminated and cfg['training'].get('stop_after_success'):
                break
        service()
    except BaseException as exc:
        # 先停车；日志、检查点或 join 失败都不能推迟停车请求。
        ex.request_emergency_stop()
        stop.set()
        fault = record_fault(guard.failure or exc)
    finally:
        phase = 'stopping'

        def cleanup(name, operation):
            try:
                operation()
            except BaseException as exc:
                cleanup_errors.append({'component': name, 'message': str(exc)})
                ex.request_emergency_stop()

        if fault is not None:
            ex.request_emergency_stop()
        # 正常回中时仍保持传感器和监测运行；停车后再停止采样。
        guard.executor_started = False
        cleanup('servo', lambda: ex.stop(timeout=shutdown_s))
        cleanup('safety', lambda: guard.stop(timeout=shutdown_s))
        if fault is None and (guard.failure is not None or ex.failure is not None):
            fault = record_fault(guard.failure or ex.failure)
        stop.set()
        if sync is not None:
            cleanup('state_sync', lambda: sync.stop(timeout=shutdown_s))
        cleanup('sensors', lambda: sensors.stop_all(timeout=shutdown_s))
        if sensors.alive_worker_names():
            cleanup_errors.append({'component': 'sensors', 'message':
                                   '仍存活的线程: ' + ', '.join(sensors.alive_worker_names())})
        if coord is not None:
            if coord.updating:
                cleanup('cancel_update', lambda: coord.cancel_update('shutdown'))
            cleanup('ppo', lambda: coord.close(timeout_s=shutdown_s))
            if cfg['training']['checkpoint_on_exit'] and not coord.updating:
                cleanup('checkpoint', lambda: coord.save_checkpoint(session, next_episode_index=next_ep,
                    metadata={'targets': cfg['targets'], 'fault': fault}))
        if cleanup_errors and fault is None:
            fault = record_fault(RuntimeError(str(cleanup_errors)))
        if fault is not None:
            # 异步日志线程故障时仍尝试写入单独的故障摘要。
            cleanup('fault_summary', lambda: (session / 'fault_summary_20260914.json').write_text(
                json.dumps({'fault': fault, 'cleanup_errors': cleanup_errors}, ensure_ascii=False, indent=2), encoding='utf-8'))
            cleanup('fault_log', lambda: _write(logs, 'faults_20260914.jsonl', fault))
        cleanup('raw_logs', raw.stop)
        for name, logger in logs.items():
            cleanup(name, logger.stop)
        for logger in [*logs.values(), *raw.loggers.values()]:
            if logger.is_alive() or logger.last_error or logger.dropped_count:
                cleanup_errors.append({'component': str(logger.path), 'message':
                    logger.last_error or f'日志线程存活={logger.is_alive()}, 丢失={logger.dropped_count}'})
        if cleanup_errors and fault is None:
            fault = record_fault(RuntimeError(str(cleanup_errors)))
        phase = 'fault' if fault else 'finished'
        if fault is not None:
            # 日志关闭阶段也可能产生错误；最终摘要覆盖写入完整清理结果。
            cleanup('final_fault_summary', lambda: (session / 'fault_summary_20260914.json').write_text(
                json.dumps({'fault': fault, 'cleanup_errors': cleanup_errors}, ensure_ascii=False, indent=2), encoding='utf-8'))
        publish(force=True, running=False)
    if fault:
        raise RuntimeError(fault['message'])
    return {'session': str(session), 'episode_count': completed, 'policy_version': agent.policy_version,
            'fin_update_count': agent.update_count, 'live_status': str(session / 'rl_live_status_20260914.json'),
            'status_warning': status_warning}
