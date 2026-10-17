"""导航的真实IMU/UWB诊断模式；不创建舵机控制器或执行线程。"""
from __future__ import annotations

import json
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from .data_logger import write_metadata_yaml
from .surface_navigation_20261010 import position_from_ranges
from .uwb_debug_20261014 import DiagnosticDecoder


def run_sensor_test(sensors, robot, cfg, session, stop, *, duration_s=None):
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    write_metadata_yaml(session / 'metadata.yaml', {
        'controller': 'surface_navigation_20261010', 'navigation': asdict(cfg),
        'robot_config': robot, 'dry_run': False, 'no_servo': True,
        'active_servo_ids': [], 'max_control_s': duration_s,
        'time_origin': 'sensor_capture_start', 'imu_calibration': False,
    })
    decoder = DiagnosticDecoder(cfg.uwb_tag)
    seen_complete = deque(maxlen=64)
    last_seq = {'imu': None, 'uwb': None}
    latest_fix = None
    latest_ranges = None
    latest_imu = None
    reason, fault = 'stopped', None
    cleanup_errors = []
    start = time.monotonic()
    print_due = start
    with ExitStack() as stack:
        files = {name: stack.enter_context((session / name).open('x', encoding='utf-8', buffering=1))
                 for name in ('raw_imu.jsonl', 'raw_uwb.jsonl', 'uwb_diagnostics.jsonl', 'navigation.jsonl')}

        def write(name, row):
            files[name].write(json.dumps(row, ensure_ascii=False) + '\n')

        def drain():
            nonlocal latest_fix, latest_ranges, latest_imu
            for name in ('imu', 'uwb'):
                for sample in sensors.buffers[name].snapshot():
                    previous = last_seq[name]
                    if previous is not None and sample.seq <= previous:
                        continue
                    last_seq[name] = sample.seq
                    write(f'raw_{name}.jsonl', sample.to_dict())
                    if name == 'imu':
                        latest_imu = sample
                        continue
                    context = {'t_ns': sample.t_ns, 'seq': sample.seq,
                               'elapsed_s': sample.t_ns / 1e9 - start}
                    if previous is not None and sample.seq != previous + 1:
                        decoder.reset_partial()
                        latest_fix = None
                        write('uwb_diagnostics.jsonl', {**context, 'status': 'buffer_gap'})
                    if not sample.ok:
                        empty = sample.error == 'no_signal_or_timeout'
                        if not empty:
                            decoder.reset_partial()
                            latest_fix = None
                        write('uwb_diagnostics.jsonl', {**context,
                              'status': 'empty_read' if empty else 'read_error', 'error': sample.error})
                        continue
                    try:
                        rows = decoder.feed(bytes.fromhex(sample.data['raw_hex']))
                    except (KeyError, ValueError, TypeError) as exc:
                        decoder.reset_partial()
                        latest_fix = None
                        write('uwb_diagnostics.jsonl', {**context, 'status': 'read_error', 'error': str(exc)})
                        continue
                    for row in rows:
                        write('uwb_diagnostics.jsonl', {**context, **row})
                        if row['status'] != 'mc':
                            continue
                        latest_ranges = row['ranges_m']
                        updated = False
                        error = None
                        # 诊断会把缺测也计为已见；定位去重仅记录完整报文，与导航一致。
                        if all(r is not None for r in latest_ranges) and row['frame_id'] not in seen_complete:
                            seen_complete.append(row['frame_id'])
                            try:
                                x, y = position_from_ranges(latest_ranges, cfg.L, cfg.W,
                                    cfg.boundary_tolerance_m, cfg.range_residual_limit_m, H=cfg.H)
                                latest_fix = {'sample_t_ns': sample.t_ns, 'uwb_frame_id': row['frame_id'],
                                              'ranges_m': latest_ranges, 'x': x, 'y': y}
                                updated = True
                            except ValueError as exc:
                                error = str(exc)
                                latest_fix = None
                        write('navigation.jsonl', {**context, 'phase': 'sensor_test',
                              'received_ranges_m': latest_ranges, 'received_frame_id': row['frame_id'],
                              'last_position': latest_fix, 'position_reused': latest_fix is not None and not updated,
                              'position_age_s': None if latest_fix is None else
                                  max(0., (sample.t_ns-latest_fix['sample_t_ns'])/1e9),
                              'position_error': error})

        try:
            print('传感器测试：真实IMU/UWB采集，不创建舵机控制器、不下发PWM；Ctrl+C停止。', flush=True)
            sensors.start_all()
            while not stop.is_set():
                drain()
                if sensors.fatal_errors():
                    raise RuntimeError(str(sensors.fatal_errors()))
                now = time.monotonic()
                if now >= print_due:
                    ranges = '尚无测距' if latest_ranges is None else ' '.join(
                        f'A{i}={r:.3f}m' if r is not None else f'A{i}=INVALID'
                        for i, r in enumerate(latest_ranges))
                    position = '尚无有效坐标' if latest_fix is None else (
                        f"x={latest_fix['x']:.3f} y={latest_fix['y']:.3f} "
                        f"完整定位年龄={max(0., now-latest_fix['sample_t_ns']/1e9):.2f}s")
                    imu = '尚无IMU' if latest_imu is None else (
                        f"IMU yaw={latest_imu.data.get('yaw_deg')} pitch={latest_imu.data.get('pitch_deg')} "
                        f"roll={latest_imu.data.get('roll_deg')} ok={latest_imu.ok} "
                        f"age={max(0., now-latest_imu.t_ns/1e9):.2f}s")
                    print(f'{ranges}; {position}; {imu}', flush=True)
                    print_due = now + 1.
                if duration_s is not None and now-start >= duration_s:
                    reason = 'time_limit'
                    break
                stop.wait(.005)
        except (KeyboardInterrupt, InterruptedError):
            reason = 'stopped'
        except Exception as exc:
            fault = f'{type(exc).__name__}: {exc}'
        finally:
            stop.set()
            try:
                sensors.stop_all(timeout=5.)
                alive = getattr(sensors, 'alive_worker_names', lambda: [])()
                if alive or sensors.fatal_errors():
                    cleanup_errors.append(str(alive or sensors.fatal_errors()))
            except Exception as exc:
                cleanup_errors.append(f'sensors: {exc}')
            try:
                drain()
                if decoder.pending:
                    write('uwb_diagnostics.jsonl', {'status': 'partial_at_exit', 'raw_hex': decoder.pending.hex()})
            except Exception as exc:
                cleanup_errors.append(f'logs: {exc}')
    result = {'session': str(session), 'reason': 'fault' if fault or cleanup_errors else reason,
              'no_servo': True, 'completed_pairs': 0, 'executor_alive': False,
              'last_position': latest_fix, 'fault': fault, 'cleanup_errors': cleanup_errors,
              'elapsed_s': time.monotonic()-start}
    (session / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if fault or cleanup_errors:
        raise RuntimeError(f'传感器测试退出：{fault or cleanup_errors}；日志：{session}')
    return result
