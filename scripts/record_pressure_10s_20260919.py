"""独立压力记录：默认立即采集30秒；不标定压力、不控制舵机。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import time

from _bootstrap import add_project_root

ROOT = add_project_root()
from control.safety import ensure_not_windows_hardware_run, load_robot_config
from drivers.depth_sensor import DepthSensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.yaml')
    parser.add_argument('--wait-s', type=float, default=0., help='开始记录前的等待秒数，默认0（不等待）')
    parser.add_argument('--duration-s', type=float, default=30., help='记录时长，默认30秒')
    parser.add_argument('--rate-hz', type=float, default=20., help='目标采样频率，默认20Hz，最大20Hz')
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'logs/pressure_20260919')
    args = parser.parse_args()
    for name in ('wait_s', 'duration_s', 'rate_hz'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != 'wait_s' and value == 0):
            parser.error(f'{name} must be finite and {"nonnegative" if name == "wait_s" else "positive"}')
    if args.rate_hz > 20:
        parser.error('rate_hz must be <= 20')
    ensure_not_windows_hardware_run()
    config = load_robot_config(args.config)
    session = args.log_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True, exist_ok=False)
    sensor = DepthSensor(config)
    summary = {'status': 'initializing', 'config': str(args.config.resolve()),
               'wait_s': args.wait_s, 'duration_s': args.duration_s, 'rate_hz': args.rate_hz,
               'i2c_bus': sensor.bus, 'i2c_address': sensor.address,
               'pressure_zero_performed': False, 'sample_count': 0}
    pressures = []
    first_elapsed = last_elapsed = None
    exit_code = 0

    def save_summary():
        (session / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        if not sensor.begin():
            raise RuntimeError('MS5837 初始化失败，请检查连接和 I2C 地址')
        print(f'日志目录：{session}\n记录时长：{args.duration_s:g} 秒。', flush=True)
        # 默认跳过等待；仅显式指定 --wait-s 时延迟开始，不调用 zero()。
        if args.wait_s > 0:
            summary.update(status='waiting', wait_started_at=datetime.now().isoformat())
            save_summary()
            print(f'等待 {args.wait_s:g} 秒。', flush=True)
            time.sleep(args.wait_s)
        summary.update(status='recording', capture_started_at=datetime.now().isoformat())
        save_summary()
        print('开始记录压力。', flush=True)
        # 每次采样后立即刷新文件；配合 nohup，水下断开 SSH 仍可独立记录。
        with (session / 'pressure.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', 'elapsed_s', 'pressure_pa', 'pressure_mbar', 'temperature_c'])
            stream.flush()
            start = time.monotonic()
            deadline = start + args.duration_s
            next_tick = start
            while time.monotonic() < deadline:
                reading = sensor.read()
                now = time.monotonic()
                if now > deadline:
                    break  # 不把记录窗口结束后才完成的采样写入本次数据。
                values = (reading.pressure_mbar*100., reading.pressure_mbar, reading.temperature_c)
                if not all(math.isfinite(v) for v in values) or reading.pressure_mbar <= 0:
                    raise ValueError('压力或温度读数无效')
                elapsed = now - start
                writer.writerow([datetime.now().isoformat(timespec='milliseconds'), elapsed, *values])
                stream.flush()
                pressures.append(values[0])
                if first_elapsed is None:
                    first_elapsed = elapsed
                last_elapsed = elapsed
                next_tick = max(next_tick + 1./args.rate_hz, now)
                time.sleep(max(0., min(next_tick, deadline) - time.monotonic()))
        if not pressures:
            raise RuntimeError('记录窗口内没有完成任何压力采样')
        summary.update(status='complete', capture_elapsed_s=time.monotonic()-start)
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
        exit_code = 130
    except Exception as exc:
        summary.update(status='error', error=f'{type(exc).__name__}: {exc}')
        exit_code = 1
    finally:
        try:
            sensor.close()
        except Exception as exc:
            summary['close_error'] = f'{type(exc).__name__}: {exc}'
            summary['status'] = 'error'
            exit_code = 1
        signal.signal(signal.SIGTERM, previous_sigterm)
        summary.update(sample_count=len(pressures), finished_at=datetime.now().isoformat(),
                       first_sample_elapsed_s=first_elapsed, last_sample_elapsed_s=last_elapsed)
        if pressures:
            summary.update(pressure_min_pa=min(pressures), pressure_max_pa=max(pressures),
                           pressure_mean_pa=sum(pressures)/len(pressures))
        save_summary()
    print(f'结束：{summary["status"]}，共 {len(pressures)} 条；日志：{session}', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
