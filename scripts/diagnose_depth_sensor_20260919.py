"""仅采集 MS5837；不导入舵机控制器，可脱离 SSH 在树莓派上完成记录。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from datetime import datetime

from _bootstrap import add_project_root

add_project_root()
from control.safety import ensure_not_windows_hardware_run, load_robot_config
from drivers.depth_sensor import DepthSensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-s', type=float, default=180.)
    parser.add_argument('--rate-hz', type=float, default=5.)
    parser.add_argument('--surface-pressure-pa', type=float, required=True,
                        help='已在空气/水面上方实测的参考压力；采集中保持不变。')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('duration_s', 'rate_hz', 'surface_pressure_pa'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.rate_hz > 20:
        parser.error('rate_hz must be <= 20')
    ensure_not_windows_hardware_run()
    sensor = DepthSensor(load_robot_config())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建文件，防止重复测试覆盖已有证据；逐条刷新以便断网后恢复读取。
    with args.output.open('x', encoding='utf-8', buffering=1) as log:
        def emit(row):
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

        try:
            if not sensor.begin():
                raise RuntimeError('MS5837 initialization failed')
            # 诊断时同时保存 ADC 与 PROM 系数，不改变驱动的采样/补偿算法。
            driver = sensor._sensor
            adc = {}
            original_read_adc = driver._read_adc

            def capture_adc(cmd):
                value = original_read_adc(cmd)
                adc[cmd] = value
                return value

            driver._read_adc = capture_adc
            start = time.monotonic()
            emit({'event': 'start', 'wall_time': datetime.now().isoformat(),
                  'surface_pressure_pa': args.surface_pressure_pa,
                  'duration_s': args.duration_s, 'rate_hz': args.rate_hz,
                  'bus': sensor.bus, 'address': sensor.address,
                  'prom_c1_to_c6': list(driver.c[1:7])})
            count = 0
            next_sample = start
            while time.monotonic() - start < args.duration_s:
                reading = sensor.read()
                pressure = reading.pressure_mbar * 100.
                count += 1
                emit({'event': 'sample', 'sample': count,
                      'wall_time': datetime.now().isoformat(timespec='milliseconds'),
                      'elapsed_s': time.monotonic() - start,
                      'pressure_pa': pressure, 'temperature_c': reading.temperature_c,
                      'depth_m': (pressure - args.surface_pressure_pa) / 9806.65,
                      'adc_d1': adc[driver._CMD_CONVERT_D1_8192],
                      'adc_d2': adc[driver._CMD_CONVERT_D2_8192]})
                next_sample += 1. / args.rate_hz
                # 超时不集中补采样，避免形成不真实的高频数据。
                next_sample = max(next_sample, time.monotonic())
                time.sleep(max(0., min(next_sample, start + args.duration_s) - time.monotonic()))
            emit({'event': 'complete', 'samples': count, 'elapsed_s': time.monotonic()-start})
        except BaseException as exc:
            emit({'event': 'error', 'error': f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            sensor.close()


if __name__ == '__main__':
    main()
