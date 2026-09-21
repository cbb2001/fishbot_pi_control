#!/usr/bin/env python3
"""固定策略下潜：整组完成后再判断保持；深度采集与动作执行并行。

模拟：python scripts/run_fixed_dive_20260920_100122.py --h0 0.3 --k 0.5 --dry-run --max-control-s 5
实机：python scripts/run_fixed_dive_20260920_100122.py --h0 0.3 --k 0.5 --confirm MOVE
"""
from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

import yaml
from _bootstrap import add_project_root

add_project_root()
from control.runtime.data_logger import create_run_log_dir
from control.runtime.fixed_dive_20260920_100122 import prepare_fixed_config, run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/robot.yaml')
    parser.add_argument('--rl-config', default='config/rl_training_20260914.yaml', help='复用采集、标定、安全及运行配置')
    parser.add_argument('--h0', type=float, required=True, help='期望深度（米），必须大于 0')
    parser.add_argument('--k', type=float, default=1., help='非负增益，默认 1')
    parser.add_argument('--action-duration-s', '--t', type=float, default=.6,
                        help='每个 action2 的时长 t（秒），默认 0.6；一组两动作共约 2t 秒')
    parser.add_argument('--initial-b2', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--surface-pressure-pa', type=float, default=None,
                        help='人工压力参考（Pa）；默认空，启动时采集稳定压力均值标定')
    parser.add_argument('--pressure-per-meter', type=float, default=None,
                        help='压力-深度系数（Pa/m），默认750；h=(P-P0)/系数')
    parser.add_argument('--dry-run', action='store_true', help='模拟传感器和 PWM，不访问硬件')
    parser.add_argument('--confirm', help='实机必须传入 MOVE')
    parser.add_argument('--max-control-s', type=float, help='标定后运行秒数；到时等本组完成再退出')
    parser.add_argument('--pretrain-wait-s', type=float, help='覆盖启动前等待时间')
    parser.add_argument('--keep-pwm', action='store_true', help='正常退出回中后保留 PWM，故障仍关闭')
    args = parser.parse_args()
    if not args.dry_run and args.confirm != 'MOVE':
        parser.error('真实运动必须显式提供 --confirm MOVE')
    raw = yaml.safe_load(Path(args.rl_config).read_text(encoding='utf-8')) or {}
    robot = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    cfg = prepare_fixed_config(raw, dry_run=args.dry_run, pretrain_wait_s=args.pretrain_wait_s)
    session = create_run_log_dir(Path(robot.get('runtime', {}).get('log_dir', 'logs')) / 'fixed_dive_20260920_100122', suffix='fixed_dive')
    stop = threading.Event()
    previous = {}
    def request_stop(signum, frame):
        stop.set()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        result = run_session(cfg, robot, session, h0=args.h0, k=args.k,
                             action_duration_s=args.action_duration_s,
                             surface_pressure_pa=args.surface_pressure_pa,
                             pressure_per_meter=args.pressure_per_meter,
                             initial_b2=args.initial_b2, dry_run=args.dry_run,
                             max_control_s=args.max_control_s, keep_pwm=args.keep_pwm, stop_event=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
