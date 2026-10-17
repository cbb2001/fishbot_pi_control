#!/usr/bin/env python3
"""可配置多阶段侧鳍下潜/上浮与尾鳍左右摆动联合控制。

侧鳍复用固定深度双向PI：浅于目标下潜、深于目标上浮，整组动作执行完再决策。
0米上浮阶段使用上一下潜目标作为误差归一化尺度；尾鳍全程独立左右摆动。

模拟：python scripts/run_fixed_dive_tail_20261009.py --dry-run --max-control-s 5
实机：python scripts/run_fixed_dive_tail_20261009.py --h0-dive1 1 --confirm MOVE
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
from control.runtime.fixed_dive_tail_20261009 import (
    TailStage, _validate_stages, prepare_fixed_config, run_session,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/robot.yaml')
    parser.add_argument('--rl-config', default='config/rl_training_20260914.yaml')
    parser.add_argument('--stage', type=float, nargs=2, action='append', metavar=('H', 'T'),
                        help='一个阶段：目标深度H（米）和持续时间T（秒）；可重复，按输入顺序执行；覆盖旧四阶段参数')
    parser.add_argument('--h0-dive1', type=float, default=1.0, help='第一阶段下潜目标深度，默认1 m')
    parser.add_argument('--h0-dive2', type=float, default=1.5, help='第三阶段下潜目标深度，默认1.5 m')
    parser.add_argument('--stage-duration-s', type=float, default=30., help='每个阶段时长，默认30 s')
    parser.add_argument('--k', '--kp', type=float, default=1., help='下潜/上浮共用比例增益，默认1')
    parser.add_argument('--ki', type=float, default=0., help='下潜/上浮共用积分增益，默认0；正值启用PI')
    parser.add_argument('--side-action-duration-s', '--t', type=float, default=.6,
                        help='侧鳍 action2 单动作时长，默认0.6 s')
    parser.add_argument('--tail-action-duration-s', type=float, default=.3,
                        help='尾鳍左右摆动单次时长，默认0.3 s')
    parser.add_argument('--tail-left-deg', type=float, default=-15., help='尾鳍左摆幅度，默认-15°')
    parser.add_argument('--tail-right-deg', type=float, default=15., help='尾鳍右摆幅度，默认+15°')
    parser.add_argument('--initial-b2', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--surface-pressure-pa', type=float, default=None)
    parser.add_argument('--pressure-per-meter', type=float, default=None)
    parser.add_argument('--pressure-drift-config', type=Path)
    parser.add_argument('--calibration-window-s', type=float)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--confirm')
    parser.add_argument('--max-control-s', type=float, help='控制总时长；默认各阶段时长之和，等当前侧鳍组合完成后退出')
    parser.add_argument('--pretrain-wait-s', type=float)
    parser.add_argument('--keep-pwm', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and args.confirm != 'MOVE':
        parser.error('真实运动必须显式提供 --confirm MOVE')
    if args.stage is not None:
        stages = tuple(TailStage(t, h, f'stage_{i}')
                       for i, (h, t) in enumerate(args.stage, start=1))
    else:
        stages = (TailStage(args.stage_duration_s, args.h0_dive1, 'dive_1m'),
                  TailStage(args.stage_duration_s, 0., 'surface_1'),
                  TailStage(args.stage_duration_s, args.h0_dive2, 'dive_1_5m'),
                  TailStage(args.stage_duration_s, 0., 'surface_2'))
    try:
        stages = _validate_stages(stages)
    except ValueError as exc:
        parser.error(str(exc))
    raw = yaml.safe_load(Path(args.rl_config).read_text(encoding='utf-8')) or {}
    robot = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    cfg = prepare_fixed_config(raw, dry_run=args.dry_run, pretrain_wait_s=args.pretrain_wait_s,
                               calibration_window_s=args.calibration_window_s)
    session = create_run_log_dir(Path(robot.get('runtime', {}).get('log_dir', 'logs')) / 'fixed_dive_tail_20261009', suffix='fixed_dive_tail')
    stop = threading.Event(); previous = {}
    def request_stop(signum, frame): stop.set()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): previous[sig] = signal.signal(sig, request_stop)
        result = run_session(cfg, robot, session, k=args.k, ki=args.ki, initial_b2=args.initial_b2,
            side_action_duration_s=args.side_action_duration_s,
            tail_action_duration_s=args.tail_action_duration_s,
            tail_left_deg=args.tail_left_deg, tail_right_deg=args.tail_right_deg,
            stages=stages, dry_run=args.dry_run, max_control_s=args.max_control_s,
            keep_pwm=args.keep_pwm, stop_event=stop, surface_pressure_pa=args.surface_pressure_pa,
            pressure_per_meter=args.pressure_per_meter, pressure_drift_config=args.pressure_drift_config)
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
