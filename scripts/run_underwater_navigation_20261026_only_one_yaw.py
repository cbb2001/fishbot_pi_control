#!/usr/bin/env python3
"""水面调向后锁定 yaw 偏差的水下多目标导航。
先放置 t1、静止标定 t2，再接收有效 UWB；胸鳍保持归中，尾鳍先调向。
角差进入 delta_yaw 范围后下潜；本水下航段不根据 IMU/估计位置更新 yaw 偏差。
t3 从首次水面调向起计时，包含后续上浮和取数；t4 从每次下潜起计时。
t4=t3 时无周期上浮；目标超时切换仍会触发重新定位。
目标为 (x,y,h,t)：h 是深度（米，向下为正），t 是到达超时秒数或 NULL。
t=NULL 时持续使用该目标，不自动切换；最后一个目标（含单目标）的 t 必须为 NULL。

单点：python scripts/run_underwater_navigation_20261026_only_one_yaw.py --x-d 10 --y-d 10 --h0 .3 --confirm MOVE
多点：python scripts/run_underwater_navigation_20261026_only_one_yaw.py --waypoint 10 10 .3 30 --waypoint 15 10 .4 NULL --confirm MOVE
模拟：python scripts/run_underwater_navigation_20261026_only_one_yaw.py --dry-run --t1 0 --t2 1 --t3 15 --t4 2
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
from control.runtime.underwater_navigation_20261026_only_one_yaw import UnderwaterConfig
from control.runtime.underwater_navigation_session_20261026_only_one_yaw import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('config/robot.yaml'))
    parser.add_argument('--output', type=Path, help='新的日志目录；默认在 logs 下自动创建时间戳目录')
    arguments = {
        'h0': '单目标水深，米，向下为正', 'x_d': '单目标 X，米', 'y_d': '单目标 Y，米',
        't1': '人工放置时间', 't2': '固定静止标定窗口',
        't3': '总任务时间，包含水面调向、上浮取数',
        't4': '每段下潜/导航最长时间，到期上浮；等于 t3 时无周期上浮，目标超时仍可触发上浮',
        'depth_kp': '胸鳍比例系数', 'depth_ki': '胸鳍积分系数',
        'fin_duration_s': '胸鳍每个半程时长，秒', 'theta_max': '尾鳍最大幅值，度',
        'theta_0': '尾鳍左基础幅值', 'theta_00': '尾鳍右基础幅值', 't_0': '尾鳍每个半程时长',
        'kp': '尾鳍航向比例系数', 'ki': '尾鳍固定航向偏差积分系数',
        'delta_s': '有效水面 UWB 到点容差，米',
        'delta_yaw': '允许开始下潜的水面角差阈值，度，须在 (0,45) 内',
        'yaw_error_scale_deg': '尾鳍 PI 航向误差归一化尺度',
    }
    for name, help_text in arguments.items():
        parser.add_argument('--'+name.replace('_', '-'), dest=name, type=float, help=help_text)
    parser.add_argument('--waypoint', dest='waypoints', nargs=4, action='append',
                        metavar=('X', 'Y', 'H', 'T'),
                        help='按顺序设置 (x,y,h,t)，可重复；t 为到达超时秒数或 NULL；末点/单点必须为 NULL；覆盖单点 x-d/y-d/h0')
    parser.add_argument('--initial-b2', type=int, choices=(-1, 1), help='胸鳍首次 b1=1 的 b2')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='全模拟，不访问硬件')
    modes.add_argument('--no-servo', action='store_true', help='真实采集、存图与估计，不初始化 PWM')
    parser.add_argument('--confirm', help='实际运动需提供 MOVE')
    args = parser.parse_args()
    if not args.dry_run and not args.no_servo and args.confirm != 'MOVE':
        parser.error('真实运动须显式提供 --confirm MOVE')
    try:
        robot = yaml.safe_load(args.config.read_text(encoding='utf8')) or {}
        cfg = UnderwaterConfig.from_robot(robot, **{
            name: getattr(args, name) for name in (*arguments, 'initial_b2', 'waypoints')
        })
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            session = args.output
        else:
            session = create_run_log_dir(
                Path(robot.get('runtime', {}).get('log_dir', 'logs'))/'underwater_navigation_20261026_only_one_yaw',
                suffix='underwater_navigation_only_one_yaw',
            )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    stop = threading.Event()
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
        result = run_session(robot, cfg, session, dry_run=args.dry_run,
                             no_servo=args.no_servo, stop_event=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), flush=True)
    return 0 if result['reason'] == 'time_limit' and not result['fault'] and not result['cleanup_errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
