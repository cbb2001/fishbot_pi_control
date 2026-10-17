#!/usr/bin/env python3
"""四基站UWB水面定点导航，仅控制三个尾鳍舵机。

UWB与test_uwb_20261014.py共用SensorManager/UWBWorker，以CRLF按行读取。
先在config/robot.yaml的surface_navigation_20261010中填写L/W/x_d/y_d。
模拟：python scripts/run_surface_navigation_20261010.py --dry-run --L 4 --W 3 --x-d 3 --y-d 1 --t1 0 --t2 .5 --max-control-s 3
实机：python scripts/run_surface_navigation_20261010.py --confirm MOVE
多点：python scripts/run_surface_navigation_20261010.py --waypoint 3 1 30 --waypoint 5 2 40 --confirm MOVE
真实传感器测试、不下发舵机指令：python scripts/run_surface_navigation_20261010.py --no-servo --t 30
"""
from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import fields
from pathlib import Path

import yaml
from _bootstrap import add_project_root

add_project_root()
from control.runtime.data_logger import create_run_log_dir
from control.runtime.surface_navigation_20261010 import NavigationConfig, validate_waypoints
from control.runtime.surface_navigation_session_20261010 import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('config/robot.yaml'))
    parser.add_argument('--waypoint', action='append', nargs=3, type=float, metavar=('X', 'Y', 'SECONDS'),
                        help='顺序目标点：X Y 本段秒数；可重复指定；不能与--x-d/--y-d/--t混用')
    descriptions = {
        'H': '基站天线与水面标签天线的垂直高度差，米，非负；本场地1.93',
        'L': 'A0到A1长度，米', 'W': 'A0到A3宽度，米', 'x_d': '目标x，米', 'y_d': '目标y，米',
        'theta_max': '最大尾鳍幅值，<=30度', 'theta_0': '左摆基础幅值，非负',
        'theta_00': '右摆基础幅值，非负，可与theta_0不同', 't_0': '每个action1时长；0直接写目标',
        't1': '人工放置等待秒数', 't2': '静止IMU标定秒数', 'kp': 'PI比例增益，非负',
        't': '定点保持总时长（秒，包含到点暂停）；到时退出；默认不限时',
        'ki': 'PI积分增益，非负，单位1/s', 'delta_yaw': '兼容旧参数；已不参与到点暂停判定',
        'delta_s': '到点距离阈值，米；暂停后漂到此范围外恢复导航', 'yaw_error_scale_deg': '角差归一化尺度，默认180度',
        'imu_filter_tau_s': 'IMU低通时间常数，秒；0关闭低通', 'yaw_sign': '航向符号：+y方向转动应为正',
        'imu_timeout_s': 'IMU有效样本超时秒数', 'uwb_timeout_s': 'UWB断流回中等待阈值（秒），恢复有效定位后继续',
        'calibration_max_std_deg': '标定最大角度标准差，度', 'calibration_min_samples': '标定最少IMU样本数',
        'boundary_tolerance_m': '兼容旧参数；已取消定位越界判断和坐标裁剪，此值不再影响定位',
        'range_residual_limit_m': '兼容旧参数；已取消测距残差检查，此值不再影响定位',
        'uwb_tag': '选择鱼的标签，如t0:0',
    }
    for field in fields(NavigationConfig):
        option = '--' + field.name.replace('_', '-')
        # 同时保留与公式相同的下划线拼写，便于实验记录直接复用。
        flags = [option] if '_' not in field.name else [option, '--'+field.name]
        if field.name == 't':
            flags.append('--max-control-s')  # 兼容旧命令，现统一为到时立即停机。
        parser.add_argument(*flags, dest=field.name, type=str if field.name == 'uwb_tag' else float,
                            help=descriptions[field.name])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='模拟采集与PWM，不访问硬件')
    modes.add_argument('--no-servo', action='store_true', help='只测试真实IMU/UWB，不创建舵机控制器；t为采集时长')
    parser.add_argument('--confirm', help='实际舵机运动必须显式提供MOVE；no-servo无需确认')
    args = parser.parse_args()
    if args.waypoint is not None and any(value is not None for value in (args.x_d, args.y_d, args.t)):
        parser.error('--waypoint不能与--x-d、--y-d、--t（含别名）混用；每点时间填在该waypoint中')
    if not args.dry_run and not args.no_servo and args.confirm != 'MOVE':
        parser.error('真实运动必须显式提供 --confirm MOVE')
    try:
        robot = yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}
        overrides = {f.name: getattr(args, f.name) for f in fields(NavigationConfig)}
        if args.waypoint is not None:
            # 点表覆盖YAML单点坐标和时长；无需再填写一个无关的单点目标。
            overrides.update(x_d=args.waypoint[0][0], y_d=args.waypoint[0][1], t=args.waypoint[0][2])
        cfg = NavigationConfig.from_robot(robot, **overrides)
        waypoints = validate_waypoints(cfg, args.waypoint)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    session = create_run_log_dir(Path(robot.get('runtime', {}).get('log_dir', 'logs')) /
                                 'surface_navigation_20261010', suffix='surface_navigation')
    stop, previous = threading.Event(), {}

    def request_stop(signum, frame):
        stop.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        result = run_session(robot, cfg, session, dry_run=args.dry_run, no_servo=args.no_servo,
                             stop_event=stop, waypoints=waypoints)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
