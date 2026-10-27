#!/usr/bin/env python3
"""定时上浮校正的水下单目标导航。先放置 t1、静止标定 t2，再立即开始任务。

实机：python scripts/run_underwater_navigation_20261026.py --h0 .3 --x-d 10 --y-d 10 --confirm MOVE
模拟：python scripts/run_underwater_navigation_20261026.py --dry-run --t1 0 --t2 1 --t3 15 --t4 2
只采集：python scripts/run_underwater_navigation_20261026.py --no-servo --t3 60
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
from control.runtime.underwater_navigation_20261026 import UnderwaterConfig
from control.runtime.underwater_navigation_session_20261026 import run_session


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,default=Path('config/robot.yaml'))
    parser.add_argument('--output',type=Path,help='新的日志目录；默认在 logs 下自动创建时间戳目录')
    arguments={'h0':'目标水深，米，向下为正','x_d':'目标 X，米','y_d':'目标 Y，米',
        't1':'人工放置时间','t2':'固定静止标定窗口','t3':'总任务时间，包含上浮取数',
        't4':'每段导航最长时间，到期上浮','depth_kp':'胸鳍比例系数','depth_ki':'胸鳍积分系数',
        'fin_duration_s':'胸鳍每个半程时长，秒','theta_max':'尾鳍最大幅值，度',
        'theta_0':'尾鳍左基础幅值','theta_00':'尾鳍右基础幅值','t_0':'尾鳍每个半程时长',
        'kp':'尾鳍航向比例系数','ki':'尾鳍航向积分系数','delta_s':'估计位置到点容差，米',
        'yaw_error_scale_deg':'尾鳍 PI 航向误差归一化尺度'}
    for name,help_text in arguments.items():
        parser.add_argument('--'+name.replace('_','-'),dest=name,type=float,help=help_text)
    parser.add_argument('--initial-b2',type=int,choices=(-1,1),help='胸鳍首次 b1=1 的 b2')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true',help='全模拟，不访问硬件')
    modes.add_argument('--no-servo',action='store_true',help='真实采集、存图与估计，不初始化 PWM')
    parser.add_argument('--confirm',help='实际运动需提供 MOVE')
    args=parser.parse_args()
    if not args.dry_run and not args.no_servo and args.confirm!='MOVE':
        parser.error('真实运动须显式提供 --confirm MOVE')
    try:
        robot=yaml.safe_load(args.config.read_text(encoding='utf8')) or {}
        cfg=UnderwaterConfig.from_robot(robot,**{n:getattr(args,n) for n in (*arguments,'initial_b2')})
        if cfg.h0>=float(robot['safety']['max_depth_m']): raise ValueError('h0 必须低于安全深度')
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False); session=args.output
        else:
            session=create_run_log_dir(Path(robot.get('runtime',{}).get('log_dir','logs'))/
                                       'underwater_navigation_20261026',suffix='underwater_navigation')
    except (OSError,ValueError,TypeError,KeyError,yaml.YAMLError) as exc:
        parser.error(str(exc))
    stop=threading.Event(); previous={}
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): previous[sig]=signal.signal(sig,lambda *_:stop.set())
        result=run_session(robot,cfg,session,dry_run=args.dry_run,no_servo=args.no_servo,stop_event=stop)
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
    print(yaml.safe_dump(result,allow_unicode=True,sort_keys=False),flush=True)
    return 0 if result['reason']=='time_limit' and not result['fault'] and not result['cleanup_errors'] else 1


if __name__=='__main__': raise SystemExit(main())
