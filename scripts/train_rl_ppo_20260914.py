#!/usr/bin/env python3
"""树莓派侧鳍在线 PPO 入口；默认只做 Fin 控制，不操作尾部三舵机。"""
from __future__ import annotations
import argparse
import signal
import threading
from pathlib import Path
import yaml
from _bootstrap import add_project_root
add_project_root()
from control.runtime.data_logger import create_run_log_dir
from control.runtime.rl_online_training_20260914 import run_session
from control.runtime.rl_training_config_20260914 import prepare_config, resolve_resume

def parse_args():
    p=argparse.ArgumentParser(description='Real machine-fish fin-only online PPO')
    p.add_argument('--rl-config',default='config/rl_training_20260914.yaml')
    p.add_argument('--config',default='config/robot.yaml')
    p.add_argument('--h0',type=float,required=True); p.add_argument('--theta0',type=float,required=True)
    p.add_argument('--confirm'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--mock-sensors',action='store_true')
    p.add_argument('--resume',nargs='?',const='latest'); p.add_argument('--seed',type=int)
    p.add_argument('--max-training-s',type=float); p.add_argument('--max-episodes',type=int)
    p.add_argument('--episode-duration-s',type=float); p.add_argument('--pretrain-wait-s',type=float)
    p.add_argument('--keep-pwm',action='store_true')
    return p.parse_args()

def main():
    args=parse_args()
    if not args.dry_run and args.confirm!='MOVE':
        raise SystemExit('真实训练必须显式提供 --confirm MOVE')
    rl_raw=yaml.safe_load(Path(args.rl_config).read_text(encoding='utf-8')) or {}
    robot_cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    cfg=prepare_config(rl_raw,dry_run=args.dry_run,overrides={
        'h0':args.h0,'theta0':args.theta0,'max_episodes':args.max_episodes,
        'max_training_s':args.max_training_s,'episode_duration_s':args.episode_duration_s,
        'pretrain_wait_s':args.pretrain_wait_s})
    root=Path(robot_cfg.get('runtime',{}).get('log_dir','logs'))/'rl_20260914'
    session=create_run_log_dir(root,suffix='fin_only')
    resume=resolve_resume(args.resume,root) if args.resume else None
    # Ctrl+C 和 SIGTERM 只置位；监测线程请求关闭 PWM，主循环统一清理。
    stop_event = threading.Event()
    previous = {}
    def request_stop(signum, frame):
        stop_event.set()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, request_stop)
        result=run_session(cfg,robot_cfg,session,dry_run=args.dry_run,seed=args.seed,resume=resume,
                           keep_pwm=args.keep_pwm,stop_event=stop_event)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(yaml.safe_dump(result,allow_unicode=True,sort_keys=False),flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
