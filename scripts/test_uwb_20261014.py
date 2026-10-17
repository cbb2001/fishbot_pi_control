#!/usr/bin/env python3
"""UWB实时诊断，沿用导航串口配置、按CRLF逐行采集，不操作舵机。

持续观察：python scripts/test_uwb_20261014.py
定时采集：python scripts/test_uwb_20261014.py --t 60
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import threading

import yaml

try:
    from ._bootstrap import add_project_root
except ImportError:
    from _bootstrap import add_project_root

add_project_root()
from control.runtime.uwb_debug_20261014 import run_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('config/robot.yaml'))
    parser.add_argument('--t', '--seconds', type=float, help='采集秒数；省略则直到Ctrl+C')
    parser.add_argument('--port', help='覆盖sensors.uwb.port')
    parser.add_argument('--baudrate', type=int, help='覆盖sensors.uwb.baudrate')
    parser.add_argument('--tag', help='默认读取导航uwb_tag，缺省t0:0')
    parser.add_argument('--output-dir', type=Path, help='日志根目录，每次建立独立子目录')
    parser.add_argument('--quiet-raw', action='store_true', help='不在终端显示原始串口片段，仍全部保存')
    args = parser.parse_args()
    stop, previous = threading.Event(), {}
    try:
        robot = yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}
        cfg = robot.setdefault('sensors', {}).setdefault('uwb', {})
        if args.port is not None:
            cfg['port'] = args.port
        if args.baudrate is not None:
            cfg['baudrate'] = args.baudrate
        tag = args.tag or robot.get('surface_navigation_20261010', {}).get('uwb_tag', 't0:0')
        base = args.output_dir or Path(robot.get('runtime', {}).get('log_dir', 'logs')) / 'uwb_debug_20261014'
        directory = base / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
        print(f'日志目录：{directory.resolve()}', flush=True)
        result = run_capture(robot, directory, duration_s=args.t, tag=tag, stop_event=stop,
                             show_raw=not args.quiet_raw, emit=lambda s: print(s, flush=True))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result['error'] or result['read_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
