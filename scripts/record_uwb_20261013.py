#!/usr/bin/env python3
"""仅采集UWB指定t秒，保存原始报文、四路测距CSV及PNG/PDF曲线。

采集：python scripts/record_uwb_20261013.py --t 30
显示：python scripts/record_uwb_20261013.py --t 30 --show
已有日志：python scripts/record_uwb_20261013.py --input logs/某次/raw_uwb.jsonl
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from _bootstrap import add_project_root
add_project_root()

import yaml
from control.runtime.uwb_capture_20261013 import capture, export_ranges, plot_ranges


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--t','--seconds',type=float,default=30.,help='纯采集时长，秒，默认30，必须>0')
    parser.add_argument('--config',type=Path,default=Path('config/robot.yaml'))
    parser.add_argument('--port',help='覆盖sensors.uwb.port，默认/dev/ttyAMA0')
    parser.add_argument('--baudrate',type=int,help='覆盖sensors.uwb.baudrate，默认115200')
    parser.add_argument('--tag',help='mc标签，默认读取导航配置或t0:0')
    parser.add_argument('--input',type=Path,help='只读取已有raw_uwb.jsonl画图，不打开串口；不使用t')
    parser.add_argument('--output-dir',type=Path,help='输出根目录；每次在其中新建唯一子目录，避免覆盖')
    parser.add_argument('--show',action='store_true',help='保存图片后显示图窗，需要图形桌面')
    args=parser.parse_args()
    if not math.isfinite(args.t) or args.t<=0:
        parser.error('t必须为正有限秒数')
    # 离线绘图不要求本机存在硬件配置文件。
    if not args.config.exists() and not args.input:
        parser.error(f'找不到配置文件：{args.config}')
    robot=(yaml.safe_load(args.config.read_text(encoding='utf-8')) or {}) if args.config.exists() else {}
    cfg=robot.get('sensors',{}).get('uwb',{})
    tag=args.tag or robot.get('surface_navigation_20261010',{}).get('uwb_tag','t0:0')
    import re
    if not re.fullmatch(r't\d+:\d+',tag):
        parser.error('tag格式应为t0:0')
    port=args.port or cfg.get('port','/dev/ttyAMA0')
    baud=args.baudrate if args.baudrate is not None else int(cfg.get('baudrate',115200))
    if baud<=0:
        parser.error('baudrate必须>0')
    if args.input and not args.input.is_file():
        parser.error(f'找不到原始日志：{args.input}')
    base=args.output_dir or (args.input.parent/'uwb_plots' if args.input else Path(robot.get('runtime',{}).get('log_dir','logs'))/'uwb_capture_20261013')
    directory=base/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True,exist_ok=False)
    result=dict(reason='offline',error=None)
    raw=args.input
    if raw is None:
        print(f'仅采集UWB：{port}，{baud} baud，{args.t:g}s；不初始化舵机或其他传感器。',flush=True)
        try:
            import serial
            with serial.Serial(port,baudrate=baud,timeout=.2) as device:
                result=capture(device,args.t,directory)
        except Exception as exc:
            result=dict(reason='error',error=f'{type(exc).__name__}: {exc}')
        raw=directory/'raw_uwb.jsonl'
    # capture结果已包含requested_duration_s，使用字典合并，避免重复关键字导致采集后崩溃。
    metadata = dict(source=str(raw.resolve()),port=port,baudrate=baud,tag=tag,
                    requested_duration_s=None if args.input else args.t)
    metadata.update(result)
    (directory/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'输出目录：{directory.resolve()}',flush=True)
    if not raw.exists():
        print(f"采集失败：{result['error']}")
        return 1
    points,summary=export_ranges(raw,directory,tag)
    summary.update(result)
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"完整有效mc帧：{summary['valid_frames']}；无效mc帧：{summary['invalid_mc_frames']}；空读/错误：{summary['empty_or_failed_reads']}",flush=True)
    try:
        plot_ranges(points,directory,show=args.show)
    except Exception as exc:
        print(f'原始数据和CSV已保存，绘图/显示失败：{exc}\n请安装matplotlib；SSH无桌面时不要加--show。')
        return 1
    print('已生成 uwb_ranges.png、uwb_ranges.pdf、uwb_ranges.csv。')
    if result['error']:
        print(f"采集提前结束：{result['error']}")
    return 1 if result['error'] else 0


if __name__=='__main__':
    raise SystemExit(main())
