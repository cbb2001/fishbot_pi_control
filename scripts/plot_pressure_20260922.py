#!/usr/bin/env python3
"""离线绘制 raw_depth.jsonl 压力曲线，不访问硬件。

示例：
python scripts/plot_pressure_20260922.py logs/某次运行/raw_depth.jsonl --show
python scripts/plot_pressure_20260922.py logs/某次运行/raw_depth.jsonl --unit Pa
依赖：python -m pip install matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import webbrowser


def read_pressure(path):
    """校验记录、按时间排序；重复时间戳保留最后一条有效记录。"""
    values = {}
    skipped = duplicates = 0
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get('ok', True) is not True:
                    raise ValueError('invalid record')
                if row.get('name', 'depth') != 'depth':
                    raise ValueError('not depth')
                timestamp = row['t_ns']
                pressure = row['data']['pressure_pa']
                # 原始时间戳保持整数，先相减再转成秒，避免丢失精度。
                if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                    raise ValueError('invalid timestamp')
                if isinstance(pressure, bool):
                    raise ValueError('invalid pressure')
                pressure = float(pressure)
                if not math.isfinite(pressure) or pressure <= 0:
                    raise ValueError('invalid pressure')
                duplicates += int(timestamp in values)
                values[timestamp] = pressure
            except (ValueError, TypeError, KeyError, OverflowError):
                skipped += 1
    if not values:
        raise ValueError('文件中没有有效的 t_ns / data.pressure_pa 记录')
    rows = sorted(values.items())
    origin = rows[0][0]
    return ([(t-origin)/1e9 for t, _ in rows],
            [p for _, p in rows], skipped, duplicates)


def plot_pressure(source, output=None, unit='mbar'):
    """保存 PNG；只做单位换算，不平滑、不重新标定压力。"""
    source = Path(source).resolve()
    output = (Path(output).resolve() if output else
              source.with_name(source.stem + '_pressure.png'))
    if output == source or output.suffix.lower() != '.png':
        raise ValueError('输出必须为 PNG 文件，且不能覆盖输入日志')
    scale = {'Pa': 1., 'kPa': 1000., 'mbar': 100.}[unit]
    times, pressures, skipped, duplicates = read_pressure(source)

    import matplotlib
    matplotlib.use('Agg')  # 无桌面的树莓派同样可以保存图片。
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'):
        if name in fonts:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False

    # 明显采样断档处断开曲线，避免把缺失区间画成连续观测。
    intervals = [b-a for a, b in zip(times, times[1:])]
    gap_limit = 3 * statistics.median(intervals) if intervals else math.inf
    xs, ys = [], []
    for i, (t, p) in enumerate(zip(times, pressures)):
        if i and t-times[i-1] > gap_limit:
            xs.append(math.nan)
            ys.append(math.nan)
        xs.append(t)
        ys.append(p/scale)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, linewidth=1, marker='.' if len(times) == 1 else None,
            label=f'Pressure ({unit})')
    ax.set(title=f'{source.parent.name} / {source.name}',
           xlabel='Time from first valid sample (s)', ylabel=f'Pressure ({unit})')
    ax.grid(alpha=.3)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.legend(loc='best')
    fig.text(.5, .015, f'N={len(times)} | duration={times[-1]:.2f} s | '
             f'min={min(pressures)/scale:.3f} | max={max(pressures)/scale:.3f} | '
             f'mean={statistics.fmean(pressures)/scale:.3f} {unit} | '
             f'skipped={skipped} | duplicates={duplicates}', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
    return output, len(times), skipped, duplicates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='raw_depth.jsonl 文件路径')
    parser.add_argument('--unit', choices=('Pa', 'kPa', 'mbar'), default='mbar',
                        help='纵轴压力单位，默认 mbar；100 Pa = 1 mbar')
    parser.add_argument('--output', type=Path, help='PNG 输出路径，默认保存到日志旁边')
    parser.add_argument('--show', action='store_true', help='生成后用默认浏览器打开图片')
    args = parser.parse_args(argv)
    try:
        output, count, skipped, duplicates = plot_pressure(args.input, args.output, args.unit)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'绘图失败：{exc}\n')
    print(f'有效采样：{count}；跳过无效记录：{skipped}；合并重复时间戳：{duplicates}')
    print(f'压力曲线：{output}')
    if args.show and not webbrowser.open(output.as_uri()):
        print('未能自动打开浏览器，请手动打开上面的图片。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
