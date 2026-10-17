#!/usr/bin/env python3
"""绘制UWB诊断日志中的A0～A3原始距离变化曲线。

示例：
python scripts/plot_uwb_debug_20261014.py logs/uwb_debug_20261014/某次运行目录
python scripts/plot_uwb_debug_20261014.py 某次运行目录/uwb_diagnostics.jsonl --show
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


DIAGNOSTICS_NAME = 'uwb_diagnostics.jsonl'


def resolve_source(source):
    source = Path(source)
    if source.is_dir():
        source = source / DIAGNOSTICS_NAME
    if not source.is_file():
        raise FileNotFoundError(f'找不到UWB诊断日志：{source}')
    return source


def load_diagnostics(source):
    source = resolve_source(source)
    points = []
    error_times = []
    with source.open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{source}第{line_number}行不是有效JSON：{exc}') from exc
            status = row.get('status')
            if status == 'range_error':
                error_times.append(float(row['elapsed_s']))
                continue
            if status != 'mc':
                continue
            ranges = row.get('ranges_m')
            if not isinstance(ranges, list) or len(ranges) != 4:
                raise ValueError(f'{source}第{line_number}行ranges_m必须包含4路距离')
            values = []
            for value in ranges:
                if value is None:
                    values.append(None)
                    continue
                value = float(value)
                values.append(value if math.isfinite(value) else None)
            points.append(dict(elapsed_s=float(row['elapsed_s']),
                               frame_id=str(row.get('frame_id', '')),
                               ranges_m=values))
    if not points:
        raise ValueError(f'{source}中没有mc测距帧')
    return points, error_times, source


def summarize(points, error_times, source):
    rows = []
    for anchor in range(4):
        values = [p['ranges_m'][anchor] for p in points
                  if p['ranges_m'][anchor] is not None]
        invalid = len(points) - len(values)
        rows.append(dict(
            valid_frames=len(values), invalid_frames=invalid,
            valid_percent=100. * len(values) / len(points),
            min_m=min(values) if values else None,
            max_m=max(values) if values else None,
            mean_m=statistics.fmean(values) if values else None,
            median_m=statistics.median(values) if values else None,
            std_m=statistics.pstdev(values) if values else None,
        ))
    times = [p['elapsed_s'] for p in points]
    return dict(source=str(Path(source).resolve()), measurement_frames=len(points),
                device_range_errors=len(error_times),
                first_elapsed_s=min(times), last_elapsed_s=max(times),
                duration_s=max(times) - min(times), ranges=rows)


def export_csv(points, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'uwb_ranges.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['elapsed_s', 'frame_id', 'A0_m', 'A1_m', 'A2_m', 'A3_m'])
        for point in points:
            writer.writerow([point['elapsed_s'], point['frame_id'],
                             *['' if value is None else value for value in point['ranges_m']]])
    return path


def plot_ranges(points, error_times, output_dir, *, title, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = [p['elapsed_s'] for p in points]
    colors = ['#1879ad', '#d97816', '#288748', '#bd4545']
    figure = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    overlay = figure.add_subplot(grid[0, :])
    axes = [figure.add_subplot(grid[1 + k // 2, k % 2]) for k in range(4)]

    for anchor in range(4):
        values = [math.nan if p['ranges_m'][anchor] is None else p['ranges_m'][anchor]
                  for p in points]
        overlay.plot(times, values, color=colors[anchor], linewidth=.9,
                     marker='.', markersize=2.2, label=f'A{anchor}')
        axes[anchor].plot(times, values, color=colors[anchor], linewidth=1,
                          marker='.', markersize=2.4)
        axes[anchor].set(title=f'A{anchor}', ylabel='Slant range (m)')

    all_axes = [overlay, *axes]
    for axis in all_axes:
        for index, error_time in enumerate(error_times):
            axis.axvline(error_time, color='#d62728', alpha=.55, linewidth=.8,
                         linestyle='--', label='RANGE_ERROR' if axis is overlay and index == 0 else None)
        axis.grid(alpha=.25)
        axis.set_xlabel('Elapsed time (s)')
    overlay.set(title='Four-anchor comparison', ylabel='Slant range (m)')
    overlay.legend(ncol=5, loc='best')
    figure.suptitle(title)
    png = output_dir / 'uwb_ranges.png'
    pdf = output_dir / 'uwb_ranges.pdf'
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    if show:
        plt.show()
    plt.close(figure)
    return png, pdf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='运行目录或uwb_diagnostics.jsonl路径')
    parser.add_argument('--output-dir', type=Path, help='输出目录；默认在运行目录的uwb_plots下新建时间戳目录')
    parser.add_argument('--show', action='store_true', help='保存后显示图窗，需要图形桌面')
    args = parser.parse_args(argv)
    try:
        points, error_times, source = load_diagnostics(args.source)
        base = args.output_dir or source.parent / 'uwb_plots' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base.mkdir(parents=True, exist_ok=False)
        csv_path = export_csv(points, base)
        summary = summarize(points, error_times, source)
        summary_path = base / 'summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        png, pdf = plot_ranges(points, error_times, base, title=source.parent.name, show=args.show)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f'测距帧：{summary["measurement_frames"]}；设备测距错误：{summary["device_range_errors"]}')
    for anchor, stats in enumerate(summary['ranges']):
        print(f'A{anchor}: 有效 {stats["valid_frames"]}/{summary["measurement_frames"]} '
              f'({stats["valid_percent"]:.2f}%)，均值={stats["mean_m"]:.3f}m，'
              f'标准差={stats["std_m"]:.3f}m')
    print(f'输出：{base.resolve()}')
    print(f'文件：{png.name}、{pdf.name}、{csv_path.name}、{summary_path.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
