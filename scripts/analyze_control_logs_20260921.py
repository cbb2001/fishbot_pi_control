#!/usr/bin/env python3
"""离线分析机器鱼运行日志；只读取文件，不连接树莓派或初始化硬件。

python scripts/analyze_control_logs_20260921.py logs/某次运行 --show
python scripts/analyze_control_logs_20260921.py --latest --logs logs --show
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import webbrowser
from pathlib import Path

import numpy as np
import yaml
import matplotlib

matplotlib.use('Agg')  # 无桌面的环境也可导出图片；--show 用浏览器展示报告。
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ('raw_depth.jsonl', 'raw_imu.jsonl', 'raw_power.jsonl')


def read_jsonl(path, warnings):
    """允许日志缺失、空文件和中断留下的残行，并将问题写入报告。"""
    if not path.exists():
        warnings.append(f'缺少 {path.name}')
        return []
    rows = []
    bad = 0
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not an object')
                rows.append(row)
            except (ValueError, TypeError):
                bad += 1
    if bad:
        warnings.append(f'{path.name}: 跳过 {bad} 行损坏记录')
    return rows


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def load_metadata(session):
    path = session / 'metadata.yaml'
    return (yaml.safe_load(path.read_text(encoding='utf-8-sig')) or {}) if path.exists() else {}


def find_latest(root, include_simulation=False):
    """优先使用目录名中的运行时间，而非拷贝文件时改变的修改时间。"""
    candidates = {p.parent for name in RAW_FILES for p in root.rglob(name)}
    candidates = [p for p in candidates if include_simulation or load_metadata(p).get('dry_run') is not True]
    if not candidates:
        raise ValueError(f'{root} 下没有可分析的运行目录')
    def key(p):
        match = re.search(r'(\d{8}_\d{6})', p.name)
        return (match.group(1) if match else '', p.stat().st_mtime, str(p))
    return max(candidates, key=key)


def samples(rows, name, warnings):
    valid = [r for r in rows if r.get('ok', True) and number(r.get('t_ns')) is not None
             and isinstance(r.get('data'), dict)]
    rejected = len(rows)-len(valid)
    if rejected:
        warnings.append(f'{name}: {rejected} 条无效采样未用于曲线/统计')
    if any(a['t_ns'] >= b['t_ns'] for a, b in zip(valid, valid[1:])):
        warnings.append(f'{name}: 时间戳重复或乱序，已排序并去重')
    return sorted({r['t_ns']: r for r in valid}.values(), key=lambda r: r['t_ns'])


def channel(rows, field, index=None):
    values = []
    for row in rows:
        value = row['data'].get(field)
        if index is not None:
            value = value[index] if isinstance(value, (list, tuple)) and len(value) > index else None
        value = number(value)
        values.append(np.nan if value is None else value)
    return np.array(values, dtype=float)


def stats(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'valid_count': int(len(finite)), 'missing_count': int(len(values)-len(finite)),
            **({k: float(fn(finite)) for k, fn in (
                ('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std))} if len(finite) else {})}


def analyze(session, output, *, start=None, end=None, surface_pressure_pa=None, rho=None, gravity=None):
    warnings = []
    meta = load_metadata(session)
    raw = {name: samples(read_jsonl(session / f'raw_{name}.jsonl', warnings), name, warnings)
           for name in ('depth', 'imu', 'power')}
    if not any(raw.values()):
        raise ValueError('没有有效的原始传感器采样')
    commands = read_jsonl(session / 'commands.jsonl', warnings)
    events = read_jsonl(session / 'events.jsonl', warnings)
    command_times = [r['t_ns'] for r in commands if number(r.get('t_ns')) is not None]
    # 所有曲线共享单调时钟，不把各传感器各自的首点都当成零点。
    origin = min(command_times) if command_times else min(r[0]['t_ns'] for r in raw.values() if r)
    origin_note = '首个动作提交' if command_times else '首个有效传感器采样'
    ends = [r['t_ns'] for r in events if r.get('event') in ('action_completed', 'fin_action_completed')
            and number(r.get('t_ns')) is not None]
    control_end = (max(ends)-origin)/1e9 if ends else None
    for name, rows in raw.items():
        raw[name] = [r for r in rows if (start is None or (r['t_ns']-origin)/1e9 >= start)
                     and (end is None or (r['t_ns']-origin)/1e9 <= end)]
    if not any(raw.values()):
        raise ValueError('指定时间范围内没有采样')

    calibration = {}
    for filename in ('calibration.json', 'calibration_20260914.json'):
        path = session / filename
        if path.exists():
            calibration = json.loads(path.read_text(encoding='utf-8-sig'))
            break
    reference = surface_pressure_pa if surface_pressure_pa is not None else number(calibration.get('p_surface_pa'))
    empirical = number(calibration.get('pressure_per_meter'))
    if calibration.get('depth_conversion') == 'linear_pressure' and (empirical is None or empirical <= 0):
        raise ValueError('经验深度标定缺少有效 pressure_per_meter')
    density = rho if rho is not None else number(calibration.get('water_density_kg_m3'))
    g = gravity if gravity is not None else number(calibration.get('gravity_mps2'))
    # 无标定文件时不读取当前 robot.yaml 猜零点，避免把旧实验按新配置重算。
    if reference is not None and empirical is None and (density is None or g is None):
        density = 1000. if density is None else density
        g = 9.80665 if g is None else g
        warnings.append('缺少密度/重力标定字段，深度换算采用默认淡水1000 kg/m³、g=9.80665 m/s²（未提供项）')
    if reference is not None and any(v <= 0 for v in (reference, density, g) if v is not None):
        raise ValueError('标定中的参考压力、密度和重力必须大于0')
    target = number(meta.get('h0'))
    if target is None:
        config = meta.get('runtime_config', meta.get('rl_config', {}))
        target = number(config.get('targets', {}).get('depth_m'))

    output.mkdir(parents=True, exist_ok=True)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei') if f in installed]
    plt.rcParams['font.sans-serif'] = fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    times = {name: np.array([(r['t_ns']-origin)/1e9 for r in rows]) for name, rows in raw.items()}
    images = []
    metrics = {}

    def figure(filename, title, sensor, panels):
        """保留原始波形，不平滑；明显采样间断处断开连线。"""
        t = times[sensor]
        if not len(t):
            return
        if not any(np.isfinite(v).any() for _, curves in panels for _, v in curves):
            warnings.append(f'{title}: 没有可绘制字段')
            return
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3*len(panels)+.5),
                                 sharex=True, squeeze=False, layout='constrained')
        gaps = np.where(np.diff(t) > max(.5, 5*np.median(np.diff(t))))[0]+1 if len(t)>1 else []
        for ax, (unit, curves) in zip(axes[:, 0], panels):
            for label, values in curves:
                metrics[f'{sensor}.{label}'] = stats(values)
                if np.isfinite(values).any():
                    ax.plot(np.insert(t, gaps, np.nan), np.insert(values, gaps, np.nan), lw=.8, label=label)
            if sensor == 'depth' and unit == '压力 / kPa' and reference is not None:
                ax.axhline(reference/1000, color='#b84b35', ls='--', label=f'水面参考 {reference/1000:g} kPa')
            if sensor == 'depth' and unit == '深度 / m' and target is not None:
                ax.axhline(target, color='#b84b35', ls='--', label=f'目标 {target:g} m')
            if command_times and control_end is not None:
                left, right = max(0., float(t.min())), min(control_end, float(t.max()))
                if left < right:
                    ax.axvspan(left, right, color='#60b894', alpha=.09)
            ax.set_ylabel(unit)
            ax.grid(alpha=.22)
            ax.ticklabel_format(axis='y', useOffset=False)
            ax.spines[['top', 'right']].set_visible(False)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='best', fontsize=9)
        axes[0, 0].set_title(f'{session.name}\n{title}')
        axes[-1, 0].set_xlabel(f'相对{origin_note}的时间 / s')
        fig.savefig(output/filename, dpi=160)
        plt.close(fig)
        images.append((title, filename))

    pressure = channel(raw['depth'], 'pressure_pa')
    depth = (pressure-reference)/(empirical if empirical is not None else density*g) if reference is not None else channel(raw['depth'], 'depth_m')
    if reference is None:
        warnings.append('缺少可用的水面压力标定：深度图仅使用日志自带 depth_m（若有），不猜测零点')
    figure('pressure_depth.png', '压力、换算深度与温度', 'depth', [
        ('压力 / kPa', [('pressure_kpa', pressure/1000)]),
        ('深度 / m', [('depth_m', depth)]),
        ('温度 / °C', [('temperature_c', channel(raw['depth'], 'temperature_c'))])])
    figure('imu_attitude.png', 'IMU 原始欧拉角（未减标定零点、未展开航向角）', 'imu', [
        ('角度 / °', [(key, channel(raw['imu'], key))]) for key in ('roll_deg', 'pitch_deg', 'yaw_deg')])
    figure('imu_motion.png', 'IMU 加速度与角速度（加速度包含重力）', 'imu', [
        ('加速度 / m/s²', [(f'acc_{axis}', channel(raw['imu'], 'acc_mps2', i)) for i, axis in enumerate('xyz')]),
        ('角速度 / rad/s', [(f'gyro_{axis}', channel(raw['imu'], 'gyro_radps', i)) for i, axis in enumerate('xyz')])])
    figure('power.png', '电压、电流与传感器报告功率', 'power', [
        (unit, [(key, channel(raw['power'], key))]) for unit, key in
        [('电压 / V', 'voltage_v'), ('电流 / A', 'current_a'), ('功率 / W', 'power_w')]])
    current = channel(raw['power'], 'current_a')
    if np.isfinite(current).any() and np.all(current[np.isfinite(current)] == 0):
        warnings.append('本时间范围电流全部为0；请核对采集，不应直接视作无负载')
    summary = {'session': str(session.resolve()), 'time_origin': origin_note, 'origin_t_ns': origin,
               'range_s': {'start': start, 'end': end}, 'control_end_s': control_end,
               'parameters': {k: meta.get(k) for k in ('dry_run', 'h0', 'k', 'action_duration_s', 'max_control_s')},
               'depth_conversion': {'surface_pressure_pa': reference, 'density_kg_m3': density,
                                    'pressure_per_meter': empirical,
                                    'gravity_mps2': g, 'reference_overridden': surface_pressure_pa is not None},
               'sampling': {}, 'metrics': metrics, 'warnings': warnings}
    for name, t in times.items():
        summary['sampling'][name] = {'count': len(t), 'max_gap_s': float(np.diff(t).max()) if len(t)>1 else None,
                                    'mean_rate_hz': float((len(t)-1)/(t[-1]-t[0])) if len(t)>1 else None}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    esc = html.escape
    report = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>机器鱼运行分析</title>',
              '<style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#243447}img{width:100%}pre{white-space:pre-wrap;background:#f3f6f8;padding:16px}a{color:#176baf}</style>',
              f'<h1>{esc(session.name)}</h1>',
              '<p>各图使用同一时间原点；负时间为首次动作前。绿色背景表示首个提交至最后完成事件的区间，包含期间的保持。曲线未平滑。缺失值和明显采样间断不连线。</p>',
              '<p>IMU为原始姿态；加速度包含重力。功率直接来自传感器日志，未用电压×电流替代。PWM指令完成不等于舵机实际位置反馈。</p>',
              '<h2>参数与深度依据</h2><pre>'+esc(json.dumps({'parameters':summary['parameters'], 'depth_conversion':summary['depth_conversion']},ensure_ascii=False,indent=2))+'</pre>']
    if warnings:
        report.append('<h2>数据提示</h2><ul>'+''.join(f'<li>{esc(w)}</li>' for w in warnings)+'</ul>')
    for title, filename in images:
        report.append(f'<h2>{esc(title)}</h2><a href="{filename}"><img src="{filename}" alt="{esc(title)}"></a>')
    report.extend(['<h2>采样与统计摘要</h2><p>统计仅针对所选时间范围；航向角均值未作圆统计，不应直接解释为平均航向。</p>',
                   '<pre>'+esc(json.dumps({'sampling':summary['sampling'],'metrics':metrics},ensure_ascii=False,indent=2))+'</pre></html>'])
    (output/'report.html').write_text('\n'.join(report), encoding='utf-8')
    return output/'report.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session', nargs='?', type=Path, help='包含 raw_*.jsonl 的具体运行目录')
    parser.add_argument('--latest', action='store_true', help='递归选择 logs 下最新的非模拟运行')
    parser.add_argument('--logs', type=Path, default=ROOT/'logs')
    parser.add_argument('--include-simulation', action='store_true', help='--latest 也允许选择 dry_run')
    parser.add_argument('--output', type=Path, help='默认输出至运行目录/analysis_20260921')
    parser.add_argument('--start-s', type=float, help='分析起始时间；0表示首个动作提交（无动作时为首个采样）')
    parser.add_argument('--end-s', type=float, help='分析结束时间，使用同一原点')
    parser.add_argument('--surface-pressure-pa', type=float, help='仅覆盖离线深度换算的参考压力（Pa），不修改日志')
    parser.add_argument('--rho', type=float, help='离线换算水密度 kg/m³')
    parser.add_argument('--gravity', type=float, help='离线换算重力加速度 m/s²')
    parser.add_argument('--show', action='store_true', help='生成后用默认浏览器打开报告')
    args = parser.parse_args()
    if bool(args.session) == args.latest:
        parser.error('请指定一个运行目录，或使用 --latest，二者选其一')
    for key in ('start_s', 'end_s', 'surface_pressure_pa', 'rho', 'gravity'):
        val = getattr(args, key)
        if val is not None and (not math.isfinite(val) or (key in ('surface_pressure_pa','rho','gravity') and val<=0)):
            parser.error(f'{key} 数值无效')
    if args.start_s is not None and args.end_s is not None and args.end_s <= args.start_s:
        parser.error('--end-s 必须大于 --start-s')
    try:
        session = args.session or find_latest(args.logs, args.include_simulation)
        report = analyze(session, args.output or session/'analysis_20260921', start=args.start_s, end=args.end_s,
                         surface_pressure_pa=args.surface_pressure_pa, rho=args.rho, gravity=args.gravity)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f'分析失败：{exc}\n')
    print(f'运行目录：{session.resolve()}\n报告：{report.resolve()}')
    if args.show:
        webbrowser.open(report.resolve().as_uri())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
