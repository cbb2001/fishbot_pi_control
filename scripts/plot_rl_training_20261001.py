"""离线绘制PPO训练总览；兼容旧日志，不启动硬件、不修改原始数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from _bootstrap import add_project_root
ROOT = add_project_root()


def read_jsonl(path):
    """容忍正在写入文件的最后一条不完整记录，其他格式错误明确报出。"""
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise ValueError(f'{path}:{i+1}: JSON格式错误')
            warnings.warn(f'跳过尚未写完的最后一条记录：{path}')
    return rows


def plot_session(session, output=None):
    session = Path(session).resolve()
    states = read_jsonl(session/'rl_state_20260914.jsonl')
    commands = read_jsonl(session/'commands.jsonl')
    updates = read_jsonl(session/'fin_ppo_updates_20260914.jsonl')
    episodes = read_jsonl(session/'episodes_20260914.jsonl')
    meta_path = session/'metadata.yaml'
    meta = yaml.safe_load(meta_path.read_text(encoding='utf-8-sig')) if meta_path.exists() else {}
    cfg = (meta or {}).get('rl_config', {})
    if not states:
        raise ValueError('没有RL状态数据，请检查训练是否已开始以及日志目录是否正确')
    # 横轴从首个动作开始；只展示实际记录的控制深度，不对旧日志擅自应用新标定。
    origin = commands[0]['t_ns'] if commands else states[0]['t_ns']
    states = [r for r in states if r['t_ns'] >= origin]
    if not states:
        raise ValueError('首个动作后暂无状态记录')
    seconds = lambda rows: [(r['t_ns']-origin)/1e9 for r in rows]
    t = seconds(states)
    targets = cfg.get('targets', {})
    training = cfg.get('training', {})
    weights = cfg.get('reward', {})
    corrected = any('pressure_drift_pa' in r for r in states)
    fig, axs = plt.subplots(7, 1, figsize=(13, 19), layout='constrained')
    axs[0].plot(t, [r['depth_m'] for r in states], lw=.8,
                label='Drift-corrected control depth' if corrected else 'Recorded control depth (legacy)')
    if 'depth_m' in targets:
        goal = targets['depth_m'];tol = training.get('depth_tolerance_m', 0.)
        axs[0].axhline(goal, color='tab:orange', ls='--', label=f'Target {goal:g} m')
        axs[0].axhspan(goal-tol, goal+tol, color='green', alpha=.08)
    axs[0].axhline(0, color='gray', lw=.6);axs[0].set_ylabel('Depth (m)')
    axs[1].plot(t, [r['yaw_deg'] for r in states], lw=.8, label='Relative yaw')
    heading = targets.get('heading_deg', 0.);tol = training.get('yaw_tolerance_deg', 5.)
    axs[1].axhspan(heading-tol, heading+tol, color='green', alpha=.1, label='Heading tolerance')
    axs[1].set_ylabel('Yaw (deg)')
    for name in ('depth', 'heading', 'pitch', 'roll'):
        rows = [r for r in states if 'r_'+name in r]
        if rows:
            axs[2].plot(seconds(rows), [r['r_'+name]*weights.get(name+'_weight',1.) for r in rows], lw=.8, label=name)
    axs[2].set_ylabel('Weighted reward')
    if corrected:
        axs[3].plot(t, [r.get('raw_pressure_pa', r['pressure_pa'])/1000 for r in states], lw=.7, label='Raw pressure')
        axs[3].plot(t, [r['pressure_pa']/1000 for r in states], lw=.8, label='Corrected P - f(t)')
    else:
        axs[3].plot(t, [r['pressure_pa']/1000 for r in states], lw=.8, label='Pressure (legacy)')
    cal_path = session/'calibration_20260914.json'
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding='utf-8-sig'))
        if cal.get('p_surface_pa') is not None:
            p0 = cal['p_surface_pa']
            axs[3].axhline(p0/1000, color='gray', ls='--', label='P0 / reference')
            if corrected:
                axs[3].plot(t, [(p0+r.get('pressure_drift_pa',0))/1000 for r in states],
                            color='tab:green', ls='--', lw=.9, label='Surface baseline P0 + f(t)')
    axs[3].set_ylabel('Pressure (kPa)')
    for b1,marker in ((0,'o'),(1,'x')):
        rows = [r for r in commands if r.get('action',{}).get('b1')==b1]
        if rows:
            axs[4].scatter(seconds(rows), [r['action']['theta'] for r in rows], s=10, marker=marker, label=f'b1={b1}')
    axs[4].set_ylabel('Action theta (deg)')
    axs[4].set_yticks([-53,0,53])
    for key, label in (('value_loss','Value loss'),('policy_loss','Policy loss')):
        rows = [r for r in updates if key in r]
        if rows:
            axs[5].plot([r.get('published_policy_version',i+1) for i,r in enumerate(rows)], [r[key] for r in rows], marker='.', label=label)
    axs[5].set_ylabel('PPO loss');axs[5].set_xlabel('Published policy version')
    rows = [r for r in updates if 'update_duration_s' in r]
    if rows:
        axs[6].plot([r.get('published_policy_version',i+1) for i,r in enumerate(rows)],
                    [r['update_duration_s']*1000 for r in rows], marker='.', label='PPO compute time')
    axs[6].set_ylabel('Update time (ms)');axs[6].set_xlabel('Published policy version')
    for ax in axs[:5]:
        ax.set_xlim(0,max(t[-1], .1));ax.set_xlabel('Seconds from first action')
        for ep in sorted({r.get('episode_index',0) for r in commands})[1:]:
            first = min(r['t_ns'] for r in commands if r.get('episode_index',0)==ep)
            ax.axvline((first-origin)/1e9, ls=':', lw=.7, color='gray')
    for ax in axs:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', ncol=2)
        elif ax in axs[5:]:
            ax.text(.5,.5,'No PPO updates recorded',ha='center',transform=ax.transAxes)
    mode = 'MOCK / dry-run' if meta.get('dry_run') else 'Real run'
    success = sum(r.get('reason')=='success' for r in episodes)
    fig.suptitle(f'{session.name} | {mode}\n{len(commands)} actions | {len(updates)} PPO updates | {success}/{len(episodes)} episodes successful')
    output = Path(output) if output else session/'analysis/training_overview.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', default='latest', help='会话目录，默认latest')
    parser.add_argument('--log-root', type=Path, default=ROOT/'logs/rl_20260914')
    parser.add_argument('--output', type=Path, help='图片输出路径，默认会话内analysis/training_overview.png')
    args = parser.parse_args()
    try:
        session = Path(args.session)
        if args.session == 'latest':
            candidates = []
            for path in args.log_root.glob('*/metadata.yaml'):
                meta = yaml.safe_load(path.read_text(encoding='utf-8-sig')) or {}
                if 'rl_config' in meta and not meta.get('dry_run',False) and (path.parent/'rl_state_20260914.jsonl').exists():
                    candidates.append(path.parent)
            if not candidates:
                raise ValueError('没有实物PPO会话；模拟日志请显式指定目录')
            session = max(candidates, key=lambda p:p.name)
        print(plot_session(session, args.output))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'绘图失败：{exc}\n')


if __name__ == '__main__':
    main()
