"""UWB独立采集和离线绘图；不创建IMU、舵机或PCA9685设备。"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path

from .surface_navigation_20261010 import UWBStreamDecoder
from .uart_workers import read_uwb_chunk


def capture(port, duration_s, directory, *, clock=time.monotonic, clock_ns=time.monotonic_ns):
    """串口由调用者管理；空读也检查截止时间，停止后保留所有已收到的字节。"""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('t必须为正有限秒数')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = clock()
    deadline = start+duration_s
    reason, error, seq, total_bytes = 'time_limit', None, 0, 0
    print_due = start
    with (directory/'raw_uwb.jsonl').open('x', encoding='utf-8', buffering=1) as f:
        try:
            while clock() < deadline:
                # 上限0.2秒，最后一次读取按剩余时长缩短，不能因无信号永久阻塞。
                port.timeout = min(.2, max(0., deadline-clock()))
                chunk = read_uwb_chunk(port)
                seq += 1
                total_bytes += len(chunk)
                row = dict(name='uwb', t_ns=clock_ns(), seq=seq,
                           elapsed_s=clock()-start,
                           data=dict(raw_hex=chunk.hex(), raw_text=chunk.decode('utf-8', errors='replace').strip(), parsed=False),
                           ok=bool(chunk), error=None if chunk else 'no_signal_or_timeout')
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                if clock() >= print_due:
                    print(f'UWB采集 {clock()-start:.1f}/{duration_s:g}s，已收 {total_bytes} 字节',flush=True)
                    print_due = clock()+1.
        except KeyboardInterrupt:
            reason = 'interrupted'
        except Exception as exc:
            reason, error = 'error', f'{type(exc).__name__}: {exc}'
            f.write(json.dumps(dict(name='uwb',t_ns=clock_ns(),seq=seq+1,elapsed_s=clock()-start,
                                   data={},ok=False,error=error),ensure_ascii=False)+'\n')
    return dict(reason=reason,error=error,requested_duration_s=duration_s,
                elapsed_s=clock()-start,received_bytes=total_bytes)


def export_ranges(raw_path, directory, tag='t0:0'):
    """从原始字节拼mc帧；不套用定位/H/残差筛选，以便观察异常测距。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    decoder = UWBStreamDecoder(tag)
    points=[]
    first_t=None
    previous_seq=None
    failures=0
    gaps=0
    last_time=0.
    chunks=0
    for line in Path(raw_path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        chunks+=1
        if first_t is None:
            first_t=row['t_ns']
        elapsed=float(row.get('elapsed_s',(row['t_ns']-first_t)/1e9))
        last_time=elapsed
        if previous_seq is not None and row['seq'] != previous_seq+1:
            decoder.reset_partial()
            gaps+=1
            points.append([elapsed,*([math.nan]*4)])
        previous_seq=row['seq']
        if not row['ok']:
            failures+=1
            # 普通串口空读不一定意味着半帧损坏，保留片段等待后续；其他错误丢弃片段。
            if row.get('error') != 'no_signal_or_timeout':
                decoder.reset_partial()
            points.append([elapsed,*([math.nan]*4)])
            continue
        invalid_before=decoder.invalid_frames
        try:
            frames=decoder.feed(bytes.fromhex(row['data']['raw_hex']))
        except (KeyError,ValueError,TypeError):
            decoder.reset_partial()
            failures+=1
            points.append([elapsed,*([math.nan]*4)])
            continue
        if decoder.invalid_frames > invalid_before:
            points.append([elapsed,*([math.nan]*4)])
        points.extend([elapsed,*frame.ranges_m] for frame in frames)
    valid=[p for p in points if all(math.isfinite(v) for v in p)]
    with (directory/'uwb_ranges.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['elapsed_s','A0_m','A1_m','A2_m','A3_m'])
        writer.writerows([[v if math.isfinite(v) else '' for v in p] for p in points])
    summary=dict(tag=tag,raw_chunks=chunks,valid_frames=len(valid),invalid_mc_frames=decoder.invalid_frames,
                 empty_or_failed_reads=failures,sequence_gaps=gaps,last_sample_elapsed_s=last_time,
                 ranges=[dict(anchor=f'A{k}',min_m=min(p[k+1] for p in valid),
                              median_m=statistics.median(p[k+1] for p in valid),max_m=max(p[k+1] for p in valid))
                         for k in range(4)] if valid else [])
    return points,summary


def plot_ranges(points, directory, *, show=False):
    """默认使用无桌面后端保存图片，--show才尝试打开交互窗口。"""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
    colors=['#1879ad','#d97816','#288748','#bd4545']
    for k,ax in enumerate(axes.flat):
        if points:
            ax.plot([p[0] for p in points],[p[k+1] for p in points],color=colors[k],lw=1,marker='.',ms=2.5)
        if not any(math.isfinite(p[k+1]) for p in points):
            ax.text(.5,.5,'No valid UWB ranges',transform=ax.transAxes,ha='center')
        ax.set(title=f'A{k}',xlabel='Elapsed time (s)',ylabel='Slant range (m)')
        ax.grid(alpha=.25)
    fig.suptitle('UWB raw ranges - no smoothing; independent Y scales')
    directory=Path(directory)
    fig.savefig(directory/'uwb_ranges.png',dpi=180)
    fig.savefig(directory/'uwb_ranges.pdf')
    if show:
        plt.show()
    plt.close(fig)
