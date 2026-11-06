#!/usr/bin/env python3
"""Replay one static capture with one causal update per t4, never continuous ZUPT."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_imu_static_20261031 import load,dump
from control.runtime.imu_periodic_static_20261031 import PeriodicStaticExperiment


def replay(d,noise,t4,settings):
    sim=PeriodicStaticExperiment(noise,t4=t4,externally_confirmed_static=True,**settings)
    events=[];trace=[];cycles=[];origins={};peaks={};all_peaks={}
    for i,t in enumerate(d['t']):
        event=sim.step(t,dict(acc_mps2=d['acc'][i],gyro_radps=d['gyro'][i],quat=d['quat'][i]))
        if not sim.models:continue
        if not origins:
            origins={m:model.p.copy() for m,model in sim.models.items()}
            peaks={m:[0.,0.] for m in sim.models};all_peaks={m:[0.,0.] for m in sim.models}
        # Evaluate BEFORE any boundary correction, so resetting cannot hide peaks.
        states=event['before'] if event and event['event']=='periodic_boundary' else sim.snapshot()
        for m,s in states.items():
            p=np.array(s['p_m']);v=np.array(s['v_mps'])
            peaks[m][0]=max(peaks[m][0],float(np.linalg.norm((p-origins[m])[:2])))
            peaks[m][1]=max(peaks[m][1],float(np.linalg.norm(v[:2])))
            all_peaks[m][0]=max(all_peaks[m][0],float(np.linalg.norm(p[:2])))
            all_peaks[m][1]=max(all_peaks[m][1],float(np.linalg.norm(v[:2])))
        if event:
            events.append(event)
            if event['event']=='periodic_boundary':
                for m,before in event['before'].items():
                    p=np.array(before['p_m']);v=np.array(before['v_mps']);after=event['after'][m]
                    cycles.append(dict(t4_s=t4,method=m,cycle=event['cycle'],time_s=event['time_s'],accepted=event['accepted'],
                        method_updated=bool(event['accepted'] and m!='free_vendor'),
                        pre_speed_xy_mps=float(np.linalg.norm(v[:2])),pre_position_xy_m=float(np.linalg.norm(p[:2])),
                        pre_segment_displacement_xy_m=float(np.linalg.norm((p-origins[m])[:2])),
                        segment_peak_displacement_xy_m=peaks[m][0],segment_peak_speed_xy_mps=peaks[m][1],
                        post_position_xy_m=float(np.linalg.norm(after['p_m'][:2])),
                        post_speed_xy_mps=float(np.linalg.norm(after['v_mps'][:2]))))
                    origins[m]=np.array(after['p_m']);peaks[m]=[0.,0.]
        if i%20==0 or event:
            trace.append(dict(time_s=float(t-sim.origin),models=sim.snapshot()))
    if not cycles:raise ValueError('Capture has no complete t4 segment')
    summaries={}
    for m,model in sim.models.items():
        group=[x for x in cycles if x['method']==m]
        values={key:np.array([r[key] for r in group]) for key in
            ('pre_speed_xy_mps','pre_segment_displacement_xy_m','pre_position_xy_m','segment_peak_displacement_xy_m')}
        summaries[m]=dict(complete_cycles=len(group),accepted_boundaries=sim.corrections,
            actual_updates=0 if m=='free_vendor' else sim.corrections,
            end_speed_xy_mps=float(np.linalg.norm(model.v[:2])),end_position_xy_m=float(np.linalg.norm(model.p[:2])),
            max_position_xy_m=all_peaks[m][0],max_speed_xy_mps=all_peaks[m][1],
            **{key:dict(mean=float(x.mean()),rms=float(np.sqrt(np.mean(x*x))),p95=float(np.percentile(x,95)),max=float(x.max()))
               for key,x in values.items()})
    return dict(t4_s=t4,initialization=sim.initialization,boundaries=sim.boundary-1,
        accepted=sim.corrections,eskf_update_count=sim.models['periodic_eskf_zupt'].update_count,
        prediction_duration_s=float(d['t'][-1]-sim.origin),summaries=summaries),events,trace,cycles


def check_live(folder,t4,events):
    path=folder/'periodic_live.jsonl'
    if not path.exists():return None
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['periodic']['t4']!=t4:return None
    live=[json.loads(x) for x in path.read_text().splitlines()]
    live=[r for r in live if r['event']=='periodic_boundary']
    replayed=[r for r in events if r['event']=='periodic_boundary']
    if len(live)!=len(replayed):raise ValueError('Live/replay boundary count differs')
    difference=0.
    for a,b in zip(live,replayed):
        if a['time_s']!=b['time_s'] or a['accepted']!=b['accepted']:raise ValueError('Live schedule differs')
        for side in ('before','after'):
            for m in a[side]:
                for key in ('p_m','v_mps','bias_body_mps2'):
                    difference=max(difference,float(np.max(np.abs(np.array(a[side][m][key])-b[side][m][key]))))
    if difference>1e-7:raise ValueError(f'Live/replay mismatch {difference}')
    return dict(boundaries=len(live),max_absolute_state_difference=difference,
                times_s=[r['time_s'] for r in live],source='Pi online periodic_live.jsonl vs independent offline replay')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--params',type=Path,default=ROOT/'config/imu_static_candidate_20261031.json')
    p.add_argument('--t4',type=float,nargs='+',default=[5.,10.,20.,30.])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    d=load(args.session);params=json.loads(args.params.read_text())
    settings=dict(window_s=2.,warmup_s=5.,startup_s=5.,tau=.2)
    meta=json.loads((args.session/'metadata.json').read_text())
    if 'periodic' in meta:
        settings={key:meta['periodic'][key] for key in settings}
        if params['noise']!=meta['periodic']['noise']:raise ValueError('Live noise parameters differ')
    report=dict(session=str(args.session),health=d['health'],settings=settings,
        methodology='One update at t4,2*t4,...; no stillness information inside segments; causal last 2s window only at boundary; fresh startup bias from 5..10s',
        origin_information='Only periodic_surface_anchor uses known absolute position zero (ideal simulated UWB). Other methods do not force position to zero.',
        no_actuation=True,interpretation='Stationary bench timing test, not physical diving/surfacing and not measured swimming accuracy.',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [args.session/'raw_imu.jsonl',Path(__file__),ROOT/'control/runtime/imu_periodic_static_20261031.py',args.params]},results={})
    all_cycles=[];all_trace={};live_checks=[]
    for t4 in args.t4:
        summary,events,trace,cycles=replay(d,params['noise'],t4,settings)
        key=f'{t4:g}';report['results'][key]=summary;all_trace[key]=trace;all_cycles+=cycles
        dump(args.output/f'events_t4_{key}.json',events)
        check=check_live(args.session,t4,events)
        if check:live_checks.append(check)
        print(json.dumps(dict(t4=t4,boundaries=summary['boundaries'],accepted=summary['accepted'],live_check=check)),flush=True)
    report['live_checks']=live_checks
    dump(args.output/'report.json',report)
    with (args.output/'cycles.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_cycles[0]));writer.writeheader();writer.writerows(all_cycles)
    selected='10' if '10' in all_trace else next(iter(all_trace))
    dump(args.output/f'trace_t4_{selected}.json',all_trace[selected])
    plot(args.output,report,all_cycles,all_trace[selected],selected)
    lines=['Intermittent static correction: PRE-CORRECTION drift, never continuous ZUPT',
           str(args.session),json.dumps(settings),'Position anchor is an extra ideal known-position observation.',
           't4 | method | cycles actual_updates | mean / p95 / max segment displacement (m) | p95 speed (m/s) | max accumulated p (m)']
    for key,result in report['results'].items():
        for m,s in result['summaries'].items():
            a=s['pre_segment_displacement_xy_m'];v=s['pre_speed_xy_mps']
            lines.append(f'{key} | {m} | {result["boundaries"]} {s["actual_updates"]} | {a["mean"]:.6f} / {a["p95"]:.6f} / {a["max"]:.6f} | {v["p95"]:.6f} | {s["max_position_xy_m"]:.6f}')
    lines+=['','Live verification: '+json.dumps(live_checks)]
    (args.output/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (args.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())


def plot(output,report,cycles,trace,key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=PeriodicStaticExperiment.METHODS
    labels=['No periodic correction','Velocity reset only','Velocity + local bias','Ideal surface position fix','ESKF: one ZUPT per boundary']
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    t=np.array([x['time_s'] for x in trace])
    for i,(m,label) in enumerate(zip(names,labels)):
        color=plt.get_cmap('tab10')(i)
        p=np.array([x['models'][m]['p_m'] for x in trace])
        axs[0,0].plot(t,np.linalg.norm(p[:,:2],axis=1),color=color,label=label,lw=1)
        group=[r for r in cycles if r['t4_s']==float(key) and r['method']==m]
        axs[0,1].plot([r['cycle'] for r in group],[r['pre_speed_xy_mps'] for r in group],'.-',color=color,lw=1)
        axs[1,0].plot([r['cycle'] for r in group],[r['pre_segment_displacement_xy_m'] for r in group],'.-',color=color,lw=1)
        intervals=sorted(float(k) for k in report['results'])
        vals=[report['results'][f'{period:g}']['summaries'][m]['pre_segment_displacement_xy_m']['p95'] for period in intervals]
        axs[1,1].plot(intervals,vals,'o-',color=color,lw=1)
    axs[0,0].set(xlabel='Elapsed prediction time (s)',ylabel='Accumulated horizontal displacement (m)',title=f't4 = {key} s; post-update state trace')
    axs[0,1].set(xlabel='Cycle',ylabel='Horizontal speed before correction (m/s)',title='No updates inside each cycle')
    axs[1,0].set(xlabel='Cycle',ylabel='Displacement within segment (m)',title='Value BEFORE boundary correction')
    axs[1,1].set(xlabel='t4 (s)',ylabel='P95 segment displacement before correction (m)',title='Correction interval comparison')
    for ax in axs.flat:ax.grid(alpha=.25)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Stationary bench: periodic correction only; vendor attitude / 0.2s LPF vs gyro-propagated ESKF')
    fig.savefig(output/'periodic_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
