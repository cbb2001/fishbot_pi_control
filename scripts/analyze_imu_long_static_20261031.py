#!/usr/bin/env python3
"""Streaming long-run IMU QA and chronological calibration assessment.

Reads raw JSONL one row at a time into compact arrays, avoiding a giant list of
Python dictionaries. Periodic results come from the actual online boundary log.
No replay can add extra rest updates to the measured online run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def read_raw(session):
    result=json.loads((session/'result.json').read_text())
    metadata=json.loads((session/'metadata.json').read_text())
    if result['reason']!='time_limit' or result['fault'] or result['executor_started']:
        raise ValueError('Need a clean completed no-actuation capture')
    n=result['sample_count'];hz=float(metadata['imu_config']['sample_rate_hz'])
    acc=np.empty((n,3));gyro=np.empty((n,3));quat=np.empty((n,4));euler=np.empty((n,3))
    stamps=np.empty(n,dtype=np.int64);last_tid=None;first_receive=last_receive=None
    missing=0;temperature_frames=0;digest=hashlib.sha256();count=0
    modulus=metadata['imu_config'].get('tid_modulus',65536)
    with (session/'raw_imu.jsonl').open('rb') as f:
        for i,line in enumerate(f):
            digest.update(line);r=json.loads(line);d=r['data']
            if i>=n or not r['ok'] or r['seq']!=i+1:raise ValueError('Frame count/sequence mismatch')
            tid=d['tid']
            if last_tid is not None:
                step=(tid-last_tid)%modulus
                if step!=1:raise ValueError(f'Non-contiguous TID at row {i+1}: {last_tid}->{tid}')
            last_tid=tid
            acc[i]=d['acc_mps2'];gyro[i]=d['gyro_radps'];quat[i]=d['quat']
            euler[i]=[d[k+'_deg'] for k in ('roll','pitch','yaw')]
            stamps[i]=r['t_ns'];last_receive=r['receive_t_ns']
            if first_receive is None:first_receive=last_receive
            raw=bytes.fromhex(d['raw_hex']);at=5;fields=set()
            while at<len(raw)-2:fields.add(raw[at]);at+=2+raw[at+1]
            if not {0x10,0x20,0x40,0x41}<=fields:raise ValueError('Missing required IMU field')
            temperature_frames+=0x01 in fields
            count=i+1
    if count!=n:raise ValueError('Incomplete raw file')
    norms=np.linalg.norm(quat,axis=1)
    if not np.isfinite(acc).all() or not np.isfinite(gyro).all() or not np.isfinite(quat).all():
        raise ValueError('Nonfinite raw IMU values')
    if np.any((norms<.8)|(norms>1.2)):raise ValueError('Quaternion norm outside range')
    quat/=norms[:,None];w,x,y,z=quat.T
    R=np.empty((n,3,3))
    R[:,0,:]=np.column_stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)))
    R[:,1,:]=np.column_stack((2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)))
    R[:,2,:]=np.column_stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))
    t=np.arange(n)/hz;receipt_span=(last_receive-first_receive)/1e9
    if abs(t[-1]-receipt_span)>max(.1,t[-1]*.001):raise ValueError('Sample clock disagrees with host clock')
    dt=np.diff(stamps)/1e9
    if np.min(dt)<=0:raise ValueError('Backward reconstructed timestamp')
    health=dict(samples=n,frame_duration_s=float(t[-1]),wall_duration_s=result['elapsed_s'],
        rate_hz=hz,host_observed_rate_hz=(n-1)/receipt_span,raw_sha256=digest.hexdigest(),
        raw_bytes=(session/'raw_imu.jsonl').stat().st_size,tid_missing=missing,
        crc_errors=d['crc_error_count'],resync_count=d['decode_resync_count'],
        temperature_frames=temperature_frames,temperature_calibration_available=bool(temperature_frames),
        original_dt_range_s=[float(dt.min()),float(dt.max())],
        executor_started=False,device_opens=result['device_opens'],periodic_result=result.get('periodic'))
    residual=acc-R[:,2,:]*9.80665
    return dict(t=t,acc=acc,gyro=gyro,quat=quat,R=R,euler=euler,residual=residual,health=health)


def periodic_summary(session):
    events=[];trace=[];methods=None;origins={};maximum={}
    with (session/'periodic_live.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['event']=='initialized':initialization=r
            elif r['event']=='state':
                if methods is None:
                    methods=list(r['models']);origins={m:np.zeros(3) for m in methods}
                    maximum={m:0. for m in methods}
                trace.append(r)
                for m,s in r['models'].items():maximum[m]=max(maximum[m],float(np.linalg.norm(s['p_m'][:2])))
            elif r['event']=='periodic_boundary':
                if methods is None:raise ValueError('Boundary before initial state')
                for m,s in r['before'].items():
                    pos=np.array(s['p_m']);maximum[m]=max(maximum[m],float(np.linalg.norm(pos[:2])))
                    events.append(dict(method=m,cycle=r['cycle'],time_s=r['time_s'],accepted=r['accepted'],
                        updated=r['accepted'] and m!='free_vendor',
                        pre_speed_xy_mps=float(np.linalg.norm(s['v_mps'][:2])),
                        pre_position_xy_m=float(np.linalg.norm(pos[:2])),
                        segment_displacement_xy_m=float(np.linalg.norm((pos-origins[m])[:2])),
                        bias_body_mps2=s['bias_body_mps2'],local_bias_mps2=r['local_effective_bias_mps2']))
                    origins[m]=np.array(r['after'][m]['p_m'])
    if not events:raise ValueError('No online periodic boundaries')
    summary={}
    for m in methods:
        rows=[r for r in events if r['method']==m]
        summary[m]=dict(boundaries=len(rows),actual_updates=sum(r['updated'] for r in rows),
            max_observed_position_xy_m=maximum[m],
            max_observed_note='10Hz online states plus every exact pre-update boundary, not a 200Hz peak bound')
        for key in ('pre_speed_xy_mps','pre_position_xy_m','segment_displacement_xy_m'):
            x=np.array([r[key] for r in rows])
            summary[m][key]=dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
    final=json.loads((session/'result.json').read_text())['periodic']
    if summary['periodic_eskf_zupt']['actual_updates']!=final['eskf_updates']:
        raise ValueError('Online boundary log/update count mismatch')
    if any(abs(r['time_s']-r['cycle']*10)>1e-7 for r in events):
        raise ValueError('Unexpected boundary time')
    return summary,events,trace,initialization


def integrate(t,a):
    dt=np.diff(t)[:,None]
    v=np.vstack((np.zeros(3),np.cumsum(.5*(a[:-1]+a[1:])*dt,axis=0)))
    p=np.vstack((np.zeros(3),np.cumsum(.5*(v[:-1]+v[1:])*dt,axis=0)))
    return v,p


def frozen_metrics(d,bias,start,end,window=10.):
    t=d['t'];mask=(t>=start)&(t<=end)
    ti=t[mask];a=np.einsum('nij,nj->ni',d['R'][mask],d['acc'][mask]-bias)-[0,0,9.80665]
    v,p=integrate(ti,a);windows=[]
    for s in np.arange(start,end-window+.001,window):
        ix=(ti>=s)&(ti<=s+window)
        if ix.sum()<2 or ti[ix][-1]-ti[ix][0]<window-.01:continue
        vv,pp=integrate(ti[ix],a[ix])
        windows.append(dict(start_s=float(s),speed_xy_mps=float(np.linalg.norm(vv[-1,:2])),
                            displacement_xy_m=float(np.linalg.norm(pp[-1,:2]))))
    values=np.array([w['displacement_xy_m'] for w in windows])
    return dict(start_s=start,end_s=end,window_count=len(windows),
        ten_second_displacement_rms_m=float(np.sqrt(np.mean(values**2))),
        ten_second_displacement_p95_m=float(np.percentile(values,95)),
        whole_speed_xy_mps=float(np.linalg.norm(v[-1,:2])),whole_displacement_xy_m=float(np.linalg.norm(p[-1,:2])),
        whole_velocity_xyz_mps=v[-1].tolist(),whole_position_xyz_m=p[-1].tolist(),windows=windows)


def assess_frozen(d):
    t=d['t'];r=d['residual'];train=(t>=60)&(t<1200)
    hz=int(d['health']['rate_hz']);blocks=r[train][:sum(train)//hz*hz].reshape(-1,hz,3).mean(1)
    candidates=dict(no_bias=np.zeros(3),startup5_mean=r[(t>=5)&(t<10)].mean(0),
        first19min_mean=r[train].mean(0),first19min_robust=np.median(blocks,axis=0),
        last120s_of_train=r[(t>=1080)&(t<1200)].mean(0))
    validation={m:frozen_metrics(d,b,1200,1800) for m,b in candidates.items()}
    chosen=min(validation,key=lambda m:validation[m]['ten_second_displacement_rms_m'])
    test={m:frozen_metrics(d,b,1800,min(2400.,float(t[-1]))) for m,b in candidates.items()}
    selected=dict(selected_method=chosen,quaternion_order='wxyz',rotation='body_to_world',gravity_mps2=9.80665,
        accel_bias_body_mps2=candidates[chosen].tolist(),gyro_bias_body_radps=d['gyro'][train].mean(0).tolist(),
        fit_interval_s=[60,1200],validation_interval_s=[1200,1800],test_interval_s=[1800,2400],
        use_status='experimental_only_not_motion_validated',temperature_model=False,auto_zupt_during_motion=False,
        scope='Effective one-pose bias; do not add to another startup bias; calibration is not intrinsic or across-temperature validated')
    return dict(candidates={k:v.tolist() for k,v in candidates.items()},selected=selected,validation=validation,test=test)


def time_blocks(d,seconds=60):
    rows=[];t=d['t']
    for start in np.arange(0,t[-1],seconds):
        mask=(t>=start)&(t<start+seconds)
        # A few frames beyond 2400s must not become a misleading 41st minute.
        if mask.sum()<.95*seconds*d['health']['rate_hz']:continue
        rows.append(dict(start_s=float(start),end_s=float(min(start+seconds,t[-1])),samples=int(mask.sum()),
            residual_mean_mps2=d['residual'][mask].mean(0).tolist(),
            residual_std_mps2=d['residual'][mask].std(0).tolist(),
            gyro_mean_radps=d['gyro'][mask].mean(0).tolist(),gyro_std_radps=d['gyro'][mask].std(0).tolist(),
            euler_mean_deg=d['euler'][mask].mean(0).tolist()))
    return rows


def save_csv(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot(output,d,trace,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hz=int(d['health']['rate_hz']);n=len(d['t'])//hz
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
    angles=np.rad2deg(np.unwrap(np.deg2rad(d['euler']),axis=0));angles-=angles[0]
    for ax,data,title in zip(axs,[d['residual'],d['gyro'],angles],
                             ['Body gravity residual (m/s2)','Gyroscope (rad/s)','Reported attitude change (deg)']):
        means=data[:n*hz].reshape(n,hz,3).mean(1)
        for k in range(3):ax.plot((np.arange(n)+.5)/60,means[:,k],label=(['Roll','Pitch','Yaw'] if ax is axs[2] else list('XYZ'))[k],lw=.8)
        ax.set_ylabel(title);ax.grid(alpha=.25);ax.legend(ncol=3,loc='upper right')
        ax.axvline(20,color='gray',ls='--',lw=.8);ax.axvline(30,color='gray',ls='--',lw=.8)
    axs[-1].set_xlabel('Capture time (min)');fig.suptitle('40-minute stationary IMU: one-second means; train / validation / test boundaries at 20 / 30 min')
    fig.savefig(output/'sensor_40min.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    methods=list(trace[0]['models']);t=np.array([x['time_s'] for x in trace])/60
    labels=['Startup calibration only','Velocity reset only','Velocity + local bias','Ideal position + velocity fix','Intermittent ESKF ZUPT']
    for i,(m,label) in enumerate(zip(methods,labels)):
        color=plt.get_cmap('tab10')(i);p=np.array([x['models'][m]['p_m'] for x in trace])
        ax=axs[0,0] if i==0 else axs[0,1]
        ax.plot(t[::10],np.linalg.norm(p[::10,:2],axis=1),label=label,color=color,lw=.9)
        rows=[r for r in events if r['method']==m]
        if i:
            for ax,key in [(axs[1,0],'segment_displacement_xy_m'),(axs[1,1],'pre_speed_xy_mps')]:
                ax.plot([r['time_s']/60 for r in rows],[r[key] for r in rows],label=label,color=color,lw=.8)
    axs[0,0].set_title('Accumulated displacement without periodic updates')
    axs[0,1].set_title('Accumulated displacement with t4 = 10 s updates')
    for ax in axs[0,:]:ax.set_ylabel('Horizontal position estimate (m)')
    axs[1,0].set_ylabel('Segment displacement BEFORE correction (m)')
    axs[1,1].set_ylabel('Speed BEFORE correction (m/s)')
    for ax in axs.flat:ax.set_xlabel('Prediction time (min)');ax.grid(alpha=.25)
    handles,labels=axs[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('40-minute bench capture: one correction per boundary; no continuous rest updates')
    fig.savefig(output/'periodic_40min.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    d=read_raw(args.session)
    if d['t'][-1]<2399:raise ValueError('Need the full approximately 40-minute capture')
    print(json.dumps(d['health']),flush=True)
    periodic,events,trace,initialization=periodic_summary(args.session)
    frozen=assess_frozen(d);blocks=time_blocks(d)
    by_five_minutes={}
    for start in range(0,2400,300):
        group=[r for r in events if start<r['time_s']<=start+300]
        by_five_minutes[str(start)]=dict(start_s=start,end_s=start+300,methods={})
        for m in periodic:
            rr=[r for r in group if r['method']==m]
            if not rr:continue
            by_five_minutes[str(start)]['methods'][m]=dict(cycles=len(rr),updates=sum(r['updated'] for r in rr),
                displacement_p95_m=float(np.percentile([r['segment_displacement_xy_m'] for r in rr],95)),
                speed_p95_mps=float(np.percentile([r['pre_speed_xy_mps'] for r in rr],95)))
    report=dict(health=d['health'],initialization=initialization,periodic=periodic,
        periodic_by_5min=by_five_minutes,bias_by_minute=blocks,frozen_calibration=frozen,
        source_session=str(args.session),source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['40 minutes in one fixed pose; no dynamic or multi-pose validation.',
            'Temperature absent in current device packets: no thermal fit.',
            'Periodic metrics use actual online events, not a continuously corrected offline trajectory.',
            'Frozen candidate fits 60..1200s, selects on 1200..1800s, tests on 1800..2400s.',
            'Absolute position reset is a separately labelled ideal simulated position observation.',
            'Effective bias includes gravity/attitude error; not intrinsic accelerometer calibration.'])
    dump(args.output/'report.json',report);dump(args.output/'calibration_candidate_40min.json',frozen['selected'])
    save_csv(args.output/'bias_by_minute.csv',blocks);save_csv(args.output/'periodic_cycles.csv',events)
    plot(args.output,d,trace,events)
    lines=['40-minute static IMU calibration assessment','',json.dumps(d['health']),
        'Periodic t4=10s: method | updates | segment displacement P95 (m) | speed P95 (m/s) | max observed accumulated p (m)']
    for m,s in periodic.items():
        lines.append(f'{m} | {s["actual_updates"]} | {s["segment_displacement_xy_m"]["p95"]:.6f} | {s["pre_speed_xy_mps"]["p95"]:.6f} | {s["max_observed_position_xy_m"]:.6f}')
    lines+=['','Frozen calibration selected: '+frozen['selected']['selected_method'],
            'Untouched final 10-minute test: method | 10s displacement RMS (m) | full interval displacement (m)']
    for m,s in frozen['test'].items():lines.append(f'{m} | {s["ten_second_displacement_rms_m"]:.6f} | {s["whole_displacement_xy_m"]:.6f}')
    lines+=['']+report['limitations']
    (args.output/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (args.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
