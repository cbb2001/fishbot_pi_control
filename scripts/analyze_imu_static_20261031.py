#!/usr/bin/env python3
"""Chronological stationary-IMU calibration, validation and untouched test replay.

No hardware access. SI units. Every evaluation window starts at p=v=0.
Raw means gravity-removed vendor-quaternion integration WITHOUT bias correction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.runtime.imu_static_calibration_20261031 import StaticESKF, rotation


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def load(folder):
    result = json.loads((folder/'result.json').read_text(encoding='utf8'))
    if result['executor_started'] or result['fault'] or result['reason'] != 'time_limit':
        raise ValueError('Not a clean IMU-only capture')
    rows = [json.loads(x) for x in (folder/'raw_imu.jsonl').read_text(encoding='utf8').splitlines()]
    if not all(r['ok'] for r in rows):
        raise ValueError('Invalid IMU samples')
    stamps = np.array([r['t_ns'] for r in rows], dtype=np.int64)
    reconstructed_t = (stamps-stamps[0])/1e9
    reconstructed_dt = np.diff(reconstructed_t)
    tid = np.array([r['data']['tid'] for r in rows])
    gaps = (np.diff(tid) % 65536)
    device_wrap = (tid[:-1] == 60000) & (tid[1:] == 1)
    gaps[device_wrap] = 1
    if np.any(reconstructed_dt <= 0) or np.max(reconstructed_dt) > .05 or np.any(gaps != 1):
        raise ValueError('Discontinuous timestamps or TID: inspect before integrating')
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf8'))
    hz=float(meta['imu_config']['sample_rate_hz'])
    # This unit wraps 60000->1, not 65535->0. Preserve original timestamps in
    # raw logs, but integrate the verified contiguous frame clock at configured Hz.
    t=np.arange(len(rows))/hz
    dt=np.diff(t)
    receipt_span=(rows[-1]['receive_t_ns']-rows[0]['receive_t_ns'])/1e9
    if abs(t[-1]-receipt_span) > max(.1,.001*t[-1]):
        raise ValueError('Configured sample rate disagrees with host receipt span')
    acc = np.array([r['data']['acc_mps2'] for r in rows])
    gyro = np.array([r['data']['gyro_radps'] for r in rows])
    quat = np.array([r['data']['quat'] for r in rows])
    euler = np.array([[r['data'][x+'_deg'] for x in ('roll','pitch','yaw')] for r in rows])
    R = np.array([rotation(q) for q in quat])
    if not np.isfinite(np.c_[acc, gyro, euler]).all():
        raise ValueError('Non-finite IMU input')
    field_counts = {}
    for row in rows:
        raw = bytes.fromhex(row['data']['raw_hex'])
        i = 5
        present = set()
        while i < len(raw)-2:
            present.add(raw[i]); i += 2+raw[i+1]
        if not {0x10, 0x20, 0x40, 0x41} <= present:
            raise ValueError('Missing acceleration/gyro/Euler/quaternion TLV')
        for key in present:
            field_counts[hex(key)] = field_counts.get(hex(key), 0)+1
    rec = np.array([r['receive_t_ns'] for r in rows], dtype=np.int64)
    health = dict(samples=len(rows), duration_s=float(t[-1]),
                  sample_rate_hz=hz, host_observed_frame_rate_hz=float((len(rows)-1)/receipt_span),
                  timebase='Contiguous verified frame sequence / configured sample_rate_hz; host duration cross-checked',
                  observed_60000_to_1_wraps=int(device_wrap.sum()),
                  legacy_logged_tid_gap_total=int(sum(r['data']['tid_gap'] for r in rows)),
                  original_reconstructed_dt_range_s=[float(reconstructed_dt.min()),float(reconstructed_dt.max())],
                  max_dt_s=float(dt.max()),
                  median_dt_s=float(np.median(dt)), tid_missing=int(sum(gaps-1)),
                  crc_errors=rows[-1]['data']['crc_error_count'],
                  decode_resync_count=rows[-1]['data']['decode_resync_count'],
                  receipt_minus_sample_s_range=[float(x) for x in [(rec-stamps).min()/1e9,(rec-stamps).max()/1e9]],
                  tlv_frame_counts=field_counts, temperature_available='0x1' in field_counts,
                  accel_norm_mean_mps2=float(np.linalg.norm(acc, axis=1).mean()),
                  gyro_unique_vectors=len(np.unique(gyro, axis=0)),
                  gyro_unchanged_adjacent_fraction=float(np.mean(np.all(np.diff(gyro,axis=0)==0,axis=1))),
                  euler_start_end_deg=[euler[0].tolist(),euler[-1].tolist()],
                  device_opens=result['device_opens'], executor_started=False)
    if 'calibrated_linear_acc_world_mps2' in rows[0]:
        frozen=json.loads((folder/'frozen_calibration.json').read_text(encoding='utf8'))
        expected=np.einsum('nij,nj->ni',R,acc-np.array(frozen['accel_bias_body_mps2']))-[0,0,frozen['gravity_mps2']]
        logged=np.array([r['calibrated_linear_acc_world_mps2'] for r in rows])
        health['live_vs_offline_acc_max_difference_mps2']=float(np.max(abs(expected-logged)))
        if health['live_vs_offline_acc_max_difference_mps2'] > 1e-10:
            raise ValueError('Live correction differs from offline computation')
    return dict(t=t, acc=acc, gyro=gyro, quat=quat, R=R, euler=euler, health=health)


def subset(d, start, end):
    mask = (d['t'] >= start) & (d['t'] < end)
    return {k:v[mask] for k,v in d.items() if isinstance(v,np.ndarray)}


def body_residual(d, g=9.80665):
    return d['acc'] - d['R'][:,2,:]*g


def block_means(t, x, seconds=1.):
    labels = np.floor((t-t[0])/seconds).astype(int)
    return np.array([x[labels==i].mean(0) for i in range(labels[-1])])


def integrate(t, a):
    dt = np.diff(t)[:,None]
    v = np.vstack((np.zeros(3), np.cumsum(.5*(a[:-1]+a[1:])*dt, axis=0)))
    p = np.vstack((np.zeros(3), np.cumsum(.5*(v[:-1]+v[1:])*dt, axis=0)))
    return v,p


def lowpass(t, a, tau):
    if not tau:
        return a.copy()
    out = np.empty_like(a); a1 = a[0].copy(); a2 = a[0].copy(); out[0] = a2
    for i,dt in enumerate(np.diff(t),1):
        gain = -np.expm1(-dt/tau)
        a1 += gain*(a[i]-a1); a2 += gain*(a1-a2); out[i] = a2
    return out


def replay_fixed(d, bias, tau=0):
    a = np.einsum('nij,nj->ni',d['R'],d['acc']-bias)-[0,0,9.80665]
    a = lowpass(d['t'],a,tau)
    return (*integrate(d['t'],a), a)


def replay_eskf(d, ba, bg, noise, known_rest):
    ekf = StaticESKF(d['quat'][0], ba, bg, noise)
    v,p = np.zeros_like(d['acc']),np.zeros_like(d['acc'])
    last_update = d['t'][0]
    for i,dt in enumerate(np.diff(d['t']),1):
        ekf.predict(.5*(d['acc'][i-1]+d['acc'][i]),
                    .5*(d['gyro'][i-1]+d['gyro'][i]),float(dt))
        if known_rest and d['t'][i]-last_update >= .1:
            start = np.searchsorted(d['t'],d['t'][i]-1.)
            aa,ww = d['acc'][start:i+1],d['gyro'][start:i+1]
            quiet = (d['t'][i]-d['t'][start] >= .9 and
                     max(aa.std(0)) < noise['quiet_acc_std_max_mps2'] and
                     np.linalg.norm(ww.mean(0)-bg) < noise['quiet_gyro_norm_max_radps'])
            # Quietness uses 1s; the gyro observation uses the new ~0.1s interval.
            obs_start=np.searchsorted(d['t'],last_update,side='right')
            ekf.update_known_rest(d['gyro'][obs_start:i+1].mean(0),
                                  externally_confirmed_static=True, quiet=quiet)
            last_update = d['t'][i]
        v[i],p[i] = ekf.v,ekf.p
    return v,p,ekf


def metrics(t, v, p):
    return dict(duration_s=float(t[-1]-t[0]), velocity_xyz_mps=v[-1].tolist(),
                displacement_xyz_m=p[-1].tolist(), speed_xy_mps=float(np.linalg.norm(v[-1,:2])),
                displacement_xy_m=float(np.linalg.norm(p[-1,:2])),
                speed_3d_mps=float(np.linalg.norm(v[-1])), displacement_3d_m=float(np.linalg.norm(p[-1])),
                max_displacement_xy_m=float(np.linalg.norm(p[:,:2],axis=1).max()))


def window_scores(d, candidates):
    scores = {key:[] for key in candidates}
    details = []
    for duration in (20,60):
        for start in np.arange(d['t'][0], d['t'][-1]-duration+.02, duration):
            win = subset(d,start,start+duration)
            if win['t'][-1]-win['t'][0] < duration-.03:
                continue
            for name,(bias,tau) in candidates.items():
                v,p,_ = replay_fixed(win,bias,tau)
                m = metrics(win['t'],v,p)
                scores[name].append(m['displacement_xy_m']**2)
                details.append(dict(method=name,window_start_s=float(start),nominal_duration_s=duration,**m))
    return {k:float(np.sqrt(np.mean(v))) for k,v in scores.items()},details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-params',type=Path,help='Independent capture: use previous fit without refitting')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    d=load(args.session); t=d['t']
    if args.frozen_params:
        params=json.loads(args.frozen_params.read_text(encoding='utf8'))
        candidates={k:(np.array(v['bias']),v['tau_s']) for k,v in params['candidates'].items()}
        noise=params['noise']; bg=np.array(params['gyro_bias_body_radps'])
        selected=params['selected_method']; test=subset(d,0,t[-1]+.001)
        protocol=dict(mode='independent_frozen_validation',source_parameters=str(args.frozen_params),test_s=[0,float(t[-1])])
        selection_scores={}; selection_details=[]; training={}
    else:
        if t[-1] < 599:
            raise ValueError('Need full 600-second capture')
        train=subset(d,60,240); validation=subset(d,240,360); test=subset(d,360,t[-1]+.001)
        residual=body_residual(train); blocks=block_means(train['t'],residual)
        bg=train['gyro'].mean(0)
        acc_density=np.maximum(blocks.std(0,ddof=1),.001)
        gyro_blocks=block_means(train['t'],train['gyro'])
        gyro_density=np.maximum(gyro_blocks.std(0,ddof=1),.0001)
        ab10=block_means(train['t'],residual,10)
        gb10=block_means(train['t'],train['gyro'],10)
        noise=dict(accel_density=acc_density.tolist(),gyro_density=gyro_density.tolist(),
                   accel_bias_rw=np.maximum(np.std(np.diff(ab10,axis=0),axis=0)/np.sqrt(10),.00001).tolist(),
                   gyro_bias_rw=np.maximum(np.std(np.diff(gb10,axis=0),axis=0)/np.sqrt(10),.000001).tolist(),
                   zupt_velocity_std_mps=.005,
                   rest_gyro_std_radps=np.maximum(block_means(train['t'],train['gyro'],.1).std(0),.0003).tolist(),
                   quiet_acc_std_max_mps2=max(.03,float(np.max(train['acc'].std(0)))*3),
                   quiet_gyro_norm_max_radps=max(.01,float(np.linalg.norm(train['gyro']-bg,axis=1).max())*2),
                   noise_estimator='1s block-mean std; 10s block differences/sqrt(10) for bias RW; conservative engineering estimates, not Allan identification',
                   fixed_hyperparameters='ZUPT 10Hz, velocity std .005m/s, 1s quiet gate; no tuning on test')
        mean=residual.mean(0); robust=np.median(blocks,axis=0)
        _,_,trained=replay_eskf(train,mean,bg,noise,True)
        candidates={'raw_gravity_only':(np.zeros(3),0.),
                    'startup5_bias':(residual[train['t']<65].mean(0),0.),
                    'mean180_bias':(mean,0.), 'robust_block_bias':(robust,0.),
                    'recent60_bias':(residual[train['t']>=180].mean(0),0.),
                    'mean180_bias_lpf02':(mean,.2),
                    'eskf_trained_bias_frozen':(trained.ba.copy(),0.)}
        # Include the uncalibrated baseline: do not force a worse fitted bias.
        selection_scores,selection_details=window_scores(validation,candidates)
        selected=min(selection_scores,key=selection_scores.get)
        training=dict(mean_body_residual_mps2=mean.tolist(),robust_body_residual_mps2=robust.tolist(),
                      residual_std_mps2=residual.std(0).tolist(),gyro_mean_radps=bg.tolist(),
                      gyro_std_radps=train['gyro'].std(0).tolist(),eskf_final_ba=trained.ba.tolist(),
                      eskf_final_bg=trained.bg.tolist(),eskf_updates=trained.update_count,
                      early_late_bias_mps2=[ab10[0].tolist(),ab10[-1].tolist()])
        protocol=dict(mode='chronological',warmup_s=[0,60],train_s=[60,240],validation_s=[240,360],
                      test_s=[360,float(t[-1])],selection='Lowest validation RMS endpoint horizontal displacement across disjoint 20s and 60s windows; no test fitting')
        params=dict(schema_version=1,quaternion_order='wxyz',rotation='body_to_world',gravity_mps2=9.80665,
                    accel_bias_body_mps2=candidates[selected][0].tolist(),gyro_bias_body_radps=bg.tolist(),
                    selected_method=selected,accel_two_stage_lpf_tau_s=candidates[selected][1],noise=noise,
                    candidates={k:dict(bias=b.tolist(),tau_s=tau) for k,(b,tau) in candidates.items()},
                    source_session=str(args.session),source_raw_sha256=hashlib.sha256((args.session/'raw_imu.jsonl').read_bytes()).hexdigest(),
                    fit_interval_s=[60,240],selection_interval_s=[240,360],
                    scope='Candidate effective single-pose bias at this temperature/power cycle, NOT a full intrinsic calibration; dynamic validation required',
                    temperature_calibration_available=False,auto_zupt_during_motion=False,imu_tid_modulus=60000,
                    integration='Replace startup b_body, never subtract both. Correct gyro only if integrating raw gyro; does not correct vendor quaternion.')
        dump(args.output/'calibration_candidate.json',params)
    curves={}; full={}; windows=[]
    for name,(bias,tau) in candidates.items():
        v,p,a=replay_fixed(test,bias,tau);curves[name]=(v,p)
        full[name]=metrics(test['t'],v,p)
    _,windows=window_scores(test,candidates)
    ba=candidates[selected][0]
    for name,rest in [('eskf_predict_only',False),('eskf_known_rest_zupt',True)]:
        v,p,ekf=replay_eskf(test,ba,bg,noise,rest);curves[name]=(v,p)
        full[name]=dict(**metrics(test['t'],v,p),zupt_updates=ekf.update_count,
                        final_accel_bias_mps2=ekf.ba.tolist(),final_gyro_bias_radps=ekf.bg.tolist())
    prefixes=[]
    for duration in (10,20,30,60,120):
        ix=np.searchsorted(test['t']-test['t'][0],duration,side='right')
        if ix<2 or test['t'][ix-1]-test['t'][0]<duration-.03:continue
        for name,(v,p) in curves.items():
            prefixes.append(dict(method=name,nominal_duration_s=duration,**metrics(test['t'][:ix],v[:ix],p[:ix])))
    if args.frozen_params and (args.session/'frozen_calibration.json').exists():
        if args.frozen_params.read_bytes() != (args.session/'frozen_calibration.json').read_bytes():
            raise ValueError('Replay parameter file differs from live capture parameter file')
    report=dict(health=d['health'],protocol=protocol,training=training,selected_method=selected,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [args.session/'raw_imu.jsonl',Path(__file__),
                                ROOT/'control/runtime/imu_static_calibration_20261031.py']},
                validation_scores_m=selection_scores,full_holdout=full,prefix_windows=prefixes,
                repeated_windows=windows,validation_windows=selection_details,
                caveats=['Physical stillness supplied by operator; IMU alone cannot prove zero velocity.',
                         'No UWB, pressure, position resets, velocity damping, endpoint detrending, or test mean subtraction.',
                         'ESKF ZUPT uses external stand-still knowledge; predict-only has NO updates.',
                         'Frozen-bias methods use vendor quaternion at every frame; ESKF propagates raw gyro, hence a different attitude source.',
                         'One pose cannot separate accelerometer bias, scale, misalignment, gravity error and tilt error.',
                         'Absent temperature TLV is not a 0-degree measurement; no thermal model fitted.',
                         'Noise estimates describe already processed, correlated device output, not intrinsic IMU noise density.',
                         'Test stationary drift does not establish swimming accuracy.'])
    dump(args.output/'report.json',report)
    with (args.output/'window_metrics.csv').open('w',newline='',encoding='utf8') as f:
        rows=prefixes+windows
        fields=sorted(set().union(*(x.keys() for x in rows)))
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    with (args.output/'holdout_timeseries.csv').open('w',newline='',encoding='utf8') as f:
        fields=['time_s']+[f'{m}_{k}{ax}' for m in curves for k in ('v','p') for ax in 'xyz']
        writer=csv.writer(f);writer.writerow(fields)
        for i in range(0,len(test['t']),20):
            writer.writerow([test['t'][i]-test['t'][0]]+[float(x) for v,p in curves.values() for arr in (v,p) for x in arr[i]])
    plot(args.output,d,test,curves,selected)
    lines=['IMU static calibration report','',f'Selected by validation: {selected}',
           json.dumps(protocol,ensure_ascii=False),f'Capture: {d["health"]}', '',
           'Whole held-out interval (p0=v0=0; units m/s and m):',
           'method | end speed XY | end displacement XY | end displacement 3D']
    for name,m in full.items():
        lines.append(f'{name} | {m["speed_xy_mps"]:.6f} | {m["displacement_xy_m"]:.6f} | {m["displacement_3d_m"]:.6f}')
    lines+=['','Caveats:']+report['caveats']
    (args.output/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(selected=selected,health=d['health'],full_holdout=full),indent=2))


def plot(output,d,test,curves,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    residual=body_residual(d)
    for k,label in enumerate('XYZ'):
        # 1-second averages reveal drift without filling the chart with 200Hz noise.
        for ax,values in [(axs[0],residual),(axs[1],d['gyro']),(axs[2],d['euler']-d['euler'][0])]:
            bm=block_means(d['t'],values)
            ax.plot(np.arange(len(bm))+.5,bm[:,k],label=('Roll','Pitch','Yaw')[k] if ax is axs[2] else label,lw=1)
    for ax,title in zip(axs,['Body gravity residual (m/s2)','Gyroscope (rad/s)','Reported Euler change (deg)']):
        ax.set_ylabel(title);ax.grid(alpha=.25);ax.legend(ncol=3,loc='upper right')
        if d['t'][-1]>599:
            for start,end,color,label in [(0,60,'gray','warmup'),(60,240,'green','train'),(240,360,'orange','validation'),(360,600,'blue','test')]:
                ax.axvspan(start,end,color=color,alpha=.05)
    axs[-1].set_xlabel('Capture time (s)');fig.suptitle('Stationary IMU diagnostics - 1 s means')
    fig.savefig(output/'sensor_diagnostics.png');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
    names=list(dict.fromkeys(['raw_gravity_only','startup5_bias','mean180_bias','mean180_bias_lpf02',selected,'eskf_predict_only','eskf_known_rest_zupt']))
    tx=test['t']-test['t'][0]
    stride=max(1,round(.1/float(np.median(np.diff(tx)))))
    for index,name in enumerate(names):
        color=plt.get_cmap('tab10')(index)
        v,p=curves[name]
        for ax,values in [(axs[0,0],v),(axs[1,0],p)]:
            ax.plot(tx[::stride],np.linalg.norm(values[::stride,:2],axis=1),label=name,lw=1.3,color=color)
        if name != 'eskf_predict_only':
            axs[0,1].plot(tx[::stride],np.linalg.norm(v[::stride,:2],axis=1),label=name,lw=1.3,color=color)
            axs[1,1].plot(tx[::stride],np.linalg.norm(p[::stride,:2],axis=1),label=name,lw=1.3,color=color)
    for ax in axs.flat:ax.grid(alpha=.25)
    axs[0,0].set_ylabel('Horizontal speed error (m/s)');axs[1,0].set_ylabel('Horizontal displacement error (m)')
    axs[0,0].set_title('All methods');axs[0,1].set_title('Bias correction / known-rest detail')
    for ax in axs[1,:]:ax.set_xlabel('Held-out time (s)')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Untouched stationary holdout; p0=v0=0; ZUPT has external rest information')
    fig.savefig(output/'drift_comparison.png');plt.close(fig)


if __name__=='__main__':main()
