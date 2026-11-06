#!/usr/bin/env python3
"""Offline raw-IMU integration and range EKF. Does not access robot hardware.

State: [x,y,vx,vy,bx,by], b is effective GLOBAL acceleration error, not a new
static sensor calibration. Nonlinearity is the four-anchor range observation.
Only navigate intervals propagate position/velocity. Pauses hold the means,
inflate position uncertainty, and preserve learned bias for the next segment.
After learning bias from a fix, restart position exactly at the logged UWB XY.
This is a UWB-anchored dead-reckoner with EKF bias learning, not a pure EKF.
No future UWB is used to revise a previously plotted segment.
"""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from control.runtime.underwater_navigation_20261026 import rotation


def read_jsonl(path):
    data=path.read_bytes();rows=[];bad=[]
    for i,line in enumerate(data.splitlines(),1):
        line=line.strip(b'\x00 \r\t')
        if not line:continue
        try:rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):bad.append(i)
    if bad or b'\x00' in data:raise ValueError(f'Incomplete source log: {path}, bad lines={bad}')
    return rows


def integrate(t,a):
    dt=np.diff(t)[:,None]
    v=np.vstack((np.zeros(2),np.cumsum(.5*(a[:-1]+a[1:])*dt,axis=0)))
    p=np.vstack((np.zeros(2),np.cumsum(.5*(v[:-1]+v[1:])*dt,axis=0)))
    return v,p


def load_session(folder,tau):
    meta=yaml.safe_load((folder/'metadata.yaml').read_text(encoding='utf8'))
    cal=json.loads((folder/'calibration.json').read_text(encoding='utf8'))
    rows=read_jsonl(folder/'imu_filtered.jsonl');raw=read_jsonl(folder/'raw_imu.jsonl');events=read_jsonl(folder/'events.jsonl')
    bytime={r['t_ns']:r['data'] for r in raw};t=np.array([r['sample_t_ns'] for r in rows],dtype=np.int64)
    angle=math.radians(-cal['yaw_zero_deg']);pool=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    bias=np.array(cal['b_body'])
    acc=np.array([pool@(rotation(bytime[int(stamp)]['quat'])@(np.array(bytime[int(stamp)]['acc_mps2'])-bias)-[0,0,cal['gravity_mps2']]) for stamp in t])[:,:2]
    diff=float(np.max(abs(acc-np.array([r['raw_linear_acc'][:2] for r in rows]))))
    assert diff<1e-10
    fixes=[r for r in events if r['event']=='position_corrected'];starts=[r['segment_start_t_ns'] for r in events if r['event']=='navigation_started']
    surfaces=[r['t_ns'] for r in events if r['event']=='surface_started']
    assert len(fixes)==len(starts)
    t0=starts[0];end=int(t[-1]);segments=[]
    for j,start in enumerate(starts):
        boundary=starts[j+1] if j+1<len(starts) else end+1
        candidates=[s for s in surfaces if start<s<boundary];stop=min(candidates) if candidates else end
        stamps=np.unique(np.r_[start,t[(t>start)&(t<stop)],stop]).astype(np.int64)
        rel=(stamps-start)/1e9
        a=np.column_stack([np.interp((stamps-t0)/1e9,(t-t0)/1e9,acc[:,k]) for k in range(2)])
        uv_raw,up_raw=integrate(rel,a)
        if tau>0:
            filtered=np.zeros_like(a);a1=np.zeros(2);a2=np.zeros(2)
            for k,dt in enumerate(np.diff(rel),1):
                gain=-math.expm1(-float(dt)/tau);a1+=gain*(a[k]-a1);a2+=gain*(a1-a2);filtered[k]=a2
            uv,up=integrate(rel,filtered)
        else:uv,up=uv_raw,up_raw
        segments.append(dict(start_ns=start,end_ns=stop,start_s=(start-t0)/1e9,end_s=(stop-t0)/1e9,
            time=rel,duration=float(rel[-1]),dv=uv,dp=up,raw_dv=uv_raw,raw_dp=up_raw))
        if j+1<len(fixes):assert stop<=fixes[j+1]['fix_t_ns']<=starts[j+1]
    depths=np.array([r['xyz'][2] for r in rows])
    for fix in fixes:
        ix=max(0,int(np.searchsorted(t,fix['fix_t_ns'],side='right'))-1)
        fix['depth_m']=float(depths[ix])
    cfg=meta['config']['navigation'];L,W,H=[cfg[k] for k in ['L','W','H']]
    ct=np.unique(np.r_[t0,t[(t>t0)&(t<=end)]]).astype(np.int64);ca=np.column_stack([np.interp((ct-t0)/1e9,(t-t0)/1e9,acc[:,k]) for k in range(2)])
    cv,cp=integrate((ct-t0)/1e9,ca);cp+=fixes[0]['xy']
    # Numerical baseline against previously delivered per-segment method2 traces.
    previous=folder.parent/'plots_two_integrations_20261030'/(folder.name[:15]+'_method2.npz')
    check=None
    if previous.exists():
        z=np.load(previous);errs=[]
        for j,s in enumerate(segments):
            ix=int(np.argmin(abs(z['t_s']-s['end_s'])))
            errs.append(np.linalg.norm(s['raw_dp'][-1]+fixes[j]['xy']-z['xy'][ix]))
        check=float(max(errs));assert check<1e-8,check
    return dict(name=folder.name,folder=str(folder),t0=t0,end_s=(end-t0)/1e9,fixes=fixes,segments=segments,
        anchors=np.array([[0,0],[L,0],[L,W],[0,W]],dtype=float),anchor_height=H,
        continuous_time=(ct-t0)/1e9,continuous_xy=cp,continuous_v=cv,
        validation=dict(raw_acc_max_difference=diff,method2_endpoint_max_difference=check),tau=tau)


def transition(T,sigma_a,sigma_b):
    I=np.eye(2);F=np.eye(6);F[:2,2:4]=T*I;F[:2,4:6]=-.5*T*T*I;F[2:4,4:6]=-T*I
    qa=sigma_a**2;qb=sigma_b**2
    small=np.array([[qa*T**3/3+qb*T**5/20,qa*T*T/2+qb*T**4/8,-qb*T**3/6],
                    [qa*T*T/2+qb*T**4/8,qa*T+qb*T**3/3,-qb*T*T/2],
                    [-qb*T**3/6,-qb*T*T/2,qb*T]])
    return F,np.kron(small,I)


def h_range(x,anchors,height):
    delta=x[:2]-anchors;pred=np.sqrt(np.sum(delta*delta,axis=1)+height*height)
    H=np.zeros((4,6));H[:,:2]=delta/pred[:,None]
    return pred,H


def range_update(x,P,ranges,anchors,height,cfg):
    prior=x.copy();cov=P.copy();z=np.array(ranges,dtype=float);R=np.eye(4)*cfg['range_sigma_m']**2
    predicted,H=h_range(prior,anchors,height);innovation=z-predicted;S=H@cov@H.T+R
    nis=float(innovation@np.linalg.solve(S,innovation))
    info=dict(nis=nis,accepted=nis<=cfg['innovation_gate'],pre_range_rms_m=float(np.sqrt(np.mean(innovation**2))))
    if not info['accepted']:return x,P,info
    current=prior.copy()
    for _ in range(cfg['iterations']):
        pred,H=h_range(current,anchors,height);S=H@cov@H.T+R
        K=np.linalg.solve(S,H@cov).T
        current=prior+K@(z-pred+H@(current-prior))
    A=np.eye(6)-K@H;P=A@cov@A.T+K@R@K.T
    P=.5*(P+P.T)
    assert np.isfinite(current).all() and np.linalg.eigvalsh(P).min()>-1e-9
    return current,P,info


def paused(P,duration,cfg,bias_enabled):
    # Freeze means, but allow unmodelled displacement from unknown surface velocity.
    P=P.copy();P[:2,:2]+=np.eye(2)*(cfg['pause_velocity_sigma_mps']*duration)**2
    if bias_enabled:P[4:6,4:6]+=np.eye(2)*cfg['bias_random_walk']**2*duration
    return P


def anchor_to_uwb(x,P,fix,anchors,height,cfg):
    """Reset to the log-accepted fix while retaining EKF bias information.

    Local range geometry approximates the new XY uncertainty. Discard old
    position cross-covariances as an independent-restart approximation; do not
    treat the forced XY mean as an error-free measurement or update bias twice.
    The log's validity decision governs the anchor. The EKF NIS gate governs
    bias learning only, and a rejection must not retain an old position origin.
    """
    x=x.copy();P=P.copy();x[:2]=fix['xy']
    _,H=h_range(x,anchors,height)
    information=H[:,:2].T@H[:,:2]/cfg['range_sigma_m']**2
    position_cov=np.linalg.solve(information,np.eye(2))
    P[:2,:]=0.;P[:,:2]=0.;P[:2,:2]=position_cov
    assert np.linalg.eigvalsh(P).min()>-1e-9
    return x,P


def replay(d,cfg,bias_enabled=True,traces=False):
    x=np.r_[d['fixes'][0]['xy'],0.,0.,0.,0.]
    P=np.diag([cfg['initial_position_sigma_m']**2]*2+[cfg['segment_velocity_sigma_mps']**2]*2+
              [cfg['initial_bias_sigma_mps2']**2 if bias_enabled else 0.]*2)
    records=[];paths=[];bias_history=[dict(time_s=0.,bias=[0.,0.])]
    for j,s in enumerate(d['segments']):
        # Every segment restarts with the user's zero-velocity prior, with uncertainty.
        P=paused(P,max(0,(s['start_ns']-d['fixes'][j]['fix_t_ns'])/1e9),cfg,bias_enabled)
        x[2:4]=0.;P[2:4,:]=0.;P[:,2:4]=0.;P[2:4,2:4]=np.eye(2)*cfg['segment_velocity_sigma_mps']**2
        before=x.copy();Pbefore=P.copy();T=s['duration']
        assert np.array_equal(before[:2],np.asarray(d['fixes'][j]['xy']))
        F,Q=transition(T,cfg['accel_noise_density'],cfg['bias_random_walk'] if bias_enabled else 0.)
        x=F@x;x[:2]+=s['dp'][-1];x[2:4]+=s['dv'][-1];P=F@P@F.T+Q
        if traces:
            tt=s['time'];p=before[:2]+before[2:4]*tt[:,None]+s['dp']-.5*tt[:,None]**2*before[4:6]
            v=before[2:4]+s['dv']-tt[:,None]*before[4:6]
            paths.append(dict(time_s=(tt+s['start_s']).tolist(),xy=p.tolist(),velocity=v.tolist(),bias=before[4:6].tolist()))
        if j+1>=len(d['fixes']):break
        f=d['fixes'][j+1];duration=(f['fix_t_ns']-s['end_ns'])/1e9
        P=paused(P,duration,cfg,bias_enabled)
        pre=x.copy();preP=P.copy();baseline=np.array(d['fixes'][j]['xy'])+s['raw_dp'][-1]
        x,P,info=range_update(x,P,f['ranges_m'],d['anchors'],d['anchor_height']+f['depth_m'],cfg)
        range_posterior=x.copy()
        x,P=anchor_to_uwb(x,P,f,d['anchors'],d['anchor_height']+f['depth_m'],cfg)
        assert np.array_equal(x[4:6],range_posterior[4:6])
        record=dict(index=j+1,time_s=(f['fix_t_ns']-d['t0'])/1e9,pause_s=duration,
            accepted_uwb_xy=f['xy'],prior_xy=pre[:2].tolist(),posterior_xy=x[:2].tolist(),
            range_posterior_xy=range_posterior[:2].tolist(),position_anchored=True,
            prior_velocity=pre[2:4].tolist(),bias_before=pre[4:6].tolist(),bias_after=x[4:6].tolist(),
            prior_position_std=np.sqrt(np.diag(preP)[:2]).tolist(),posterior_bias_std=np.sqrt(np.diag(P)[4:6]).tolist(),
            pre_update_difference_m=float(np.linalg.norm(pre[:2]-f['xy'])),
            post_update_difference_m=float(np.linalg.norm(x[:2]-f['xy'])),
            baseline_difference_m=float(np.linalg.norm(baseline-f['xy'])),baseline_xy=baseline.tolist(),**info)
        records.append(record);bias_history.append(dict(time_s=record['time_s'],bias=x[4:6].tolist()))
    errors=np.array([r['pre_update_difference_m'] for r in records]);base=np.array([r['baseline_difference_m'] for r in records]);split=max(2,len(records)//2)
    def score(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)),max=float(max(v)))
    return dict(config=cfg,bias_enabled=bias_enabled,records=records,paths=paths,bias_history=bias_history,
        position_policy='exact logged UWB XY + EKF-bias-corrected relative displacement',
        metrics=dict(all=score(errors),train=score(errors[:split]),validation=score(errors[split:]),
            baseline_all=score(base),baseline_train=score(base[:split]),baseline_validation=score(base[split:]),
            rejected=sum(not r['accepted'] for r in records),split=split),final_state=x.tolist(),
        maximum_bias_norm=float(max(np.linalg.norm(r['bias_after']) for r in records)))


def self_test():
    # Analytic range Jacobian, covariance semigroup, synthetic bias convergence, future-data causality.
    anchors=np.array([[0.,0.],[50.,0.],[50.,26.5],[0.,26.5]])
    x=np.array([4.,7.,.1,-.2,.02,-.03]);_,H=h_range(x,anchors,1.9)
    for k in range(6):
        dx=np.zeros(6);dx[k]=1e-5
        numeric=(h_range(x+dx,anchors,1.9)[0]-h_range(x-dx,anchors,1.9)[0])/2e-5
        assert np.allclose(H[:,k],numeric,atol=1e-8)
    F,Q=transition(2.7,.1,.003);F1,Q1=transition(1.1,.1,.003);F2,Q2=transition(1.6,.1,.003)
    assert np.allclose(F,F2@F1) and np.allclose(Q,F2@Q1@F2.T+Q2,atol=1e-12)
    assert np.linalg.eigvalsh(Q).min()>0
    cfg=dict(range_sigma_m=.05,innovation_gate=1e6,iterations=3,accel_noise_density=.001,bias_random_walk=.00001,
        initial_bias_sigma_mps2=.1,initial_position_sigma_m=.1,segment_velocity_sigma_mps=.005,pause_velocity_sigma_mps=.1)
    truth=np.array([.04,-.025]);x=np.array([4.,7.,0.,0.,0.,0.]);P=np.diag([.01,.01,.001,.001,.01,.01]);z=h_range(x,anchors,1.9)[0]
    for _ in range(10):
        x[2:4]=0.;P[2:4,:]=0.;P[:,2:4]=0.;P[2:4,2:4]=np.eye(2)*.005**2
        F,Q=transition(10.,.001,.00001);x=F@x;x[:2]+=.5*truth*100;x[2:4]+=truth*10;P=F@P@F.T+Q
        x,P,info=range_update(x,P,z,anchors,1.9,cfg)
        learned=x[4:6].copy();bias_cov=P[4:6,4:6].copy()
        x,P=anchor_to_uwb(x,P,dict(xy=[4.,7.]),anchors,1.9,cfg)
        assert np.array_equal(x[:2],[4.,7.]) and np.array_equal(x[4:6],learned)
        assert np.array_equal(P[4:6,4:6],bias_cov) and np.linalg.eigvalsh(P[:2,:2]).min()>0
    assert np.linalg.norm(x[4:6]-truth)<.002,(x,truth)
    recovered=x[4:6].copy()
    # A log-accepted fix remains the origin even when NIS rejects bias learning.
    rejected,P_rejected,info=range_update(x,P,z+100,anchors,1.9,{**cfg,'innovation_gate':1.})
    assert not info['accepted']
    anchored,_=anchor_to_uwb(rejected,P_rejected,dict(xy=[5.,8.]),anchors,1.9,cfg)
    assert np.array_equal(anchored[:2],[5.,8.]) and np.array_equal(anchored[4:6],recovered)
    return dict(range_jacobian='passed',covariance_propagation='passed',synthetic_bias_recovery=recovered.tolist(),
        exact_uwb_anchor_preserves_bias='passed',bias_gate_does_not_block_log_accepted_anchor='passed')


def plots(d,default,tuned,ablation,out,swept=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected_label='UWB起点＋EKF补偿（'+('前半程选参' if swept else '指定参数')+'）'
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
    name=d['name'][:15];fixxy=np.array([f['xy'] for f in d['fixes']]);fixt=np.array([max(0,(f['fix_t_ns']-d['t0'])/1e9) for f in d['fixes']])
    fig,ax=plt.subplots(2,2,figsize=(15,10));fig.suptitle(name+'：① 原始线加速度连续积分；② UWB起点＋EKF补偿后的分段积分',fontsize=14)
    ax[0,0].plot(d['continuous_xy'][:,0],d['continuous_xy'][:,1],label='① 连续原始积分',c='#888888')
    ax[0,0].scatter(fixxy[:,0],fixxy[:,1],c='black',s=22,label='接受UWB坐标');ax[0,0].set_title('连续积分参考（无中途校正）')
    for j,s in enumerate(d['segments']):
        p=np.array(d['fixes'][j]['xy'])+s['raw_dp'];ax[0,1].plot(p[:,0],p[:,1],c='#bbbbbb',lw=1,label='原方法2：UWB直接重置' if j==0 else None)
    variants=([(default,'#d88937','EKF：配置参数')] if default['config']!=tuned['config'] else [])+[(tuned,'#247f9b',selected_label)]
    for r,color,label in variants:
        for j,path in enumerate(r['paths']):
            p=np.array(path['xy']);ax[0,1].plot(p[:,0],p[:,1],c=color,lw=1.2,label=label if j==0 else None)
    ax[0,1].scatter(fixxy[:,0],fixxy[:,1],c='black',s=22,label='接受UWB坐标');ax[0,1].set_title('每段严格从UWB坐标出发；上浮/等待不积分')
    for a in ax[0]:a.set_xlabel('X / m');a.set_ylabel('Y / m');a.set_aspect('equal',adjustable='datalim')
    for j,s in enumerate(d['segments']):ax[1,0].plot(s['time']+s['start_s'],s['raw_dp'][:,1]+d['fixes'][j]['xy'][1],c='#bbb',label='原方法2' if j==0 else None)
    for j,path in enumerate(tuned['paths']):ax[1,0].plot(path['time_s'],np.array(path['xy'])[:,1],c='#247f9b',label=selected_label if j==0 else None)
    ax[1,0].scatter(fixt,fixxy[:,1],s=20,c='black',label='UWB');ax[1,0].set_xlabel('首次导航后 / s');ax[1,0].set_ylabel('Y / m');ax[1,0].set_title('全局Y＝本次UWB的Y＋补偿后的本段积分位移')
    xx=np.arange(1,len(tuned['records'])+1);split=tuned['metrics']['split']
    curves=[([r['baseline_difference_m'] for r in tuned['records']],'#999','原方法2')]
    curves += [([r['pre_update_difference_m'] for r in v['records']],color,label) for v,color,label in variants]
    curves += [
        ([r['pre_update_difference_m'] for r in ablation['records']],'#a65a93','同参EKF，不学习偏差')]
    for vals,color,label in curves:
        ax[1,1].plot(xx,vals,'o-',c=color,label=label)
    ax[1,1].axvspan(split+.5,len(xx)+.5,color='#e9e1c9',alpha=.4,label='后半程验证' if swept else '后半程')
    ax[1,1].set_xlabel('第几次重新定位');ax[1,1].set_ylabel('更新前与接受UWB坐标之差 / m');ax[1,1].set_title('每个点在使用该次UWB之前计算；并非真实定位精度')
    for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/(name+'_comparison.png'),dpi=170);fig.savefig(out/(name+'_comparison.pdf'));plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(14,9));fig.suptitle(name+'：EKF状态和更新诊断')
    b=np.array([r['bias'] for r in tuned['bias_history']]);bt=[r['time_s'] for r in tuned['bias_history']]
    ax[0,0].step(bt,b[:,0],where='post',label='bx');ax[0,0].step(bt,b[:,1],where='post',label='by');ax[0,0].set_ylabel('等效加速度偏差 / m/s²');ax[0,0].set_xlabel('时间 / s')
    for j,path in enumerate(tuned['paths']):ax[0,1].plot(path['time_s'],np.linalg.norm(path['velocity'],axis=1),label='估计水平速度' if j==0 else None)
    ax[0,1].set_xlabel('时间 / s');ax[0,1].set_ylabel('m/s')
    ax[1,0].plot(xx,[r['nis'] for r in tuned['records']],'o-',label='四路测距NIS');ax[1,0].axhline(tuned['config']['innovation_gate'],ls='--',c='red',label='拒绝门限');ax[1,0].set_yscale('symlog',linthresh=1)
    ax[1,0].set_xlabel('重新定位编号');ax[1,0].set_ylabel('NIS')
    ax[1,1].plot(xx,[r['pre_update_difference_m'] for r in tuned['records']],'o-',label='更新前差值')
    ax[1,1].plot(xx,[r['post_update_difference_m'] for r in tuned['records']],'o-',label='起点强制对齐后为0（非精度指标）');ax[1,1].set_xlabel('重新定位编号');ax[1,1].set_ylabel('与原始UWB坐标之差 / m')
    for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/(name+'_diagnostics.png'),dpi=170);fig.savefig(out/(name+'_diagnostics.pdf'));plt.close(fig)
    # User-requested standalone figures, without other algorithms overlaid.
    for mode in (1,2):
        fig,ax=plt.subplots(1,3,figsize=(16,5));fig.suptitle(name+('：① 原始全局加速度连续积分' if mode==1 else '：② UWB起点＋EKF补偿后的分段积分（'+('前半程选参' if swept else '指定参数')+'）'))
        paths=[dict(time_s=d['continuous_time'],xy=d['continuous_xy'])] if mode==1 else tuned['paths']
        for j,path in enumerate(paths):
            p=np.asarray(path['xy']);t=np.asarray(path['time_s']);color='#247f9b' if mode==1 else plt.cm.tab20(j%20)
            ax[0].plot(t,p[:,0],c=color);ax[1].plot(t,p[:,1],c=color);ax[2].plot(p[:,0],p[:,1],c=color,label=f'第{j+1}段' if mode==2 else '连续积分')
        for k in range(2):ax[k].scatter(fixt,fixxy[:,k],c='black',s=18);ax[k].set_xlabel('时间 / s');ax[k].set_ylabel(('X' if k==0 else 'Y')+' / m')
        ax[2].scatter(fixxy[:,0],fixxy[:,1],c='black',s=22,label='接受UWB坐标');ax[2].set_xlabel('X / m');ax[2].set_ylabel('Y / m');ax[2].set_aspect('equal',adjustable='datalim');ax[2].legend(fontsize=7,ncol=2)
        for a in ax:a.grid(alpha=.2)
        fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(out/(name+f'_method{mode}.png'),dpi=170);fig.savefig(out/(name+f'_method{mode}.pdf'));plt.close(fig)


def validate_config(c):
    keys=['range_sigma_m','accel_noise_density','bias_random_walk','initial_bias_sigma_mps2','initial_position_sigma_m','segment_velocity_sigma_mps','pause_velocity_sigma_mps','accel_tau_s','innovation_gate']
    for k in keys:
        if not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<0:raise ValueError(f'Invalid {k}')
    if c['range_sigma_m']==0 or c['innovation_gate']==0:raise ValueError('range_sigma_m and innovation_gate must be positive')
    if type(c['iterations']) is not int or not 1<=c['iterations']<=10:raise ValueError('iterations must be 1..10')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/underwater_ekf_replay_20261030.yaml')
    parser.add_argument('--output',type=Path);parser.add_argument('--sweep',action='store_true')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    checks=self_test()
    if args.self_test:print(json.dumps(checks,indent=2));return
    settings=yaml.safe_load(args.config.read_text(encoding='utf8'));cfg=settings['ekf'];validate_config(cfg)
    out=args.output or ROOT/settings['output'];out.mkdir(parents=True,exist_ok=True)
    data_cache={}
    def inputs_for(c):
        tau=float(c['accel_tau_s'])
        if tau not in data_cache:
            data_cache[tau]=[load_session(ROOT/Path(p),tau) for p in settings['sessions']]
        return data_cache[tau]
    data=inputs_for(cfg)
    defaults=[replay(d,cfg,traces=True) for d in data]
    candidates=[];chosen=cfg.copy()
    if args.sweep:
        grid=settings['sweep'];keys=list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            c={**cfg,**dict(zip(keys,values))};validate_config(c);runs=[replay(d,c) for d in inputs_for(c)]
            # Hyperparameter selection sees only the first half's PRE-update differences.
            score=np.mean([r['metrics']['train']['mean']/r['metrics']['baseline_train']['mean']+
                .1*sum(not v['accepted'] for v in r['records'][:r['metrics']['split']])/r['metrics']['split'] for r in runs])
            candidates.append(dict(config=c,training_score=float(score),metrics=[r['metrics'] for r in runs]))
        candidates.sort(key=lambda r:r['training_score']);chosen=candidates[0]['config']
        print(f'Completed {len(candidates)} parameter combinations; selected on first halves only.',flush=True)
    data=inputs_for(chosen)
    tuned=[replay(d,chosen,traces=True) for d in data];ablations=[replay(d,chosen,bias_enabled=False,traces=True) for d in data]
    # Perturb only the final observation: all earlier predictions must remain identical.
    for d,r in zip(data,tuned):
        altered=copy.deepcopy(d);altered['fixes'][-1]['ranges_m']=[v+2 for v in altered['fixes'][-1]['ranges_m']]
        rr=replay(altered,chosen)
        assert np.allclose([v['prior_xy'] for v in r['records']],[v['prior_xy'] for v in rr['records']],atol=1e-12)
        for j,(path,segment) in enumerate(zip(r['paths'],d['segments'])):
            xy=np.asarray(path['xy']);tt=segment['time'][:,None]
            assert np.array_equal(xy[0],np.asarray(d['fixes'][j]['xy']))
            expected=np.asarray(d['fixes'][j]['xy'])+segment['dp']-.5*tt**2*np.asarray(path['bias'])
            assert np.allclose(xy,expected,rtol=0,atol=1e-12)
            assert np.array_equal(path['velocity'][0],[0.,0.])
        d['validation']['segment_start_max_difference_m']=float(max(np.linalg.norm(np.asarray(p['xy'][0])-f['xy']) for p,f in zip(r['paths'],d['fixes'])))
    checks['no_future_observation_leakage']='passed'
    checks['all_segment_origins_equal_uwb']='passed'
    checks['uwb_plus_corrected_relative_displacement']='passed'
    config_out=copy.deepcopy(settings);config_out['ekf']=chosen
    try:config_out['output']=out.resolve().relative_to(ROOT).as_posix()
    except ValueError:config_out['output']=str(out)
    (out/'selected_config.yaml').write_text(yaml.safe_dump(config_out,allow_unicode=True,sort_keys=False),encoding='utf8')
    (out/'parameter_search.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf8')
    records=[]
    for d,a,b,c in zip(data,defaults,tuned,ablations):
        plots(d,a,b,c,out,args.sweep)
        records.append(dict(session=d['name'],input_validation=d['validation'],default=a,tuned=b,no_bias_learning=c))
        print(d['name'],'baseline',b['metrics']['baseline_all'],'EKF',b['metrics']['all'],'validation',b['metrics']['validation'],flush=True)
    selection='first half of each session; second half held out for parameter selection' if args.sweep else 'fixed config; no parameter search in this run'
    (out/'results.json').write_text(json.dumps(dict(checks=checks,selection=selection,sessions=records),ensure_ascii=False,indent=2),encoding='utf8')
    write_report(out,data,defaults,tuned,ablations,candidates,checks,args.sweep)
    with zipfile.ZipFile(out/'ekf_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png','.pdf','.txt','.yaml']:z.write(p,p.name)
    print('Output:',out,flush=True)


def write_report(out,data,defaults,tuned,ablations,candidates,checks,swept):
    lines=['水下导航原始IMU积分与UWB测距EKF离线回放','',
        '旧实验不是没有校正：位置直接设为接受UWB坐标，水平速度清零，但没有根据历史观测学习后续积分的加速度偏差。',
        '本模型状态[x,y,vx,vy,bx,by]，b为等效全局水平加速度误差，包含动态姿态补偿残差，不等同于传感器静态零偏。',
        '传播：p+=v*T+原始加速度双积分-0.5*b*T²；v+=原始加速度积分-b*T。',
        '四基站测距观测h_i=sqrt((x-Axi)²+(y-Ayi)²+(H+depth)²)，使用采集时刻之前已有压力深度。',
        '非线性测距观测使用迭代EKF线性化，Joseph形式更新协方差，NIS门限拒绝异常观测。',
        '位置策略已改为：全局位置=本次日志接受的UWB坐标+经EKF偏差补偿后的本航段相对积分位移。',
        '先用测距更新EKF偏差，再将位置严格重置为本次UWB坐标；保留偏差均值/方差，下一段水平速度清零。',
        '这是UWB位置重置与EKF偏差学习的组合；位置方差按测距局部几何重建，并近似清除与其他状态的旧交叉协方差，非纯EKF后验。',
        'NIS门限只决定是否学习该次偏差；日志已判有效的UWB坐标仍用于重置起点。重置后差值为0是构造结果，不代表真实误差为0。',
        '只有导航阶段积分；上浮/等待冻结位置均值，但加入未知运动的方差(sigma_pause*暂停时间)²，避免把未积分运动全当作IMU偏差。',
        '每段开始仍采用零速度先验，保留其不确定性；这是原任务假设，不是单帧UWB测得零速度。',
        '输入按原始采样时间积分约200Hz；无观测区间的线性状态传播/协方差采用等价闭式表达，不是降低IMU频率。',
        '偏差只在UWB更新时改变均值，并用于下一段，不回溯修改此前轨迹；无未来观测泄漏。',
        '评价使用下一次UWB更新前的预测差值，接受UWB坐标只是带误差参考；包含上浮阶段真实位移，不能当作真实定位精度。',
        '当NIS拒绝偏差更新时该次差值仍计入评分，不删除困难样本。',
        '初始标定、四元数、坐标轴沿用日志，没重估姿态；只分析X/Y，Z来自压力。',
        ('前半程调参、后半程评估；后半程仍正常用已经到达的UWB在线学习，不冻结学习。' if swept else '本次使用指定参数，前/后半程仅分别统计，是否独立验证取决于此前调参所用数据。'),
        '这只有两次旧实验，不是独立实机验证；模型还会混淆真正的初始速度、UWB误差和等效偏差。',
        '本脚本不连接硬件，不修改现有导航控制。','',
        '校验：'+json.dumps(checks,ensure_ascii=False),'',
        ('扫描参数组合数：'+str(len(candidates))) if swept else '本次未扫描参数，采用配置文件中的参数。',
        '选用参数：'+json.dumps(tuned[0]['config'],ensure_ascii=False,indent=2),'']
    for d,default,r,ablation in zip(data,defaults,tuned,ablations):
        lines+=[d['name'],'指标是更新前与接受UWB坐标的欧氏距离。',
            '方案 | 全程均值m | 全程中位数m | 全程最大m | 前半程均值m | 后半程均值m']
        for label,all_,train,val in [('原方法2',r['metrics']['baseline_all'],r['metrics']['baseline_train'],r['metrics']['baseline_validation']),
            ('预设EKF',default['metrics']['all'],default['metrics']['train'],default['metrics']['validation']),
            ('前半程选参EKF' if swept else '指定参数EKF',r['metrics']['all'],r['metrics']['train'],r['metrics']['validation']),
            ('同参EKF不学习偏差',ablation['metrics']['all'],ablation['metrics']['train'],ablation['metrics']['validation'])]:
            lines.append(f"{label} | {all_['mean']:.4f} | {all_['median']:.4f} | {all_['max']:.4f} | {train['mean']:.4f} | {val['mean']:.4f}")
        lines+=['更新编号 | 普通积分差m | EKF更新前差m | 起点重置后差m | 学到bx | 学到by | 接受偏差更新 | NIS']
        for v in r['records']:lines.append(f"{v['index']} | {v['baseline_difference_m']:.4f} | {v['pre_update_difference_m']:.4f} | {v['post_update_difference_m']:.4f} | {v['bias_after'][0]:+.5f} | {v['bias_after'][1]:+.5f} | {v['accepted']} | {v['nis']:.3f}")
        lines+=['','观测拒绝次数：预设EKF='+str(default['metrics']['rejected'])+'，选参EKF='+str(r['metrics']['rejected'])+'，不学习偏差='+str(ablation['metrics']['rejected'])+'。',
            '不学习偏差分支同时取消该状态的不确定性，可能更容易触发门限；其差异不完全来自均值校正。',
            '分割点：前'+str(r['metrics']['split'])+('次更新用于选参，其后仅用于评估。' if swept else '次更新为前半程统计。'),'']
    lines+=['运行方法（从项目根目录）：',
        'python scripts/analyze_underwater_ekf_20261030.py --sweep',
        'python scripts/analyze_underwater_ekf_20261030.py --config logs/underwater_navigation_20261026/ekf_replay_20261030/selected_config.yaml --output logs/underwater_navigation_20261026/ekf_manual_trial',
        'python scripts/analyze_underwater_ekf_20261030.py --self-test','',
        '调参入口：config/underwater_ekf_replay_20261030.yaml。--sweep只扫描sweep部分列出的值；不加--sweep则只运行ekf部分。',
        'range_sigma_m：测距噪声，越小越信UWB；initial_bias_sigma_mps2：偏差初始不确定性，越大越容易学习大偏差。',
        'bias_random_walk：越大越允许偏差随时间改变；accel_noise_density：未建模加速度的不确定性。',
        'pause_velocity_sigma_mps：越大越承认上浮时存在未知移动，降低把这段移动误学为偏差的倾向。',
        'segment_velocity_sigma_mps：零速度假设的不确定性，不应因为希望更好的结果而不合理地设为0。',
        'accel_tau_s：0为不低通；非零时导航阶段用两级低通，段首清零；改变此项后要重跑。',
        'innovation_gate：四维测距NIS拒绝门限；调低可能拒绝合法的大创新，必须一起检查拒绝次数。','',
        '方法参考：MathWorks Extended Kalman Filters（非线性测量、Jacobian、预测/更新）：',
        'https://www.mathworks.com/help/fusion/ug/extended-kalman-filters.html']
    (out/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
