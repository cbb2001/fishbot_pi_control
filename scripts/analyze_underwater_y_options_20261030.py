#!/usr/bin/env python3
"""Offline, causal comparison of global/body bias tracking and acceleration LPF.

All navigation segments restart at accepted UWB XY with zero velocity. Only
past fixes may affect a segment. Body model estimates two body-horizontal
effective acceleration residuals; it does not re-estimate attitude or body Z.
"""
from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
import argparse
import numpy as np
import yaml
import analyze_underwater_ekf_20261030 as base

ROOT=base.ROOT


def cumulative(t,values):
    dt=np.diff(t).reshape((-1,)+(1,)*(values.ndim-1))
    return np.concatenate([np.zeros_like(values[:1]),np.cumsum((values[:-1]+values[1:])*dt/2,axis=0)])


def lowpass(t,a,tau):
    if tau==0:return a.copy()
    y=np.zeros_like(a);first=np.zeros(a.shape[1:]);second=first.copy()
    for k,dt in enumerate(np.diff(t),1):
        gain=-np.expm1(-dt/tau);first+=gain*(a[k]-first);second+=gain*(first-second);y[k]=second
    return y


def prepare(folder):
    d=base.load_session(folder,0.)
    raw=base.read_jsonl(folder/'raw_imu.jsonl');rows=base.read_jsonl(folder/'imu_filtered.jsonl')
    cal=json.loads((folder/'calibration.json').read_text(encoding='utf8'))
    angle=np.deg2rad(-cal['yaw_zero_deg']);pool=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    bytime={r['t_ns']:r['data'] for r in raw}
    times=np.array([(r['sample_t_ns']-d['t0'])/1e9 for r in rows])
    acc=np.array([r['raw_linear_acc'][:2] for r in rows])
    rotations=np.array([(pool@base.rotation(bytime[r['sample_t_ns']]['quat']))[:2,:2] for r in rows])
    for s in d['segments']:
        t=s['time']+s['start_s']
        s['acc']=np.column_stack([np.interp(t,times,acc[:,k]) for k in range(2)])
        s['body_projection']=np.stack([np.interp(t,times,rotations[:,i,j]) for i in range(2) for j in range(2)],axis=1).reshape(-1,2,2)
    return d


def inputs(d,model,tau):
    """Integrate acceleration and bias sensitivity at the original IMU timestamps."""
    prepared=[]
    for s in d['segments']:
        t=s['time'];T=s['duration'];a=lowpass(t,s['acc'],tau)
        dv,dp=base.integrate(t,a)
        if model=='global':
            Bv=t[:,None,None]*np.eye(2);Bp=.5*t[:,None,None]**2*np.eye(2)
            F,Qb=base.transition(T,0.,1.)
        else:
            # Filter the bias sensitivity with the same causal filter as acceleration.
            # Thus a constant body bias has exactly the modelled filtered effect.
            R=lowpass(t,s['body_projection'],tau)
            Bv=cumulative(t,R);Bp=cumulative(t,Bv)
            F=np.eye(6);F[:2,2:4]=T*np.eye(2);F[:2,4:6]=-Bp[-1];F[2:4,4:6]=-Bv[-1]
            # Continuous bias random-walk covariance, integrated through the time-
            # varying body projection. For LPF this is a quasi-static approximation;
            # filter states are not augmented stochastic states.
            remv=Bv[-1]-Bv;remp=Bp[-1]-Bp-(T-t)[:,None,None]*Bv
            G=np.concatenate([-remp,-remv,np.broadcast_to(np.eye(2),remv.shape)],axis=1)
            GG=np.einsum('nik,njk->nij',G,G)
            Qb=cumulative(t,GG)[-1]
        _,Qa=base.transition(T,1.,0.)
        prepared.append(dict(dv=dv,dp=dp,Bv=Bv,Bp=Bp,F=F,Qa=Qa,Qb=Qb))
    return prepared


def replay(d,prepared,cfg,traces=False):
    x=np.r_[d['fixes'][0]['xy'],np.zeros(4)]
    P=np.diag([cfg['initial_position_sigma_m']**2]*2+[cfg['segment_velocity_sigma_mps']**2]*2+[cfg['initial_bias_sigma_mps2']**2]*2)
    records=[];paths=[]
    for j,(s,u) in enumerate(zip(d['segments'],prepared)):
        P=base.paused(P,max(0,(s['start_ns']-d['fixes'][j]['fix_t_ns'])/1e9),cfg,True)
        x[2:4]=0.;P[2:4,:]=0.;P[:,2:4]=0.;P[2:4,2:4]=np.eye(2)*cfg['segment_velocity_sigma_mps']**2
        before=x.copy();assert np.array_equal(before[:2],d['fixes'][j]['xy'])
        x=u['F']@x;x[:2]+=u['dp'][-1];x[2:4]+=u['dv'][-1]
        P=u['F']@P@u['F'].T+cfg['accel_noise_density']**2*u['Qa']+cfg['bias_random_walk']**2*u['Qb']
        if traces:
            xy=before[:2]+u['dp']-np.einsum('nij,j->ni',u['Bp'],before[4:6])
            assert np.array_equal(xy[0],d['fixes'][j]['xy']) and np.allclose(xy[-1],x[:2])
            paths.append(dict(time_s=(s['time']+s['start_s']).tolist(),xy=xy.tolist(),bias=before[4:6].tolist()))
        if j+1==len(d['fixes']):break
        f=d['fixes'][j+1];pause=(f['fix_t_ns']-s['end_ns'])/1e9
        P=base.paused(P,pause,cfg,True);prior=x.copy()
        x,P,info=base.range_update(x,P,f['ranges_m'],d['anchors'],d['anchor_height']+f['depth_m'],cfg)
        x,P=base.anchor_to_uwb(x,P,f,d['anchors'],d['anchor_height']+f['depth_m'],cfg)
        error=prior[:2]-f['xy'];raw_error=np.array(d['fixes'][j]['xy'])+s['raw_dp'][-1]-f['xy']
        records.append(dict(index=j+1,prior_xy=prior[:2].tolist(),error_xy=error.tolist(),raw_error_xy=raw_error.tolist(),
            bias_before=before[4:6].tolist(),bias_after=x[4:6].tolist(),pause_s=pause,**info))
    split=max(2,len(records)//2)
    def metric(rr):
        e=np.array([v['error_xy'] for v in rr]);length=np.linalg.norm(e,axis=1)
        return dict(xy_mean=float(length.mean()),xy_median=float(np.median(length)),y_mae=float(abs(e[:,1]).mean()),
            y_rmse=float(np.sqrt(np.mean(e[:,1]**2))),x_mae=float(abs(e[:,0]).mean()),rejected=sum(not r['accepted'] for r in rr))
    return dict(records=records,paths=paths,split=split,all=metric(records),train=metric(records[:split]),validation=metric(records[split:]))


def test_body_model(cfg):
    anchors=np.array([[0.,0.],[50.,0.],[50.,26.5],[0.,26.5]])
    truth=np.array([.04,-.025]);xy=np.array([4.,7.]);segments=[];fixes=[]
    for j in range(10):
        t=np.linspace(0,10,1001);angle=.15*j+.02*t
        R=np.zeros((len(t),2,2));R[:,0,0]=R[:,1,1]=np.cos(angle);R[:,1,0]=np.sin(angle);R[:,0,1]=-np.sin(angle)
        a=np.einsum('nij,j->ni',R,truth);dv,dp=base.integrate(t,a)
        segments.append(dict(time=t,duration=10.,start_s=j*10.,start_ns=j*10**10,end_ns=(j+1)*10**10,
            acc=a,body_projection=R,raw_dp=dp,raw_dv=dv))
        fixes.append(dict(xy=xy.tolist(),fix_t_ns=j*10**10,depth_m=0.,ranges_m=base.h_range(np.r_[xy,np.zeros(4)],anchors,1.9)[0].tolist()))
    d=dict(segments=segments,fixes=fixes,anchors=anchors,anchor_height=1.9,t0=0)
    c={**cfg,'range_sigma_m':.05,'segment_velocity_sigma_mps':.005,'initial_bias_sigma_mps2':.1,
        'bias_random_walk':.00001,'accel_noise_density':.001,'innovation_gate':1e6}
    for tau in [0.,.2]:
        r=replay(d,inputs(d,'body',tau),c,True)
        estimated=np.array(r['records'][-1]['bias_after'])
        assert np.linalg.norm(estimated-truth)<.002,(tau,estimated)
    return 'known rotating body bias recovered; tau=0 and 0.2 s'


def artifacts(out,data,results,candidates,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    styles={'current':('当前EKF','#777777'),'fast_bias':('只加快偏差跟踪','#dd842d'),
        'lowpass_0_2':('当前EKF＋低通0.2s','#9972b2'),'selected_global':('组合候选（全局偏差）','#007e95'),
        'selected_body':('组合候选（机身偏差）','#3b8f43')}
    lines=['Y向漂移改进探索：严格UWB起点，导航段积分，上浮/等待不积分。',
        '当前结果与之前脚本数值一致。240组候选仅用两次实验各自前半程的XY差值评分，等权归一化。',
        '这些日志已经反复分析过；后半程只是回放检查，不是新的独立验证集。',
        '全部指标均为使用下一次UWB之前的差值，包含暂停期间真实位移及UWB误差，不是真实轨迹误差。',
        '全局模型估计全局等效偏差；机身模型估计机身X/Y等效偏差并按逐样本姿态旋转，不估计姿态误差或机身Z偏差。',
        '机身模型滤波后随机游走协方差采用准静态近似，没有扩充低通随机状态。',
        '加快偏差跟踪：仅将bias_random_walk从0.003改为0.03，其余参数不变。',
        '低通0.2s：仅增加两级因果低通，各段清零；它会引入延迟，不能消除持续加速度偏差。',
        '全局/机身组合候选分别按前半程选参，不按最后几个点挑选。低通选到扫描边界0.5s，不证明更强滤波更准确。',
        '前半程综合评分第一是机身偏差模型，但两次后半程效果不稳定，不能推荐直接替换。','']
    for i,d in enumerate(data):
        name=d['name'][:15];n=len(results['current']['sessions'][i]['records']);xx=np.arange(1,n+1)
        fig,ax=plt.subplots(2,2,figsize=(16,10))
        fig.suptitle(name+'：Y向漂移改进试算（无未来观测；每段严格从UWB坐标出发）',fontsize=14)
        for key in ['current','lowpass_0_2','selected_global']:
            label,color=styles[key]
            for j,p in enumerate(results[key]['sessions'][i]['paths']):
                ax[0,0].plot(p['time_s'],np.asarray(p['xy'])[:,1],c=color,lw=1.3,label=label if j==0 else None)
        ft=[max(0,(f['fix_t_ns']-d['t0'])/1e9) for f in d['fixes']]
        ax[0,0].scatter(ft,[f['xy'][1] for f in d['fixes']],s=28,c='black',label='接受UWB位置',zorder=5)
        ax[0,0].set(xlabel='首次导航后 / s',ylabel='全局Y / m',title='下一段起点重置；上浮/等待期间断开')
        for key in ['current','fast_bias','lowpass_0_2','selected_global']:
            label,color=styles[key];r=results[key]['sessions'][i]
            ax[0,1].plot(xx,[v['error_xy'][1] for v in r['records']],'o-',c=color,label=label)
        ax[0,1].axhline(0,c='black',lw=.7);ax[0,1].set(xlabel='第几次重新定位',ylabel='预测Y − 下一次UWB Y / m',title='更新前Y差值：越接近0越好；不是实测真值')
        rr=results['current']['sessions'][i]['records']
        needed=[2*v['raw_error_xy'][1]/d['segments'][j]['duration']**2 for j,v in enumerate(rr)]
        ax[1,0].plot(xx,needed,'o--',c='black',label='用下一次UWB反推（仅诊断，不用于预测）')
        ax[1,0].plot(xx,[v['bias_before'][1] for v in rr],'o-',c=styles['current'][1],label='当前EKF航段开始时的by')
        ax[1,0].set(xlabel='导航航段编号',ylabel='等效Y加速度偏差 / m/s²',title='参考偏差包含上浮移动和UWB误差，不能当作真实零偏')
        labels=list(styles);width=.36;bx=np.arange(len(labels))
        ax[1,1].bar(bx-width/2,[results[k]['sessions'][i]['validation']['x_mae'] for k in labels],width,label='后半程X绝对差均值')
        ax[1,1].bar(bx+width/2,[results[k]['sessions'][i]['validation']['y_mae'] for k in labels],width,label='后半程Y绝对差均值')
        ax[1,1].set_xticks(bx,['当前','加快跟踪','低通0.2s','组合/全局','组合/机身']);ax[1,1].set(ylabel='m',title='同时检查X和Y；避免只改善最后几个Y点')
        for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/(name+'_y_comparison.png'),dpi=170);fig.savefig(out/(name+'_y_comparison.pdf'));plt.close(fig)
        lines += [d['name'],'方案 | 全程XY均值 | 全程Y绝对差均值 | 后半程XY均值 | 后半程Y绝对差均值 | 拒绝次数']
        for key in styles:
            r=results[key]['sessions'][i];lines.append(f"{styles[key][0]} | {r['all']['xy_mean']:.4f} | {r['all']['y_mae']:.4f} | {r['validation']['xy_mean']:.4f} | {r['validation']['y_mae']:.4f} | {r['all']['rejected']}")
        lines+=['航段 | 原始积分ΔY | 相邻UWB ΔY | 航段开始by | 事后参考by | 当前Y差值 | 组合全局Y差值']
        for j,r in enumerate(rr):
            rawdy=d['segments'][j]['raw_dp'][-1,1];uwbdy=d['fixes'][j+1]['xy'][1]-d['fixes'][j]['xy'][1]
            lines.append(f"{j+1} | {rawdy:.4f} | {uwbdy:.4f} | {r['bias_before'][1]:.5f} | {needed[j]:.5f} | {r['error_xy'][1]:.4f} | {results['selected_global']['sessions'][i]['records'][j]['error_xy'][1]:.4f}")
        lines+=['']
    lines+=['各方案参数：']
    for key in styles:lines += [key,json.dumps(results[key]['config'],ensure_ascii=False)]
    lines+=['','当前不替换原分析结果，不改实机导航。全局组合候选配置可交给原分析脚本回放；不是已经验证的最佳参数。',
        'python scripts/analyze_underwater_ekf_20261030.py --config logs/underwater_navigation_20261026/ekf_y_options_20261030/global_candidate.yaml --output logs/underwater_navigation_20261026/ekf_y_candidate_trial']
    (out/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    exported={**settings,'ekf':results['selected_global']['config'],'output':'logs/underwater_navigation_20261026/ekf_y_candidate_trial'}
    (out/'global_candidate.yaml').write_text(yaml.safe_dump(exported,allow_unicode=True,sort_keys=False),encoding='utf8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'logs/underwater_navigation_20261026/ekf_replay_20261030/selected_config.yaml')
    ap.add_argument('--output',type=Path,default=ROOT/'logs/underwater_navigation_20261026/ekf_y_options_20261030')
    args=ap.parse_args();settings=yaml.safe_load(args.config.read_text(encoding='utf8'));cfg=settings['ekf']
    body_check=test_body_model(cfg)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=[prepare(ROOT/p) for p in settings['sessions']]
    cache={}
    def get(model,tau):
        if (model,tau) not in cache:cache[model,tau]=[inputs(d,model,tau) for d in data]
        return cache[model,tau]
    current=[replay(d,u,cfg,True) for d,u in zip(data,get('global',cfg['accel_tau_s']))]
    # The new replay must reproduce the previous implementation before comparison.
    for d,r in zip(data,current):
        old=base.replay(d,cfg,True,True)
        assert np.allclose([x['prior_xy'] for x in r['records']],[x['prior_xy'] for x in old['records']],atol=1e-10)
    candidates=[]
    for model,tau in itertools.product(['global','body'],[0.,.075,.2,.5]):
        prepared=get(model,tau)
        for rw,pause,initial in itertools.product([.001,.003,.01,.03,.06],[.1,.3,.6],[.03,.1]):
            c={**cfg,'accel_tau_s':tau,'bias_random_walk':rw,'pause_velocity_sigma_mps':pause,'initial_bias_sigma_mps2':initial}
            runs=[replay(d,u,c) for d,u in zip(data,prepared)]
            # Equal session weights, first halves only. Keep X in the objective;
            # cannot choose a curve just because its Y looks flatter.
            score=float(np.mean([r['train']['xy_mean']/b['train']['xy_mean']+.1*r['train']['rejected']/r['split'] for r,b in zip(runs,current)]))
            candidates.append(dict(model=model,config=c,score=score,metrics=[{k:r[k] for k in ['all','train','validation']} for r in runs]))
        print('Completed',model,'tau',tau,flush=True)
    candidates.sort(key=lambda c:c['score'])
    selected=candidates[0]
    named={'current':dict(model='global',config=cfg)}
    named['fast_bias']={'model':'global','config':{**cfg,'bias_random_walk':.03}}
    named['lowpass_0_2']={'model':'global','config':{**cfg,'accel_tau_s':.2}}
    named['selected_global']=next(c for c in candidates if c['model']=='global')
    named['selected_body']=next(c for c in candidates if c['model']=='body')
    results={}
    for key,c in named.items():
        results[key]=dict(model=c['model'],config=c['config'],sessions=[replay(d,u,c['config'],True) for d,u in zip(data,get(c['model'],c['config']['accel_tau_s']))])
    # Check causality by perturbing only the last fix. All predictions before
    # that observation must remain unchanged, for both coordinate models.
    for key in ['selected_global','selected_body']:
        c=named[key]
        for d,u,r in zip(data,get(c['model'],c['config']['accel_tau_s']),results[key]['sessions']):
            changed=copy.deepcopy(d);changed['fixes'][-1]['ranges_m']=[v+2 for v in changed['fixes'][-1]['ranges_m']]
            rr=replay(changed,u,c['config'])
            assert np.allclose([v['prior_xy'] for v in r['records']],[v['prior_xy'] for v in rr['records']],atol=1e-12)
    (out/'parameter_search.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf8')
    payload=dict(selection='first halves only; exploratory reuse of these two existing logs, not independent validation',
        chosen_model=selected['model'],checks=['original replay reproduced','all origins equal UWB','no future observations',body_check],
        sessions=[d['name'] for d in data],variants=results)
    (out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    artifacts(out,data,results,candidates,settings)
    for key,r in results.items():
        print(key,r['model'],r['config'],flush=True)
        for d,s in zip(data,r['sessions']):print(d['name'],s['all'],s['validation'],flush=True)
    print('Selected on early data:',selected['model'],selected['score'],flush=True)


if __name__=='__main__':main()
