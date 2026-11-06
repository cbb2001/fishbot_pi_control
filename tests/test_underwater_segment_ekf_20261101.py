"""分段积分、五帧平均与 EKF 的数学/阶段边界回归，不访问硬件。"""
from dataclasses import replace
import math

import numpy as np
import pytest
import yaml

from control.runtime.surface_navigation_20261010 import NavigationConfig, UWBFrame
from control.runtime.underwater_navigation_20261026 import (
    AcceptedUWB, UnderwaterConfig, UWBObservation, NavigationEstimator,
)
from control.runtime.underwater_ekf_20261101 import EKFConfig, NavigationEKF, range_model, transition


def config(**kw):
    return UnderwaterConfig(NavigationConfig(L=10,W=6,H=1,x_d=8,y_d=4),**kw)


def observation(i,x=3.,y=2.,stamp=None):
    ranges=tuple(math.sqrt((x-a)**2+(y-b)**2+1) for a,b in ((0,0),(10,0),(10,6),(0,6)))
    return UWBObservation(i*100_000_000 if stamp is None else stamp,UWBFrame(ranges,str(i)),.8)


def imu(ax=0.,bias=(0.,0.,0.)):
    return dict(acc_mps2=(np.array([ax,0.,9.80665])+bias).tolist(),quat=[1,0,0,0],tid=None)


def fix_at(x=3.,y=2.,stamp=0):
    gate=AcceptedUWB(config(uwb_average_frames=1))
    obs=observation(1,x,y,stamp)
    return gate.add(obs,stamp)


def test_five_complete_distinct_frames_are_averaged_before_position_solution():
    gate=AcceptedUWB(config())
    rows=[observation(i,x=3+i*.01) for i in range(1,6)]
    for obs in rows[:4]:
        assert gate.add(obs,obs.t_ns) is None
        assert gate.add(obs,obs.t_ns) is None  # 重复报文不增加计数。
    bad=UWBObservation(450_000_000,UWBFrame((1,None,3,4),'partial'),None)
    assert gate.add(bad,bad.t_ns) is None
    fix=gate.add(rows[4],rows[4].t_ns)
    ranges=np.mean([o.frame.ranges_m for o in rows],axis=0)
    a,b,c,d=ranges**2
    assert fix.ranges_m==pytest.approx(ranges)
    assert fix.xy==pytest.approx([(a-b+d-c+200)/40,(a-d+b-c+72)/24])
    assert fix.quality['frames']==5 and fix.quality['surface_depth_m']==0
    assert fix.quality['frame_ids']==[str(i) for i in range(1,6)]
    assert gate.add(observation(6),600_000_000) is None  # 同一组只发布一次。


def test_batch_reset_discards_old_samples_and_rejects_jump_and_stale():
    gate=AcceptedUWB(config(uwb_average_frames=2))
    assert gate.add(observation(1),100_000_000) is None
    gate.reset(150_000_000)
    assert gate.add(observation(2),200_000_000) is None
    reference=gate.add(observation(3),300_000_000); gate.commit(reference)
    gate.reset(350_000_000)
    bad=observation(4)
    bad=replace(bad,frame=UWBFrame(tuple(v+8 for v in bad.frame.ranges_m),'jump'))
    assert gate.add(bad,400_000_000) is None and gate.reason=='range_jump'
    assert gate.add(observation(5),900_000_000) is None and gate.reason=='stale'
    assert gate.add(observation(6),600_000_000) is None
    fix=gate.add(observation(7),700_000_000)
    assert fix.quality['frames']==2 and gate.last_accepted is reference


def test_same_uart_timestamp_distinct_frames_count_but_old_ids_do_not():
    gate=AcceptedUWB(config())
    fixes=[gate.add(observation(i,stamp=100_000_000),100_000_000) for i in range(1,6)]
    assert fixes[:4]==[None]*4 and fixes[4] is not None
    gate.commit(fixes[4]); gate.reset(100_000_000)
    assert gate.add(observation(1,stamp=200_000_000),200_000_000) is None


def test_only_navigation_integrates_reset_drops_surface_samples_and_keeps_calibration():
    bias=np.array([.1,.02,-.03]); est=NavigationEstimator(config(accel_tau_s=0),bias=bias)
    for i in range(101): est.imu(i*5_000_000,imu(10,bias))
    assert est.xyz==pytest.approx([0,0,0]) and est.v==pytest.approx([0,0,0])
    est.apply_surface_fix(fix_at(stamp=500_000_000),500_000_000)
    est.start_navigation(600_000_000,1_600_000_000)
    for i in range(101,321): est.imu(i*5_000_000,imu(1,bias))
    assert est.xyz[:2]==pytest.approx([3.5,2],abs=.006)
    assert not est.integrating
    frozen=np.array(est.xyz)
    for i in range(321,501): est.imu(i*5_000_000,imu(100,bias))
    assert est.xyz==pytest.approx(frozen)
    est.pressure(2_500_000_000,.4)
    est.apply_surface_fix(fix_at(4,3,2_500_000_000),2_500_000_000)
    assert est.xyz==pytest.approx([4,3,0])
    assert est.v==pytest.approx([0,0,0])
    assert est.a1==pytest.approx([0,0,0]) and est.a2==pytest.approx([0,0,0])
    assert est.relative_xy==pytest.approx([0,0]) and est.last_integrated_ns is None
    assert est.bias==pytest.approx(bias) and est.last_imu==2_500_000_000
    assert not hasattr(est,'history')  # 新估计器不保留可回放的上浮/等待积分历史。


def test_new_segment_cannot_bridge_pause_and_filter_starts_from_zero():
    est=NavigationEstimator(config())
    est.apply_surface_fix(fix_at(),0)
    for i in range(201): est.imu(i*5_000_000,imu(100))
    est.start_navigation(1_000_000_000,2_000_000_000)
    est.imu(1_005_000_000,imu())
    assert est.xyz==pytest.approx([3,2,0]) and est.v==pytest.approx([0,0,0])
    assert est.a1==pytest.approx([0,0,0]) and est.a2==pytest.approx([0,0,0])


def test_nonlinear_range_jacobian_covariance_and_bias_learning():
    cfg=EKFConfig(); ekf=NavigationEKF(cfg,10,6,1)
    x=np.array([3.,2.,.2,.1,.2,-.1,.03,-.02,.05]); predicted,H=range_model(x,ekf.anchors,1.)
    for j in range(9):
        dx=np.zeros(9); dx[j]=1e-5
        numeric=(range_model(x+dx,ekf.anchors,1.)[0]-range_model(x-dx,ekf.anchors,1.)[0])/2e-5
        assert H[:,j]==pytest.approx(numeric,abs=1e-8)
    F,Q=transition(.1,cfg.accel_noise_density,cfg.bias_random_walk)
    F2,Q2=transition(.2,cfg.accel_noise_density,cfg.bias_random_walk)
    assert F@F==pytest.approx(F2)
    assert F@Q@F.T+Q==pytest.approx(Q2)
    obs=fix_at(); ekf.surface_fix(0,obs.xy,obs.ranges_m)
    for cycle in range(12):
        ekf.resume()
        for _ in range(1000): ekf.predict(.01,np.array([.04,-.025,.02]))
        info=ekf.surface_fix((cycle+1)*10_000_000_000,obs.xy,obs.ranges_m)
        assert info['accepted']
        assert ekf.x[:6]==pytest.approx([3,2,0,0,0,0])
        assert np.linalg.eigvalsh(ekf.P).min()>-1e-10
    assert ekf.x[6:]==pytest.approx([.04,-.025,.02],abs=.01)


def test_pause_inflates_uncertainty_without_motion_and_outlier_still_resets_anchor():
    ekf=NavigationEKF(EKFConfig(innovation_gate=.00001),10,6,1)
    f=fix_at(); ekf.surface_fix(0,f.xy,f.ranges_m)
    ekf.resume()
    ekf.predict(1,np.array([1.,0.,0.])); before=ekf.x.copy(); cov=ekf.P.copy()
    ekf.pause(1_000_000_000)
    other=fix_at(7,4,3_000_000_000)
    info=ekf.surface_fix(other.t_ns,other.xy,other.ranges_m)
    assert info['accepted'] is False and info['pause_s']==2
    assert info['prior_position_variance'][0]>cov[0,0]
    assert ekf.x[:6]==pytest.approx([7,4,0,0,0,0])
    assert ekf.x[6:]==pytest.approx(before[6:])


def test_yaml_ekf_parameters_are_strict_and_match_logged_baseline():
    with open('config/robot.yaml',encoding='utf8') as f: robot=yaml.safe_load(f)
    cfg=UnderwaterConfig.from_robot(robot)
    assert cfg.uwb_average_frames==5 and cfg.ekf.range_sigma_m==.3
    robot['underwater_navigation_20261026']['ekf']['range_sigma_m']=.8
    assert UnderwaterConfig.from_robot(robot).ekf.range_sigma_m==.8
    del robot['underwater_navigation_20261026']['ekf']['iterations']
    with pytest.raises(ValueError,match='iterations'): UnderwaterConfig.from_robot(robot)
    for count in (0,2.5,True):
        with pytest.raises(ValueError): config(uwb_average_frames=count)
    with pytest.raises(ValueError): EKFConfig(range_sigma_m=0)


@pytest.mark.parametrize('enabled,retain',[(False,True),(True,False)])
def test_optional_bias_reset_does_not_change_startup_calibration(enabled,retain):
    bias=np.array([.1,.02,.03])
    c=config(ekf=EKFConfig(enabled=enabled,retain_bias_on_fix=retain))
    est=NavigationEstimator(c,bias=bias,yaw_zero_deg=30.)
    alignment=est.pool.copy()
    est.ekf.x[6:]=[.1,-.2,.3]
    est.apply_surface_fix(fix_at(),0)
    assert est.ekf.x[6:]==pytest.approx([0,0,0])
    assert est.bias==pytest.approx(bias) and est.pool==pytest.approx(alignment)
    assert est.v==pytest.approx([0,0,0])


def test_new_estimator_still_checks_health_and_tid_when_paused_or_reset():
    est=NavigationEstimator(config())
    est.imu(0,{**imu(),'tid':10})
    est.apply_surface_fix(fix_at(),0)
    assert est.imu(5_000_000,{**imu(100),'tid':10}) is False
    assert est.imu(10_000_000,{**imu(100),'tid':9}) is False
    with pytest.raises(ValueError,match='gap'):
        est.imu(100_000_000,{**imu(),'tid':11})
    assert est.xyz==pytest.approx([3,2,0])


def test_pressure_corrects_vertical_velocity_and_bias_with_down_positive_convention():
    est=NavigationEstimator(config(accel_tau_s=0))
    est.apply_surface_fix(fix_at(),0); est.imu(0,imu())
    est.start_navigation(0,21_000_000_000)
    # 真实静止，去重力后的Z向上加速度残差为-0.05，EKF应学到bz=+0.05。
    data=imu(); data['acc_mps2'][2]-=.05
    for i in range(1,4001):
        t=i*5_000_000
        est.imu(t,data)
        if i%20==0: est.pressure(t,0.)
    assert abs(est.xyz[2])<.01 and abs(est.v[2])<.015
    assert est.ekf.x[8]==pytest.approx(.05,abs=.008)
    assert est.ekf.last_pressure_update['accepted']
    assert np.linalg.eigvalsh(est.ekf.P).min()>-1e-10
    est.pause_navigation(20_000_000_000)
    learned=est.ekf.x[6:].copy()
    est.apply_surface_fix(fix_at(4,3,20_000_000_000),20_000_000_000)
    assert est.xyz==pytest.approx([4,3,0]) and est.v==pytest.approx([0,0,0])
    assert est.relative_xyz==pytest.approx([0,0,0])
    assert abs(est.ekf.x[8])>.03  # 学习保留，积分初始状态清零。


def test_pressure_outlier_does_not_overwrite_state_or_lose_covariance_symmetry():
    ekf=NavigationEKF(EKFConfig(),10,6,1)
    f=fix_at(); ekf.surface_fix(0,f.xy,f.ranges_m)
    before=ekf.x.copy(); cov=ekf.P.copy()
    info=ekf.update_depth(10.)
    assert not info['accepted']
    assert ekf.x==pytest.approx(before) and ekf.P==pytest.approx(cov)
    assert ekf.update_depth(.001)['accepted']
    assert ekf.P==pytest.approx(ekf.P.T)


def test_interleaved_pressure_and_imu_predict_time_once():
    est=NavigationEstimator(config(accel_tau_s=0,ekf=EKFConfig(enabled=False)))
    est.apply_surface_fix(fix_at(),0); est.imu(0,imu(1))
    est.start_navigation(0,1_000_000_000)
    est.imu(5_000_000,imu(1))
    est.pressure(7_500_000,0.)
    est.imu(10_000_000,imu(1))
    assert est.v[0]==pytest.approx(.01)
    assert est.xyz[0]==pytest.approx(3+.5*.01**2)
    assert not est.pressure(7_500_000,1.)
