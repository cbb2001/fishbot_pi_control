import math
from dataclasses import replace

import numpy as np
import pytest

from control.runtime.underwater_navigation_20261026 import (
    UnderwaterConfig, UnderwaterScheduler, InertialEstimator, StableUWB, UWBObservation,
    AcceptedUWB, SurfaceFinPolicy,
)
from control.runtime.surface_navigation_20261010 import NavigationConfig, UWBFrame, NavigationTailAction
from control.runtime.fixed_dive_20260920_100122 import ContinuousFinAction
from control.runtime.rl_actions_20260914 import build_calibration
import yaml


def config(**kwargs):
    return UnderwaterConfig(NavigationConfig(L=10, W=6, H=1, x_d=8, y_d=4), **kwargs)


def imu(ax=0., yaw=0.):
    angle = math.radians(yaw)/2
    return dict(acc_mps2=[ax, 0, 9.80665], quat=[math.cos(angle), 0, 0, math.sin(angle)],
                yaw_deg=yaw, pitch_deg=0., roll_deg=0., gyro_radps=[0,0,0])


def observation(i, x=3., y=2., depth=0., bad=0.):
    ranges = tuple(math.sqrt((x-a)**2+(y-b)**2+(1+depth)**2)+(bad if k==2 else 0)
                   for k,(a,b) in enumerate(((0,0),(10,0),(10,6),(0,6))))
    return UWBObservation(i*100_000_000, UWBFrame(ranges, str(i)), depth)


def test_static_bias_and_pressure_constrain_vertical_position():
    est = InertialEstimator(config(), bias=[.1,0,0])
    for i in range(401):
        est.imu(i*5_000_000, imu(.1))
        if i%20==0: est.pressure(i*5_000_000, .3)
    assert est.xyz == pytest.approx([0,0,.3])
    assert est.v == pytest.approx([0,0,0])


def test_acceleration_rotates_into_same_pool_frame_as_heading():
    est = InertialEstimator(config(accel_tau_s=0), yaw_zero_deg=90)
    for i in range(201): est.imu(i*5_000_000, imu(1., 90))
    assert est.xyz == pytest.approx([.5,0,0], abs=1e-9)
    assert est.v[:2] == pytest.approx([1,0], abs=1e-9)


def test_duplicate_samples_do_not_integrate_and_large_gap_is_rejected():
    est=InertialEstimator(config())
    est.imu(0, imu())
    assert est.imu(0, imu(100)) is False
    with pytest.raises(ValueError, match='IMU.*gap'): est.imu(60_000_000, imu())
    assert est.xyz == [0.,0.,0.]


def test_delayed_fix_replays_filter_checkpoints_without_double_filtering():
    c=config()
    a,b=InertialEstimator(c),InertialEstimator(c)
    for i in range(101):
        for e in (a,b): e.imu(i*5_000_000, imu(math.sin(i/20)))
    b.correct(500_000_000, [4,3], [.2,-.1])
    for i in range(101,151):
        for e in (a,b): e.imu(i*5_000_000, imu(math.sin(i/20)))
    a.correct(500_000_000, [4,3], [.2,-.1])
    assert a.xyz == pytest.approx(b.xyz)
    assert a.v == pytest.approx(b.v)
    assert a.filtered_acc == pytest.approx(b.filtered_acc)
    with pytest.raises(ValueError, match='history'): a.correct(-1, [1,1], [0,0])


def test_fix_clears_both_acceleration_stages_without_carrying_surface_motion():
    bias=np.array([.1,.02,-.03])
    est=InertialEstimator(config(),bias=bias)
    moving={**imu(), 'acc_mps2':[.7,-.2,10.2]}
    for i in range(101):
        est.imu(i*5_000_000,moving)
        if i%20==0: est.pressure(i*5_000_000,.2+i*.001)
    assert np.linalg.norm(est.a1)>0 and np.linalg.norm(est.a2)>0
    depth_state=(est.xyz[2],est.v[2],est.depth,est.last_depth)
    raw_acc=est.raw_acc.copy()
    est.correct(500_000_000,[4,3],[0,0])
    assert est.a1==pytest.approx([0,0,0])
    assert est.a2==pytest.approx([0,0,0])
    assert est.xyz[:2]==pytest.approx([4,3])
    assert est.v[:2]==pytest.approx([0,0])
    assert (est.xyz[2],est.v[2],est.depth,est.last_depth)==depth_state
    assert est.bias==pytest.approx(bias)
    assert est.raw_acc==pytest.approx(raw_acc)
    assert len(est.history)==1
    stationary={**imu(),'acc_mps2':(bias+[0,0,9.80665]).tolist()}
    for i in range(101,141): est.imu(i*5_000_000,stationary)
    assert est.xyz[:2]==pytest.approx([4,3])
    assert est.v[:2]==pytest.approx([0,0],abs=1e-12)
    assert est.a1==pytest.approx([0,0,0],abs=1e-12)
    assert est.a2==pytest.approx([0,0,0],abs=1e-12)


@pytest.mark.parametrize('reset_filter',[True,False])
def test_delayed_between_sample_fix_matches_timely_fix_and_repeated_correction(reset_filter):
    a,b=InertialEstimator(config()),InertialEstimator(config())
    for i in range(201):
        for est in (a,b):
            est.imu(i*5_000_000,imu(math.sin(i/20)))
            if i%20==0: est.pressure(i*5_000_000,.2+i*.001)
        if i==100:
            # 校正位于两帧 IMU 之间；a 在后续数据处理完后才收到这帧定位。
            b.correct(502_000_000,[2,3],[0,0],reset_accel_filter=reset_filter)
    a.correct(502_000_000,[2,3],[0,0],reset_accel_filter=reset_filter)
    assert a.xyz==pytest.approx(b.xyz)
    assert a.v==pytest.approx(b.v)
    assert a.a1==pytest.approx(b.a1)
    assert a.a2==pytest.approx(b.a2)
    assert a.last_depth==b.last_depth
    # 再次定位，回放必须使用含前次滤波重置的检查点。
    a.correct(802_000_000,[5,6],[0,0],reset_accel_filter=reset_filter)
    b.correct(802_000_000,[5,6],[0,0],reset_accel_filter=reset_filter)
    assert a.xyz==pytest.approx(b.xyz)
    assert a.v==pytest.approx(b.v)
    assert a.a2==pytest.approx(b.a2)


def test_diagnostic_fix_can_preserve_continuous_acceleration_filter():
    est=InertialEstimator(config())
    for i in range(101): est.imu(i*5_000_000,imu(.5))
    a1,a2=est.a1.copy(),est.a2.copy()
    est.correct(500_000_000,[2,3],[0,0],reset_accel_filter=False)
    assert est.a1==pytest.approx(a1)
    assert est.a2==pytest.approx(a2)
    assert est.v[:2]==pytest.approx([0,0])


def test_stable_window_fits_slow_drift_and_requires_new_frame_evidence():
    gate=StableUWB(config())
    fix=None
    for i in range(1,31):
        obs=observation(i,x=3+.02*i*.1)
        fix=gate.add(obs, obs.t_ns) or fix
    assert fix is not None
    assert fix.xy[0] == pytest.approx(3+.02*fix.t_ns/1e9)
    assert fix.velocity == pytest.approx([.02,0], abs=1e-8)
    duplicate=gate.add(obs, obs.t_ns+100_000_000)
    assert duplicate is None


def test_bad_geometry_jump_and_old_frames_cannot_publish_a_fix():
    for case in ('geometry','jump','partial','duplicate'):
        gate=StableUWB(config())
        for i in range(1,40):
            obs=observation(i, bad=3 if case=='geometry' or (case=='jump' and i%2) else 0)
            if case=='partial': obs=replace(obs,frame=UWBFrame((1,None,None,None),str(i)))
            if case=='duplicate': obs=replace(obs,frame=replace(obs.frame,frame_id='same'))
            assert gate.add(obs,obs.t_ns) is None


def test_partial_recovery_requires_two_distinct_fresh_frames_since_stage_boundary():
    gate=StableUWB(config())
    gate.reset(1_000_000_000)
    def partial(t, id): return UWBObservation(t,UWBFrame((2,None,None,None),id),0)
    assert not gate.signal(partial(900_000_000,'old'),1_100_000_000)
    assert not gate.signal(partial(1_100_000_000,'a'),1_100_000_000)
    assert not gate.signal(partial(1_200_000_000,'a'),1_200_000_000)
    assert gate.signal(partial(1_200_000_000,'b'),1_200_000_000)


def test_centered_fin_restarts_b1_but_only_completed_b2_advances():
    with open('config/robot.yaml',encoding='utf8') as stream: cal=build_calibration(yaml.safe_load(stream))
    scheduler=UnderwaterScheduler(cal)
    scheduler.schedule('tail',NavigationTailAction(10,0),start_t_ns=0)
    scheduler.schedule('fin',ContinuousFinAction(10,.3,0,0),start_t_ns=0)
    scheduler.complete('fin',endpoint_written=True,completion_t_ns=300_000_000)
    scheduler.schedule('fin',ContinuousFinAction(-10,.3,1,1),start_t_ns=300_000_000)
    scheduler.hold_groups(cal.centers,('fin',))
    assert scheduler.fin_sequence.next_b1==0 and scheduler.fin_sequence.last_b2==0
    assert scheduler.active['tail'] is not None
    scheduler.schedule('fin',ContinuousFinAction(10,.3,0,0),start_t_ns=400_000_000)


def test_invalid_configuration_rejected_before_hardware():
    with pytest.raises(ValueError): config(t4=float('nan'))
    with pytest.raises(ValueError): UnderwaterConfig(NavigationConfig(L=10,W=6,x_d=3,y_d=2))
    with pytest.raises(ValueError): UnderwaterConfig(NavigationConfig(L=10,W=6,H=1,x_d=3,y_d=2,yaw_sign=-1))


def test_repeated_or_backward_tid_is_rejected_even_when_uart_timestamp_advances():
    est=InertialEstimator(config(accel_tau_s=0))
    est.imu(0,{**imu(), 'tid':65535})
    assert est.imu(5_000_000,{**imu(100), 'tid':65535}) is False
    assert est.imu(10_000_000,{**imu(100), 'tid':65534}) is False
    assert est.imu(15_000_000,{**imu(), 'tid':0}) is True
    assert est.v == pytest.approx([0,0,0])


def test_delayed_fix_replays_pressure_and_preserves_bias_and_vertical_filter():
    a,b=InertialEstimator(config()),InertialEstimator(config())
    for i in range(201):
        for est in (a,b):
            est.imu(i*5_000_000,imu(math.sin(i/20)))
            if i%20==0: est.pressure(i*5_000_000,.2+i*.001)
        if i==100: b.correct(500_000_000,[2,3],[.1,.2])
    a.correct(500_000_000,[2,3],[.1,.2])
    assert a.xyz == pytest.approx(b.xyz)
    assert a.v == pytest.approx(b.v)
    assert a.depth == b.depth


def test_filtered_outlier_does_not_prevent_eventual_stable_fix():
    gate=StableUWB(config())
    fixes=[]
    for i in range(1,45):
        obs=observation(i,bad=1.5 if i==20 else 0)
        fix=gate.add(obs,obs.t_ns)
        if fix: fixes.append(fix)
    assert fixes and fixes[-1].xy==pytest.approx([3,2])
    assert fixes[-1].velocity==pytest.approx([0,0],abs=1e-8)


def test_explicit_n1_accepts_first_full_frame_without_window_or_pressure():
    gate=AcceptedUWB(config(uwb_average_frames=1))
    obs=replace(observation(1),depth=None)
    fix=gate.add(obs,obs.t_ns)
    assert fix.xy==pytest.approx([3.,2.])
    assert fix.ranges_m==obs.frame.ranges_m
    assert fix.velocity==[0.,0.]
    assert gate.last_accepted is None  # 失败的估计器校正不能改变比较基准。
    gate.commit(fix)
    assert gate.last_accepted is fix


@pytest.mark.parametrize('anchor',range(4))
def test_raw_uwb_seven_meter_limit_keeps_last_accepted_reference_across_stages(anchor):
    gate=AcceptedUWB(config(uwb_average_frames=1))
    first=observation(1)
    reference=gate.add(first,first.t_ns); gate.commit(reference)
    gate.reset(200_000_000)
    ranges=list(reference.ranges_m); ranges[anchor]+=7.001
    rejected=UWBObservation(300_000_000,UWBFrame(tuple(ranges),'bad'),None)
    assert gate.add(rejected,rejected.t_ns) is None and gate.reason=='range_jump'
    assert gate.last_accepted is reference
    ranges[anchor]-=.001  # 恰好7米也必须接受，不额外要求几何/边界合理。
    accepted=UWBObservation(400_000_000,UWBFrame(tuple(ranges),'ok'),None)
    fix=gate.add(accepted,accepted.t_ns)
    assert fix is not None and fix.ranges_m==tuple(ranges)
    gate.commit(fix); gate.reset(500_000_000)
    ranges[anchor]+=6.
    following=UWBObservation(600_000_000,UWBFrame(tuple(ranges),'next'),None)
    assert gate.add(following,following.t_ns) is not None  # 相对最新已接受帧，非首帧。


@pytest.mark.parametrize('ranges',[(1,None,3,4),(1,0,3,4),(1,float('nan'),3,4),(1,float('inf'),3,4)])
def test_raw_uwb_incomplete_frames_never_replace_reference(ranges):
    gate=AcceptedUWB(config(uwb_average_frames=1)); obs=observation(1)
    first=gate.add(obs,obs.t_ns); gate.commit(first)
    bad=UWBObservation(200_000_000,UWBFrame(ranges,'bad'),None)
    assert gate.add(bad,bad.t_ns) is None
    assert gate.last_accepted is first


def test_raw_uwb_requires_new_fresh_post_center_frame_and_resets_horizontal_velocity():
    gate=AcceptedUWB(config(uwb_average_frames=1)); gate.reset(100_000_000)
    assert gate.add(observation(1),100_000_000) is None
    assert gate.add(observation(2),600_000_000) is None
    obs=observation(7); fix=gate.add(obs,obs.t_ns); gate.commit(fix)
    gate.reset(0)
    assert gate.add(obs,obs.t_ns) is None
    est=InertialEstimator(config(accel_tau_s=0))
    for i in range(141):
        est.imu(i*5_000_000,imu(1.))
        if i%20==0: est.pressure(i*5_000_000,.2)
    assert est.v[0]>.5
    est.correct(fix.t_ns,fix.xy,fix.velocity)
    assert est.xyz==pytest.approx([3,2,.2])
    assert est.v[:2]==pytest.approx([0,0])


def test_full_surface_stroke_rejects_incompatible_tip_limit_instead_of_shrinking():
    with open('config/robot.yaml',encoding='utf8') as stream: cal=build_calibration(yaml.safe_load(stream))
    cal.limits[5]=(90.,92.)
    with pytest.raises(ValueError,match='limits'):
        SurfaceFinPolicy(cal,.3)


def test_impossible_subnanosecond_timers_are_rejected():
    with pytest.raises(ValueError): config(t4=1e-20)


def test_robot_yaml_is_source_for_sampling_pwm_calibration_and_runtime():
    with open('config/robot.yaml',encoding='utf8') as stream: robot=yaml.safe_load(stream)
    robot['sensors']['imu']['sample_rate_hz']=100
    robot['servo']['discrete_executor']['command_hz']=80
    robot['servo']['discrete_executor']['center_settle_s']=.75
    robot['logging']['flush_interval_s']=.4
    robot['logging']['write_queue_maxsize']=1234
    section=robot['underwater_navigation_20261026']
    section.update(control_poll_s=.007,merge_delay_s=.015,status_hz=3,
                   calibration_max_angle_std_deg=.6,gravity_mps2=9.81,water_density_kg_m3=998)
    cfg=UnderwaterConfig.from_robot(robot)
    assert cfg.imu_rate_hz==100 and cfg.command_hz==80
    assert cfg.center_settle_s==.75
    assert cfg.log_flush_s==.4 and cfg.log_queue_size==1234
    assert cfg.control_poll_s==.007 and cfg.merge_delay_s==.015 and cfg.status_hz==3
    assert cfg.calibration_max_angle_std_deg==.6 and cfg.gravity_mps2==9.81
    assert cfg.water_density_kg_m3==998


def test_missing_yaml_setting_is_reported_instead_of_silent_default():
    with open('config/robot.yaml',encoding='utf8') as stream: robot=yaml.safe_load(stream)
    del robot['underwater_navigation_20261026']['accel_tau_s']
    with pytest.raises(ValueError,match='accel_tau_s'): UnderwaterConfig.from_robot(robot)


def test_terminal_center_hold_must_fit_shutdown_budget():
    with pytest.raises(ValueError,match='shutdown_timeout_s'):
        config(center_settle_s=3.,shutdown_timeout_s=3.)
    with pytest.raises(ValueError,match='center_settle_s'):
        config(center_settle_s=0.)


def test_calibration_counts_follow_configured_imu_rate_and_threshold():
    from control.runtime.underwater_navigation_session_20261026 import calibrate_window
    from control.runtime.sample import SensorSample
    with open('config/robot.yaml',encoding='utf8') as stream: robot=yaml.safe_load(stream)
    robot['sensors']['imu']['sample_rate_hz']=100
    robot['underwater_navigation_20261026'].update(t2=1.,calibration_max_angle_std_deg=.5)
    cfg=UnderwaterConfig.from_robot(robot)
    imus=[SensorSample('imu',i*10_000_000,i,imu()) for i in range(100)]
    depths=[SensorSample('depth',i*100_000_000,i,{'pressure_pa':101325.}) for i in range(10)]
    cal,_,_=calibrate_window(cfg,imus,depths,1_000_000_000)
    assert cal.minimum_imu_samples==90
    for i,sample in enumerate(imus): sample.data['yaw_deg']=(-1 if i%2 else 1)*.8
    with pytest.raises(ValueError,match='unstable'):
        calibrate_window(cfg,imus,depths,1_000_000_000)
