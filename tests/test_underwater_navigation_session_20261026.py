import json
import threading
from dataclasses import replace

import pytest
import yaml

from control.runtime.underwater_navigation_20261026 import UnderwaterConfig, UWBObservation
from control.runtime.underwater_navigation_session_20261026 import NavigationMission, run_session, calibrate_window
from control.runtime.surface_navigation_20261010 import UWBFrame
from control.runtime.sample import SensorSample
from control.runtime.rl_actions_20260914 import build_calibration
from control.runtime.rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.underwater_navigation_20261026 import UnderwaterScheduler


def robot_config():
    with open('config/robot.yaml',encoding='utf8') as f: return yaml.safe_load(f)


def mission():
    robot=robot_config(); cfg=replace(UnderwaterConfig.from_robot(robot),t4=.2,t3=10)
    cal=build_calibration(robot); now=[0]; scheduler=UnderwaterScheduler(cal)
    ex=RLServoExecutor(DryRunRLServoController(),scheduler,RLServoStateTracker(cal),clock_ns=lambda:now[0])
    events=[]
    m=NavigationMission(cfg,cal,ex,lambda kind,**data:events.append((kind,data)))
    m.enter_navigation(0); m.depth=.1
    return m,ex,now,events


def test_t4_cancels_depth_control_then_starts_full_surface_stroke():
    m,ex,now,events=mission()
    m.tick(0,[1,1,0],0)
    ex._consume_requests()
    assert ex.scheduler.active['tail'] and ex.scheduler.active['fin']
    now[0]=210_000_000; m.tick(now[0],[1,1,0],0); ex._consume_requests()
    assert m.phase=='surface'
    assert ex.scheduler.active['tail'] is None
    assert ex.scheduler.active['fin'] is None
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    action=ex.scheduler.active['fin'].action
    assert action.b1==0
    assert action.theta+ex.scheduler.centers[4]==ex.scheduler.limits[4][1]
    assert action.t==m.cfg.fin_duration_s
    assert ex.scheduler.active['tail'] is None


def test_partial_uwb_interrupts_fin_and_center_ack_is_required_for_acquire():
    m,ex,now,events=mission()
    m.tick(0,[1,1,0],0); ex._consume_requests()
    now[0]=210_000_000; m.tick(now[0],[1,1,0],0); ex._consume_requests()
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    now[0]=220_000_000
    m.on_uwb(UWBObservation(now[0],UWBFrame((None,None,2,None),'a'),0),now[0])
    assert m.phase=='surface' and m.center_ticket is not None
    ex._consume_requests(); m.tick(now[0],[1,1,0],0)
    assert m.phase=='acquire'
    assert all(v is None for v in ex.scheduler.active.values())
    assert ex.scheduler.fin_sequence.next_b1==0
    start=m.motion_start
    m.enter_navigation(2_000_000_000)
    assert m.motion_start==start and m.segment_start==2_000_000_000


def test_surface_fixed_strokes_ignore_pressure_and_preserve_b2_across_resume():
    m,ex,now,events=mission()
    from control.runtime.fixed_dive_20260920_100122 import FixedFinSequence
    ex.scheduler.fin_sequence=FixedFinSequence(last_b2=1)
    now[0]=210_000_000
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    actions=[]
    for depth in (0.,-.03,0.,.04):
        m.on_depth(now[0],depth)
        m.tick(now[0],[1,1,depth],0); ex._consume_requests()
        action=ex.scheduler.active['fin'].action; actions.append(action)
        now[0]+=round(m.cfg.fin_duration_s*1e9)
        ex.tick_once()
    assert [a.b1 for a in actions]==[0,1,0,1]
    assert [a.b2 for a in actions if a.b1]==[-1,1]
    assert [a.theta+ex.scheduler.centers[4] for a in actions]==[164,58,164,58]
    assert ex.scheduler.active['tail'] is None
    # 中断下一组b1=0；归中不推进b2，下次导航继续从-1开始。
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    now[0]+=1
    m.on_uwb(UWBObservation(now[0],UWBFrame((2,None,None,None),'stop'),0),now[0])
    ex._consume_requests(); m.tick(now[0],[1,1,0],0)
    m.enter_navigation(now[0]+1)
    assert m.depth_policy.next_b2==-1


def test_surface_and_acquire_wait_past_old_phase_timeouts_until_total_deadline():
    m,ex,now,events=mission()
    m.cfg=replace(m.cfg,t3=100,fix_timeout_s=.01,surface_timeout_s=.01)
    now[0]=210_000_000
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    now[0]=40_000_000_000
    m.tick(now[0],[1,1,0],0); ex._consume_requests()
    assert m.phase=='surface' and m.reason is None
    m.on_uwb(UWBObservation(now[0]+1,UWBFrame((2,None,None,None),'a'),0),now[0]+1)
    ex._consume_requests(); m.tick(now[0]+1,[1,1,0],0)
    assert m.phase=='acquire'
    m.tick(90_000_000_000,[1,1,0],0)
    assert m.reason is None
    m.tick(100_000_000_000,[1,1,0],0)
    assert m.reason=='time_limit'


def test_calibration_rejects_short_coverage_even_with_enough_samples():
    data=dict(acc_mps2=[0,0,9.80665],quat=[1,0,0,0],gyro_radps=[0,0,0],yaw_deg=0,pitch_deg=0,roll_deg=0)
    imus=[SensorSample('imu',i*100_000, i,data) for i in range(200)]
    depths=[SensorSample('depth',i*100_000,i,dict(pressure_pa=101325)) for i in range(10)]
    with pytest.raises(ValueError,match='coverage'):
        calibrate_window(replace(UnderwaterConfig.from_robot(robot_config()),t2=1),imus,depths,1_000_000_000)


def test_dry_session_cycles_and_preserves_pressure_origin(tmp_path):
    robot=robot_config(); cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=7,t4=.4,
        settle_s=.05,window_s=.5,min_span_s=.3,min_frames=4,stable_s=.1,stable_frames=2)
    result=run_session(robot,cfg,tmp_path,dry_run=True)
    assert result['reason']=='time_limit' and result['fault'] is None, result
    events=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text(encoding='utf8').splitlines()]
    assert sum(e['event']=='navigation_started' for e in events)>=2
    assert any(e['event']=='surface_started' for e in events)
    assert any(e['event']=='uwb_acquire_started' for e in events)
    assert result['pressure_origin_t_ns']==json.loads((tmp_path/'calibration.json').read_text(encoding='utf8'))['pressure_drift_origin_t_ns']
    assert not result['alive_workers'] and not result['cleanup_errors']
    assert yaml.safe_load(yaml.safe_dump(result))['reason']=='time_limit'  # CLI结果必须能序列化。
    centered=result['terminal_center']
    assert centered['command_ack_t_ns'] is not None
    assert centered['hold_completed_t_ns']-centered['command_ack_t_ns'] >= cfg.center_settle_s*1e9
    assert centered['pwm_disabled_t_ns'] >= centered['hold_completed_t_ns']
    assert any(e['event']=='terminal_center_status' for e in events)
    accepted=[e for e in events if e['event']=='uwb_accepted']
    fixes=[e for e in events if e['event']=='position_corrected']
    assert accepted and result['last_accepted_uwb']['frame_id']==accepted[-1]['frame_id']
    assert all(e['velocity']==[0.,0.] and e['quality']['mode']=='mean_valid_ranges' for e in fixes)
    assert all(e['quality']['frames']==5 and e['xyz'][2]==0 for e in fixes)
    assert all(e['integration_reset'] and e['relative_xy']==[0.,0.] for e in fixes)
    assert all(e['accel_filter_reset'] is True for e in fixes)
    metadata=yaml.safe_load((tmp_path/'metadata.yaml').read_text(encoding='utf8'))
    assert metadata['uwb_accel_filter_reset']=='both_stages_zero_at_fix'
    assert metadata['imu_integration_mode']=='navigate_only'
    assert metadata['ekf_state_order']==['x','y','z','vx','vy','vz','bx','by','bz']
    assert all(e['quality']['reference_frame_id']==a['frame_id'] for a,e in zip(accepted,accepted[1:]))
    samples=[json.loads(s) for s in (tmp_path/'imu_filtered.jsonl').read_text(encoding='utf8').splitlines()]
    assert any(s['integrating'] for s in samples)
    assert all(len(s['ekf_bias_mps2'])==3 and len(s['ekf_variance'])==9 for s in samples)
    assert all(not s['integrating'] for s in samples if s['phase']!='navigate')
    for a,b in zip(samples,samples[1:]):
        if a['phase']==b['phase'] and a['phase'] in ('surface','acquire'):
            assert a['xyz'][:2]==b['xyz'][:2]


def test_session_averages_five_valid_frames_across_uart_batches(tmp_path,monkeypatch):
    import control.runtime.underwater_navigation_session_20261026 as module
    from control.runtime.surface_navigation_20261010 import UWBStreamDecoder
    original=module._UnderwaterMockWorker.read_once
    expected={}

    def read_once(worker):
        sample=original(worker)
        mission=worker.manager.mission
        if worker.name!='uwb' or mission is None or mission.phase!='acquire': return sample
        base=UWBStreamDecoder('t0:0').feed(bytes.fromhex(sample.data['raw_hex']))[0]
        packets=[]
        # 同一UART批次：超限帧、两个合格帧。只计合格帧，跨批累计满5帧再恢复。
        for tag,jump in (('f1',20.),('f2',1.),('f3',3.)):
            ranges=list(base.ranges_m); ranges[1]+=jump
            frame_id=f'{tag}:{sample.seq:08x}:t0:0'
            expected[frame_id]=ranges
            packets.append('mc 00 '+' '.join(f'{round(v*1000):08x}' for v in ranges)+
                           f' 4705 {tag} {sample.seq:08x} t0:0 0000\r\n')
        sample.data['raw_hex']=''.join(packets).encode().hex()
        return sample

    monkeypatch.setattr(module._UnderwaterMockWorker,'read_once',read_once)
    robot=robot_config()
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=3.5,t4=.4,settle_s=20)
    result=run_session(robot,cfg,tmp_path,dry_run=True)
    assert result['reason']=='time_limit' and result['fault'] is None,result
    events=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text(encoding='utf8').splitlines()]
    fixes=[e for e in events if e['event']=='position_corrected' and e['phase']=='acquire']
    assert fixes
    for fix in fixes:
        import numpy as np
        ids=fix['quality']['frame_ids']
        assert len(ids)==5 and len(set(ids))==5
        assert all(not id.startswith('f1:') for id in ids)
        assert fix['ranges_m']==pytest.approx(np.mean([expected[id] for id in ids],axis=0))
        assert fix['velocity']==[0.,0.]
        assert fix['xyz'][2]==0
    quality=[json.loads(s) for s in (tmp_path/'navigation.jsonl').read_text(encoding='utf8').splitlines()]
    assert any(e.get('reason')=='range_jump' for e in quality)


def test_no_servo_diagnostic_does_not_wait_for_uwb_or_construct_pwm(tmp_path):
    from control.runtime.underwater_navigation_session_20261026 import UnderwaterMockSensors
    robot=robot_config(); cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.2)
    stop=threading.Event(); sensors=UnderwaterMockSensors(robot,cfg,stop,None)
    sensors.disable_uwb=True
    def forbidden(): raise AssertionError('no-servo constructed hardware')
    result=run_session(robot,cfg,tmp_path,no_servo=True,sensors=sensors,controller=forbidden,stop_event=stop)
    assert result['reason']=='time_limit' and not result['executor_started'], result


def test_four_valid_frames_after_surface_do_not_resume_navigation(tmp_path,monkeypatch):
    import control.runtime.underwater_navigation_session_20261026 as module
    original=module._UnderwaterMockWorker.read_once
    count=[0]
    def read_once(worker):
        sample=original(worker)
        m=worker.manager.mission
        if worker.name=='uwb' and m is not None and m.phase=='acquire':
            count[0]+=1
            if count[0]>4: sample.data['raw_hex']=b'RANGE_ERROR\r\n'.hex()
        return sample
    monkeypatch.setattr(module._UnderwaterMockWorker,'read_once',read_once)
    robot=robot_config()
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=2.,t4=.2)
    result=run_session(robot,cfg,tmp_path,dry_run=True)
    assert result['reason']=='time_limit' and result['fault'] is None,result
    events=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text(encoding='utf8').splitlines()]
    assert any(e['event']=='uwb_acquire_started' for e in events)
    assert sum(e['event']=='navigation_started' for e in events)==1
    assert not any(e['event']=='position_corrected' and e['phase']=='acquire' for e in events)
    rows=[json.loads(s) for s in (tmp_path/'navigation.jsonl').read_text(encoding='utf8').splitlines()]
    assert max(e['quality']['frames'] for e in rows if e['phase']=='acquire' and e['event']=='uwb_quality')==4


def test_overcurrent_during_terminal_center_hold_disables_pwm_immediately(tmp_path,monkeypatch):
    import control.runtime.underwater_navigation_session_20261026 as module
    from control.runtime.rl_safety_20260914 import RLSafetyFault
    original=module.RLSafetyMonitor.check_once

    def fail_during_hold(guard):
        original(guard)
        if guard.executor.terminal_center_ack_t_ns is not None:
            raise RLSafetyFault('OVERCURRENT','injected during terminal center hold')

    monkeypatch.setattr(module.RLSafetyMonitor,'check_once',fail_during_hold)
    robot=robot_config()
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.3,t4=.4)
    result=run_session(robot,cfg,tmp_path,dry_run=True)
    assert result['reason']=='fault' and 'OVERCURRENT' in result['fault'],result
    centered=result['terminal_center']
    assert centered['command_ack_t_ns'] is not None
    assert centered['hold_completed_t_ns'] is None
    assert centered['pwm_disabled_t_ns']-centered['command_ack_t_ns'] < cfg.center_settle_s*1e9
    assert not result['alive_workers']


def test_camera_missing_invalid_or_stale_samples_cannot_be_successful():
    from types import SimpleNamespace
    from control.runtime.ring_buffer import RingBuffer
    from control.runtime.underwater_navigation_session_20261026 import check_camera_health
    buffer=RingBuffer(4)
    sensors=SimpleNamespace(buffers={'vision':buffer},workers=[],fatal_errors=lambda:{})
    with pytest.raises(RuntimeError,match='camera'):
        check_camera_health(sensors,now_ns=6_000_000_000,started_ns=0)
    buffer.append(SensorSample('vision',6_000_000_000,1,{},ok=False,error='camera_read_failed'))
    with pytest.raises(RuntimeError,match='camera'):
        check_camera_health(sensors,now_ns=6_000_000_000,started_ns=0)
    buffer.append(SensorSample('vision',6_000_000_000,2,{},ok=True))
    with pytest.raises(RuntimeError,match='camera'):
        check_camera_health(sensors,now_ns=7_000_000_000,started_ns=0)


def test_camera_shutdown_reports_fatal_error_and_live_writer():
    from types import SimpleNamespace
    from control.runtime.underwater_navigation_session_20261026 import check_camera_health
    sensors=SimpleNamespace(workers=[],fatal_errors=lambda:{'vision':'release failed'})
    with pytest.raises(RuntimeError,match='release failed'):
        check_camera_health(sensors,now_ns=0,started_ns=0,shutdown=True)
    writer=SimpleNamespace(last_error=None,dropped_frames=0,
        _thread=SimpleNamespace(is_alive=lambda:True),
        index_logger=SimpleNamespace(last_error=None,dropped_count=0,is_alive=lambda:False))
    sensors.fatal_errors=lambda:{}
    sensors.workers=[SimpleNamespace(_closed_image_writer=writer)]
    with pytest.raises(RuntimeError,match='thread'):
        check_camera_health(sensors,now_ns=0,started_ns=0,shutdown=True)


def test_image_save_false_is_reported_as_persistent_error(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from control.runtime.camera_worker import ImageWriter,ImageSaveTask
    writer=ImageWriter(log_dir=tmp_path)
    monkeypatch.setattr(writer,'_load_cv2',lambda:SimpleNamespace(imwrite=lambda *args:False,IMWRITE_JPEG_QUALITY=1))
    try:
        writer._write_task(ImageSaveTask(None,1,1,'camera/test.jpg',1,1,'jpg',85))
        assert writer.last_error=='cv2_imwrite_failed'
    finally: writer.stop()


def test_safety_fault_at_deadline_is_retained_in_result(tmp_path,monkeypatch):
    import control.runtime.underwater_navigation_session_20261026 as module
    class LateFaultGuard(module.RLSafetyMonitor):
        def stop(self,timeout=5.):
            super().stop(timeout)
            self.failure=RuntimeError('OVERCURRENT at deadline')
    monkeypatch.setattr(module,'RLSafetyMonitor',LateFaultGuard)
    robot=robot_config(); cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.1)
    stop=threading.Event(); sensors=module.UnderwaterMockSensors(robot,cfg,stop,None)
    result=module.run_session(robot,cfg,tmp_path,no_servo=True,sensors=sensors,stop_event=stop)
    assert result['reason']=='fault' and 'OVERCURRENT' in result['fault'],result


def test_sensor_switches_rates_and_driver_reference_are_not_overwritten(tmp_path):
    robot=robot_config()
    robot['sensors']['vision'].update(enabled=False,save_frames=False)
    robot['sensors']['imu']['sample_rate_hz']=100
    robot['depth_sensor']['surface_pressure_mbar']=1025.
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=1,t3=.8)
    from control.runtime.underwater_navigation_session_20261026 import UnderwaterMockSensors
    stop=threading.Event(); sensors=UnderwaterMockSensors(robot,cfg,stop,None)
    result=run_session(robot,cfg,tmp_path,no_servo=True,sensors=sensors,stop_event=stop)
    assert result['fault'] is None,result
    metadata=yaml.safe_load((tmp_path/'metadata.yaml').read_text(encoding='utf8'))
    assert metadata['robot']['sensors']['vision']['enabled'] is False
    assert metadata['robot']['sensors']['vision']['save_frames'] is False
    assert metadata['robot']['sensors']['imu']['sample_rate_hz']==100
    assert metadata['robot']['depth_sensor']['surface_pressure_mbar']==1025.
    assert metadata['uwb_accel_filter_reset']=='both_stages_zero_at_fix'
    fixes=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text(encoding='utf8').splitlines()
           if json.loads(s)['event']=='position_corrected']
    assert fixes and all(e['accel_filter_reset'] is True for e in fixes)
    samples=[json.loads(s) for s in (tmp_path/'imu_filtered.jsonl').read_text(encoding='utf8').splitlines()]
    assert samples and all(not s['integrating'] and s['velocity']==[0.,0.,0.] for s in samples)


def test_incompatible_pressure_rate_fails_before_sampling(tmp_path):
    robot=robot_config(); robot['sensors']['depth']['rate_hz']=20
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.1,fix_timeout_s=.2)
    with pytest.raises(ValueError,match='sample_rate|采样率'):
        run_session(robot,cfg,tmp_path,dry_run=True)
