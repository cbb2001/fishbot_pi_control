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


def test_first_navigation_starts_immediately_and_t4_centers_only_tail():
    m,ex,now,events=mission()
    m.tick(0,[1,1,0],0)
    ex._consume_requests()
    assert ex.scheduler.active['tail'] and ex.scheduler.active['fin']
    now[0]=210_000_000; m.tick(now[0],[1,1,0],0); ex._consume_requests()
    assert m.phase=='surface'
    assert ex.scheduler.active['tail'] is None
    assert ex.scheduler.active['fin'] is not None
    assert m.depth_policy.h0==.3  # 当前组合结束之前不能切换目标。


def test_partial_uwb_interrupts_fin_and_center_ack_is_required_for_acquire():
    m,ex,now,events=mission()
    m.tick(0,[1,1,0],0); ex._consume_requests()
    now[0]=210_000_000; m.tick(now[0],[1,1,0],0); ex._consume_requests()
    for t,id in ((220_000_000,'a'),(230_000_000,'b')):
        now[0]=t; m.on_uwb(UWBObservation(t,UWBFrame((2,None,None,None),id),0),t)
    assert m.phase=='surface' and m.center_ticket is not None
    ex._consume_requests(); m.tick(now[0],[1,1,0],0)
    assert m.phase=='acquire'
    assert all(v is None for v in ex.scheduler.active.values())
    assert ex.scheduler.fin_sequence.next_b1==0
    start=m.motion_start
    m.enter_navigation(2_000_000_000)
    assert m.motion_start==start and m.segment_start==2_000_000_000


def test_surface_pressure_zero_pause_requires_three_new_samples_to_resume():
    m,ex,now,events=mission()
    m.phase='surface'; m.stage_start=0; m.depth=0.
    m.tick(0,[1,1,0],0); ex._consume_requests(); m.tick(1,[1,1,0],0)
    assert m.pressure_paused
    for t in (100_000_000,200_000_000):
        m.on_depth(t,.03); m.tick(t,[1,1,.03],0)
    assert m.pressure_paused
    m.on_depth(200_000_000,.03)  # 重复时间戳不能充当第三条。
    assert m.pressure_paused
    m.on_depth(300_000_000,.03); m.tick(300_000_000,[1,1,.03],0)
    assert not m.pressure_paused and m.depth_policy.h0==0


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


def test_no_servo_diagnostic_does_not_wait_for_uwb_or_construct_pwm(tmp_path):
    from control.runtime.underwater_navigation_session_20261026 import UnderwaterMockSensors
    robot=robot_config(); cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.2)
    stop=threading.Event(); sensors=UnderwaterMockSensors(robot,cfg,stop,None)
    sensors.disable_uwb=True
    def forbidden(): raise AssertionError('no-servo constructed hardware')
    result=run_session(robot,cfg,tmp_path,no_servo=True,sensors=sensors,controller=forbidden,stop_event=stop)
    assert result['reason']=='time_limit' and not result['executor_started'], result


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
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=1,t3=.2)
    from control.runtime.underwater_navigation_session_20261026 import UnderwaterMockSensors
    stop=threading.Event(); sensors=UnderwaterMockSensors(robot,cfg,stop,None)
    result=run_session(robot,cfg,tmp_path,no_servo=True,sensors=sensors,stop_event=stop)
    assert result['fault'] is None,result
    metadata=yaml.safe_load((tmp_path/'metadata.yaml').read_text(encoding='utf8'))
    assert metadata['robot']['sensors']['vision']['enabled'] is False
    assert metadata['robot']['sensors']['vision']['save_frames'] is False
    assert metadata['robot']['sensors']['imu']['sample_rate_hz']==100
    assert metadata['robot']['depth_sensor']['surface_pressure_mbar']==1025.


def test_incompatible_pressure_rate_fails_before_sampling(tmp_path):
    robot=robot_config(); robot['sensors']['depth']['rate_hz']=20
    cfg=replace(UnderwaterConfig.from_robot(robot),t1=0,t2=.6,t3=.1,fix_timeout_s=.2)
    with pytest.raises(ValueError,match='sample_rate|采样率'):
        run_session(robot,cfg,tmp_path,dry_run=True)
