"""验证采样时间基准、去漂移 P0、深度恢复和模型边界。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from control.runtime.pressure_drift_20261001 import PressureDrift, CorrectedSensors
from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.fixed_dive_20260920_100122 import FixedDepthCalibration
from control.runtime.rl_sensor_calibration_20260914 import RLSensorCalibration


def model():
    return PressureDrift(Path(__file__).resolve().parents[1]/'config/pressure_drift_10hz.json')


def test_lookup_interpolation_and_endpoints():
    drift = model()
    assert drift.drift_pa(0) == 0
    assert drift.drift_pa(.05) == pytest.approx(drift.values[1]/2)
    assert drift.drift_pa(7200) == drift.values[-1]
    for t in (-.1, 7200.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            drift.drift_pa(t)


def test_surface_mean_corrects_drift_and_preserves_raw_buffer():
    drift = model()
    buf = RingBuffer(100)
    start = 1_000_000_000
    p0 = 104000.
    for i in range(50):
        t = i*.1
        buf.append(SensorSample('depth', start+i*100_000_000, i,
                               {'pressure_pa': p0+drift.drift_pa(t)}))
    sensors = CorrectedSensors(SimpleNamespace(buffers={'depth': buf}), drift)
    # 首次访问 latest 也必须从第一条有效记录确定起点。
    assert sensors.buffers['depth'].latest().data['pressure_pa'] == pytest.approx(p0)
    assert drift.origin_t_ns == start
    base = RLSensorCalibration({'minimum_depth_samples': 45})
    base.calibrate_depth(sensors.buffers['depth'].snapshot())
    assert base.p_surface_pa == pytest.approx(p0)
    assert buf.latest().data['pressure_pa'] != pytest.approx(p0)
    assert 'raw_pressure_pa' not in buf.latest().data
    # 经过等待后才下潜；使用采样时刻，不使用当前墙钟或控制开始时刻。
    t = 60.
    buf.append(SensorSample('depth', start+int(t*1e9), 50,
                           {'pressure_pa':p0+drift.drift_pa(t)+9806.65*.2}))
    cal = FixedDepthCalibration(base, 9806.65, drift)
    assert cal.depth_m(sensors.buffers['depth'].latest().data['pressure_pa']) == pytest.approx(.2)
    assert drift.origin_t_ns == start
    buf.append(SensorSample('depth', start+7201*10**9, 51, {'pressure_pa':104000.}))
    with pytest.raises(ValueError, match='超出查表'):
        sensors.buffers['depth'].latest()


def test_invalid_first_sample_does_not_set_origin():
    drift = model()
    buf = RingBuffer(10)
    buf.append(SensorSample('depth', 1, 0, {}, ok=False, error='read failed'))
    buf.append(SensorSample('depth', 2, 1, {'pressure_pa': -1}))
    buf.append(SensorSample('depth', 3, 2, {'pressure_pa': 104000}))
    view = CorrectedSensors(SimpleNamespace(buffers={'depth':buf}), drift)
    rows = view.buffers['depth'].snapshot()
    assert drift.origin_t_ns == 3
    assert [s.ok for s in rows] == [False, False, True]


def test_wrong_rate_model_rejected(tmp_path):
    config = model().config
    config['sample_rate_hz'] = 20
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        PressureDrift(path)


def test_observed_origin_survives_ring_eviction_and_failed_read():
    from control.runtime.pressure_drift_20261001 import ObservedPressureBuffer
    drift = model()
    buf = ObservedPressureBuffer(RingBuffer(2), drift)
    for i in range(5):
        buf.append(SensorSample('depth', 10**9+i*10**8, i, {'pressure_pa':104000.}))
    assert drift.origin_t_ns == 10**9
    assert len(buf.snapshot()) == 2
    buf.append(SensorSample('depth', 2*10**9, 5, {}, ok=False, error='I2C reset'))
    buf.append(SensorSample('depth', 3*10**9, 6, {'pressure_pa':104000.}))
    assert not buf.latest().ok
    assert drift.origin_t_ns == 10**9


def test_corrected_view_is_causal_and_uses_sample_time():
    drift = model()
    buf = RingBuffer(20)
    start = 10**9
    p0 = 104000.
    buf.append(SensorSample('depth', start, 0, {'pressure_pa':p0,'depth_m':-99.}))
    view = CorrectedSensors(SimpleNamespace(buffers={'depth':buf}), drift)
    view.buffers['depth'].latest()
    base = RLSensorCalibration({'surface_pressure_mbar':p0/100})
    view.buffers['depth'].calibration = base
    for t,h in [(60.,.2),(61.,.4)]:
        buf.append(SensorSample('depth', start+int(t*1e9), int(t),
                               {'pressure_pa':p0+drift.drift_pa(t)+9806.65*h,'depth_m':-99.}))
    row = view.buffers['depth'].get_latest_before(start+int(60.5e9))
    assert row.data['drift_elapsed_s'] == 60.
    assert row.data['depth_m'] == pytest.approx(.2)
    assert buf.latest().data['depth_m'] == -99.


def test_safety_checks_corrected_depth_not_raw_pressure():
    import queue
    import threading
    from control.runtime.rl_safety_20260914 import RLSafetyMonitor, RLSafetyFault
    drift = model()
    start = 10**9
    now = start+600*10**9
    buffers = {n:RingBuffer(10) for n in ('depth','imu','power')}
    buffers['depth'].append(SensorSample('depth', start, 0, {'pressure_pa':104000.}))
    view = CorrectedSensors(SimpleNamespace(buffers=buffers, fatal_errors=lambda:[]), drift)
    view.buffers['depth'].latest()
    for name,data in [('imu',{}),('power',{'voltage_v':12.,'current_a':1.,'power_w':12.})]:
        buffers[name].append(SensorSample(name,now,1,data))
    # 漂移使未经补偿的深度低于0.3m，补偿后真实0.4m应触发保护。
    buffers['depth'].append(SensorSample('depth',now,1,
        {'pressure_pa':104000.+drift.drift_pa(600)+.4*9806.65}))
    ex = SimpleNamespace(failure=None,is_alive=lambda:True)
    guard = RLSafetyMonitor(view, ex, queue.Queue(), threading.Event(), {},
        {'safety':{'min_voltage_v':10.5,'max_current_a':10,'max_depth_m':.3}}, clock_ns=lambda:now)
    base = RLSensorCalibration({'surface_pressure_mbar':1040})
    guard.calibration = FixedDepthCalibration(base,9806.65,drift)
    with pytest.raises(RLSafetyFault, match='OVERDEPTH'):
        guard.check_once()
