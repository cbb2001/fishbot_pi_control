"""固定下潜策略：测试整组边界、角度语义及实际 PWM 执行路径。"""
import importlib
import json
import math
from pathlib import Path

import pytest
import yaml

from control.runtime.rl_actions_20260914 import build_calibration, build_rl_trajectory, validate_trajectory
from control.runtime.rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker


@pytest.fixture
def api():
    try:
        return importlib.import_module('control.runtime.fixed_dive_20260920_100122')
    except ModuleNotFoundError:
        pytest.fail('固定下潜控制模块尚未实现')


@pytest.fixture
def calibration():
    centers = [85, 95, 95, 111, 90, 143, 90]
    limits = [(55, 115), (65, 125), (65, 125), (58, 164), (0, 180), (90, 196), (0, 180)]
    return build_calibration({'servo': {'channels': [
        {'servo_id': i, 'center_angle': c, 'min_angle': lo, 'max_angle': hi}
        for i, (c, (lo, hi)) in enumerate(zip(centers, limits), 1)
    ]}})


def test_crossing_target_finishes_pair_then_holds_and_restarts(api, calibration):
    policy = api.FixedDivePolicy(calibration, h0=.4, k=.5)
    first = policy.next_action(.2)
    assert (first.theta, first.t, first.b1, first.b2) == (-13.25, .6, 0, 0)
    # 途中穿越阈值不能改变本组动作，必须先获得第一动作完成回执。
    assert policy.next_action(.6) == first
    policy.acknowledge(first)
    second = policy.next_action(.6)
    assert (second.theta, second.t, second.b1, second.b2) == (13.25, .6, 1, 1)
    policy.acknowledge(second)
    assert policy.next_action(.6) is None
    assert policy.next_action(.4) is None
    first2 = policy.next_action(.3)
    assert first2.theta == pytest.approx(-6.625)
    policy.acknowledge(first2)
    second2 = policy.next_action(.3)
    assert second2.b2 == -1
    policy.acknowledge(second2)
    first3 = policy.next_action(.3)
    policy.acknowledge(first3)
    assert policy.next_action(.3).b2 == 1


def test_extreme_gain_limits_roots_and_tip_without_changing_coupling(api, calibration):
    # 5号正向仅有18度余量，因此正向根部总行程最多21.2度。
    calibration.limits[5] = (0., 108.)
    policy = api.FixedDivePolicy(calibration, h0=.4, k=100.)
    first = policy.next_action(-.5)
    assert first.theta == pytest.approx(-10.6)
    policy.acknowledge(first)
    second = policy.next_action(-.5)
    assert second.theta == pytest.approx(10.6)
    traj = build_rl_trajectory(second, first.theta, calibration)
    assert traj.evaluate_all(.3)[5] == pytest.approx(108.)
    validate_trajectory(traj, calibration.limits)


def test_continuous_actions_use_existing_pwm_executor_and_mirror(api, calibration):
    now = [0]
    scheduler = api.FixedDiveScheduler(calibration)
    tracker = RLServoStateTracker(calibration)
    controller = DryRunRLServoController(calibration.centers)
    ex = RLServoExecutor(controller, scheduler, tracker, clock_ns=lambda: now[0],
                         active_servo_ids=(4, 5, 6, 7))
    policy = api.FixedDivePolicy(calibration, h0=.4, k=.5)
    for _ in range(4):
        action = policy.next_action(.2)
        scheduler.schedule('fin', action, start_t_ns=now[0])
        now[0] += 600_000_000
        refs = ex.tick_once()
        done = ex.drain_completed()
        assert len(done) == 1 and done[0].endpoint_written
        policy.acknowledge(done[0].action)
        assert refs[4] + refs[6] == pytest.approx(254.)
    assert set(tracker.latest_commanded_angles()) >= {4, 5, 6, 7}
    assert all(set(row) <= {4, 5, 6, 7} for row in controller.commands)
    held = ex.tick_once()
    now[0] += 10_000_000_000
    assert ex.tick_once() == held


@pytest.mark.parametrize('h0,k', [(0, 1), (-1, 1), (.3, -1), (math.nan, 1), (.3, math.inf)])
def test_invalid_parameters_rejected(api, calibration, h0, k):
    with pytest.raises(ValueError):
        api.FixedDivePolicy(calibration, h0=h0, k=k)


@pytest.mark.parametrize('duration', [.25, .8, 1.2])
def test_configurable_duration_controls_both_trajectories_and_completion(api, calibration, duration):
    policy = api.FixedDivePolicy(calibration, h0=.4, k=.5, action_duration_s=duration)
    scheduler = api.FixedDiveScheduler(calibration)
    now = 0
    for b1 in (0, 1):
        action = policy.next_action(.2 if b1 == 0 else .6)
        assert action.t == duration
        assert action.b1 == b1
        item = scheduler.schedule('fin', action, start_t_ns=now)
        assert item.end_t_ns == now + round(duration * 1e9)
        assert scheduler.tick(item.end_t_ns-1)[1] == []
        assert scheduler.tick(item.end_t_ns)[1] == [item]
        traj = item.trajectory
        assert traj.evaluate_all(duration)[4] == pytest.approx(97.75 if b1 == 0 else 124.25)
        if b1:
            assert traj.evaluate_all(duration/2)[5] == pytest.approx(112.5)
            assert traj.evaluate_all(duration)[5] == 90.
        scheduler.complete('fin', endpoint_written=True, completion_t_ns=item.end_t_ns)
        policy.acknowledge(action)
        now = item.end_t_ns
    assert policy.next_action(.6) is None


@pytest.mark.parametrize('duration', [0, -1, math.nan, math.inf, 1e-12, 1e308])
def test_invalid_duration_rejected_before_running(api, calibration, duration):
    with pytest.raises(ValueError):
        api.FixedDivePolicy(calibration, h0=.3, action_duration_s=duration)


@pytest.mark.parametrize('manual_reference', [False, True])
def test_full_session_holds_and_resumes_with_opposite_b2(api, monkeypatch, tmp_path, manual_reference):
    from control.runtime.rl_mock_sensors_20260914 import _RLSensorWorker
    root = Path(__file__).resolve().parents[1]
    robot = yaml.safe_load((root/'config/robot.yaml').read_text(encoding='utf-8'))
    raw = yaml.safe_load((root/'config/rl_training_20260914.yaml').read_text(encoding='utf-8'))
    cfg = api.prepare_fixed_config(raw, dry_run=True)
    original = _RLSensorWorker.read_once

    def read(worker):
        sample = original(worker)
        surface, elapsed = worker.manager._motion_state()
        if worker.name == 'depth' and not surface:
            # 第一组中途达到目标，保持一小段后重新变浅，第二组必须 b2=-1。
            depth = .5 if .15 <= elapsed < 1.6 else .1
            sample.data['pressure_pa'] = robot['depth_sensor']['surface_pressure_mbar']*100 + depth*750
        return sample

    monkeypatch.setattr(_RLSensorWorker, 'read_once', read)
    reference = robot['depth_sensor']['surface_pressure_mbar']*100 if manual_reference else None
    result = api.run_session(cfg, robot, tmp_path, h0=.3, k=.5, dry_run=True, max_control_s=2.,
                             surface_pressure_pa=reference)
    cal = json.loads((tmp_path/'calibration.json').read_text())
    assert cal['pressure_per_meter'] == 750
    assert cal['depth_conversion'] == 'linear_pressure'
    assert cal['p_surface_pa'] == pytest.approx(robot['depth_sensor']['surface_pressure_mbar']*100)
    assert cal['pressure_reference_mode'] == ('fixed_config' if manual_reference else 'sample_mean')
    assert (cal['depth_sample_count'] == 0) == manual_reference
    commands = [json.loads(v) for v in (tmp_path/'commands.jsonl').read_text().splitlines()]
    events = [json.loads(v) for v in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert result['completed_pairs'] == 2
    assert [(v['action']['b1'], v['action']['b2']) for v in commands] == [(0, 0), (1, 1), (0, 0), (1, -1)]
    hold = next(v for v in events if v['event'] == 'hold_endpoint')
    second_done = next(v for v in events if v.get('request_id') == 2)
    assert hold['t_ns'] >= second_done['t_ns']
    assert hold['t_ns'] < commands[2]['t_ns']
    assert json.loads((tmp_path/'result.json').read_text())['fault'] is None


def test_low_voltage_fails_before_any_action_submission(api, tmp_path):
    root = Path(__file__).resolve().parents[1]
    robot = yaml.safe_load((root/'config/robot.yaml').read_text(encoding='utf-8'))
    cfg = api.prepare_fixed_config(yaml.safe_load((root/'config/rl_training_20260914.yaml').read_text(encoding='utf-8')), dry_run=True)
    robot['safety']['min_voltage_v'] = 20.
    with pytest.raises(RuntimeError, match='LOW_VOLTAGE'):
        api.run_session(cfg, robot, tmp_path, h0=.3, dry_run=True, max_control_s=1.)
    assert (tmp_path/'commands.jsonl').read_text() == ''
    assert 'LOW_VOLTAGE' in json.loads((tmp_path/'result.json').read_text())['fault']['message']


def test_empirical_depth_and_manual_pressure_precedence(api):
    from control.runtime.rl_sensor_calibration_20260914 import RLSensorCalibration
    manual = RLSensorCalibration({'surface_pressure_mbar': 1040.})
    manual.calibrate_depth([105000.] * 50)
    adapter = api.FixedDepthCalibration(manual, 750.)
    assert adapter.depth_m(104150.) == pytest.approx(.2)
    assert adapter.depth_m(103925.) == pytest.approx(-.1)
    assert api.FixedDepthCalibration(manual, 1500.).depth_m(104150.) == pytest.approx(.1)
    automatic = RLSensorCalibration({'minimum_depth_samples': 3})
    automatic.calibrate_depth([104000., 104010., 103990.])
    assert api.FixedDepthCalibration(automatic, 750.).depth_m(104000.) == 0
    with pytest.raises(ValueError):
        api.FixedDepthCalibration(manual, 0)
