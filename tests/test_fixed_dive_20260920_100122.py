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
    # 深于目标时执行上浮组合；只有回到精确目标深度才保持。
    up_first = policy.next_action(.6)
    assert (up_first.theta, up_first.b1, up_first.b2) == (pytest.approx(13.25), 0, 0)
    policy.acknowledge(up_first)
    up_second = policy.next_action(.6)
    assert (up_second.theta, up_second.b1, up_second.b2) == (pytest.approx(-13.25), 1, -1)
    policy.acknowledge(up_second)
    assert policy.next_action(.4) is None
    first2 = policy.next_action(.3)
    assert first2.theta == pytest.approx(-6.625)
    policy.acknowledge(first2)
    second2 = policy.next_action(.3)
    assert second2.b2 == 1
    policy.acknowledge(second2)
    first3 = policy.next_action(.3)
    policy.acknowledge(first3)
    assert policy.next_action(.3).b2 == -1


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
    assert policy.next_action(.6).theta == pytest.approx(13.25)


@pytest.mark.parametrize('duration', [0, -1, math.nan, math.inf, 1e-12, 1e308])
def test_invalid_duration_rejected_before_running(api, calibration, duration):
    with pytest.raises(ValueError):
        api.FixedDivePolicy(calibration, h0=.3, action_duration_s=duration)


@pytest.mark.parametrize('manual_reference', [False, True])
def test_full_session_reverses_after_pair_with_opposite_b2(api, monkeypatch, tmp_path, manual_reference):
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
            # 第一组中途超过目标，第二组改为上浮且 b2=-1；组内变浅仍执行完。
            depth = .5 if .15 <= elapsed < 1.6 else .1
            sample.data['pressure_pa'] = robot['depth_sensor']['surface_pressure_mbar']*100 + depth*9806.65
        return sample

    monkeypatch.setattr(_RLSensorWorker, 'read_once', read)
    reference = robot['depth_sensor']['surface_pressure_mbar']*100 if manual_reference else None
    result = api.run_session(cfg, robot, tmp_path, h0=.3, k=.5, dry_run=True, max_control_s=2.,
                             surface_pressure_pa=reference)
    cal = json.loads((tmp_path/'calibration.json').read_text())
    assert cal['pressure_per_meter'] == pytest.approx(9806.65)
    assert cal['depth_conversion'] == 'pressure_drift_lookup'
    assert cal['p_surface_pa'] == pytest.approx(robot['depth_sensor']['surface_pressure_mbar']*100)
    assert cal['pressure_reference_mode'] == ('fixed_config' if manual_reference else 'sample_mean')
    assert (cal['depth_sample_count'] == 0) == manual_reference
    commands = [json.loads(v) for v in (tmp_path/'commands.jsonl').read_text().splitlines()]
    events = [json.loads(v) for v in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert result['completed_pairs'] == 2
    assert [(v['action']['b1'], v['action']['b2']) for v in commands] == [(0, 0), (1, 1), (0, 0), (1, -1)]
    assert [v['pair']['direction'] for v in commands] == ['diving', 'diving', 'surfacing', 'surfacing']
    assert commands[2]['action']['theta'] > 0 > commands[3]['action']['theta']
    assert not any(v['event'] == 'hold_endpoint' for v in events)
    assert json.loads((tmp_path/'result.json').read_text())['fault'] is None
    depth_rows = [json.loads(v) for v in (tmp_path/'raw_depth.jsonl').read_text().splitlines()]
    assert cal['pressure_drift_origin_t_ns'] == depth_rows[0]['t_ns']
    intervals = [(b['t_ns']-a['t_ns'])/1e9 for a,b in zip(depth_rows, depth_rows[1:])]
    assert sum(intervals)/len(intervals) == pytest.approx(.1, abs=.035)
    assert (tmp_path/'pressure_drift.json').is_file()
    decisions = [json.loads(v) for v in (tmp_path/'depth_control.jsonl').read_text().splitlines()]
    for row in decisions:
        assert row['depth_m'] == pytest.approx((row['raw_pressure_pa']-row['surface_pressure_pa'])/9806.65)


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


@pytest.mark.parametrize('direction', [1, -1], ids=['diving', 'surfacing'])
def test_pi_accumulates_actual_sample_intervals_and_freezes_pair(api, calibration, direction):
    p = api.FixedDivePolicy(calibration, h0=.4, k=.5, ki=.2)
    depth = .4 - direction*.2
    p.observe_depth(depth, 1_000_000_000)  # 启动标定阶段不积分。
    assert p.i_term == 0
    p.start_control(2_000_000_000)
    p.observe_depth(depth, 1_900_000_000)  # 控制前的旧样本仍不积分。
    p.observe_depth(depth, 2_000_000_000)
    p.observe_depth(depth, 2_100_000_000)
    p.observe_depth(depth, 2_300_000_000)
    assert p.i_term == pytest.approx(direction*.2*.5*.3)
    first = p.next_action(depth)
    assert first.theta == pytest.approx(-direction*(.5*.5+.03)*53)
    p.observe_depth(depth, 2_500_000_000)
    assert p.next_action(depth) == first
    p.acknowledge(first)
    second = p.next_action(depth)
    assert second.theta == -first.theta
    p.acknowledge(second)
    assert abs(p.next_action(depth).theta) > abs(first.theta)


@pytest.mark.parametrize('direction', [1, -1], ids=['dive_to_surface', 'surface_to_dive'])
def test_pi_saturation_prevents_windup_but_allows_unwinding(api, calibration, direction):
    p = api.FixedDivePolicy(calibration, h0=.4, k=1., ki=1.)
    p.start_control(0)
    for i in range(101):
        p.observe_depth(.4-direction*.4, i*100_000_000)
    assert p.i_term == 0  # P本身占满输出，不能再累加I。
    for i in range(101, 202):
        p.observe_depth(.4-direction*.2, i*100_000_000)
    assert p.i_term == pytest.approx(direction*.5)
    first = p.next_action(.4-direction*.2)
    assert first.theta == -direction*53
    assert p.pair_info['saturated'] is True
    # 刚穿越目标时旧I远大于新P；不能让下一组继续朝旧方向拍动。
    reverse_depth = .4+direction*.02
    p.observe_depth(reverse_depth, 20_200_000_000)
    assert p.i_term == pytest.approx(direction*.495)
    assert p.integral_status == 'unwinding'
    assert p.next_action(reverse_depth) == first
    p.acknowledge(first)
    second = p.next_action(reverse_depth)
    assert second.theta == direction*53  # 中途到达目标仍完成已冻结的组合。
    p.acknowledge(second)
    reverse_first = p.next_action(reverse_depth)
    assert reverse_first.theta == pytest.approx(direction*2.65)
    assert p.pair_info['direction'] == ('surfacing' if direction > 0 else 'diving')
    assert p.pair_info['integral_status'] == 'direction_reset'
    assert p.i_term == 0
    p.acknowledge(reverse_first)
    reverse_second = p.next_action(reverse_depth)
    assert reverse_second.theta == pytest.approx(-direction*2.65)
    assert reverse_second.b2 == -second.b2
    p.acknowledge(reverse_second)
    p.observe_depth(reverse_depth, 20_300_000_000)
    assert p.i_term == pytest.approx(-direction*.005)
    assert abs(p.next_action(reverse_depth).theta) > abs(reverse_first.theta)


@pytest.mark.parametrize('direction', [1, -1])
def test_pi_fast_integral_reversal_respects_new_direction_saturation(api, calibration, direction):
    p = api.FixedDivePolicy(calibration, h0=.4, k=1., ki=20.)
    p.start_control(0)
    p.observe_depth(.4-direction*.004, 0)
    p.observe_depth(.4-direction*.004, 100_000_000)
    assert p.i_term == pytest.approx(direction*.02)
    p.observe_depth(.4+direction*.4, 200_000_000)
    assert p.i_term == 0  # 新方向P已经占满输出，退积分跨零后也不能再累积。
    assert p.integral_status == 'anti_windup'


def test_pi_repeated_old_and_missing_samples_do_not_accumulate(api, calibration):
    p = api.FixedDivePolicy(calibration,h0=.4,k=.2,ki=.1)
    p.start_control(0)
    p.observe_depth(.2,0)
    p.observe_depth(.2,100_000_000)
    initial = p.i_term
    for t in [100_000_000,50_000_000,10_000_000_000]:
        p.observe_depth(.2,t)
    assert p.i_term == initial
    assert p.integral_status == 'gap_skipped'
    p.observe_depth(.2,10_100_000_000)
    assert p.i_term == pytest.approx(initial+.005)


def test_ki_zero_preserves_original_proportional_output(api, calibration):
    p = api.FixedDivePolicy(calibration,h0=.4,k=.5,ki=0)
    p.start_control(0)
    for i in range(100):
        p.observe_depth(.2,i*100_000_000)
    assert p.i_term == 0
    assert p.next_action(.2).theta == -13.25


@pytest.mark.parametrize('ki', [-1, math.nan, math.inf])
def test_invalid_integral_gain_rejected(api, calibration, ki):
    with pytest.raises(ValueError):
        api.FixedDivePolicy(calibration,h0=.4,ki=ki)


def test_pi_session_increases_amplitude_for_persistent_shallow_depth(api, monkeypatch, tmp_path):
    from control.runtime.rl_mock_sensors_20260914 import _RLSensorWorker
    root = Path(__file__).resolve().parents[1]
    robot = yaml.safe_load((root/'config/robot.yaml').read_text(encoding='utf-8'))
    raw = yaml.safe_load((root/'config/rl_training_20260914.yaml').read_text(encoding='utf-8'))
    cfg = api.prepare_fixed_config(raw,dry_run=True)
    original = _RLSensorWorker.read_once
    def read(worker):
        sample = original(worker)
        surface, _ = worker.manager._motion_state()
        if worker.name == 'depth' and not surface:
            sample.data['pressure_pa'] = robot['depth_sensor']['surface_pressure_mbar']*100 + .1*9806.65
        return sample
    monkeypatch.setattr(_RLSensorWorker,'read_once',read)
    result = api.run_session(cfg,robot,tmp_path,h0=.3,k=.2,ki=.2,dry_run=True,
                            action_duration_s=.2,max_control_s=1.5)
    commands = [json.loads(x) for x in (tmp_path/'commands.jsonl').read_text().splitlines()]
    pairs = commands[::2]
    assert len(pairs) >= 2 and result['completed_pairs'] >= 2
    assert json.loads((tmp_path/'result.json').read_text())['fault'] is None
    assert pairs[-1]['pair']['i_term_deg'] < pairs[0]['pair']['i_term_deg']
    assert abs(pairs[-1]['action']['theta']) > abs(pairs[0]['action']['theta'])
    for first,second in zip(commands[::2],commands[1::2]):
        assert second['action']['theta'] == -first['action']['theta']
    rows = [json.loads(x) for x in (tmp_path/'depth_control.jsonl').read_text().splitlines()]
    assert all(x['i_term_deg']==0 for x in rows if x['phase']=='servo_initialization')


def test_pi_integral_respects_asymmetric_tip_limits(api, calibration):
    calibration.limits[5] = (0.,108.)
    p = api.FixedDivePolicy(calibration,h0=.4,k=0.,ki=10.)
    p.start_control(0)
    for i in range(50):
        p.observe_depth(.2,i*100_000_000)
    assert p.i_term*53 == pytest.approx(10.6)
    previous = 0.
    for _ in range(4):
        action = p.next_action(.2)
        validate_trajectory(build_rl_trajectory(action,previous,calibration),calibration.limits)
        previous = action.theta
        p.acknowledge(action)
