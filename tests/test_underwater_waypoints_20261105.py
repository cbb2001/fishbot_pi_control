"""多目标到点/超时切换回归：使用手工时钟与虚拟执行器，不访问硬件。"""
from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from control.runtime.rl_actions_20260914 import build_calibration
from control.runtime.rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.surface_navigation_20261010 import NavigationConfig, UWBFrame
from control.runtime.underwater_navigation_20261026 import (
    AcceptedUWB, NavigationEstimator, UnderwaterConfig, UnderwaterScheduler,
    UnderwaterWaypoint, UWBObservation, validate_safety_depth,
)
from control.runtime.underwater_navigation_session_20261026 import NavigationMission


ROOT = Path(__file__).resolve().parents[1]


def robot_config():
    return yaml.safe_load((ROOT / 'config/robot.yaml').read_text(encoding='utf8'))


def config(points=None, **changes):
    nav = NavigationConfig(L=10, W=6, H=1, x_d=8, y_d=4, delta_s=.25)
    return UnderwaterConfig(nav, waypoints=points, **changes)


def mission(points, **changes):
    cfg = config(tuple(UnderwaterWaypoint(*point) for point in points),
                 **{'t3': 10., 't4': 5., **changes})
    cal = build_calibration(robot_config())
    now = [0]
    ex = RLServoExecutor(DryRunRLServoController(), UnderwaterScheduler(cal),
                         RLServoStateTracker(cal), clock_ns=lambda: now[0])
    events = []
    m = NavigationMission(cfg, cal, ex, lambda kind, **data: events.append((kind, data)))
    m.enter_navigation(now[0])
    m.on_depth(0, .1)
    return m, ex, now, events


def advance(m, ex, now, seconds, xyz=(1., 1., .1), *, acknowledge=True):
    now[0] = round(seconds * 1e9)
    m.tick(now[0], xyz, 0.)
    if acknowledge:
        ex._consume_requests()
        m.tick(now[0], xyz, 0.)
        ex._consume_requests()


def events_of(events, kind):
    return [data for name, data in events if name == kind]


def enter_acquire(m, ex, now):
    advance(m, ex, now, m.cfg.t4 + .01)
    assert m.phase == 'surface'
    now[0] += 1_000_000
    m.on_uwb(UWBObservation(now[0], UWBFrame((2., None, None, None), 'surface'), 0.), now[0])
    ex._consume_requests()
    m.tick(now[0], (1., 1., 0.), 0.)
    assert m.phase == 'acquire'
    assert all(value is None for value in ex.scheduler.active.values())


def test_legacy_single_point_resolves_to_null_deadline_and_is_immutable():
    cfg = config(h0=.6)
    assert cfg.route == (UnderwaterWaypoint(8., 4., .6, None),)
    with pytest.raises(FrozenInstanceError):
        cfg.route[0].x = 1.


def test_yaml_route_parses_mappings_sequences_null_and_zero_depth():
    robot = robot_config()
    robot['underwater_navigation_20261026']['waypoints'] = [
        {'x': 2, 'y': 3, 'h': .4, 't': 12}, [4, 5, 0, 'NULL'],
    ]
    cfg = UnderwaterConfig.from_robot(robot)
    assert cfg.route == (UnderwaterWaypoint(2, 3, .4, 12), UnderwaterWaypoint(4, 5, 0, None))


@pytest.mark.parametrize('token', [None, 'NULL', 'null', 'Null'])
def test_terminal_null_spellings(token):
    robot = robot_config()
    cfg = UnderwaterConfig.from_robot(robot, waypoints=[[1, 1, .3, token]])
    assert cfg.route[0].t is None


def test_explicit_route_overrides_yaml_route_and_single_target_flags_conflict():
    robot = robot_config()
    robot['underwater_navigation_20261026']['waypoints'] = [[1, 1, .3, None]]
    cfg = UnderwaterConfig.from_robot(robot, waypoints=[['2', '3', '.4', '4'], ['4', '5', '.6', 'NULL']])
    assert cfg.route == (UnderwaterWaypoint(2, 3, .4, 4), UnderwaterWaypoint(4, 5, .6))
    for flag in ('x_d', 'y_d', 'h0'):
        with pytest.raises(ValueError, match='waypoints'):
            UnderwaterConfig.from_robot(robot, **{flag: .3})


@pytest.mark.parametrize('rows', [
    [], [[1, 1, .3]], [{'x': 1, 'y': 1, 'h': .3}],
    [[1, 1, .3, 1]], [[1, 1, .3, None], [2, 2, .4, 5]],
    [[-1, 1, .3, None]], [[11, 1, .3, None]], [[1, 7, .3, None]],
    [[1, 1, -.1, None]], [[True, 1, .3, None]], [[1, 1, True, None]],
    [[1, 1, .3, True], [2, 2, .4, None]],
    [[1, 1, .3, 0], [2, 2, .4, None]],
    [[1, 1, .3, -1], [2, 2, .4, None]],
    [[1, 1, .3, math.inf], [2, 2, .4, None]],
    [[1, 1, .3, math.nan], [2, 2, .4, None]],
    [[1, 1, .3, 1e-20], [2, 2, .4, None]],
    [[math.nan, 1, .3, None]], [[1, 1, math.inf, None]],
])
def test_invalid_routes_rejected_before_motion(rows):
    with pytest.raises((ValueError, TypeError)):
        config(rows)


def test_safety_depth_checks_every_waypoint_including_later_points():
    cfg = config((UnderwaterWaypoint(2, 2, .1, 2), UnderwaterWaypoint(3, 3, 1.2)))
    with pytest.raises(ValueError):
        validate_safety_depth(cfg, 1.)
    validate_safety_depth(replace(cfg, waypoints=(UnderwaterWaypoint(1, 1, 0),)), 1.)


def test_xy_boundary_reaches_even_when_depth_is_far_from_target():
    m, ex, now, events = mission([(2, 2, .8, 3), (8, 4, .4, None)])
    advance(m, ex, now, .1, (1.75, 2., 0.))
    assert m.waypoint_index == 1
    assert m.target == UnderwaterWaypoint(8, 4, .4)
    assert m.depth_policy.h0 == .4
    completed = events_of(events, 'waypoint_completed')
    assert len(completed) == 1 and completed[0]['reason'] == 'reached'
    assert completed[0]['waypoint_index'] == 1
    assert completed[0]['target'] == [2, 2, .8]


def test_timeout_switches_at_deadline_and_does_not_reset_mission_or_t4():
    m, ex, now, events = mission([(2, 2, .4, .2), (8, 4, .7, None)])
    advance(m, ex, now, .199)
    assert m.waypoint_index == 0
    advance(m, ex, now, .2)
    assert m.waypoint_index == 1 and m.target.h == .7
    assert m.motion_start == 0 and m.segment_start == 0
    assert len(events_of(events, 'navigation_started')) == 1
    assert events_of(events, 'waypoint_completed')[0]['reason'] == 'timeout'
    started = events_of(events, 'waypoint_started')
    assert [event['waypoint_index'] for event in started] == [1, 2]
    assert started[-1]['timeout_s'] is None


def test_zero_depth_waypoint_uses_valid_depth_control_after_switch():
    m, ex, now, _ = mission([(2, 2, .4, .2), (8, 4, 0, None)])
    advance(m, ex, now, .2)
    assert m.target.h == 0 and m.depth_policy.h0 == 0
    assert m.depth_policy.error_scale_m > 0
    assert ex.scheduler.active['fin'] is not None
    assert m.depth_policy.pair_info['direction'] == 'surfacing'


def test_null_intermediate_point_never_advances_even_after_arrival():
    m, ex, now, events = mission([(2, 2, .4, None), (8, 4, .7, None)], t4=20)
    advance(m, ex, now, .1, (2., 2., .1))
    assert m.waypoint_index == 0 and ex.scheduler.active['tail'] is None
    advance(m, ex, now, 9., (1., 1., .1))
    assert m.waypoint_index == 0 and ex.scheduler.active['tail'] is not None
    assert not events_of(events, 'waypoint_completed')


def test_last_point_holds_then_recorrects_and_ends_only_at_t3():
    m, ex, now, events = mission([(2, 2, .4, .2), (8, 4, .7, None)], t4=20)
    advance(m, ex, now, .2)
    advance(m, ex, now, .3, (8., 4., .1))
    assert m.waypoint_index == 1 and m.reason is None
    assert ex.scheduler.active['tail'] is None
    assert ex.scheduler.active['fin'] is not None
    advance(m, ex, now, .4, (7., 4., .1))
    assert ex.scheduler.active['tail'] is not None
    advance(m, ex, now, 10., acknowledge=False)
    assert m.reason == 'time_limit'
    assert len(events_of(events, 'waypoint_completed')) == 1


def test_waypoint_ack_blocks_new_actions_and_late_completion_is_ignored():
    m, ex, now, events = mission([(2, 2, .4, 3), (8, 4, .7, None)])
    advance(m, ex, now, 0.)
    old_epoch = m.epoch
    old = ex.scheduler.active['fin']
    old_request = m.inflight['fin']
    submitted_before = len(events_of(events, 'action_submitted'))
    advance(m, ex, now, .1, (2., 2., .1), acknowledge=False)
    assert m.center_ticket is not None and not m.center_ticket.is_set()
    assert m.center_purpose == 'waypoint'
    advance(m, ex, now, .11, (2., 2., .1), acknowledge=False)
    assert len(events_of(events, 'action_submitted')) == submitted_before
    ex._consume_requests()
    m.tick(now[0], (2., 2., .1), 0.)
    ex._consume_requests()
    assert m.waypoint_index == 1 and m.depth_policy.h0 == .7
    assert m.epoch > old_epoch
    ex.completion_queue.put(SimpleNamespace(agent='fin', request_id=old_request,
        context=old_epoch, action=old.action, completion_t_ns=now[0]))
    m.tick(now[0], (2., 2., .1), 0.)
    assert events_of(events, 'late_completion_ignored')
    assert m.depth_policy.completed_pairs == 0


def test_t4_at_waypoint_deadline_enters_surface_without_new_navigation_motion():
    m, ex, now, events = mission([(2, 2, .4, .2), (8, 4, .7, None)], t4=.2)
    advance(m, ex, now, .2)
    assert m.phase == 'surface'
    assert ex.scheduler.active['tail'] is None
    assert len(events_of(events, 'navigation_started')) == 1
    assert events_of(events, 'surface_started')
    advance(m, ex, now, .21)
    assert m.waypoint_index == 1
    assert m.phase == 'surface' and ex.scheduler.active['tail'] is None


def test_delayed_waypoint_ack_past_t4_does_not_submit_new_target_motion():
    m, ex, now, events = mission([(2, 2, .4, 3), (8, 4, .7, None)], t4=.2)
    advance(m, ex, now, 0.)
    advance(m, ex, now, .19, (2., 2., .1), acknowledge=False)
    assert m.center_purpose == 'waypoint'
    submitted_before = len(events_of(events, 'action_submitted'))
    now[0] = 250_000_000
    ex._consume_requests()
    m.tick(now[0], (2., 2., .1), 0.)
    assert m.phase == 'surface' and m.center_purpose == 'surface'
    assert len(events_of(events, 'action_submitted')) == submitted_before
    assert m.motion_start == 0 and m.segment_start == 0


def test_waypoint_change_keeps_ekf_velocity_origin_covariance_and_segment_clock():
    cfg = config((UnderwaterWaypoint(2, 2, .4, .2), UnderwaterWaypoint(8, 4, .7)),
                 t3=10, t4=5, accel_tau_s=0)
    estimator = NavigationEstimator(cfg)
    gate = AcceptedUWB(replace(cfg, uwb_average_frames=1))
    ranges = tuple(math.sqrt((1-a)**2 + (1-b)**2 + 1)
                   for a, b in ((0, 0), (10, 0), (10, 6), (0, 6)))
    fix = gate.add(UWBObservation(0, UWBFrame(ranges, 'initial'), 0.), 0)
    estimator.apply_surface_fix(fix, 0)
    cal = build_calibration(robot_config())
    now = [0]
    ex = RLServoExecutor(DryRunRLServoController(), UnderwaterScheduler(cal),
                         RLServoStateTracker(cal), clock_ns=lambda: now[0])

    def event(kind, **data):
        # 会话的真实事件接线：若切点错误复用 navigation_started，会重置滤波与积分时钟。
        if kind == 'navigation_started':
            end = min(data['segment_start_t_ns'] + round(cfg.t4 * 1e9),
                      data['motion_start_t_ns'] + round(cfg.t3 * 1e9))
            estimator.start_navigation(data['segment_start_t_ns'], end)
        elif kind == 'surface_started':
            estimator.pause_navigation(m.stage_start)

    m = NavigationMission(cfg, cal, ex, event)
    m.enter_navigation(0)
    m.on_depth(0, .1)
    imu = {'acc_mps2': [1., .2, cfg.gravity_mps2], 'quat': [1., 0., 0., 0.]}
    for index in range(1, 41):
        estimator.imu(index * 5_000_000, imu)
    assert np.linalg.norm(estimator.v) > 0
    assert np.linalg.norm(estimator.relative_xy) > 0
    arrays = {name: getattr(estimator, name).copy() for name in ('p', 'v', 'a1', 'a2', 'origin_xy')}
    ekf_x, ekf_p = estimator.ekf.x.copy(), estimator.ekf.P.copy()
    clocks = (estimator.segment_start_ns, estimator.segment_end_ns,
              estimator.last_integrated_ns, estimator.last_filter_ns)
    relative = estimator.relative_xyz
    advance(m, ex, now, .2, estimator.xyz)
    assert m.waypoint_index == 1 and estimator.integrating
    assert (estimator.segment_start_ns, estimator.segment_end_ns,
            estimator.last_integrated_ns, estimator.last_filter_ns) == clocks
    for name, expected in arrays.items():
        np.testing.assert_array_equal(getattr(estimator, name), expected)
    np.testing.assert_array_equal(estimator.ekf.x, ekf_x)
    np.testing.assert_array_equal(estimator.ekf.P, ekf_p)
    assert estimator.relative_xyz == relative
    estimator.imu(205_000_000, imu)
    assert estimator.relative_xy[0] > relative[0]


def test_acquire_timeout_changes_target_but_waits_for_uwb_before_motion():
    m, ex, now, events = mission([(2, 2, .4, .4), (8, 4, .7, None)], t4=.2)
    enter_acquire(m, ex, now)
    advance(m, ex, now, .4)
    assert m.waypoint_index == 1 and m.phase == 'acquire'
    assert all(value is None for value in ex.scheduler.active.values())
    assert len(events_of(events, 'navigation_started')) == 1
    point_start = m.waypoint_start
    now[0] = 1_000_000_000
    m.enter_navigation(now[0])
    assert m.waypoint_start == point_start
    assert m.depth_policy.h0 == .7 and m.motion_start == 0
    assert m.segment_start == now[0]


def test_stale_frozen_xy_in_acquire_does_not_count_as_new_arrival():
    m, ex, now, events = mission([(2, 2, .4, 3), (8, 4, .7, None)], t4=.2)
    enter_acquire(m, ex, now)
    advance(m, ex, now, .4, (2., 2., 0.))
    assert m.waypoint_index == 0 and not events_of(events, 'waypoint_completed')


def test_navigation_resume_preserves_finite_waypoint_elapsed_time():
    m, ex, now, events = mission([(2, 2, .4, 2), (8, 4, .7, None)], t4=.2)
    enter_acquire(m, ex, now)
    now[0] = 1_900_000_000
    m.enter_navigation(now[0])
    assert m.waypoint_start == 0
    advance(m, ex, now, 2.)
    assert m.waypoint_index == 1
    assert events_of(events, 'waypoint_completed')[0]['reason'] == 'timeout'


def test_averaged_surface_uwb_position_can_trigger_next_target():
    m, ex, now, events = mission([(3, 2, .8, 3), (8, 4, .7, None)], t4=.2)
    enter_acquire(m, ex, now)
    gate = AcceptedUWB(m.cfg)
    gate.reset(now[0])
    ranges = tuple(math.sqrt((3-a)**2 + (2-b)**2 + 1)
                   for a, b in ((0, 0), (10, 0), (10, 6), (0, 6)))
    for index in range(1, 6):
        now[0] += 100_000_000
        fix = gate.add(UWBObservation(now[0], UWBFrame(ranges, str(index)), .8), now[0])
        if index < 5:
            assert fix is None and m.waypoint_index == 0 and m.phase == 'acquire'
    estimator = NavigationEstimator(m.cfg)
    estimator.apply_surface_fix(fix, now[0])
    assert estimator.xyz == pytest.approx([3., 2., 0.])
    m.enter_navigation(now[0])
    advance(m, ex, now, now[0] / 1e9, estimator.xyz)
    assert m.waypoint_index == 1
    assert events_of(events, 'waypoint_completed')[0]['reason'] == 'reached'


def test_overlapping_targets_advance_in_order_until_null():
    m, ex, now, events = mission([(2, 2, .4, 2), (2, 2, .7, 2),
                                 (2, 2, .2, None), (8, 4, .4, None)])
    for index in range(1, 8):
        advance(m, ex, now, index * .01, (2., 2., 0.))
    assert m.waypoint_index == 2
    assert [event['waypoint_index'] for event in events_of(events, 'waypoint_completed')] == [1, 2]
    assert len(events_of(events, 'navigation_started')) == 1


@pytest.mark.parametrize('waypoint_args', [
    ['--waypoint', '2', '2', '.3', '5'],
    ['--waypoint', '2', '2', '.3', '0', '--waypoint', '3', '3', '.4', 'NULL'],
    ['--waypoint', '2', '2', '.3', 'NULL', '--x-d', '3'],
])
def test_cli_invalid_route_exits_before_creating_session(tmp_path, waypoint_args):
    output = tmp_path / 'must_not_start'
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/run_underwater_navigation_20261026.py'),
        '--dry-run', '--output', str(output), *waypoint_args], cwd=ROOT,
        capture_output=True, text=True, encoding='utf8', timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists()


def test_cli_repeated_waypoints_override_yaml_route(tmp_path, monkeypatch):
    robot = robot_config()
    robot['underwater_navigation_20261026']['waypoints'] = [[1, 1, .3, None]]
    source = tmp_path / 'robot.yaml'
    source.write_text(yaml.safe_dump(robot), encoding='utf8')
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    namespace = runpy.run_path(str(ROOT / 'scripts/run_underwater_navigation_20261026.py'),
                               run_name='underwater_waypoint_cli_test')
    captured = []

    def capture_run(_robot, cfg, _session, **kwargs):
        captured.append((cfg, kwargs))
        return {'reason': 'time_limit', 'fault': None, 'cleanup_errors': []}

    monkeypatch.setitem(namespace['main'].__globals__, 'run_session', capture_run)
    monkeypatch.setattr(sys, 'argv', ['run_underwater_navigation_20261026.py',
        '--config', str(source), '--dry-run', '--output', str(tmp_path / 'session'),
        '--waypoint', '2', '3', '.4', '4', '--waypoint', '4', '5', '.6', 'NULL'])
    assert namespace['main']() == 0
    cfg, options = captured[0]
    assert options['dry_run'] is True
    assert cfg.route == (UnderwaterWaypoint(2, 3, .4, 4), UnderwaterWaypoint(4, 5, .6))
