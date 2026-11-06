"""Only-one-yaw mission regressions using the real scheduler/PWM acknowledgements."""
import json
import threading
from dataclasses import replace

import pytest
import yaml

from control.runtime.rl_actions_20260914 import build_calibration
from control.runtime.rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.surface_navigation_20261010 import UWBFrame
from control.runtime.underwater_navigation_20261026 import RawUWBFix, UWBObservation, UnderwaterScheduler
from control.runtime.underwater_navigation_20261026_only_one_yaw import UnderwaterConfig
from control.runtime.underwater_navigation_session_20261026_only_one_yaw import NavigationMission, run_session


def robot_config():
    with open('config/robot.yaml', encoding='utf8') as stream:
        return yaml.safe_load(stream)


def fix(x=1., y=1., t_ns=0):
    # The mission receives only the already-accepted mean fix from AcceptedUWB.
    return RawUWBFix(t_ns, f'accepted:{t_ns}', (2., 3., 4., 5.), [x, y], [0., 0.],
                     {'mode': 'mean_valid_ranges', 'frames': 5})


def mission(*, waypoints=((5., 1., .3, None),), t3=10., t4=2., yaw=0.):
    robot = robot_config()
    base = UnderwaterConfig.from_robot(robot)
    cfg = replace(base, t3=t3, t4=t4, waypoints=waypoints,
                  navigation=replace(base.navigation, x_d=waypoints[0][0], y_d=waypoints[0][1],
                                     ki=0., delta_yaw=10., delta_s=.2))
    cal = build_calibration(robot)
    now = [0]
    scheduler = UnderwaterScheduler(cal)
    ex = RLServoExecutor(DryRunRLServoController(), scheduler, RLServoStateTracker(cal),
                         clock_ns=lambda: now[0])
    events = []
    m = NavigationMission(cfg, cal, ex, lambda kind, **data: events.append((kind, data)))
    m.enter_surface_alignment(fix(), now[0], yaw)
    ex._consume_requests()  # Initial all-group center acknowledged before alignment.
    m.depth = .1
    return m, ex, now, events


def poll(m, ex, now, *, yaw=0., xyz=(1., 1., 0.)):
    m.tick(now[0], xyz, yaw)
    ex._consume_requests()


def start_dive(m, ex, now, *, yaw=0.):
    for _ in range(4):
        poll(m, ex, now, yaw=yaw)
        if m.phase == 'navigate':
            return
    raise AssertionError(f'no dive after alignment and center acknowledgement: {m.phase}')


def complete_half(m, ex, now, *, yaw=0.):
    active = ex.scheduler.active['tail']
    assert active is not None
    now[0] = active.start_t_ns + round(active.action.t * 1e9)
    ex.tick_once()
    poll(m, ex, now, yaw=yaw)


def test_surface_alignment_uses_existing_large_angle_strokes_without_pectoral_motion():
    m, ex, now, _ = mission(yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    first = ex.scheduler.active['tail'].action
    assert first.theta == 30.
    assert ex.scheduler.active['fin'] is None
    complete_half(m, ex, now, yaw=-90.)
    assert ex.scheduler.active['tail'].action.theta == 20.
    complete_half(m, ex, now, yaw=-30.)
    assert ex.scheduler.active['tail'].action.theta < 0.  # First small-angle PI half.
    assert ex.scheduler.active['fin'] is None
    assert all(ex.scheduler.committed_angles[sid] == ex.scheduler.centers[sid]
               for sid in (4, 5, 6, 7))
    assert m.phase != 'navigate'


def test_fins_wait_for_center_ack_and_yaw_is_latched_from_latest_surface_heading():
    m, ex, now, _ = mission(yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 100_000_000
    m.tick(now[0], [1., 1., 0.], -10.)
    assert m.phase != 'navigate'
    assert ex.scheduler.active['fin'] is None
    # A queued center request is not evidence that the tail is centered yet.
    m.tick(now[0], [1., 1., 0.], -9.)
    assert m.phase != 'navigate'
    assert ex.scheduler.active['fin'] is None
    ex._consume_requests()
    start_dive(m, ex, now, yaw=-8.)
    assert m.frozen_yaw_error_deg == pytest.approx(8.)
    assert m.segment_start == now[0]
    assert m.motion_start == 0
    poll(m, ex, now, yaw=-170.)
    assert ex.scheduler.active['fin'] is not None


def test_heading_drift_during_center_ack_resumes_surface_alignment():
    m, ex, now, _ = mission(yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 100_000_000
    m.tick(now[0], [1., 1., 0.], -8.)
    ex._consume_requests()
    poll(m, ex, now, yaw=-20.)
    assert m.phase != 'navigate'
    assert ex.scheduler.active['fin'] is None
    assert ex.scheduler.active['tail'] is not None
    start_dive(m, ex, now, yaw=-5.)
    assert m.frozen_yaw_error_deg == pytest.approx(5.)


def test_dive_waits_for_imu_sample_collected_after_center_ack():
    m, ex, now, events = mission(yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 100_000_000
    m.tick(now[0], [1., 1., 0.], -8., yaw_t_ns=now[0]-20_000_000)
    ex._consume_requests()
    ack = m.center_ticket.ack_t_ns
    for sample_t in (ack-1, ack):
        m.tick(now[0], [1., 1., 0.], -8., yaw_t_ns=sample_t)
        assert m.phase == 'align'
        assert ex.scheduler.active['fin'] is None
        assert m.frozen_yaw_error_deg is None
    now[0] = ack+5_000_000
    m.tick(now[0], [1., 1., 0.], -6., yaw_t_ns=now[0])
    ex._consume_requests()
    assert m.phase == 'navigate'
    assert m.frozen_yaw_error_deg == pytest.approx(6.)
    frozen = [data for kind, data in events if kind == 'yaw_frozen'][-1]
    assert frozen['yaw_sample_t_ns'] > frozen['center_ack_t_ns']


def test_underwater_position_and_live_imu_cannot_change_frozen_yaw_or_advance_waypoint():
    route = ((5., 1., .3, 30.), (8., 1., .6, None))
    m, ex, now, _ = mission(waypoints=route, yaw=-8., t4=5.)
    start_dive(m, ex, now, yaw=-8.)
    expected = m.frozen_policy.decide(now[0])
    for index, yaw in enumerate((179., -179., 75., -120.), 1):
        now[0] = index * 100_000_000
        poll(m, ex, now, yaw=yaw, xyz=(5., 1., .3))
        actual = m.frozen_policy.decide(now[0])
        assert actual.yaw_error_deg == expected.yaw_error_deg
        assert (actual.theta_left, actual.theta_right) == pytest.approx(
            (expected.theta_left, expected.theta_right))
        assert m.waypoint_index == 0
        assert m.phase == 'navigate'
    assert not m.tail_held


def test_equal_t4_t3_has_no_intermediate_surface_even_when_alignment_takes_time():
    m, ex, now, events = mission(t3=2., t4=2., yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 500_000_000
    start_dive(m, ex, now, yaw=0.)
    now[0] = 1_999_000_000
    poll(m, ex, now, yaw=100.)
    assert m.phase == 'navigate' and m.reason is None
    now[0] = 2_000_000_000
    poll(m, ex, now, yaw=100.)
    assert m.reason == 'time_limit'
    assert not any(kind == 'surface_started' for kind, _ in events)


def test_total_time_bounds_surface_alignment_before_any_dive():
    m, ex, now, events = mission(t3=1., t4=1., yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 1_000_000_000
    poll(m, ex, now, yaw=-90.)
    assert m.reason == 'time_limit'
    assert not any(kind == 'navigation_started' for kind, _ in events)
    assert ex.scheduler.active['fin'] is None


def surface_and_acquire(m, ex, now):
    now[0] = m.segment_start + round(m.cfg.t4 * 1e9)
    poll(m, ex, now)
    assert m.phase == 'surface'
    assert all(item is None for item in ex.scheduler.active.values())
    poll(m, ex, now)
    assert ex.scheduler.active['fin'] is not None
    assert ex.scheduler.active['tail'] is None
    now[0] += 1
    m.on_uwb(UWBObservation(now[0], UWBFrame((None, 2., None, None), 'surface-signal'), 0.), now[0])
    assert m.phase == 'surface'
    m.tick(now[0], [1., 1., 0.], 0.)
    assert m.phase == 'surface'  # No average acquisition before the PWM center ack.
    ex._consume_requests()
    m.tick(now[0], [1., 1., 0.], 0.)
    assert m.phase == 'acquire'
    assert all(item is None for item in ex.scheduler.active.values())


def test_t4_preserves_partial_uwb_surface_stop_and_acquire_center_barrier():
    m, ex, now, _ = mission(t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    assert ex.scheduler.fin_sequence.next_b1 == 0
    now[0] += 1
    m.enter_surface_alignment(fix(2., 1., now[0]), now[0], -80.)
    poll(m, ex, now, yaw=-80.)
    poll(m, ex, now, yaw=-80.)
    assert ex.scheduler.active['tail'] is not None
    assert ex.scheduler.active['fin'] is None
    assert m.motion_start == 0


def test_only_accepted_surface_position_switches_to_next_target_and_depth():
    route = ((5., 1., .3, 30.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(5.05, 1., now[0]), now[0], 0.)
    assert m.waypoint_index == 1
    assert m.target == route[1]
    start_dive(m, ex, now)
    assert m.depth_policy.h0 == .6
    assert m.motion_start == 0
    assert m.frozen_policy.decide(now[0]).distance_m == pytest.approx(2.95)


def test_final_arrival_keeps_tail_centered_and_depth_control_active_until_total_time():
    route = ((5., 1., .3, 30.), (8., 1., .6, None))
    m, ex, now, _ = mission(waypoints=route, t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(5., 1., now[0]), now[0], 0.)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(8.05, 1., now[0]), now[0], 170.)
    start_dive(m, ex, now, yaw=170.)
    poll(m, ex, now, yaw=170.)
    assert m.waypoint_index == len(route) - 1
    assert m.reason is None
    assert ex.scheduler.active['tail'] is None
    assert ex.scheduler.active['fin'] is not None
    assert m.depth_policy.h0 == .6
    now[0] = round(m.cfg.t3 * 1e9)
    poll(m, ex, now, yaw=170.)
    assert m.reason == 'time_limit'


def test_one_accepted_fix_can_skip_overlapping_intermediate_targets():
    route = ((5., 1., .3, 30.), (5.1, 1., .5, 30.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(5.05, 1., now[0]), now[0], 0.)
    assert m.waypoint_index == 2
    assert m.target == route[-1]
    reached = [data['waypoint_index'] for kind, data in events if kind == 'waypoint_reached']
    assert reached == [0, 1]
    start_dive(m, ex, now)
    assert m.depth_policy.h0 == .6


def test_final_hold_can_resume_navigation_only_after_another_accepted_surface_fix():
    m, ex, now, _ = mission(t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(5., 1., now[0]), now[0], 170.)
    start_dive(m, ex, now, yaw=170.)
    assert m.final_reached
    poll(m, ex, now, yaw=0., xyz=(1., 1., .3))
    assert m.final_reached and ex.scheduler.active['tail'] is None
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(4., 1., now[0]), now[0], 0.)
    start_dive(m, ex, now)
    assert not m.final_reached
    assert m.waypoint_index == 0
    assert ex.scheduler.active['tail'] is not None


def test_null_intermediate_point_never_advances_on_arrival_or_elapsed_time():
    route = ((5., 1., .3, None), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, t3=2000., t4=.2)
    start_dive(m, ex, now)
    assert m.waypoint_start == 0
    surface_and_acquire(m, ex, now)
    now[0] += 1
    m.enter_surface_alignment(fix(5.05, 1., now[0]), now[0], 170.)
    start_dive(m, ex, now, yaw=170.)
    assert m.waypoint_index == 0 and m.target == route[0]
    assert m.target_reached and not m.final_reached
    assert ex.scheduler.active['tail'] is None
    assert ex.scheduler.active['fin'] is not None
    now[0] = 1_000_000_000_000
    assert not m.update_waypoint_timeout(now[0])
    assert m.waypoint_index == 0 and m.waypoint_start == 0
    assert not any(kind == 'waypoint_changed' for kind, _ in events)


def test_numeric_point_timeout_forces_fresh_surface_fix_even_when_t4_equals_t3():
    route = ((5., 1., .3, 1.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, t3=10., t4=10., yaw=-8.)
    start_dive(m, ex, now, yaw=-8.)
    assert ex.scheduler.active['tail'] is not None
    assert ex.scheduler.active['fin'] is not None
    now[0] = 999_999_999
    poll(m, ex, now, yaw=-170.)
    assert m.waypoint_index == 0
    now[0] = 1_000_000_000
    poll(m, ex, now, yaw=-170.)
    assert m.waypoint_index == 1 and m.target == route[1]
    assert m.waypoint_start == now[0] and m.motion_start == 0
    assert m.phase == 'surface'
    assert m.frozen_yaw_error_deg is None
    assert all(action is None for action in ex.scheduler.active.values())
    changes = [data for kind, data in events if kind == 'waypoint_changed']
    assert len(changes) == 1 and changes[0]['reason'] == 'timeout'
    poll(m, ex, now)
    assert ex.scheduler.active['fin'] is not None
    assert ex.scheduler.active['tail'] is None
    now[0] += 1
    m.on_uwb(UWBObservation(now[0], UWBFrame((2., None, None, None), 'fresh-signal'), 0.), now[0])
    poll(m, ex, now)
    poll(m, ex, now)
    assert m.phase == 'acquire'
    assert m.frozen_yaw_error_deg is None
    now[0] += 1
    m.enter_surface_alignment(fix(2., 1., now[0]), now[0], -6.)
    start_dive(m, ex, now, yaw=-6.)
    assert m.frozen_yaw_error_deg == pytest.approx(6.)
    assert m.depth_policy.h0 == .6


def test_timeout_during_alignment_cancels_tail_and_waits_for_center_then_new_uwb():
    route = ((5., 1., .3, 1.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    assert ex.scheduler.active['tail'] is not None
    now[0] = 1_000_000_000
    m.tick(now[0], [1., 1., 0.], -90.)
    assert m.waypoint_index == 1
    assert m.phase == 'reacquire'
    assert m.center_ticket is not None and m.center_purpose == 'acquire'
    assert ex.scheduler.active['fin'] is None
    ex._consume_requests()
    m.tick(now[0], [1., 1., 0.], -90.)
    assert m.phase == 'acquire'
    assert all(action is None for action in ex.scheduler.active.values())
    assert m.frozen_yaw_error_deg is None
    assert not any(kind == 'navigation_started' for kind, _ in events)


def test_timeout_preempts_pending_dive_center_ack_without_starting_fins():
    route = ((5., 1., .3, 1.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 900_000_000
    m.tick(now[0], [1., 1., 0.], -8.)
    assert m.center_purpose == 'dive'
    now[0] = 1_000_000_000
    m.tick(now[0], [1., 1., 0.], -8.)
    assert m.waypoint_index == 1 and m.center_purpose == 'acquire'
    ex._consume_requests()
    poll(m, ex, now, yaw=-8.)
    assert m.phase == 'acquire'
    assert ex.scheduler.active['fin'] is None
    assert not any(kind == 'yaw_frozen' for kind, _ in events)


def test_timeout_while_surfacing_preserves_surface_sequence_and_stage_boundary():
    route = ((5., 1., .3, 1.), (8., 1., .6, None))
    m, ex, now, _ = mission(waypoints=route, t4=.2)
    start_dive(m, ex, now)
    now[0] = 200_000_000
    poll(m, ex, now)
    poll(m, ex, now)
    assert m.phase == 'surface'
    old_stage, old_policy = m.stage_start, m.surface_policy
    old_request = m.inflight['fin']
    now[0] = 1_000_000_000
    poll(m, ex, now)
    assert m.waypoint_index == 1
    assert m.phase == 'surface' and m.stage_start == old_stage
    assert m.surface_policy is old_policy
    assert m.inflight['fin'] == old_request
    assert m.center_ticket is None


def test_point_budget_includes_alignment_surface_and_acquire_without_reset_on_refix():
    route = ((5., 1., .3, 3.), (8., 1., .6, None))
    m, ex, now, _ = mission(waypoints=route, t3=30., t4=.2, yaw=-90.)
    poll(m, ex, now, yaw=-90.)
    now[0] = 500_000_000
    start_dive(m, ex, now)
    assert m.segment_start == now[0] and m.waypoint_start == 0
    surface_and_acquire(m, ex, now)
    now[0] = 900_000_000
    m.enter_surface_alignment(fix(2., 1., now[0]), now[0], 0.)
    start_dive(m, ex, now)
    assert m.waypoint_start == 0  # A correction does not grant a new point budget.
    surface_and_acquire(m, ex, now)
    boundary = m.stage_start
    now[0] = 2_999_999_999
    poll(m, ex, now)
    assert m.waypoint_index == 0
    now[0] = 3_000_000_000
    poll(m, ex, now)
    assert m.waypoint_index == 1 and m.waypoint_start == now[0]
    assert m.phase == 'acquire' and m.stage_start == boundary
    assert m.center_ticket is None
    assert m.motion_start == 0


def test_total_deadline_wins_over_simultaneous_point_timeout():
    route = ((5., 1., .3, 1.), (8., 1., .6, None))
    m, ex, now, events = mission(waypoints=route, t3=1., t4=1.)
    start_dive(m, ex, now)
    now[0] = 1_000_000_000
    assert not m.update_waypoint_timeout(now[0])
    poll(m, ex, now)
    assert m.reason == 'time_limit'
    assert m.waypoint_index == 0
    assert not any(kind == 'waypoint_changed' for kind, _ in events)


def test_timeout_preserves_completed_fin_pair_b2_before_cancelling_old_controls():
    route = ((5., 1., .3, .6), (8., 1., .6, None))
    m, ex, now, _ = mission(waypoints=route, t4=10.)
    start_dive(m, ex, now)
    first = ex.scheduler.active['fin'].action
    assert first.b1 == 0
    now[0] = 300_000_000
    ex.tick_once()
    poll(m, ex, now)
    second = ex.scheduler.active['fin'].action
    assert second.b1 == 1
    now[0] = 600_000_000
    ex.tick_once()  # Completion is queued at the same instant as point timeout.
    poll(m, ex, now)
    assert m.waypoint_index == 1 and m.phase == 'surface'
    assert ex.scheduler.fin_sequence.last_b2 == second.b2
    assert ex.scheduler.fin_sequence.next_b1 == 0
    poll(m, ex, now)
    assert m.surface_policy.sequence.last_b2 == second.b2


def test_reaching_numeric_target_starts_next_budget_at_transition_not_initial_launch():
    route = ((5., 1., .3, 3.), (8., 1., .6, 2.), (9., 1., .4, None))
    m, ex, now, events = mission(waypoints=route, t3=10., t4=.2)
    start_dive(m, ex, now)
    surface_and_acquire(m, ex, now)
    now[0] = 1_000_000_000
    m.enter_surface_alignment(fix(5., 1., now[0]), now[0], 0.)
    start_dive(m, ex, now)
    assert m.waypoint_index == 1 and m.waypoint_start == 1_000_000_000
    assert m.motion_start == 0
    now[0] = 2_999_999_999
    assert not m.update_waypoint_timeout(now[0])
    now[0] += 1
    assert m.update_waypoint_timeout(now[0])
    assert m.waypoint_index == 2 and m.waypoint_start == 3_000_000_000
    changes = [data['reason'] for kind, data in events if kind == 'waypoint_changed']
    assert changes == ['reached', 'timeout']


def dry_config(robot, *, t3=2.5):
    cfg = UnderwaterConfig.from_robot(robot)
    # Mock UWB sits at (L/4,W/2); use an eastward target for short cycle tests.
    return replace(cfg, t1=0., t2=.6, t3=t3, t4=.2,
                   waypoints=((cfg.navigation.L/2., cfg.navigation.W/2., .3, None),))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def test_dry_session_reacquires_five_frames_and_never_updates_underwater_yaw(tmp_path):
    robot = robot_config()
    cfg = dry_config(robot)
    result = run_session(robot, cfg, tmp_path, dry_run=True)
    assert result['reason'] == 'time_limit' and result['fault'] is None, result
    assert not result['cleanup_errors'] and not result['alive_workers']
    events = read_jsonl(tmp_path/'events.jsonl')
    starts = [row for row in events if row['event'] == 'navigation_started']
    assert len(starts) >= 2
    assert all(abs(row['frozen_yaw_error_deg']) <= cfg.navigation.delta_yaw for row in starts)
    frozen = [row for row in events if row['event'] == 'yaw_frozen']
    assert all(row['yaw_sample_t_ns'] > row['center_ack_t_ns'] for row in frozen)
    corrections = [row for row in events if row['event'] == 'position_corrected']
    assert any(row['phase'] == 'acquire' for row in corrections)
    assert all(row['quality']['frames'] == 5 and row['xyz'][2] == 0 for row in corrections)
    samples = read_jsonl(tmp_path/'imu_filtered.jsonl')
    underwater = [row for row in samples if row['phase'] == 'navigate']
    assert underwater and all(not row['yaw_updated'] for row in underwater)
    assert all(not row['integrating'] for row in samples if row['phase'] != 'navigate')
    centers = result['terminal_center']
    assert centers['hold_completed_t_ns']-centers['command_ack_t_ns'] >= cfg.center_settle_s*1e9
    assert centers['pwm_disabled_t_ns'] >= centers['hold_completed_t_ns']


def test_dry_timeout_switch_requires_new_mean_fix_before_next_surface_yaw_freeze(tmp_path):
    robot = robot_config()
    cfg = dry_config(robot, t3=3.5)
    cfg = replace(cfg, t4=cfg.t3, waypoints=(
        (cfg.navigation.L/2., cfg.navigation.W/2., .3, .3),
        (cfg.navigation.L*.75, cfg.navigation.W/2., .4, None),
    ))
    result = run_session(robot, cfg, tmp_path, dry_run=True)
    assert result['reason'] == 'time_limit' and result['fault'] is None, result
    assert not result['cleanup_errors'] and not result['alive_workers']
    assert result['waypoint_index'] == 1
    events = read_jsonl(tmp_path/'events.jsonl')
    starts = [row for row in events if row['event'] == 'navigation_started']
    assert [row['waypoint_index'] for row in starts] == [0, 1]
    changes = [row for row in events if row['event'] == 'waypoint_changed']
    assert len(changes) == 1 and changes[0]['reason'] == 'timeout'
    after_change = events[events.index(changes[0])+1:]
    surface = next(row for row in after_change if row['event'] == 'surface_started')
    assert surface['reason'] == 'waypoint_timeout'
    acquire = next(row for row in after_change if row['event'] == 'uwb_acquire_started')
    corrected = next(row for row in after_change if row['event'] == 'position_corrected')
    aligned = next(row for row in after_change if row['event'] == 'surface_alignment_started')
    frozen = next(row for row in after_change if row['event'] == 'yaw_frozen')
    assert corrected['quality']['frames'] == cfg.uwb_average_frames
    assert min(corrected['quality']['sample_times_ns']) > acquire['center_ack_t_ns']
    assert after_change.index(acquire) < after_change.index(corrected) < after_change.index(aligned)
    assert after_change.index(aligned) < after_change.index(frozen)
    assert frozen['fix_t_ns'] == corrected['fix_t_ns']
    assert frozen['yaw_sample_t_ns'] > frozen['center_ack_t_ns']
    samples = read_jsonl(tmp_path/'imu_filtered.jsonl')
    assert all(not row['yaw_updated'] for row in samples if row['phase'] == 'navigate')


def test_four_valid_frames_after_surface_do_not_realign_or_resume(tmp_path, monkeypatch):
    import control.runtime.underwater_navigation_session_20261026_only_one_yaw as module
    original = module._UnderwaterMockWorker.read_once
    count = [0]

    def read_once(worker):
        sample = original(worker)
        m = worker.manager.mission
        if worker.name == 'uwb' and m is not None and m.phase == 'acquire':
            count[0] += 1
            if count[0] > 4:
                sample.data['raw_hex'] = b'RANGE_ERROR\r\n'.hex()
        return sample

    monkeypatch.setattr(module._UnderwaterMockWorker, 'read_once', read_once)
    robot = robot_config()
    result = run_session(robot, dry_config(robot, t3=2.), tmp_path, dry_run=True)
    assert result['reason'] == 'time_limit' and result['fault'] is None, result
    events = read_jsonl(tmp_path/'events.jsonl')
    assert any(row['event'] == 'uwb_acquire_started' for row in events)
    assert sum(row['event'] == 'surface_alignment_started' for row in events) == 1
    assert sum(row['event'] == 'navigation_started' for row in events) == 1
    assert not any(row['event'] == 'position_corrected' and row['phase'] == 'acquire' for row in events)


def test_no_servo_does_not_require_uwb_or_initialize_pwm(tmp_path):
    from control.runtime.underwater_navigation_session_20261026_only_one_yaw import UnderwaterMockSensors
    robot = robot_config()
    cfg = dry_config(robot, t3=.2)
    stop = threading.Event()
    sensors = UnderwaterMockSensors(robot, cfg, stop, None)
    sensors.disable_uwb = True

    def forbidden():
        raise AssertionError('no-servo initialized PWM')

    result = run_session(robot, cfg, tmp_path, no_servo=True, sensors=sensors,
                         controller=forbidden, stop_event=stop)
    assert result['reason'] == 'time_limit' and result['fault'] is None, result
    assert not result['executor_started']
    assert not result['cleanup_errors'] and not result['alive_workers']

