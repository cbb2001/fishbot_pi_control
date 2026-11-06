"""Configuration and surface-frozen tail control regressions (no hardware)."""
import copy
import importlib.util
import inspect
import math
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
import yaml

from control.runtime.rl_actions_20260914 import build_calibration
from control.runtime.surface_navigation_20261010 import NavigationConfig, NavigationPolicy
from control.runtime.underwater_navigation_20261026 import UnderwaterConfig as BaseConfig
from control.runtime.underwater_navigation_20261026_only_one_yaw import (
    UnderwaterConfig, FrozenYawPolicy, validate_safety_depth,
)


def robot_config():
    return yaml.safe_load(Path('config/robot.yaml').read_text(encoding='utf8'))


def nav_config(**overrides):
    values = dict(L=10, W=6, H=1, x_d=8, y_d=3, delta_yaw=10,
                  kp=.5, ki=.2, yaw_error_scale_deg=90)
    values.update(overrides)
    return NavigationConfig(**values)


def test_default_config_preserves_every_original_field_and_uses_single_point():
    robot = robot_config()
    original = copy.deepcopy(robot)
    old = BaseConfig.from_robot(robot)
    new = UnderwaterConfig.from_robot(robot)
    for field in fields(BaseConfig):
        if field.name == 'waypoints':
            continue  # 两个入口的路线表示不同，下面单独验证本入口的四元点表。
        assert getattr(new, field.name) == getattr(old, field.name), field.name
    assert new.waypoints == ((old.navigation.x_d, old.navigation.y_d, old.h0, None),)
    assert robot == original


def test_main_four_component_route_does_not_override_only_one_yaw_route():
    robot = robot_config()
    robot['underwater_navigation_20261026']['waypoints'] = [
        {'x': 1, 'y': 2, 'h': .3, 't': 10},
        {'x': 2, 'y': 3, 'h': .4, 't': None},
    ]
    original = copy.deepcopy(robot)
    cfg = UnderwaterConfig.from_robot(robot)
    assert cfg.waypoints == ((cfg.navigation.x_d, cfg.navigation.y_d, cfg.h0, None),)
    assert robot == original
    robot['underwater_navigation_20261026_only_one_yaw'] = {'waypoints': [[3, 4, .6, None]]}
    assert UnderwaterConfig.from_robot(robot).waypoints == ((3., 4., .6, None),)
    assert UnderwaterConfig.from_robot(robot, waypoints=[[4, 5, .7, None]]).waypoints == ((4., 5., .7, None),)


def test_overlay_and_explicit_overrides_have_predictable_precedence():
    robot = robot_config()
    robot['underwater_navigation_20261026_only_one_yaw'] = {
        't3': 45, 't4': 45, 'kp': .6, 'delta_yaw': 8,
        'waypoints': [[1, 2, .3, 20], [2, 3, .4, None]],
    }
    original = copy.deepcopy(robot)
    overlay = UnderwaterConfig.from_robot(robot)
    assert overlay.t3 == overlay.t4 == 45
    assert overlay.navigation.kp == .6
    assert overlay.navigation.delta_yaw == 8
    assert overlay.waypoints == ((1., 2., .3, 20.), (2., 3., .4, None))
    explicit = UnderwaterConfig.from_robot(
        robot, t3=60, kp=.7, waypoints=[[2, 2, .6, None]], t4=None)
    assert explicit.t3 == 60 and explicit.t4 == 45
    assert explicit.navigation.kp == .7
    assert explicit.waypoints == ((2., 2., .6, None),)
    assert robot == original


def test_single_point_cli_fields_remain_supported_without_point_table():
    cfg = UnderwaterConfig.from_robot(robot_config(), x_d=1, y_d=2, h0=.7, delta_yaw=4)
    assert cfg.waypoints == ((1., 2., .7, None),)
    assert cfg.navigation.delta_yaw == 4


@pytest.mark.parametrize('points', [
    [], '1,2,.3,NULL', [[1, 2]], [[1, 2, .3]], [[1, 2, .3, None, 5]], [None],
    [[True, 2, .3, None]], [[1, False, .3, None]], [[1, 2, True, None]],
    [[-1, 2, .3, None]], [[11, 2, .3, None]], [[1, -1, .3, None]], [[1, 7, .3, None]],
    [[1, 2, 0, None]], [[1, 2, -.1, None]], [[math.nan, 2, .3, None]],
    [[1, math.inf, .3, None]], [[1, 2, math.nan, None]], [[1, 2, math.inf, None]],
])
def test_waypoint_validation_rejects_malformed_or_nonphysical_values(points):
    with pytest.raises((ValueError, TypeError)):
        UnderwaterConfig(nav_config(), waypoints=points)


def test_all_points_are_validated_against_safety_depth_before_start():
    robot = robot_config()
    limit = float(robot['safety']['max_depth_m'])
    with pytest.raises(ValueError, match='第 2 个目标点'):
        UnderwaterConfig.from_robot(robot, waypoints=[[1, 2, .3, 20], [2, 3, limit, None]])
    cfg = UnderwaterConfig(nav_config(), waypoints=((1, 2, .3, 20), (2, 3, .8, None)))
    with pytest.raises(ValueError, match='第 2 个目标点'):
        validate_safety_depth(cfg, .8)
    validate_safety_depth(cfg, .81)


@pytest.mark.parametrize('threshold', [45, 90, 180])
def test_alignment_threshold_requires_small_angle_entry(threshold):
    with pytest.raises(ValueError, match='delta_yaw'):
        UnderwaterConfig(nav_config(delta_yaw=threshold))


def test_only_one_yaw_overlay_cannot_duplicate_shared_hardware_settings():
    robot = robot_config()
    robot['underwater_navigation_20261026_only_one_yaw'] = {'imu_rate_hz': 100}
    with pytest.raises(ValueError, match='sample_rate_hz'):
        UnderwaterConfig.from_robot(robot)


def test_only_one_yaw_overlay_requires_mapping():
    robot = robot_config()
    robot['underwater_navigation_20261026_only_one_yaw'] = [1, 2, 3]
    with pytest.raises(ValueError, match='映射'):
        UnderwaterConfig.from_robot(robot)


@pytest.mark.parametrize('error', [-10., -5., -.1, 0., .1, 5., 10.])
@pytest.mark.parametrize('ki', [0., .2])
def test_frozen_policy_preserves_original_small_angle_pi_and_limits(error, ki):
    cfg = nav_config(ki=ki)
    cal = build_calibration(robot_config())
    original = NavigationPolicy(cfg, cal)
    frozen = FrozenYawPolicy(cfg, cal)
    # Target is east of the accepted fix, so desired bearing is exactly zero.
    frozen.freeze(2, 3, 0, -error, 6)
    # Include repeated, reversed, and long-gap timestamps to preserve PI semantics.
    for stamp in (0, 10_000_000, 200_000_000, 200_000_000, 100_000_000,
                  1_000_000_000, 10_000_000_000, 10_100_000_000):
        expected = asdict(original.decide(2, 3, -error, stamp))
        decision = asdict(frozen.decide(stamp))
        assert decision.keys() == expected.keys()
        for key in expected:
            if isinstance(expected[key], (float, int)):
                assert decision[key] == pytest.approx(expected[key]), key
            else:
                assert decision[key] == expected[key], key
        assert decision['control_mode'] == 'pi'
        assert frozen.lower <= decision['theta_left'] <= 0
        assert 0 <= decision['theta_right'] <= frozen.upper


def test_underwater_decision_accepts_only_time_and_never_recomputes_heading(monkeypatch):
    import control.runtime.underwater_navigation_20261026_only_one_yaw as module

    policy = FrozenYawPolicy(nav_config(), build_calibration(robot_config()))
    policy.freeze(2, 3, 0, -5, 6)
    assert list(inspect.signature(policy.decide).parameters) == ['t_ns']

    def forbidden(*args, **kwargs):
        raise AssertionError('Underwater control recomputed a yaw angle')

    monkeypatch.setattr(module, 'wrap180', forbidden)
    # Even a different target in config cannot change the frozen geometry.
    policy.config = replace(policy.config, x_d=1, y_d=0)
    first = policy.decide(0)
    later = policy.decide(1_000_000_000)
    assert first.yaw_error_deg == later.yaw_error_deg == 5
    assert first.distance_m == later.distance_m == 6
    assert later.pi_output > first.pi_output  # fixed error, ongoing original PI


def test_new_surface_fix_resets_integral_and_wraps_shortest_error():
    policy = FrozenYawPolicy(nav_config(), build_calibration(robot_config()))
    with pytest.raises(RuntimeError, match='freeze'):
        policy.decide(0)
    policy.freeze(2, 3, 0, -5, 6)
    policy.decide(0)
    policy.decide(1_000_000_000)
    assert policy.integral > 0
    policy.freeze(3, 3, -179, 179, 5)
    assert policy.integral == 0 and policy.last_t_ns is None
    assert policy.decide(2_000_000_000).yaw_error_deg == 2


@pytest.mark.parametrize('error', [-179, -45, -10.01, 10.01, 45, 179])
def test_freeze_rejects_errors_not_yet_surface_aligned(error):
    policy = FrozenYawPolicy(nav_config(), build_calibration(robot_config()))
    with pytest.raises(ValueError):
        policy.freeze(2, 3, 0, -error, 6)


def test_cli_passes_original_parameters_and_repeated_xyz_depth_timeout_points(monkeypatch, tmp_path):
    script = Path('scripts/run_underwater_navigation_20261026_only_one_yaw.py').resolve()
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('only_one_yaw_cli_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = {}

    def run_session(robot, cfg, session, **kwargs):
        received.update(cfg=cfg, session=session, **kwargs)
        return dict(reason='time_limit', fault=None, cleanup_errors=[])

    monkeypatch.setattr(module, 'run_session', run_session)
    monkeypatch.setattr(sys, 'argv', [
        str(script), '--dry-run', '--output', str(tmp_path/'run'),
        '--t1', '0', '--t2', '1', '--t3', '5', '--t4', '5',
        '--depth-kp', '.8', '--depth-ki', '.15', '--initial-b2', '-1',
        '--theta-0', '7', '--theta-00', '8', '--delta-yaw', '6',
        '--waypoint', '1', '2', '.3', '30', '--waypoint', '2', '3', '.4', 'NULL',
    ])
    assert module.main() == 0
    cfg = received['cfg']
    assert cfg.waypoints == ((1., 2., .3, 30.), (2., 3., .4, None))
    assert cfg.t3 == cfg.t4 == 5
    assert cfg.depth_kp == .8 and cfg.depth_ki == .15
    assert cfg.initial_b2 == -1
    assert cfg.navigation.theta_0 == 7 and cfg.navigation.theta_00 == 8
    assert cfg.navigation.delta_yaw == 6
    assert received['dry_run'] and not received['no_servo']


@pytest.mark.parametrize('null', [None, 'NULL', 'null', 'NuLl', ' NULL '])
def test_null_is_normalized_for_single_and_intermediate_targets(null):
    cfg = UnderwaterConfig(nav_config(), waypoints=[[1, 2, .3, null]])
    assert cfg.waypoints == ((1., 2., .3, None),)
    cfg = UnderwaterConfig(nav_config(), waypoints=[[1, 2, .3, null], [3, 4, .4, None]])
    assert cfg.waypoints == ((1., 2., .3, None), (3., 4., .4, None))


@pytest.mark.parametrize('timeout', [1, .25, '30', 1e-9])
def test_positive_timeout_is_normalized_for_intermediate_target(timeout):
    cfg = UnderwaterConfig(nav_config(), waypoints=[[1, 2, .3, timeout], [3, 4, .4, None]])
    assert cfg.waypoints[0][3] == float(timeout)


@pytest.mark.parametrize('timeout', [0, -1, math.inf, -math.inf, math.nan,
                                     True, False, 'None', 'forever', '', 1e-12, 1e300])
def test_invalid_intermediate_timeout_is_rejected(timeout):
    with pytest.raises((ValueError, TypeError)):
        UnderwaterConfig(nav_config(), waypoints=[[1, 2, .3, timeout], [3, 4, .4, None]])


@pytest.mark.parametrize('points', [
    [[1, 2, .3, 30]],
    [[1, 2, .3, None], [3, 4, .4, 30]],
    [[1, 2, .3, 10], [3, 4, .4, '20']],
])
def test_last_target_and_single_target_must_have_null_timeout(points):
    with pytest.raises(ValueError, match='最后一个目标点.*t 必须为 NULL'):
        UnderwaterConfig(nav_config(), waypoints=points)


def test_yaml_null_and_quoted_null_are_accepted_in_own_overlay():
    robot = robot_config()
    robot['underwater_navigation_20261026_only_one_yaw'] = yaml.safe_load('''
waypoints:
  - [1, 2, 0.3, 12.5]
  - [2, 3, 0.4, NULL]
  - [3, 4, 0.5, "null"]
''')
    assert UnderwaterConfig.from_robot(robot).waypoints == (
        (1., 2., .3, 12.5), (2., 3., .4, None), (3., 4., .5, None))


@pytest.mark.parametrize('waypoint_args', [
    ['--waypoint', '1', '2', '.3'],
    ['--waypoint', '1', '2', '.3', '20'],
    ['--waypoint', '1', '2', '.3', '-1', '--waypoint', '2', '3', '.4', 'NULL'],
])
def test_invalid_cli_waypoint_fails_before_hardware_or_output(monkeypatch, tmp_path, waypoint_args):
    script = Path('scripts/run_underwater_navigation_20261026_only_one_yaw.py').resolve()
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('only_one_yaw_cli_invalid_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path/'must_not_exist'

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid CLI parameters reached run_session')

    monkeypatch.setattr(module, 'run_session', forbidden)
    monkeypatch.setattr(sys, 'argv', [str(script), '--dry-run', '--output', str(output), *waypoint_args])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not output.exists()
