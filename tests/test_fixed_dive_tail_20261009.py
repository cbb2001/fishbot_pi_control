"""联合控制回归：双向PI、零米目标、完整组合、阶段切换和尾鳍并行。"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from control.runtime import fixed_dive_tail_20261009 as api
from control.runtime.fixed_dive_20260920_100122 import FixedDivePolicy


@pytest.mark.parametrize('count', [1, 2, 5, 10])
def test_arbitrary_stage_count(count):
    stages = tuple(api.TailStage(i + 1., i * .1, f'stage_{i}') for i in range(count))
    assert api._validate_stages(stages) == stages


def test_empty_stages_rejected():
    with pytest.raises(ValueError):
        api._validate_stages([])


@pytest.mark.parametrize('duration', [0., -1., math.nan, math.inf])
def test_invalid_stage_duration_rejected(duration):
    with pytest.raises(ValueError):
        api._validate_stages((api.TailStage(duration, .5, 'invalid'),))


def test_cli_custom_stages_and_tail_parameters(robot, tmp_path):
    root = Path(__file__).resolve().parents[1]
    robot.setdefault('runtime', {})['log_dir'] = str(tmp_path / 'logs')
    config = tmp_path / 'robot.yaml'
    config.write_text(yaml.safe_dump(robot), encoding='utf-8')
    result = subprocess.run([
        sys.executable, str(root / 'scripts/run_fixed_dive_tail_20261009.py'),
        '--config', str(config), '--dry-run',
        '--stage', '.3', '1', '--stage', '.7', '1.2', '--stage', '0', '1.4',
        '--side-action-duration-s', '.1', '--tail-action-duration-s', '.17',
        '--tail-left-deg', '-12', '--tail-right-deg', '18',
    ], cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    session = next((tmp_path / 'logs').glob('fixed_dive_tail_20261009/*'))
    metadata = yaml.safe_load((session / 'metadata.yaml').read_text(encoding='utf-8'))
    assert [(s['target_depth_m'], s['duration_s']) for s in metadata['stages']] == [
        (.3, 1.), (.7, 1.2), (0., 1.4)]
    assert metadata['max_control_s'] == pytest.approx(3.6)
    assert json.loads((session / 'result.json').read_text())['stage_index'] == 2
    commands = [json.loads(line) for line in (session / 'commands.jsonl').read_text().splitlines()]
    tails = [row for row in commands if row['agent'] == 'tail']
    assert set(row['stage_index'] for row in tails) == {0, 1, 2}
    assert all(row['action']['theta'] == (-12. if i % 2 == 0 else 18.)
               for i, row in enumerate(tails))
    assert all(row['action']['t'] == pytest.approx(.17) for row in tails)


@pytest.fixture
def robot():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / 'config/robot.yaml').read_text(encoding='utf-8'))


@pytest.fixture
def calibration(robot):
    return api.build_calibration(robot)


@pytest.mark.parametrize('target,scale', [(0., 1.), (0., 1.5), (.4, .4)])
def test_surfacing_pi_finishes_pair_across_target_and_alternates_b2(calibration, target, scale):
    policy = FixedDivePolicy(calibration, h0=target, error_scale_m=scale, k=.2, ki=.1)
    span = calibration.limits[4][1] - calibration.centers[4]
    depth = target + .2 * scale
    policy.start_control(0)
    policy.observe_depth(depth, 0)
    policy.observe_depth(depth, 100_000_000)
    first = policy.next_action(depth)
    assert first.b1 == 0
    assert first.theta == pytest.approx((.2 * .2 + .1 * .2 * .1) * span)
    assert policy.pair_info['direction'] == 'surfacing'
    assert not policy.pair_info['limited']
    # 中途到达目标仍需执行完整组合；组后精确到达目标才保持。
    assert policy.next_action(target) == first
    policy.acknowledge(first)
    second = policy.next_action(target)
    assert (second.b1, second.b2) == (1, 1)
    assert second.theta == -first.theta
    policy.acknowledge(second)
    assert policy.next_action(target) is None
    # 随后变浅进入下潜，b2承接上浮组合继续取反。
    dive_first = policy.next_action(target - .1 * scale)
    assert dive_first.theta < 0
    policy.acknowledge(dive_first)
    dive_second = policy.next_action(depth)
    assert (dive_second.b1, dive_second.b2) == (1, -1)
    assert dive_second.theta == -dive_first.theta


def test_stage_target_reset_preserves_b2_and_resets_integral_time(calibration):
    policy = FixedDivePolicy(calibration, h0=.4, k=.2, ki=.1)
    policy.start_control(0)
    policy.observe_depth(.2, 0)
    policy.observe_depth(.2, 100_000_000)
    assert policy.i_term > 0
    first = policy.next_action(.2)
    with pytest.raises(RuntimeError, match='组合'):
        policy.set_target(0., error_scale_m=.4)
    assert policy.h0 == .4
    policy.acknowledge(first)
    second = policy.next_action(.2)
    policy.acknowledge(second)
    policy.set_target(0., error_scale_m=.4)
    assert policy.h0 == 0.
    assert policy.i_term == 0.
    assert policy.integral_status == 'stage_reset'
    assert policy.next_b2 == -1
    policy.observe_depth(.2, 200_000_000)
    assert policy.i_term == 0.  # 不把跨阶段间隔计入新积分。
    policy.observe_depth(.2, 300_000_000)
    assert policy.i_term == pytest.approx(-.005)
    assert policy.next_action(.2).theta > 0


@pytest.mark.parametrize('scale', [None, 0., -1., math.nan, math.inf])
def test_zero_target_requires_valid_explicit_error_scale(calibration, scale):
    with pytest.raises(ValueError):
        FixedDivePolicy(calibration, h0=0., error_scale_m=scale)


@pytest.mark.parametrize('target', [math.nan, math.inf, -1.])
def test_invalid_stage_target_rejected(target):
    stages = (api.TailStage(1., target, 'invalid'),) + api.DEFAULT_STAGES[1:]
    with pytest.raises(ValueError):
        api._validate_stages(stages)


def test_four_stages_reverse_sides_while_tail_runs_and_stop_by_default(robot, monkeypatch, tmp_path):
    from control.runtime.rl_mock_sensors_20260914 import _RLSensorWorker
    from control.runtime.rl_servo_executor_20260914 import RLServoExecutor

    root = Path(__file__).resolve().parents[1]
    cfg = api.prepare_fixed_config(yaml.safe_load(
        (root / 'config/rl_training_20260914.yaml').read_text(encoding='utf-8')), dry_run=True)
    stages = tuple(api.TailStage(1.2, h, name) for h, name in (
        (.3, 'dive_1'), (0., 'surface_1'), (.7, 'dive_2'), (0., 'surface_2')))
    original_read = _RLSensorWorker.read_once
    original_tick = RLServoExecutor.tick_once
    concurrent_writes = []

    def read(worker):
        sample = original_read(worker)
        surface, _ = worker.manager._motion_state()
        if worker.name == 'depth' and not surface:
            # 标定后固定在0.5米：同阶段越过0.3米，然后分别触发上浮/下潜/上浮。
            p0 = float(worker.manager.config['depth_sensor']['surface_pressure_mbar']) * 100.
            convention = worker.manager._convention
            sample.data['pressure_pa'] = p0 + .5 * convention.water_density_kg_m3 * convention.gravity_mps2
        return sample

    def tick(executor, *args, **kwargs):
        concurrent = all(executor.scheduler.active.values())
        refs = original_tick(executor, *args, **kwargs)
        if concurrent:
            concurrent_writes.append(refs)
        return refs

    monkeypatch.setattr(_RLSensorWorker, 'read_once', read)
    monkeypatch.setattr(RLServoExecutor, 'tick_once', tick)
    result = api.run_session(cfg, robot, tmp_path, k=.2, ki=.1,
                             side_action_duration_s=.2, tail_action_duration_s=.1,
                             stages=stages, pressure_per_meter=750., dry_run=True)

    def rows(name):
        return [json.loads(line) for line in (tmp_path / name).read_text(encoding='utf-8').splitlines()]

    commands = rows('commands.jsonl')
    fins = [row for row in commands if row['agent'] == 'fin']
    tails = [row for row in commands if row['agent'] == 'tail']
    events = rows('events.jsonl')
    switches = [row for row in events if row['event'] == 'stage_changed']
    assert result['stage_index'] == 3
    assert json.loads((tmp_path / 'result.json').read_text())['fault'] is None
    assert [(row['stage_index'], row['target_depth_m'], row['error_scale_m']) for row in switches] == [
        (1, 0., .3), (2, .7, .7), (3, 0., .7)]
    assert all(row['integral_status'] == 'stage_reset' for row in switches)
    assert set(row['stage_index'] for row in fins) == {0, 1, 2, 3}
    assert len(fins) == 2 * result['completed_pairs']
    completed_ids = {row['request_id'] for row in events if row['event'] == 'action_completed'}
    assert all(row['request_id'] in completed_ids for row in fins)
    assert any(row['pair']['direction'] == 'surfacing' for row in fins if row['stage_index'] == 0)
    for index, (first, second) in enumerate(zip(fins[::2], fins[1::2])):
        assert first['stage_index'] == second['stage_index']
        assert first['pair'] == second['pair']  # 深度/阶段边界不更改组内冻结的PI输出。
        assert first['action']['b1'] == 0 and second['action']['b1'] == 1
        assert second['action']['b2'] == (1 if index % 2 == 0 else -1)
        assert first['action']['theta'] == -second['action']['theta']
        pair = first['pair']
        assert pair['target_depth_m'] == stages[first['stage_index']].target_depth_m
        if first['stage_index'] in (1, 3):
            assert pair['direction'] == 'surfacing' and first['action']['theta'] > 0
        elif first['stage_index'] == 2:
            assert pair['direction'] == 'diving' and first['action']['theta'] < 0
    for row in switches:
        stage_first = next(v for v in fins if v['stage_index'] == row['stage_index'])
        assert stage_first['pair']['i_term_deg'] == 0.
    assert all(row['action']['theta'] == (-15. if i % 2 == 0 else 15.) for i, row in enumerate(tails))
    assert set(row['stage_index'] for row in tails) == {0, 1, 2, 3}
    assert concurrent_writes and all(set(refs) == set(range(1, 8)) for refs in concurrent_writes)
    depth_rows = rows('depth_control.jsonl')
    assert any(row['phase'] == 'diving' for row in depth_rows)
    assert any(row['phase'] == 'surfacing' for row in depth_rows)
    assert all(math.isfinite(row['p_term_deg'] + row['i_term_deg']) for row in depth_rows)
    cal = json.loads((tmp_path / 'calibration.json').read_text())
    for row in depth_rows:
        assert row['depth_m'] == pytest.approx(
            (row['raw_pressure_pa'] - row['pressure_drift_pa'] - cal['p_surface_pa']) / 750.)
    metadata = yaml.safe_load((tmp_path / 'metadata.yaml').read_text(encoding='utf-8'))
    assert metadata['max_control_s'] == pytest.approx(4.8)
    assert metadata['robot_config']['sensors']['depth']['rate_hz'] == 10.
