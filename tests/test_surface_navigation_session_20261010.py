"""使用真实传感器缓冲区和舵机执行线程，硬件I/O用可控输入替代。"""
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from control.runtime.surface_navigation_20261010 import NavigationConfig
from control.runtime.surface_navigation_session_20261010 import run_session
from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor


class ScriptedSensors:
    def __init__(self, *, drop_uwb=False, drop_imu=False, moving_calibration=False):
        self.buffers = {name: RingBuffer(1000) for name in ('imu', 'uwb')}
        self.stop = threading.Event()
        self.drop_uwb, self.drop_imu = drop_uwb, drop_imu
        self.moving_calibration = moving_calibration

    def start_all(self):
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.produce, daemon=True)
        self.thread.start()

    def ranges_at(self, elapsed):
        return (1732, 3317, 3742, 2449)

    def produce(self):
        seq = 0
        uwb_seq = 0
        next_uwb = 0.
        # 4x3矩形、鱼在(1,1)、与基站高度差1m，距离以毫米取整。
        while not self.stop.is_set():
            elapsed = time.monotonic()-self.start
            now = time.monotonic_ns()
            if not self.drop_imu or elapsed < .8:
                yaw = (60. if seq % 2 else 0.) if self.moving_calibration else 179.9
                self.buffers['imu'].append(SensorSample('imu', now, seq,
                    dict(yaw_deg=yaw, pitch_deg=0., roll_deg=0.)))
            if elapsed >= next_uwb and (not self.drop_uwb or elapsed < .8):
                next_uwb = elapsed + .08
                ranges = self.ranges_at(elapsed)
                packet = ('mc 00 ' + ' '.join(f'{r:08x}' for r in ranges)
                          + f' 4705 {seq % 256:02x} {seq:08x} t0:0 0000\r\n')
                self.buffers['uwb'].append(SensorSample('uwb', now, uwb_seq, {'raw_hex': packet.encode().hex()}))
                uwb_seq += 1
            seq += 1
            self.stop.wait(.005)

    def stop_all(self, timeout=5.):
        self.stop.set()
        self.thread.join(timeout)

    def fatal_errors(self):
        return []


@pytest.fixture
def robot():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/robot.yaml').read_text(encoding='utf-8'))


@pytest.fixture
def cfg():
    return NavigationConfig(L=4., W=3., x_d=3., y_d=1., t1=.05, t2=.4,
                            calibration_min_samples=10, t_0=.04, imu_timeout_s=.15, uwb_timeout_s=.25)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_outside_uwb_position_drives_navigation_without_boundary_hold(robot, cfg, tmp_path):
    class OutsideSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            return tuple(round(math.sqrt((-.5-a)**2+(1.-b)**2+1.)*1000)
                         for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))

    result = run_session(robot, cfg, tmp_path, sensors=OutsideSensors(),
                         controller=DryRunRLServoController(), max_control_s=.3)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    assert result['last_position']['x'] == pytest.approx(-.5, abs=.002)
    assert read_rows(tmp_path / 'commands.jsonl')
    assert all(row['x'] < 0 for row in read_rows(tmp_path / 'navigation.jsonl'))
    events = read_rows(tmp_path / 'events.jsonl')
    assert not any(row['event'] in ('uwb_unavailable', 'uwb_invalid_position') for row in events)


@pytest.mark.parametrize('no_servo', [False, True])
def test_large_residual_uwb_is_used_for_navigation_and_sensor_test(robot, cfg, tmp_path, no_servo):
    class MeasuredSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            return (3222, 65528, 54581, 25524)

    configured = replace(cfg, L=50., W=26.5, H=1.93, range_residual_limit_m=.001)
    result = run_session(robot, configured, tmp_path, sensors=MeasuredSensors(),
                         controller=DryRunRLServoController(), no_servo=no_servo, max_control_s=.3)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    assert result['last_position']['x'] == pytest.approx(-8.056, abs=.001)
    if not no_servo:
        assert read_rows(tmp_path / 'commands.jsonl')
        assert not any(row['event'] == 'uwb_unavailable' for row in read_rows(tmp_path / 'events.jsonl'))


def test_waypoints_interrupt_old_pair_and_change_target(robot, cfg, tmp_path):
    # 长动作尚未走完就应切点，不能补发上一点的右摆。
    result = run_session(robot, replace(cfg, t_0=2., t=.05), tmp_path,
                         sensors=ScriptedSensors(), controller=DryRunRLServoController(),
                         waypoints=[(3., 1., .3), (1., 2., .4)])
    assert result['reason'] == 'time_limit' and result['fault'] is None
    commands = read_rows(tmp_path / 'commands.jsonl')
    assert [row['waypoint_index'] for row in commands] == [1, 2]
    assert all(row['pair_index'] == 0 for row in commands)
    assert commands[0]['decision']['yaw_d_deg'] == pytest.approx(0., abs=.1)
    assert commands[1]['decision']['yaw_d_deg'] == pytest.approx(90., abs=.1)
    events = read_rows(tmp_path / 'events.jsonl')
    stages = [row for row in events if row['event'] == 'waypoint_started']
    assert [row['waypoint_index'] for row in stages] == [1, 2]
    assert (stages[1]['t_ns']-stages[0]['t_ns'])/1e9 == pytest.approx(.3, abs=.12)
    assert any(row['event'] == 'tail_held' for row in events)
    assert sum(row['event'] == 'calibration_completed' for row in events) == 1


def test_waypoints_hold_until_stage_time_then_resume(robot, cfg, tmp_path):
    run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(),
                controller=DryRunRLServoController(), waypoints=[(1., 1., .3), (3., 1., .4)])
    commands = read_rows(tmp_path / 'commands.jsonl')
    assert commands and all(row['waypoint_index'] == 2 for row in commands)
    events = read_rows(tmp_path / 'events.jsonl')
    assert any(row['event'] == 'holding_started' for row in events)
    stages = [row for row in events if row['event'] == 'waypoint_started']
    assert (stages[1]['t_ns']-stages[0]['t_ns'])/1e9 >= .3


def test_waypoint_deadline_centers_even_when_stdout_blocks(robot, cfg, tmp_path, monkeypatch):
    from control.runtime import surface_navigation_session_20261010 as runtime
    hardware = DryRunRLServoController()
    blocked = False
    samples = []

    def slow_print(message, **kwargs):
        nonlocal blocked
        if message.startswith('x=') and not blocked:
            blocked = True
            # 模拟SSH终端输出阻塞；执行器应独立在第一段到时后保持中位。
            time.sleep(.4)
            samples.append(list(hardware.commands)[-3:])

    monkeypatch.setattr(runtime, 'print', slow_print, raising=False)
    result = run_session(robot, replace(cfg, t_0=2.), tmp_path, sensors=ScriptedSensors(),
                         controller=hardware, waypoints=[(3., 1., .15), (1., 2., .7)])
    assert result['reason'] == 'time_limit'
    assert samples == [[{0: 85.}, {1: 95.}, {2: 95.}]]


def test_waypoints_advance_while_waiting_for_uwb(robot, cfg, tmp_path):
    result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(drop_uwb=True),
                         controller=DryRunRLServoController(),
                         waypoints=[(3., 1., .8), (1., 2., .3)])
    assert result['reason'] == 'time_limit'
    events = read_rows(tmp_path / 'events.jsonl')
    waiting = next(row['t_ns'] for row in events if row['event'] == 'uwb_unavailable')
    stage2 = next(row['t_ns'] for row in events
                  if row['event'] == 'waypoint_started' and row['waypoint_index'] == 2)
    assert stage2 > waiting
    assert all(row['waypoint_index'] == 1 for row in read_rows(tmp_path / 'commands.jsonl'))


def test_uwb_recovers_to_current_waypoint_after_switch(robot, cfg, tmp_path):
    class InvalidThenRecovered(ScriptedSensors):
        def ranges_at(self, elapsed):
            # 斜距小于配置高度触发回中等待，恢复时第一段已结束。
            return (1000, 1000, 1000, 1000) if .75 < elapsed < 1.4 else super().ranges_at(elapsed)

    run_session(robot, replace(cfg, t_0=2., ki=.5, H=1.5), tmp_path,
                sensors=InvalidThenRecovered(), controller=DryRunRLServoController(),
                waypoints=[(3., 2., .65), (1., 2., .7)])
    events = read_rows(tmp_path / 'events.jsonl')
    recovered = next(row for row in events if row['event'] == 'uwb_recovered')
    assert recovered['waypoint_index'] == 2
    # Windows时钟分辨率可能使恢复事件与紧随其后的动作同戳，不能用严格大于筛掉动作。
    commands = [row for row in read_rows(tmp_path / 'commands.jsonl') if row['t_ns'] >= recovered['t_ns']]
    assert commands and all(row['waypoint_index'] == 2 for row in commands)
    assert commands[0]['pair_index'] == 0
    nav = next(row for row in read_rows(tmp_path / 'navigation.jsonl') if row['waypoint_index'] == 2)
    assert nav['integral'] == 0.


def test_no_servo_waypoints_use_sum_without_starting_executor(robot, cfg, tmp_path, monkeypatch):
    from control.runtime import surface_navigation_session_20261010 as runtime

    def forbidden(*args, **kwargs):
        raise AssertionError('no-servo must not create an executor')

    monkeypatch.setattr(runtime, 'RLServoExecutor', forbidden)
    result = run_session(robot, replace(cfg, t=.01), tmp_path, no_servo=True,
                         sensors=ScriptedSensors(), waypoints=[(1., 1., .1), (2., 2., .2)])
    assert result['reason'] == 'time_limit'
    metadata = yaml.safe_load((tmp_path / 'metadata.yaml').read_text(encoding='utf-8'))
    assert metadata['max_control_s'] == pytest.approx(.3)


@pytest.mark.parametrize('waypoints', [[], [(5, 1, 1)], [(1, -1, 1)],
                                     [(1, 1, 0)], [(1, 1, -1)], [(1, 1, float('nan'))],
                                     [(1, 1, float('inf'))], [(1, 1)]])
def test_invalid_waypoints_rejected_before_sensor_start(robot, cfg, tmp_path, waypoints):
    sensors = ScriptedSensors()
    with pytest.raises(ValueError):
        run_session(robot, cfg, tmp_path, sensors=sensors,
                    controller=DryRunRLServoController(), waypoints=waypoints)
    assert not hasattr(sensors, 'thread')


def test_one_pair_per_new_uwb_no_fin_pwm_and_no_early_motion(robot, cfg, tmp_path):
    hardware = DryRunRLServoController()
    result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(), controller=hardware, max_control_s=.6)
    assert result['reason'] == 'time_limit'
    commands = read_rows(tmp_path / 'commands.jsonl')
    assert len(commands) >= 4
    ids = []
    for left, right in zip(commands[::2], commands[1::2]):
        assert left['action']['theta'] < 0 < right['action']['theta']
        assert left['uwb_frame_id'] == right['uwb_frame_id']
        ids.append(left['uwb_frame_id'])
    assert len(ids) == len(set(ids))
    assert all(set(row).issubset({0, 1, 2}) for row in hardware.commands)
    events = read_rows(tmp_path / 'events.jsonl')
    cal_done = next(row['t_ns'] for row in events if row['event'] == 'calibration_completed')
    assert all(row['t_ns'] >= cal_done for row in commands)
    assert result['completed_pairs'] >= 2


def test_time_limit_interrupts_first_action_without_waiting_for_pair(robot, cfg, tmp_path):
    result = run_session(robot, replace(cfg, t_0=2.), tmp_path,
                         sensors=ScriptedSensors(), controller=DryRunRLServoController(), max_control_s=.15)
    assert result['reason'] == 'time_limit'
    assert result['fault'] is None and result['completed_pairs'] == 0
    commands = read_rows(tmp_path / 'commands.jsonl')
    assert len(commands) == 1  # 放置/标定不占运动时间；到时不再执行右摆。
    assert not result['executor_alive']


def test_yaml_t_sets_motion_limit(robot, cfg, tmp_path):
    configured = replace(cfg, t=.15, t_0=2.)
    result = run_session(robot, configured, tmp_path,
                         sensors=ScriptedSensors(), controller=DryRunRLServoController())
    assert result['reason'] == 'time_limit' and result['completed_pairs'] == 0


def test_no_time_limit_keeps_running_until_stop_request(robot, cfg, tmp_path):
    stop = threading.Event()
    timer = threading.Timer(1.5, stop.set)
    timer.start()
    try:
        result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(),
                             controller=DryRunRLServoController(), stop_event=stop)
    finally:
        timer.cancel()
        timer.join()
    assert result['reason'] == 'stopped'
    assert result['completed_pairs'] >= 2


@pytest.mark.parametrize('drop_uwb,drop_imu,expected', [(False, True, 'IMU_TIMEOUT')])
def test_sensor_loss_stops_inflight_pair(robot, cfg, tmp_path, drop_uwb, drop_imu, expected):
    with pytest.raises(RuntimeError, match=expected):
        run_session(robot, replace(cfg, t_0=1.), tmp_path,
                    sensors=ScriptedSensors(drop_uwb=drop_uwb, drop_imu=drop_imu),
                    controller=DryRunRLServoController(), max_control_s=2.)
    result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert expected in result['fault']
    assert result['completed_pairs'] == 0
    assert not result['executor_alive']


def test_unstable_calibration_never_initializes_pwm(robot, cfg, tmp_path):
    hardware = DryRunRLServoController()
    with pytest.raises(RuntimeError, match='不稳定'):
        run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(moving_calibration=True), controller=hardware)
    assert not hardware.commands


def test_already_at_target_keeps_monitoring_until_time_limit_without_pwm(robot, cfg, tmp_path):
    hardware = DryRunRLServoController()
    result = run_session(robot, replace(cfg, x_d=1.05, y_d=1.), tmp_path,
                         sensors=ScriptedSensors(), controller=hardware, max_control_s=1.)
    assert result['reason'] == 'time_limit'
    assert result['completed_pairs'] == 0
    assert read_rows(tmp_path / 'commands.jsonl') == []
    assert not hardware.commands  # 到点判断必须发生在执行器第一次中位PWM之前。


@pytest.mark.parametrize('arrival_x', [2.95, 3.05])
def test_arrive_hold_drift_resume_then_hold_again(robot, cfg, tmp_path, arrival_x):
    class DriftingSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            # 先航行，在第一摆未完成时到达；保持后漂走，再次到达。
            x = 1. if elapsed < .85 or 1.35 <= elapsed < 1.8 else arrival_x
            return tuple(round(math.sqrt((x-a)**2+(1.-b)**2+1.)*1000)
                         for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))

    result = run_session(robot, replace(cfg, t_0=2.), tmp_path, sensors=DriftingSensors(),
                         controller=DryRunRLServoController(), max_control_s=1.8)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    events = read_rows(tmp_path / 'events.jsonl')
    held = [r for r in events if r['event'] == 'holding_started']
    centered = [r for r in events if r['event'] == 'tail_held']
    assert len(centered) == 2
    assert all(r['angles_deg'] == {'1': 85., '2': 95., '3': 95.} for r in centered)
    resumed = [r for r in events if r['event'] == 'navigation_resumed']
    assert len(held) == 2 and len(resumed) == 1
    commands = read_rows(tmp_path / 'commands.jsonl')
    assert len(commands) == 2
    assert all(r['pair_index'] == 0 for r in commands)  # 取消的右摆不会在恢复后补发。
    assert commands[0]['t_ns'] < held[0]['t_ns'] < resumed[0]['t_ns'] <= commands[1]['t_ns'] < held[1]['t_ns']
    assert len([r for r in events if r['event'] == 'motion_started']) == 1
    assert result['completed_pairs'] == 0


def test_heading_change_inside_distance_band_does_not_resume(robot, cfg, tmp_path):
    class NearbyDrift(ScriptedSensors):
        def ranges_at(self, elapsed):
            # 越过目标后目标航向约180度，但位置仍在0.2m容差内。
            x = 1. if elapsed < .85 else 1.1
            return tuple(round(math.sqrt((x-a)**2+(1.-b)**2+1.)*1000)
                         for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))

    hardware = DryRunRLServoController()
    result = run_session(robot, replace(cfg, x_d=1.05), tmp_path, sensors=NearbyDrift(),
                         controller=hardware, max_control_s=1.)
    assert result['reason'] == 'time_limit'
    assert not hardware.commands
    assert not [r for r in read_rows(tmp_path / 'events.jsonl') if r['event'] == 'navigation_resumed']


def test_uwb_timeout_centers_and_waits_until_time_limit(robot, cfg, tmp_path):
    result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(drop_uwb=True),
                         controller=DryRunRLServoController(), max_control_s=1.)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    events = read_rows(tmp_path / 'events.jsonl')
    assert any(r['event'] == 'uwb_unavailable' for r in events)
    assert any(r['event'] == 'tail_held' and r['angles_deg'] == {'1':85., '2':95., '3':95.} for r in events)


@pytest.mark.parametrize('invalid_value', [0, 0xffffffff])
def test_incomplete_uwb_reuses_last_complete_until_new_fix(robot, cfg, tmp_path, invalid_value):
    class IncompleteSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            values = super().ranges_at(elapsed)
            if .8 <= elapsed < 1.4:
                return (9999, 9999, 9999, invalid_value)
            if elapsed >= 1.4:
                return (2449, 2449, 3000, 3000)  # 新位置(2,1)，不能混用旧距离。
            return values

    result = run_session(robot, cfg, tmp_path,
                         sensors=IncompleteSensors(), controller=DryRunRLServoController(), max_control_s=1.4)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    events = read_rows(tmp_path / 'events.jsonl')
    assert not any(r['event'] == 'uwb_unavailable' for r in events)
    commands = read_rows(tmp_path / 'commands.jsonl')
    reused = [r for r in commands if r.get('position_reused')]
    assert len(reused) >= 4  # 缺测持续超过uwb_timeout_s时仍连续完成多组动作。
    assert any(r['position_age_s'] > cfg.uwb_timeout_s for r in reused)
    assert all(r['decision']['x'] == pytest.approx(1., abs=.002) for r in reused)
    assert all(r['decision']['y'] == pytest.approx(1., abs=.002) for r in reused)
    assert len({r['uwb_frame_id'] for r in reused}) == 1
    assert result['last_position']['x'] == pytest.approx(2., abs=.002)
    assert commands[-1]['uwb_frame_id'] != reused[-1]['uwb_frame_id']


@pytest.mark.parametrize('same_chunk', [False, True])
def test_bad_then_good_batch_still_cancels_old_pair(robot, cfg, tmp_path, same_chunk):
    sensors = ScriptedSensors()
    original_snapshot = sensors.buffers['uwb'].snapshot
    injected = None

    def snapshot():
        nonlocal injected
        rows = original_snapshot()
        if not rows or rows[-1].t_ns/1e9-sensors.start < .8:
            return rows
        if injected is None:
            injected = rows[-1].t_ns
        index = next(i for i, row in enumerate(rows) if row.t_ns == injected)
        good = rows[index]
        # 同块使用斜距小于高度的坏测距；跨样本使用缺失测距。
        bad = (b'mc 00 000003e8 000003e8 000003e8 000003e8 4705 ee fffffffe t0:0 0000\r\n'
               if same_chunk else
               b'mc 00 00000000 00000cf5 00000e9e 00000991 4705 ee fffffffe t0:0 0000\r\n')
        if same_chunk:
            return [*rows[:index], SensorSample('uwb', good.t_ns, good.seq,
                    {'raw_hex': (bad+bytes.fromhex(good.data['raw_hex'])).hex()}), *rows[index+1:]]
        return [*rows[:index], SensorSample('uwb', good.t_ns-1, good.seq,
                    {'raw_hex': bad.hex()}), *rows[index:]]

    sensors.buffers['uwb'].snapshot = snapshot
    result = run_session(robot, replace(cfg, t_0=2., uwb_timeout_s=3., H=1.5), tmp_path,
                         sensors=sensors, controller=DryRunRLServoController(), max_control_s=1.)
    assert result['reason'] == 'time_limit'
    events = read_rows(tmp_path / 'events.jsonl')
    kinds = [r['event'] for r in events]
    if same_chunk:
        assert kinds.index('uwb_unavailable') < kinds.index('tail_held') < kinds.index('uwb_recovered')
    else:
        assert 'uwb_unavailable' not in kinds
    commands = read_rows(tmp_path / 'commands.jsonl')
    if same_chunk:
        assert len(commands) == 2 and all(r['pair_index'] == 0 for r in commands)
    else:
        assert len(commands) == 1  # 不完整报文不打断当前长动作。


@pytest.mark.parametrize('all_incomplete', [False, True])
def test_no_servo_reads_real_sensor_path_without_executor(robot, cfg, tmp_path, monkeypatch, all_incomplete):
    from control.runtime import surface_navigation_session_20261010 as runtime

    class Sensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            values = super().ranges_at(elapsed)
            return (*values[:3], 0xffffffff) if all_incomplete or elapsed > .1 else values

    def sensor_factory(config, *, mock, stop_event, log_dir):
        assert mock is False
        assert config['sensors']['imu']['enabled'] and config['sensors']['uwb']['enabled']
        return Sensors()

    def forbidden(*args, **kwargs):
        raise AssertionError('传感器测试不得创建舵机执行器或读取舵机标定')

    monkeypatch.setattr(runtime, 'SensorManager', sensor_factory)
    monkeypatch.setattr(runtime, 'RLServoExecutor', forbidden)
    monkeypatch.setattr(runtime, 'build_calibration', forbidden)
    result = run_session(robot, cfg, tmp_path, no_servo=True, controller=forbidden, max_control_s=.35)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    assert result['completed_pairs'] == 0 and not result['executor_alive']
    assert not (tmp_path / 'commands.jsonl').exists()
    assert read_rows(tmp_path / 'raw_imu.jsonl')
    assert read_rows(tmp_path / 'raw_uwb.jsonl')
    diagnostics = read_rows(tmp_path / 'uwb_diagnostics.jsonl')
    assert any(r.get('ranges_m', [0]*4)[3] is None for r in diagnostics)
    if all_incomplete:
        assert result['last_position'] is None
    else:
        assert result['last_position']['x'] == pytest.approx(1., abs=.002)
        assert any(r['position_reused'] for r in read_rows(tmp_path / 'navigation.jsonl'))
    metadata = yaml.safe_load((tmp_path / 'metadata.yaml').read_text(encoding='utf-8'))
    assert metadata['no_servo'] is True and metadata['active_servo_ids'] == []


def test_cli_no_servo_does_not_require_move_confirmation(robot, tmp_path, monkeypatch):
    import importlib.util
    from control.runtime import surface_navigation_session_20261010 as runtime
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('navigation_cli', root / 'scripts/run_surface_navigation_20261010.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    robot['runtime']['log_dir'] = str(tmp_path / 'logs')
    path = tmp_path / 'robot.yaml'
    path.write_text(yaml.safe_dump(robot), encoding='utf-8')
    monkeypatch.setattr(runtime, 'SensorManager', lambda *a, **kw: ScriptedSensors())
    monkeypatch.setattr(sys, 'argv', ['navigation', '--config', str(path), '--no-servo', '--t', '.15'])
    assert cli.main() == 0
    result_path, = (tmp_path / 'logs').glob('surface_navigation_20261010/*/result.json')
    assert json.loads(result_path.read_text(encoding='utf-8'))['no_servo'] is True


def test_no_servo_finishes_even_without_any_sensor_data(robot, cfg, tmp_path):
    class SilentSensors(ScriptedSensors):
        def produce(self):
            self.stop.wait()

    sensors = SilentSensors()
    result = run_session(robot, cfg, tmp_path, no_servo=True, sensors=sensors, max_control_s=.05)
    assert result['reason'] == 'time_limit' and result['last_position'] is None
    assert not sensors.thread.is_alive()
    assert (tmp_path / 'raw_uwb.jsonl').read_text() == ''
    assert (tmp_path / 'raw_imu.jsonl').read_text() == ''


def test_no_servo_accepts_complete_report_after_incomplete_with_same_id(robot, cfg, tmp_path):
    class SameIdSensors(ScriptedSensors):
        def produce(self):
            packet = b'mc 00 000006c4 00000cf5 00000e9e 00000991 4705 01 00000001 t0:0 0000\r\n'
            for seq, data in enumerate((packet.replace(b'00000991', b'ffffffff'), packet), 1):
                self.buffers['uwb'].append(SensorSample('uwb', time.monotonic_ns(), seq,
                                                       {'raw_hex': data.hex()}))
            self.stop.wait()

    result = run_session(robot, cfg, tmp_path, no_servo=True, sensors=SameIdSensors(), max_control_s=.03)
    assert result['last_position'] is not None
    assert result['last_position']['x'] == pytest.approx(1., abs=.002)


def test_incomplete_reuse_still_stops_on_real_stream_loss(robot, cfg, tmp_path):
    class EventuallySilent(ScriptedSensors):
        def ranges_at(self, elapsed):
            if elapsed >= 1.15:
                self.drop_uwb = True
            values = super().ranges_at(elapsed)
            return (*values[:3], 0xffffffff) if elapsed >= .8 else values

    result = run_session(robot, cfg, tmp_path, sensors=EventuallySilent(),
                         controller=DryRunRLServoController(), max_control_s=1.4)
    assert result['reason'] == 'time_limit'
    events = read_rows(tmp_path / 'events.jsonl')
    assert any(r['event'] == 'uwb_incomplete' and r['position_reused'] for r in events)
    centered = next(r for r in events if r['event'] == 'tail_held')
    assert not any(r['t_ns'] > centered['t_ns'] for r in read_rows(tmp_path / 'commands.jsonl'))


def test_reused_fix_recomputes_heading_between_action_pairs(robot, cfg, tmp_path):
    class IncompleteSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            values = super().ranges_at(elapsed)
            return (*values[:3], 0xffffffff) if elapsed >= .8 else values

    sensors = IncompleteSensors()
    snapshot = sensors.buffers['imu'].snapshot

    def changing_heading():
        return [SensorSample('imu', row.t_ns, row.seq,
                {**row.data, 'yaw_deg': 179.9 if row.t_ns/1e9-sensors.start < 1.1 else 119.9})
                for row in snapshot()]

    sensors.buffers['imu'].snapshot = changing_heading
    run_session(robot, replace(cfg, imu_filter_tau_s=0), tmp_path, sensors=sensors,
                controller=DryRunRLServoController(), max_control_s=1.3)
    reused = [r for r in read_rows(tmp_path / 'navigation.jsonl') if r['position_reused']]
    assert len({r['uwb_frame_id'] for r in reused}) == 1
    assert any(abs(r['yaw_deg']) < 1. for r in reused)
    assert any(abs(r['yaw_deg']+60.) < 1. for r in reused)
    assert reused[-1]['theta_left'] < reused[0]['theta_left'] - 1.


@pytest.mark.parametrize('when', ['before_request', 'during_write'])
def test_manual_stop_during_hold_transition_is_not_fault(robot, cfg, tmp_path, monkeypatch, when):
    stop = threading.Event()
    in_hold = threading.Event()

    class ArrivingSensors(ScriptedSensors):
        def ranges_at(self, elapsed):
            x = 1. if elapsed < .85 else 2.95
            return tuple(round(math.sqrt((x-a)**2+(1.-b)**2+1.)*1000)
                         for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))

    class StopDuringHold(DryRunRLServoController):
        def write_angle(self, channel, angle):
            super().write_angle(channel, angle)
            if in_hold.is_set():
                stop.set()

    if when == 'before_request':
        original = RLServoExecutor.request_hold

        def request(executor):
            stop.set()
            return original(executor)

        monkeypatch.setattr(RLServoExecutor, 'request_hold', request)
    else:
        original = RLServoExecutor._hold

        def hold(executor, ticket):
            in_hold.set()
            return original(executor, ticket)

        monkeypatch.setattr(RLServoExecutor, '_hold', hold)
    result = run_session(robot, replace(cfg, t_0=2.), tmp_path, sensors=ArrivingSensors(),
                         controller=StopDuringHold(), stop_event=stop, max_control_s=2.)
    assert result['reason'] == 'stopped' and result['fault'] is None
    assert not result['executor_alive']


def test_stop_during_pwm_batch_is_normal_stop_and_releases_pwm(robot, cfg, tmp_path):
    stop = threading.Event()

    class StopDuringWrite(DryRunRLServoController):
        stopped = False

        def write_angle(self, channel, angle):
            super().write_angle(channel, angle)
            stop.set()  # 确定性地在一次三舵机写入的第一路之后触发Ctrl+C语义。

        def stop_all(self, channels):
            self.stopped = True

    hardware = StopDuringWrite()
    result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(), controller=hardware, stop_event=stop)
    assert result['reason'] == 'stopped' and result['fault'] is None
    assert hardware.stopped and not result['executor_alive']


def test_intentional_stop_does_not_hide_subsequent_pwm_shutdown_failure(robot, cfg, tmp_path):
    stop = threading.Event()

    class FailedShutdown(DryRunRLServoController):
        def write_angle(self, channel, angle):
            super().write_angle(channel, angle)
            stop.set()

        def stop_all(self, channels):
            raise OSError('injected PWM disable I/O failure')

    with pytest.raises(RuntimeError, match='PWM disable I/O failure'):
        run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(),
                    controller=FailedShutdown(), stop_event=stop)
    result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
    assert result['reason'] == 'fault'
    assert 'PWM disable I/O failure' in result['fault']


@pytest.mark.parametrize('delay_prefix', [False, True])
def test_frame_started_during_calibration_cannot_start_motion(robot, cfg, tmp_path, delay_prefix):
    class SplitAcrossCalibration(ScriptedSensors):
        def produce(self):
            prefix = b'mc 00 000006c4 00000cf5 '
            suffix = b'00000e9e 00000991 4705 01 00000001 t0:0 0000\r\n'
            seq, prefix_sent, suffix_sent = 0, False, False
            while not self.stop.is_set():
                now = time.monotonic_ns()
                self.buffers['imu'].append(SensorSample('imu', now, seq,
                    dict(yaw_deg=0., roll_deg=0., pitch_deg=0.)))
                elapsed = time.monotonic()-self.start
                if elapsed >= (.5 if delay_prefix else .1) and not prefix_sent:
                    timestamp = now-400_000_000 if delay_prefix else now
                    self.buffers['uwb'].append(SensorSample('uwb', timestamp, 1, {'raw_hex': prefix.hex()}))
                    prefix_sent = True
                if elapsed >= .5 and not suffix_sent:
                    self.buffers['uwb'].append(SensorSample('uwb', now, 2, {'raw_hex': suffix.hex()}))
                    suffix_sent = True
                seq += 1
                self.stop.wait(.005)

    hardware = DryRunRLServoController()
    result = run_session(robot, cfg, tmp_path, sensors=SplitAcrossCalibration(), controller=hardware, max_control_s=.3)
    assert result['reason'] == 'time_limit'
    assert read_rows(tmp_path / 'commands.jsonl') == []
    assert hardware.commands and all(row == {channel: {0:85., 1:95., 2:95.}[channel]}
                                     for row in hardware.commands for channel in row)


@pytest.mark.parametrize('interruption', ['timeout', 'serial_error', 'buffer_gap'])
def test_navigation_split_frame_survives_only_plain_timeout(robot, cfg, tmp_path, interruption):
    """空读和等待定位不能丢半帧，断线/缓冲缺口则禁止跨缺失字节拼接。"""
    class SplitAfterCalibration(ScriptedSensors):
        def produce(self):
            packet = b'mc 00 000006c4 00000cf5 00000e9e 00000991 4705 01 00000001 t0:0 0000\r\n'
            steps = [(0.52, 1, packet[:-1], None),
                     (0.55, 2, b'', 'no_signal_or_timeout' if interruption == 'timeout' else 'OSError: disconnected'),
                     (0.82, 3, packet[-1:], None)]
            if interruption == 'buffer_gap':
                steps = [steps[0], steps[2]]
            while not self.stop.is_set():
                now = time.monotonic_ns()
                self.buffers['imu'].append(SensorSample('imu', now, 1,
                    dict(yaw_deg=0., roll_deg=0., pitch_deg=0.)))
                elapsed = time.monotonic() - self.start
                if steps and elapsed >= steps[0][0]:
                    _, seq, chunk, error = steps.pop(0)
                    self.buffers['uwb'].append(SensorSample('uwb', now, seq,
                        {'raw_hex': chunk.hex()}, ok=error is None, error=error))
                self.stop.wait(.005)

    hardware = DryRunRLServoController()
    result = run_session(robot, cfg, tmp_path, sensors=SplitAfterCalibration(),
                         controller=hardware, max_control_s=.55)
    assert result['reason'] == 'time_limit' and result['fault'] is None
    commands = read_rows(tmp_path / 'commands.jsonl')
    if interruption == 'timeout':
        assert commands, '半帧经过空读及定位超时等待后仍应恢复出完整定位'
        assert commands[0]['uwb_frame_id'] == '01:00000001:t0:0'
    else:
        assert commands == []


def test_uwb_loss_while_controller_initializes_only_allows_center_pwm(robot, cfg, tmp_path):
    hardware = DryRunRLServoController()

    def slow_controller():
        time.sleep(1.)
        return hardware

    result = run_session(robot, cfg, tmp_path, sensors=ScriptedSensors(drop_uwb=True),
                         controller=slow_controller, max_control_s=.3)
    assert result['reason'] == 'time_limit'
    assert read_rows(tmp_path / 'commands.jsonl') == []
    assert hardware.commands and all(row == {channel: {0:85., 1:95., 2:95.}[channel]}
                                     for row in hardware.commands for channel in row)


@pytest.mark.parametrize('time_option', ['--t', '--max-control-s'])
def test_cli_dry_run_overrides_and_zero_duration(robot, tmp_path, time_option):
    robot['runtime']['log_dir'] = str(tmp_path / 'logs')
    config = tmp_path / 'robot.yaml'
    config.write_text(yaml.safe_dump(robot), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-X', 'utf8', str(root / 'scripts/run_surface_navigation_20261010.py'),
               '--config', str(config), '--dry-run', '--L', '4', '--W', '3', '--x-d', '3', '--y-d', '1',
               '--t1', '0', '--t2', '.5', '--t-0', '0', time_option, '.6', '--ki', '.2']
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    session = next((tmp_path / 'logs' / 'surface_navigation_20261010').iterdir())
    metadata = yaml.safe_load((session / 'metadata.yaml').read_text(encoding='utf-8'))
    assert metadata['navigation']['ki'] == .2
    result = json.loads((session / 'result.json').read_text(encoding='utf-8'))
    assert result['completed_pairs'] >= 1 and result['fault'] is None


def test_cli_rejects_missing_geometry_before_creating_logs(tmp_path):
    config = tmp_path / 'robot.yaml'
    config.write_text('surface_navigation_20261010: {L: null, W: null}\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'scripts/run_surface_navigation_20261010.py'),
        '--config', str(config), '--dry-run'], capture_output=True, text=True, encoding='utf-8')
    assert completed.returncode != 0 and 'L' in completed.stderr


@pytest.mark.parametrize('points', [[(3, 1, .3)], [(3, 1, .3), (1, 2, .3), (2, 2, .3)]])
def test_cli_waypoints_override_yaml_single_target_and_time(robot, tmp_path, points):
    robot['runtime']['log_dir'] = str(tmp_path / 'logs')
    robot['surface_navigation_20261010'].update(x_d=None, y_d=None, t=.01)
    config = tmp_path / 'robot.yaml'
    config.write_text(yaml.safe_dump(robot), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-X', 'utf8', str(root / 'scripts/run_surface_navigation_20261010.py'),
               '--config', str(config), '--dry-run', '--L', '4', '--W', '3',
               '--t1', '0', '--t2', '.5', '--t-0', '0']
    for point in points:
        command += ['--waypoint', *map(str, point)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    session = next((tmp_path / 'logs' / 'surface_navigation_20261010').iterdir())
    commands = read_rows(session / 'commands.jsonl')
    assert sorted(set(row['waypoint_index'] for row in commands)) == list(range(1, len(points)+1))
    events = read_rows(session / 'events.jsonl')
    stages = [row for row in events if row['event'] == 'waypoint_started']
    assert [(row['x_d'], row['y_d']) for row in stages] == [point[:2] for point in points]
    assert events[-1]['event'] == 'route_completed'


@pytest.mark.parametrize('extra', [
    ['--t', '10'], ['--x_d', '2'], ['--y-d', '2'], ['--max-control-s', '10'],
    ['--waypoint', '1', '1', '0'], ['--waypoint', '1000', '1', '1'],
    ['--waypoint', '1', '1', 'nan'],
])
def test_cli_rejects_conflicting_or_invalid_waypoints_before_logs(robot, tmp_path, extra):
    robot['runtime']['log_dir'] = str(tmp_path / 'logs')
    config = tmp_path / 'robot.yaml'
    config.write_text(yaml.safe_dump(robot), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'scripts/run_surface_navigation_20261010.py'),
        '--config', str(config), '--dry-run', '--waypoint', '1', '1', '1', *extra],
        capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert not (tmp_path / 'logs').exists()
