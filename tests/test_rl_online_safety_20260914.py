"""用真实训练循环和模拟 I/O 验证故障停车，不访问树莓派硬件。"""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml
from control.runtime import rl_online_training_20260914 as runtime
from control.runtime.rl_mock_sensors_20260914 import _RLSensorWorker
from control.runtime.rl_training_config_20260914 import prepare_config

ROOT = Path(__file__).resolve().parents[1]


class OnlineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = prepare_config(yaml.safe_load((ROOT / 'config/rl_training_20260914.yaml').read_text(encoding='utf-8')), dry_run=True)
        self.cfg['state']['enforce_data_freshness'] = True
        self.cfg['training'].update(max_episodes=1, episode_duration_s=4.1, success_enabled=False)
        self.robot = yaml.safe_load((ROOT / 'config/robot.yaml').read_text(encoding='utf-8'))
        self.robot['sensors']['power']['rate_hz'] = 100
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.controllers = []
        owner = self

        class Controller(runtime.DryRunRLServoController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.off = threading.Event()
                owner.controllers.append(self)

            def stop_all(self, *args, **kwargs):
                self.off.set()

        self.controller_patch = patch.object(runtime, 'DryRunRLServoController', Controller)
        self.controller_patch.start()
        self.addCleanup(self.controller_patch.stop)

    def run_session(self, **kwargs):
        return runtime.run_session(self.cfg, self.robot, self.path, dry_run=True, seed=3, **kwargs)

    def test_bad_power_before_calibration_never_writes_servos(self):
        self.robot['safety']['min_voltage_v'] = 20
        with self.assertRaisesRegex(RuntimeError, 'LOW_VOLTAGE'):
            self.run_session()
        self.assertFalse(self.controllers[0].commands)
        status = json.loads((self.path / 'rl_live_status_20260914.json').read_text())
        self.assertEqual(status['fault']['fault_code'], 'LOW_VOLTAGE')

    def test_fault_during_inference_stops_before_network_returns(self):
        # 关闭决策过期拦截也不能关闭独立低电压停车。
        self.cfg['state']['stop_on_stale_decision'] = False
        self.cfg['state']['enforce_data_freshness'] = False
        inject = threading.Event()
        original_read = _RLSensorWorker.read_once
        original_act = runtime.FinPPOAgent.act
        owner = self

        def read(worker):
            sample = original_read(worker)
            if worker.name == 'power' and inject.is_set():
                sample.data['voltage_v'] = 8
            return sample

        def act(agent, obs, *args, **kwargs):
            inject.set()
            owner.assertTrue(owner.controllers[0].off.wait(.8), '主线程阻塞时仍应独立关闭 PWM')
            return original_act(agent, obs, *args, **kwargs)

        with patch.object(_RLSensorWorker, 'read_once', read), patch.object(runtime.FinPPOAgent, 'act', act):
            with self.assertRaisesRegex(RuntimeError, 'LOW_VOLTAGE'):
                self.run_session()
        commands = (self.path / 'commands.jsonl').read_text()
        self.assertEqual(commands, '')

    def test_healthy_run_updates_and_saves_calibration(self):
        self.cfg['training']['max_episodes'] = 2
        result = self.run_session()
        self.assertGreaterEqual(result['fin_update_count'], 2)
        self.assertTrue((self.path / 'calibration_20260914.json').is_file())
        calibration = json.loads((self.path / 'calibration_20260914.json').read_text())
        self.assertEqual(calibration['pressure_reference_mode'], 'fixed_config')
        self.assertEqual(calibration['depth_sample_count'], 0)
        self.assertAlmostEqual(calibration['p_surface_pa'], self.robot['depth_sensor']['surface_pressure_mbar']*100.)
        self.assertTrue(self.controllers[0].off.is_set())
        self.assertFalse((self.path / 'faults_20260914.jsonl').read_text())
        self.assertTrue((self.path / 'update_holds_20260914.jsonl').read_text())
        commands = [json.loads(x) for x in (self.path/'commands.jsonl').read_text().splitlines()]
        completions = {x['request_id']: x for x in
            (json.loads(line) for line in (self.path/'events.jsonl').read_text().splitlines())
            if x.get('event') == 'fin_action_completed'}
        self.assertEqual({x['episode_index'] for x in commands}, {0, 1})
        self.assertGreater(len({x['policy_version'] for x in commands}), 1)
        previous_theta, last_direction = 0, 0
        for i, row in enumerate(commands):
            action = row['action']
            self.assertEqual(action['b1'], i % 2)  # 全会话连续检查，不在episode重置。
            self.assertEqual(row['sequence_before'], {'previous_theta': previous_theta,
                'next_b1': i % 2, 'last_effective_direction': last_direction})
            delta = action['theta'] - previous_theta
            direction = (action['b2'] if delta > 0 else -action['b2']) if action['b1'] and delta else 0
            self.assertEqual(row['effective_direction'], direction)
            self.assertTrue(row['action_mask'][row['action_index']])
            if direction:
                self.assertNotEqual(last_direction, direction)
                last_direction = direction
            previous_theta = action['theta']
            self.assertEqual(completions[row['request_id']]['sequence_after'],
                {'previous_theta': previous_theta, 'next_b1': 1-i % 2,
                 'last_effective_direction': last_direction})
        transitions = [json.loads(x) for x in (self.path/'fin_transitions_20260914.jsonl').read_text().splitlines()]
        self.assertTrue(all(len(x['action_mask']) == 18 for x in transitions))
        self.assertTrue(all(len(x['observation']) == 18 + 5*self.cfg['state']['fin_history_length'] for x in transitions))
        # 同版本恢复权重后仍重新回中，首个action从新的b1=0执行序列开始。
        source = self.path
        self.path = source/'resumed'
        self.cfg['training'].update(max_episodes=1, episode_duration_s=.8)
        resumed = self.run_session(resume=source)
        first = json.loads((self.path/'commands.jsonl').read_text().splitlines()[0])
        self.assertEqual(first['sequence_before'], {'previous_theta':0., 'next_b1':0, 'last_effective_direction':0})
        self.assertEqual(first['episode_index'], 2)
        self.assertEqual(first['action']['b1'], 0)
        self.assertEqual(resumed['policy_version'], result['policy_version'])

    def test_old_132_checkpoint_rejected_before_sensors_or_servos_start(self):
        old = self.path/'old_checkpoint.json'
        old.write_text(json.dumps({'format_version':2, 'mode':'fin_only_v2'}))
        from control.runtime.rl_mock_sensors_20260914 import MockRLSensorManager
        with patch.object(MockRLSensorManager, 'start_all') as start:
            with self.assertRaisesRegex(RuntimeError, '132'):
                self.run_session(resume=old)
        start.assert_not_called()
        self.assertFalse(self.controllers[0].commands)

    def test_overcurrent_during_ppo_update_cancels_without_publishing(self):
        # 本用例只注入过流；年龄拦截有独立测试，避免树莓派采样抖动抢先改变故障原因。
        self.cfg['state']['enforce_data_freshness'] = False
        inject = threading.Event()
        original_read = _RLSensorWorker.read_once
        original_update = runtime.FinPPOAgent.update
        owner = self

        def read(worker):
            sample = original_read(worker)
            if worker.name == 'power' and inject.is_set():
                sample.data['current_a'] = 20
            return sample

        def update(agent, *args, **kwargs):
            inject.set()
            owner.assertTrue(owner.controllers[0].off.wait(.8))
            return original_update(agent, *args, **kwargs)

        self.cfg['training']['episode_duration_s'] = 5
        with patch.object(_RLSensorWorker, 'read_once', read), patch.object(runtime.FinPPOAgent, 'update', update):
            with self.assertRaisesRegex(RuntimeError, 'OVERCURRENT'):
                self.run_session()
        status = json.loads((self.path / 'rl_live_status_20260914.json').read_text())
        self.assertEqual(status['published_policy_version'], 0)
        self.assertFalse(status['training_running'])

    def test_sync_failure_stops_and_cleanup_failure_does_not_skip_sensors(self):
        original_stop = runtime.SensorManager.stop_all
        sensor_closed = threading.Event()
        def close(manager, *args, **kwargs):
            original_stop(manager, *args, **kwargs)
            sensor_closed.set()
        original_sync_stop = runtime.RLStateSyncWorker.stop
        def bad_stop(worker, *args, **kwargs):
            original_sync_stop(worker, *args, **kwargs)
            raise RuntimeError('injected cleanup failure')
        with patch.object(runtime.RLStateBuilder, 'build', side_effect=RuntimeError('injected state failure')), \
             patch.object(runtime.RLStateSyncWorker, 'stop', bad_stop), \
             patch.object(runtime.SensorManager, 'stop_all', close):
            with self.assertRaisesRegex(RuntimeError, 'injected state failure'):
                self.run_session()
        self.assertTrue(sensor_closed.is_set())
        self.assertTrue(self.controllers[0].off.is_set())
        status = json.loads((self.path / 'rl_live_status_20260914.json').read_text())
        self.assertTrue(any(e['component'] == 'state_sync' for e in status['cleanup_errors']))

    def test_manual_stop_during_action_disables_output(self):
        stop = threading.Event()
        original_submit = runtime.RLServoExecutor.submit
        def submit(ex, *args, **kwargs):
            rid = original_submit(ex, *args, **kwargs)
            stop.set()
            return rid
        with patch.object(runtime.RLServoExecutor, 'submit', submit):
            with self.assertRaisesRegex(RuntimeError, 'STOP_REQUESTED'):
                self.run_session(stop_event=stop)
        self.assertTrue(self.controllers[0].off.is_set())
        self.assertEqual((self.path / 'fin_transitions_20260914.jsonl').read_text(), '')

    def test_overdepth_uses_calibrated_live_pressure(self):
        self.robot['safety']['max_depth_m'] = .1
        self.cfg['targets']['depth_m'] = .05
        with self.assertRaisesRegex(RuntimeError, 'OVERDEPTH'):
            self.run_session()
        self.assertTrue(self.controllers[0].off.is_set())

    def test_stale_policy_input_never_submits_action(self):
        original_act = runtime.FinPPOAgent.act
        def slow_act(agent, *args, **kwargs):
            time.sleep(.2)
            return original_act(agent, *args, **kwargs)
        with patch.object(runtime.FinPPOAgent, 'act', slow_act):
            with self.assertRaisesRegex(RuntimeError, 'DECISION_STALE'):
                self.run_session()
        self.assertEqual((self.path / 'commands.jsonl').read_text(), '')

    def test_disabled_stale_decision_logs_warning_and_submits_action(self):
        self.cfg['state']['stop_on_stale_decision'] = False
        self.cfg['training']['episode_duration_s'] = 1.2
        original_act = runtime.FinPPOAgent.act
        def slow_act(agent, *args, **kwargs):
            time.sleep(.2)
            return original_act(agent, *args, **kwargs)
        with patch.object(runtime.FinPPOAgent, 'act', slow_act):
            self.run_session()
        self.assertTrue((self.path / 'commands.jsonl').read_text())
        events = [json.loads(line) for line in (self.path / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        warnings = [e for e in events if e.get('event') == 'stale_decision_allowed']
        self.assertTrue(warnings)
        self.assertGreater(warnings[0]['decision_age_ms'], self.cfg['state']['max_state_age_ms'])
        self.assertFalse((self.path / 'faults_20260914.jsonl').read_text())

    def test_stale_decision_switch_defaults_true_and_rejects_strings(self):
        self.cfg['state'].pop('stop_on_stale_decision', None)
        self.assertTrue(prepare_config(self.cfg)['state']['stop_on_stale_decision'])
        self.cfg['state']['stop_on_stale_decision'] = 'false'
        with self.assertRaisesRegex(ValueError, 'stop_on_stale_decision'):
            prepare_config(self.cfg)

    def test_disabled_freshness_continues_with_frozen_sensor_data(self):
        self.cfg['state'].update(enforce_data_freshness=False, stop_on_stale_decision=True)
        self.cfg['training']['episode_duration_s'] = 3.3
        frozen = threading.Event()
        cached = {}
        read_original = _RLSensorWorker.read_once
        act_original = runtime.FinPPOAgent.act
        def read(worker):
            if frozen.is_set() and worker.name in ('imu', 'depth', 'power'):
                if worker.name not in cached:
                    cached[worker.name] = read_original(worker)
                return cached[worker.name]
            return read_original(worker)
        def act(agent, *args, **kwargs):
            frozen.set()
            time.sleep(.2)
            return act_original(agent, *args, **kwargs)
        with patch.object(_RLSensorWorker, 'read_once', read), patch.object(runtime.FinPPOAgent, 'act', act):
            self.run_session()
        states = [json.loads(x) for x in (self.path/'rl_state_20260914.jsonl').read_text(encoding='utf-8').splitlines()]
        for name in ('imu', 'depth', 'power'):
            self.assertGreater(max(s['sensor_age_ms'][name] for s in states), self.cfg['state']['sensor_age'][name]['fault_ms'])
        self.assertGreaterEqual(len((self.path/'commands.jsonl').read_text().splitlines()), 4)
        self.assertFalse((self.path/'faults_20260914.jsonl').read_text())

    def test_freshness_switch_defaults_enabled_and_validates_boolean(self):
        self.cfg['state'].pop('enforce_data_freshness', None)
        self.assertTrue(prepare_config(self.cfg)['state']['enforce_data_freshness'])
        self.cfg['state']['enforce_data_freshness'] = 'false'
        with self.assertRaisesRegex(ValueError, 'enforce_data_freshness'):
            prepare_config(self.cfg)

    def test_disabled_freshness_stalled_state_still_exits_at_episode_limit(self):
        self.cfg['state']['enforce_data_freshness'] = False
        self.cfg['training']['episode_duration_s'] = .45
        stalled = threading.Event()
        watchdog_stop = threading.Event()
        watchdog_fired = threading.Event()
        original_act = runtime.FinPPOAgent.act
        original_sample = runtime.RLStateSyncWorker.sample_once
        def act(agent, *args, **kwargs):
            stalled.set()
            return original_act(agent, *args, **kwargs)
        def sample(worker, *args, **kwargs):
            if not stalled.is_set():
                return original_sample(worker, *args, **kwargs)
        # 仅给测试设上界，防止旧代码无限等待动作终点之后的状态。
        def watchdog():
            watchdog_fired.set()
            watchdog_stop.set()
        timer = threading.Timer(5., watchdog)
        timer.start()
        try:
            with patch.object(runtime.FinPPOAgent, 'act', act), \
                 patch.object(runtime.RLStateSyncWorker, 'sample_once', sample):
                self.run_session(stop_event=watchdog_stop)
        finally:
            timer.cancel()
        self.assertFalse(watchdog_fired.is_set())
        self.assertFalse((self.path/'fin_transitions_20260914.jsonl').read_text())
        events = [json.loads(x) for x in (self.path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(e.get('event') == 'transition_discarded_no_post_action_state' for e in events))


if __name__ == '__main__':
    unittest.main()
