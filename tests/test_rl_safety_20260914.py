import queue
import threading
import unittest
from types import SimpleNamespace

from control.runtime.rl_safety_20260914 import RLSafetyMonitor, RLSafetyFault
from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.buffers = {n: RingBuffer(10) for n in ('imu', 'depth', 'power')}
        self.sensors = SimpleNamespace(buffers=self.buffers, fatal_errors=lambda: [])
        self.stopped = threading.Event()
        self.ex = SimpleNamespace(failure=None, is_alive=lambda: True,
                                  request_emergency_stop=self.stopped.set)
        self.stop = threading.Event()
        self.q = queue.Queue()
        self.guard = RLSafetyMonitor(self.sensors, self.ex, self.q, self.stop,
                                    {'state': {'max_state_age_ms': 150}},
                                    {'safety': {'min_voltage_v': 10.5, 'max_current_a': 10, 'max_depth_m': 5}},
                                    clock_ns=lambda: self.now)
        self.sample('imu', {})
        self.sample('depth', {'pressure_pa': 101325})
        self.sample('power', {'voltage_v': 12, 'current_a': .3, 'power_w': 3.6})

    def sample(self, name, data, **kw):
        self.buffers[name].append(SensorSample(name, self.now, 1, data, **kw))

    def test_voltage_current_and_nonfinite_are_faults(self):
        for values, code in [({'voltage_v': 10}, 'LOW_VOLTAGE'),
                             ({'current_a': -11}, 'OVERCURRENT'),
                             ({'power_w': float('nan')}, 'POWER_INVALID')]:
            with self.subTest(code=code):
                self.sample('power', {'voltage_v': 12, 'current_a': .3, 'power_w': 3.6, **values})
                with self.assertRaises(RLSafetyFault) as caught:
                    self.guard.check_once()
                self.assertEqual(caught.exception.code, code)

    def test_imu_timeout_cannot_be_disabled_by_manual_config(self):
        self.now += 110_000_000
        with self.assertRaisesRegex(RLSafetyFault, 'IMU_STALE'):
            self.guard.check_once()

    def test_disabled_data_freshness_accepts_old_samples_and_state(self):
        self.guard = RLSafetyMonitor(self.sensors, self.ex, self.q, self.stop,
            {'state': {'enforce_data_freshness': False}},
            {'safety': {'min_voltage_v': 10.5, 'max_current_a': 10, 'max_depth_m': 5}},
            clock_ns=lambda: self.now)
        self.now += 10_000_000_000
        self.guard.statebuf = SimpleNamespace(latest=lambda: SimpleNamespace(t_ns=1_000_000_000))
        self.guard.check_once()
        self.assertEqual(set(self.guard.data_age_warnings), {'imu', 'depth', 'power', 'rl_state'})
        self.assertEqual(self.guard.latest_data_age_ms['imu'], 10000)
        self.sample('power', {'voltage_v': 9, 'current_a': .3, 'power_w': 2.7})
        with self.assertRaisesRegex(RLSafetyFault, 'LOW_VOLTAGE'):
            self.guard.check_once()

    def test_disabled_freshness_still_rejects_missing_invalid_and_future(self):
        self.guard = RLSafetyMonitor(self.sensors, self.ex, self.q, self.stop,
            {'state': {'enforce_data_freshness': False}},
            {'safety': {'min_voltage_v': 10.5, 'max_current_a': 10, 'max_depth_m': 5}},
            clock_ns=lambda: self.now)
        self.now += 6_000_000_000
        self.buffers['imu'] = RingBuffer(10)
        with self.assertRaisesRegex(RLSafetyFault, 'IMU_MISSING'):
            self.guard.check_once()
        self.sample('imu', {}, ok=False, error='decode failed')
        with self.assertRaisesRegex(RLSafetyFault, 'IMU_INVALID'):
            self.guard.check_once()
        self.buffers['imu'].append(SensorSample('imu', self.now+1, 2, {}))
        with self.assertRaises(RLSafetyFault):
            self.guard.check_once()

    def test_depth_uses_new_calibration_not_legacy_depth(self):
        self.guard.calibration = SimpleNamespace(depth_m=lambda p: (p - 101325) / 9806.65)
        self.sample('depth', {'pressure_pa': 101325, 'depth_m': 999})
        self.guard.check_once()
        self.sample('depth', {'pressure_pa': 101325 + 9806.65 * 6})
        with self.assertRaisesRegex(RLSafetyFault, 'OVERDEPTH'):
            self.guard.check_once()

    def test_state_stall_and_thread_faults(self):
        self.guard.statebuf = SimpleNamespace(latest=lambda: SimpleNamespace(t_ns=self.now - 200_000_000))
        with self.assertRaisesRegex(RLSafetyFault, 'RL_STATE_STALE'):
            self.guard.check_once()
        self.guard.statebuf = None
        self.q.put({'fault_code': 'TEST_WORKER_FAILURE', 'message': 'injected'})
        with self.assertRaisesRegex(RLSafetyFault, 'TEST_WORKER_FAILURE'):
            self.guard.check_once()

    def test_missing_sensor_allowed_only_during_startup_grace(self):
        self.buffers['power'] = RingBuffer(10)
        self.guard.check_once()
        self.now += 6_000_000_000
        self.sample('imu', {})
        self.sample('depth', {'pressure_pa': 101325})
        with self.assertRaisesRegex(RLSafetyFault, 'POWER_MISSING'):
            self.guard.check_once()

    def test_watchdog_stops_output_without_main_thread_polling(self):
        self.sample('power', {'voltage_v': 9, 'current_a': .3, 'power_w': 2.7})
        self.guard.start()
        try:
            self.assertTrue(self.stopped.wait(1))
            self.assertTrue(self.stop.is_set())
            with self.assertRaisesRegex(RLSafetyFault, 'LOW_VOLTAGE'):
                self.guard.raise_if_failed()
        finally:
            self.guard.stop()

    def test_invalid_sample_and_silent_thread_exit(self):
        self.sample('imu', {}, ok=False, error='CRC decode failed')
        with self.assertRaisesRegex(RLSafetyFault, 'IMU_INVALID'):
            self.guard.check_once()
        self.sample('imu', {})
        self.ex.is_alive = lambda: False
        self.guard.executor_started = True
        with self.assertRaisesRegex(RLSafetyFault, 'SERVO_WORKER_STOPPED'):
            self.guard.check_once()

    def test_future_sample_is_not_treated_as_fresh(self):
        self.buffers['depth'].append(SensorSample('depth', self.now + 1, 2, {'pressure_pa': 101325}))
        with self.assertRaisesRegex(RLSafetyFault, 'DEPTH_STALE'):
            self.guard.check_once()


class SafetyConfigTests(unittest.TestCase):
    def test_invalid_runtime_or_startup_rejected_before_starting(self):
        from pathlib import Path
        import copy
        import yaml
        from control.runtime.rl_training_config_20260914 import prepare_config
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/rl_training_20260914.yaml').read_text(encoding='utf-8'))
        for section, key, value in [('runtime', 'safety_poll_interval_s', 0),
                                    ('runtime', 'poll_interval_s', float('nan')),
                                    ('runtime', 'torch_num_threads', 1.5),
                                    ('startup', 'pretrain_wait_s', -1),
                                    ('state', 'max_state_age_ms', float('inf'))]:
            cfg = copy.deepcopy(config)
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_config(cfg)


if __name__ == '__main__':
    unittest.main()
