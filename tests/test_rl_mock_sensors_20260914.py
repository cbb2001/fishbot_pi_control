import time
import unittest

from control.runtime.rl_mock_sensors_20260914 import MockRLSensorManager
from control.runtime.rl_sensor_calibration_20260914 import wait_for_calibration
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.rl_state_builder_20260914 import RLStateBuilder


class RLMockCalibrationTests(unittest.TestCase):
    def test_surface_calibration_then_valid_tilted_depth_state(self):
        for rotation, sign in (('body_to_world', 1), ('world_to_body', -1)):
            with self.subTest(rotation=rotation, sign=sign):
                config = {
                    'runtime': {'mock': False},
                    'sensors': {'imu': {'sample_rate_hz': 100},
                                'depth': {'rate_hz': 100},
                                'power': {'rate_hz': 100},
                                'uwb': {'enabled': False}, 'vision': {'enabled': False}},
                    'startup': {'calibration_window_s': .2, 'calibration_timeout_s': 2.},
                    'calibration': {'minimum_imu_samples': 8, 'minimum_depth_samples': 8,
                                    'quaternion_rotation': rotation, 'gravity_world_z_sign': sign},
                }
                sensors = MockRLSensorManager(config)
                try:
                    sensors.set_surface_mode(True)
                    sensors.start_all()
                    calibration = wait_for_calibration(sensors, config)
                    self.assertTrue(calibration.ready)
                    self.assertGreaterEqual(calibration.state_dict()['imu_sample_count'], 8)
                    self.assertAlmostEqual(calibration.p_surface_pa, 101325.)
                    builder = RLStateBuilder(sensors, calibration, RLServoStateTracker(), config)
                    surface = builder.build()
                    self.assertAlmostEqual(surface.depth_m, 0.)
                    for value in (surface.pitch_deg, surface.yaw_deg, surface.roll_deg):
                        self.assertAlmostEqual(value, 0.)
                    sensors.set_surface_mode(False)
                    time.sleep(.07)
                    submerged = builder.build()
                    self.assertGreater(submerged.depth_m, 0.)
                    self.assertNotEqual(submerged.raw_angles_deg, surface.raw_angles_deg)
                    self.assertLess(calibration.validate_orientation(
                        sensors.buffers['imu'].latest().data), .001)
                    for value in submerged.linear_acc_mps2:
                        self.assertAlmostEqual(value, 0., places=8)
                    self.assertEqual(sensors.fatal_errors(), [])
                finally:
                    sensors.stop_all(timeout=1.)
                self.assertEqual(sensors.alive_worker_names(), [])


if __name__ == '__main__':
    unittest.main()
