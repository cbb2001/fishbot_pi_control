import math
import unittest

from control.runtime.rl_sensor_calibration_20260914 import (
    RLSensorCalibration, CalibrationError, circular_mean_deg, remove_gravity, wrap180,
)


def stationary_imu(pitch=0., yaw=0., roll=0.):
    p,y,r = [math.radians(v)/2 for v in (pitch,yaw,roll)]
    cp,sp,cy,sy,cr,sr = math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
    g = 9.80665
    return {'pitch_deg':pitch,'yaw_deg':yaw,'roll_deg':roll,'quat':quat,
            'acc_mps2':[-g*math.sin(2*p),g*math.sin(2*r)*math.cos(2*p),g*math.cos(2*r)*math.cos(2*p)],
            'gyro_radps':[0.,0.,0.]}


class CalibrationTests(unittest.TestCase):
    def test_fixed_reference_is_not_replaced_by_pressure_samples(self):
        c = self.calibration(surface_pressure_mbar=1038.131)
        c.fit([stationary_imu()]*2, [90000., 130000.])
        self.assertTrue(c.ready)
        self.assertAlmostEqual(c.p_surface_pa, 103813.1)
        self.assertAlmostEqual(c.depth_m(103813.1 + 9806.65 * .3), .3)
        self.assertEqual(c.state_dict()['pressure_reference_mode'], 'fixed_config')
        self.assertEqual(c.state_dict()['depth_sample_count'], 0)
        self.assertIsNone(c.state_dict()['pressure_std_pa'])
        c.calibrate_depth([120000.]*100)
        self.assertAlmostEqual(c.p_surface_pa, 103813.1)

    def test_fixed_reference_still_requires_stable_imu(self):
        c = self.calibration(surface_pressure_mbar=1038.131)
        with self.assertRaises(CalibrationError):
            c.fit([stationary_imu(yaw=-20), stationary_imu(yaw=20)], [])
        self.assertFalse(c.ready)

    def test_fixed_reference_skips_pressure_window(self):
        from control.runtime.rl_sensor_calibration_20260914 import wait_for_calibration
        from control.runtime.ring_buffer import RingBuffer
        from control.runtime.sample import SensorSample
        from unittest.mock import Mock
        buffers = {'imu': RingBuffer(20)}
        for i in range(11):
            buffers['imu'].append(SensorSample('imu', 1_000_000_000 + i*100_000_000, i, stationary_imu()))
        cfg = {'startup': {'calibration_window_s': 1., 'calibration_timeout_s': 2.},
               'calibration': {'minimum_imu_samples': 2, 'surface_pressure_mbar': 1038.131}}
        c = wait_for_calibration(buffers, cfg, clock_ns=Mock(side_effect=[1_000_000_000, 2_000_000_000]))
        self.assertTrue(c.ready)
        self.assertAlmostEqual(c.depth_m(103813.1), 0.)

    def test_invalid_fixed_reference_rejected(self):
        for value in (None, True, False, 0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.calibration(surface_pressure_mbar=value)

    def test_disabled_freshness_calibrates_valid_window_with_delayed_last_sample(self):
        from control.runtime.rl_sensor_calibration_20260914 import wait_for_calibration
        from control.runtime.ring_buffer import RingBuffer
        from control.runtime.sample import SensorSample
        from unittest.mock import Mock
        buffers = {name: RingBuffer(20) for name in ('imu', 'depth')}
        for i in range(9):
            stamp = 1_000_000_000 + i*100_000_000
            buffers['imu'].append(SensorSample('imu', stamp, i, stationary_imu()))
            buffers['depth'].append(SensorSample('depth', stamp, i, {'pressure_pa': 101325.}))
        cfg = {'state': {'enforce_data_freshness': False},
               'startup': {'calibration_window_s': 1., 'calibration_timeout_s': 2.},
               'calibration': {'minimum_imu_samples': 2, 'minimum_depth_samples': 2}}
        clock = Mock(side_effect=[1_000_000_000, 2_000_000_000, 4_000_000_000])
        self.assertTrue(wait_for_calibration(buffers, cfg, clock_ns=clock).ready)

    def calibration(self, **kw):
        return RLSensorCalibration({'minimum_imu_samples':2,'minimum_depth_samples':2,**kw})

    def test_circular_mean_handles_boundary(self):
        self.assertEqual(wrap180(358),-2.)
        self.assertAlmostEqual(abs(circular_mean_deg([179,-179])),180.)
        with self.assertRaises(CalibrationError):
            circular_mean_deg([0,180])

    def test_three_axes_zero_and_raw_gravity_at_tilt(self):
        c = self.calibration()
        raw = stationary_imu(-20,179,30)
        c.fit([raw,raw],[101325.,101325.])
        self.assertTrue(c.ready)
        for value in c.angles(-20,179,30): self.assertAlmostEqual(value,0.)
        for value in c.linear_acceleration(raw): self.assertAlmostEqual(value,0.,places=8)
        self.assertEqual(c.state_dict()['quaternion_order'],'wxyz')

    def test_yaw_wrap_calibration(self):
        c = self.calibration()
        c.fit([stationary_imu(yaw=179),stationary_imu(yaw=-179)],[101325.,101325.])
        self.assertAlmostEqual(c.angles(0,-179,0)[1],1.)

    def test_static_gravity_known_quaternion(self):
        result = remove_gravity([0.,9.80665,0.],[math.sqrt(.5),math.sqrt(.5),0.,0.])
        for value in result: self.assertAlmostEqual(value,0.,places=8)
        self.assertEqual(remove_gravity([1.,2.,12.80665],[1.,0.,0.,0.]),(1.,2.,3.))

    def test_unknown_or_bad_quaternion_rejected(self):
        for q in (None,[0,0,0,0],[1,0,0],[float('nan'),0,0,0]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                remove_gravity([0,0,9.80665],q)

    def test_unstable_angles_pressure_and_translation_rejected(self):
        c = self.calibration()
        with self.assertRaises(CalibrationError):
            c.fit([stationary_imu(yaw=-20),stationary_imu(yaw=20)],[101325]*2)
        self.assertFalse(c.ready)
        with self.assertRaises(CalibrationError):
            c.fit([stationary_imu()]*2,[100000,102000])
        self.assertFalse(c.imu_calibrated)  # fit commits neither zero on failure
        moving = stationary_imu(); moving['acc_mps2'][0] = 2.
        with self.assertRaises(CalibrationError):
            c.fit([moving]*2,[101325]*2)

    def test_coordinate_direction_and_raw_euler_consistency(self):
        c = self.calibration(quaternion_rotation='world_to_body')
        with self.assertRaises(CalibrationError):
            c.fit([stationary_imu(roll=30)]*2,[101325]*2)
        wrong_euler = stationary_imu(); wrong_euler['pitch_deg'] = 30
        with self.assertRaises(CalibrationError):
            self.calibration().fit([wrong_euler]*2,[101325]*2)

    def test_current_surface_pressure_and_depth_sign(self):
        c = self.calibration()
        c.fit([stationary_imu()]*2,[110000,110000])
        self.assertAlmostEqual(c.depth_m(110000),0.)
        self.assertAlmostEqual(c.depth_m(110000+9806.65),1.)
        self.assertLess(c.depth_m(109000),0.)

    def test_missing_nonfinite_and_invalid_configuration(self):
        c = self.calibration()
        for samples in ([{}]*2,[{'pressure_pa':float('nan')}]*2):
            with self.assertRaises(ValueError): c.calibrate_depth(samples)
        for config in ({'gravity_mps2':0},{'minimum_imu_samples':0},{'max_angle_std_deg':float('nan')}):
            with self.assertRaises(ValueError): self.calibration(**config)


if __name__ == '__main__': unittest.main()
