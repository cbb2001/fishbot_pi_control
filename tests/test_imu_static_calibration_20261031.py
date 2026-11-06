"""Offline physical/numerical regression tests; never import hardware drivers."""
import unittest
import numpy as np
from control.runtime.imu_static_calibration_20261031 import (
    FrozenIMUCalibration, StaticESKF, quat_exp, quat_mul, rotation,
)


def noise():
    return dict(accel_density=[.005]*3,gyro_density=[.0003]*3,
                accel_bias_rw=[.0001]*3,gyro_bias_rw=[.00001]*3,
                zupt_velocity_std_mps=.005,rest_gyro_std_radps=[.0003]*3)


class CalibrationTests(unittest.TestCase):
    def test_rotated_gravity_and_body_bias(self):
        ba=np.array([.03,-.04,.02]);bg=np.array([.001,-.002,.003])
        params=dict(quaternion_order='wxyz',rotation='body_to_world',gravity_mps2=9.80665,
                    accel_bias_body_mps2=ba,gyro_bias_body_radps=bg)
        calibration=FrozenIMUCalibration(params)
        for rv in ([0,0,0],[.4,-.7,.2],[1.5,.1,-2.],[0,0,2.8]):
            q=quat_exp(np.array(rv));R=rotation(q);desired=np.array([.3,-.2,.1])
            measured=R.T@(desired+[0,0,9.80665])+ba
            a,w=calibration.correct(dict(quat=q,acc_mps2=measured,gyro_radps=bg))
            np.testing.assert_allclose(a,desired,atol=1e-12)
            np.testing.assert_allclose(w,0,atol=1e-12)

    def test_predict_preserves_real_acceleration_without_rest_updates(self):
        kf=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise())
        for _ in range(2000):kf.predict([1,0,9.80665],[0,0,0],.005)
        np.testing.assert_allclose(kf.v,[10,0,0],atol=1e-9)
        np.testing.assert_allclose(kf.p,[50,0,0],atol=1e-9)
        self.assertEqual(kf.update_count,0)

    def test_quietness_alone_cannot_stop_constant_velocity(self):
        kf=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise());kf.v[0]=.4
        for _ in range(200):
            kf.predict([0,0,9.80665],[0,0,0],.005)
            self.assertFalse(kf.update_known_rest([0,0,0],quiet=True))
        np.testing.assert_allclose(kf.v,[.4,0,0],atol=1e-12)
        np.testing.assert_allclose(kf.p,[.4,0,0],atol=1e-12)

    def test_static_updates_reduce_drift_and_keep_covariance_positive(self):
        ba=np.array([.025,-.015,.01]);bg=np.array([.001,-.0015,.0008])
        aided=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise())
        free=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise())
        for i in range(4000):
            for kf in (aided,free):kf.predict(ba+[0,0,9.80665],bg,.005)
            if i%20==19:aided.update_known_rest(bg,externally_confirmed_static=True,quiet=True)
        self.assertLess(np.linalg.norm(aided.v),.002)
        self.assertLess(np.linalg.norm(aided.p),np.linalg.norm(free.p)*.01)
        self.assertGreater(np.linalg.eigvalsh(aided.P).min(),-1e-12)
        self.assertAlmostEqual(np.linalg.norm(aided.q),1.,places=12)
        np.testing.assert_allclose(aided.bg,bg,atol=1e-4)

    def test_turn_in_place_does_not_create_translation(self):
        kf=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise())
        for _ in range(1000):kf.predict([0,0,9.80665],[0,0,.2],.005)
        np.testing.assert_allclose(kf.p,0,atol=1e-10)
        np.testing.assert_allclose(rotation(kf.q),rotation(quat_exp(np.array([0,0,1.]))),atol=1e-11)

    def test_invalid_gap_and_quaternion_rejected(self):
        kf=StaticESKF([1,0,0,0],[0,0,0],[0,0,0],noise())
        with self.assertRaises(ValueError):kf.predict([0,0,9.8],[0,0,0],.1)
        with self.assertRaises(ValueError):rotation([0,0,0,0])


if __name__=='__main__':unittest.main()
