import unittest
import numpy as np
from control.runtime.imu_periodic_static_20261031 import PeriodicStaticExperiment,VendorIntegrator


def noise():
    return dict(accel_density=[.005]*3,gyro_density=[.0003]*3,accel_bias_rw=[.0001]*3,
        gyro_bias_rw=[.00001]*3,zupt_velocity_std_mps=.005,rest_gyro_std_radps=[.0003]*3,
        quiet_acc_std_max_mps2=.03,quiet_gyro_norm_max_radps=.01)


def data(a=0.,w=0.):
    return dict(acc_mps2=[a,0,9.80665],gyro_radps=[w,0,0],quat=[1,0,0,0])


class PeriodicTests(unittest.TestCase):
    def make(self):
        obj=PeriodicStaticExperiment(noise(),t4=1.,window_s=.2,warmup_s=0.,startup_s=.2,
                                    externally_confirmed_static=True)
        for i in range(41):obj.step(i*.005,data())
        return obj

    def test_no_updates_inside_segment_and_exactly_one_per_boundary(self):
        obj=self.make();kf=obj.models['periodic_eskf_zupt']
        for i in range(41,240):
            self.assertIsNone(obj.step(i*.005,data()))
            self.assertEqual(kf.update_count,0)
        event=obj.step(1.2,data())
        self.assertEqual(event['cycle'],1);self.assertEqual(kf.update_count,1)
        for i in range(241,440):obj.step(i*.005,data())
        self.assertEqual(kf.update_count,1)
        obj.step(2.2,data());self.assertEqual(kf.update_count,2)

    def test_velocity_only_keeps_position_while_anchor_resets_it(self):
        obj=self.make()
        for model in obj.models.values():model.v[0]=.4
        for i in range(41,240):obj.step(i*.005,data())
        self.assertAlmostEqual(obj.models['periodic_velocity'].v[0],.4)
        event=obj.step(1.2,data())
        self.assertAlmostEqual(event['before']['periodic_velocity']['p_m'][0],.4)
        self.assertAlmostEqual(event['after']['periodic_velocity']['p_m'][0],.4)
        self.assertEqual(event['after']['periodic_velocity']['v_mps'][0],0.)
        self.assertEqual(event['after']['periodic_surface_anchor']['p_m'][0],0.)
        self.assertAlmostEqual(obj.models['free_vendor'].v[0],.4)

    def test_bias_changes_only_at_boundary_and_uses_past_window(self):
        obj=self.make()
        for i in range(41,240):
            obj.step(i*.005,data(.01))
            self.assertEqual(obj.models['periodic_velocity_bias'].bias[0],0.)
        event=obj.step(1.2,data(.01))
        self.assertAlmostEqual(obj.models['periodic_velocity_bias'].bias[0],.01)
        self.assertLessEqual(event['window_end_capture_s'],1.2)
        self.assertGreaterEqual(event['window_start_capture_s'],.9999999)

    def test_disturbed_boundary_is_skipped_not_delayed_into_prediction(self):
        obj=self.make()
        for i in range(41,241):event=obj.step(i*.005,data(.1 if i%2 else -.1,w=.2))
        self.assertFalse(event['accepted']);self.assertEqual(obj.corrections,0)
        for i in range(241,300):obj.step(i*.005,data())
        self.assertEqual(obj.corrections,0)

    def test_matches_existing_navigation_horizontal_propagation(self):
        from types import SimpleNamespace
        from control.runtime.underwater_navigation_20261026 import InertialEstimator
        cfg=SimpleNamespace(gravity_mps2=9.80665,accel_tau_s=.2,imu_gap_s=.05,history_s=5.)
        nav=InertialEstimator(cfg);bench=VendorIntegrator([0,0,0],tau=.2)
        for i in range(501):
            row=data(.1*np.sin(i*.03));row['tid']=i+1
            nav.imu(i*5000000,row);bench.predict(np.array(row['acc_mps2']),np.eye(3),0 if i==0 else .005)
        np.testing.assert_allclose(nav.p[:2],bench.p[:2],atol=1e-12)
        np.testing.assert_allclose(nav.v[:2],bench.v[:2],atol=1e-12)


if __name__=='__main__':unittest.main()
