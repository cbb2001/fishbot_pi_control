import math
import unittest
from control.runtime.rl_reward_20260914 import RLReward


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.calc=RLReward(target_depth_m=.5,target_heading_deg=0.)
        self.state={'depth_m':.5,'yaw_deg':0.,'pitch_deg':0.,'roll_deg':0.}

    def test_perfect_and_scale_gaussians(self):
        perfect=self.calc.compute(self.state)
        self.assertEqual((perfect.r_depth,perfect.r_heading,perfect.r_pitch,perfect.r_roll),(1.,1.,1.,1.))
        self.assertEqual(perfect.reward,4.)
        for key,delta,component in (('depth_m',.1,'r_depth'),('yaw_deg',15,'r_heading'),('pitch_deg',10,'r_pitch'),('roll_deg',10,'r_roll')):
            state=dict(self.state); state[key]+=delta
            reward=self.calc.compute(state)
            self.assertAlmostEqual(getattr(reward,component),math.exp(-.5))
            for other in ('r_depth','r_heading','r_pitch','r_roll'):
                if other!=component:self.assertEqual(getattr(reward,other),1.)

    def test_nonreward_variables_cannot_change_reward(self):
        state={**self.state,'power_w':1e6,'acc_mps2':[100,100,100],'servo1_estimated_deg':999,'duration_s':.6}
        self.assertEqual(self.calc.compute(state).reward,4.)

    def test_wrapped_yaw_and_roll_errors(self):
        state={**self.state,'yaw_deg':-179,'roll_deg':181}
        r=self.calc.compute(state,target_heading_deg=179)
        self.assertEqual(r.heading_error_deg,2.)
        self.assertEqual(r.roll_error_deg,-179.)
        state['roll_deg']=-179
        self.assertEqual(self.calc.compute(state).r_roll,r.r_roll)

    def test_bad_scales_weights_missing_values_and_nan_rejected(self):
        for cfg in ({'depth_scale_m':0},{'roll_scale_deg':-1},{'pitch_weight':-1},{'depth_weight':float('inf')}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):RLReward(cfg)
        for bad in ({},dict(self.state,pitch_deg=float('nan'))):
            with self.assertRaises(ValueError):self.calc.compute(bad)

    def test_configured_weights_and_targets(self):
        calc=RLReward({'targets':{'depth_m':.5,'heading_deg':0},'reward':{'depth_weight':2.,'heading_weight':3.}})
        self.assertEqual(calc.compute(self.state).reward,7.)
        self.assertEqual(calc.theoretical_max,7.)


if __name__=='__main__':unittest.main()
