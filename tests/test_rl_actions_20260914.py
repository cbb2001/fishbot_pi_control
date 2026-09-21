import unittest
from control.runtime.rl_actions_20260914 import *
from control.runtime.rl_action_scheduler_20260914 import RLActionScheduler

class RLActionTests(unittest.TestCase):
    def test_spaces_and_half_fin_motion(self):
        self.assertEqual(len(TailActionSpace()), 35); self.assertEqual(len(FinActionSpace()), 18)
        self.assertAlmostEqual(evaluate_fin_tip_motion(90,180,.3,.6),180)
        self.assertAlmostEqual(evaluate_fin_tip_motion(90,180,.6,.6),90)
    def test_mirror_and_invalid_action(self):
        self.assertEqual(mirror_fin_angles({4:111,5:180}), {6:143,7:0})
        with self.assertRaises(ActionValidationError): TailAction(31,.2)

    def test_fin_decode_has_unique_executable_actions(self):
        space = FinActionSpace()
        self.assertEqual(space.nvec, (3, 2, 3))
        actions = space.all()
        self.assertEqual(len(set(actions)), 18)
        self.assertEqual(space.decode(0, 0, 0), FinAction(-53, .3, 0, 0))
        self.assertEqual(space.decode(2, 1, 2), FinAction(53, .6, 1, 1))
        self.assertEqual(FinAction(0, .3, 0, -1).b2, 0)
        self.assertEqual(FinAction(0, .3, 0, 1).b2, 0)
        with self.assertRaises(ActionValidationError): FinAction(0, .2, 0, 0)
        with self.assertRaises(ActionValidationError): FinAction(0, .3, 1, 0)

    def test_coupling_and_independent_right_tip_center(self):
        cfg = {'servo': {'channels': [dict(servo_id=i, center_angle=100 if i == 7 else 90, min_angle=-200, max_angle=300) for i in ALL_SERVO_IDS]}, 'discrete_action_coupling': {'left': {'root_reference_span_deg': 100, 'tip_reference_span_deg': 50}}}
        cal = build_calibration(cfg)
        tr = build_rl_trajectory(FinAction(53, .6, 1, 1), 0, cal)
        self.assertEqual(tr.evaluate_all(.3), {4:116.5, 5:116.5, 6:63.5, 7:73.5})
        self.assertEqual(tr.evaluate_all(.6)[7], 100)

    def test_fin_only_validation_ignores_tail_and_checks_right_tip_peak(self):
        cfg = {'servo': {'channels': [dict(servo_id=i, center_angle=90, min_angle=0, max_angle=180) for i in ALL_SERVO_IDS]}}
        cfg['servo']['channels'][0].update(min_angle=90, max_angle=90)
        cal = build_calibration(cfg)
        self.assertTrue(validate_action_space(cal, agents=('fin',)))
        cfg['servo']['channels'][6]['min_angle'] = 10
        with self.assertRaises(ActionValidationError): validate_action_space(build_calibration(cfg), agents=('fin',))

    def test_scheduler_preserves_right_tip_center_and_checks_mirrored_peak(self):
        centers = {i:90. for i in ALL_SERVO_IDS}
        centers[7] = 100.
        limits = {i:(0.,200.) for i in ALL_SERVO_IDS}
        s = RLActionScheduler(centers, limits=limits)
        s.schedule('fin', FinAction(53,.3,0,0), start_t_ns=0)
        self.assertEqual(s.references_at(400_000_000)[7], 100.)
        s.complete('fin', endpoint_written=True, completion_t_ns=400_000_000)
        self.assertEqual(s.references_at(500_000_000)[7], 100.)
        limits[7] = (99.,200.)
        with self.assertRaises(ActionValidationError): s.schedule('fin', FinAction(-53,.3,1,1), start_t_ns=500_000_000)

if __name__ == '__main__': unittest.main()
