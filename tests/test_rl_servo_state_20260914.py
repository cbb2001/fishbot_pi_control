import unittest
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.rl_actions_20260914 import FinAction, build_rl_trajectory

class RLServoStateTests(unittest.TestCase):
    def test_timestamp_query_and_command_estimate(self):
        t=RLServoStateTracker({i:90 for i in range(1,8)})
        t.record_reference_segment(0,1_000_000_000,{i:90 for i in range(1,8)},{i:100 for i in range(1,8)})
        t.record_successful_batch(500_000_000,{i:95 for i in range(1,8)})
        p=t.get_servo_pose_at(500_000_000)
        self.assertAlmostEqual(p.reference_angle_deg[1],95); self.assertEqual(p.commanded_angle_deg[1],95)
    def test_nonmonotonic_commands_rejected(self):
        t=RLServoStateTracker({i:90 for i in range(1,8)}); t.record_successful_batch(2,{1:90})
        with self.assertRaises(ValueError): t.record_successful_batch(1,{1:90})

    def test_fin_subset_segment_preserves_tail(self):
        t = RLServoStateTracker({i:90 for i in range(1,8)})
        t.record_reference_segment(0, 1_000_000_000, {4:90, 5:90, 6:90, 7:90}, {4:110, 5:90, 6:70, 7:90})
        pose = t.get_servo_pose_at(500_000_000)
        self.assertEqual(pose.reference_angle_deg[4], 100)
        self.assertEqual(pose.reference_angle_deg[1], 90)

    def test_recorded_fin_trajectory_tracks_tip_return_and_successful_estimate(self):
        t = RLServoStateTracker({i:90 for i in range(1,8)})
        tr = build_rl_trajectory(FinAction(53, .6, 1, 1), 0, t.centers)
        t.record_trajectory(0, tr)
        t.record_successful_batch(300_000_000, {4:116.5,5:135,6:63.5,7:45})
        pose = t.get_servo_pose_at(300_000_000)
        self.assertEqual(pose.reference_angle_deg[5], 135)
        self.assertEqual(pose.estimated_angle_deg[5], 135)
        self.assertEqual(t.get_servo_pose_at(600_000_000).reference_angle_deg[5], 90)
        self.assertFalse(pose.to_dict()['feedback_available'])

if __name__ == '__main__': unittest.main()
