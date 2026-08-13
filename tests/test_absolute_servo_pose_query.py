from __future__ import annotations

import unittest

from control.runtime.absolute_action_scheduler import AbsoluteActionScheduler
from control.runtime.absolute_discrete_actions import (
    AbsoluteMission, FinAbsoluteAction, TailAbsoluteAction, build_absolute_calibration,
    quintic_smoothstep,
)
from control.runtime.absolute_servo_state_tracker import ServoStateTracker
from control.safety import load_robot_config


class AbsoluteServoPoseQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cal = build_absolute_calibration(load_robot_config())
        mission = AbsoluteMission("query", (TailAbsoluteAction(100, 90, 90, 20, 20, 20),),
                                  (FinAbsoluteAction(141, 20, 1),), ())
        self.scheduler = AbsoluteActionScheduler(mission, self.cal)
        self.tracker = ServoStateTracker(self.cal)
        self.tracker.initialize(900)
        self.tracker.set_mission_start(1_000)
        transitions = self.scheduler.start(1_000, {
            "tail": {1: 80, 2: 90, 3: 90}, "left_fin": {4: 121, 5: 90},
            "right_fin": {6: 143, 7: 90}})
        self.tracker.record_actions_started(x.started for x in transitions if x.started)

    def test_before_start_start_middle_and_after_completion(self) -> None:
        self.assertEqual(
            self.tracker.get_servo_pose_at(999).data["reference_angles_deg"][1],
            self.cal.servos[1].center_deg,
        )
        start = self.tracker.get_servo_pose_at(1_000).data
        self.assertEqual(start["tail_action_progress"], 0)
        midpoint = self.tracker.get_servo_pose_at(500_001_000).data
        self.assertAlmostEqual(midpoint["reference_angles_deg"][1], 90)
        self.assertNotAlmostEqual(midpoint["reference_angles_deg"][1],
                                  80 + 20 * 0.25)  # 证明不是简单线性插值。
        tail = self.scheduler.active("tail")
        completion = self.scheduler.complete("tail", tail.planned_end_t_ns,
                                             tail.trajectory.target_angles_deg)
        self.tracker.record_transition(completion)
        after = self.tracker.get_servo_pose_at(tail.planned_end_t_ns + 1).data
        self.assertEqual(after["reference_angles_deg"][1], 100)

    def test_tip_historical_segments_and_command_history(self) -> None:
        left = self.scheduler.active("left_fin")
        duration = left.duration_ns
        quarter = self.tracker.get_servo_pose_at(1_000 + duration // 4).data
        self.assertAlmostEqual(
            quarter["reference_angles_deg"][5],
            self.cal.servos[5].center_deg
            + 20
            / self.cal.left_coupling.root_reference_span_deg
            * self.cal.left_coupling.tip_reference_span_deg,
        )
        end = self.tracker.get_servo_pose_at(1_000 + duration).data
        self.assertAlmostEqual(
            end["reference_angles_deg"][5],
            self.cal.servos[5].center_deg,
        )
        self.tracker.record_successful_command(1, 2_000, 81)
        self.tracker.record_successful_command(1, 3_000, 82)
        self.assertIsNone(self.tracker.get_servo_pose_at(1_999).data["commanded_angles_deg"][1])
        self.assertEqual(self.tracker.get_servo_pose_at(2_500).data["commanded_angles_deg"][1], 81)


if __name__ == "__main__":
    unittest.main()
