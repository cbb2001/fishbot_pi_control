from __future__ import annotations

import unittest

from control.runtime.action_scheduler_20260725 import ActionScheduler
from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    FinAction,
    TailAction,
    build_calibration,
)
from control.runtime.servo_state_tracker_20260725 import (
    ESTIMATION_MODE,
    ServoStateTracker,
)
from control.safety import load_robot_config


class ServoStateTracker20260725Tests(unittest.TestCase):
    """覆盖 previous/current 历史语义和最近成功命令二分查询。"""

    def setUp(self) -> None:
        self.calibration = build_calibration(load_robot_config())
        self.mission_start = 1_000_000_000
        mission = DiscreteAbsoluteMission(
            "tracker",
            (TailAction(20.0, 2.0),),
            (FinAction(150.0, 2.0, 1, 1),),
            (),
        )
        self.scheduler = ActionScheduler(mission, self.calibration)
        self.tracker = ServoStateTracker(self.calibration)
        self.tracker.initialize(self.mission_start - 100)
        self.tracker.set_mission_start(self.mission_start)
        transitions = self.scheduler.start(self.mission_start)
        self.tracker.record_actions_started(
            transition.started
            for transition in transitions
            if transition.started is not None
        )

    def test_before_mission_returns_initial_reference_and_previous_values(self) -> None:
        pose = self.tracker.get_servo_pose_at(self.mission_start - 1).data
        self.assertEqual(
            pose["reference_angles_deg"],
            self.calibration.initial_angles_deg,
        )
        self.assertEqual(pose["previous_action1_theta"], 0.0)
        self.assertEqual(pose["previous_action2_theta"], 121.0)
        self.assertEqual(pose["previous_action3_theta"], 143.0)
        self.assertIsNone(pose["tail_action_index"])

    def test_active_pose_keeps_previous_uncommitted_until_endpoint_write(self) -> None:
        midpoint = self.tracker.get_servo_pose_at(
            self.mission_start + 1_000_000_000
        ).data
        self.assertEqual(midpoint["tail_action_index"], 0)
        self.assertEqual(midpoint["tail_action_progress"], 0.5)
        self.assertEqual(midpoint["reference_angles_deg"][1], 95.0)
        self.assertEqual(midpoint["previous_action1_theta"], 0.0)
        self.assertEqual(midpoint["current_action1_theta"], 20.0)

        tail = self.scheduler.active("tail")
        # 规定时间已经结束，但尚未调用成功终点提交；previous 仍为 0。
        due_pose = self.tracker.get_servo_pose_at(tail.planned_end_t_ns).data
        self.assertEqual(due_pose["reference_angles_deg"][1], 105.0)
        self.assertEqual(due_pose["tail_action_progress"], 1.0)
        self.assertEqual(due_pose["previous_action1_theta"], 0.0)

    def test_successful_completion_updates_historical_previous_at_completion(self) -> None:
        tail = self.scheduler.active("tail")
        completion_t_ns = tail.planned_end_t_ns + 123
        transition = self.scheduler.complete(
            "tail",
            completion_t_ns,
            tail.trajectory.target_angles_deg,
            endpoint_written=True,
        )
        self.tracker.record_transition(transition)
        before = self.tracker.get_servo_pose_at(completion_t_ns - 1).data
        after = self.tracker.get_servo_pose_at(completion_t_ns).data
        self.assertEqual(before["previous_action1_theta"], 0.0)
        self.assertEqual(after["previous_action1_theta"], 20.0)
        self.assertEqual(after["current_action1_theta"], 20.0)
        self.assertEqual(after["tail_action_index"], 0)
        self.assertEqual(after["tail_action_progress"], 1.0)

    def test_commanded_query_uses_latest_success_not_later_than_query(self) -> None:
        self.tracker.record_successful_command(1, self.mission_start + 10, 85.0)
        self.tracker.record_successful_command(1, self.mission_start + 20, 86.0)
        before = self.tracker.get_servo_pose_at(self.mission_start + 9).data
        middle = self.tracker.get_servo_pose_at(self.mission_start + 15).data
        after = self.tracker.get_servo_pose_at(self.mission_start + 20).data
        self.assertIsNone(before["commanded_angles_deg"][1])
        self.assertEqual(middle["commanded_angles_deg"][1], 85.0)
        self.assertEqual(after["commanded_angles_deg"][1], 86.0)

    def test_estimated_is_reference_and_explicitly_not_feedback(self) -> None:
        pose = self.tracker.get_servo_pose_at(
            self.mission_start + 500_000_000
        ).data
        self.assertEqual(
            pose["estimated_angles_deg"],
            pose["reference_angles_deg"],
        )
        self.assertFalse(pose["feedback_available"])
        self.assertEqual(pose["estimation_mode"], ESTIMATION_MODE)

    def test_elapsed_query_uses_same_monotonic_axis(self) -> None:
        direct = self.tracker.get_servo_pose_at(
            self.mission_start + 750_000_000
        ).data
        elapsed = self.tracker.get_servo_pose_at_elapsed_s(0.75).data
        self.assertEqual(elapsed, direct)

    def test_fractional_nanosecond_planned_end_is_exact_target(self) -> None:
        mission = DiscreteAbsoluteMission(
            "fractional_nanosecond",
            (TailAction(30.0, 1.4e-9),),
            (),
            (),
        )
        scheduler = ActionScheduler(mission, self.calibration)
        tracker = ServoStateTracker(self.calibration)
        start_t_ns = 2_000_000_000
        tracker.initialize(start_t_ns - 1)
        tracker.set_mission_start(start_t_ns)
        transitions = scheduler.start(start_t_ns)
        tracker.record_actions_started(
            transition.started
            for transition in transitions
            if transition.started is not None
        )

        tail = scheduler.active("tail")
        self.assertEqual(tail.planned_end_t_ns, start_t_ns + 1)
        pose = tracker.get_servo_pose_at(tail.planned_end_t_ns).data
        self.assertEqual(pose["tail_action_progress"], 1.0)
        self.assertEqual(
            pose["reference_angles_deg"],
            {
                1: 115.0,
                2: 125.0,
                3: 125.0,
                4: 121.0,
                5: 94.0,
                6: 143.0,
                7: 90.0,
            },
        )

    def test_history_owns_private_trajectory_dictionaries(self) -> None:
        tail = self.scheduler.active("tail")
        tail.trajectory.start_angles_deg[1] = -999.0
        tail.trajectory.target_angles_deg[1] = 999.0

        start_pose = self.tracker.get_servo_pose_at(self.mission_start).data
        end_pose = self.tracker.get_servo_pose_at(tail.planned_end_t_ns).data
        self.assertEqual(start_pose["reference_angles_deg"][1], 85.0)
        self.assertEqual(end_pose["reference_angles_deg"][1], 105.0)


if __name__ == "__main__":
    unittest.main()
