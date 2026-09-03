from __future__ import annotations

import unittest

from control.runtime.action_scheduler_20260725 import ActionScheduler
from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    FinAction,
    TailAction,
    build_calibration,
    quintic_smoothstep,
)
from control.runtime.sensor_manager import SENSOR_NAMES
from control.runtime.sensor_sync_worker import attach_servo_state_by_sensor
from control.runtime.servo_state_tracker_20260725 import ServoStateTracker
from control.safety import load_robot_config


class ServoPoseQuery20260725Tests(unittest.TestCase):
    """验证任意时间戳重放、保持阶段和传感器 sample_t_ns 对齐。"""

    def setUp(self) -> None:
        self.calibration = build_calibration(load_robot_config())
        self.start = 10_000_000_000
        mission = DiscreteAbsoluteMission(
            "pose",
            (TailAction(10.0, 1.0), TailAction(10.0, 1.0)),
            (FinAction(164.0, 2.0, 1, 1),),
            (FinAction(143.0, 1.0, 1, -1),),
        )
        self.scheduler = ActionScheduler(mission, self.calibration)
        self.tracker = ServoStateTracker(self.calibration)
        self.tracker.initialize(self.start - 1)
        self.tracker.set_mission_start(self.start)
        transitions = self.scheduler.start(self.start)
        self.tracker.record_actions_started(
            transition.started
            for transition in transitions
            if transition.started is not None
        )

    def _complete_tail_first(self) -> None:
        tail = self.scheduler.active("tail")
        transition = self.scheduler.complete(
            "tail",
            tail.planned_end_t_ns,
            tail.trajectory.target_angles_deg,
            endpoint_written=True,
        )
        self.tracker.record_transition(transition)

    def test_historical_motion_uses_quintic_not_linear_interpolation(self) -> None:
        query = self.start + 250_000_000
        pose = self.tracker.get_servo_pose_at(query).data
        expected = 85.0 + 10.0 * quintic_smoothstep(0.25)
        self.assertAlmostEqual(pose["reference_angles_deg"][1], expected)
        self.assertNotAlmostEqual(pose["reference_angles_deg"][1], 87.5)

    def test_fin_tip_reaches_peak_at_quarter_and_holds_until_three_quarters(
        self,
    ) -> None:
        start_pose = self.tracker.get_servo_pose_at(self.start).data
        quarter_pose = self.tracker.get_servo_pose_at(
            self.start + 500_000_000
        ).data
        half_pose = self.tracker.get_servo_pose_at(
            self.start + 1_000_000_000
        ).data
        three_quarters_pose = self.tracker.get_servo_pose_at(
            self.start + 1_500_000_000
        ).data
        end_pose = self.tracker.get_servo_pose_at(
            self.start + 2_000_000_000
        ).data
        expected_peak = 135.0
        self.assertEqual(start_pose["reference_angles_deg"][5], 90.0)
        self.assertEqual(quarter_pose["reference_angles_deg"][5], expected_peak)
        self.assertAlmostEqual(half_pose["reference_angles_deg"][5], expected_peak)
        self.assertEqual(
            three_quarters_pose["reference_angles_deg"][5],
            expected_peak,
        )
        self.assertEqual(end_pose["reference_angles_deg"][5], 90.0)
        self.assertAlmostEqual(half_pose["left_tip_peak_angle_deg"], expected_peak)

    def test_equal_theta_second_action_holds_exact_angle_for_full_duration(self) -> None:
        self._complete_tail_first()
        second = self.scheduler.active("tail")
        self.assertEqual(second.previous_theta, second.current_theta)
        for offset_ns in (0, 100_000_000, 500_000_000, 999_999_999):
            pose = self.tracker.get_servo_pose_at(
                second.actual_start_t_ns + offset_ns
            ).data
            self.assertEqual(pose["reference_angles_deg"][1], 95.0)
            self.assertEqual(pose["reference_angles_deg"][2], 105.0)
            self.assertEqual(pose["reference_angles_deg"][3], 105.0)
            self.assertEqual(pose["tail_action_index"], 1)
        middle = self.tracker.get_servo_pose_at(
            second.actual_start_t_ns + 500_000_000
        ).data
        self.assertEqual(middle["tail_action_progress"], 0.5)

    def test_each_sensor_sample_uses_its_own_timestamp(self) -> None:
        command_t_ns = self.start + 500_000_000
        self.tracker.record_successful_command(1, command_t_ns, 90.0)
        sample = {"t_ns": self.start + 800_000_000}
        for index, name in enumerate(SENSOR_NAMES):
            sample[name] = {
                "sample_t_ns": self.start + index * 250_000_000
            }
        attach_servo_state_by_sensor(sample, self.tracker)
        for index, name in enumerate(SENSOR_NAMES):
            state = sample["servo_state_by_sensor"][name]["servo_state"]
            expected_t_ns = self.start + index * 250_000_000
            self.assertEqual(state["query_t_ns"], expected_t_ns)
            if expected_t_ns < command_t_ns:
                self.assertIsNone(state["commanded_angles_deg"][1])
            else:
                self.assertEqual(state["commanded_angles_deg"][1], 90.0)

    def test_synchronized_top_level_contains_previous_and_current_theta(self) -> None:
        sample = {"t_ns": self.start + 250_000_000}
        for name in SENSOR_NAMES:
            sample[name] = {"sample_t_ns": None}
        attach_servo_state_by_sensor(sample, self.tracker)
        for key in (
            "previous_action1_theta",
            "previous_action2_theta",
            "previous_action3_theta",
            "current_action1_theta",
            "current_action2_theta",
            "current_action3_theta",
        ):
            self.assertIn(key, sample)
        self.assertEqual(sample["previous_action1_theta"], 0.0)
        self.assertEqual(sample["current_action1_theta"], 10.0)

    def test_pose_after_sequence_finish_holds_final_posture(self) -> None:
        self._complete_tail_first()
        second = self.scheduler.active("tail")
        transition = self.scheduler.complete(
            "tail",
            second.planned_end_t_ns,
            second.trajectory.target_angles_deg,
            endpoint_written=True,
        )
        self.tracker.record_transition(transition)
        pose = self.tracker.get_servo_pose_at(
            second.planned_end_t_ns + 5_000_000_000
        ).data
        self.assertEqual(pose["reference_angles_deg"][1], 95.0)
        self.assertEqual(pose["previous_action1_theta"], 10.0)
        self.assertEqual(pose["tail_action_progress"], 1.0)


if __name__ == "__main__":
    unittest.main()
