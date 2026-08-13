from __future__ import annotations

import unittest

from control.runtime.action_scheduler import ActionScheduler
from control.runtime.discrete_actions import DiscreteMission, FinAction, TailAction, build_robot_calibration
from control.runtime.sensor_sync_worker import attach_servo_state_by_sensor
from control.runtime.servo_state_tracker import ServoStateTracker
from control.safety import load_robot_config


class ServoStateTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = build_robot_calibration(load_robot_config())
        self.tracker = ServoStateTracker(self.calibration)
        self.mission = DiscreteMission(
            "tracker",
            (TailAction(2.0, 10.0), TailAction(1.0, -10.0)),
            (FinAction(1.0, 1, 0.5),),
            (),
        )
        self.scheduler = ActionScheduler(self.mission)
        self.start_ns = 1_000_000_000
        transitions = self.scheduler.start(
            self.start_ns,
            {
                "tail": {1: 80.0, 2: 90.0, 3: 90.0},
                "left_fin": {4: 121.0, 5: 90.0},
                "right_fin": {6: 143.0, 7: 90.0},
            },
        )
        self.tracker.initialize(self.start_ns - 1)
        self.tracker.set_mission_start(self.start_ns)
        self.tracker.record_actions_started(
            transition.started for transition in transitions if transition.started is not None
        )

    def test_reference_angles_are_computed_at_arbitrary_timestamp(self) -> None:
        snapshot = self.tracker.query(self.start_ns + 1_000_000_000)
        self.assertAlmostEqual(snapshot["reference_angles_deg"][1], 85.0)
        self.assertEqual(snapshot["tail_action_index"], 0)
        self.assertAlmostEqual(snapshot["tail_action_progress"], 0.5)
        self.assertEqual(snapshot["estimation_mode"], "assumed_perfect_tracking")
        self.assertFalse(snapshot["feedback_available"])
        self.assertEqual(
            snapshot["servo_reference_angles_deg"],
            snapshot["reference_angles_deg"],
        )
        self.assertIn("left_action_index", snapshot)
        self.assertIn("right_action_progress", snapshot)

    def test_command_query_uses_latest_success_not_later_than_sample(self) -> None:
        self.tracker.record_successful_command(1, 100, 80.0)
        self.tracker.record_successful_command(1, 200, 90.0)
        self.assertIsNone(self.tracker.query(99)["commanded_angles_deg"][1])
        self.assertEqual(self.tracker.query(150)["commanded_angles_deg"][1], 80.0)
        self.assertEqual(self.tracker.query(200)["commanded_angles_deg"][1], 90.0)

    def test_transition_is_right_continuous_at_completion(self) -> None:
        completion = self.start_ns + 2_000_000_100
        transition = self.scheduler.complete(
            "tail", completion, {1: 90.0, 2: 100.0, 3: 100.0}
        )
        self.tracker.record_transition(transition)
        snapshot = self.tracker.query(completion)
        self.assertEqual(snapshot["tail_action_index"], 1)
        self.assertEqual(snapshot["tail_action_progress"], 0.0)
        self.assertEqual(snapshot["tail_cumulative_angles_deg"][1], 90.0)

    def test_sensor_mapping_uses_each_sensor_sample_timestamp(self) -> None:
        self.tracker.record_successful_command(1, 95, 80.0)
        self.tracker.record_successful_command(1, 105, 90.0)
        sample = {
            "imu": {"sample_t_ns": 110},
            "depth": {"sample_t_ns": 100},
            "power": {"sample_t_ns": None},
            "uwb": {"sample_t_ns": None},
            "vision": {"sample_t_ns": None},
        }
        attach_servo_state_by_sensor(sample, self.tracker)
        states = sample["servo_state_by_sensor"]
        self.assertEqual(states["imu"]["servo_state"]["commanded_angles_deg"][1], 90.0)
        self.assertEqual(states["depth"]["servo_state"]["commanded_angles_deg"][1], 80.0)
        self.assertIsNone(states["power"]["servo_state"])


if __name__ == "__main__":
    unittest.main()
