from __future__ import annotations

import unittest

from control.runtime.absolute_action_scheduler import AbsoluteActionScheduler
from control.runtime.absolute_discrete_actions import (
    AbsoluteMission, FinAbsoluteAction, TailAbsoluteAction, build_absolute_calibration,
)
from control.safety import load_robot_config


class AbsoluteActionSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cal = build_absolute_calibration(load_robot_config())

    def test_parallel_start_and_serial_completion_start(self) -> None:
        mission = AbsoluteMission("parallel",
            (TailAbsoluteAction(90, 90, 90, 10, 10, 10), TailAbsoluteAction(80, 90, 90, 10, 10, 10)),
            (FinAbsoluteAction(141, 20, 0),), (FinAbsoluteAction(163, 20, 0),))
        scheduler = AbsoluteActionScheduler(mission, self.cal)
        starts = scheduler.start(1_000_000_000, {
            "tail": {1: 80, 2: 90, 3: 90}, "left_fin": {4: 121, 5: 90},
            "right_fin": {6: 143, 7: 90}})
        self.assertEqual({x.started.actual_start_t_ns for x in starts}, {1_000_000_000})
        tail = scheduler.active("tail")
        self.assertEqual(tail.planned_end_t_ns, 2_000_000_000)
        with self.assertRaises(RuntimeError):
            scheduler.complete("tail", tail.planned_end_t_ns - 1, tail.trajectory.target_angles_deg)
        completion = tail.planned_end_t_ns + 123
        transition = scheduler.complete("tail", completion, tail.trajectory.target_angles_deg)
        self.assertEqual(transition.started.actual_start_t_ns, completion)
        self.assertEqual(transition.started.trajectory.start_angles_deg[1], 90)

    def test_empty_sequences_complete_at_mission_start(self) -> None:
        scheduler = AbsoluteActionScheduler(
            AbsoluteMission("one", (TailAbsoluteAction(80, 90, 90, 1, 1, 1),), (), ()), self.cal)
        scheduler.start(10, {"tail": {1: 80, 2: 90, 3: 90},
                             "left_fin": {4: 121, 5: 90}, "right_fin": {6: 143, 7: 90}})
        self.assertFalse(scheduler.all_finished())


if __name__ == "__main__":
    unittest.main()
