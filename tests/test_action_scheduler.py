from __future__ import annotations

import unittest

from control.runtime.action_scheduler import ActionScheduler
from control.runtime.discrete_actions import DiscreteMission, FinAction, TailAction


class ActionSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mission = DiscreteMission(
            "scheduler",
            (TailAction(1.0, 5.0), TailAction(2.0, -5.0)),
            (FinAction(3.0, 1, 0.5),),
            (FinAction(4.0, -1, 0.5),),
        )
        self.scheduler = ActionScheduler(self.mission)
        self.start_ns = 1_000_000_000
        self.starts = {
            "tail": {1: 80.0, 2: 90.0, 3: 90.0},
            "left_fin": {4: 121.0, 5: 90.0},
            "right_fin": {6: 143.0, 7: 90.0},
        }

    def test_three_sequences_start_together_and_advance_independently(self) -> None:
        transitions = self.scheduler.start(self.start_ns, self.starts)
        self.assertEqual(len(transitions), 3)
        self.assertEqual(
            {transition.started.actual_start_t_ns for transition in transitions},
            {self.start_ns},
        )
        completion_ns = self.start_ns + 1_000_000_000 + 123
        transition = self.scheduler.complete(
            "tail", completion_ns, {1: 85.0, 2: 95.0, 3: 95.0}
        )
        self.assertEqual(transition.started.actual_start_t_ns, completion_ns)
        self.assertEqual(transition.started.planned_start_t_ns, self.start_ns + 1_000_000_000)
        self.assertEqual(self.scheduler.active("left_fin").action_index, 0)
        self.assertEqual(self.scheduler.active("right_fin").action_index, 0)

    def test_endpoint_cannot_complete_early(self) -> None:
        self.scheduler.start(self.start_ns, self.starts)
        with self.assertRaises(RuntimeError):
            self.scheduler.complete(
                "tail", self.start_ns + 999_999_999, {1: 85.0, 2: 95.0, 3: 95.0}
            )

    def test_mission_completion_is_maximum_actual_sequence_completion(self) -> None:
        self.scheduler.start(self.start_ns, self.starts)
        tail_first_end = self.start_ns + 1_000_000_010
        self.scheduler.complete("tail", tail_first_end, {1: 85.0, 2: 95.0, 3: 95.0})
        self.scheduler.complete(
            "tail", tail_first_end + 2_000_000_010, {1: 80.0, 2: 90.0, 3: 90.0}
        )
        self.scheduler.complete(
            "left_fin", self.start_ns + 3_000_000_100, {4: 121.0, 5: 90.0}
        )
        right_end = self.start_ns + 4_000_000_500
        self.scheduler.complete("right_fin", right_end, {6: 143.0, 7: 90.0})
        self.assertTrue(self.scheduler.all_finished())
        self.assertEqual(self.scheduler.mission_completion_t_ns(), right_end)


if __name__ == "__main__":
    unittest.main()
