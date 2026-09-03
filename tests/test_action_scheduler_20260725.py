from __future__ import annotations

import unittest

from control.runtime.action_scheduler_20260725 import ActionScheduler
from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    FinAction,
    TailAction,
    build_calibration,
    validate_mission,
)
from control.safety import load_robot_config


class ActionScheduler20260725Tests(unittest.TestCase):
    """验证三路同起点、路内串行和 previous theta 事务提交。"""

    def setUp(self) -> None:
        self.calibration = build_calibration(load_robot_config())
        self.mission = DiscreteAbsoluteMission(
            "parallel",
            (TailAction(10.0, 1.0), TailAction(10.0, 0.75)),
            (FinAction(150.0, 2.0, 1, 1), FinAction(100.0, 0.5, 0, -1)),
            (FinAction(175.0, 0.5, 1, -1), FinAction(120.0, 1.25, 1, 1)),
        )
        validate_mission(self.mission, self.calibration)

    def _scheduler(self) -> ActionScheduler:
        return ActionScheduler(self.mission, self.calibration)

    def test_first_actions_share_one_mission_start_and_different_end_times(self) -> None:
        scheduler = self._scheduler()
        transitions = scheduler.start(1_000_000_000)
        starts = [transition.started for transition in transitions]
        self.assertEqual(
            {action.actual_start_t_ns for action in starts if action is not None},
            {1_000_000_000},
        )
        self.assertEqual(scheduler.active("tail").planned_end_t_ns, 2_000_000_000)
        self.assertEqual(
            scheduler.active("left_fin").planned_end_t_ns,
            3_000_000_000,
        )
        self.assertEqual(
            scheduler.active("right_fin").planned_end_t_ns,
            1_500_000_000,
        )

    def test_hold_action_is_not_due_before_full_t(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(10)
        tail = scheduler.active("tail")
        final = tail.trajectory.target_angles_deg
        transition = scheduler.complete(
            "tail",
            tail.planned_end_t_ns,
            final,
            endpoint_written=True,
        )
        hold = transition.started
        self.assertEqual(hold.previous_theta, hold.current_theta)
        self.assertFalse(scheduler.sequences["tail"].is_due(hold.planned_end_t_ns - 1))
        self.assertTrue(scheduler.sequences["tail"].is_due(hold.planned_end_t_ns))

    def test_failed_or_early_endpoint_does_not_advance_or_commit(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(100)
        right = scheduler.active("right_fin")
        initial_previous = scheduler.previous_thetas["right_fin"]
        with self.assertRaises(RuntimeError):
            scheduler.complete(
                "right_fin",
                right.planned_end_t_ns,
                right.trajectory.target_angles_deg,
                endpoint_written=False,
            )
        self.assertIs(scheduler.active("right_fin"), right)
        self.assertEqual(scheduler.previous_thetas["right_fin"], initial_previous)
        with self.assertRaises(RuntimeError):
            scheduler.complete(
                "right_fin",
                right.planned_end_t_ns - 1,
                right.trajectory.target_angles_deg,
                endpoint_written=True,
            )
        self.assertEqual(scheduler.previous_thetas["right_fin"], initial_previous)

    def test_successful_completion_commits_previous_and_starts_next_at_completion(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(1_000)
        right = scheduler.active("right_fin")
        completion_t_ns = right.planned_end_t_ns + 321
        left_before = scheduler.active("left_fin")
        left_progress_before = left_before.progress_at(completion_t_ns)
        left_index_before = scheduler.sequences["left_fin"].current_action_index
        transition = scheduler.complete(
            "right_fin",
            completion_t_ns,
            right.trajectory.target_angles_deg,
            endpoint_written=True,
        )
        self.assertTrue(transition.finished.completed)
        self.assertTrue(transition.finished.endpoint_written)
        self.assertEqual(
            transition.finished.previous_theta_before_commit,
            143.0,
        )
        self.assertEqual(
            transition.finished.previous_theta_after_commit,
            175.0,
        )
        self.assertEqual(scheduler.previous_thetas["right_fin"], 175.0)
        self.assertEqual(transition.started.actual_start_t_ns, completion_t_ns)
        self.assertEqual(transition.started.previous_theta, 175.0)
        self.assertTrue(transition.started.start_command_written)
        # 右侧切换不能改变尚未到期的左侧动作对象和绝对时间。
        self.assertIs(scheduler.active("left_fin"), left_before)
        self.assertEqual(left_before.actual_start_t_ns, 1_000)
        self.assertEqual(
            scheduler.sequences["left_fin"].current_action_index,
            left_index_before,
        )
        self.assertEqual(
            scheduler.active("left_fin").progress_at(completion_t_ns),
            left_progress_before,
        )

    def test_action2_successful_completion_commits_previous_theta(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(5_000)
        left = scheduler.active("left_fin")
        self.assertEqual(scheduler.previous_thetas["left_fin"], 111.0)

        transition = scheduler.complete(
            "left_fin",
            left.planned_end_t_ns,
            left.trajectory.target_angles_deg,
            endpoint_written=True,
        )

        self.assertTrue(transition.finished.completed)
        self.assertTrue(transition.finished.endpoint_written)
        self.assertEqual(
            transition.finished.previous_theta_before_commit,
            111.0,
        )
        self.assertEqual(
            transition.finished.previous_theta_after_commit,
            150.0,
        )
        self.assertEqual(scheduler.previous_thetas["left_fin"], 150.0)
        self.assertEqual(
            transition.started.actual_start_t_ns,
            transition.finished.completion_t_ns,
        )
        self.assertEqual(transition.started.previous_theta, 150.0)

    def test_incomplete_endpoint_mapping_is_rejected_without_commit(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(0)
        tail = scheduler.active("tail")
        with self.assertRaises(RuntimeError):
            scheduler.complete(
                "tail",
                tail.planned_end_t_ns,
                {1: 95.0, 2: 105.0},
                endpoint_written=True,
            )
        self.assertEqual(scheduler.previous_thetas["tail"], 0.0)
        self.assertIs(scheduler.active("tail"), tail)

    def test_all_three_sequences_must_finish_before_mission_finishes(self) -> None:
        scheduler = self._scheduler()
        scheduler.start(0)
        while scheduler.active("right_fin") is not None:
            action = scheduler.active("right_fin")
            scheduler.complete(
                "right_fin",
                action.planned_end_t_ns,
                action.trajectory.target_angles_deg,
                endpoint_written=True,
            )
        self.assertFalse(scheduler.all_finished())
        while scheduler.active("tail") is not None:
            action = scheduler.active("tail")
            scheduler.complete(
                "tail",
                action.planned_end_t_ns,
                action.trajectory.target_angles_deg,
                endpoint_written=True,
            )
        self.assertFalse(scheduler.all_finished())
        while scheduler.active("left_fin") is not None:
            action = scheduler.active("left_fin")
            scheduler.complete(
                "left_fin",
                action.planned_end_t_ns,
                action.trajectory.target_angles_deg,
                endpoint_written=True,
            )
        self.assertTrue(scheduler.all_finished())
        self.assertEqual(
            scheduler.mission_completion_t_ns(),
            scheduler.sequences["left_fin"].completion_t_ns,
        )

    def test_empty_sequences_are_finished_at_common_start(self) -> None:
        mission = DiscreteAbsoluteMission(
            "tail_only",
            (TailAction(0.0, 1.0),),
            (),
            (),
        )
        scheduler = ActionScheduler(mission, self.calibration)
        scheduler.start(123)
        self.assertTrue(scheduler.sequences["left_fin"].sequence_finished)
        self.assertEqual(scheduler.sequences["left_fin"].completion_t_ns, 123)
        self.assertFalse(scheduler.all_finished())


if __name__ == "__main__":
    unittest.main()
