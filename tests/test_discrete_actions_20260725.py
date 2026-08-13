from __future__ import annotations

import copy
import math
import unittest

from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    FinAction,
    MissionValidationError,
    TailAction,
    build_calibration,
    build_trajectory,
    quintic_smoothstep,
    smooth_motion,
    validate_mission,
)
from control.safety import load_robot_config


class DiscreteActions20260725Tests(unittest.TestCase):
    """覆盖新动作定义的边界、三分支数学和完整任务预验证。"""

    def setUp(self) -> None:
        self.config = load_robot_config()
        self.calibration = build_calibration(self.config)

    def test_default_reference_values_are_exact(self) -> None:
        self.assertEqual(
            self.calibration.initial_angles_deg,
            {1: 85.0, 2: 95.0, 3: 95.0, 4: 121.0, 5: 94.0, 6: 143.0, 7: 90.0},
        )
        self.assertEqual(self.calibration.initial_previous_thetas["tail"], 0.0)
        self.assertEqual(self.calibration.initial_previous_thetas["left_fin"], 121.0)
        self.assertEqual(self.calibration.initial_previous_thetas["right_fin"], 143.0)
        self.assertEqual(self.calibration.tail.theta_min_deg, -30.0)
        self.assertEqual(self.calibration.tail.theta_max_deg, 30.0)
        self.assertEqual(self.calibration.left_fin.root_servo_id, 4)
        self.assertEqual(self.calibration.left_fin.tip_servo_id, 5)
        self.assertEqual(self.calibration.left_fin.root_reference_span_deg, 106.0)
        self.assertEqual(self.calibration.left_fin.tip_reference_span_deg, 86.0)
        self.assertEqual(self.calibration.right_fin.root_servo_id, 6)
        self.assertEqual(self.calibration.right_fin.tip_servo_id, 7)
        self.assertEqual(self.calibration.right_fin.root_reference_span_deg, 106.0)
        self.assertEqual(self.calibration.right_fin.tip_reference_span_deg, 86.0)

    def test_reference_center_outside_mechanical_range_is_rejected_at_startup(self) -> None:
        config = copy.deepcopy(self.config)
        servo6 = next(
            raw for raw in config["servo"]["channels"] if raw["servo_id"] == 6
        )
        servo6["center_angle"] = 197.0
        with self.assertRaisesRegex(
            MissionValidationError,
            r"servo_id=6.*配置中位=197.*允许范围90.*196",
        ):
            build_calibration(config)

    def test_quintic_has_exact_boundaries_and_zero_endpoint_derivatives(self) -> None:
        self.assertEqual(quintic_smoothstep(-1.0), 0.0)
        self.assertEqual(quintic_smoothstep(0.0), 0.0)
        self.assertEqual(quintic_smoothstep(1.0), 1.0)
        self.assertEqual(quintic_smoothstep(2.0), 1.0)
        h = 1e-5
        first_at_zero = (quintic_smoothstep(h) - quintic_smoothstep(0.0)) / h
        first_at_one = (quintic_smoothstep(1.0) - quintic_smoothstep(1.0 - h)) / h
        self.assertAlmostEqual(first_at_zero, 0.0, places=7)
        self.assertAlmostEqual(first_at_one, 0.0, places=7)

    def test_smooth_motion_returns_exact_start_end_and_hold(self) -> None:
        self.assertEqual(smooth_motion(85.0, 115.0, 0.0, 2.0), 85.0)
        self.assertEqual(smooth_motion(85.0, 115.0, 2.0, 2.0), 115.0)
        self.assertEqual(smooth_motion(85.0, 115.0, 3.0, 2.0), 115.0)
        self.assertEqual(smooth_motion(85.0, 85.0, 0.7, 2.0), 85.0)

    def test_smoothstep_near_one_never_overshoots_mechanical_maximum(self) -> None:
        u = 0.999999
        self.assertLessEqual(quintic_smoothstep(u), 1.0)

        trajectory = build_trajectory(
            "left_fin",
            FinAction(174.0, 1.0, 0, 1),
            121.0,
            self.calibration,
        )
        root_angle = trajectory.evaluate(u)[4]
        mechanical_max = self.calibration.servos[4].allowed_max_deg
        self.assertLessEqual(root_angle, mechanical_max)
        self.calibration.servos[4].validate(root_angle, "near-end root angle")

    def test_action1_theta_boundaries_targets_and_full_duration(self) -> None:
        for theta in (-30.0, 30.0):
            mission = DiscreteAbsoluteMission(
                "tail_boundary",
                (TailAction(theta, 2.0),),
                (),
                (),
            )
            validate_mission(mission, self.calibration)
        trajectory = build_trajectory(
            "tail",
            TailAction(30.0, 2.0),
            0.0,
            self.calibration,
        )
        self.assertEqual(
            trajectory.target_angles_deg,
            {1: 115.0, 2: 125.0, 3: 125.0},
        )
        self.assertEqual(trajectory.evaluate(0.0), {1: 85.0, 2: 95.0, 3: 95.0})
        quarter = trajectory.evaluate(0.5)
        quarter_factor = quintic_smoothstep(0.25)
        self.assertEqual(
            quarter,
            {
                1: 85.0 + 30.0 * quarter_factor,
                2: 95.0 + 30.0 * quarter_factor,
                3: 95.0 + 30.0 * quarter_factor,
            },
        )
        self.assertNotAlmostEqual(quarter[1], 92.5)
        self.assertEqual(trajectory.evaluate(1.0), {1: 100.0, 2: 110.0, 3: 110.0})
        self.assertEqual(trajectory.evaluate(2.0), trajectory.target_angles_deg)

    def test_action1_rejects_out_of_range_nonpositive_and_nonfinite(self) -> None:
        for action in (
            TailAction(-30.001, 1.0),
            TailAction(30.001, 1.0),
            TailAction(0.0, 0.0),
            TailAction(0.0, -1.0),
            TailAction(0.0, math.inf),
        ):
            mission = DiscreteAbsoluteMission("bad", (action,), (), ())
            with self.assertRaises(MissionValidationError):
                validate_mission(mission, self.calibration)

    def test_action1_equal_theta_is_exact_hold_not_zero_duration(self) -> None:
        trajectory = build_trajectory(
            "tail",
            TailAction(15.0, 1.25),
            15.0,
            self.calibration,
        )
        self.assertEqual(trajectory.duration_s, 1.25)
        expected = {1: 100.0, 2: 110.0, 3: 110.0}
        for elapsed in (0.0, 0.5, 1.249999, 1.25, 2.0):
            self.assertEqual(trajectory.evaluate(elapsed), expected)

    def test_action2_increasing_and_decreasing_use_abs_delta_peak(self) -> None:
        increasing = build_trajectory(
            "left_fin",
            FinAction(174.0, 2.0, 1, 1),
            121.0,
            self.calibration,
        )
        decreasing = build_trajectory(
            "left_fin",
            FinAction(68.0, 2.0, 1, -1),
            121.0,
            self.calibration,
        )
        expected_positive_peak = 94.0 + 53.0 / 106.0 * 86.0
        expected_negative_peak = 94.0 - 53.0 / 106.0 * 86.0
        self.assertAlmostEqual(increasing.tip_peak_angle_deg, expected_positive_peak)
        self.assertAlmostEqual(decreasing.tip_peak_angle_deg, expected_negative_peak)
        self.assertEqual(increasing.evaluate(0.0), {4: 121.0, 5: 94.0})
        self.assertAlmostEqual(increasing.evaluate(1.0)[4], 147.5)
        self.assertAlmostEqual(increasing.evaluate(1.0)[5], expected_positive_peak)
        self.assertEqual(increasing.evaluate(2.0), {4: 174.0, 5: 94.0})
        self.assertEqual(decreasing.evaluate(0.0), {4: 121.0, 5: 94.0})
        self.assertAlmostEqual(decreasing.evaluate(1.0)[4], 94.5)
        self.assertAlmostEqual(decreasing.evaluate(1.0)[5], expected_negative_peak)
        self.assertEqual(decreasing.evaluate(2.0), {4: 68.0, 5: 94.0})

    def test_action2_b1_zero_and_equal_theta_hold_tip_exactly(self) -> None:
        no_tip = build_trajectory(
            "left_fin",
            FinAction(150.0, 2.0, 0, -1),
            121.0,
            self.calibration,
        )
        for elapsed in (0.0, 0.5, 1.0, 1.999, 2.0):
            self.assertEqual(no_tip.evaluate(elapsed)[5], 94.0)

        equal = build_trajectory(
            "left_fin",
            FinAction(121.0, 2.0, 1, -1),
            121.0,
            self.calibration,
        )
        for elapsed in (0.0, 0.5, 1.0, 1.999, 2.0):
            self.assertEqual(equal.evaluate(elapsed), {4: 121.0, 5: 94.0})

    def test_action3_formula_is_not_mirrored(self) -> None:
        positive = build_trajectory(
            "right_fin",
            FinAction(196.0, 2.0, 1, 1),
            143.0,
            self.calibration,
        )
        negative = build_trajectory(
            "right_fin",
            FinAction(90.0, 2.0, 1, -1),
            143.0,
            self.calibration,
        )
        self.assertAlmostEqual(positive.tip_peak_angle_deg, 90.0 + 43.0)
        self.assertAlmostEqual(negative.tip_peak_angle_deg, 90.0 - 43.0)
        self.assertEqual(positive.evaluate(0.0), {6: 143.0, 7: 90.0})
        self.assertAlmostEqual(positive.evaluate(1.0)[6], 169.5)
        self.assertAlmostEqual(positive.evaluate(1.0)[7], 133.0)
        self.assertEqual(negative.evaluate(0.0), {6: 143.0, 7: 90.0})
        self.assertAlmostEqual(negative.evaluate(1.0)[6], 116.5)
        self.assertAlmostEqual(negative.evaluate(1.0)[7], 47.0)
        self.assertEqual(positive.evaluate(2.0), {6: 196.0, 7: 90.0})
        self.assertEqual(negative.evaluate(2.0), {6: 90.0, 7: 90.0})

    def test_action3_root_limits_equal_hold_and_b1_zero(self) -> None:
        for theta in (90.0, 196.0):
            mission = DiscreteAbsoluteMission(
                "right_root_boundary",
                (),
                (),
                (FinAction(theta, 1.0, 0, 1),),
            )
            validate_mission(mission, self.calibration)

        for theta in (89.999, 196.001):
            mission = DiscreteAbsoluteMission(
                "bad_right_root",
                (),
                (),
                (FinAction(theta, 1.0, 0, 1),),
            )
            with self.assertRaises(MissionValidationError):
                validate_mission(mission, self.calibration)

        equal = build_trajectory(
            "right_fin",
            FinAction(143.0, 2.0, 1, -1),
            143.0,
            self.calibration,
        )
        self.assertEqual(equal.duration_s, 2.0)
        for elapsed in (0.0, 0.5, 1.0, 1.999999, 2.0):
            self.assertEqual(equal.evaluate(elapsed), {6: 143.0, 7: 90.0})

        no_tip = build_trajectory(
            "right_fin",
            FinAction(196.0, 2.0, 0, -1),
            143.0,
            self.calibration,
        )
        for elapsed in (0.0, 0.5, 1.0, 1.999999, 2.0):
            self.assertEqual(no_tip.evaluate(elapsed)[7], 90.0)

    def test_fin_flags_duration_and_root_limits_are_strict(self) -> None:
        for theta in (68.0, 174.0):
            mission = DiscreteAbsoluteMission(
                "left_root_boundary",
                (),
                (FinAction(theta, 1.0, 0, 1),),
                (),
            )
            validate_mission(mission, self.calibration)

        bad_actions = (
            FinAction(121.0, 0.0, 0, 1),
            FinAction(121.0, -1.0, 0, 1),
            FinAction(121.0, 1.0, 2, 1),
            FinAction(121.0, 1.0, True, 1),
            FinAction(121.0, 1.0, 0, 0),
            FinAction(121.0, 1.0, 0, True),
            FinAction(67.999, 1.0, 0, 1),
            FinAction(174.001, 1.0, 0, 1),
        )
        for action in bad_actions:
            mission = DiscreteAbsoluteMission("bad", (), (action,), ())
            with self.assertRaises(MissionValidationError):
                validate_mission(mission, self.calibration)

    def test_fin_flags_reject_float_equivalents(self) -> None:
        for action in (
            FinAction(121.0, 1.0, 1.0, 1),
            FinAction(121.0, 1.0, 0, -1.0),
        ):
            mission = DiscreteAbsoluteMission("float_flag", (), (action,), ())
            with self.assertRaises(MissionValidationError):
                validate_mission(mission, self.calibration)

    def test_tip_peak_out_of_range_rejects_entire_mission_with_context(self) -> None:
        config = copy.deepcopy(self.config)
        servo5 = next(
            raw for raw in config["servo"]["channels"] if raw["servo_id"] == 5
        )
        servo5["max_angle"] = 100.0
        calibration = build_calibration(config)
        mission = DiscreteAbsoluteMission(
            "bad_peak",
            (),
            (FinAction(174.0, 2.0, 1, 1),),
            (),
        )
        with self.assertRaisesRegex(
            MissionValidationError,
            r"动作组=left_fin.*动作索引=0.*previous theta=121.*current theta=174",
        ):
            validate_mission(mission, calibration)

    def test_right_tip_peak_out_of_range_rejects_entire_mission(self) -> None:
        config = copy.deepcopy(self.config)
        servo7 = next(
            raw for raw in config["servo"]["channels"] if raw["servo_id"] == 7
        )
        servo7["max_angle"] = 100.0
        calibration = build_calibration(config)
        mission = DiscreteAbsoluteMission(
            "bad_right_peak",
            (),
            (),
            (FinAction(196.0, 2.0, 1, 1),),
        )
        with self.assertRaisesRegex(
            MissionValidationError,
            r"动作组=right_fin.*动作索引=0.*previous theta=143.*current theta=196",
        ):
            validate_mission(mission, calibration)

    def test_tail_absolute_targets_still_obey_each_servo_limit(self) -> None:
        config = copy.deepcopy(self.config)
        servo2 = next(
            raw for raw in config["servo"]["channels"] if raw["servo_id"] == 2
        )
        servo2["max_angle"] = 120.0
        calibration = build_calibration(config)
        mission = DiscreteAbsoluteMission(
            "tail_mechanical_limit",
            (TailAction(30.0, 1.0),),
            (),
            (),
        )
        with self.assertRaisesRegex(MissionValidationError, "2号舵机目标角=125"):
            validate_mission(mission, calibration)


if __name__ == "__main__":
    unittest.main()
