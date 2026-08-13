from __future__ import annotations

import math
import copy
import unittest

from control.runtime.absolute_discrete_actions import (
    AbsoluteMission, FinAbsoluteAction, MissionValidationError, TailAbsoluteAction,
    build_absolute_calibration, build_trajectory, quintic_smoothstep,
    validate_absolute_mission,
)
from control.safety import load_robot_config


class AbsoluteDiscreteActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = build_absolute_calibration(load_robot_config())

    def test_quintic_endpoints_and_endpoint_derivatives(self) -> None:
        self.assertEqual(quintic_smoothstep(0), 0)
        self.assertEqual(quintic_smoothstep(1), 1)
        h = 1e-5
        self.assertAlmostEqual((quintic_smoothstep(h) - quintic_smoothstep(0)) / h, 0, places=7)
        self.assertAlmostEqual((quintic_smoothstep(1) - quintic_smoothstep(1-h)) / h, 0, places=7)

    def test_tail_uses_absolute_targets_and_independent_durations(self) -> None:
        action = TailAbsoluteAction(90, 110, 100, 20, 10, 5)
        trajectory = build_trajectory("tail", action, {1: 80, 2: 90, 3: 90}, self.calibration)
        self.assertEqual(trajectory.target_angles_deg, {1: 90, 2: 110, 3: 100})
        self.assertEqual(trajectory.servo_durations_s, {1: 0.5, 2: 2.0, 3: 2.0})
        self.assertEqual(trajectory.duration_s, 2.0)
        at_one_second = trajectory.evaluate(1.0)
        self.assertEqual(at_one_second[1], 90)
        self.assertAlmostEqual(at_one_second[2], 100)

    def test_zero_delta_is_immediate(self) -> None:
        tail = build_trajectory("tail", TailAbsoluteAction(80, 90, 90, 1, 1, 1),
                                {1: 80, 2: 90, 3: 90}, self.calibration)
        self.assertEqual(tail.duration_s, 0)
        fin = build_trajectory("left_fin", FinAbsoluteAction(121, 1, 1),
                               {4: 121, 5: 90}, self.calibration)
        self.assertEqual(fin.duration_s, 0)
        self.assertEqual(
            fin.evaluate(0),
            {4: 121, 5: self.calibration.servos[5].center_deg},
        )

    def test_left_fin_coupling_sign_and_segments(self) -> None:
        up = build_trajectory("left_fin", FinAbsoluteAction(174, 53, 1),
                              {4: 121, 5: 90}, self.calibration)
        center = self.calibration.servos[5].center_deg
        span = self.calibration.left_coupling.tip_reference_span_deg
        self.assertAlmostEqual(up.peak_angle_deg, center + 53 / 106 * span)
        self.assertEqual(up.duration_s, 1)
        self.assertEqual(up.evaluate(0)[5], center)
        self.assertAlmostEqual(up.evaluate(.25)[5], center + 53 / 106 * span)
        self.assertAlmostEqual(up.evaluate(.75)[5], center + 53 / 106 * span)
        self.assertAlmostEqual(up.evaluate(1)[5], center)
        down = build_trajectory("left_fin", FinAbsoluteAction(68, 53, 1),
                                {4: 121, 5: 90}, self.calibration)
        self.assertAlmostEqual(down.peak_angle_deg, center - 53 / 106 * span)

    def test_b_zero_holds_tip_center(self) -> None:
        trajectory = build_trajectory("right_fin", FinAbsoluteAction(170, 20, 0),
                                      {6: 143, 7: 90}, self.calibration)
        for elapsed in (0, trajectory.duration_s / 4, trajectory.duration_s):
            self.assertEqual(trajectory.evaluate(elapsed)[7], 90)

    def test_speed_limits_and_nonpositive_speed_rejected(self) -> None:
        for speed in (
            self.calibration.tail_max_speed_deg_s + 1,
            0,
            -1,
            math.inf,
        ):
            mission = AbsoluteMission("bad", (TailAbsoluteAction(80, 90, 90, speed, 1, 1),), (), ())
            with self.assertRaises(MissionValidationError):
                validate_absolute_mission(mission, self.calibration)
        mission = AbsoluteMission(
            "bad",
            (),
            (
                FinAbsoluteAction(
                    121,
                    self.calibration.fin_max_speed_deg_s + 1,
                    0,
                ),
            ),
            (),
        )
        with self.assertRaises(MissionValidationError):
            validate_absolute_mission(mission, self.calibration)

    def test_tip_peak_out_of_range_rejects_whole_mission(self) -> None:
        # 收紧配置中的 5 号机械上限；映射参数本身绝不能替代或修改该限位。
        config = copy.deepcopy(load_robot_config())
        next(raw for raw in config["servo"]["channels"] if raw["servo_id"] == 5)["max_angle"] = 170
        calibration = build_absolute_calibration(config)
        mission = AbsoluteMission("bad", (),
            (FinAbsoluteAction(68, 20, 0), FinAbsoluteAction(174, 20, 1)), ())
        with self.assertRaisesRegex(MissionValidationError, "5号舵机目标角"):
            validate_absolute_mission(mission, calibration)

    def test_right_formula_is_not_mirrored(self) -> None:
        trajectory = build_trajectory("right_fin", FinAbsoluteAction(196, 53, 1),
                                      {6: 143, 7: 90}, self.calibration)
        self.assertAlmostEqual(trajectory.peak_angle_deg, 135)

    def test_right_tip_peak_out_of_range_rejects_whole_mission(self) -> None:
        config = copy.deepcopy(load_robot_config())
        next(raw for raw in config["servo"]["channels"] if raw["servo_id"] == 7)["max_angle"] = 170
        calibration = build_absolute_calibration(config)
        mission = AbsoluteMission("bad", (), (),
            (FinAbsoluteAction(90, 20, 0), FinAbsoluteAction(196, 20, 1)))
        with self.assertRaisesRegex(MissionValidationError, "7号舵机目标角"):
            validate_absolute_mission(mission, calibration)


if __name__ == "__main__":
    unittest.main()
