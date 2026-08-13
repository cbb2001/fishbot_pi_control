from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from control.runtime.discrete_actions import (
    DiscreteMission,
    FinAction,
    MissionValidationError,
    TailAction,
    build_robot_calibration,
    evaluate_fin_action,
    evaluate_tail_action,
    quintic_smoothstep,
    smooth_segment,
    tail_action_boundaries,
    validate_discrete_mission,
)
from control.runtime.manual_action_provider import ManualSequenceProvider
from control.safety import load_robot_config


class DiscreteActionMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_robot_config()
        cls.calibration = build_robot_calibration(cls.config)

    def test_quintic_endpoints_and_zero_endpoint_velocity(self) -> None:
        self.assertEqual(quintic_smoothstep(0.0), 0.0)
        self.assertEqual(quintic_smoothstep(1.0), 1.0)
        epsilon = 1e-6
        start_velocity = (quintic_smoothstep(epsilon) - quintic_smoothstep(0.0)) / epsilon
        end_velocity = (quintic_smoothstep(1.0) - quintic_smoothstep(1.0 - epsilon)) / epsilon
        self.assertLess(abs(start_velocity), 1e-8)
        self.assertLess(abs(end_velocity), 1e-8)

    def test_tail_positive_negative_and_configured_direction(self) -> None:
        starts = {servo_id: self.calibration.servos[servo_id].center_deg for servo_id in (1, 2, 3)}
        positive = evaluate_tail_action(TailAction(1.0, 10.0), 1.0, starts, self.calibration)
        negative = evaluate_tail_action(TailAction(1.0, -10.0), 1.0, starts, self.calibration)
        for servo_id in (1, 2, 3):
            self.assertGreater(positive[servo_id], starts[servo_id])
            self.assertLess(negative[servo_id], starts[servo_id])

        config = copy.deepcopy(self.config)
        servo_2 = next(item for item in config["servo"]["channels"] if item["servo_id"] == 2)
        original_left = servo_2["direction"]["left_reference_angle"]
        original_right = servo_2["direction"]["right_reference_angle"]
        servo_2["direction"].update(
            {
                "left_reference_angle": original_right,
                "right_reference_angle": original_left,
                "angle_increases_toward": "right",
                "angle_decreases_toward": "left",
            }
        )
        reversed_calibration = build_robot_calibration(config)
        reversed_target = evaluate_tail_action(
            TailAction(1.0, 10.0), 1.0, starts, reversed_calibration
        )
        self.assertEqual(reversed_target[2], starts[2] - 10.0)

    def test_tail_actions_accumulate_and_intermediate_violation_is_rejected(self) -> None:
        mission = DiscreteMission(
            "tail_accumulation",
            (TailAction(1.0, 10.0), TailAction(1.0, -5.0)),
            (),
            (),
        )
        validate_discrete_mission(mission, self.calibration)
        boundaries = tail_action_boundaries(mission, self.calibration)
        center = self.calibration.servos[1].center_deg
        self.assertEqual(boundaries[0][1][1], center + 10.0)
        self.assertEqual(boundaries[1][0][1], center + 10.0)
        self.assertEqual(boundaries[1][1][1], center + 5.0)

        returns_legal = DiscreteMission(
            "invalid_intermediate",
            (TailAction(1.0, 31.0), TailAction(1.0, -31.0)),
            (),
            (),
        )
        with self.assertRaises(MissionValidationError) as caught:
            validate_discrete_mission(returns_legal, self.calibration)
        self.assertIn("tail_actions[0]", str(caught.exception))

    def test_left_fin_b1_all_key_points(self) -> None:
        action = FinAction(8.0, 1, 0.5)
        root = self.calibration.servos[4]
        tip = self.calibration.servos[5]
        root_bottom = root.center_deg + 0.5 * (root.bottom_reference_deg - root.center_deg)
        root_top = root.center_deg + 0.5 * (root.top_reference_deg - root.center_deg)
        tip_bottom = tip.center_deg + 0.5 * (tip.bottom_reference_deg - tip.center_deg)
        expected = {
            0.0: (root.center_deg, tip.center_deg),
            2.0: (root_bottom, tip.center_deg),
            3.0: (smooth_segment(root_bottom, root_top, 0.25), tip_bottom),
            6.0: (root_top, tip_bottom),
            7.0: (smooth_segment(root_top, root.center_deg, 0.5), tip.center_deg),
            8.0: (root.center_deg, tip.center_deg),
        }
        for elapsed, (root_expected, tip_expected) in expected.items():
            values = evaluate_fin_action(action, elapsed, (4, 5), self.calibration)
            self.assertAlmostEqual(values[4], root_expected)
            self.assertAlmostEqual(values[5], tip_expected)

    def test_left_fin_b_minus_one_is_symmetric(self) -> None:
        plus = FinAction(8.0, 1, 0.5)
        minus = FinAction(8.0, -1, 0.5)
        root = self.calibration.servos[4]
        for elapsed in (0.0, 2.0, 3.0, 6.0, 7.0, 8.0):
            up = evaluate_fin_action(plus, elapsed, (4, 5), self.calibration)
            down = evaluate_fin_action(minus, elapsed, (4, 5), self.calibration)
            self.assertAlmostEqual(up[4] + down[4], 2.0 * root.center_deg)
        tip_at_three_eighths = evaluate_fin_action(minus, 3.0, (4, 5), self.calibration)
        expected_tip_top = self.calibration.servos[5].center_deg + 0.5 * (
            self.calibration.servos[5].top_reference_deg
            - self.calibration.servos[5].center_deg
        )
        self.assertEqual(tip_at_three_eighths[5], expected_tip_top)

    def test_right_fin_uses_right_physical_references(self) -> None:
        values = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.0, (6, 7), self.calibration)
        self.assertEqual(values[6], 90.0)
        self.assertEqual(values[7], 90.0)
        tip_at_three_eighths = evaluate_fin_action(
            FinAction(4.0, 1, 1.0), 1.5, (6, 7), self.calibration
        )
        self.assertEqual(tip_at_three_eighths[7], 180.0)
        reversed_tip = evaluate_fin_action(
            FinAction(4.0, -1, 1.0), 1.5, (6, 7), self.calibration
        )
        self.assertEqual(reversed_tip[7], 0.0)

    def test_r_zero_holds_center_and_r_one_reaches_calibration(self) -> None:
        for servo_ids in ((4, 5), (6, 7)):
            for b in (-1, 1):
                for elapsed in (0.0, 0.7, 2.0, 3.8, 4.0):
                    values = evaluate_fin_action(
                        FinAction(4.0, b, 0.0), elapsed, servo_ids, self.calibration
                    )
                    for servo_id in servo_ids:
                        self.assertEqual(values[servo_id], self.calibration.servos[servo_id].center_deg)
        left = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.0, (4, 5), self.calibration)
        self.assertEqual(left[4], self.calibration.servos[4].bottom_reference_deg)
        left_tip = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.5, (4, 5), self.calibration)
        self.assertEqual(left_tip[5], self.calibration.servos[5].bottom_reference_deg)

    def test_b1_matches_confirmed_servo_angle_directions(self) -> None:
        left_root = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.0, (4, 5), self.calibration)
        left_tip = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.5, (4, 5), self.calibration)
        right_root = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.0, (6, 7), self.calibration)
        right_tip = evaluate_fin_action(FinAction(4.0, 1, 1.0), 1.5, (6, 7), self.calibration)

        self.assertGreater(left_root[4], self.calibration.servos[4].center_deg)
        self.assertLess(left_tip[5], self.calibration.servos[5].center_deg)
        self.assertLess(right_root[6], self.calibration.servos[6].center_deg)
        self.assertGreater(right_tip[7], self.calibration.servos[7].center_deg)

    def test_only_robot_min_max_apply(self) -> None:
        config = copy.deepcopy(self.config)
        config["safety"]["servo"]["max_test_amplitude_deg"] = 0.0
        calibration = build_robot_calibration(config)
        tip = calibration.servos[5]
        safe_bottom_ratio = (
            (tip.center_deg - tip.min_deg)
            / (tip.center_deg - tip.bottom_reference_deg)
        )
        mission = DiscreteMission(
            "full_range",
            (),
            (FinAction(1.0, 1, safe_bottom_ratio),),
            (),
        )
        validate_discrete_mission(mission, calibration)


class ManualProviderTests(unittest.TestCase):
    def test_provider_allows_empty_individual_sequence(self) -> None:
        text = """name: tail_only
tail_actions:
  - {t: 1.0, A: 0.0}
left_fin_actions: []
right_fin_actions: []
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mission.yaml"
            path.write_text(text, encoding="utf-8")
            mission = ManualSequenceProvider(path).load()
        self.assertEqual(mission.name, "tail_only")
        self.assertEqual(len(mission.tail_actions), 1)

    def test_provider_rejects_bool_number_and_unknown_field(self) -> None:
        text = """name: bad
tail_actions:
  - {t: true, A: 0, frequency: 1}
left_fin_actions: []
right_fin_actions: []
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mission.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(MissionValidationError):
                ManualSequenceProvider(path).load()


if __name__ == "__main__":
    unittest.main()
