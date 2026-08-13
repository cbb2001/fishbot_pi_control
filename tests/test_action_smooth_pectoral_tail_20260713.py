from __future__ import annotations

import argparse
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_action_smooth_pectoral_tail_20260713 as action  # noqa: E402
from control.safety import load_robot_config  # noqa: E402


class SmoothActionMechanicalLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_robot_config()

    @staticmethod
    def motion_args(
        *,
        tail_left: float,
        tail_right: float,
        root_ratio: float = 0.5,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            root_amplitude_ratio=root_ratio,
            tip_amplitude_deg=20.0,
            tail_left_amplitude_deg=tail_left,
            tail_right_amplitude_deg=tail_right,
        )

    def test_cli_has_no_second_mechanical_margin(self) -> None:
        with mock.patch.object(sys, "argv", ["run_action_smooth_pectoral_tail_20260713.py"]):
            args = action.parse_args()
        self.assertFalse(hasattr(args, "mechanical_limit_margin_deg"))

    def test_targets_equal_to_configured_min_max_are_accepted(self) -> None:
        targets = action._prepare_targets(
            self.config,
            self.motion_args(tail_left=30.0, tail_right=30.0),
        )
        for servo_id in (1, 2, 3):
            raw = next(
                item for item in self.config["servo"]["channels"]
                if int(item["servo_id"]) == servo_id
            )
            self.assertEqual(
                targets["tail"][servo_id]["left_target"],
                float(raw["max_angle"]),
            )
            self.assertEqual(
                targets["tail"][servo_id]["right_target"],
                float(raw["min_angle"]),
            )

    def test_full_root_ratio_reaches_configured_physical_endpoints(self) -> None:
        config = copy.deepcopy(self.config)
        config["safety"]["servo"]["max_test_amplitude_deg"] = 0.001
        targets = action._prepare_targets(
            config,
            self.motion_args(
                tail_left=10.0,
                tail_right=10.0,
                root_ratio=1.0,
            ),
        )
        for servo_id in (4, 6):
            raw = next(
                item for item in config["servo"]["channels"]
                if int(item["servo_id"]) == servo_id
            )
            self.assertEqual(
                targets["roots"][servo_id]["bottom"],
                float(raw["direction"]["bottom_reference_angle"]),
            )
            self.assertEqual(
                targets["roots"][servo_id]["top"],
                float(raw["direction"]["top_reference_angle"]),
            )

    def test_target_outside_configured_limit_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            action._prepare_targets(
                self.config,
                self.motion_args(tail_left=30.1, tail_right=30.0),
            )
        message = str(caught.exception)
        self.assertIn("servo_id=1", message)
        servo_1 = next(
            item for item in self.config["servo"]["channels"]
            if int(item["servo_id"]) == 1
        )
        self.assertIn(
            f"target={float(servo_1['max_angle']) + 0.1:.3f}",
            message,
        )
        self.assertIn("config/robot.yaml", message)
        self.assertIn(
            f"{float(servo_1['min_angle']):.3f}.."
            f"{float(servo_1['max_angle']):.3f}",
            message,
        )

    def test_configured_center_outside_min_max_is_rejected_before_motion(self) -> None:
        config = copy.deepcopy(self.config)
        servo_7 = next(
            item
            for item in config["servo"]["channels"]
            if int(item["servo_id"]) == 7
        )
        servo_7["center_angle"] = 181.0
        with self.assertRaises(SystemExit) as caught:
            action._prepare_targets(
                config,
                self.motion_args(tail_left=10.0, tail_right=10.0),
            )
        message = str(caught.exception)
        self.assertIn("servo_id=7", message)
        self.assertIn("center_angle=181.000", message)
        self.assertIn(
            f"{float(servo_7['min_angle']):.3f}.."
            f"{float(servo_7['max_angle']):.3f}",
            message,
        )


if __name__ == "__main__":
    unittest.main()
