from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_action_roll_20260713 as action  # noqa: E402
from control.runtime.roll_motion import (  # noqa: E402
    RootPairTargets,
    TailServoMotion,
    build_roll_root_targets,
    evaluate_roll_trajectory,
)
from control.safety import load_robot_config  # noqa: E402


class RollTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_robot_config()

    def make_args(self, *argv: str):
        return action.parse_args(self.config, list(argv))

    def make_targets(self, *argv: str):
        args = self.make_args(*argv)
        action._validate_scalar_args(args)
        return args, action._prepare_targets(self.config, args)

    @staticmethod
    def sample(elapsed_s, args, targets):
        return evaluate_roll_trajectory(
            elapsed_s,
            pectoral_frequency_hz=args.pectoral_frequency_hz,
            roots=targets["roots"],
            tip_centers_deg=targets["tip_centers_deg"],
            tip_amplitude_deg=args.tip_amplitude_deg,
            tip_transition_s=args.tip_transition_s,
            tip_direction=args.tip_direction,
            tail_frequency_hz=args.tail_frequency_hz,
            tail_left_amplitude_deg=args.tail_left_amplitude_deg,
            tail_right_amplitude_deg=args.tail_right_amplitude_deg,
            tail_first_direction=args.tail_first_direction,
            tail_servos=targets["tail_motion"],
        )


class RollTrajectoryTests(RollTestCase):
    def test_right_roll_required_endpoints(self) -> None:
        args, targets = self.make_targets(
            "--roll-direction", "right",
            "--root-amplitude-ratio", "0.1",
        )
        period = 1.0 / args.pectoral_frequency_hz
        start = self.sample(0.0, args, targets).commands_by_servo_id_deg
        half = self.sample(period / 2.0, args, targets).commands_by_servo_id_deg
        end = self.sample(period, args, targets).commands_by_servo_id_deg

        endpoints = targets["physical_endpoints"]
        self.assertAlmostEqual(start[4], endpoints["servo_4_bottom_deg"])
        self.assertAlmostEqual(start[6], endpoints["servo_6_top_deg"])
        self.assertAlmostEqual(
            half[4],
            endpoints["servo_4_bottom_deg"]
            + 0.1 * (endpoints["servo_4_top_deg"] - endpoints["servo_4_bottom_deg"]),
        )
        self.assertAlmostEqual(
            half[6],
            endpoints["servo_6_top_deg"]
            + 0.1 * (endpoints["servo_6_bottom_deg"] - endpoints["servo_6_top_deg"]),
        )
        self.assertAlmostEqual(end[4], endpoints["servo_4_bottom_deg"])
        self.assertAlmostEqual(end[6], endpoints["servo_6_top_deg"])
        self.assertAlmostEqual(start[5], targets["tips"][5]["center"])
        self.assertAlmostEqual(start[7], targets["tips"][7]["center"])
        self.assertAlmostEqual(half[5], targets["tips"][5]["center"])
        self.assertAlmostEqual(half[7], targets["tips"][7]["center"])

    def test_left_roll_required_endpoints(self) -> None:
        args, targets = self.make_targets(
            "--roll-direction", "left",
            "--root-amplitude-ratio", "0.1",
        )
        period = 1.0 / args.pectoral_frequency_hz
        start = self.sample(0.0, args, targets).commands_by_servo_id_deg
        half = self.sample(period / 2.0, args, targets).commands_by_servo_id_deg
        end = self.sample(period, args, targets).commands_by_servo_id_deg

        endpoints = targets["physical_endpoints"]
        self.assertAlmostEqual(start[4], endpoints["servo_4_top_deg"])
        self.assertAlmostEqual(start[6], endpoints["servo_6_bottom_deg"])
        self.assertAlmostEqual(
            half[4],
            endpoints["servo_4_top_deg"]
            + 0.1 * (endpoints["servo_4_bottom_deg"] - endpoints["servo_4_top_deg"]),
        )
        self.assertAlmostEqual(
            half[6],
            endpoints["servo_6_bottom_deg"]
            + 0.1 * (endpoints["servo_6_top_deg"] - endpoints["servo_6_bottom_deg"]),
        )
        self.assertAlmostEqual(end[4], endpoints["servo_4_top_deg"])
        self.assertAlmostEqual(end[6], endpoints["servo_6_bottom_deg"])

    def test_root_servos_use_same_ratio_and_opposed_physical_height(self) -> None:
        for direction in ("right", "left"):
            args, targets = self.make_targets(
                "--roll-direction", direction,
                "--root-amplitude-ratio", "0.1",
            )
            roots = targets["roots"]
            endpoints = targets["physical_endpoints"]
            self.assertAlmostEqual(
                abs(roots.servo_4_target_deg - roots.servo_4_start_deg),
                0.1 * abs(endpoints["servo_4_bottom_deg"] - endpoints["servo_4_top_deg"]),
            )
            self.assertAlmostEqual(
                abs(roots.servo_6_target_deg - roots.servo_6_start_deg),
                0.1 * abs(endpoints["servo_6_top_deg"] - endpoints["servo_6_bottom_deg"]),
            )
            period = 1.0 / args.pectoral_frequency_hz
            for index in range(41):
                angles = self.sample(
                    period * index / 40.0,
                    args,
                    targets,
                ).commands_by_servo_id_deg
                height4 = (
                    endpoints["servo_4_bottom_deg"] - angles[4]
                ) / (
                    endpoints["servo_4_bottom_deg"]
                    - endpoints["servo_4_top_deg"]
                )
                height6 = (
                    angles[6] - endpoints["servo_6_bottom_deg"]
                ) / (
                    endpoints["servo_6_top_deg"]
                    - endpoints["servo_6_bottom_deg"]
                )
                self.assertAlmostEqual(height4 + height6, 1.0, places=12)

    def test_root_ratio_is_applied_to_each_distinct_physical_span(self) -> None:
        expected = {
            "right": (150.0, 125.0, 180.0, 165.0),
            "left": (50.0, 75.0, 120.0, 135.0),
        }
        for direction, values in expected.items():
            with self.subTest(direction=direction):
                roots = build_roll_root_targets(
                    roll_direction=direction,
                    root_amplitude_ratio=0.25,
                    servo_4_top_deg=50.0,
                    servo_4_bottom_deg=150.0,
                    servo_6_top_deg=180.0,
                    servo_6_bottom_deg=120.0,
                )
                self.assertEqual(
                    (
                        roots.servo_4_start_deg,
                        roots.servo_4_target_deg,
                        roots.servo_6_start_deg,
                        roots.servo_6_target_deg,
                    ),
                    values,
                )
                self.assertEqual(
                    abs(roots.servo_4_target_deg - roots.servo_4_start_deg),
                    25.0,
                )
                self.assertEqual(
                    abs(roots.servo_6_target_deg - roots.servo_6_start_deg),
                    15.0,
                )

    def test_tip_offsets_are_identical_for_both_tip_directions(self) -> None:
        for tip_direction, expected_sign in (("positive", 1), ("negative", -1)):
            args, targets = self.make_targets(
                "--tip-direction", tip_direction,
                "--tip-amplitude-deg", "13",
            )
            period = 1.0 / args.pectoral_frequency_hz
            for index in range(101):
                angles = self.sample(
                    period * index / 100.0,
                    args,
                    targets,
                ).commands_by_servo_id_deg
                offset5 = angles[5] - targets["tips"][5]["center"]
                offset7 = angles[7] - targets["tips"][7]["center"]
                self.assertAlmostEqual(offset5, offset7, places=12)
                self.assertGreaterEqual(expected_sign * offset5, -1e-12)

    def test_all_required_tip_boundaries_and_shared_phase(self) -> None:
        args, targets = self.make_targets()
        period = 1.0 / args.pectoral_frequency_hz
        checkpoints = {
            0.0: 0.0,
            args.tip_transition_s: 1.0,
            period / 2.0 - args.tip_transition_s: 1.0,
            period / 2.0: 0.0,
            period: 0.0,
        }
        for elapsed, expected_h in checkpoints.items():
            sample = self.sample(elapsed, args, targets)
            self.assertGreaterEqual(sample.pectoral_phase, 0.0)
            self.assertLess(sample.pectoral_phase, 1.0)
            self.assertAlmostEqual(sample.tip_envelope_h, expected_h)

    def test_root_and_tip_positions_and_velocities_are_continuous(self) -> None:
        args, targets = self.make_targets()
        period = 1.0 / args.pectoral_frequency_hz
        epsilon = 1e-5
        boundaries = (
            args.tip_transition_s,
            period / 2.0 - args.tip_transition_s,
            period / 2.0,
            period,
        )
        for boundary in boundaries:
            before = self.sample(boundary - epsilon, args, targets).commands_by_servo_id_deg
            at = self.sample(boundary, args, targets).commands_by_servo_id_deg
            after = self.sample(boundary + epsilon, args, targets).commands_by_servo_id_deg
            for servo_id in (4, 5, 6, 7):
                self.assertLess(abs(before[servo_id] - at[servo_id]), 1e-3)
                self.assertLess(abs(after[servo_id] - at[servo_id]), 1e-3)
                left_velocity = (at[servo_id] - before[servo_id]) / epsilon
                right_velocity = (after[servo_id] - at[servo_id]) / epsilon
                self.assertLess(abs(left_velocity - right_velocity), 0.02)

    def test_all_periodic_targets_remain_in_prevalidated_ranges(self) -> None:
        for direction in ("left", "right"):
            for tip_direction in ("positive", "negative"):
                args, targets = self.make_targets(
                    "--roll-direction", direction,
                    "--tip-direction", tip_direction,
                    "--tail-left-amplitude-deg", "8",
                    "--tail-right-amplitude-deg", "3",
                )
                period = 1.0 / args.pectoral_frequency_hz
                for index in range(401):
                    sample = self.sample(period * index / 400.0, args, targets)
                    for servo_id, angle in sample.commands_by_servo_id_deg.items():
                        mechanical_min, mechanical_max = targets["mechanical_ranges"][servo_id]
                        self.assertGreaterEqual(angle, mechanical_min - 1e-10)
                        self.assertLessEqual(angle, mechanical_max + 1e-10)


class TailTrajectoryTests(RollTestCase):
    def test_tail_starts_center_and_reaches_unequal_extrema(self) -> None:
        args, targets = self.make_targets(
            "--tail-frequency-hz", "0.5",
            "--tail-left-amplitude-deg", "12",
            "--tail-right-amplitude-deg", "7",
            "--tail-first-direction", "left",
        )
        tail_period = 1.0 / args.tail_frequency_hz
        start = self.sample(0.0, args, targets).commands_by_servo_id_deg
        left = self.sample(tail_period / 4.0, args, targets).commands_by_servo_id_deg
        right = self.sample(3.0 * tail_period / 4.0, args, targets).commands_by_servo_id_deg
        end = self.sample(tail_period, args, targets).commands_by_servo_id_deg
        for servo_id in (1, 2, 3):
            center = targets["tail"][servo_id]["center"]
            self.assertAlmostEqual(start[servo_id], center)
            self.assertAlmostEqual(left[servo_id], center + 12.0)
            self.assertAlmostEqual(right[servo_id], center - 7.0)
            self.assertAlmostEqual(end[servo_id], center)

    def test_tail_first_direction_changes_sign_without_prepositioning(self) -> None:
        for first_direction, expected_sign in (("left", 1), ("right", -1)):
            args, targets = self.make_targets(
                "--tail-first-direction", first_direction,
                "--tail-left-amplitude-deg", "9",
                "--tail-right-amplitude-deg", "4",
            )
            at_zero = self.sample(0.0, args, targets).commands_by_servo_id_deg
            just_after = self.sample(1e-4, args, targets).commands_by_servo_id_deg
            for servo_id in (1, 2, 3):
                center = targets["tail"][servo_id]["center"]
                self.assertAlmostEqual(at_zero[servo_id], center)
                self.assertGreater(expected_sign * (just_after[servo_id] - center), 0.0)

    def test_tail_mapping_applies_individual_sign_and_scale(self) -> None:
        roots = RootPairTargets("right", 160.0, 150.0, 190.0, 180.0)
        tail_servos = {
            1: TailServoMotion(80.0, 1.0, 1.0),
            2: TailServoMotion(90.0, -1.0, 0.5),
            3: TailServoMotion(100.0, 1.0, 1.5),
        }
        sample = evaluate_roll_trajectory(
            0.25,
            pectoral_frequency_hz=1.0,
            roots=roots,
            tip_centers_deg={5: 90.0, 7: 90.0},
            tip_amplitude_deg=0.0,
            tip_transition_s=0.1,
            tip_direction="positive",
            tail_frequency_hz=1.0,
            tail_left_amplitude_deg=8.0,
            tail_right_amplitude_deg=3.0,
            tail_first_direction="left",
            tail_servos=tail_servos,
        )
        angles = sample.commands_by_servo_id_deg
        self.assertAlmostEqual(angles[1], 88.0)
        self.assertAlmostEqual(angles[2], 86.0)
        self.assertAlmostEqual(angles[3], 112.0)

    def test_unequal_tail_waveform_is_position_continuous_at_center(self) -> None:
        args, targets = self.make_targets(
            "--tail-frequency-hz", "0.5",
            "--tail-left-amplitude-deg", "12",
            "--tail-right-amplitude-deg", "7",
        )
        crossing = 1.0 / (2.0 * args.tail_frequency_hz)
        epsilon = 1e-8
        before = self.sample(crossing - epsilon, args, targets).commands_by_servo_id_deg
        at = self.sample(crossing, args, targets).commands_by_servo_id_deg
        after = self.sample(crossing + epsilon, args, targets).commands_by_servo_id_deg
        for servo_id in (1, 2, 3):
            self.assertLess(abs(before[servo_id] - at[servo_id]), 1e-5)
            self.assertLess(abs(after[servo_id] - at[servo_id]), 1e-5)


class RollValidationTests(RollTestCase):
    def test_default_parameters_are_conservative_and_valid(self) -> None:
        args, targets = self.make_targets()
        self.assertEqual(args.roll_direction, "right")
        self.assertEqual(args.tip_direction, "negative")
        self.assertEqual(args.root_amplitude_ratio, 0.1)
        self.assertEqual(args.tip_amplitude_deg, 10.0)
        self.assertEqual(args.tail_left_amplitude_deg, 5.0)
        self.assertEqual(args.tail_right_amplitude_deg, 5.0)
        self.assertEqual(args.start_delay_s, 20.0)
        self.assertFalse(hasattr(args, "mechanical_limit_margin_deg"))
        self.assertEqual(len(targets["centers"]), 7)

    def test_tip_transition_must_be_strictly_less_than_quarter_period(self) -> None:
        args = self.make_args(
            "--pectoral-frequency-hz", "1",
            "--tip-transition-s", "0.25",
        )
        with self.assertRaises(SystemExit) as caught:
            action._validate_scalar_args(args)
        self.assertIn("--tip-transition-s", str(caught.exception))

    def test_all_required_scalar_bounds_fail_before_target_preparation(self) -> None:
        invalid_cases = (
            ("--duration", "0"),
            ("--pectoral-frequency-hz", "0"),
            ("--root-amplitude-ratio", "-0.1"),
            ("--root-amplitude-ratio", "1.1"),
            ("--tip-amplitude-deg", "-1"),
            ("--tip-transition-s", "0"),
            ("--tail-frequency-hz", "0"),
            ("--tail-left-amplitude-deg", "-1"),
            ("--tail-right-amplitude-deg", "-1"),
            ("--start-delay-s", "-1"),
            ("--initial-hold-s", "-1"),
            ("--tail-ramp-down-s", "0"),
            ("--return-to-center-s", "0"),
            ("--command-hz", "0"),
            ("--duration", "nan"),
        )
        for option, value in invalid_cases:
            with self.subTest(option=option, value=value):
                args = self.make_args(option, value)
                with self.assertRaises(SystemExit):
                    action._validate_scalar_args(args)

    def test_full_root_ratio_reaches_configured_physical_endpoints(self) -> None:
        for direction in ("right", "left"):
            with self.subTest(direction=direction):
                args, targets = self.make_targets(
                    "--roll-direction", direction,
                    "--root-amplitude-ratio", "1",
                )
                roots = targets["roots"]
                endpoints = targets["physical_endpoints"]
                values = (
                    (
                        endpoints["servo_4_bottom_deg"],
                        endpoints["servo_4_top_deg"],
                        endpoints["servo_6_top_deg"],
                        endpoints["servo_6_bottom_deg"],
                    )
                    if direction == "right"
                    else (
                        endpoints["servo_4_top_deg"],
                        endpoints["servo_4_bottom_deg"],
                        endpoints["servo_6_bottom_deg"],
                        endpoints["servo_6_top_deg"],
                    )
                )
                self.assertEqual(
                    (
                        roots.servo_4_start_deg,
                        roots.servo_4_target_deg,
                        roots.servo_6_start_deg,
                        roots.servo_6_target_deg,
                    ),
                    values,
                )

    def test_configured_min_max_are_used_directly_without_extra_margin(self) -> None:
        args, targets = self.make_targets("--roll-direction", "right")
        self.assertFalse(hasattr(args, "mechanical_limit_margin_deg"))
        for servo_id in (4, 6):
            raw = next(
                item for item in self.config["servo"]["channels"]
                if int(item["servo_id"]) == servo_id
            )
            self.assertEqual(
                targets["mechanical_ranges"][servo_id],
                (float(raw["min_angle"]), float(raw["max_angle"])),
            )
        endpoints = targets["physical_endpoints"]
        self.assertEqual(targets["roots"].servo_4_start_deg, endpoints["servo_4_bottom_deg"])
        self.assertEqual(targets["roots"].servo_6_start_deg, endpoints["servo_6_top_deg"])

    def test_legacy_global_test_amplitude_does_not_override_servo_limits(self) -> None:
        config = copy.deepcopy(self.config)
        config["safety"]["servo"]["max_test_amplitude_deg"] = 0.001
        for option, value in (
            ("--root-amplitude-ratio", "0.2"),
            ("--tip-amplitude-deg", "16"),
            ("--tail-left-amplitude-deg", "16"),
            ("--tail-right-amplitude-deg", "16"),
        ):
            with self.subTest(option=option, value=value):
                args = self.make_args(option, value)
                action._validate_scalar_args(args)
                targets = action._prepare_targets(config, args)
                self.assertNotIn("max_test_amplitude_deg", targets)

    def test_tip_and_tail_limits_are_checked_independently(self) -> None:
        tip_args = self.make_args(
            "--tip-direction", "positive",
            "--tip-amplitude-deg", "100",
        )
        action._validate_scalar_args(tip_args)
        with self.assertRaises(SystemExit) as tip_error:
            action._prepare_targets(self.config, tip_args)
        self.assertIn("servo_id=5", str(tip_error.exception))
        self.assertIn("--tip-amplitude-deg=100.000", str(tip_error.exception))

        tail_args = self.make_args("--tail-left-amplitude-deg", "40")
        action._validate_scalar_args(tail_args)
        with self.assertRaises(SystemExit) as tail_error:
            action._prepare_targets(self.config, tail_args)
        self.assertIn("servo_id=1", str(tail_error.exception))
        self.assertIn("--tail-left-amplitude-deg=40.000", str(tail_error.exception))

    def test_missing_physical_reference_is_not_guessed_from_min_max(self) -> None:
        config = copy.deepcopy(self.config)
        servo4 = next(
            item
            for item in config["servo"]["channels"]
            if int(item["servo_id"]) == 4
        )
        del servo4["direction"]["top_reference_angle"]
        args = self.make_args()
        action._validate_scalar_args(args)
        with self.assertRaises(SystemExit) as caught:
            action._prepare_targets(config, args)
        self.assertIn("missing explicit direction.top_reference_angle", str(caught.exception))
        self.assertIn("will not be guessed", str(caught.exception))

    def test_mock_sensors_does_not_bypass_move_confirmation(self) -> None:
        mock_only = self.make_args("--mock-sensors")
        with self.assertRaises(SystemExit):
            action._validate_motion_authorization(mock_only)

        dry_run = self.make_args("--dry-run", "--mock-sensors")
        action._validate_motion_authorization(dry_run)

        confirmed = self.make_args("--mock-sensors", "--confirm", "MOVE")
        action._validate_motion_authorization(confirmed)

    def test_dry_run_profile_covers_one_complete_period_and_all_fields(self) -> None:
        args, targets = self.make_targets(
            "--pectoral-frequency-hz", "0.5",
            "--tip-transition-s", "0.2",
            "--tail-frequency-hz", "0.25",
            "--tail-left-amplitude-deg", "8",
            "--tail-right-amplitude-deg", "3",
        )
        rows = action._build_dry_run_profile(args, targets)
        period = max(
            1.0 / args.pectoral_frequency_hz,
            1.0 / args.tail_frequency_hz,
        )
        self.assertEqual(rows[0]["time_s"], 0.0)
        self.assertEqual(rows[-1]["time_s"], period)
        for key in (
            "time_s",
            "theta_1",
            "theta_2",
            "theta_3",
            "theta_4",
            "theta_5",
            "theta_6",
            "theta_7",
            "tail_offset_deg",
        ):
            self.assertIn(key, rows[0])

        with tempfile.TemporaryDirectory() as tmp:
            path = action._write_dry_run_profile(Path(tmp), args, targets)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


class RollLifecycleTests(RollTestCase):
    @staticmethod
    def short_argv(*extra: str) -> list[str]:
        return [
            "--mock-sensors",
            "--duration", "0.04",
            "--pectoral-frequency-hz", "1",
            "--root-amplitude-ratio", "0.05",
            "--tip-amplitude-deg", "3",
            "--tip-transition-s", "0.1",
            "--tail-frequency-hz", "1",
            "--tail-left-amplitude-deg", "4",
            "--tail-right-amplitude-deg", "2",
            "--initial-hold-s", "0",
            "--tail-ramp-down-s", "0.02",
            "--return-to-center-s", "0.02",
            "--command-hz", "100",
            *extra,
        ]

    def test_dry_run_lifecycle_events_delay_order_and_log_files(self) -> None:
        required_events = (
            "start_delay_begin",
            "start_delay_end",
            "sensor_recording_begin",
            "initial_pose_begin",
            "initial_pose_ready",
            "baseline_hold_begin",
            "baseline_hold_end",
            "roll_action_begin",
            "roll_action_end",
            "tail_ramp_down_begin",
            "tail_ramp_down_end",
            "return_to_center_begin",
            "return_to_center_end",
            "sensor_recording_end",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "logs"
            countdown_calls: list[float] = []

            def fake_countdown(delay_s: float) -> None:
                countdown_calls.append(delay_s)
                self.assertFalse(base_dir.exists())

            with (
                mock.patch.object(action, "_resolve_log_base_dir", return_value=base_dir),
                mock.patch.object(action, "countdown_start_delay", side_effect=fake_countdown),
                mock.patch.object(
                    action,
                    "PCA9685ServoController",
                    side_effect=AssertionError("dry-run must not construct PCA9685"),
                ),
            ):
                result = action.main(
                    self.short_argv(
                        "--dry-run",
                        "--start-delay-s", "0.1",
                        "--roll-direction", "left",
                        "--tip-direction", "positive",
                    )
                )

            self.assertEqual(result, 0)
            self.assertEqual(countdown_calls, [0.1])
            run_dirs = list(base_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            log_dir = run_dirs[0]
            events = [
                json.loads(line)
                for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            event_types = [entry["event_type"] for entry in events]
            positions = [event_types.index(name) for name in required_events]
            self.assertEqual(positions, sorted(positions))
            event_times = [entry["t_ns"] for entry in events]
            self.assertTrue(all(isinstance(value, int) for value in event_times))
            self.assertEqual(event_times, sorted(event_times))

            commands = [
                json.loads(line)
                for line in (log_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(commands), 0)
            command_times = [entry["t_ns"] for entry in commands]
            self.assertEqual(command_times, sorted(command_times))
            self.assertTrue(all(entry["roll_direction"] == "left" for entry in commands))
            self.assertTrue(all(entry["tip_direction"] == "positive" for entry in commands))
            action_commands = [
                entry for entry in commands
                if entry["action_state"] == "roll_action"
            ]
            self.assertEqual(action_commands[0]["action_elapsed_s"], 0.0)
            phase_zero = action_commands[0]["commands_by_servo_id_deg"]
            tail_centers = {
                str(item["servo_id"]): float(item["center_angle"])
                for item in self.config["servo"]["channels"]
                if int(item["servo_id"]) in (1, 2, 3)
            }
            self.assertEqual(
                {servo_id: phase_zero[servo_id] for servo_id in ("1", "2", "3")},
                tail_centers,
            )
            servo_4 = next(item for item in self.config["servo"]["channels"] if item["servo_id"] == 4)
            servo_6 = next(item for item in self.config["servo"]["channels"] if item["servo_id"] == 6)
            self.assertEqual(phase_zero["4"], float(servo_4["direction"]["top_reference_angle"]))
            servo_5 = next(item for item in self.config["servo"]["channels"] if item["servo_id"] == 5)
            servo_7 = next(item for item in self.config["servo"]["channels"] if item["servo_id"] == 7)
            self.assertEqual(phase_zero["5"], float(servo_5["center_angle"]))
            self.assertEqual(phase_zero["6"], float(servo_6["direction"]["bottom_reference_angle"]))
            self.assertEqual(phase_zero["7"], float(servo_7["center_angle"]))

            for relative_path in (
                "metadata.yaml",
                "synchronized_sensors.jsonl",
                "events.jsonl",
                "commands.jsonl",
                "raw_imu.jsonl",
                "raw_uwb.jsonl",
                "raw_depth.jsonl",
                "raw_power.jsonl",
                "raw_vision.jsonl",
                "camera_index.jsonl",
                "camera",
                action.DRY_RUN_PROFILE_FILE,
            ):
                self.assertTrue((log_dir / relative_path).exists(), relative_path)

    def test_keep_pwm_controls_normal_release_and_failed_recenter_forces_release(self) -> None:
        class TrackingController(action.DryRunServoController):
            def __init__(self, config):
                super().__init__(config)
                self.stop_calls: list[list[int] | None] = []

            def stop_all(self, channels=None) -> None:
                self.stop_calls.append(None if channels is None else list(channels))
                super().stop_all(channels)

        def run_case(*extra: str, fail_recenter: bool = False):
            with tempfile.TemporaryDirectory() as tmp:
                base_dir = Path(tmp) / "logs"
                controllers: list[TrackingController] = []

                def controller_factory(config):
                    controller = TrackingController(config)
                    controllers.append(controller)
                    return controller

                patches = [
                    mock.patch.object(action, "_resolve_log_base_dir", return_value=base_dir),
                    mock.patch.object(action, "countdown_start_delay", return_value=None),
                    mock.patch.object(action, "ensure_not_windows_hardware_run", return_value=None),
                    mock.patch.object(action, "PCA9685ServoController", side_effect=controller_factory),
                ]
                if fail_recenter:
                    patches.append(
                        mock.patch.object(
                            action,
                            "_execute_return_to_center",
                            side_effect=RuntimeError("synthetic recenter failure"),
                        )
                    )
                with patches[0], patches[1], patches[2], patches[3]:
                    if fail_recenter:
                        with patches[4], self.assertRaises(RuntimeError):
                            action.main(
                                self.short_argv(
                                    "--confirm", "MOVE",
                                    "--start-delay-s", "0",
                                    *extra,
                                )
                            )
                    else:
                        action.main(
                            self.short_argv(
                                "--confirm", "MOVE",
                                "--start-delay-s", "0",
                                *extra,
                            )
                        )
                self.assertEqual(len(controllers), 1)
                return len(controllers[0].stop_calls)

        self.assertEqual(run_case(), 1)
        self.assertEqual(run_case("--keep-pwm"), 0)
        self.assertEqual(run_case("--keep-pwm", fail_recenter=True), 1)

    def test_cleanup_before_action_does_not_start_a_tail_excursion(self) -> None:
        args, targets = self.make_targets(
            "--tail-left-amplitude-deg", "12",
            "--tail-right-amplitude-deg", "7",
            "--tail-ramp-down-s", "1",
        )
        start_commands = dict(targets["centers"])
        tail_1_channel = int(targets["tail"][1]["channel"])
        start_commands[tail_1_channel] += 3.0
        runtime_state = {
            "last_commands": start_commands,
            "action_elapsed_s": None,
        }
        samples: list[tuple[dict[int, float], dict[str, object]]] = []

        def fake_run_stage(**kwargs):
            for elapsed in (0.0, 0.5, 1.0):
                samples.append(kwargs["command_builder"](elapsed))
            kwargs["runtime_state"]["last_commands"] = dict(samples[-1][0])
            return dict(samples[-1][0])

        with mock.patch.object(action, "_run_stage", side_effect=fake_run_stage):
            action._execute_tail_ramp_down(
                self.config,
                args,
                mock.Mock(),
                targets,
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                0.0,
                runtime_state,
            )

        for commands, state in samples:
            self.assertIsNone(state["tail_phase_elapsed_s"])
            for servo_id in (1, 2, 3):
                channel = int(targets["tail"][servo_id]["channel"])
                center = targets["tail"][servo_id]["center"]
                if servo_id == 1:
                    self.assertGreaterEqual(commands[channel], center)
                    self.assertLessEqual(commands[channel], center + 3.0)
                else:
                    self.assertAlmostEqual(commands[channel], center)
        self.assertAlmostEqual(samples[-1][0][tail_1_channel], targets["tail"][1]["center"])


if __name__ == "__main__":
    unittest.main()
