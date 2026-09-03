from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mission_dive_roll_20260713 as mission  # noqa: E402
from control.runtime.action_utils import DryRunServoController  # noqa: E402
from control.safety import load_robot_config  # noqa: E402


class DiveRollTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_robot_config()

    def make_args(self, *argv: str):
        return mission.parse_args(self.config, list(argv))

    def make_targets(self, *argv: str):
        args = self.make_args(*argv)
        mission._validate_scalar_args(args)
        return args, mission._prepare_targets(self.config, args)

    @staticmethod
    def by_servo_id(commands, targets):
        return {
            targets["channel_to_servo_id"][channel]: angle
            for channel, angle in commands.items()
        }

    @staticmethod
    def short_argv(*extra: str) -> list[str]:
        return [
            "--mock-sensors",
            "--dive-duration", "0.04",
            "--dive-pectoral-tilt", "3",
            "--transition-s", "0.02",
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


class DiveTrajectoryTests(DiveRollTestCase):
    def test_default_dive_pose_uses_centered_roots_and_mirrored_tip_tilt(self) -> None:
        args, targets = self.make_targets()
        pose = targets["dive_pose_by_servo_id"]
        centers = targets["centers_by_servo_id"]

        self.assertEqual(pose[4], centers[4])
        self.assertEqual(pose[6], centers[6])
        self.assertEqual(pose[5], centers[5] + args.dive_pectoral_tilt)
        self.assertEqual(pose[7], centers[7] - args.dive_pectoral_tilt)

    def test_dive_tail_starts_at_center_and_reaches_asymmetric_extrema(self) -> None:
        args, targets = self.make_targets(
            "--tail-frequency-hz", "0.5",
            "--tail-left-amplitude-deg", "12",
            "--tail-right-amplitude-deg", "7",
            "--tail-first-direction", "left",
        )
        period = 1.0 / args.tail_frequency_hz
        start = self.by_servo_id(mission._dive_commands(0.0, args, targets)[0], targets)
        left = self.by_servo_id(
            mission._dive_commands(period / 4.0, args, targets)[0],
            targets,
        )
        right = self.by_servo_id(
            mission._dive_commands(3.0 * period / 4.0, args, targets)[0],
            targets,
        )

        for servo_id in mission.TAIL_SERVO_IDS:
            self.assertAlmostEqual(start[servo_id], targets["tail"][servo_id]["center"])
            self.assertAlmostEqual(left[servo_id], targets["tail"][servo_id]["left_target"])
            self.assertAlmostEqual(right[servo_id], targets["tail"][servo_id]["right_target"])

    def test_dive_end_to_transition_start_is_continuous_for_every_servo(self) -> None:
        args, targets = self.make_targets(
            "--dive-duration", "0.37",
            "--tail-frequency-hz", "0.7",
            "--tail-left-amplitude-deg", "11",
            "--tail-right-amplitude-deg", "4",
        )
        dive_end = self.by_servo_id(
            mission._dive_commands(args.dive_duration, args, targets)[0],
            targets,
        )
        transition_start = self.by_servo_id(
            mission._dive_to_roll_commands(0.0, args, targets)[0],
            targets,
        )
        for servo_id in mission.ALL_SERVO_IDS:
            self.assertAlmostEqual(dive_end[servo_id], transition_start[servo_id], places=12)

    def test_transition_end_to_roll_phase_zero_is_continuous_for_every_servo(self) -> None:
        args, targets = self.make_targets(
            "--roll-direction", "left",
            "--root-amplitude-ratio", "0.4",
            "--dive-duration", "0.37",
            "--transition-s", "0.23",
        )
        transition_end = self.by_servo_id(
            mission._dive_to_roll_commands(args.transition_s, args, targets)[0],
            targets,
        )
        roll_start = self.by_servo_id(
            mission._roll_commands(0.0, args, targets)[0],
            targets,
        )
        for servo_id in mission.ALL_SERVO_IDS:
            self.assertAlmostEqual(transition_end[servo_id], roll_start[servo_id], places=12)

    def test_root_ratio_one_reaches_explicit_endpoints_in_both_directions(self) -> None:
        for direction in ("right", "left"):
            with self.subTest(direction=direction):
                _, targets = self.make_targets(
                    "--roll-direction", direction,
                    "--root-amplitude-ratio", "1",
                )
                roots = targets["roots"]
                endpoints = targets["physical_endpoints"]
                if direction == "right":
                    expected = (
                        endpoints["servo_4_bottom_deg"],
                        endpoints["servo_4_top_deg"],
                        endpoints["servo_6_top_deg"],
                        endpoints["servo_6_bottom_deg"],
                    )
                else:
                    expected = (
                        endpoints["servo_4_top_deg"],
                        endpoints["servo_4_bottom_deg"],
                        endpoints["servo_6_bottom_deg"],
                        endpoints["servo_6_top_deg"],
                    )
                actual = (
                    roots.servo_4_start_deg,
                    roots.servo_4_target_deg,
                    roots.servo_6_start_deg,
                    roots.servo_6_target_deg,
                )
                self.assertEqual(actual, expected)


class DiveRollValidationTests(DiveRollTestCase):
    def test_old_max_test_amplitude_does_not_restrict_combined_targets(self) -> None:
        config = copy.deepcopy(self.config)
        config["safety"]["servo"]["max_test_amplitude_deg"] = 0.001
        args = mission.parse_args(
            config,
            [
                "--root-amplitude-ratio", "1",
                "--dive-pectoral-tilt", "90",
                "--tail-left-amplitude-deg", "30",
                "--tail-right-amplitude-deg", "30",
            ],
        )
        mission._validate_scalar_args(args)
        targets = mission._prepare_targets(config, args)

        self.assertNotIn("max_test_amplitude_deg", targets)
        self.assertEqual(targets["dive_pose_by_servo_id"][5], 180.0)
        self.assertEqual(targets["dive_pose_by_servo_id"][7], 0.0)
        for servo_id in mission.TAIL_SERVO_IDS:
            self.assertEqual(
                targets["tail"][servo_id]["left_target"],
                targets["limits"][servo_id].max_angle,
            )
            self.assertEqual(
                targets["tail"][servo_id]["right_target"],
                targets["limits"][servo_id].min_angle,
            )
        servo_4 = next(item for item in config["servo"]["channels"] if item["servo_id"] == 4)
        servo_6 = next(item for item in config["servo"]["channels"] if item["servo_id"] == 6)
        self.assertEqual(
            targets["roots"].servo_4_target_deg,
            float(servo_4["direction"]["top_reference_angle"]),
        )
        self.assertEqual(
            targets["roots"].servo_6_target_deg,
            float(servo_6["direction"]["bottom_reference_angle"]),
        )

    def test_dive_tilt_just_outside_tip_limits_is_rejected_preflight(self) -> None:
        args = self.make_args("--dive-pectoral-tilt", "90.1")
        mission._validate_scalar_args(args)
        with self.assertRaises(SystemExit) as caught:
            mission._prepare_targets(self.config, args)
        message = str(caught.exception)
        self.assertIn("servo_id=5", message)
        self.assertIn("--dive-pectoral-tilt=90.100", message)
        self.assertIn("calculated target=180.100", message)
        self.assertIn("0.000..180.000", message)

    def test_invalid_dive_duration_tilt_and_transition_are_rejected(self) -> None:
        invalid_cases = (
            ("--dive-duration", "0"),
            ("--dive-duration", "-1"),
            ("--dive-duration", "nan"),
            ("--dive-duration", "inf"),
            ("--dive-pectoral-tilt", "-0.1"),
            ("--dive-pectoral-tilt", "nan"),
            ("--dive-pectoral-tilt", "inf"),
            ("--transition-s", "0"),
            ("--transition-s", "-1"),
            ("--transition-s", "nan"),
            ("--transition-s", "inf"),
        )
        for option, value in invalid_cases:
            with self.subTest(option=option, value=value):
                args = self.make_args(option, value)
                with self.assertRaises(SystemExit):
                    mission._validate_scalar_args(args)

    def test_invalid_root_ratio_is_rejected_by_self_contained_validation(self) -> None:
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value):
                args = self.make_args("--root-amplitude-ratio", value)
                with self.assertRaises(SystemExit):
                    mission._validate_scalar_args(args)

    def test_mock_sensors_does_not_bypass_real_motion_confirmation(self) -> None:
        with (
            mock.patch.object(
                mission,
                "countdown_start_delay",
                side_effect=AssertionError("authorization must fail before countdown"),
            ),
            mock.patch.object(
                mission,
                "PCA9685ServoController",
                side_effect=AssertionError("authorization must fail before PCA9685"),
            ),
            self.assertRaises(SystemExit) as caught,
        ):
            mission.main(["--mock-sensors"])
        self.assertIn("--confirm MOVE", str(caught.exception))

    def test_combined_dry_run_profile_contains_full_mission_and_all_angles(self) -> None:
        args, targets = self.make_targets(
            "--dive-duration", "0.2",
            "--transition-s", "0.1",
            "--duration", "0.2",
            "--pectoral-frequency-hz", "1",
            "--tip-transition-s", "0.1",
            "--tail-frequency-hz", "1",
            "--command-hz", "20",
        )
        rows = mission._build_dry_run_profile(args, targets)
        expected_phases = (
            "dive_initial_pose",
            "dive_action",
            "dive_to_roll_transition",
            "initial_hold",
            "roll_action",
            "tail_ramp_down",
            "return_to_center",
        )
        phase_order = tuple(dict.fromkeys(row["phase"] for row in rows))
        self.assertEqual(phase_order, expected_phases)
        for phase in expected_phases:
            first = next(row for row in rows if row["phase"] == phase)
            self.assertEqual(first["phase_elapsed_s"], 0.0)
        for row in rows:
            for servo_id in mission.ALL_SERVO_IDS:
                self.assertIn(f"theta_{servo_id}", row)
        self.assertAlmostEqual(
            rows[-1]["sequence_time_s"],
            (2.0 * args.transition_s)
            + args.dive_duration
            + args.initial_hold_s
            + args.duration
            + args.tail_ramp_down_s
            + args.return_to_center_s,
        )
        for servo_id in mission.ALL_SERVO_IDS:
            self.assertAlmostEqual(
                rows[-1][f"theta_{servo_id}"],
                targets["centers_by_servo_id"][servo_id],
            )


class DiveRollLifecycleTests(DiveRollTestCase):
    def test_short_dry_run_lifecycle_order_logs_and_zeroed_roll_clock(self) -> None:
        required_events = (
            "start_delay_begin",
            "start_delay_end",
            "sensor_recording_begin",
            "safe_center_begin",
            "safe_center_ready",
            "dive_initial_pose_begin",
            "dive_initial_pose_ready",
            "dive_action_begin",
            "dive_action_end",
            "dive_to_roll_transition_begin",
            "initial_pose_begin",
            "initial_pose_ready",
            "dive_to_roll_transition_end",
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
            with (
                mock.patch.object(mission, "_resolve_log_base_dir", return_value=base_dir),
                mock.patch.object(mission, "countdown_start_delay", return_value=None),
                mock.patch.object(
                    mission,
                    "PCA9685ServoController",
                    side_effect=AssertionError("dry-run must not construct PCA9685"),
                ),
            ):
                result = mission.main(
                    self.short_argv("--dry-run", "--start-delay-s", "0")
                )

            self.assertEqual(result, 0)
            run_dirs = list(base_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            log_dir = run_dirs[0]

            events = [
                json.loads(line)
                for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            event_types = [entry["event_type"] for entry in events]
            positions = [event_types.index(event) for event in required_events]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(
                [entry["t_ns"] for entry in events],
                sorted(entry["t_ns"] for entry in events),
            )

            commands = [
                json.loads(line)
                for line in (log_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            dive_commands = [entry for entry in commands if entry["action_state"] == "dive_action"]
            self.assertEqual(dive_commands[0]["dive_elapsed_s"], 0.0)
            self.assertAlmostEqual(dive_commands[-1]["dive_elapsed_s"], 0.04)

            roll_commands = [entry for entry in commands if entry["action_state"] == "roll_action"]
            self.assertEqual(roll_commands[0]["state_elapsed_s"], 0.0)
            self.assertEqual(roll_commands[0]["roll_elapsed_s"], 0.0)
            self.assertEqual(roll_commands[0]["action_elapsed_s"], 0.0)
            self.assertTrue((log_dir / mission.DRY_RUN_PROFILE_FILE).exists())

    def test_pwm_release_keep_and_failed_recenter_semantics(self) -> None:
        class TrackingController(DryRunServoController):
            def __init__(self, config):
                super().__init__(config)
                self.stop_calls: list[list[int] | None] = []

            def stop_all(self, channels=None) -> None:
                self.stop_calls.append(None if channels is None else list(channels))
                super().stop_all(channels)

        def run_case(*extra: str, fail_recenter: bool = False) -> int:
            with tempfile.TemporaryDirectory() as tmp:
                base_dir = Path(tmp) / "logs"
                controllers: list[TrackingController] = []

                def controller_factory(config):
                    controller = TrackingController(config)
                    controllers.append(controller)
                    return controller

                with (
                    mock.patch.object(mission, "_resolve_log_base_dir", return_value=base_dir),
                    mock.patch.object(mission, "countdown_start_delay", return_value=None),
                    mock.patch.object(mission, "ensure_not_windows_hardware_run", return_value=None),
                    mock.patch.object(
                        mission,
                        "PCA9685ServoController",
                        side_effect=controller_factory,
                    ),
                ):
                    if fail_recenter:
                        with (
                            mock.patch.object(
                                mission,
                                "_execute_return_to_center",
                                side_effect=RuntimeError("synthetic recenter failure"),
                            ),
                            self.assertRaises(RuntimeError),
                        ):
                            mission.main(
                                self.short_argv(
                                    "--confirm", "MOVE",
                                    "--start-delay-s", "0",
                                    *extra,
                                )
                            )
                    else:
                        mission.main(
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

    def test_repeated_ctrl_c_during_cleanup_still_closes_logs_and_disables_pwm(self) -> None:
        class TrackingController(DryRunServoController):
            def __init__(self, config):
                super().__init__(config)
                self.stop_calls: list[list[int] | None] = []

            def stop_all(self, channels=None) -> None:
                self.stop_calls.append(None if channels is None else list(channels))
                super().stop_all(channels)

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "logs"
            controllers: list[TrackingController] = []

            def controller_factory(config):
                controller = TrackingController(config)
                controllers.append(controller)
                return controller

            with (
                mock.patch.object(mission, "_resolve_log_base_dir", return_value=base_dir),
                mock.patch.object(mission, "countdown_start_delay", return_value=None),
                mock.patch.object(mission, "ensure_not_windows_hardware_run", return_value=None),
                mock.patch.object(
                    mission,
                    "PCA9685ServoController",
                    side_effect=controller_factory,
                ),
                mock.patch.object(mission, "_run_stage", side_effect=KeyboardInterrupt),
                mock.patch.object(
                    mission,
                    "_execute_tail_ramp_down",
                    side_effect=KeyboardInterrupt,
                ),
                mock.patch.object(
                    mission,
                    "_execute_return_to_center",
                    side_effect=KeyboardInterrupt,
                ),
            ):
                result = mission.main(
                    self.short_argv(
                        "--confirm", "MOVE",
                        "--keep-pwm",
                        "--start-delay-s", "0",
                    )
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(controllers), 1)
            self.assertEqual(len(controllers[0].stop_calls), 1)
            run_dir = next(base_dir.iterdir())
            event_types = [
                json.loads(line)["event_type"]
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("sensor_recording_end", event_types)
            self.assertIn("logger_stopped", event_types)


if __name__ == "__main__":
    unittest.main()
