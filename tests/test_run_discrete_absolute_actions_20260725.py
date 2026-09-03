from __future__ import annotations

import copy
import json
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_discrete_absolute_actions_20260725 as action_script  # noqa: E402
from control.runtime.discrete_absolute_actions_20260725 import (  # noqa: E402
    MissionValidationError,
    build_calibration,
    validate_mission,
)
from control.runtime.manual_action_provider_20260725 import (  # noqa: E402
    ManualSequenceProvider,
)
from control.runtime.sensor_sync_worker import SensorSyncWorker  # noqa: E402
from control.safety import load_robot_config  # noqa: E402


class RunDiscreteAbsoluteActions20260725Tests(unittest.TestCase):
    """覆盖 CLI 安全门、严格新 YAML 和完整 mock/dry-run 产物。"""

    def _write_yaml(self, path: Path, data) -> None:
        import yaml

        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def test_default_start_delay_is_five_seconds(self) -> None:
        """未显式传参时，在日志和硬件初始化前等待 5 秒。"""

        args = action_script.parse_args(["--mission", "mission.yaml"])
        self.assertEqual(args.start_delay_s, 5.0)

    def test_provider_accepts_only_new_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            valid = base / "valid.yaml"
            self._write_yaml(
                valid,
                {
                    "name": "valid",
                    "tail_actions": [{"theta": 0.0, "t": 1.0}],
                    "left_fin_actions": [
                        {"theta": 121.0, "t": 1.0, "b1": 0, "b2": 1}
                    ],
                    "right_fin_actions": [],
                },
            )
            mission = ManualSequenceProvider(valid).load()
            self.assertEqual(mission.tail_actions[0].theta, 0.0)
            self.assertEqual(mission.left_fin_actions[0].b2, 1)

            legacy = base / "legacy.yaml"
            self._write_yaml(
                legacy,
                {
                    "name": "legacy",
                    "tail_actions": [
                        {
                            "theta1": 85,
                            "theta2": 95,
                            "theta3": 95,
                            "v1": 1,
                            "v2": 1,
                            "v3": 1,
                        }
                    ],
                    "left_fin_actions": [],
                    "right_fin_actions": [],
                },
            )
            with self.assertRaises(MissionValidationError):
                ManualSequenceProvider(legacy).load()

    def test_shipped_mission_sync_scope_and_readme_commands(self) -> None:
        mission_path = (
            PROJECT_ROOT
            / "missions"
            / "discrete_absolute_actions_20260725_example.yaml"
        )
        mission = ManualSequenceProvider(mission_path).load()
        validate_mission(
            mission,
            build_calibration(load_robot_config()),
        )
        self.assertEqual(
            mission.name,
            "discrete_absolute_actions_20260725_action4_down_tail_fins_up_A",
        )
        self.assertEqual(len(mission.tail_actions), 141)
        self.assertEqual(len(mission.left_fin_actions), 140)
        self.assertEqual(len(mission.right_fin_actions), 140)
        self.assertEqual(mission.tail_actions[0].theta, 15.0)
        self.assertEqual(mission.left_fin_actions[1].b2, 1)
        self.assertEqual(mission.right_fin_actions[1].b2, -1)

        sync_text = (
            PROJECT_ROOT / "codex_pi_workflow" / "sync_to_pi.ps1"
        ).read_text(encoding="utf-8")
        items_block = sync_text.split("$itemsToSync = @(", 1)[1].split(")", 1)[0]
        self.assertIn('"missions"', items_block)

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for fragment in (
            "scripts/run_discrete_absolute_actions_20260725.py",
            "missions/discrete_absolute_actions_20260725_example.yaml",
            "--dry-run --mock-sensors",
            "--confirm MOVE",
            r".\codex_pi_workflow\run_on_pi.ps1",
        ):
            self.assertIn(fragment, readme)

    def test_provider_rejects_boolean_b_fields_even_though_bool_is_int(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            self._write_yaml(
                path,
                {
                    "name": "bad",
                    "tail_actions": [],
                    "left_fin_actions": [
                        {"theta": 121, "t": 1, "b1": True, "b2": 1}
                    ],
                    "right_fin_actions": [],
                },
            )
            with self.assertRaises(MissionValidationError):
                ManualSequenceProvider(path).load()

    def test_provider_rejects_numeric_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric-string.yaml"
            self._write_yaml(
                path,
                {
                    "name": "numeric-string",
                    "tail_actions": [{"theta": "0.0", "t": 1.0}],
                    "left_fin_actions": [],
                    "right_fin_actions": [],
                },
            )
            with self.assertRaises(MissionValidationError):
                ManualSequenceProvider(path).load()

    def test_cli_has_no_old_speed_frequency_or_travel_parameters(self) -> None:
        args = action_script.parse_args(
            [
                "--mission",
                "mission.yaml",
                "--dry-run",
                "--mock-sensors",
            ]
        )
        for forbidden in (
            "frequency",
            "phase",
            "speed",
            "v",
            "travel_ratio",
            "max_test_amplitude_deg",
        ):
            self.assertFalse(hasattr(args, forbidden))

    def test_real_motion_without_confirmation_stops_before_log_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = copy.deepcopy(load_robot_config())
            config["logging"]["base_dir"] = str(base / "logs")
            config_path = base / "robot.yaml"
            mission_path = base / "mission.yaml"
            self._write_yaml(config_path, config)
            self._write_yaml(
                mission_path,
                {
                    "name": "confirm",
                    "tail_actions": [{"theta": 0, "t": 0.01}],
                    "left_fin_actions": [],
                    "right_fin_actions": [],
                },
            )
            result = action_script.main(
                [
                    "--config",
                    str(config_path),
                    "--mission",
                    str(mission_path),
                    "--start-delay-s",
                    "0",
                ]
            )
            self.assertEqual(result, 2)
            self.assertFalse((base / "logs").exists())

    def test_complete_mock_dry_run_saves_mission_metadata_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = copy.deepcopy(load_robot_config())
            config["logging"]["base_dir"] = str(base / "logs")
            config["runtime"]["synchronized_sample_hz"] = 200
            config["servo"]["discrete_executor"].update(
                {
                    "command_hz": 1000,
                    "initial_move_s": 0,
                    "center_settle_s": 0,
                    "safe_recenter_s": 0,
                    "join_timeout_s": 2,
                }
            )
            config_path = base / "robot.yaml"
            mission_path = base / "mission.yaml"
            mission_data = {
                "name": "end_to_end",
                "tail_actions": [
                    {"theta": 0.0, "t": 0.008},
                    {"theta": 0.0, "t": 0.006},
                ],
                "left_fin_actions": [
                    {"theta": 121.0, "t": 0.005, "b1": 1, "b2": 1}
                ],
                "right_fin_actions": [
                    {"theta": 143.0, "t": 0.011, "b1": 0, "b2": -1}
                ],
            }
            self._write_yaml(config_path, config)
            self._write_yaml(mission_path, mission_data)

            result = action_script.main(
                [
                    "--config",
                    str(config_path),
                    "--mission",
                    str(mission_path),
                    "--dry-run",
                    "--mock-sensors",
                    "--start-delay-s",
                    "0",
                ]
            )
            self.assertEqual(result, 0)
            run_dirs = list((base / "logs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            log_dir = run_dirs[0]
            self.assertEqual(
                (log_dir / action_script.SAVED_MISSION_FILENAME).read_text(
                    encoding="utf-8"
                ),
                mission_path.read_text(encoding="utf-8"),
            )
            for filename in (
                "metadata.yaml",
                "commands.jsonl",
                "events.jsonl",
                "synchronized_sensors.jsonl",
            ):
                self.assertTrue((log_dir / filename).exists(), filename)

            import yaml

            metadata = yaml.safe_load(
                (log_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["manual_action_sequence"], mission_data)
            self.assertEqual(metadata["script_version"], "20260725-v1")
            self.assertEqual(
                metadata["action_reference"]["left_fin"][
                    "tip_reference_span_deg"
                ],
                90.0,
            )
            commands = [
                json.loads(line)
                for line in (log_dir / "commands.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(any(row["control_phase"] == "mission" for row in commands))
            events = [
                json.loads(line)
                for line in (log_dir / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            event_types = {row["event_type"] for row in events}
            self.assertIn("mission_loaded", event_types)
            self.assertIn("mission_validated", event_types)
            self.assertIn("mission_finished", event_types)
            self.assertIn("logger_stopped", event_types)

    def test_runtime_failure_is_propagated_to_motion_stop(self) -> None:
        class ExplodingSynchronizer:
            buffers = {}

            @staticmethod
            def build(_scheduled_t_ns):
                raise RuntimeError("injected background failure")

        failures = queue.Queue()
        worker = SensorSyncWorker(
            synchronizer=ExplodingSynchronizer(),
            tracker=object(),
            sync_logger=object(),
            raw_loggers=object(),
            sample_hz=100.0,
            shutdown_event=threading.Event(),
            failure_queue=failures,
        )
        worker.start()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertIn("RuntimeError: injected background failure", worker.last_error)

        reported = []
        seen = set()
        stop = threading.Event()
        action_script._drain_failure_queue(
            failures,
            reported,
            seen,
            stop,
        )
        self.assertTrue(stop.is_set())
        self.assertEqual(reported[0]["source"], "SensorSyncWorker")
        self.assertIn("injected background failure", reported[0]["error"])

        # 相同错误只记录一次，避免主循环每 50 ms 重复膨胀。
        action_script._record_runtime_failure(
            {
                "source": reported[0]["source"],
                "error": reported[0]["error"],
                "t_ns": reported[0]["t_ns"] + 1,
            },
            reported,
            seen,
            stop,
        )
        self.assertEqual(len(reported), 1)

    def test_ctrl_c_stops_motion_and_runs_bounded_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = copy.deepcopy(load_robot_config())
            config["logging"]["base_dir"] = str(base / "logs")
            config["runtime"]["synchronized_sample_hz"] = 200
            config["servo"]["discrete_executor"].update(
                {
                    "command_hz": 500,
                    "initial_move_s": 0,
                    "center_settle_s": 0,
                    "safe_recenter_s": 0,
                    "join_timeout_s": 2,
                }
            )
            config_path = base / "robot.yaml"
            mission_path = base / "mission.yaml"
            self._write_yaml(config_path, config)
            self._write_yaml(
                mission_path,
                {
                    "name": "ctrl_c_cleanup",
                    "tail_actions": [{"theta": 10.0, "t": 0.5}],
                    "left_fin_actions": [],
                    "right_fin_actions": [],
                },
            )

            actual_executor_class = action_script.ServoExecutor
            actual_drain = action_script._drain_failure_queue
            executor_holder = {}
            drain_calls = 0

            def capture_executor(*args, **kwargs):
                executor = actual_executor_class(*args, **kwargs)
                executor_holder["executor"] = executor
                return executor

            def interrupt_main_once(*args, **kwargs):
                nonlocal drain_calls
                drain_calls += 1
                if drain_calls == 1:
                    executor = executor_holder["executor"]
                    if not executor.wait_mission_started(1.0):
                        raise AssertionError("ServoExecutor 未在时限内启动 mission")
                    raise KeyboardInterrupt()
                return actual_drain(*args, **kwargs)

            with mock.patch.object(
                action_script,
                "ServoExecutor",
                side_effect=capture_executor,
            ), mock.patch.object(
                action_script,
                "_drain_failure_queue",
                side_effect=interrupt_main_once,
            ):
                result = action_script.main(
                    [
                        "--config",
                        str(config_path),
                        "--mission",
                        str(mission_path),
                        "--dry-run",
                        "--mock-sensors",
                        "--start-delay-s",
                        "0",
                    ]
                )

            self.assertEqual(result, 130)
            executor = executor_holder["executor"]
            self.assertFalse(executor.is_alive())
            self.assertTrue(executor.motion_stop_event.is_set())
            self.assertTrue(executor.result.interrupted)
            self.assertTrue(executor.result.safe_recentered)

            run_dirs = list((base / "logs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            events = [
                json.loads(line)
                for line in (run_dirs[0] / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            event_types = {row["event_type"] for row in events}
            self.assertTrue(
                {
                    "mission_started",
                    "mission_interrupted",
                    "safe_recenter_started",
                    "safe_recenter_finished",
                    "logger_stopped",
                }.issubset(event_types)
            )
            logger_stopped = next(
                row for row in events if row["event_type"] == "logger_stopped"
            )
            self.assertTrue(logger_stopped["data"]["ctrl_c"])


if __name__ == "__main__":
    unittest.main()
