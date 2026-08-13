from __future__ import annotations

import builtins
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from control.runtime.action_scheduler import ActionScheduler
from control.runtime.data_logger import JsonlLogger
from control.runtime.discrete_actions import (
    DiscreteMission,
    FinAction,
    TailAction,
    build_robot_calibration,
)
from control.runtime.event_logger import EventLogger
from control.runtime.servo_executor import (
    ServoExecutionConfig,
    ServoExecutor,
    create_servo_controller,
)
from control.runtime.servo_state_tracker import ServoStateTracker
from control.safety import load_robot_config, servo_limits_from_config


class FakeController:
    def __init__(
        self,
        config,
        *,
        fail_at_call: int | None = None,
        fatal_at_call: int | None = None,
    ) -> None:
        self.limits = {
            int(raw["channel"]): servo_limits_from_config(config, raw)
            for raw in config["servo"]["channels"]
        }
        self.fail_at_call = fail_at_call
        self.fatal_at_call = fatal_at_call
        self.call_count = 0
        self.last_angles = {}
        self.stopped = False

    def limits_for(self, channel):
        return self.limits[int(channel)]

    def write_angle(self, channel, angle):
        self.call_count += 1
        if self.fatal_at_call == self.call_count:
            raise KeyboardInterrupt("injected background exception")
        if self.fail_at_call == self.call_count:
            raise OSError("injected PWM failure")
        self.last_angles[int(channel)] = self.limits_for(channel).validate(float(angle))

    def stop_all(self, channels=None):
        self.stopped = True


class ServoExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_robot_config()
        self.calibration = build_robot_calibration(self.config)

    def _run_executor(self, mission, *, controller, motion_event=None):
        temporary = tempfile.TemporaryDirectory()
        log_dir = Path(temporary.name)
        command_logger = JsonlLogger(log_dir / "commands.jsonl", flush_interval_s=0.1)
        event_logger = EventLogger(log_dir / "events.jsonl", flush_interval_s=0.1)
        command_logger.start()
        event_logger.start()
        tracker = ServoStateTracker(self.calibration)
        scheduler = ActionScheduler(mission)
        shutdown_event = threading.Event()
        motion_stop_event = motion_event or threading.Event()
        failures = queue.Queue()
        executor = ServoExecutor(
            config=self.config,
            calibration=self.calibration,
            scheduler=scheduler,
            tracker=tracker,
            command_logger=command_logger,
            event_logger=event_logger,
            execution_config=ServoExecutionConfig(
                command_hz=500.0,
                center_settle_s=0.001,
                safe_recenter_s=0.004,
                join_timeout_s=2.0,
            ),
            dry_run=True,
            keep_pwm=False,
            shutdown_event=shutdown_event,
            motion_stop_event=motion_stop_event,
            failure_queue=failures,
            controller_factory=lambda _config, _dry_run: controller,
        )
        executor.start()
        return temporary, log_dir, command_logger, event_logger, scheduler, executor

    def test_parallel_sequences_finish_and_log_timing_fields(self) -> None:
        mission = DiscreteMission(
            "executor",
            (TailAction(0.006, 1.0), TailAction(0.004, -1.0)),
            (FinAction(0.008, 1, 0.1),),
            (FinAction(0.012, -1, 0.1),),
        )
        controller = FakeController(self.config)
        temporary, log_dir, command_logger, event_logger, scheduler, executor = self._run_executor(
            mission, controller=controller
        )
        try:
            executor.join(2.0)
            self.assertFalse(executor.is_alive())
            self.assertTrue(executor.result.mission_finished)
            self.assertTrue(executor.result.safe_recentered)
            self.assertTrue(scheduler.all_finished())
            command_logger.stop()
            event_logger.stop()
            commands = [
                json.loads(line)
                for line in (log_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            mission_commands = [entry for entry in commands if entry["control_phase"] == "mission"]
            self.assertGreater(len(mission_commands), 1)
            self.assertTrue(all(entry["lateness_us"] >= 0.0 for entry in commands))
            for key in (
                "scheduled_t_ns",
                "write_start_t_ns",
                "write_end_t_ns",
                "lateness_us",
                "write_duration_us",
                "reference_angles_deg",
                "estimated_angles_deg",
            ):
                self.assertIn(key, mission_commands[0])
            events = [
                json.loads(line)
                for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            event_types = [entry["event_type"] for entry in events]
            self.assertIn("tail_action_finished", event_types)
            self.assertIn("left_fin_action_finished", event_types)
            self.assertIn("right_fin_action_finished", event_types)
        finally:
            command_logger.stop()
            event_logger.stop()
            temporary.cleanup()

    def test_endpoint_write_failure_prevents_completion_and_forces_cleanup(self) -> None:
        mission = DiscreteMission("failure", (TailAction(0.002, 1.0),), (), ())
        controller = FakeController(self.config, fail_at_call=15)
        temporary, _, command_logger, event_logger, scheduler, executor = self._run_executor(
            mission, controller=controller
        )
        try:
            executor.join(2.0)
            self.assertFalse(executor.is_alive())
            self.assertFalse(scheduler.all_finished())
            self.assertEqual(executor.result.reason, "pwm_write_failed")
            self.assertTrue(executor.result.safe_recentered)
            self.assertTrue(controller.stopped)
        finally:
            command_logger.stop()
            event_logger.stop()
            temporary.cleanup()

    def test_motion_stop_interrupts_and_recenters(self) -> None:
        mission = DiscreteMission("interrupt", (TailAction(0.2, 2.0),), (), ())
        controller = FakeController(self.config)
        motion_event = threading.Event()
        temporary, _, command_logger, event_logger, _, executor = self._run_executor(
            mission, controller=controller, motion_event=motion_event
        )
        try:
            self.assertTrue(executor.wait_mission_started(1.0))
            motion_event.set()
            executor.join(2.0)
            self.assertFalse(executor.result.mission_finished)
            self.assertTrue(executor.result.interrupted)
            self.assertTrue(executor.result.safe_recentered)
        finally:
            command_logger.stop()
            event_logger.stop()
            temporary.cleanup()

    def test_unhandled_background_exception_is_reported_and_thread_exits(self) -> None:
        mission = DiscreteMission("background", (TailAction(0.002, 1.0),), (), ())
        controller = FakeController(self.config, fatal_at_call=15)
        temporary, _, command_logger, event_logger, _, executor = self._run_executor(
            mission, controller=controller
        )
        try:
            executor.join(2.0)
            self.assertFalse(executor.is_alive())
            self.assertEqual(executor.result.reason, "background_exception")
            self.assertTrue(executor.result.safe_recentered)
            self.assertIn("KeyboardInterrupt", executor.result.error)
        finally:
            command_logger.stop()
            event_logger.stop()
            temporary.cleanup()

    def test_dry_run_factory_never_imports_real_driver(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "drivers.pca9685_servo":
                raise AssertionError("dry-run imported real PCA9685 driver")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            controller = create_servo_controller(self.config, True)
        self.assertEqual(type(controller).__name__, "DryRunServoController")


if __name__ == "__main__":
    unittest.main()
