from __future__ import annotations

import builtins
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from control.runtime.absolute_action_scheduler import AbsoluteActionScheduler
from control.runtime.absolute_discrete_actions import AbsoluteMission, TailAbsoluteAction, build_absolute_calibration
from control.runtime.absolute_servo_executor import ServoExecutionConfig, ServoExecutor, create_servo_controller
from control.runtime.absolute_servo_state_tracker import ServoStateTracker
from control.runtime.data_logger import JsonlLogger
from control.runtime.event_logger import EventLogger
from control.safety import load_robot_config, servo_limits_from_config


class FakeController:
    def __init__(self, config, fail_at=None):
        self.limits = {int(x["channel"]): servo_limits_from_config(config, x) for x in config["servo"]["channels"]}
        self.fail_at, self.calls, self.stopped = fail_at, 0, False
    def limits_for(self, channel): return self.limits[channel]
    def write_angle(self, channel, angle):
        self.calls += 1
        if self.calls == self.fail_at: raise OSError("injected endpoint failure")
        self.limits_for(channel).validate(angle)
    def stop_all(self, channels=None): self.stopped = True


class AbsoluteServoExecutorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_robot_config(); self.cal = build_absolute_calibration(self.config)

    def _start(self, controller):
        temp = tempfile.TemporaryDirectory(); path = Path(temp.name)
        command = JsonlLogger(path / "commands.jsonl", flush_interval_s=.01)
        events = EventLogger(path / "events.jsonl", flush_interval_s=.01)
        command.start(); events.start()
        mission = AbsoluteMission("executor", (TailAbsoluteAction(80.1, 90, 90, 100, 100, 100),), (), ())
        scheduler = AbsoluteActionScheduler(mission, self.cal); tracker = ServoStateTracker(self.cal)
        executor = ServoExecutor(config=self.config, calibration=self.cal, scheduler=scheduler,
            tracker=tracker, command_logger=command, event_logger=events,
            execution_config=ServoExecutionConfig(500, .001, .004, 2), dry_run=True,
            keep_pwm=False, shutdown_event=threading.Event(), motion_stop_event=threading.Event(),
            failure_queue=queue.Queue(), controller_factory=lambda _c, _d: controller)
        executor.start()
        return temp, path, command, events, scheduler, executor

    def test_endpoint_write_completes_and_logs_required_fields(self):
        temp, path, command, events, scheduler, executor = self._start(FakeController(self.config))
        try:
            executor.join(2); self.assertTrue(executor.result.mission_finished)
            self.assertTrue(executor.result.safe_recentered); self.assertTrue(scheduler.all_finished())
            command.stop(); events.stop()
            rows = [json.loads(x) for x in (path / "commands.jsonl").read_text().splitlines()]
            mission = next(x for x in rows if x["control_phase"] == "mission")
            for key in ("scheduled_t_ns", "write_start_t_ns", "write_end_t_ns", "lateness_us",
                        "write_duration_us", "reference_angles_deg", "commanded_angles_deg",
                        "estimated_angles_deg", "tail_servo_durations_s"):
                self.assertIn(key, mission)
        finally:
            command.stop(); events.stop(); temp.cleanup()

    def test_failed_write_never_completes_sequence(self):
        controller = FakeController(self.config, fail_at=15)
        temp, _, command, events, scheduler, executor = self._start(controller)
        try:
            executor.join(2); self.assertFalse(scheduler.all_finished())
            self.assertEqual(executor.result.reason, "pwm_write_failed"); self.assertTrue(controller.stopped)
        finally:
            command.stop(); events.stop(); temp.cleanup()

    def test_dry_run_does_not_import_real_driver(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "drivers.pca9685_servo": raise AssertionError("真实驱动被导入")
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=guarded):
            create_servo_controller(self.config, True)


if __name__ == "__main__": unittest.main()
