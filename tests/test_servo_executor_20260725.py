from __future__ import annotations

import builtins
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from control.runtime.action_scheduler_20260725 import ActionScheduler
from control.runtime.data_logger import JsonlLogger
from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    TailAction,
    build_calibration,
)
from control.runtime.event_logger import EventLogger
from control.runtime.servo_executor_20260725 import (
    ServoExecutionConfig,
    ServoExecutor,
    create_servo_controller,
)
from control.runtime.servo_state_tracker_20260725 import ServoStateTracker
from control.safety import load_robot_config, servo_limits_from_config


class FakeController:
    """记录线程和写入次数的无硬件测试控制器。"""

    def __init__(self, config, fail_at_call=None):
        self.limits = {
            int(raw["channel"]): servo_limits_from_config(config, raw)
            for raw in config["servo"]["channels"]
        }
        self.fail_at_call = fail_at_call
        self.calls = 0
        self.thread_ids = []
        self.stopped = False

    def limits_for(self, channel):
        return self.limits[int(channel)]

    def write_angle(self, channel, angle):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        if self.calls == self.fail_at_call:
            raise OSError("injected write failure")
        self.limits_for(channel).validate(float(angle))

    def stop_all(self, channels=None):
        self.thread_ids.append(threading.get_ident())
        self.stopped = True


class EndpointDroppingJsonlLogger(JsonlLogger):
    """只拒绝任务终点记录，用于隔离日志失败与物理终点提交语义。"""

    def write(self, obj):
        if (
            obj.get("control_phase") == "mission"
            and obj.get("tail_action_progress") == 1.0
        ):
            self.dropped_count += 1
            return False
        return super().write(obj)


class ServoExecutor20260725Tests(unittest.TestCase):
    """验证单线程所有权、保持跳写、终点强写和失败事务边界。"""

    def setUp(self) -> None:
        self.config = load_robot_config()
        self.calibration = build_calibration(self.config)

    def _run_executor(
        self,
        action,
        controller,
        *,
        drop_endpoint_command=False,
    ):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name)
        logger_class = (
            EndpointDroppingJsonlLogger
            if drop_endpoint_command
            else JsonlLogger
        )
        command_logger = logger_class(
            path / "commands.jsonl",
            flush_interval_s=0.01,
        )
        event_logger = EventLogger(
            path / "events.jsonl",
            flush_interval_s=0.01,
        )
        command_logger.start()
        event_logger.start()
        mission = DiscreteAbsoluteMission("executor", (action,), (), ())
        scheduler = ActionScheduler(mission, self.calibration)
        tracker = ServoStateTracker(self.calibration)
        failures = queue.Queue()
        executor = ServoExecutor(
            config=self.config,
            calibration=self.calibration,
            scheduler=scheduler,
            tracker=tracker,
            command_logger=command_logger,
            event_logger=event_logger,
            execution_config=ServoExecutionConfig(
                command_hz=200.0,
                center_settle_s=0.0,
                safe_recenter_s=0.0,
                join_timeout_s=2.0,
                initial_move_s=0.0,
            ),
            dry_run=True,
            keep_pwm=False,
            shutdown_event=threading.Event(),
            motion_stop_event=threading.Event(),
            failure_queue=failures,
            controller_factory=lambda _config, _dry_run: controller,
        )
        executor.start()
        executor.join(2.0)
        command_logger.stop(2.0)
        event_logger.stop()
        return (
            temporary,
            path,
            scheduler,
            tracker,
            executor,
            failures,
        )

    def test_hold_skips_unchanged_ticks_but_forces_start_and_endpoint(self) -> None:
        controller = FakeController(self.config)
        (
            temporary,
            path,
            scheduler,
            _tracker,
            executor,
            _failures,
        ) = self._run_executor(TailAction(0.0, 0.05), controller)
        try:
            self.assertTrue(executor.result.mission_finished)
            self.assertTrue(executor.result.safe_recentered)
            self.assertTrue(scheduler.all_finished())
            self.assertGreater(executor.result.unchanged_skip_count, 0)
            rows = [
                json.loads(line)
                for line in (path / "commands.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            mission_rows = [
                row for row in rows if row["control_phase"] == "mission"
            ]
            self.assertTrue(
                any(not row["pwm_write_performed"] for row in mission_rows)
            )
            endpoint = next(
                row
                for row in mission_rows
                if row["tail_action_progress"] == 1.0
            )
            self.assertTrue(endpoint["pwm_write_performed"])
            self.assertIsNone(endpoint["write_skip_reason"])
            skipped = next(
                row
                for row in mission_rows
                if not row["pwm_write_performed"]
            )
            self.assertEqual(skipped["write_skip_reason"], "unchanged_command")
        finally:
            temporary.cleanup()

    def test_all_controller_access_comes_from_one_executor_thread(self) -> None:
        controller = FakeController(self.config)
        temporary, _, _, _, executor, _ = self._run_executor(
            TailAction(1.0, 0.03),
            controller,
        )
        try:
            self.assertTrue(executor.result.mission_finished)
            self.assertEqual(len(set(controller.thread_ids)), 1)
            self.assertNotEqual(controller.thread_ids[0], threading.get_ident())
            self.assertTrue(controller.stopped)
        finally:
            temporary.cleanup()

    def test_failed_batch_does_not_complete_or_commit_previous_theta(self) -> None:
        # 7 路初始化 + 7 路任务起点之后，第 15 次调用在任务周期中失败。
        controller = FakeController(self.config, fail_at_call=15)
        temporary, path, scheduler, _tracker, executor, failures = self._run_executor(
            TailAction(5.0, 0.05),
            controller,
        )
        try:
            self.assertFalse(scheduler.all_finished())
            self.assertEqual(scheduler.previous_thetas["tail"], 0.0)
            self.assertEqual(executor.result.reason, "pwm_write_failed")
            self.assertFalse(executor.result.mission_finished)
            self.assertTrue(controller.stopped)
            self.assertFalse(failures.empty())
            records = [
                json.loads(line)
                for line in (path / "commands.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            failed = next(
                row
                for row in records
                if row["write_skip_reason"] == "pwm_write_failed"
            )
            self.assertFalse(failed["pwm_write_performed"])
            self.assertIn("write_error", failed)
            self.assertIn("successful_commands_by_servo_id_deg", failed)
        finally:
            temporary.cleanup()

    def test_command_log_drop_does_not_undo_successful_endpoint_commit(self) -> None:
        """日志入队失败会停止任务，但不能增加 previous theta 的提交条件。"""

        controller = FakeController(self.config)
        (
            temporary,
            _path,
            scheduler,
            _tracker,
            executor,
            failures,
        ) = self._run_executor(
            TailAction(5.0, 0.03),
            controller,
            drop_endpoint_command=True,
        )
        try:
            self.assertEqual(scheduler.previous_thetas["tail"], 5.0)
            self.assertTrue(scheduler.all_finished())
            self.assertFalse(executor.result.mission_finished)
            self.assertEqual(executor.result.reason, "motion_stop_requested")
            self.assertTrue(executor.result.safe_recentered)
            self.assertFalse(failures.empty())
        finally:
            temporary.cleanup()

    def test_commands_and_events_contain_required_state_fields(self) -> None:
        controller = FakeController(self.config)
        temporary, path, _, _, executor, _ = self._run_executor(
            TailAction(1.0, 0.03),
            controller,
        )
        try:
            self.assertTrue(executor.result.mission_finished)
            commands = [
                json.loads(line)
                for line in (path / "commands.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            mission_row = next(
                row for row in commands if row["control_phase"] == "mission"
            )
            for key in (
                "scheduled_t_ns",
                "write_start_t_ns",
                "write_end_t_ns",
                "lateness_us",
                "write_duration_us",
                "pwm_write_performed",
                "write_skip_reason",
                "reference_angles_deg",
                "commanded_angles_deg",
                "estimated_angles_deg",
                "previous_action1_theta",
                "current_action1_theta",
                "tail_action_duration_s",
            ):
                self.assertIn(key, mission_row)
            events = [
                json.loads(line)
                for line in (path / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            event_types = {row["event_type"] for row in events}
            self.assertTrue(
                {
                    "servo_state_initialized",
                    "mission_started",
                    "tail_action_started",
                    "tail_action_finished",
                    "mission_finished",
                    "safe_recenter_started",
                    "safe_recenter_finished",
                }.issubset(event_types)
            )
            finished = next(
                row
                for row in events
                if row["event_type"] == "tail_action_finished"
            )["data"]
            self.assertTrue(finished["endpoint_written"])
            self.assertEqual(
                finished["completion_reason"],
                "time_elapsed_and_endpoint_written",
            )
            self.assertEqual(finished["previous_theta_before_commit"], 0.0)
            self.assertEqual(finished["previous_theta_after_commit"], 1.0)
        finally:
            temporary.cleanup()

    def test_dry_run_factory_never_imports_real_driver(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "drivers.pca9685_servo":
                raise AssertionError("dry-run 不得导入真实 PCA9685 驱动")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            controller = create_servo_controller(self.config, True)
        self.assertEqual(controller.last_angles, {})


if __name__ == "__main__":
    unittest.main()
