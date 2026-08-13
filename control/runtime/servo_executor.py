from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from control.runtime.action_scheduler import (
    ActionScheduler,
    ScheduledAction,
    scheduled_action_to_dict,
)
from control.runtime.data_logger import JsonlLogger
from control.runtime.discrete_actions import (
    ALL_SERVO_IDS,
    LEFT_FIN_SERVO_IDS,
    RIGHT_FIN_SERVO_IDS,
    TAIL_SERVO_IDS,
    RobotCalibration,
    action_parameters,
    sequence_servo_ids,
    smooth_segment,
)
from control.runtime.event_logger import EventLogger
from control.runtime.servo_state_tracker import ServoStateTracker
from control.safety import ServoLimits, servo_limits_from_config


class ServoController(Protocol):
    """ServoExecutor 使用的最小控制器接口。"""

    def limits_for(self, channel: int) -> ServoLimits: ...

    def write_angle(self, channel: int, angle: float) -> None: ...

    def stop_all(self, channels: list[int] | None = None) -> None: ...


class PwmWriteError(RuntimeError):
    """表示一批舵机目标未能全部成功写入。"""


@dataclass(frozen=True)
class ServoExecutionConfig:
    """不改变动作数学语义的执行器设置。"""

    command_hz: float = 50.0
    center_settle_s: float = 1.0
    safe_recenter_s: float = 2.0
    join_timeout_s: float = 10.0

    @property
    def period_ns(self) -> int:
        return max(1, int(round(1_000_000_000 / self.command_hz)))

    def to_dict(self) -> dict[str, float]:
        return {
            "command_hz": self.command_hz,
            "center_settle_s": self.center_settle_s,
            "safe_recenter_s": self.safe_recenter_s,
            "join_timeout_s": self.join_timeout_s,
        }


@dataclass
class ServoExecutorResult:
    mission_started: bool = False
    mission_finished: bool = False
    interrupted: bool = False
    safe_recentered: bool = False
    pwm_released: bool = False
    mission_start_t_ns: int | None = None
    mission_completion_t_ns: int | None = None
    reason: str = "not_started"
    error: str | None = None
    skipped_tick_count: int = 0


class DryRunServoController:
    """只执行同样的运行时限位检查，不导入或初始化真实 PCA9685。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._limits: dict[int, ServoLimits] = {}
        for raw in config.get("servo", {}).get("channels", []):
            if isinstance(raw, dict) and "channel" in raw:
                self._limits[int(raw["channel"])] = servo_limits_from_config(config, raw)
        self.last_angles: dict[int, float] = {}

    def limits_for(self, channel: int) -> ServoLimits:
        if int(channel) not in self._limits:
            raise ValueError(f"Dry-run channel {channel} 未配置。")
        return self._limits[int(channel)]

    def write_angle(self, channel: int, angle: float) -> None:
        target = self.limits_for(int(channel)).validate(float(angle))
        self.last_angles[int(channel)] = target

    def stop_all(self, channels: list[int] | None = None) -> None:
        selected = list(self.last_angles) if channels is None else [int(value) for value in channels]
        for channel in selected:
            self.last_angles.pop(channel, None)


class ServoExecutor:
    """唯一拥有 ServoKit/PCA9685 的长期线程，执行初始化、任务和安全回中。"""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        calibration: RobotCalibration,
        scheduler: ActionScheduler,
        tracker: ServoStateTracker,
        command_logger: JsonlLogger,
        event_logger: EventLogger,
        execution_config: ServoExecutionConfig,
        dry_run: bool,
        keep_pwm: bool,
        shutdown_event: threading.Event,
        motion_stop_event: threading.Event,
        failure_queue: queue.Queue[dict[str, Any]],
        controller_factory: Callable[[dict[str, Any], bool], ServoController] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wait_fn: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.scheduler = scheduler
        self.tracker = tracker
        self.command_logger = command_logger
        self.event_logger = event_logger
        self.execution_config = execution_config
        self.dry_run = bool(dry_run)
        self.keep_pwm = bool(keep_pwm)
        self.shutdown_event = shutdown_event
        self.motion_stop_event = motion_stop_event
        self.failure_queue = failure_queue
        self.controller_factory = controller_factory or create_servo_controller
        self.clock_ns = clock_ns
        self.wait_fn = wait_fn or (lambda event, seconds: event.wait(max(0.0, seconds)))
        self.result = ServoExecutorResult()
        self._thread: threading.Thread | None = None
        self._controller: ServoController | None = None
        self._mission_started_event = threading.Event()
        self._finished_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_guarded, name="servo-executor", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def wait_finished(self, timeout: float | None = None) -> bool:
        return self._finished_event.wait(timeout)

    def wait_mission_started(self, timeout: float | None = None) -> bool:
        return self._mission_started_event.wait(timeout)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self.result.reason = "background_exception"
            self.result.error = f"{type(exc).__name__}: {exc}"
            self.result.interrupted = True
            self.motion_stop_event.set()
            self.failure_queue.put(
                {
                    "source": "ServoExecutor",
                    "error": self.result.error,
                    "t_ns": self.clock_ns(),
                }
            )
        finally:
            self._finished_event.set()

    def _run(self) -> None:
        normal_completion = False
        try:
            self._controller = self.controller_factory(self.config, self.dry_run)
            self._write_batch(
                scheduled_t_ns=self.clock_ns(),
                reference_angles_deg=self.calibration.centers_deg,
                phase="safe_center_initialized",
                mission_start_t_ns=None,
            )
            if self.motion_stop_event.wait(max(0.0, self.execution_config.center_settle_s)):
                self.result.reason = "interrupted_before_mission"
                self.result.interrupted = True
                return

            initialized_t_ns = self.clock_ns()
            self.tracker.initialize(initialized_t_ns)
            self.event_logger.write(
                "servo_state_initialized",
                {
                    "feedback_available": False,
                    "position_estimation_mode": "assumed_perfect_tracking",
                    "centers_deg": self.calibration.centers_deg,
                    "position_note": "该角度没有硬件反馈，只是模型估计值。",
                },
                t_ns=initialized_t_ns,
            )

            mission_start_t_ns = self.clock_ns()
            self.result.mission_started = True
            self.result.mission_start_t_ns = mission_start_t_ns
            self.result.reason = "running"
            self.tracker.set_mission_start(mission_start_t_ns)
            transitions = self.scheduler.start(
                mission_start_t_ns,
                {
                    "tail": {
                        servo_id: self.calibration.servos[servo_id].center_deg
                        for servo_id in TAIL_SERVO_IDS
                    },
                    "left_fin": {
                        servo_id: self.calibration.servos[servo_id].center_deg
                        for servo_id in LEFT_FIN_SERVO_IDS
                    },
                    "right_fin": {
                        servo_id: self.calibration.servos[servo_id].center_deg
                        for servo_id in RIGHT_FIN_SERVO_IDS
                    },
                },
            )
            started_actions = [item.started for item in transitions if item.started is not None]
            self.tracker.record_actions_started(started_actions)
            self.event_logger.write(
                "mission_started",
                {"mission_start_t_ns": mission_start_t_ns},
                t_ns=mission_start_t_ns,
            )
            for action in started_actions:
                self._write_action_event(action, started=True)
            self._mission_started_event.set()

            if self.scheduler.all_finished():
                normal_completion = True
            else:
                self._run_mission_ticks(mission_start_t_ns)
                normal_completion = self.scheduler.all_finished()

            if normal_completion:
                completion_t_ns = self.scheduler.mission_completion_t_ns()
                self.result.mission_finished = True
                self.result.mission_completion_t_ns = completion_t_ns
                self.result.reason = "all_sequences_completed"
                self.event_logger.write(
                    "mission_finished",
                    {
                        "mission_start_t_ns": mission_start_t_ns,
                        "mission_completion_t_ns": completion_t_ns,
                    },
                    t_ns=completion_t_ns,
                )
            else:
                self.result.interrupted = True
                if self.result.reason == "running":
                    self.result.reason = "motion_stop_requested"
                self.event_logger.write(
                    "mission_interrupted",
                    {"reason": self.result.reason, "error": self.result.error},
                    t_ns=self.clock_ns(),
                )
        except PwmWriteError as exc:
            self.result.reason = "pwm_write_failed"
            self.result.error = str(exc)
            self.result.interrupted = True
            self.motion_stop_event.set()
            if self.result.mission_started:
                self.event_logger.write(
                    "mission_interrupted",
                    {"reason": self.result.reason, "error": self.result.error},
                    t_ns=self.clock_ns(),
                )
        finally:
            if self._controller is not None:
                self._safe_recenter_and_release(normal_completion)

    def _run_mission_ticks(self, executor_start_t_ns: int) -> None:
        period_ns = self.execution_config.period_ns
        tick_index = 0
        while not self.scheduler.all_finished() and not self.motion_stop_event.is_set():
            scheduled_t_ns = executor_start_t_ns + tick_index * period_ns
            now_ns = self.clock_ns()
            if now_ns < scheduled_t_ns:
                if not self._wait_until(scheduled_t_ns, self.motion_stop_event):
                    break
                now_ns = self.clock_ns()
            latest_due_index = max(0, (now_ns - executor_start_t_ns) // period_ns)
            if latest_due_index > tick_index:
                self.result.skipped_tick_count += int(latest_due_index - tick_index)
                tick_index = int(latest_due_index)
                scheduled_t_ns = executor_start_t_ns + tick_index * period_ns

            snapshot = self.tracker.query(scheduled_t_ns)
            reference = {
                int(servo_id): float(angle)
                for servo_id, angle in snapshot["reference_angles_deg"].items()
            }
            write_end_t_ns = self._write_batch(
                scheduled_t_ns=scheduled_t_ns,
                reference_angles_deg=reference,
                phase="mission",
                mission_start_t_ns=executor_start_t_ns,
                state_snapshot=snapshot,
            )
            due_sequences = self.scheduler.due_sequences(scheduled_t_ns)
            for sequence_name in due_sequences:
                final_angles = {
                    servo_id: reference[servo_id]
                    for servo_id in sequence_servo_ids(sequence_name)
                }
                transition = self.scheduler.complete(
                    sequence_name,
                    write_end_t_ns,
                    final_angles,
                )
                self.tracker.record_transition(transition)
                if transition.finished is not None:
                    self._write_action_event(transition.finished, started=False)
                if transition.started is not None:
                    self._write_action_event(transition.started, started=True)
            tick_index += 1

    def _write_batch(
        self,
        *,
        scheduled_t_ns: int,
        reference_angles_deg: dict[int, float],
        phase: str,
        mission_start_t_ns: int | None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> int:
        if self._controller is None:
            raise RuntimeError("舵机控制器尚未初始化。")
        by_channel = {
            self.calibration.servos[servo_id].channel: float(reference_angles_deg[servo_id])
            for servo_id in ALL_SERVO_IDS
        }
        # 先验证完整批次，防止已知越界时只写入一部分通道。
        for channel, angle in by_channel.items():
            self._controller.limits_for(channel).validate(angle)

        write_start_t_ns = self.clock_ns()
        successful_by_servo: dict[int, float] = {}
        per_servo_write_end_t_ns: dict[int, int] = {}
        error: str | None = None
        for channel in sorted(by_channel):
            servo_id = self.calibration.channel_to_servo_id[channel]
            angle = by_channel[channel]
            try:
                self._controller.write_angle(channel, angle)
                per_servo_end = self.clock_ns()
                successful_by_servo[servo_id] = angle
                per_servo_write_end_t_ns[servo_id] = per_servo_end
                self.tracker.record_successful_command(servo_id, per_servo_end, angle)
            except Exception as exc:
                error = f"servo_id={servo_id}, channel={channel}: {type(exc).__name__}: {exc}"
                break
        write_end_t_ns = self.clock_ns()
        snapshot = state_snapshot or self.tracker.query(scheduled_t_ns)
        command_record = {
            "scheduled_t_ns": int(scheduled_t_ns),
            "write_start_t_ns": write_start_t_ns,
            "write_end_t_ns": write_end_t_ns,
            "lateness_us": (write_start_t_ns - int(scheduled_t_ns)) / 1_000.0,
            "write_duration_us": (write_end_t_ns - write_start_t_ns) / 1_000.0,
            "mission_elapsed_s": None
            if mission_start_t_ns is None
            else (int(scheduled_t_ns) - mission_start_t_ns) / 1_000_000_000.0,
            "control_phase": phase,
            "ok": error is None,
            "error": error,
            "commands_deg": {str(channel): angle for channel, angle in by_channel.items()},
            "commands_by_servo_id_deg": {
                str(servo_id): reference_angles_deg[servo_id] for servo_id in ALL_SERVO_IDS
            },
            "successful_commands_by_servo_id_deg": {
                str(servo_id): angle for servo_id, angle in successful_by_servo.items()
            },
            "per_servo_write_end_t_ns": {
                str(servo_id): t_ns for servo_id, t_ns in per_servo_write_end_t_ns.items()
            },
            "reference_angles_deg": snapshot["reference_angles_deg"],
            "estimated_angles_deg": snapshot["estimated_angles_deg"],
            "position_estimation_mode": snapshot["estimation_mode"],
            "tail_action_index": snapshot["tail_action_index"],
            "left_fin_action_index": snapshot["left_fin_action_index"],
            "right_fin_action_index": snapshot["right_fin_action_index"],
            "tail_action_progress": snapshot["tail_action_progress"],
            "left_fin_action_progress": snapshot["left_fin_action_progress"],
            "right_fin_action_progress": snapshot["right_fin_action_progress"],
            "left_action_index": snapshot["left_fin_action_index"],
            "right_action_index": snapshot["right_fin_action_index"],
            "left_action_progress": snapshot["left_fin_action_progress"],
            "right_action_progress": snapshot["right_fin_action_progress"],
        }
        self.command_logger.write(command_record)
        if error is not None:
            self.event_logger.write(
                "pwm_write_failed",
                {
                    "scheduled_t_ns": int(scheduled_t_ns),
                    "write_start_t_ns": write_start_t_ns,
                    "write_end_t_ns": write_end_t_ns,
                    "successful_commands_by_servo_id_deg": successful_by_servo,
                    "error": error,
                },
                t_ns=write_end_t_ns,
            )
            raise PwmWriteError(error)
        return write_end_t_ns

    def _safe_recenter_and_release(self, normal_completion: bool) -> None:
        assert self._controller is not None
        start_t_ns = self.clock_ns()
        start_angles = self.tracker.latest_commanded_angles()
        centers = self.calibration.centers_deg
        duration_ns = max(1, int(round(self.execution_config.safe_recenter_s * 1_000_000_000)))
        self.tracker.record_reference_segment(
            "safe_recenter",
            start_t_ns,
            start_t_ns + duration_ns,
            start_angles,
            centers,
        )
        self.event_logger.write(
            "safe_recenter_started",
            {"start_angles_deg": start_angles, "target_angles_deg": centers},
            t_ns=start_t_ns,
        )
        recenter_error: str | None = None
        try:
            period_ns = self.execution_config.period_ns
            tick_count = max(1, (duration_ns + period_ns - 1) // period_ns)
            for tick_index in range(int(tick_count)):
                scheduled_t_ns = start_t_ns + tick_index * period_ns
                self._wait_until(scheduled_t_ns, self.shutdown_event)
                elapsed_ns = min(duration_ns, max(0, scheduled_t_ns - start_t_ns))
                progress = elapsed_ns / duration_ns
                reference = {
                    servo_id: smooth_segment(start_angles[servo_id], centers[servo_id], progress)
                    for servo_id in ALL_SERVO_IDS
                }
                self._write_batch(
                    scheduled_t_ns=scheduled_t_ns,
                    reference_angles_deg=reference,
                    phase="safe_recenter",
                    mission_start_t_ns=self.result.mission_start_t_ns,
                )
            endpoint_scheduled_t_ns = start_t_ns + duration_ns
            self._wait_until(endpoint_scheduled_t_ns, self.shutdown_event)
            endpoint_t_ns = self._write_batch(
                scheduled_t_ns=endpoint_scheduled_t_ns,
                reference_angles_deg=centers,
                phase="safe_recenter",
                mission_start_t_ns=self.result.mission_start_t_ns,
            )
            self.result.safe_recentered = True
            self.event_logger.write(
                "safe_recenter_finished",
                {"final_angles_deg": centers},
                t_ns=endpoint_t_ns,
            )
        except BaseException as exc:
            recenter_error = f"{type(exc).__name__}: {exc}"
            self.result.safe_recentered = False
            self.event_logger.write(
                "safe_recenter_failed",
                {"error": recenter_error},
                t_ns=self.clock_ns(),
            )

        configured_release = bool(
            self.config.get("safety", {}).get("servo", {}).get("release_pwm_after_tests", True)
        )
        must_release = recenter_error is not None or (configured_release and not self.keep_pwm)
        if must_release:
            try:
                self._controller.stop_all(
                    [self.calibration.servos[servo_id].channel for servo_id in ALL_SERVO_IDS]
                )
                self.result.pwm_released = True
            except BaseException as exc:
                release_error = f"{type(exc).__name__}: {exc}"
                self.result.error = (
                    release_error if self.result.error is None else f"{self.result.error}; {release_error}"
                )
        elif normal_completion:
            self.result.pwm_released = False

    def _write_action_event(self, action: ScheduledAction, *, started: bool) -> None:
        prefix = {
            "tail": "tail_action",
            "left_fin": "left_fin_action",
            "right_fin": "right_fin_action",
        }[action.sequence_name]
        data = scheduled_action_to_dict(action)
        data["action_parameters"] = action_parameters(action.action)
        event_type = f"{prefix}_{'started' if started else 'finished'}"
        event_t_ns = action.actual_start_t_ns if started else action.completion_t_ns
        self.event_logger.write(event_type, data, t_ns=event_t_ns)

    def _wait_until(self, deadline_t_ns: int, stop_event: threading.Event) -> bool:
        """用绝对单调截止时间等待，处理系统 wait 提前返回和停止信号。"""

        while True:
            remaining_ns = int(deadline_t_ns) - self.clock_ns()
            if remaining_ns <= 0:
                return True
            if self.wait_fn(stop_event, remaining_ns / 1_000_000_000.0):
                return False


def create_servo_controller(config: dict[str, Any], dry_run: bool) -> ServoController:
    """仅在真实执行分支中延迟导入 Raspberry Pi PCA9685 驱动。"""

    if dry_run:
        return DryRunServoController(config)
    from drivers.pca9685_servo import PCA9685ServoController

    return PCA9685ServoController(config)


def execution_config_from_robot(
    config: dict[str, Any],
    *,
    command_hz_override: float | None = None,
) -> ServoExecutionConfig:
    """读取并验证离散执行器配置，控制频率不属于动作定义。"""

    raw = config.get("servo", {}).get("discrete_executor", {})
    values = {
        "command_hz": float(raw.get("command_hz", 50.0)),
        "center_settle_s": float(raw.get("center_settle_s", 1.0)),
        "safe_recenter_s": float(raw.get("safe_recenter_s", 2.0)),
        "join_timeout_s": float(raw.get("join_timeout_s", 10.0)),
    }
    if command_hz_override is not None:
        values["command_hz"] = float(command_hz_override)
    for key, value in values.items():
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"servo.discrete_executor.{key} 必须是有限正数。")
    return ServoExecutionConfig(**values)
