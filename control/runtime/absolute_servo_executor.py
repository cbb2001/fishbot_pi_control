from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from control.runtime.absolute_action_scheduler import (
    AbsoluteActionScheduler, ScheduledAbsoluteAction, action_event_fields,
)
from control.runtime.absolute_discrete_actions import (
    ALL_SERVO_IDS, LEFT_FIN_SERVO_IDS, RIGHT_FIN_SERVO_IDS, TAIL_SERVO_IDS,
    AbsoluteRobotCalibration, quintic_smoothstep, sequence_servo_ids,
)
from control.runtime.absolute_servo_state_tracker import ServoStateTracker
from control.runtime.data_logger import JsonlLogger
from control.runtime.event_logger import EventLogger
from control.safety import ServoLimits, servo_limits_from_config


class ServoController(Protocol):
    def limits_for(self, channel: int) -> ServoLimits: ...
    def write_angle(self, channel: int, angle: float) -> None: ...
    def stop_all(self, channels: list[int] | None = None) -> None: ...


class PwmWriteError(RuntimeError):
    """完整批次中至少一路 PCA9685 写入失败。"""


@dataclass(frozen=True)
class ServoExecutionConfig:
    command_hz: float = 50.0
    center_settle_s: float = 1.0
    safe_recenter_s: float = 2.0
    join_timeout_s: float = 10.0

    @property
    def period_ns(self) -> int:
        return max(1, int(round(1e9 / self.command_hz)))

    def to_dict(self) -> dict[str, float]:
        return dict(command_hz=self.command_hz, center_settle_s=self.center_settle_s,
                    safe_recenter_s=self.safe_recenter_s, join_timeout_s=self.join_timeout_s)


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
    """执行同样的机械限位检查，但不会导入真实硬件驱动。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._limits = {int(raw["channel"]): servo_limits_from_config(config, raw)
                        for raw in config.get("servo", {}).get("channels", [])
                        if isinstance(raw, dict) and "channel" in raw}
        self.last_angles: dict[int, float] = {}

    def limits_for(self, channel: int) -> ServoLimits:
        return self._limits[int(channel)]

    def write_angle(self, channel: int, angle: float) -> None:
        self.last_angles[int(channel)] = self.limits_for(channel).validate(float(angle))

    def stop_all(self, channels: list[int] | None = None) -> None:
        self.last_angles.clear()


class ServoExecutor:
    """唯一允许访问 ServoKit/PCA9685 的长期线程。

    三个动作序列只是调度状态，绝不各自创建 I2C 线程；任何异常均通过停止事件
    和 failure_queue 传播，随后本线程尽力安全回中。
    """

    def __init__(self, *, config: dict[str, Any], calibration: AbsoluteRobotCalibration,
                 scheduler: AbsoluteActionScheduler, tracker: ServoStateTracker,
                 command_logger: JsonlLogger, event_logger: EventLogger,
                 execution_config: ServoExecutionConfig, dry_run: bool, keep_pwm: bool,
                 shutdown_event: threading.Event, motion_stop_event: threading.Event,
                 failure_queue: queue.Queue[dict[str, Any]],
                 controller_factory: Callable[[dict[str, Any], bool], ServoController] | None = None,
                 clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self.config, self.calibration, self.scheduler, self.tracker = config, calibration, scheduler, tracker
        self.command_logger, self.event_logger = command_logger, event_logger
        self.execution_config, self.dry_run, self.keep_pwm = execution_config, dry_run, keep_pwm
        self.shutdown_event, self.motion_stop_event, self.failure_queue = shutdown_event, motion_stop_event, failure_queue
        self.controller_factory = controller_factory or create_servo_controller
        self.clock_ns = clock_ns
        self.result = ServoExecutorResult()
        self._thread: threading.Thread | None = None
        self._controller: ServoController | None = None
        self._mission_started = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_guarded, name="absolute-servo-executor", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def wait_mission_started(self, timeout: float | None = None) -> bool:
        return self._mission_started.wait(timeout)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self.result.reason = "background_exception"
            self.result.error = f"{type(exc).__name__}: {exc}"
            self.result.interrupted = True
            self.motion_stop_event.set()
            self.failure_queue.put({"source": "ServoExecutor", "error": self.result.error,
                                    "t_ns": self.clock_ns()})

    def _run(self) -> None:
        normal = False
        try:
            self._controller = self.controller_factory(self.config, self.dry_run)
            self._write_batch(self.clock_ns(), self.calibration.centers_deg, "safe_center_initialized", None)
            if self.motion_stop_event.wait(max(0.0, self.execution_config.center_settle_s)):
                self.result.reason, self.result.interrupted = "interrupted_before_mission", True
                return
            initialized_t_ns = self.clock_ns()
            self.tracker.initialize(initialized_t_ns)
            self.event_logger.write("servo_state_initialized", {
                "feedback_available": False, "position_estimation_mode": "assumed_perfect_tracking",
                "centers_deg": self.calibration.centers_deg}, t_ns=initialized_t_ns)
            mission_start = self.clock_ns()
            self.result.mission_started, self.result.mission_start_t_ns = True, mission_start
            self.result.reason = "running"
            self.tracker.set_mission_start(mission_start)
            transitions = self.scheduler.start(mission_start, {
                "tail": {i: self.calibration.servos[i].center_deg for i in TAIL_SERVO_IDS},
                "left_fin": {i: self.calibration.servos[i].center_deg for i in LEFT_FIN_SERVO_IDS},
                "right_fin": {i: self.calibration.servos[i].center_deg for i in RIGHT_FIN_SERVO_IDS}})
            started = [x.started for x in transitions if x.started is not None]
            self.tracker.record_actions_started(started)
            self.event_logger.write("mission_started", {"mission_start_t_ns": mission_start}, t_ns=mission_start)
            for action in started:
                self._action_event(action, True)
            self._mission_started.set()
            self._mission_loop(mission_start)
            normal = self.scheduler.all_finished()
            if normal:
                completion = self.scheduler.mission_completion_t_ns()
                self.result.mission_finished = True
                self.result.mission_completion_t_ns = completion
                self.result.reason = "all_sequences_completed"
                self.event_logger.write("mission_finished", {"mission_completion_t_ns": completion}, t_ns=completion)
            else:
                self.result.interrupted = True
                self.result.reason = "motion_stop_requested"
                self.event_logger.write("mission_interrupted", {"reason": self.result.reason}, t_ns=self.clock_ns())
        except PwmWriteError as exc:
            self.result.reason, self.result.error, self.result.interrupted = "pwm_write_failed", str(exc), True
            self.motion_stop_event.set()
            self.event_logger.write("mission_interrupted", {"reason": self.result.reason,
                                                              "error": self.result.error}, t_ns=self.clock_ns())
        finally:
            if self._controller is not None:
                self._safe_recenter(normal)

    def _mission_loop(self, executor_start_t_ns: int) -> None:
        tick = 0
        while not self.scheduler.all_finished() and not self.motion_stop_event.is_set():
            scheduled = executor_start_t_ns + tick * self.execution_config.period_ns
            self._wait_until(scheduled, self.motion_stop_event)
            now = self.clock_ns()
            latest = max(0, (now - executor_start_t_ns) // self.execution_config.period_ns)
            if latest > tick:
                self.result.skipped_tick_count += int(latest - tick)
                tick = int(latest)
                scheduled = executor_start_t_ns + tick * self.execution_config.period_ns
            snapshot = self.tracker.get_servo_pose_at(scheduled).to_dict()
            reference = {int(k): float(v) for k, v in snapshot["reference_angles_deg"].items()}
            write_end = self._write_batch(scheduled, reference, "mission", executor_start_t_ns, snapshot)
            # 当前批次已成功完整写入，因此到期序列的终点也已成功写入。
            for name in self.scheduler.due_sequences(scheduled):
                active = self.scheduler.active(name)
                assert active is not None
                endpoint = active.trajectory.evaluate(active.trajectory.duration_s)
                transition = self.scheduler.complete(name, write_end, endpoint)
                self.tracker.record_transition(transition)
                self._action_event(transition.finished, False)
                if transition.started is not None:
                    self._action_event(transition.started, True)
            tick += 1

    def _write_batch(self, scheduled_t_ns: int, reference: dict[int, float], phase: str,
                     mission_start_t_ns: int | None, snapshot: dict[str, Any] | None = None) -> int:
        assert self._controller is not None
        by_channel = {self.calibration.servos[i].channel: reference[i] for i in ALL_SERVO_IDS}
        for channel, angle in by_channel.items():
            self._controller.limits_for(channel).validate(angle)
        write_start = self.clock_ns()
        successful: dict[int, float] = {}
        try:
            for channel in sorted(by_channel):
                servo_id = self.calibration.channel_to_servo_id[channel]
                self._controller.write_angle(channel, by_channel[channel])
                end = self.clock_ns()
                successful[servo_id] = by_channel[channel]
                self.tracker.record_successful_command(servo_id, end, by_channel[channel])
        except Exception as exc:
            write_end = self.clock_ns()
            error = f"servo_id={servo_id}, channel={channel}: {type(exc).__name__}: {exc}"
            self.event_logger.write("pwm_write_failed", {"error": error,
                                                          "successful_commands_by_servo_id_deg": successful},
                                    t_ns=write_end)
            raise PwmWriteError(error) from exc
        write_end = self.clock_ns()
        pose = snapshot or self.tracker.get_servo_pose_at(scheduled_t_ns).to_dict()
        self.command_logger.write({"scheduled_t_ns": scheduled_t_ns, "write_start_t_ns": write_start,
            "write_end_t_ns": write_end, "lateness_us": (write_start - scheduled_t_ns) / 1e3,
            "write_duration_us": (write_end - write_start) / 1e3,
            "mission_elapsed_s": None if mission_start_t_ns is None else (scheduled_t_ns - mission_start_t_ns) / 1e9,
            "control_phase": phase, "reference_angles_deg": pose["reference_angles_deg"],
            "commanded_angles_deg": self.tracker.latest_commanded_angles(),
            "estimated_angles_deg": pose["estimated_angles_deg"],
            "tail_action_index": pose["tail_action_index"], "left_action_index": pose["left_action_index"],
            "right_action_index": pose["right_action_index"], "tail_action_progress": pose["tail_action_progress"],
            "left_action_progress": pose["left_action_progress"], "right_action_progress": pose["right_action_progress"],
            "tail_servo_durations_s": pose["tail_servo_durations_s"],
            "left_action_duration_s": pose["left_action_duration_s"],
            "right_action_duration_s": pose["right_action_duration_s"]})
        return write_end

    def _action_event(self, action: ScheduledAbsoluteAction | None, started: bool) -> None:
        if action is None:
            return
        prefix = {"tail": "tail_action", "left_fin": "left_fin_action",
                  "right_fin": "right_fin_action"}[action.sequence_name]
        t_ns = action.actual_start_t_ns if started else action.completion_t_ns
        self.event_logger.write(prefix + ("_started" if started else "_finished"),
                                action_event_fields(action), t_ns=t_ns)

    def _safe_recenter(self, normal: bool) -> None:
        assert self._controller is not None
        start = self.clock_ns()
        origins, centers = self.tracker.latest_commanded_angles(), self.calibration.centers_deg
        self.event_logger.write("safe_recenter_started", {"start_angles_deg": origins,
                                                            "target_angles_deg": centers}, t_ns=start)
        duration_ns = max(0, int(round(self.execution_config.safe_recenter_s * 1e9)))
        try:
            ticks = max(1, duration_ns // self.execution_config.period_ns)
            for tick in range(int(ticks) + 1):
                scheduled = start + min(duration_ns, tick * self.execution_config.period_ns)
                self._wait_until(scheduled, self.shutdown_event)
                p = 1.0 if duration_ns == 0 else (scheduled - start) / duration_ns
                factor = quintic_smoothstep(p)
                reference = {i: origins[i] + (centers[i] - origins[i]) * factor for i in ALL_SERVO_IDS}
                self._write_batch(scheduled, reference, "safe_recenter", self.result.mission_start_t_ns)
            self._write_batch(start + duration_ns, centers, "safe_recenter", self.result.mission_start_t_ns)
            self.result.safe_recentered = True
            self.event_logger.write("safe_recenter_finished", {"final_angles_deg": centers}, t_ns=self.clock_ns())
        except BaseException as exc:
            self.event_logger.write("safe_recenter_failed", {"error": f"{type(exc).__name__}: {exc}"},
                                    t_ns=self.clock_ns())
        finally:
            release = not self.keep_pwm and bool(self.config.get("safety", {}).get("servo", {})
                                                  .get("release_pwm_after_tests", True))
            if release:
                try:
                    self._controller.stop_all()
                    self.result.pwm_released = True
                except Exception:
                    pass

    def _wait_until(self, deadline_ns: int, event: threading.Event) -> None:
        while not event.is_set():
            remaining = deadline_ns - self.clock_ns()
            if remaining <= 0:
                return
            event.wait(min(remaining / 1e9, 0.05))


def create_servo_controller(config: dict[str, Any], dry_run: bool) -> ServoController:
    """dry-run 分支在返回前不会触碰真实 PCA9685 模块。"""

    if dry_run:
        return DryRunServoController(config)
    from drivers.pca9685_servo import PCA9685ServoController
    return PCA9685ServoController(config)


def execution_config_from_robot(config: dict[str, Any]) -> ServoExecutionConfig:
    raw = config.get("servo", {}).get("discrete_executor", {})
    result = ServoExecutionConfig(float(raw.get("command_hz", 50)),
                                  float(raw.get("center_settle_s", 1)),
                                  float(raw.get("safe_recenter_s", 2)),
                                  float(raw.get("join_timeout_s", 10)))
    if min(result.command_hz, result.center_settle_s, result.safe_recenter_s,
           result.join_timeout_s) <= 0:
        raise ValueError("discrete_executor 参数必须大于 0。")
    return result
