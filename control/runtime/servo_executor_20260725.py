"""由唯一硬件线程合并并执行 20260725 三组动作的七路舵机命令。"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from control.runtime.action_scheduler_20260725 import (
    ActionScheduler,
    ScheduledAction,
    action_event_fields,
)
from control.runtime.data_logger import JsonlLogger
from control.runtime.discrete_absolute_actions_20260725 import (
    ALL_SERVO_IDS,
    RobotCalibration,
    smooth_motion,
)
from control.runtime.event_logger import EventLogger
from control.runtime.servo_state_tracker_20260725 import ServoStateTracker
from control.safety import ServoLimits, servo_limits_from_config


class ServoController(Protocol):
    """执行器线程内部使用的最小舵机控制器协议。"""

    def limits_for(self, channel: int) -> ServoLimits:
        """返回指定 PCA9685 通道对应的机械角度约束。"""

        ...

    def write_angle(self, channel: int, angle: float) -> None:
        """向指定通道写入已经通过限位验证的目标角。"""

        ...

    def stop_all(self, channels: list[int] | None = None) -> None:
        """停止全部或指定通道的 PWM 输出。"""

        ...


class PwmWriteError(RuntimeError):
    """统一七路命令批次中至少一路写入失败。"""


class RequiredLogWriteError(RuntimeError):
    """必需命令或事件未能进入异步日志队列。"""


@dataclass(frozen=True)
class ServoExecutionConfig:
    """控制循环配置；command_hz 不是动作参数。"""

    command_hz: float = 50.0
    center_settle_s: float = 1.0
    safe_recenter_s: float = 2.0
    join_timeout_s: float = 10.0
    initial_move_s: float = 2.0

    @property
    def period_ns(self) -> int:
        """把控制频率转换为至少一纳秒的整数控制周期。"""

        return max(1, int(round(1_000_000_000 / self.command_hz)))

    def to_dict(self) -> dict[str, float]:
        """返回可写入 metadata 的执行器配置快照。"""

        return {
            "command_hz": self.command_hz,
            "center_settle_s": self.center_settle_s,
            "safe_recenter_s": self.safe_recenter_s,
            "join_timeout_s": self.join_timeout_s,
            "initial_move_s": self.initial_move_s,
        }


@dataclass
class ServoExecutorResult:
    """主线程可读取的执行摘要，不包含任何“实测舵机到位”断言。"""

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
    pwm_write_count: int = 0
    unchanged_skip_count: int = 0


@dataclass(frozen=True)
class _BatchResult:
    """描述统一批次的结束时刻以及是否实际执行了 PWM 写入。"""

    write_end_t_ns: int
    pwm_write_performed: bool


class DryRunServoController:
    """保留完整限位和调用语义，但不导入 ServoKit 或访问 PCA9685。"""

    def __init__(self, config: dict[str, Any]) -> None:
        """从机器人配置构造纯内存通道限位和命令状态。"""

        self._limits = {
            int(raw["channel"]): servo_limits_from_config(config, raw)
            for raw in config.get("servo", {}).get("channels", [])
            if isinstance(raw, dict) and "channel" in raw
        }
        self.last_angles: dict[int, float] = {}
        self.write_thread_ids: list[int] = []

    def limits_for(self, channel: int) -> ServoLimits:
        """返回 dry-run 通道仍需遵守的真实配置机械限位。"""

        return self._limits[int(channel)]

    def write_angle(self, channel: int, angle: float) -> None:
        """模拟一次单通道硬件写入，并记录调用线程用于测试所有权。"""

        channel_id = int(channel)
        target = self.limits_for(channel_id).validate(float(angle))
        self.write_thread_ids.append(threading.get_ident())
        self.last_angles[channel_id] = target

    def stop_all(self, channels: list[int] | None = None) -> None:
        """模拟释放 PWM；不触碰任何硬件库。"""

        if channels is None:
            self.last_angles.clear()
            return
        for channel in channels:
            self.last_angles.pop(int(channel), None)


class ServoExecutor:
    """唯一创建并访问 PCA9685 控制器的长期线程。

    三条动作序列只提供逻辑状态。本线程每个绝对截止时间合并 1～7 号参考角，
    完成机械限位检查和统一写入，再分别提交已经到期的动作组。任何写入失败
    都是全局致命错误，因而不会推进索引或提前更新任一 previous theta。
    """

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
        controller_factory: Callable[
            [dict[str, Any], bool], ServoController
        ]
        | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        """注入调度、状态、日志和停止信号，并延迟到线程内创建控制器。"""

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
        self.result = ServoExecutorResult()
        self._thread: threading.Thread | None = None
        self._controller: ServoController | None = None
        self._mission_started = threading.Event()
        self._last_complete_command: dict[int, float] | None = None

    def start(self) -> None:
        """启动唯一硬件所有者线程。"""

        if self._thread and self._thread.is_alive():
            raise RuntimeError("ServoExecutor 已经启动。")
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="servo-executor-20260725",
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """有界等待唯一执行器线程结束。"""

        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        """报告执行器线程当前是否仍在运行。"""

        return bool(self._thread and self._thread.is_alive())

    def wait_mission_started(self, timeout: float | None = None) -> bool:
        """等待任务共用起点已经建立，超时则返回假。"""

        return self._mission_started.wait(timeout)

    def _run_guarded(self) -> None:
        """在线程最外层捕获异常，并通过失败队列传播给主线程。"""

        try:
            self._run()
        except BaseException as exc:
            self.result.reason = "background_exception"
            self.result.error = f"{type(exc).__name__}: {exc}"
            self.result.interrupted = True
            self.motion_stop_event.set()
            self._report_failure("ServoExecutor", self.result.error)

    def _run(self) -> None:
        """执行初始化、任务循环和无条件安全回中生命周期。"""

        normal_completion = False
        try:
            # 工厂调用位于本线程内部；主线程和三个逻辑序列均拿不到控制器。
            self._controller = self.controller_factory(self.config, self.dry_run)
            if not self._move_to_initial_pose():
                self.result.reason = "interrupted_before_mission"
                self.result.interrupted = True
                return
            if self.motion_stop_event.wait(self.execution_config.center_settle_s):
                self.result.reason = "interrupted_before_mission"
                self.result.interrupted = True
                return

            initialized_t_ns = self.clock_ns()
            self.tracker.initialize(initialized_t_ns)
            self._write_event_required(
                "servo_state_initialized",
                {
                    "feedback_available": False,
                    "position_estimation_mode": "assumed_perfect_tracking",
                    "reference_initial_angles_deg": self.calibration.initial_angles_deg,
                    "physical_position_confirmed": False,
                },
                t_ns=initialized_t_ns,
            )

            mission_start_t_ns = self.clock_ns()
            transitions = self.scheduler.start(mission_start_t_ns)
            started = [
                transition.started
                for transition in transitions
                if transition.started is not None
            ]

            # 三组首动作的起点姿态必须在同一任务起点至少成功写一次；即使
            # 它与初始化命令完全相同，也不能被 unchanged 优化跳过。在整批
            # 成功前不向 tracker 发布活动动作，避免传感器先看到一个实际上
            # 未成功开始的任务。
            with self.tracker.atomic_update():
                initial_snapshot = self._initial_start_snapshot(
                    mission_start_t_ns,
                    started,
                )
                self._write_batch(
                    mission_start_t_ns,
                    initial_snapshot["reference_angles_deg"],
                    control_phase="mission",
                    mission_start_t_ns=mission_start_t_ns,
                    snapshot=initial_snapshot,
                    force_write=True,
                )
                self.scheduler.mark_initial_starts_written()
                self.tracker.set_mission_start(mission_start_t_ns)
                self.tracker.record_actions_started(started)
            self.result.mission_started = True
            self.result.mission_start_t_ns = mission_start_t_ns
            self.result.reason = "running"
            self._write_event_required(
                "mission_started",
                {"mission_start_t_ns": mission_start_t_ns},
                t_ns=mission_start_t_ns,
            )
            for action in started:
                self._write_action_event(action, started=True)
            self._mission_started.set()

            self._mission_loop(mission_start_t_ns)
            normal_completion = (
                self.scheduler.all_finished()
                and not self.motion_stop_event.is_set()
            )
            if normal_completion:
                completion_t_ns = self.scheduler.mission_completion_t_ns()
                self.result.mission_finished = True
                self.result.mission_completion_t_ns = completion_t_ns
                self.result.reason = "all_sequences_completed"
                self._write_event_required(
                    "mission_finished",
                    {"mission_completion_t_ns": completion_t_ns},
                    t_ns=completion_t_ns,
                )
            else:
                self.result.interrupted = True
                self.result.reason = "motion_stop_requested"
                self._write_event_required(
                    "mission_interrupted",
                    {"reason": self.result.reason},
                    t_ns=self.clock_ns(),
                )
        except PwmWriteError as exc:
            self.result.reason = "pwm_write_failed"
            self.result.error = str(exc)
            self.result.interrupted = True
            self.motion_stop_event.set()
            self._write_event_nonfatal(
                "mission_interrupted",
                {"reason": self.result.reason, "error": self.result.error},
                t_ns=self.clock_ns(),
            )
            self._report_failure("ServoExecutor", self.result.error)
        except RequiredLogWriteError as exc:
            self.result.reason = "required_log_enqueue_failed"
            self.result.error = str(exc)
            self.result.interrupted = True
            self.motion_stop_event.set()
            self._write_event_nonfatal(
                "mission_interrupted",
                {"reason": self.result.reason, "error": self.result.error},
                t_ns=self.clock_ns(),
            )
            self._report_failure("ServoExecutor.logging", self.result.error)
        finally:
            if self._controller is not None:
                self._safe_recenter(normal_completion)

    def _mission_loop(self, executor_start_t_ns: int) -> None:
        """用绝对 deadline 执行，不使用 threading.Timer 或累计 sleep。"""

        tick_index = 1
        while (
            not self.scheduler.all_finished()
            and not self.motion_stop_event.is_set()
        ):
            scheduled_t_ns = (
                executor_start_t_ns + tick_index * self.execution_config.period_ns
            )
            self._wait_until(scheduled_t_ns, self.motion_stop_event)
            if self.motion_stop_event.is_set():
                break
            now_t_ns = self.clock_ns()
            latest_due_tick = max(
                0,
                (now_t_ns - executor_start_t_ns)
                // self.execution_config.period_ns,
            )
            if latest_due_tick > tick_index:
                self.result.skipped_tick_count += int(
                    latest_due_tick - tick_index
                )
                tick_index = int(latest_due_tick)
                scheduled_t_ns = (
                    executor_start_t_ns
                    + tick_index * self.execution_config.period_ns
                )

            with self.tracker.atomic_update():
                snapshot = self.tracker.get_servo_pose_at(
                    scheduled_t_ns
                ).to_dict()
                reference = {
                    int(servo_id): float(angle)
                    for servo_id, angle in snapshot[
                        "reference_angles_deg"
                    ].items()
                }
                due_sequences = self.scheduler.due_sequences(scheduled_t_ns)
                batch = self._write_batch(
                    scheduled_t_ns,
                    reference,
                    control_phase="mission",
                    mission_start_t_ns=executor_start_t_ns,
                    snapshot=snapshot,
                    # 到期动作的精确终点必须至少实际写入一次，不能因保持或
                    # 浮点完全相等而被 unchanged 优化跳过。
                    force_write=bool(due_sequences),
                )

                transitions = []
                start_next = not self.motion_stop_event.is_set()
                for sequence_name in due_sequences:
                    active = self.scheduler.active(sequence_name)
                    if active is None:
                        continue
                    final_angles = {
                        servo_id: reference[servo_id]
                        for servo_id in active.trajectory.target_angles_deg
                    }
                    transition = self.scheduler.complete(
                        sequence_name,
                        batch.write_end_t_ns,
                        final_angles,
                        endpoint_written=batch.pwm_write_performed,
                        start_next=start_next,
                    )
                    transitions.append(transition)

                # 同一七路命令可能同时完成多个动作组；统一发布历史，避免
                # 同步传感器线程看到组间混合状态或 write_end 后的旧状态。
                self.tracker.record_transitions(transitions)
            for transition in transitions:
                self._write_action_event(transition.finished, started=False)
                if transition.started is not None:
                    self._write_action_event(transition.started, started=True)
            tick_index += 1

    def _write_batch(
        self,
        scheduled_t_ns: int,
        reference_angles_deg: dict[int, float],
        *,
        control_phase: str,
        mission_start_t_ns: int | None,
        snapshot: dict[str, Any] | None = None,
        force_write: bool,
    ) -> _BatchResult:
        """验证、可选跳写、执行七路统一命令并异步记录 commands.jsonl。"""

        if self._controller is None:
            raise RuntimeError("舵机控制器尚未初始化。")
        reference = {
            int(servo_id): float(angle)
            for servo_id, angle in reference_angles_deg.items()
        }
        if set(reference) != set(ALL_SERVO_IDS):
            raise RuntimeError("统一舵机命令必须完整包含 1～7 号舵机。")

        # 预验证已经检查完整 mission；运行时仍在每批命令进入驱动前重复检查
        # top/bottom 物理端点与 min/max 的交集，防止未来 ActionProvider、
        # 跟踪器缺陷或浮点异常绕过机械约束。这里严格拒绝，绝不 clamp。
        for servo_id, angle in reference.items():
            self.calibration.servos[servo_id].validate(
                angle,
                "运行时参考命令",
            )
        by_channel = {
            self.calibration.servos[servo_id].channel: reference[servo_id]
            for servo_id in ALL_SERVO_IDS
        }
        for channel, angle in by_channel.items():
            self._controller.limits_for(channel).validate(angle)

        unchanged = (
            self._last_complete_command is not None
            and all(
                reference[servo_id] == self._last_complete_command[servo_id]
                for servo_id in ALL_SERVO_IDS
            )
        )
        if unchanged and not force_write:
            write_start_t_ns = self.clock_ns()
            write_end_t_ns = write_start_t_ns
            performed = False
            skip_reason = "unchanged_command"
            self.result.unchanged_skip_count += 1
        else:
            write_start_t_ns = self.clock_ns()
            successful_commands: dict[int, float] = {}
            try:
                for channel in sorted(by_channel):
                    servo_id = self.calibration.channel_to_servo_id[channel]
                    self._controller.write_angle(channel, by_channel[channel])
                    channel_write_end_t_ns = self.clock_ns()
                    successful_commands[servo_id] = by_channel[channel]
                    # 单路调用成功即属于该路的最近成功命令；整批稍后失败时
                    # 调度器仍不会提交动作，但安全回中可以使用真实已写子集。
                    self.tracker.record_successful_command(
                        servo_id,
                        channel_write_end_t_ns,
                        by_channel[channel],
                    )
            except Exception as exc:
                write_end_t_ns = self.clock_ns()
                error = (
                    f"servo_id={servo_id}, channel={channel}: "
                    f"{type(exc).__name__}: {exc}"
                )
                # 即使统一批次中途失败，也保存完整时序和已成功子集，便于
                # 审计“计划发送什么、写到哪一路失败”，同时仍不提交动作。
                try:
                    failed_record = self._build_command_record(
                        scheduled_t_ns=scheduled_t_ns,
                        reference=reference,
                        control_phase=control_phase,
                        mission_start_t_ns=mission_start_t_ns,
                        snapshot=snapshot,
                        write_start_t_ns=write_start_t_ns,
                        write_end_t_ns=write_end_t_ns,
                        performed=False,
                        skip_reason="pwm_write_failed",
                    )
                    failed_record.update(
                        {
                            "write_error": error,
                            "partial_pwm_writes_performed": bool(
                                successful_commands
                            ),
                            "successful_commands_by_servo_id_deg": dict(
                                successful_commands
                            ),
                        }
                    )
                    if not self.command_logger.write(failed_record):
                        self._report_failure(
                            "ServoExecutor.command_logger",
                            "PWM 失败批次的 commands.jsonl 记录未入队。",
                        )
                except Exception as log_exc:
                    self._report_failure(
                        "ServoExecutor.command_logger",
                        "记录 PWM 失败批次时异常："
                        f"{type(log_exc).__name__}: {log_exc}",
                    )
                self._write_event_nonfatal(
                    "pwm_write_failed",
                    {
                        "error": error,
                        "successful_commands_by_servo_id_deg": successful_commands,
                    },
                    t_ns=write_end_t_ns,
                )
                raise PwmWriteError(error) from exc
            write_end_t_ns = self.clock_ns()
            performed = True
            skip_reason = None
            self._last_complete_command = dict(reference)
            self.result.pwm_write_count += 1

        command_record = self._build_command_record(
            scheduled_t_ns=scheduled_t_ns,
            reference=reference,
            control_phase=control_phase,
            mission_start_t_ns=mission_start_t_ns,
            snapshot=snapshot,
            write_start_t_ns=write_start_t_ns,
            write_end_t_ns=write_end_t_ns,
            performed=performed,
            skip_reason=skip_reason,
        )
        self._write_command_record_nonfatal(command_record)
        return _BatchResult(write_end_t_ns, performed)

    def _write_command_record_nonfatal(
        self,
        command_record: dict[str, Any],
    ) -> None:
        """命令日志失败会停止后续动作，但不能撤销已经成功的物理终点。

        previous theta 的提交判据严格只有“规定时间经过”和“终点硬件写入
        成功”。因此这里报告全局日志故障并置停止信号，却仍让调用方完成
        当前到期动作的 transaction；调用方看到停止信号后不会启动下一动作。
        """

        try:
            accepted = self.command_logger.write(command_record)
            if accepted:
                return
            error = "commands.jsonl 异步队列已满，统一命令记录未入队。"
        except Exception as exc:
            error = (
                "commands.jsonl 入队异常："
                f"{type(exc).__name__}: {exc}"
            )
        self.motion_stop_event.set()
        self._report_failure("ServoExecutor.command_logger", error)

    def _build_command_record(
        self,
        *,
        scheduled_t_ns: int,
        reference: dict[int, float],
        control_phase: str,
        mission_start_t_ns: int | None,
        snapshot: dict[str, Any] | None,
        write_start_t_ns: int,
        write_end_t_ns: int,
        performed: bool,
        skip_reason: str | None,
    ) -> dict[str, Any]:
        """构造成功、跳写和部分失败批次共用的完整命令审计记录。"""

        pose = snapshot or self.tracker.get_servo_pose_at(
            int(scheduled_t_ns)
        ).to_dict()
        return {
            "t_ns": int(scheduled_t_ns),
            "scheduled_t_ns": int(scheduled_t_ns),
            "write_start_t_ns": write_start_t_ns,
            "write_end_t_ns": write_end_t_ns,
            "lateness_us": (write_start_t_ns - int(scheduled_t_ns)) / 1_000.0,
            "write_duration_us": (
                write_end_t_ns - write_start_t_ns
            )
            / 1_000.0,
            "pwm_write_performed": performed,
            "write_skip_reason": skip_reason,
            "mission_elapsed_s": (
                None
                if mission_start_t_ns is None
                else (int(scheduled_t_ns) - mission_start_t_ns)
                / 1_000_000_000.0
            ),
            "control_phase": control_phase,
            "reference_angles_deg": dict(reference),
            "commanded_angles_deg": self.tracker.latest_commanded_angles(),
            "estimated_angles_deg": dict(reference),
            "tail_action_index": pose.get("tail_action_index"),
            "left_action_index": pose.get("left_action_index"),
            "right_action_index": pose.get("right_action_index"),
            "previous_action1_theta": pose.get("previous_action1_theta"),
            "previous_action2_theta": pose.get("previous_action2_theta"),
            "previous_action3_theta": pose.get("previous_action3_theta"),
            "current_action1_theta": pose.get("current_action1_theta"),
            "current_action2_theta": pose.get("current_action2_theta"),
            "current_action3_theta": pose.get("current_action3_theta"),
            "tail_action_progress": pose.get("tail_action_progress"),
            "left_action_progress": pose.get("left_action_progress"),
            "right_action_progress": pose.get("right_action_progress"),
            "tail_action_duration_s": pose.get("tail_action_duration_s"),
            "left_action_duration_s": pose.get("left_action_duration_s"),
            "right_action_duration_s": pose.get("right_action_duration_s"),
            "left_tip_peak_angle_deg": pose.get("left_tip_peak_angle_deg"),
            "right_tip_peak_angle_deg": pose.get("right_tip_peak_angle_deg"),
            "feedback_available": False,
            "estimation_mode": "assumed_perfect_tracking",
        }

    def _write_action_event(
        self,
        action: ScheduledAction | None,
        *,
        started: bool,
    ) -> None:
        """把动作开始或结束状态写成对应动作组的必需事件。"""

        if action is None:
            return
        prefix = {
            "tail": "tail_action",
            "left_fin": "left_fin_action",
            "right_fin": "right_fin_action",
        }[action.sequence_name]
        event_t_ns = (
            action.actual_start_t_ns if started else action.completion_t_ns
        )
        self._write_event_required(
            prefix + ("_started" if started else "_finished"),
            action_event_fields(action),
            t_ns=event_t_ns,
        )

    def _initial_start_snapshot(
        self,
        mission_start_t_ns: int,
        started: list[ScheduledAction],
    ) -> dict[str, Any]:
        """构造尚未公开的三个首动作起点快照，供统一起点命令日志使用。"""

        snapshot = self.tracker.get_servo_pose_at(mission_start_t_ns).to_dict()
        snapshot["mission_elapsed_s"] = 0.0
        field_names = {
            "tail": (
                "tail_action_index",
                "previous_action1_theta",
                "current_action1_theta",
                "tail_action_progress",
                "tail_action_duration_s",
                None,
            ),
            "left_fin": (
                "left_action_index",
                "previous_action2_theta",
                "current_action2_theta",
                "left_action_progress",
                "left_action_duration_s",
                "left_tip_peak_angle_deg",
            ),
            "right_fin": (
                "right_action_index",
                "previous_action3_theta",
                "current_action3_theta",
                "right_action_progress",
                "right_action_duration_s",
                "right_tip_peak_angle_deg",
            ),
        }
        for action in started:
            (
                index_key,
                previous_key,
                current_key,
                progress_key,
                duration_key,
                peak_key,
            ) = field_names[action.sequence_name]
            snapshot[index_key] = action.action_index
            snapshot[previous_key] = action.previous_theta
            snapshot[current_key] = action.current_theta
            snapshot[progress_key] = 0.0
            snapshot[duration_key] = action.duration_s
            if peak_key is not None:
                snapshot[peak_key] = action.trajectory.tip_peak_angle_deg
        return snapshot

    def _write_event_required(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        t_ns: int | None,
    ) -> None:
        """将任务语义事件入队；队列饱和时停止后续动作而不静默丢失。"""

        try:
            accepted = self.event_logger.write(event_type, data, t_ns=t_ns)
        except Exception as exc:
            raise RequiredLogWriteError(
                f"events.jsonl 写入 {event_type} 异常："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not accepted:
            raise RequiredLogWriteError(
                f"events.jsonl 异步队列已满，事件 {event_type} 未入队。"
            )

    def _write_event_nonfatal(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        t_ns: int | None,
    ) -> None:
        """安全清理和原始故障路径尽力记事件，失败时改走主线程错误队列。"""

        try:
            accepted = self.event_logger.write(event_type, data, t_ns=t_ns)
            if accepted:
                return
            error = f"events.jsonl 异步队列已满，事件 {event_type} 未入队。"
        except Exception as exc:
            error = (
                f"events.jsonl 写入 {event_type} 异常："
                f"{type(exc).__name__}: {exc}"
            )
        self._report_failure("ServoExecutor.event_logger", error)

    def _move_to_initial_pose(self) -> bool:
        """按五次曲线进入 20260725 初始化姿态。

        系统没有位置反馈，因此这里不能知道舵机轴的真实起点。模型起点只能取
        robot.yaml 中各通道的配置中位；默认配置下它与新动作初始化姿态相同，
        轨迹就是精确常值保持。若显式配置了不同的 20260725 参考中位，则会从
        通道配置中位平滑过渡，并在起点和终点各强制成功写入一次。
        """

        if self._controller is None:
            raise RuntimeError("舵机控制器尚未初始化。")
        start_t_ns = self.clock_ns()
        origins = {
            servo_id: float(
                self._controller.limits_for(
                    self.calibration.servos[servo_id].channel
                ).center_angle
            )
            for servo_id in ALL_SERVO_IDS
        }
        targets = self.calibration.initial_angles_deg
        self._write_event_required(
            "initial_pose_move_started",
            {
                "model_start_angles_deg": origins,
                "target_angles_deg": targets,
                "feedback_available": False,
                "physical_start_position_confirmed": False,
            },
            t_ns=start_t_ns,
        )
        duration_ns = max(
            0,
            int(round(self.execution_config.initial_move_s * 1_000_000_000)),
        )
        if duration_ns == 0:
            self._write_batch(
                start_t_ns,
                targets,
                control_phase="initial_pose_move",
                mission_start_t_ns=None,
                force_write=True,
            )
        else:
            tick_index = 0
            while True:
                offset_ns = min(
                    duration_ns,
                    tick_index * self.execution_config.period_ns,
                )
                scheduled_t_ns = start_t_ns + offset_ns
                self._wait_until(scheduled_t_ns, self.motion_stop_event)
                if self.motion_stop_event.is_set():
                    return False
                reference = {
                    servo_id: smooth_motion(
                        origins[servo_id],
                        targets[servo_id],
                        offset_ns / 1_000_000_000.0,
                        duration_ns / 1_000_000_000.0,
                    )
                    for servo_id in ALL_SERVO_IDS
                }
                self._write_batch(
                    scheduled_t_ns,
                    reference,
                    control_phase="initial_pose_move",
                    mission_start_t_ns=None,
                    force_write=offset_ns in (0, duration_ns),
                )
                if offset_ns == duration_ns:
                    break
                tick_index += 1
        self._write_event_required(
            "initial_pose_move_finished",
            {
                "final_reference_angles_deg": targets,
                "feedback_available": False,
                "physical_position_confirmed": False,
            },
            t_ns=self.clock_ns(),
        )
        return True

    def _safe_recenter(self, normal_completion: bool) -> None:
        """从最近成功命令按五次平滑回到 20260725 七路参考中位。"""

        if self._controller is None:
            return
        recenter_error: str | None = None
        try:
            start_t_ns = self.clock_ns()
            origins = self.tracker.latest_commanded_angles()
            centers = self.calibration.initial_angles_deg
            duration_ns = max(
                0,
                int(
                    round(
                        self.execution_config.safe_recenter_s
                        * 1_000_000_000
                    )
                ),
            )
            self.tracker.record_reference_segment(
                "safe_recenter",
                start_t_ns,
                start_t_ns + duration_ns,
                origins,
                centers,
            )
            self._write_event_nonfatal(
                "safe_recenter_started",
                {
                    "start_angles_deg": origins,
                    "target_angles_deg": centers,
                    "mission_completed_normally": normal_completion,
                },
                t_ns=start_t_ns,
            )
            # 动作停止事件不能取消安全回中；主线程也会等执行器退出后才停止
            # 其他生产者。使用本地未置位事件可避免外部 shutdown_event 已
            # 置位时把原本数秒的轨迹瞬间突发写完。
            recenter_wait_event = threading.Event()
            if duration_ns > 0:
                tick_index = 0
                while True:
                    offset_ns = min(
                        duration_ns,
                        tick_index * self.execution_config.period_ns,
                    )
                    scheduled_t_ns = start_t_ns + offset_ns
                    self._wait_until(scheduled_t_ns, recenter_wait_event)
                    reference = {
                        servo_id: smooth_motion(
                            origins[servo_id],
                            centers[servo_id],
                            offset_ns / 1_000_000_000.0,
                            duration_ns / 1_000_000_000.0,
                        )
                        for servo_id in ALL_SERVO_IDS
                    }
                    self._write_batch(
                        scheduled_t_ns,
                        reference,
                        control_phase="safe_recenter",
                        mission_start_t_ns=self.result.mission_start_t_ns,
                        force_write=offset_ns == duration_ns,
                    )
                    if offset_ns == duration_ns:
                        break
                    tick_index += 1
            else:
                self._write_batch(
                    start_t_ns,
                    centers,
                    control_phase="safe_recenter",
                    mission_start_t_ns=self.result.mission_start_t_ns,
                    force_write=True,
                )
            self.result.safe_recentered = True
            self._write_event_nonfatal(
                "safe_recenter_finished",
                {"final_angles_deg": centers},
                t_ns=self.clock_ns(),
            )
        except BaseException as exc:
            recenter_error = f"{type(exc).__name__}: {exc}"
            self.result.safe_recentered = False
            self.result.reason = "safe_recenter_failed"
            self.result.error = (
                recenter_error
                if self.result.error is None
                else f"{self.result.error}; safe_recenter={recenter_error}"
            )
            self._write_event_nonfatal(
                "safe_recenter_failed",
                {"error": recenter_error},
                t_ns=self.clock_ns(),
            )
            self._report_failure("ServoExecutor.safe_recenter", recenter_error)
        finally:
            configured_release = bool(
                self.config.get("safety", {})
                .get("servo", {})
                .get("release_pwm_after_tests", True)
            )
            # 回中失败时忽略 --keep-pwm：未知的半程姿态继续带力比释放 PWM
            # 风险更高；正常完成时才遵从配置和用户显式保持选项。
            release_pwm = recenter_error is not None or (
                not self.keep_pwm and configured_release
            )
            if release_pwm:
                try:
                    self._controller.stop_all()
                    self.result.pwm_released = True
                except Exception as exc:
                    self._report_failure(
                        "ServoExecutor.stop_all",
                        f"{type(exc).__name__}: {exc}",
                    )

    def _wait_until(
        self,
        deadline_t_ns: int,
        stop_event: threading.Event,
    ) -> None:
        """短间隔等待绝对截止时间，并允许 stop_event 及时打断。"""

        while not stop_event.is_set():
            remaining_ns = int(deadline_t_ns) - self.clock_ns()
            if remaining_ns <= 0:
                return
            stop_event.wait(min(remaining_ns / 1_000_000_000.0, 0.05))

    def _report_failure(self, source: str, error: str) -> None:
        """把线程致命错误连同单调时间戳放入主线程失败队列。"""

        self.failure_queue.put(
            {"source": source, "error": error, "t_ns": self.clock_ns()}
        )


def create_servo_controller(
    config: dict[str, Any],
    dry_run: bool,
) -> ServoController:
    """延迟导入真实驱动，保证 dry-run 连 import 都不会触发。"""

    if dry_run:
        return DryRunServoController(config)
    from drivers.pca9685_servo import PCA9685ServoController

    return PCA9685ServoController(config)


def execution_config_from_robot(
    config: dict[str, Any],
) -> ServoExecutionConfig:
    """读取执行器周期配置并拒绝非有限或非法时间。"""

    raw = config.get("servo", {}).get("discrete_executor", {})
    result = ServoExecutionConfig(
        command_hz=float(raw.get("command_hz", 50.0)),
        center_settle_s=float(raw.get("center_settle_s", 1.0)),
        safe_recenter_s=float(raw.get("safe_recenter_s", 2.0)),
        join_timeout_s=float(raw.get("join_timeout_s", 10.0)),
        initial_move_s=float(raw.get("initial_move_s", 2.0)),
    )
    values = result.to_dict()
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("discrete_executor 参数必须是有限数。")
    if result.command_hz <= 0.0 or result.join_timeout_s <= 0.0:
        raise ValueError("command_hz 和 join_timeout_s 必须大于 0。")
    if (
        result.initial_move_s < 0.0
        or result.center_settle_s < 0.0
        or result.safe_recenter_s < 0.0
    ):
        raise ValueError(
            "initial_move_s、center_settle_s 和 safe_recenter_s 不能小于 0。"
        )
    if result.join_timeout_s <= result.safe_recenter_s:
        raise ValueError(
            "join_timeout_s 必须大于 safe_recenter_s，"
            "以保证 Ctrl+C 后安全回中线程能在关闭日志前完成。"
        )
    return result
