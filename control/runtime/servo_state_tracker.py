from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable

from control.runtime.action_scheduler import ScheduledAction, SchedulerTransition
from control.runtime.discrete_actions import (
    LEFT_FIN_SERVO_IDS,
    RIGHT_FIN_SERVO_IDS,
    SEQUENCE_NAMES,
    TAIL_SERVO_IDS,
    FinAction,
    RobotCalibration,
    TailAction,
    evaluate_fin_action,
    evaluate_tail_action,
    smooth_segment,
)


ESTIMATION_MODE = "assumed_perfect_tracking"


@dataclass
class TrackedAction:
    sequence_name: str
    action_index: int
    action: TailAction | FinAction
    planned_start_t_ns: int
    planned_end_t_ns: int
    actual_start_t_ns: int
    start_angles_deg: dict[int, float]
    completion_t_ns: int | None = None
    final_angles_deg: dict[int, float] | None = None
    completion_reason: str | None = None


@dataclass(frozen=True)
class ReferenceSegment:
    """动作任务之外的初始化或安全回中参考轨迹。"""

    phase: str
    start_t_ns: int
    end_t_ns: int
    start_angles_deg: dict[int, float]
    end_angles_deg: dict[int, float]


class ServoStateTracker:
    """保存模型参考、成功命令和动作提交历史；所有角度均无硬件反馈。"""

    def __init__(self, calibration: RobotCalibration) -> None:
        self.calibration = calibration
        self._lock = RLock()
        self._initialized_t_ns: int | None = None
        self._mission_start_t_ns: int | None = None
        self._actions: dict[str, list[TrackedAction]] = {name: [] for name in SEQUENCE_NAMES}
        self._command_times: dict[int, list[int]] = {
            servo_id: [] for servo_id in calibration.servos
        }
        self._command_angles: dict[int, list[float]] = {
            servo_id: [] for servo_id in calibration.servos
        }
        self._reference_segments: list[ReferenceSegment] = []
        self._tail_cumulative_angles_deg = {
            servo_id: calibration.servos[servo_id].center_deg for servo_id in TAIL_SERVO_IDS
        }

    def initialize(self, initialized_t_ns: int) -> None:
        """在七路中位命令成功且稳定后提交已知模型状态。"""

        with self._lock:
            self._initialized_t_ns = int(initialized_t_ns)

    def set_mission_start(self, mission_start_t_ns: int) -> None:
        with self._lock:
            self._mission_start_t_ns = int(mission_start_t_ns)

    def record_actions_started(self, actions: Iterable[ScheduledAction]) -> None:
        with self._lock:
            for action in actions:
                self._actions[action.sequence_name].append(_tracked(action))

    def record_transition(self, transition: SchedulerTransition) -> None:
        """原子记录完成和下一动作启动，避免查询看到中间空档。"""

        with self._lock:
            if transition.finished is not None:
                tracked = self._find_action(
                    transition.finished.sequence_name,
                    transition.finished.action_index,
                )
                tracked.completion_t_ns = transition.finished.completion_t_ns
                tracked.final_angles_deg = dict(transition.finished.final_angles_deg or {})
                tracked.completion_reason = transition.finished.completion_reason
                if transition.finished.sequence_name == "tail":
                    self._tail_cumulative_angles_deg = dict(tracked.final_angles_deg)
            if transition.started is not None:
                self._actions[transition.started.sequence_name].append(_tracked(transition.started))

    def record_successful_command(self, servo_id: int, write_end_t_ns: int, angle_deg: float) -> None:
        """保存单路最近一次真正成功返回的 PCA9685 目标命令。"""

        with self._lock:
            times = self._command_times[int(servo_id)]
            if times and int(write_end_t_ns) < times[-1]:
                raise ValueError("成功命令时间戳必须单调不减。")
            times.append(int(write_end_t_ns))
            self._command_angles[int(servo_id)].append(float(angle_deg))

    def record_reference_segment(
        self,
        phase: str,
        start_t_ns: int,
        end_t_ns: int,
        start_angles_deg: dict[int, float],
        end_angles_deg: dict[int, float],
    ) -> None:
        with self._lock:
            self._reference_segments.append(
                ReferenceSegment(
                    phase=phase,
                    start_t_ns=int(start_t_ns),
                    end_t_ns=max(int(start_t_ns), int(end_t_ns)),
                    start_angles_deg=dict(start_angles_deg),
                    end_angles_deg=dict(end_angles_deg),
                )
            )

    def latest_commanded_angles(self) -> dict[int, float]:
        with self._lock:
            result: dict[int, float] = {}
            for servo_id, values in self._command_angles.items():
                if values:
                    result[servo_id] = values[-1]
                else:
                    result[servo_id] = self.calibration.servos[servo_id].center_deg
            return result

    def query(self, sample_t_ns: int) -> dict[str, Any]:
        """按指定样本自己的单调时间戳查询七路模型与最近成功命令。"""

        t_ns = int(sample_t_ns)
        with self._lock:
            reference, indices, progress, phase = self._reference_at(t_ns)
            commanded = {
                servo_id: self._command_at(servo_id, t_ns)
                for servo_id in sorted(self.calibration.servos)
            }
            initialized = self._initialized_t_ns is not None and t_ns >= self._initialized_t_ns
            result = {
                "sample_t_ns": t_ns,
                "state_initialized": initialized,
                "feedback_available": False,
                "position_note": "该角度没有硬件反馈，只是模型估计值。",
                "estimation_mode": ESTIMATION_MODE,
                "control_phase": phase,
                "reference_angles_deg": dict(reference),
                "commanded_angles_deg": commanded,
                "estimated_angles_deg": dict(reference),
                "tail_action_index": indices["tail"],
                "left_fin_action_index": indices["left_fin"],
                "right_fin_action_index": indices["right_fin"],
                "tail_action_progress": progress["tail"],
                "left_fin_action_progress": progress["left_fin"],
                "right_fin_action_progress": progress["right_fin"],
                "tail_cumulative_angles_deg": self._tail_cumulative_at(t_ns),
            }
            # 保留简洁内部名称，同时提供同步日志契约中的显式 servo_* 和左右别名。
            result["servo_reference_angles_deg"] = dict(reference)
            result["servo_commanded_angles_deg"] = dict(commanded)
            result["servo_estimated_angles_deg"] = dict(reference)
            result["left_action_index"] = result["left_fin_action_index"]
            result["right_action_index"] = result["right_fin_action_index"]
            result["left_action_progress"] = result["left_fin_action_progress"]
            result["right_action_progress"] = result["right_fin_action_progress"]
            return result

    def _reference_at(
        self,
        t_ns: int,
    ) -> tuple[dict[int, float], dict[str, int | None], dict[str, float | None], str]:
        reference = self.calibration.centers_deg
        indices: dict[str, int | None] = {name: None for name in SEQUENCE_NAMES}
        progress: dict[str, float | None] = {name: None for name in SEQUENCE_NAMES}
        phase = "pre_mission" if self._mission_start_t_ns is None or t_ns < self._mission_start_t_ns else "mission"

        for sequence_name in SEQUENCE_NAMES:
            sequence_reference, active, local_progress = self._sequence_reference(sequence_name, t_ns)
            reference.update(sequence_reference)
            if active is not None:
                indices[sequence_name] = active.action_index
                progress[sequence_name] = local_progress

        for segment in self._reference_segments:
            if t_ns < segment.start_t_ns:
                continue
            duration_ns = max(1, segment.end_t_ns - segment.start_t_ns)
            p = min(1.0, max(0.0, (t_ns - segment.start_t_ns) / duration_ns))
            reference = {
                servo_id: smooth_segment(
                    segment.start_angles_deg[servo_id],
                    segment.end_angles_deg[servo_id],
                    p,
                )
                for servo_id in self.calibration.servos
            }
            indices = {name: None for name in SEQUENCE_NAMES}
            progress = {name: None for name in SEQUENCE_NAMES}
            phase = segment.phase
        return reference, indices, progress, phase

    def _sequence_reference(
        self,
        sequence_name: str,
        t_ns: int,
    ) -> tuple[dict[int, float], TrackedAction | None, float | None]:
        servo_ids = _servo_ids(sequence_name)
        held = {servo_id: self.calibration.servos[servo_id].center_deg for servo_id in servo_ids}
        active: TrackedAction | None = None
        for action in self._actions[sequence_name]:
            if t_ns < action.actual_start_t_ns:
                break
            if action.completion_t_ns is None or t_ns < action.completion_t_ns:
                active = action
                break
            if action.final_angles_deg:
                held = dict(action.final_angles_deg)

        if active is None:
            return held, None, None
        elapsed_s = max(0.0, (t_ns - active.actual_start_t_ns) / 1_000_000_000.0)
        progress = min(1.0, elapsed_s / active.action.t_s)
        if sequence_name == "tail":
            assert isinstance(active.action, TailAction)
            values = evaluate_tail_action(
                active.action,
                elapsed_s,
                active.start_angles_deg,
                self.calibration,
            )
        else:
            assert isinstance(active.action, FinAction)
            values = evaluate_fin_action(
                active.action,
                elapsed_s,
                servo_ids,
                self.calibration,
            )
        return values, active, progress

    def _command_at(self, servo_id: int, t_ns: int) -> float | None:
        times = self._command_times[servo_id]
        index = bisect_right(times, t_ns) - 1
        if index < 0:
            return None
        return self._command_angles[servo_id][index]

    def _tail_cumulative_at(self, t_ns: int) -> dict[int, float]:
        committed = {
            servo_id: self.calibration.servos[servo_id].center_deg
            for servo_id in TAIL_SERVO_IDS
        }
        for action in self._actions["tail"]:
            if action.completion_t_ns is None or action.completion_t_ns > t_ns:
                break
            if action.final_angles_deg:
                committed = dict(action.final_angles_deg)
        return committed

    def _find_action(self, sequence_name: str, action_index: int) -> TrackedAction:
        for action in reversed(self._actions[sequence_name]):
            if action.action_index == action_index:
                return action
        raise KeyError(f"找不到 {sequence_name}[{action_index}] 的跟踪记录。")


def _tracked(action: ScheduledAction) -> TrackedAction:
    return TrackedAction(
        sequence_name=action.sequence_name,
        action_index=action.action_index,
        action=action.action,
        planned_start_t_ns=action.planned_start_t_ns,
        planned_end_t_ns=action.planned_end_t_ns,
        actual_start_t_ns=action.actual_start_t_ns,
        start_angles_deg=dict(action.start_angles_deg),
        completion_t_ns=action.completion_t_ns,
        final_angles_deg=None if action.final_angles_deg is None else dict(action.final_angles_deg),
        completion_reason=action.completion_reason,
    )


def _servo_ids(sequence_name: str) -> tuple[int, ...]:
    if sequence_name == "tail":
        return TAIL_SERVO_IDS
    if sequence_name == "left_fin":
        return LEFT_FIN_SERVO_IDS
    if sequence_name == "right_fin":
        return RIGHT_FIN_SERVO_IDS
    raise KeyError(sequence_name)
