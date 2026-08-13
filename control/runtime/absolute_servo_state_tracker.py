from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable

from control.runtime.absolute_action_scheduler import ScheduledAbsoluteAction, SchedulerTransition
from control.runtime.absolute_discrete_actions import (
    AbsoluteRobotCalibration, SEQUENCE_NAMES, sequence_servo_ids,
)


ESTIMATION_MODE = "assumed_perfect_tracking"


@dataclass
class TrackedAction:
    scheduled: ScheduledAbsoluteAction


@dataclass(frozen=True)
class ServoPoseSnapshot:
    """指定单调时间戳下的无反馈七路舵机姿态快照。"""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


class ServoStateTracker:
    """保存参考轨迹和成功命令历史；estimated 绝不表示真实测量角度。"""

    def __init__(self, calibration: AbsoluteRobotCalibration) -> None:
        self.calibration = calibration
        self._lock = RLock()
        self.initialized_t_ns: int | None = None
        self.mission_start_t_ns: int | None = None
        self._actions: dict[str, list[TrackedAction]] = {name: [] for name in SEQUENCE_NAMES}
        self._command_times = {servo_id: [] for servo_id in calibration.servos}
        self._command_angles = {servo_id: [] for servo_id in calibration.servos}

    def initialize(self, initialized_t_ns: int) -> None:
        self.initialized_t_ns = int(initialized_t_ns)

    def set_mission_start(self, mission_start_t_ns: int) -> None:
        self.mission_start_t_ns = int(mission_start_t_ns)

    def record_actions_started(self, actions: Iterable[ScheduledAbsoluteAction]) -> None:
        with self._lock:
            for action in actions:
                self._actions[action.sequence_name].append(TrackedAction(action))

    def record_transition(self, transition: SchedulerTransition) -> None:
        with self._lock:
            if transition.started is not None:
                self._actions[transition.started.sequence_name].append(TrackedAction(transition.started))

    def record_successful_command(self, servo_id: int, write_end_t_ns: int,
                                  angle_deg: float) -> None:
        """只在 PCA9685 调用成功返回后追加 commanded 历史。"""

        with self._lock:
            times = self._command_times[servo_id]
            if times and write_end_t_ns < times[-1]:
                raise ValueError("成功写入时间必须单调不减。")
            times.append(int(write_end_t_ns))
            self._command_angles[servo_id].append(float(angle_deg))

    def latest_commanded_angles(self) -> dict[int, float]:
        with self._lock:
            return {servo_id: (values[-1] if values else self.calibration.servos[servo_id].center_deg)
                    for servo_id, values in self._command_angles.items()}

    def get_servo_pose_at(self, t_ns: int) -> ServoPoseSnapshot:
        """按任意历史 monotonic_ns 时间重算参考角并检索最近成功命令。"""

        query_t_ns = int(t_ns)
        with self._lock:
            reference = self.calibration.centers_deg
            indices: dict[str, int | None] = {name: None for name in SEQUENCE_NAMES}
            progress: dict[str, float | None] = {name: None for name in SEQUENCE_NAMES}
            durations: dict[str, Any] = {"tail": {}, "left_fin": None, "right_fin": None}
            for name in SEQUENCE_NAMES:
                held = {servo_id: self.calibration.servos[servo_id].center_deg
                        for servo_id in sequence_servo_ids(name)}
                for tracked in self._actions[name]:
                    action = tracked.scheduled
                    if query_t_ns < action.actual_start_t_ns:
                        break
                    if action.completion_t_ns is None or query_t_ns < action.completion_t_ns:
                        elapsed = max(0.0, (query_t_ns - action.actual_start_t_ns) / 1e9)
                        held = action.trajectory.evaluate(elapsed)
                        indices[name] = action.action_index
                        progress[name] = action.progress_at(query_t_ns)
                        durations[name] = (dict(action.trajectory.servo_durations_s)
                                           if name == "tail" else action.trajectory.duration_s)
                        break
                    held = dict(action.trajectory.target_angles_deg)
                reference.update(held)
            commanded = {servo_id: self._command_at(servo_id, query_t_ns)
                         for servo_id in sorted(self.calibration.servos)}
            elapsed = None if self.mission_start_t_ns is None else max(
                0.0, (query_t_ns - self.mission_start_t_ns) / 1e9)
            data = {"query_t_ns": query_t_ns, "servo_pose_query_t_ns": query_t_ns,
                    "mission_elapsed_s": elapsed,
                    "reference_angles_deg": reference,
                    "commanded_angles_deg": commanded,
                    "estimated_angles_deg": dict(reference),
                    "tail_action_index": indices["tail"],
                    "left_action_index": indices["left_fin"],
                    "right_action_index": indices["right_fin"],
                    "tail_action_progress": progress["tail"],
                    "left_action_progress": progress["left_fin"],
                    "right_action_progress": progress["right_fin"],
                    "tail_servo_durations_s": durations["tail"],
                    "left_action_duration_s": durations["left_fin"],
                    "right_action_duration_s": durations["right_fin"],
                    "feedback_available": False, "estimation_mode": ESTIMATION_MODE}
            return ServoPoseSnapshot(data)

    def get_servo_pose_at_elapsed_s(self, elapsed_s: float) -> ServoPoseSnapshot:
        if self.mission_start_t_ns is None:
            raise RuntimeError("mission_start_t_ns 尚未设置。")
        return self.get_servo_pose_at(self.mission_start_t_ns + int(round(elapsed_s * 1e9)))

    def query(self, t_ns: int) -> dict[str, Any]:
        """为同步线程提供向后兼容的字典包装。"""

        return self.get_servo_pose_at(t_ns).to_dict()

    def _command_at(self, servo_id: int, t_ns: int) -> float | None:
        index = bisect_right(self._command_times[servo_id], t_ns) - 1
        return None if index < 0 else self._command_angles[servo_id][index]
