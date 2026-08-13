"""记录 20260725 动作历史，并按任意单调时间戳重建无反馈舵机姿态。"""

from __future__ import annotations

import math
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable, Iterator

from control.runtime.action_scheduler_20260725 import (
    ScheduledAction,
    SchedulerTransition,
)
from control.runtime.discrete_absolute_actions_20260725 import (
    RobotCalibration,
    SEQUENCE_NAMES,
    ActionTrajectory,
    sequence_servo_ids,
    smooth_motion,
)


ESTIMATION_MODE = "assumed_perfect_tracking"


@dataclass
class TrackedAction:
    """状态跟踪器私有的动作历史副本，避免读取调度器的可变对象。"""

    sequence_name: str
    action_index: int
    trajectory: ActionTrajectory
    actual_start_t_ns: int
    planned_end_t_ns: int
    completion_t_ns: int | None = None
    endpoint_written: bool = False
    completion_reason: str | None = None

    @classmethod
    def from_scheduled(cls, action: ScheduledAction) -> "TrackedAction":
        """复制调度动作及其轨迹，隔离后续可变完成状态。"""

        trajectory = action.trajectory
        private_trajectory = ActionTrajectory(
            sequence_name=trajectory.sequence_name,
            action=trajectory.action,
            previous_theta=trajectory.previous_theta,
            current_theta=trajectory.current_theta,
            start_angles_deg=dict(trajectory.start_angles_deg),
            target_angles_deg=dict(trajectory.target_angles_deg),
            duration_s=trajectory.duration_s,
            tip_peak_angle_deg=trajectory.tip_peak_angle_deg,
        )
        return cls(
            action.sequence_name,
            action.action_index,
            private_trajectory,
            action.actual_start_t_ns,
            action.planned_end_t_ns,
            action.completion_t_ns,
            action.endpoint_written,
            action.completion_reason,
        )

    @property
    def duration_ns(self) -> int:
        """返回与调度器一致、由动作秒数舍入得到的整数纳秒时长。"""

        return max(1, int(round(self.trajectory.duration_s * 1_000_000_000)))

    def progress_at(self, t_ns: int) -> float:
        """计算指定单调时刻在本历史动作中的闭区间进度。"""

        raw = (int(t_ns) - self.actual_start_t_ns) / self.duration_ns
        return min(1.0, max(0.0, raw))


@dataclass(frozen=True)
class ReferenceSegment:
    """任务动作之外的安全回中连续参考段。"""

    phase: str
    start_t_ns: int
    end_t_ns: int
    start_angles_deg: dict[int, float]
    target_angles_deg: dict[int, float]


@dataclass(frozen=True)
class ServoPoseSnapshot:
    """指定 monotonic_ns 时刻的七路无反馈姿态快照。"""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """返回浅拷贝，避免调用者替换快照顶层字段。"""

        return dict(self.data)


@dataclass(frozen=True)
class _SequencePose:
    """封装单个动作组在查询时刻的姿态和动作上下文。"""

    angles_deg: dict[int, float]
    action_index: int | None
    progress: float | None
    previous_theta: float
    current_theta: float
    duration_s: float | None
    tip_peak_angle_deg: float | None


class ServoStateTracker:
    """保存理论参考轨迹和成功命令历史。

    本类没有舵机角度反馈。estimated_angles_deg 第一版严格等于连续参考轨迹，
    只能称为“假设完美跟随的模型估计”，不能称为实测或真实舵机角度。
    """

    def __init__(self, calibration: RobotCalibration) -> None:
        """初始化三组动作历史、成功命令历史和安全参考段存储。"""

        self.calibration = calibration
        self._lock = RLock()
        self.initialized_t_ns: int | None = None
        self.mission_start_t_ns: int | None = None
        self._actions: dict[str, list[TrackedAction]] = {
            name: [] for name in SEQUENCE_NAMES
        }
        self._command_times: dict[int, list[int]] = {
            servo_id: [] for servo_id in calibration.servos
        }
        self._command_angles: dict[int, list[float]] = {
            servo_id: [] for servo_id in calibration.servos
        }
        self._reference_segments: list[ReferenceSegment] = []

    def initialize(self, initialized_t_ns: int) -> None:
        """记录模型初始化时间；不声称物理舵机已经由反馈确认到位。"""

        with self._lock:
            self.initialized_t_ns = int(initialized_t_ns)

    def set_mission_start(self, mission_start_t_ns: int) -> None:
        """设置三组共用的任务单调时钟起点。"""

        with self._lock:
            self.mission_start_t_ns = int(mission_start_t_ns)

    @contextmanager
    def atomic_update(self) -> Iterator[None]:
        """阻止查询观察统一硬件批次与动作历史提交之间的中间窗口。

        锁是可重入锁，因此执行器在上下文内仍可调用本类的命令记录和批量
        transition 方法。传感器查询会短暂等待 I2C 批次结束，随后一次看到
        与同一 write_end_t_ns 对应的成功命令和动作提交结果。
        """

        with self._lock:
            yield

    def record_actions_started(self, actions: Iterable[ScheduledAction]) -> None:
        """复制初始三组动作，使历史查询不依赖调度器锁。"""

        with self._lock:
            for action in actions:
                self._append_started(action)

    def record_transition(self, transition: SchedulerTransition) -> None:
        """原子记录已提交的结束动作和紧接着开始的下一动作。"""

        self.record_transitions((transition,))

    def record_transitions(
        self,
        transitions: Iterable[SchedulerTransition],
    ) -> None:
        """一次发布同一统一命令批次触发的全部动作组切换。

        多组可能在同一个七路终点批次到期。用一把锁提交整个 transition
        集合，传感器查询只会看到“全部提交前”或“全部提交后”，不会看到
        例如尾鳍已经进入下一动作、左右胸鳍仍停留在上一动作的混合快照。
        """

        with self._lock:
            for transition in transitions:
                if transition.finished is not None:
                    finished = transition.finished
                    record = self._find_record(finished)
                    record.completion_t_ns = finished.completion_t_ns
                    record.endpoint_written = finished.endpoint_written
                    record.completion_reason = finished.completion_reason
                if transition.started is not None:
                    self._append_started(transition.started)

    def record_successful_batch(
        self,
        write_end_t_ns: int,
        angles_deg: dict[int, float],
    ) -> None:
        """仅在统一 PCA9685 批次完整成功返回后追加命令历史。"""

        timestamp = int(write_end_t_ns)
        with self._lock:
            for servo_id in sorted(angles_deg):
                self._record_successful_command_locked(
                    int(servo_id),
                    timestamp,
                    float(angles_deg[servo_id]),
                )

    def record_reference_segment(
        self,
        phase: str,
        start_t_ns: int,
        end_t_ns: int,
        start_angles_deg: dict[int, float],
        target_angles_deg: dict[int, float],
    ) -> None:
        """发布任务外七路参考曲线，供同步传感器按 sample_t_ns 查询。"""

        if set(start_angles_deg) != set(self.calibration.servos) or set(
            target_angles_deg
        ) != set(self.calibration.servos):
            raise ValueError("任务外参考段必须完整包含全部七路舵机。")
        with self._lock:
            self._reference_segments.append(
                ReferenceSegment(
                    phase=str(phase),
                    start_t_ns=int(start_t_ns),
                    end_t_ns=max(int(start_t_ns), int(end_t_ns)),
                    start_angles_deg={
                        int(servo_id): float(angle)
                        for servo_id, angle in start_angles_deg.items()
                    },
                    target_angles_deg={
                        int(servo_id): float(angle)
                        for servo_id, angle in target_angles_deg.items()
                    },
                )
            )

    def record_successful_command(
        self,
        servo_id: int,
        write_end_t_ns: int,
        angle_deg: float,
    ) -> None:
        """测试及兼容入口：记录单路已成功返回的硬件写入。"""

        with self._lock:
            self._record_successful_command_locked(
                int(servo_id),
                int(write_end_t_ns),
                float(angle_deg),
            )

    def latest_commanded_angles(self) -> dict[int, float]:
        """返回最近成功命令；尚无命令时使用初始化参考姿态供清理使用。"""

        with self._lock:
            initial = self.calibration.initial_angles_deg
            return {
                servo_id: (
                    values[-1] if values else float(initial[servo_id])
                )
                for servo_id, values in self._command_angles.items()
            }

    def get_servo_pose_at(self, t_ns: int) -> ServoPoseSnapshot:
        """查询任务期间任意历史单调时间戳的参考、命令和估计姿态。"""

        query_t_ns = int(t_ns)
        with self._lock:
            group_pose = {
                name: self._sequence_pose_at(name, query_t_ns)
                for name in SEQUENCE_NAMES
            }
            reference = dict(self.calibration.initial_angles_deg)
            for pose in group_pose.values():
                reference.update(pose.angles_deg)
            segment_reference = self._reference_segment_at(query_t_ns)
            control_phase = (
                "pre_mission"
                if self.mission_start_t_ns is None
                or query_t_ns < self.mission_start_t_ns
                else "mission"
            )
            if segment_reference is not None:
                control_phase, reference = segment_reference
            commanded = {
                servo_id: self._command_at_locked(servo_id, query_t_ns)
                for servo_id in sorted(self.calibration.servos)
            }
            mission_elapsed_s = (
                None
                if self.mission_start_t_ns is None
                else max(0.0, (query_t_ns - self.mission_start_t_ns) / 1e9)
            )
            tail = group_pose["tail"]
            left = group_pose["left_fin"]
            right = group_pose["right_fin"]
            data = {
                "query_t_ns": query_t_ns,
                "servo_pose_query_t_ns": query_t_ns,
                "mission_elapsed_s": mission_elapsed_s,
                "control_phase": control_phase,
                "reference_angles_deg": reference,
                "commanded_angles_deg": commanded,
                "estimated_angles_deg": dict(reference),
                "tail_action_index": (
                    None if segment_reference is not None else tail.action_index
                ),
                "left_action_index": (
                    None if segment_reference is not None else left.action_index
                ),
                "right_action_index": (
                    None if segment_reference is not None else right.action_index
                ),
                "previous_action1_theta": tail.previous_theta,
                "previous_action2_theta": left.previous_theta,
                "previous_action3_theta": right.previous_theta,
                "current_action1_theta": tail.current_theta,
                "current_action2_theta": left.current_theta,
                "current_action3_theta": right.current_theta,
                "tail_action_progress": (
                    None if segment_reference is not None else tail.progress
                ),
                "left_action_progress": (
                    None if segment_reference is not None else left.progress
                ),
                "right_action_progress": (
                    None if segment_reference is not None else right.progress
                ),
                "tail_action_duration_s": tail.duration_s,
                "left_action_duration_s": left.duration_s,
                "right_action_duration_s": right.duration_s,
                "left_tip_peak_angle_deg": left.tip_peak_angle_deg,
                "right_tip_peak_angle_deg": right.tip_peak_angle_deg,
                "feedback_available": False,
                "estimation_mode": ESTIMATION_MODE,
            }
            return ServoPoseSnapshot(data)

    def _reference_segment_at(
        self,
        query_t_ns: int,
    ) -> tuple[str, dict[int, float]] | None:
        """返回最后一个已开始的任务外参考段；调用者已持有 tracker 锁。"""

        for segment in reversed(self._reference_segments):
            if query_t_ns < segment.start_t_ns:
                continue
            duration_ns = segment.end_t_ns - segment.start_t_ns
            if duration_ns <= 0 or query_t_ns >= segment.end_t_ns:
                return segment.phase, dict(segment.target_angles_deg)
            elapsed_s = (query_t_ns - segment.start_t_ns) / 1_000_000_000.0
            duration_s = duration_ns / 1_000_000_000.0
            return (
                segment.phase,
                {
                    servo_id: smooth_motion(
                        segment.start_angles_deg[servo_id],
                        segment.target_angles_deg[servo_id],
                        elapsed_s,
                        duration_s,
                    )
                    for servo_id in sorted(self.calibration.servos)
                },
            )
        return None

    def get_servo_pose_at_elapsed_s(self, elapsed_s: float) -> ServoPoseSnapshot:
        """以 mission_start_t_ns 为原点查询指定秒数。"""

        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed_s 必须是有限数。")
        with self._lock:
            mission_start = self.mission_start_t_ns
        if mission_start is None:
            raise RuntimeError("mission_start_t_ns 尚未设置。")
        return self.get_servo_pose_at(
            mission_start + int(round(elapsed * 1_000_000_000))
        )

    def query(self, t_ns: int) -> dict[str, Any]:
        """为现有 SensorSyncWorker 提供稳定的字典适配接口。"""

        return self.get_servo_pose_at(t_ns).to_dict()

    def _append_started(self, action: ScheduledAction) -> None:
        """把新启动动作复制进对应组历史，并拒绝重复记录。"""

        records = self._actions[action.sequence_name]
        if records and (
            records[-1].action_index == action.action_index
            and records[-1].actual_start_t_ns == action.actual_start_t_ns
        ):
            raise RuntimeError("同一个调度动作不能重复写入状态历史。")
        records.append(TrackedAction.from_scheduled(action))

    def _find_record(self, action: ScheduledAction) -> TrackedAction:
        """按动作组、索引和实际起点定位待提交的历史副本。"""

        for record in reversed(self._actions[action.sequence_name]):
            if (
                record.action_index == action.action_index
                and record.actual_start_t_ns == action.actual_start_t_ns
            ):
                return record
        raise RuntimeError("未找到待完成动作的状态历史。")

    def _record_successful_command_locked(
        self,
        servo_id: int,
        write_end_t_ns: int,
        angle_deg: float,
    ) -> None:
        """在已持锁条件下追加一条时间单调的单舵机成功命令。"""

        if servo_id not in self._command_times:
            raise KeyError(f"未知舵机 ID：{servo_id}")
        times = self._command_times[servo_id]
        if times and write_end_t_ns < times[-1]:
            raise ValueError("成功命令时间必须单调不减。")
        times.append(write_end_t_ns)
        self._command_angles[servo_id].append(angle_deg)

    def _command_at_locked(self, servo_id: int, t_ns: int) -> float | None:
        """二分查询不晚于指定时刻的最近成功命令角。"""

        index = bisect_right(self._command_times[servo_id], int(t_ns)) - 1
        if index < 0:
            return None
        return self._command_angles[servo_id][index]

    def _sequence_pose_at(self, sequence_name: str, query_t_ns: int) -> _SequencePose:
        """按动作实际起止历史重放一个动作组，不读取当前全局 previous 状态。"""

        servo_ids = sequence_servo_ids(sequence_name)
        initial = self.calibration.initial_angles_deg
        held = {servo_id: initial[servo_id] for servo_id in servo_ids}
        committed_previous = self.calibration.initial_previous_thetas[sequence_name]
        current_theta = committed_previous
        action_index: int | None = None
        progress: float | None = None
        duration_s: float | None = None
        tip_peak: float | None = None

        for record in self._actions[sequence_name]:
            if query_t_ns < record.actual_start_t_ns:
                break
            completion = record.completion_t_ns
            if completion is None or query_t_ns < completion:
                # 调度器以 round(t*1e9) 得到唯一整数纳秒边界。轨迹查询也
                # 必须使用同一个归一化进度，否则亚纳秒小数会出现“已经 due
                # 但 evaluate 尚未到原始浮点 t”的不一致，终点甚至可能不是
                # 精确 target。
                normalized_progress = record.progress_at(query_t_ns)
                if normalized_progress <= 0.0:
                    held = dict(record.trajectory.start_angles_deg)
                elif normalized_progress >= 1.0:
                    held = dict(record.trajectory.target_angles_deg)
                else:
                    held = record.trajectory.evaluate(
                        normalized_progress * record.trajectory.duration_s
                    )
                return _SequencePose(
                    held,
                    record.action_index,
                    normalized_progress,
                    record.trajectory.previous_theta,
                    record.trajectory.current_theta,
                    record.trajectory.duration_s,
                    record.trajectory.tip_peak_angle_deg,
                )

            # completion_t_ns 时 previous theta 已原子提交；若同一时刻存在
            # 下一动作，循环的下一轮会立即以其 previous theta 覆盖这些字段。
            held = dict(record.trajectory.target_angles_deg)
            committed_previous = record.trajectory.current_theta
            current_theta = committed_previous
            action_index = record.action_index
            progress = 1.0
            duration_s = record.trajectory.duration_s
            tip_peak = record.trajectory.tip_peak_angle_deg

        return _SequencePose(
            held,
            action_index,
            progress,
            committed_previous,
            current_theta,
            duration_s,
            tip_peak,
        )
