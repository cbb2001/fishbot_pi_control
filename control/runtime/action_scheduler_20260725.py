"""调度三条同起点、组内串行且可独立推进的 20260725 动作序列。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from control.runtime.discrete_absolute_actions_20260725 import (
    DiscreteAbsoluteMission,
    DiscreteAction,
    RobotCalibration,
    SEQUENCE_NAMES,
    ActionTrajectory,
    build_trajectory,
)


@dataclass
class ScheduledAction:
    """一个动作组中已经确定起点和绝对时间边界的动作实例。"""

    sequence_name: str
    action_index: int
    action: DiscreteAction
    trajectory: ActionTrajectory
    actual_start_t_ns: int
    planned_end_t_ns: int
    completion_t_ns: int | None = None
    completed: bool = False
    endpoint_written: bool = False
    start_command_written: bool = False
    completion_reason: str | None = None
    final_commanded_angles_deg: dict[int, float] | None = None
    previous_theta_before_commit: float | None = None
    previous_theta_after_commit: float | None = None

    @property
    def planned_start_t_ns(self) -> int:
        """兼容事件日志中的 planned_start_t_ns 命名。"""

        return self.actual_start_t_ns

    @property
    def previous_theta(self) -> float:
        """返回本动作轨迹冻结的上一动作 theta。"""

        return self.trajectory.previous_theta

    @property
    def current_theta(self) -> float:
        """返回本动作的目标 theta。"""

        return self.trajectory.current_theta

    @property
    def theta_delta(self) -> float:
        """返回目标 theta 相对 previous theta 的带符号变化量。"""

        return self.trajectory.theta_delta

    @property
    def absolute_theta_delta(self) -> float:
        """返回 theta 变化量的绝对值。"""

        return self.trajectory.absolute_theta_delta

    @property
    def duration_s(self) -> float:
        """返回动作在 YAML 中声明的完整持续秒数。"""

        return self.trajectory.duration_s

    @property
    def duration_ns(self) -> int:
        """使用 round 将 YAML 秒数转换为唯一的整数纳秒持续时间。"""

        return max(1, int(round(self.duration_s * 1_000_000_000)))

    def progress_at(self, t_ns: int) -> float:
        """返回动作独立局部进度，保持动作也正常从 0 增长到 1。"""

        raw = (int(t_ns) - self.actual_start_t_ns) / self.duration_ns
        return min(1.0, max(0.0, raw))


@dataclass(frozen=True)
class SchedulerTransition:
    """一次成功终点提交所产生的结束和可选下一动作。"""

    finished: ScheduledAction | None
    started: ScheduledAction | None


class SequenceScheduler:
    """单个动作组的严格串行调度器。

    previous theta 只在 complete(endpoint_written=True) 内提交。解析、预验证、
    动作执行中途以及失败写入都不会修改该值。
    """

    def __init__(
        self,
        sequence_name: str,
        actions: Iterable[DiscreteAction],
        initial_previous_theta: float,
        calibration: RobotCalibration,
    ) -> None:
        """创建单组调度状态，并保留规定的 previous theta 初值。"""

        if sequence_name not in SEQUENCE_NAMES:
            raise KeyError(sequence_name)
        self.sequence_name = sequence_name
        self.actions = tuple(actions)
        self.calibration = calibration
        self.previous_theta = float(initial_previous_theta)
        self.current_action_index: int | None = None
        self.actual_start_t_ns: int | None = None
        self.planned_end_t_ns: int | None = None
        self.sequence_finished = False
        self.completion_t_ns: int | None = None
        self._next_action_index = 0
        self._active: ScheduledAction | None = None
        self._started = False

    @property
    def active_action(self) -> ScheduledAction | None:
        """返回本组当前活动动作；序列尚未开始或已结束时返回空。"""

        return self._active

    def start(self, mission_start_t_ns: int) -> ScheduledAction | None:
        """在三组共用的任务起点启动本组第一个动作。"""

        if self._started:
            raise RuntimeError(f"序列 {self.sequence_name} 已经启动。")
        self._started = True
        if not self.actions:
            self.sequence_finished = True
            self.completion_t_ns = int(mission_start_t_ns)
            return None
        return self._start_next(int(mission_start_t_ns))

    def is_due(self, t_ns: int) -> bool:
        """判断本组动作规定 t 是否已经经过。"""

        return self._active is not None and int(t_ns) >= self._active.planned_end_t_ns

    def mark_start_command_written(self) -> None:
        """记录动作起点姿态已经成功写入统一 PCA9685 批次。"""

        if self._active is None:
            raise RuntimeError(f"序列 {self.sequence_name} 没有活动动作。")
        self._active.start_command_written = True

    def complete(
        self,
        completion_t_ns: int,
        final_commanded_angles_deg: dict[int, float],
        *,
        endpoint_written: bool,
        start_next: bool = True,
    ) -> SchedulerTransition:
        """在时间到期且终点批次成功后原子提交 previous theta。

        若统一停止信号恰在终点写入期间到达，当前动作仍满足完成条件并提交
        previous theta，但 start_next=False 会阻止创建一个永不执行的下一动作。
        """

        action = self._active
        if action is None:
            raise RuntimeError(f"序列 {self.sequence_name} 没有可完成动作。")
        completion = int(completion_t_ns)
        if completion < action.planned_end_t_ns:
            raise RuntimeError(f"序列 {self.sequence_name} 的动作时间尚未结束。")
        if not endpoint_written:
            raise RuntimeError(
                f"序列 {self.sequence_name} 的终点命令未成功写入，禁止推进。"
            )
        expected = action.trajectory.target_angles_deg
        for servo_id, target in expected.items():
            if (
                servo_id not in final_commanded_angles_deg
                or float(final_commanded_angles_deg[servo_id]) != float(target)
            ):
                raise RuntimeError(
                    f"序列 {self.sequence_name} 的终点命令不完整，禁止推进。"
                )

        before = self.previous_theta
        action.endpoint_written = True
        action.completed = True
        action.completion_t_ns = completion
        action.completion_reason = "time_elapsed_and_endpoint_written"
        action.final_commanded_angles_deg = {
            servo_id: float(final_commanded_angles_deg[servo_id])
            for servo_id in expected
        }
        action.previous_theta_before_commit = before
        self.previous_theta = action.current_theta
        action.previous_theta_after_commit = self.previous_theta
        self._active = None

        if self._next_action_index < len(self.actions) and start_next:
            started = self._start_next(completion)
            # 新动作的起点正是上一动作刚刚成功写入的终点；同一个批次既是
            # 上一动作终点命令，也是下一动作 actual_start 时刻的起点命令。
            started.start_command_written = True
        elif self._next_action_index >= len(self.actions):
            started = None
            self.sequence_finished = True
            self.completion_t_ns = completion
            self.current_action_index = None
            self.actual_start_t_ns = None
            self.planned_end_t_ns = None
        else:
            started = None
            self.current_action_index = None
            self.actual_start_t_ns = None
            self.planned_end_t_ns = None
        return SchedulerTransition(action, started)

    def _start_next(self, start_t_ns: int) -> ScheduledAction:
        """用已提交的 previous theta 在指定绝对时刻创建下一动作。"""

        action_index = self._next_action_index
        action = self.actions[action_index]
        trajectory = build_trajectory(
            self.sequence_name,
            action,
            self.previous_theta,
            self.calibration,
        )
        duration_ns = max(1, int(round(trajectory.duration_s * 1_000_000_000)))
        scheduled = ScheduledAction(
            self.sequence_name,
            action_index,
            action,
            trajectory,
            int(start_t_ns),
            int(start_t_ns) + duration_ns,
        )
        self._next_action_index += 1
        self._active = scheduled
        self.current_action_index = action_index
        self.actual_start_t_ns = scheduled.actual_start_t_ns
        self.planned_end_t_ns = scheduled.planned_end_t_ns
        return scheduled


class ActionScheduler:
    """持有三个 SequenceScheduler，并提供单执行器所需的统一视图。"""

    def __init__(
        self,
        mission: DiscreteAbsoluteMission,
        calibration: RobotCalibration,
    ) -> None:
        """为任务的 tail、left_fin、right_fin 分别创建独立组调度器。"""

        self.mission = mission
        self.calibration = calibration
        initial = calibration.initial_previous_thetas
        self.sequences = {
            name: SequenceScheduler(
                name,
                mission.actions_for(name),
                initial[name],
                calibration,
            )
            for name in SEQUENCE_NAMES
        }
        self.mission_start_t_ns: int | None = None

    def start(self, mission_start_t_ns: int) -> tuple[SchedulerTransition, ...]:
        """让三个序列严格使用同一个 mission_start_t_ns。"""

        if self.mission_start_t_ns is not None:
            raise RuntimeError("动作调度器已经启动。")
        self.mission_start_t_ns = int(mission_start_t_ns)
        return tuple(
            SchedulerTransition(None, sequence.start(self.mission_start_t_ns))
            for sequence in self.sequences.values()
        )

    def active(self, sequence_name: str) -> ScheduledAction | None:
        """取得指定动作组的当前活动动作。"""

        return self.sequences[sequence_name].active_action

    def due_sequences(self, t_ns: int) -> tuple[str, ...]:
        """独立判断三组到期状态，不在组间设置同步屏障。"""

        return tuple(
            name for name, sequence in self.sequences.items() if sequence.is_due(t_ns)
        )

    def mark_initial_starts_written(self) -> None:
        """在任务起点统一批次成功后标记三个非空序列。"""

        for sequence in self.sequences.values():
            if sequence.active_action is not None:
                sequence.mark_start_command_written()

    def complete(
        self,
        sequence_name: str,
        completion_t_ns: int,
        final_commanded_angles_deg: dict[int, float],
        *,
        endpoint_written: bool,
        start_next: bool = True,
    ) -> SchedulerTransition:
        """只推进指定到期组，其他组的索引、起点和局部进度保持不变。"""

        return self.sequences[sequence_name].complete(
            completion_t_ns,
            final_commanded_angles_deg,
            endpoint_written=endpoint_written,
            start_next=start_next,
        )

    def all_finished(self) -> bool:
        """仅当三条动作序列都结束时返回真。"""

        return all(sequence.sequence_finished for sequence in self.sequences.values())

    def mission_completion_t_ns(self) -> int | None:
        """返回三组实际完成时刻的最大值；任务未完成时返回空。"""

        if not self.all_finished():
            return None
        values = [
            sequence.completion_t_ns
            for sequence in self.sequences.values()
            if sequence.completion_t_ns is not None
        ]
        return max(values) if values else self.mission_start_t_ns

    @property
    def previous_thetas(self) -> dict[str, float]:
        """返回当前已提交的三个 previous theta。"""

        return {
            name: sequence.previous_theta
            for name, sequence in self.sequences.items()
        }


def action_event_fields(action: ScheduledAction) -> dict:
    """生成 action started/finished 共用的完整结构化事件字段。"""

    return {
        "action_group": action.sequence_name,
        "action_index": action.action_index,
        "action_parameters": action.action.to_dict(),
        "previous_theta": action.previous_theta,
        "current_theta": action.current_theta,
        "theta_delta": action.theta_delta,
        "absolute_theta_delta": action.absolute_theta_delta,
        "duration_s": action.duration_s,
        "planned_start_t_ns": action.planned_start_t_ns,
        "actual_start_t_ns": action.actual_start_t_ns,
        "planned_end_t_ns": action.planned_end_t_ns,
        "completion_t_ns": action.completion_t_ns,
        "completed": action.completed,
        "endpoint_written": action.endpoint_written,
        "start_command_written": action.start_command_written,
        "completion_reason": action.completion_reason,
        "start_angles_deg": dict(action.trajectory.start_angles_deg),
        "target_angles_deg": dict(action.trajectory.target_angles_deg),
        "tip_peak_angle_deg": action.trajectory.tip_peak_angle_deg,
        "final_commanded_angles_deg": action.final_commanded_angles_deg,
        "previous_theta_before_commit": action.previous_theta_before_commit,
        "previous_theta_after_commit": action.previous_theta_after_commit,
    }
