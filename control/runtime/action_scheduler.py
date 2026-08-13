from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from control.runtime.discrete_actions import (
    DiscreteMission,
    FinAction,
    SEQUENCE_NAMES,
    TailAction,
    action_duration_s,
)


NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class ScheduledAction:
    """单个动作的理想计划、实际启动和完成状态。"""

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

    @property
    def endpoint_due_t_ns(self) -> int:
        return self.actual_start_t_ns + _duration_ns(self.action)

    def progress_at(self, t_ns: int) -> float:
        duration_ns = max(1, _duration_ns(self.action))
        return min(1.0, max(0.0, (int(t_ns) - self.actual_start_t_ns) / duration_ns))


@dataclass(frozen=True)
class SchedulerTransition:
    """一次原子调度转换，包含刚完成动作和紧接着启动的动作。"""

    finished: ScheduledAction | None
    started: ScheduledAction | None


class ActionScheduler:
    """维护三路并行、各自严格串行的动作游标。"""

    def __init__(self, mission: DiscreteMission) -> None:
        self.mission = mission
        self.mission_start_t_ns: int | None = None
        self._planned_offsets = self._build_planned_offsets()
        self._active: dict[str, ScheduledAction | None] = {
            name: None for name in SEQUENCE_NAMES
        }
        self._next_index = {name: 0 for name in SEQUENCE_NAMES}
        self._sequence_completion_t_ns: dict[str, int | None] = {
            name: None for name in SEQUENCE_NAMES
        }

    def start(
        self,
        mission_start_t_ns: int,
        initial_angles_by_sequence: dict[str, dict[int, float]],
    ) -> tuple[SchedulerTransition, ...]:
        """在同一个任务时刻启动三条序列的首个动作。"""

        if self.mission_start_t_ns is not None:
            raise RuntimeError("调度器已经启动。")
        self.mission_start_t_ns = int(mission_start_t_ns)
        transitions: list[SchedulerTransition] = []
        for name in SEQUENCE_NAMES:
            if self.mission.actions_for(name):
                started = self._start_next(name, mission_start_t_ns, initial_angles_by_sequence[name])
                transitions.append(SchedulerTransition(None, started))
            else:
                self._sequence_completion_t_ns[name] = int(mission_start_t_ns)
        return tuple(transitions)

    def active(self, sequence_name: str) -> ScheduledAction | None:
        return self._active[sequence_name]

    def due_sequences(self, t_ns: int) -> tuple[str, ...]:
        """返回已过实际持续时间但尚未提交终点写入的序列。"""

        return tuple(
            name
            for name in SEQUENCE_NAMES
            if self._active[name] is not None
            and int(t_ns) >= self._active[name].endpoint_due_t_ns
        )

    def complete(
        self,
        sequence_name: str,
        completion_t_ns: int,
        final_angles_deg: dict[int, float],
    ) -> SchedulerTransition:
        """提交完成，并让下一动作从本次实际完成时刻启动。"""

        active = self._active[sequence_name]
        if active is None:
            raise RuntimeError(f"序列 {sequence_name} 当前没有活动动作。")
        if int(completion_t_ns) < active.endpoint_due_t_ns:
            raise RuntimeError(f"序列 {sequence_name} 尚未经过规定动作时间。")
        active.completion_t_ns = int(completion_t_ns)
        active.final_angles_deg = dict(final_angles_deg)
        active.completion_reason = "time_elapsed_and_endpoint_written"
        self._active[sequence_name] = None

        actions = self.mission.actions_for(sequence_name)
        if self._next_index[sequence_name] < len(actions):
            started = self._start_next(
                sequence_name,
                completion_t_ns,
                final_angles_deg,
            )
        else:
            started = None
            self._sequence_completion_t_ns[sequence_name] = int(completion_t_ns)
        return SchedulerTransition(active, started)

    def all_finished(self) -> bool:
        return all(value is not None for value in self._sequence_completion_t_ns.values())

    def mission_completion_t_ns(self) -> int | None:
        if not self.all_finished():
            return None
        return max(int(value) for value in self._sequence_completion_t_ns.values() if value is not None)

    def action_state_at(self, t_ns: int) -> dict[str, int | float | None]:
        state: dict[str, int | float | None] = {}
        for name in SEQUENCE_NAMES:
            action = self._active[name]
            state[f"{name}_action_index"] = None if action is None else action.action_index
            state[f"{name}_action_progress"] = None if action is None else action.progress_at(t_ns)
        return state

    def _start_next(
        self,
        sequence_name: str,
        actual_start_t_ns: int,
        start_angles_deg: dict[int, float],
    ) -> ScheduledAction:
        index = self._next_index[sequence_name]
        action = self.mission.actions_for(sequence_name)[index]
        if self.mission_start_t_ns is None:
            raise RuntimeError("调度器尚未初始化 mission_start_t_ns。")
        planned_offset_ns, duration_ns = self._planned_offsets[sequence_name][index]
        scheduled = ScheduledAction(
            sequence_name=sequence_name,
            action_index=index,
            action=action,
            planned_start_t_ns=self.mission_start_t_ns + planned_offset_ns,
            planned_end_t_ns=self.mission_start_t_ns + planned_offset_ns + duration_ns,
            actual_start_t_ns=int(actual_start_t_ns),
            start_angles_deg=dict(start_angles_deg),
        )
        self._active[sequence_name] = scheduled
        self._next_index[sequence_name] += 1
        return scheduled

    def _build_planned_offsets(self) -> dict[str, tuple[tuple[int, int], ...]]:
        result: dict[str, tuple[tuple[int, int], ...]] = {}
        for name in SEQUENCE_NAMES:
            offset_ns = 0
            entries: list[tuple[int, int]] = []
            for action in self.mission.actions_for(name):
                duration_ns = _duration_ns(action)
                entries.append((offset_ns, duration_ns))
                offset_ns += duration_ns
            result[name] = tuple(entries)
        return result


def scheduled_action_to_dict(action: ScheduledAction) -> dict[str, Any]:
    """生成动作事件使用的公共字段。"""

    return {
        "sequence_name": action.sequence_name,
        "action_index": action.action_index,
        "planned_start_t_ns": action.planned_start_t_ns,
        "planned_end_t_ns": action.planned_end_t_ns,
        "actual_start_t_ns": action.actual_start_t_ns,
        "completion_t_ns": action.completion_t_ns,
        "completion_reason": action.completion_reason,
        "start_angles_deg": dict(action.start_angles_deg),
        "final_angles_deg": None if action.final_angles_deg is None else dict(action.final_angles_deg),
    }


def _duration_ns(action: TailAction | FinAction) -> int:
    return max(1, int(round(action_duration_s(action) * NANOSECONDS_PER_SECOND)))
