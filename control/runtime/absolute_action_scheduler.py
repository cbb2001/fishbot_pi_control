from __future__ import annotations

from dataclasses import dataclass

from control.runtime.absolute_discrete_actions import (
    AbsoluteAction, AbsoluteMission, AbsoluteRobotCalibration, ActionTrajectory,
    SEQUENCE_NAMES, build_trajectory,
)


@dataclass
class ScheduledAbsoluteAction:
    sequence_name: str
    action_index: int
    action: AbsoluteAction
    actual_start_t_ns: int
    planned_start_t_ns: int
    planned_end_t_ns: int
    trajectory: ActionTrajectory
    completion_t_ns: int | None = None
    completion_reason: str | None = None
    final_commanded_angles_deg: dict[int, float] | None = None

    @property
    def duration_ns(self) -> int:
        return max(0, int(round(self.trajectory.duration_s * 1_000_000_000)))

    def progress_at(self, t_ns: int) -> float:
        if self.duration_ns == 0:
            return 1.0
        return min(1.0, max(0.0, (int(t_ns) - self.actual_start_t_ns) / self.duration_ns))


@dataclass(frozen=True)
class SchedulerTransition:
    finished: ScheduledAbsoluteAction | None
    started: ScheduledAbsoluteAction | None


class AbsoluteActionScheduler:
    """三路并行、每路串行；下一动作只由成功终点写入推进。"""

    def __init__(self, mission: AbsoluteMission, calibration: AbsoluteRobotCalibration) -> None:
        self.mission, self.calibration = mission, calibration
        self.mission_start_t_ns: int | None = None
        self._active = {name: None for name in SEQUENCE_NAMES}
        self._next = {name: 0 for name in SEQUENCE_NAMES}
        self._completed_t = {name: None for name in SEQUENCE_NAMES}

    def start(self, mission_start_t_ns: int,
              initial_angles: dict[str, dict[int, float]]) -> tuple[SchedulerTransition, ...]:
        if self.mission_start_t_ns is not None:
            raise RuntimeError("调度器已经启动。")
        self.mission_start_t_ns = int(mission_start_t_ns)
        result = []
        for name in SEQUENCE_NAMES:
            if self.mission.actions_for(name):
                result.append(SchedulerTransition(None, self._start_next(name, mission_start_t_ns,
                                                                          initial_angles[name])))
            else:
                self._completed_t[name] = int(mission_start_t_ns)
        return tuple(result)

    def active(self, name: str) -> ScheduledAbsoluteAction | None:
        return self._active[name]

    def due_sequences(self, t_ns: int) -> tuple[str, ...]:
        return tuple(name for name, action in self._active.items()
                     if action is not None and int(t_ns) >= action.planned_end_t_ns)

    def complete(self, name: str, completion_t_ns: int,
                 final_angles: dict[int, float]) -> SchedulerTransition:
        active = self._active[name]
        if active is None or completion_t_ns < active.planned_end_t_ns:
            raise RuntimeError(f"序列 {name} 尚不能完成。")
        active.completion_t_ns = int(completion_t_ns)
        active.completion_reason = "time_elapsed_and_endpoint_written"
        active.final_commanded_angles_deg = dict(final_angles)
        self._active[name] = None
        if self._next[name] < len(self.mission.actions_for(name)):
            started = self._start_next(name, completion_t_ns, final_angles)
        else:
            started = None
            self._completed_t[name] = int(completion_t_ns)
        return SchedulerTransition(active, started)

    def all_finished(self) -> bool:
        return all(value is not None for value in self._completed_t.values())

    def mission_completion_t_ns(self) -> int | None:
        return None if not self.all_finished() else max(int(x) for x in self._completed_t.values()
                                                         if x is not None)

    def _start_next(self, name: str, start_t_ns: int,
                    start_angles: dict[int, float]) -> ScheduledAbsoluteAction:
        index = self._next[name]
        action = self.mission.actions_for(name)[index]
        trajectory = build_trajectory(name, action, start_angles, self.calibration)
        duration_ns = max(0, int(round(trajectory.duration_s * 1_000_000_000)))
        scheduled = ScheduledAbsoluteAction(name, index, action, int(start_t_ns), int(start_t_ns),
                                             int(start_t_ns) + duration_ns, trajectory)
        self._active[name] = scheduled
        self._next[name] += 1
        return scheduled


def action_event_fields(action: ScheduledAbsoluteAction) -> dict:
    return {"action_index": action.action_index, "action_parameters": action.action.to_dict(),
            "planned_start_t_ns": action.planned_start_t_ns,
            "actual_start_t_ns": action.actual_start_t_ns,
            "planned_end_t_ns": action.planned_end_t_ns,
            "completion_t_ns": action.completion_t_ns,
            "completion_reason": action.completion_reason,
            "start_angles_deg": action.trajectory.start_angles_deg,
            "target_angles_deg": action.trajectory.target_angles_deg,
            "final_commanded_angles_deg": action.final_commanded_angles_deg,
            "servo_durations_s": action.trajectory.servo_durations_s,
            "action_duration_s": action.trajectory.duration_s}
