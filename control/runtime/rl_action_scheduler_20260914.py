"""Deterministic scheduler for RL tail and synchronized pectoral actions."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable
from .rl_actions_20260914 import *

@dataclass
class ScheduledRLAction:
    agent: str
    action: TailAction | FinAction
    trajectory: RLActionTrajectory
    start_t_ns: int
    completion_t_ns: int | None = None
    endpoint_written: bool = False
    request_id: int | None = None
    context: object = None
    @property
    def end_t_ns(self): return self.start_t_ns + round(self.action.t * 1e9)
    def progress_at(self, t_ns: int) -> float:
        return max(0.0, min(1.0, (int(t_ns)-self.start_t_ns)/(self.end_t_ns-self.start_t_ns)))

class RLActionScheduler:
    """Schedules one tail and one fin action at a time.

    A fin action automatically returns both left and mirrored right references
    from :meth:`references_at`; ``previous_theta`` advances only after the
    endpoint write is acknowledged.
    """
    def __init__(self, centers: dict[int,float] | None = None, clock_ns: Callable[[],int] | None = None, limits: dict[int,tuple[float,float]] | None = None, coupling=None):
        calibration = centers if hasattr(centers, 'centers') else None
        self.centers = dict(calibration.centers if calibration else (centers or {1:85,2:95,3:95,4:111,5:90,6:143,7:90}))
        self.limits = limits or (calibration.limits if calibration else {i:(0.0,180.0) for i in ALL_SERVO_IDS})
        self.coupling = dict(coupling if coupling is not None else (calibration.coupling if calibration else {}))
        self.clock_ns = clock_ns or time.monotonic_ns
        self.previous_theta = {"tail":0.0, "fin":0.0}
        self.fin_sequence = FinSequenceState()
        self._fin_sequence_interrupted = False
        self.committed_angles = dict(self.centers)
        self.active: dict[str,ScheduledRLAction|None] = {"tail":None,"fin":None}
        self._index = {"tail":0,"fin":0}
    def schedule(self, agent: str, action: TailAction|FinAction, start_t_ns: int|None = None) -> ScheduledRLAction:
        key = "tail" if isinstance(action,TailAction) else "fin"
        if key == "fin" and agent in ("left_fin", "action2", "fin"):
            agent = "fin"
        if agent != key: raise ValueError("agent/action mismatch")
        if self.active[key] is not None: raise RuntimeError(f"{key} action is still active")
        if key == 'fin':
            if self._fin_sequence_interrupted:
                raise RuntimeError('侧鳍动作中途取消，须重新启动并回中后开始新序列')
            self.fin_sequence.validate(action)
        traj = build_rl_trajectory(action, self.previous_theta[key], self, start_angles_deg=self.committed_angles)
        # Reject invalid trajectories explicitly; silent clamping would alter
        # the action selected by PPO and make training data inconsistent.
        validate_trajectory(traj, self.limits)
        item = ScheduledRLAction(key, action, traj, int(self.clock_ns() if start_t_ns is None else start_t_ns))
        self.active[key] = item; self._index[key] += 1
        return item
    def references_at(self, t_ns: int|None = None) -> dict[int,float]:
        t = int(self.clock_ns() if t_ns is None else t_ns)
        out = dict(self.committed_angles)
        for key,item in self.active.items():
            if item is None: continue
            out.update(item.trajectory.evaluate_all((t-item.start_t_ns)/1e9))
        return out
    def complete(self, agent: str, *, endpoint_written: bool, completion_t_ns: int|None = None) -> ScheduledRLAction:
        item = self.active.get(agent)
        if item is None: raise RuntimeError("no active action")
        t = int(self.clock_ns() if completion_t_ns is None else completion_t_ns)
        if t < item.end_t_ns: raise RuntimeError("action duration has not elapsed")
        if not endpoint_written: raise RuntimeError("endpoint PWM write required before commit")
        if agent == 'fin':
            # 完成回执入队前原子替换不可变状态；采样线程只读取完整快照。
            self.fin_sequence = self.fin_sequence.after(item.action)
        item.completion_t_ns, item.endpoint_written = t, True
        self.previous_theta[agent] = float(item.action.theta)
        self.committed_angles.update(item.trajectory.evaluate_all(item.action.t))
        self.active[agent] = None
        return item
    def hold(self, angles_deg):
        """在成功写入的姿态取消当前动作；下一动作从此姿态连续启动。"""
        cancelled = [item for item in self.active.values() if item is not None]
        if self.active['fin'] is not None:
            self._fin_sequence_interrupted = True
        self.committed_angles.update(angles_deg)
        for key in self.active:
            self.active[key] = None
        self.previous_theta['fin'] = self.committed_angles[4]-self.centers[4]
        self.previous_theta['tail'] = self.committed_angles[1]-self.centers[1]
        return cancelled
    def tick(self, t_ns: int|None = None) -> tuple[dict[int,float], list[ScheduledRLAction]]:
        t = int(self.clock_ns() if t_ns is None else t_ns)
        refs = self.references_at(t); done=[]
        for key,item in list(self.active.items()):
            if item is not None and t >= item.end_t_ns:
                done.append(item)
        return refs, done

ActionScheduler = RLActionScheduler
