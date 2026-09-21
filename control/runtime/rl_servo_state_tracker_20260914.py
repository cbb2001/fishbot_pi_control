"""Timestamped model based servo state estimator used by RL observations."""
from __future__ import annotations
import bisect
import threading
from dataclasses import dataclass
from typing import Any
from .rl_actions_20260914 import *

@dataclass(frozen=True)
class RLServoPose:
    query_t_ns: int
    reference_angle_deg: dict[int,float]
    commanded_angle_deg: dict[int,float|None]
    estimated_angle_deg: dict[int,float]
    def to_dict(self):
        return {"query_t_ns":self.query_t_ns,"reference_angles_deg":dict(self.reference_angle_deg),"commanded_angles_deg":dict(self.commanded_angle_deg),"estimated_angles_deg":dict(self.estimated_angle_deg),"feedback_available":False,"estimation_mode":"last_successful_command_no_feedback"}

class RLServoStateTracker:
    """Keeps successful PWM writes and frozen trajectories for arbitrary time queries."""
    def __init__(self, centers: dict[int,float] | Any | None = None, limits: dict[int,tuple[float,float]] | None = None):
        # Accept the RobotCalibration object produced by the legacy config
        # loader as a convenience, while keeping a dependency free dict API.
        if centers is not None and not isinstance(centers, dict) and hasattr(centers, "initial_angles_deg"):
            calibration = centers
            limits = limits or {sid: (float(s.get('min_angle',0.0)), float(s.get('max_angle',180.0))) if isinstance(s,dict) else (float(s.min_angle), float(s.max_angle)) for sid, s in calibration.servos.items()}
            centers = calibration.initial_angles_deg
        self.centers = dict(centers or {1:85,2:95,3:95,4:111,5:90,6:143,7:90})
        self.limits = limits or {i:(0.0,180.0) for i in ALL_SERVO_IDS}
        self._times = {i:[] for i in ALL_SERVO_IDS}; self._angles = {i:[] for i in ALL_SERVO_IDS}
        self._segments = []; self._lock = threading.RLock()
    def record_successful_batch(self, t_ns: int, angles_deg: dict[int,float]) -> None:
        with self._lock:
            for sid,a in angles_deg.items():
                if sid not in self._times: raise KeyError(sid)
                lo,hi=self.limits[sid]
                if not lo <= float(a) <= hi: raise ValueError(f"servo {sid} angle out of limits")
                if self._times[sid] and t_ns < self._times[sid][-1]: raise ValueError("timestamps must be monotonic")
                self._times[sid].append(int(t_ns)); self._angles[sid].append(float(a))
    record_successful_command = lambda self, servo_id, write_end_t_ns, angle_deg: self.record_successful_batch(write_end_t_ns,{servo_id:angle_deg})
    def record_reference_segment(self, *args, phase: str = "rl", start_t_ns: int | None = None, end_t_ns: int | None = None, start_angles_deg=None, target_angles_deg=None):
        # Supports both ``(start, end, start_angles, target_angles)`` and the
        # legacy ``(phase, start, end, start_angles, target_angles)`` order.
        if args:
            if isinstance(args[0], str):
                phase, start_t_ns, end_t_ns, start_angles_deg, target_angles_deg = args
            else:
                start_t_ns, end_t_ns, start_angles_deg, target_angles_deg = args
        if start_t_ns is None or end_t_ns is None or start_angles_deg is None or target_angles_deg is None:
            raise TypeError("reference segment requires start/end and angle maps")
        if not start_angles_deg or set(start_angles_deg)!=set(target_angles_deg) or not set(start_angles_deg).issubset(self.centers): raise ValueError("segment servo sets must match known servos")
        if end_t_ns < start_t_ns: raise ValueError("negative segment duration")
        orig, target = dict(start_angles_deg), dict(target_angles_deg)
        duration = (end_t_ns-start_t_ns)/1e9
        evaluate = lambda e: {sid:smooth_motion(orig[sid],target[sid],e,duration) if duration>0 else target[sid] for sid in orig}
        with self._lock: self._segments.append((int(start_t_ns), evaluate))
    def record_trajectory(self, start_t_ns, trajectory):
        """记录包含尖端回摆的完整曲线，不将无反馈估计称作实测。"""
        with self._lock: self._segments.append((int(start_t_ns), trajectory.evaluate_all))
    def _command_at(self,sid,t):
        idx=bisect.bisect_right(self._times[sid],t)-1
        return None if idx<0 else self._angles[sid][idx]
    def get_servo_pose_at(self,t_ns:int) -> RLServoPose:
        t=int(t_ns)
        with self._lock:
            ref=dict(self.centers)
            for start,evaluate in self._segments:
                if t < start: continue
                ref.update(evaluate((t-start)/1e9))
            cmd={sid:self._command_at(sid,t) for sid in ALL_SERVO_IDS}
            estimated={sid:(cmd[sid] if cmd[sid] is not None else ref[sid]) for sid in ALL_SERVO_IDS}
            return RLServoPose(t,ref,cmd,estimated)
    def query(self,t_ns:int)->dict[str,Any]: return self.get_servo_pose_at(t_ns).to_dict()
    def latest_commanded_angles(self):
        with self._lock:
            return {sid:(vals[-1] if vals else self.centers[sid]) for sid,vals in self._angles.items()}

# Name used by a few integrations and by the 20260725 tracker adapter.
ServoStateTracker = RLServoStateTracker
