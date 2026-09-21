"""Logical episode endings from continuous sensor evidence and monotonic time.

The first decision is latched until ``start``. Callers may finish the current
action endpoint before consuming it; no physical reset or actuator command is
performed here. Feed every buffered RL state in timestamp order, including
states collected while PPO runs, and call ``check_time`` even without samples.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

from .rl_sensor_calibration_20260914 import finite_float, wrap180


@dataclass(frozen=True)
class EpisodeDecision:
    reason: str
    terminated: bool
    truncated: bool
    episode_index: int
    end_ns: int
    elapsed_s: float

    def to_dict(self):
        return asdict(self)

    as_dict = to_dict


class EpisodeMonitor:
    """Require joint depth/yaw tolerance over fresh, uninterrupted samples.

    Configuration accepts the whole config, a training mapping, or its
    termination subsection. Flat training keys remain the primary interface.
    For RLState objects, hold time advances on the older of the IMU/depth
    sample timestamps, so repeated synchronized snapshots cannot create new
    evidence. Lightweight mappings without sensor timestamps use ``t_ns``.
    """
    def __init__(self, config=None, *, target_depth_m=None, target_heading_deg=None):
        root = dict(config or {})
        training = dict(root.get('training', root))
        cfg = {**training, **training.get('termination', {})}
        targets = root.get('targets', {})
        self.target_depth_m = finite_float(
            targets.get('depth_m', cfg.get('target_depth_m', 0.))
            if target_depth_m is None else target_depth_m, 'target_depth_m')
        self.target_heading_deg = finite_float(
            targets.get('heading_deg', cfg.get('target_heading_deg', 0.))
            if target_heading_deg is None else target_heading_deg, 'target_heading_deg')
        for name, default, positive in (
            ('episode_duration_s', 60., True),
            ('depth_tolerance_m', .05, False),
            ('yaw_tolerance_deg', 5., False),
            ('success_hold_s', 2., False),
            ('min_episode_s', 2., False),
            ('max_sample_gap_s', .2, True),
        ):
            value = finite_float(cfg.get(name, default), name)
            if value < 0 or (positive and value == 0):
                raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
            setattr(self, name, value)
        count = cfg.get('min_actions', 1)
        if isinstance(count, bool) or finite_float(count, 'min_actions') < 0 or int(count) != count:
            raise ValueError('min_actions must be a nonnegative integer')
        self.min_actions = int(count)
        self.success_enabled = bool(cfg.get('success_enabled', True))
        self._duration_ns = round(self.episode_duration_s * 1e9)
        self._hold_ns = round(self.success_hold_s * 1e9)
        self._minimum_ns = round(self.min_episode_s * 1e9)
        self._gap_ns = round(self.max_sample_gap_s * 1e9)
        self.episode_index = None
        self.start_ns = None
        self.decision = None

    @staticmethod
    def _timestamp(value):
        if isinstance(value, bool) or not math.isfinite(value) or int(value) != value:
            raise ValueError('timestamp must be a finite integer')
        return int(value)

    def start(self, index, start_ns):
        self.episode_index = int(index)
        self.start_ns = self._timestamp(start_ns)
        self.decision = None
        self._last_state_ns = None
        self._last_evidence_ns = None
        self._last_sensor_ns = {}
        self._reset_hold()

    def _reset_hold(self):
        self._hold_start_ns = None
        self._hold_samples = 0

    def _finish(self, reason, end_ns):
        self.decision = EpisodeDecision(
            reason=reason, terminated=reason == 'success', truncated=reason == 'time_limit',
            episode_index=self.episode_index, end_ns=end_ns,
            elapsed_s=(end_ns - self.start_ns) / 1e9)
        return self.decision

    def check_time(self, now_ns, *, action_count=0):
        """Wall-clock duration includes inference/update waits and idle time."""
        if self.start_ns is None:
            raise RuntimeError('start an episode before checking termination')
        if self.decision is not None:
            return self.decision
        now_ns = self._timestamp(now_ns)
        if now_ns - self.start_ns >= self._duration_ns:
            return self._finish('time_limit', now_ns)
        return None

    def observe(self, state, *, action_count=0):
        if self.start_ns is None:
            raise RuntimeError('start an episode before observing states')
        if self.decision is not None:
            return self.decision

        def get(key, default=None):
            return state.get(key, default) if isinstance(state, Mapping) else getattr(state, key, default)

        try:
            now_ns = self._timestamp(get('t_ns'))
        except (TypeError, ValueError, OverflowError):
            self._reset_hold()
            return None
        decision = self.check_time(now_ns)
        if decision is not None or not self.success_enabled:
            return decision
        if now_ns < self.start_ns or (self._last_state_ns is not None and now_ns < self._last_state_ns):
            self._reset_hold()
            return None
        if now_ns == self._last_state_ns:
            return None
        if self._last_state_ns is not None and now_ns - self._last_state_ns > self._gap_ns:
            self._reset_hold()
        self._last_state_ns = now_ns

        try:
            depth = finite_float(get('depth_m'), 'depth_m')
            yaw = finite_float(get('yaw_deg'), 'yaw_deg')
            observation = get('observation')
            if observation is not None and not all(math.isfinite(v) for v in observation):
                raise ValueError('nonfinite observation')
            stamps = get('sensor_sample_t_ns')
            if stamps is None:
                evidence_ns = now_ns
            else:
                stamps = {name: self._timestamp(stamps[name]) for name in ('imu', 'depth')}
                for name, stamp in stamps.items():
                    if not self.start_ns <= stamp <= now_ns or now_ns - stamp > self._gap_ns:
                        raise ValueError('sensor evidence is stale, future, or predates episode')
                    if stamp < self._last_sensor_ns.get(name, stamp):
                        raise ValueError('sensor timestamp moved backwards')
                self._last_sensor_ns = stamps
                evidence_ns = min(stamps.values())
        except (TypeError, ValueError, KeyError, OverflowError):
            self._reset_hold()
            return None

        # Tiny numerical allowance keeps exact decimal tolerance edges inclusive.
        in_target = (abs(depth - self.target_depth_m) <= self.depth_tolerance_m + 1e-12
                     and abs(wrap180(yaw - self.target_heading_deg)) <= self.yaw_tolerance_deg + 1e-12)
        if not in_target:
            self._reset_hold()
            self._last_evidence_ns = evidence_ns
            return None
        if self._last_evidence_ns is not None:
            if evidence_ns < self._last_evidence_ns:
                self._reset_hold()
                return None
            if evidence_ns == self._last_evidence_ns:
                return None
            if evidence_ns - self._last_evidence_ns > self._gap_ns:
                self._reset_hold()
        self._last_evidence_ns = evidence_ns
        if self._hold_start_ns is None:
            self._hold_start_ns = evidence_ns
        self._hold_samples += 1
        if (self._hold_samples >= 2 and evidence_ns - self._hold_start_ns >= self._hold_ns
                and now_ns - self.start_ns >= self._minimum_ns and action_count >= self.min_actions):
            return self._finish('success', now_ns)
        return None
