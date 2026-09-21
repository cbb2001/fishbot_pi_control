"""Immutable episode data and duration-aware generalized advantage estimation."""

from __future__ import annotations
import math
from dataclasses import asdict, dataclass, replace
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class Transition:
    """Endpoint-confirmed action and exact normalized input used by behavior policy."""

    observation: tuple[float, ...]
    action: tuple[int, ...]
    reward: float
    next_observation: tuple[float, ...]
    old_log_prob: float
    old_value: float
    next_value: float
    normalized_observation: tuple[float, ...]
    duration_s: float = 0.2
    behavior_policy_version: int = 0
    episode_index: int = 0
    episode_role: str = "trainable"
    terminated: bool = False
    truncated: bool = False
    included_in_training: bool = True
    exclusion_reason: str | None = None
    reward_mean: float | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    reward_sample_count: int | None = None
    mean_r_depth: float | None = None
    mean_r_heading: float | None = None
    mean_r_pitch: float | None = None
    mean_r_roll: float | None = None
    action_mask: tuple[bool, ...] | None = None

    def __post_init__(self):
        for name in ("observation", "next_observation", "normalized_observation"):
            vals = tuple(float(v) for v in getattr(self, name))
            if not vals or not all(math.isfinite(v) for v in vals):
                raise ValueError(f"{name} 包含 NaN/Inf")
            object.__setattr__(self, name, vals)
        if (
            len(
                {
                    len(self.observation),
                    len(self.next_observation),
                    len(self.normalized_observation),
                }
            )
            != 1
        ):
            raise ValueError("observation 维度不一致")
        if any(isinstance(v, bool) or int(v) != v for v in self.action):
            raise ValueError("action 必须为分类索引")
        object.__setattr__(self, "action", tuple(int(v) for v in self.action))
        if self.action_mask is not None:
            mask = tuple(self.action_mask)
            if not mask or any(not isinstance(v, (bool, np.bool_)) for v in mask) or not any(mask):
                raise ValueError('action_mask 必须为非空布尔掩码')
            object.__setattr__(self, 'action_mask', tuple(bool(v) for v in mask))
        for name in ("reward", "old_log_prob", "old_value", "next_value", "duration_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 包含 NaN/Inf")
        if self.duration_s <= 0:
            raise ValueError("duration_s 必须大于 0")
        if self.episode_role not in ("trainable", "bridge"):
            raise ValueError("未知 episode_role")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FrozenRollout:
    transitions: tuple[Transition, ...]
    source_policy_version: int

    def __post_init__(self):
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not self.transitions:
            raise ValueError("不能冻结空 rollout")
        if any(
            t.behavior_policy_version != self.source_policy_version
            for t in self.transitions
        ):
            raise ValueError("rollout 混合 policy 版本")
        if any(not t.included_in_training for t in self.transitions):
            raise ValueError("排除 transition 不能训练")

    @property
    def source_episode_indices(self):
        return tuple(dict.fromkeys(t.episode_index for t in self.transitions))

    def __len__(self):
        return len(self.transitions)


def compute_gae(
    transitions: Iterable[Transition],
    gamma=0.99,
    gae_lambda=0.95,
    time_aware_discount=True,
    reference_dt_s=0.2,
):
    """Use duration-aware gamma and lambda; truncation bootstraps but stops trace."""
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1 and reference_dt_s > 0):
        raise ValueError("非法 GAE 配置")
    items = tuple(transitions)
    adv = np.zeros(len(items), dtype=np.float64)
    following = 0.0
    for i in range(len(items) - 1, -1, -1):
        tr = items[i]
        exponent = tr.duration_s / reference_dt_s if time_aware_discount else 1.0
        discount = gamma**exponent
        trace = gae_lambda**exponent
        delta = (
            tr.reward
            + discount * (0.0 if tr.terminated else tr.next_value)
            - tr.old_value
        )
        same = (
            i + 1 < len(items)
            and items[i + 1].episode_index == tr.episode_index
            and items[i + 1].behavior_policy_version == tr.behavior_policy_version
        )
        following = delta + (
            discount * trace * following
            if same and not (tr.terminated or tr.truncated)
            else 0.0
        )
        adv[i] = following
    returns = adv + np.asarray([t.old_value for t in items])
    if not np.isfinite(adv).all() or not np.isfinite(returns).all():
        raise ValueError("GAE 产生 NaN/Inf")
    return returns.astype(np.float32), adv.astype(np.float32)


class RolloutBuffer:
    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        self._items = []

    def add(self, transition):
        if len(self._items) >= self.capacity:
            raise OverflowError("rollout 达到容量上限")
        if (
            self._items
            and self._items[0].behavior_policy_version
            != transition.behavior_policy_version
        ):
            raise ValueError("rollout 混合 policy")
        self._items.append(transition)

    def end_episode(self, terminated=False):
        if self._items:
            self._items[-1] = replace(
                self._items[-1], terminated=bool(terminated), truncated=not terminated
            )

    def freeze(self):
        if not self._items:
            raise ValueError("不能冻结空 rollout")
        return FrozenRollout(tuple(self._items), self._items[0].behavior_policy_version)

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)
