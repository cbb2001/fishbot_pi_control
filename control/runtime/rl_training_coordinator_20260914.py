"""Episode-end pair updates and atomic pair publication."""

from __future__ import annotations
from dataclasses import dataclass, replace
import threading
import math
import time
from .rl_rollout_buffer_20260914 import FrozenRollout, Transition
from .rl_ppo_update_worker_20260914 import PPOUpdateWorker


@dataclass(frozen=True)
class EpisodeContext:
    episode_index: int
    episode_role: str
    published_policy_version: int


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    episode_role: str
    published_policy_version: int
    ppo_update_started: bool
    policy_published_at_end: bool
    candidate_ready_at_end: bool
    tail_trainable_transitions: int
    fin_trainable_transitions: int


class RLTrainingCoordinator:
    def __init__(self, tail_agent, fin_agent, config=None, update_worker=None):
        self.tail_agent = tail_agent
        self.fin_agent = fin_agent
        self.config = config or {}
        self._lock = threading.RLock()
        self.episode_index = -1
        self.episode_role = "trainable"
        self.running = False
        self.published_policy_version = tail_agent.policy_version
        self.pending = {"tail": [], "fin": []}
        self.current = {"tail": [], "fin": []}
        self._result = None
        self.tail_transition_count = 0
        self.fin_transition_count = 0
        self.last_results = []
        if tail_agent.policy_version != fin_agent.policy_version:
            raise ValueError("Tail/Fin published policy 版本不同")
        train = self.config.get("training", self.config)
        for flag in (
            "background_update",
            "publish_only_at_episode_boundary",
            "require_both_agents_ready_for_publish",
            "exclude_bridge_episodes_from_training",
        ):
            if train.get(flag, True) is not True:
                raise ValueError(f"本实现要求 {flag}=true")
        if train.get("update_trigger", "episode_end") != "episode_end":
            raise ValueError("仅支持 episode_end PPO update")
        self.max_rollout_size = int(train.get("max_rollout_size", 100000))
        if self.max_rollout_size <= 0:
            raise ValueError("max_rollout_size 必须为正整数")
        self.worker = update_worker or PPOUpdateWorker(
            {"tail": tail_agent, "fin": fin_agent}
        )

    def begin_episode(self, episode_index=None):
        if self.running:
            raise RuntimeError("Episode 已开始")
        self.episode_index = (
            self.episode_index + 1 if episode_index is None else int(episode_index)
        )
        self.episode_role = (
            "bridge" if self.worker.busy or self._result is not None else "trainable"
        )
        self.current = {"tail": [], "fin": []}
        self.running = True
        return EpisodeContext(
            self.episode_index, self.episode_role, self.published_policy_version
        )

    start_episode = begin_episode

    def add_transition(self, name, transition):
        if not self.running:
            raise RuntimeError("没有正在运行的 Episode")
        if name not in self.current:
            raise ValueError("agent 必须是 tail 或 fin")
        if transition.behavior_policy_version != self.published_policy_version:
            raise ValueError("transition policy 版本错误")
        if transition.episode_index != self.episode_index:
            raise ValueError("transition Episode 错误")
        if len(self.current[name]) + len(self.pending[name]) >= self.max_rollout_size:
            raise OverflowError("rollout 达到容量上限")
        if self.episode_role == "bridge":
            transition = replace(
                transition,
                episode_role="bridge",
                included_in_training=False,
                exclusion_reason="bridge_episode",
            )
        elif transition.episode_role != "trainable":
            raise ValueError("transition episode_role 与当前 Episode 不一致")
        self.current[name].append(transition)
        (self.tail_agent if name == "tail" else self.fin_agent).environment_steps += 1
        if name == "tail":
            self.tail_transition_count += 1
        else:
            self.fin_transition_count += 1
        return transition

    record_transition = add_transition

    def process_update_results(self):
        r = self.worker.poll()
        if r is not None:
            if r.error:
                raise RuntimeError(f"PPO 后台训练失败:\n{r.error}")
            self._result = r
            self.last_results = [r]
        return [] if r is None else [r]

    def end_episode(self, terminated=False, termination_reason="duration"):
        if not self.running:
            raise RuntimeError("Episode 未开始")
        self.running = False
        self.process_update_results()
        published = False
        started = False
        counts = {"tail": 0, "fin": 0}
        ready = self._result is not None
        behavior_version = self.published_policy_version
        if self.episode_role == "trainable":
            for name in ("tail", "fin"):
                items = self.current[name]
                if items:
                    items[-1] = replace(
                        items[-1], terminated=bool(terminated), truncated=not terminated
                    )
                eligible = [t for t in items if t.included_in_training]
                self.pending[name].extend(eligible)
                counts[name] = len(eligible)
        if self._result is not None and not terminated:
            if self._result.source_policy_version != self.published_policy_version:
                raise RuntimeError("candidate source policy 已过时")
            if not (self._result.tail_metrics and self._result.fin_metrics):
                raise RuntimeError("必须同时完成两个 candidate")
            if any(
                a.candidate_model is None
                or a.candidate_policy_version != behavior_version + 1
                for a in (self.tail_agent, self.fin_agent)
            ):
                raise RuntimeError("candidate 版本或模型缺失")
            with self._lock:
                self.tail_agent.publish_candidate()
                self.fin_agent.publish_candidate()
                self.published_policy_version += 1
            self._result = None
            self.pending = {"tail": [], "fin": []}
            published = True
            self.worker.tail_candidate_ready = self.worker.fin_candidate_ready = False
        elif not terminated and not self.worker.busy:
            if (
                len(self.pending["tail"])
                >= self.tail_agent.config.min_transitions_per_update
                and len(self.pending["fin"])
                >= self.fin_agent.config.min_transitions_per_update
            ):
                started = self.worker.submit_pair(
                    FrozenRollout(
                        tuple(self.pending["tail"]), self.published_policy_version
                    ),
                    FrozenRollout(
                        tuple(self.pending["fin"]), self.published_policy_version
                    ),
                    self.published_policy_version,
                )
                if started:
                    self.pending = {"tail": [], "fin": []}
        return EpisodeResult(
            self.episode_index,
            self.episode_role,
            behavior_version,
            started,
            published,
            ready,
            counts["tail"],
            counts["fin"],
        )

    finish_episode = end_episode

    def act_tail(self, o, deterministic=False):
        with self._lock:
            return self.tail_agent.act(o, deterministic)

    def act_fin(self, o, deterministic=False, *, action_mask=None):
        with self._lock:
            return self.fin_agent.act(o, deterministic, action_mask=action_mask)

    def snapshot(self):
        return {
            "episode_index": self.episode_index,
            "episode_role": self.episode_role,
            "published_policy_version": self.published_policy_version,
            "candidate_training_active": self.worker.busy or self.worker.active,
            "tail_candidate_ready": self.worker.tail_candidate_ready,
            "fin_candidate_ready": self.worker.fin_candidate_ready,
            "tail_rollout_size": len(self.pending["tail"])
            + (len(self.current["tail"]) if self.running else 0),
            "fin_rollout_size": len(self.pending["fin"])
            + (len(self.current["fin"]) if self.running else 0),
            "tail_transition_count": self.tail_transition_count,
            "fin_transition_count": self.fin_transition_count,
            "tail_update_count": self.tail_agent.update_count,
            "fin_update_count": self.fin_agent.update_count,
            "latest_tail_policy_loss": self.tail_agent.latest_metrics.get(
                "policy_loss"
            ),
            "latest_tail_value_loss": self.tail_agent.latest_metrics.get("value_loss"),
            "latest_fin_policy_loss": self.fin_agent.latest_metrics.get("policy_loss"),
            "latest_fin_value_loss": self.fin_agent.latest_metrics.get("value_loss"),
        }

    def request_checkpoint(self, directory, metadata=None):
        context = {
            "episode_index": self.episode_index,
            "next_episode_index": self.episode_index + 1,
            "tail_transition_count": self.tail_transition_count,
            "fin_transition_count": self.fin_transition_count,
        }
        context.update(metadata or {})
        self.worker.request_checkpoint(directory, context)

    def load_checkpoint(self, directory):
        if self.running or self.worker.busy:
            raise RuntimeError("resume 只能在训练启动前执行")
        from .rl_ppo_agent_20260914 import load_checkpoint_pair

        metadata = load_checkpoint_pair(directory, self.tail_agent, self.fin_agent)
        self.published_policy_version = self.tail_agent.policy_version
        self.episode_index = int(metadata.get("next_episode_index", 0)) - 1
        self.tail_transition_count = self.tail_agent.environment_steps
        self.fin_transition_count = self.fin_agent.environment_steps
        return metadata

    def close(self):
        self.worker.stop()


TrainingCoordinator = RLTrainingCoordinator


class FinTrainingCoordinator:
    """Fin-only collection/update cycle with no bridge actions or Tail objects.

    Caller completes the current action, freezes here, and holds its endpoint
    while polling. Publishing is permitted as soon as the update succeeds.
    Sensors and actuator health remain the runner's responsibility while waiting.
    """

    def __init__(self, agent, config=None):
        from .rl_ppo_update_worker_20260914 import FinPPOUpdateWorker

        if tuple(agent.action_dims) != (18,):
            raise ValueError("Fin-only trainer 要求 18 动作带掩码的 Fin Agent")
        self.agent = agent
        self.config = config or {}
        training = self.config.get("training", {})
        every = training.get("update_every_actions", 5)
        if isinstance(every, bool) or int(every) != every or every < 1:
            raise ValueError("update_every_actions 必须是正整数")
        self.update_every_actions = int(every)
        if self.update_every_actions < agent.config.min_transitions_per_update:
            raise ValueError("update_every_actions 不能小于 min_transitions_per_update")
        timeout = self.config.get("runtime", {}).get("update_timeout_s", 2.0)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("update_timeout_s 必须为正有限数")
        self.update_timeout_s = float(timeout)
        self.worker = FinPPOUpdateWorker(agent)
        self.pending = []
        self.last_rollout = None
        self.last_metrics = None
        self.episode_index = -1
        self.next_episode_index = 0
        self.running = False
        self._closed = False
        self._cancel_reason = None
        self._fault = None
        self._force_update = False
        self._episode_transitions = 0
        self._episode_updates = 0
        self._episode_discarded = 0
        self._discard_reason = None

    @property
    def published_policy_version(self):
        return self.agent.policy_version

    @property
    def updating(self):
        return self.worker.busy

    @property
    def ready(self):
        return not self.updating and len(self.pending) >= self.update_every_actions

    def begin_episode(self, index=None):
        if self._closed or self._fault:
            raise RuntimeError("Fin trainer 已关闭或处于故障状态")
        if self.running or self.updating or self.pending:
            raise RuntimeError("必须先结束前一 Episode 并清空 rollout")
        self.episode_index = self.next_episode_index if index is None else int(index)
        if self.episode_index < 0:
            raise ValueError("episode_index 必须非负")
        self.next_episode_index = self.episode_index + 1
        self.running = True
        self._episode_transitions = self._episode_updates = self._episode_discarded = 0
        self._discard_reason = None
        return EpisodeContext(
            self.episode_index, "trainable", self.published_policy_version
        )

    def add_transition(self, transition):
        if not self.running or self.updating or self._closed or self._fault:
            raise RuntimeError("Fin update/暂停期间禁止采集动作 transition")
        if transition.behavior_policy_version != self.agent.policy_version:
            raise ValueError("Fin rollout 不能混合 policy 版本")
        if transition.episode_index != self.episode_index:
            raise ValueError("Fin rollout 不能跨 Episode")
        if (
            not transition.included_in_training
            or transition.episode_role != "trainable"
        ):
            raise ValueError("Fin-only rollout 只接受 trainable transition")
        if len(self.pending) >= self.update_every_actions:
            raise RuntimeError("已达到 update_every_actions，应暂停动作并更新")
        self.agent.validate_transition_action(transition)
        self.pending.append(transition)
        self.agent.environment_steps += 1
        self._episode_transitions += 1
        return transition

    def start_update(self, force=False):
        if self._closed or self._fault:
            raise RuntimeError("Fin trainer 已关闭或故障")
        if self.updating:
            return False
        if not self.running:
            raise RuntimeError("Episode 尚未开始")
        threshold = (
            self.agent.config.min_transitions_per_update
            if force
            else self.update_every_actions
        )
        if len(self.pending) < threshold:
            return False
        # The hold during learning is not a sampled action. Cut this rollout at
        # its final action endpoint, while preserving V(real_next_observation).
        self.pending[-1] = replace(
            self.pending[-1], truncated=not self.pending[-1].terminated
        )
        rollout = FrozenRollout(tuple(self.pending), self.agent.policy_version)
        if not self.worker.submit(rollout):
            return False
        self.last_rollout = rollout
        self.pending.clear()
        self._force_update = bool(force)
        self._cancel_reason = None
        return True

    def cancel_update(self, reason="cancelled"):
        self._cancel_reason = str(reason)
        self.worker.cancel()

    def poll_update(self):
        if (
            self.updating
            and self._cancel_reason is None
            and time.monotonic() - self.worker.started_t > self.update_timeout_s
        ):
            self.cancel_update("update_timeout")
            self._fault = "update_timeout"
            raise TimeoutError("Fin PPO update 超过 update_timeout_s，已请求取消")
        result = self.worker.poll()
        if result is None:
            return None
        if result.error:
            self._fault = result.error
            raise RuntimeError(f"Fin PPO 后台训练失败:\n{result.error}")
        if result.cancelled or self._cancel_reason:
            self.worker._clear_candidate()
            return {
                "cancelled": True,
                "reason": self._cancel_reason or "cancelled",
                "published": False,
                "source_policy_version": result.source_policy_version,
            }
        if (
            result.source_policy_version != self.agent.policy_version
            or self.agent.candidate_policy_version != self.agent.policy_version + 1
        ):
            self._fault = "stale_candidate"
            self.worker._clear_candidate()
            raise RuntimeError("Fin candidate policy 来源版本错误")
        if not self.agent.publish_candidate():
            raise RuntimeError("Fin candidate 缺失")
        self._episode_updates += 1
        self.last_metrics = dict(
            result.metrics,
            published=True,
            published_policy_version=self.agent.policy_version,
            rollout_boundary_reason=(
                "episode_end" if self._force_update else "update_hold"
            ),
            update_hold_excluded_from_rollout=True,
        )
        return self.last_metrics

    def discard_pending(self, reason):
        count = len(self.pending)
        self.pending.clear()
        self._episode_discarded += count
        self._discard_reason = str(reason) if count else self._discard_reason
        return {
            "discarded_transitions": count,
            "discard_reason": str(reason) if count else None,
        }

    def end_episode(self, terminated=False):
        if not self.running or self.updating:
            raise RuntimeError("必须等待 Fin update 结束后再关闭 Episode")
        if (
            not terminated
            and len(self.pending) >= self.agent.config.min_transitions_per_update
        ):
            raise RuntimeError(
                "Episode 尾批达到最小样本量，应先 start_update(force=True)"
            )
        self.discard_pending("terminated" if terminated else "insufficient_tail_batch")
        self.running = False
        return {
            "episode_index": self.episode_index,
            "next_episode_index": self.next_episode_index,
            "episode_role": "trainable",
            "published_policy_version": self.agent.policy_version,
            "fin_steps": self._episode_transitions,
            "fin_updates": self._episode_updates,
            "discarded_transitions": self._episode_discarded,
            "discard_reason": self._discard_reason,
            "terminated": bool(terminated),
        }

    def snapshot(self):
        return {
            "episode_index": self.episode_index,
            "next_episode_index": self.next_episode_index,
            "episode_role": "trainable",
            "published_policy_version": self.agent.policy_version,
            "fin_rollout_size": len(self.pending),
            "fin_update_count": self.agent.update_count,
            "fin_transition_count": self.agent.environment_steps,
            "candidate_training_active": self.updating,
            "fin_candidate_ready": self.agent.candidate_model is not None,
            "latest_fin_policy_loss": self.agent.latest_metrics.get("policy_loss"),
            "latest_fin_value_loss": self.agent.latest_metrics.get("value_loss"),
            "update_wait_elapsed_s": (
                0.0 if not self.updating else time.monotonic() - self.worker.started_t
            ),
        }

    def save_checkpoint(self, directory, next_episode_index=None, metadata=None):
        if self.updating or self.agent.candidate_model is not None:
            raise RuntimeError("Fin 更新期间不能保存 checkpoint")
        from .rl_ppo_agent_20260914 import save_fin_checkpoint

        return save_fin_checkpoint(
            directory,
            self.agent,
            (
                self.next_episode_index
                if next_episode_index is None
                else next_episode_index
            ),
            metadata,
        )

    def load_checkpoint(self, directory):
        if self.running or self.updating or self.pending:
            raise RuntimeError("resume 只允许在空闲启动阶段执行")
        from .rl_ppo_agent_20260914 import load_fin_checkpoint

        metadata = load_fin_checkpoint(directory, self.agent)
        self.next_episode_index = metadata["next_episode_index"]
        self.episode_index = self.next_episode_index - 1
        return metadata

    def close(self, timeout_s=10.0):
        self.worker.close(timeout_s)
        self._closed = True
