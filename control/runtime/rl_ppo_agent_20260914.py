"""CPU PPO agents with immutable published observation normalizers."""

from __future__ import annotations
import copy, os, random, math, json, hashlib, uuid
from pathlib import Path
from dataclasses import asdict, dataclass
import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except Exception as e:
    torch = None
    nn = None
    Categorical = None
    _TORCH_ERROR = e
from .rl_rollout_buffer_20260914 import Transition, FrozenRollout, compute_gae
from .rl_actions_20260914 import FIN_THETAS, FIN_DURATIONS, FIN_MODES, FinActionSpace, FinSequenceState

TAIL_THETA_VALUES = (-30, -20, -10, 0, 10, 20, 30)
TAIL_T_VALUES = (0.2, 0.3, 0.4, 0.5, 0.6)
FIN_THETA_VALUES = FIN_THETAS
FIN_T_VALUES = FIN_DURATIONS


@dataclass
class PPOConfig:
    actor_hidden_sizes: tuple = (128, 128)
    critic_hidden_sizes: tuple = (128, 128)
    activation: str = "tanh"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 32
    update_epochs: int = 10
    min_transitions_per_update: int = 64
    time_aware_discount: bool = True
    discount_reference_dt_s: float = 0.2
    target_kl: float | None = None

    @classmethod
    def from_dict(cls, d):
        extra = set(d or {}) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f"未知 PPO 配置: {sorted(extra)}")
        return cls(**(d or {}))

    def __post_init__(self):
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl 必须为 null 或正有限数")
        for name in ("actor_hidden_sizes", "critic_hidden_sizes"):
            sizes = tuple(getattr(self, name))
            if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in sizes):
                raise ValueError("网络隐藏层必须是正整数")
            setattr(self, name, sizes)
        if self.activation not in ("tanh", "relu", "elu"):
            raise ValueError("activation 必须为 tanh/relu/elu")
        for name in ("learning_rate", "max_grad_norm", "discount_reference_dt_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正有限数")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须在 [0,1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio 必须在 (0,1)")
        for name in ("entropy_coef", "value_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 非法")
        for name in ("minibatch_size", "update_epochs", "min_transitions_per_update"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v or v <= 0:
                raise ValueError(f"{name} 必须为正整数")
            setattr(self, name, int(v))


@dataclass(frozen=True)
class RunningMeanStd:
    mean: np.ndarray
    var: np.ndarray
    count: float

    def __post_init__(self):
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        var = np.array(self.var, dtype=np.float64, copy=True)
        if (
            mean.ndim != 1
            or mean.shape != var.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(var).all()
            or (var < 0).any()
            or not math.isfinite(self.count)
            or self.count <= 0
        ):
            raise ValueError("非法 normalizer 状态")
        mean.setflags(write=False)
        var.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "var", var)

    @classmethod
    def create(cls, n):
        return cls(np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64), 1e-4)

    def update(self, x):
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if (
            a.ndim != 2
            or a.shape[1] != len(self.mean)
            or not len(a)
            or not np.isfinite(a).all()
        ):
            raise ValueError("normalizer 样本维度错误或 NaN/Inf")
        n = len(a)
        m = a.mean(0)
        v = a.var(0)
        d = m - self.mean
        total = self.count + n
        mean = self.mean + d * n / total
        var = (self.var * self.count + v * n + d * d * self.count * n / total) / total
        return RunningMeanStd(mean, var, total)


class NormalizedInputAdapter(nn.Module if nn else object):
    """Keep learned network coordinates fixed while running statistics evolve.

    Clipping happens AFTER this affine transform. Clipping each newly normalized
    vector before compensation would change the policy outside the clip range.
    Only buffers change, so Adam's learned-parameter moments remain valid.
    """

    def __init__(self, width, clip):
        super().__init__()
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("offset", torch.zeros(width))
        self.clip = float(clip)

    def forward(self, x):
        return torch.clamp(x * self.scale + self.offset, -self.clip, self.clip)

    def adapt_statistics(self, old, new):
        ratio = torch.tensor(
            np.sqrt(new.var + 1e-8) / np.sqrt(old.var + 1e-8), dtype=self.scale.dtype
        )
        shift = torch.tensor(
            (new.mean - old.mean) / np.sqrt(old.var + 1e-8), dtype=self.scale.dtype
        )
        with torch.no_grad():
            self.offset.add_(self.scale * shift)
            self.scale.mul_(ratio)


def _mlp(inp, hid, out, act, clip):
    layers = [NormalizedInputAdapter(inp, clip)]
    last = inp
    A = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[act]
    for h in hid:
        layers += [nn.Linear(last, int(h)), A()]
        last = int(h)
    return nn.Sequential(*layers, nn.Linear(last, out))


class _Net(nn.Module if nn else object):
    def __init__(self, n, d, c, clip=10.0):
        super().__init__()
        self.d = tuple(d)
        self.actor = _mlp(n, c.actor_hidden_sizes, sum(d), c.activation, clip)
        self.critic = _mlp(n, c.critic_hidden_sizes, 1, c.activation, clip)

    def distributions(self, o, action_masks=None):
        logits = self.actor(o)
        if action_masks is not None:
            if len(self.d) != 1 or action_masks.dtype != torch.bool or action_masks.shape != logits.shape or not action_masks.any(dim=-1).all():
                raise ValueError('非法联合动作 mask：维度错误或无合法动作')
            logits = logits.masked_fill(~action_masks, -torch.inf)
        return [Categorical(logits=x) for x in torch.split(logits, self.d, dim=-1)]

    def evaluate(self, o, a, action_masks=None):
        ds = self.distributions(o, action_masks)
        if action_masks is not None and not action_masks.gather(1, a).all():
            raise ValueError('rollout action 被行为策略 mask 禁止')
        lp = sum(x.log_prob(a[:, i]) for i, x in enumerate(ds))
        en = sum(x.entropy() for x in ds)
        return lp, en, self.critic(o).squeeze(-1)


@dataclass(frozen=True)
class ActionDecision:
    action: tuple[int, ...]
    log_prob: float
    value: float
    normalized_observation: tuple[float, ...]
    policy_version: int
    action_mask: tuple[bool, ...] | None = None


class BasePPOAgent:
    action_dims = ()
    action_names = ()
    requires_action_mask = False

    def __init__(
        self,
        observation_dim=None,
        config=None,
        seed=None,
        history_length=0,
        normalized_clip=10.0,
    ):
        if torch is None:
            raise RuntimeError(f"PyTorch is required for PPO training: {_TORCH_ERROR}")
        self.history_length = int(history_length)
        if self.history_length < 0 or self.history_length != history_length:
            raise ValueError("history_length 必须为非负整数")
        if observation_dim is None:
            observation_dim = 15 + self.history_length * (
                3 if self.action_dims == (7, 5) else 5
            )
        if int(observation_dim) != observation_dim or observation_dim <= 0:
            raise ValueError("observation_dim 必须为正整数")
        if not math.isfinite(normalized_clip) or normalized_clip <= 0:
            raise ValueError("normalized_clip 必须为正有限数")
        self.normalized_clip = float(normalized_clip)
        self.observation_dim = int(observation_dim)
        self.config = (
            PPOConfig.from_dict(config)
            if isinstance(config, dict)
            else (config or PPOConfig())
        )
        self.seed = int(
            seed if seed is not None else random.SystemRandom().randrange(2**31)
        )
        if not 0 <= self.seed < 2**63 - 1 or (seed is not None and self.seed != seed):
            raise ValueError("seed 必须为非负整数，且小于 2**63-1")
        self.inference_rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.update_rng = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.published_model = _Net(
                self.observation_dim,
                self.action_dims,
                self.config,
                self.normalized_clip,
            )
        self.published_model.eval()
        self.training_model = copy.deepcopy(self.published_model)
        self.candidate_model = None
        self.optimizer = torch.optim.Adam(
            self.training_model.parameters(), lr=self.config.learning_rate
        )
        self.normalizer = RunningMeanStd.create(self.observation_dim)
        self.candidate_normalizer = None
        self.policy_version = 0
        self.update_count = 0
        self.environment_steps = 0
        self.published_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.candidate_optimizer_state = None
        self.candidate_policy_version = None
        self.latest_metrics = {}
        self.normalizer_initialized = False
        self._inference_started = False

    def initialize_normalizer(self, observations):
        """Warm up once with raw observations before any policy inference.

        A variance floor of one keeps a single stationary calibration frame from
        turning sensor noise into enormous normalized inputs.
        """
        if (
            self.normalizer_initialized
            or self._inference_started
            or self.policy_version
            or self.environment_steps
        ):
            raise RuntimeError("normalizer 只能在全新训练首次 inference 前初始化")
        data = np.asarray(observations, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if (
            data.ndim != 2
            or data.shape[1] != self.observation_dim
            or not len(data)
            or not np.isfinite(data).all()
        ):
            raise ValueError("normalizer warmup 维度错误或 NaN/Inf")
        self.normalizer = RunningMeanStd(
            data.mean(0), np.maximum(data.var(0), 1.0), float(len(data))
        )
        self.normalizer_initialized = True

    def normalize(self, x, normalizer=None):
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (self.observation_dim,) or not np.isfinite(x).all():
            raise ValueError("observation 维度错误或 NaN/Inf")
        n = normalizer or self.normalizer
        result = ((x - n.mean) / np.sqrt(n.var + 1e-8)).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("normalized observation NaN/Inf")
        return result

    def validate_action_mask(self, action_mask):
        if action_mask is None:
            if self.requires_action_mask:
                raise ValueError('Fin policy 必须提供当前序列 action_mask')
            return None
        mask = np.asarray(action_mask)
        if len(self.action_dims) != 1 or mask.dtype != np.bool_ or mask.shape != self.action_dims or not mask.any():
            raise ValueError('非法 action_mask：要求非空布尔联合动作掩码')
        return tuple(bool(v) for v in mask)

    def act(self, observation, deterministic=False, *, action_mask=None):
        mask = self.validate_action_mask(action_mask)
        self._inference_started = True
        no = self.normalize(observation)
        o = torch.as_tensor(no).unsqueeze(0)
        with torch.no_grad():
            ds = self.published_model.distributions(
                o, None if mask is None else torch.tensor([mask], dtype=torch.bool)
            )
            vals = [
                (
                    torch.argmax(d.probs, -1)
                    if deterministic
                    else torch.multinomial(
                        d.probs, 1, generator=self.inference_rng
                    ).squeeze(-1)
                )
                for d in ds
            ]
            a = torch.stack(vals, 1)
            lp = sum(d.log_prob(v) for d, v in zip(ds, vals))
            v = self.published_model.critic(o).squeeze(-1)
        if not torch.isfinite(lp).all() or not torch.isfinite(v).all():
            raise ValueError("PPO inference 产生 NaN/Inf")
        return ActionDecision(
            tuple(int(x) for x in a[0]),
            float(lp[0]),
            float(v[0]),
            tuple(float(x) for x in no),
            self.policy_version,
            mask,
        )

    def value(self, observation):
        self._inference_started = True
        with torch.no_grad():
            v = float(
                self.published_model.critic(
                    torch.as_tensor(self.normalize(observation)).unsqueeze(0)
                ).item()
            )
        if not math.isfinite(v):
            raise ValueError("PPO value 产生 NaN/Inf")
        return v

    def validate_transition_action(self, transition):
        self.decode_action(transition.action)
        mask = self.validate_action_mask(transition.action_mask)
        if mask is not None and not mask[transition.action[0]]:
            raise ValueError('rollout action 被行为策略 mask 禁止')

    sample_action = act
    get_action = act

    def decode_action(self, a):
        if len(a) != len(self.action_dims) or any(
            int(v) != v or not 0 <= v < dim for v, dim in zip(a, self.action_dims)
        ):
            raise ValueError("动作分类索引越界")
        tab = (
            (TAIL_THETA_VALUES, TAIL_T_VALUES)
            if self.action_dims == (7, 5)
            else (FIN_THETA_VALUES, FIN_T_VALUES, (0, 1), (-1, 1))
        )
        return {n: tab[i][int(v)] for i, (n, v) in enumerate(zip(self.action_names, a))}

    def evaluate_actions(self, normalized_observations, actions, published=True, *, action_masks=None):
        """Evaluate already normalized, frozen rollout inputs without renormalization."""
        o = torch.as_tensor(np.asarray(normalized_observations, dtype=np.float32))
        a = torch.as_tensor(actions, dtype=torch.long)
        if (
            o.ndim != 2
            or o.shape[1] != self.observation_dim
            or not torch.isfinite(o).all()
        ):
            raise ValueError("normalized observation 非法")
        if action_masks is None:
            self.validate_action_mask(None)
            masks = None
        else:
            masks = torch.tensor([self.validate_action_mask(m) for m in action_masks], dtype=torch.bool)
        return (self.published_model if published else self.training_model).evaluate(o, a, masks)

    def update(self, rollout, cancel_event=None):
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("PPO update cancelled / 已取消")

        check_cancelled()
        ts = tuple(
            rollout.transitions if isinstance(rollout, FrozenRollout) else rollout
        )
        n = len(ts)
        if n < self.config.min_transitions_per_update:
            return None
        if self.candidate_model is not None:
            raise RuntimeError("candidate 未发布，不能重复训练")
        if any(
            t.behavior_policy_version != self.policy_version
            or not t.included_in_training
            for t in ts
        ):
            raise ValueError("PPO rollout 来源版本或训练资格错误")
        for t in ts:
            self.validate_transition_action(t)
            expected = self.normalize(t.observation)
            if not np.array_equal(
                expected, np.asarray(t.normalized_observation, dtype=np.float32)
            ):
                raise ValueError("transition 输入与 published normalizer 不一致")
        self.training_model.load_state_dict(self.published_model.state_dict())
        self.optimizer.load_state_dict(copy.deepcopy(self.published_optimizer_state))
        ret, adv = compute_gae(
            ts,
            self.config.gamma,
            self.config.gae_lambda,
            self.config.time_aware_discount,
            self.config.discount_reference_dt_s,
        )
        if n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        o = torch.as_tensor(
            np.asarray([t.normalized_observation for t in ts], dtype=np.float32)
        )
        a = torch.as_tensor(np.asarray([t.action for t in ts]))
        masks = None if ts[0].action_mask is None else torch.tensor([t.action_mask for t in ts], dtype=torch.bool)
        old = torch.as_tensor([t.old_log_prob for t in ts])
        rt = torch.as_tensor(ret)
        ad = torch.as_tensor(adv)
        self.optimizer.zero_grad()
        metrics = {}
        sums = np.zeros(6)
        steps = 0
        early_stopped = False
        completed_epochs = 0
        for _ in range(self.config.update_epochs):
            for ix in torch.randperm(n, generator=self.update_rng).split(
                min(self.config.minibatch_size, n)
            ):
                check_cancelled()
                lp, en, v = self.training_model.evaluate(o[ix], a[ix], None if masks is None else masks[ix])
                log_ratio = lp - old[ix]
                ratio = torch.exp(log_ratio)
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approximate_kl):
                    raise FloatingPointError("PPO KL NaN/Inf")
                # Check before the next gradient step; no additional update once
                # the sampled policy already exceeds the configured KL budget.
                if (
                    self.config.target_kl is not None
                    and steps
                    and approximate_kl.item() > self.config.target_kl
                ):
                    early_stopped = True
                    break
                pl = -torch.min(
                    ratio * ad[ix],
                    torch.clamp(
                        ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                    )
                    * ad[ix],
                ).mean()
                vl = (v - rt[ix]).pow(2).mean()
                entropy = en.mean()
                loss = (
                    pl
                    + self.config.value_coef * vl
                    - self.config.entropy_coef * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss NaN/Inf")
                self.optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(
                    self.training_model.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                if any(
                    not torch.isfinite(p).all()
                    for p in self.training_model.parameters()
                ):
                    raise FloatingPointError("optimizer 参数 NaN/Inf")
                if any(
                    torch.is_tensor(v) and not torch.isfinite(v).all()
                    for state in self.optimizer.state.values()
                    for v in state.values()
                ):
                    raise FloatingPointError("optimizer 状态 NaN/Inf")
                with torch.no_grad():
                    sums += np.asarray(
                        [
                            pl.item(),
                            vl.item(),
                            entropy.item(),
                            ((ratio - 1) - log_ratio).mean().item(),
                            ((ratio - 1).abs() > self.config.clip_ratio)
                            .float()
                            .mean()
                            .item(),
                            gn.item(),
                        ]
                    )
                    steps += 1
            if early_stopped:
                break
            completed_epochs += 1
        check_cancelled()
        candidate_model = copy.deepcopy(self.training_model).eval()
        candidate_normalizer = self.normalizer.update(
            np.asarray([t.observation for t in ts], dtype=np.float64)
        )
        candidate_model.actor[0].adapt_statistics(self.normalizer, candidate_normalizer)
        candidate_model.critic[0].adapt_statistics(
            self.normalizer, candidate_normalizer
        )
        check_cancelled()
        self.candidate_model = candidate_model
        self.candidate_normalizer = candidate_normalizer
        self.candidate_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.candidate_policy_version = self.policy_version + 1
        variance = float(np.var(ret))
        explained = (
            1 - float(np.var(ret - np.asarray([t.old_value for t in ts]))) / variance
            if variance > 1e-12
            else 0.0
        )
        metrics = dict(
            zip(
                (
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                    "gradient_norm",
                ),
                (sums / steps).tolist(),
            )
        )
        metrics.update(
            rollout_size=n,
            update_epochs=self.config.update_epochs,
            minibatch_size=self.config.minibatch_size,
            optimizer_steps=steps,
            learning_rate=self.config.learning_rate,
            explained_variance=explained,
            source_policy_version=self.policy_version,
            candidate_policy_version=self.candidate_policy_version,
            source_episode_indices=list(dict.fromkeys(t.episode_index for t in ts)),
            update_count=self.update_count + 1,
            early_stopped=early_stopped,
            completed_epochs=completed_epochs,
            target_kl=self.config.target_kl,
        )
        self.latest_metrics = metrics
        return metrics

    def publish_candidate(self):
        if self.candidate_model is None:
            return False
        # Pointer swaps only: no model copying or disk I/O on the inference thread.
        self.published_model = self.candidate_model
        self.normalizer = self.candidate_normalizer
        self.published_optimizer_state = self.candidate_optimizer_state
        self.policy_version = self.candidate_policy_version
        self.update_count += 1
        self.candidate_model = None
        self.candidate_policy_version = None
        return True

    def save_checkpoint(self, path):
        """Standalone/offline helper. Runtime calls worker.request_checkpoint instead."""
        _atomic_torch_save(self.checkpoint_state(), Path(path))

    def checkpoint_state(self):
        return {
            "format_version": 1,
            "model": self.published_model.state_dict(),
            "optimizer": self.published_optimizer_state,
            "normalizer": {
                "mean": torch.tensor(self.normalizer.mean),
                "var": torch.tensor(self.normalizer.var),
                "count": self.normalizer.count,
            },
            "policy_version": self.policy_version,
            "environment_steps": self.environment_steps,
            "update_count": self.update_count,
            "architecture": self.architecture(),
            "seed": self.seed,
            "inference_rng": self.inference_rng.get_state(),
            "update_rng": self.update_rng.get_state(),
            "normalizer_initialized": self.normalizer_initialized,
        }

    def architecture(self):
        tables = (
            (TAIL_THETA_VALUES, TAIL_T_VALUES)
            if self.action_dims == (7, 5)
            else (FIN_THETA_VALUES, FIN_T_VALUES, FIN_MODES)
        )
        return {
            "agent": type(self).__name__,
            "observation_dim": self.observation_dim,
            "action_dims": list(self.action_dims),
            "action_definition": {
                name: list(v) for name, v in zip(self.action_names, tables)
            },
            "history_length": self.history_length,
            "normalized_clip": self.normalized_clip,
            "ppo_config": json.loads(json.dumps(asdict(self.config))),
        }

    def validate_checkpoint(self, d):
        if d.get("format_version") != 1 or d.get("architecture") != self.architecture():
            raise ValueError("checkpoint 网络、动作定义、history 或 PPO 配置不匹配")
        for key in ("policy_version", "environment_steps", "update_count", "seed"):
            if not isinstance(d.get(key), int) or d[key] < 0:
                raise ValueError(f"checkpoint {key} 非法")
        q = d["normalizer"]
        norm = RunningMeanStd(q["mean"].numpy(), q["var"].numpy(), q["count"])
        if len(norm.mean) != self.observation_dim:
            raise ValueError("checkpoint normalizer 维度错误")
        reference = self.published_model.state_dict()
        if set(reference) != set(d["model"]):
            raise ValueError("checkpoint 参数名称不匹配")
        for key, v in d["model"].items():
            if v.shape != reference[key].shape or not torch.isfinite(v).all():
                raise ValueError("checkpoint 参数形状错误或 NaN/Inf")
        # Validate optimizer and RNG on scratch objects before touching live policies.
        scratch = copy.deepcopy(self.published_model)
        opt = torch.optim.Adam(scratch.parameters(), lr=self.config.learning_rate)
        opt.load_state_dict(d["optimizer"])
        for parameter, state in opt.state.items():
            for name, v in state.items():
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    raise ValueError("checkpoint optimizer NaN/Inf")
                if (
                    name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
                    and v.shape != parameter.shape
                ):
                    raise ValueError("checkpoint optimizer 动量形状错误")
        for key in ("inference_rng", "update_rng"):
            torch.Generator().set_state(d[key])
        return norm

    def restore_state(self, d):
        norm = self.validate_checkpoint(d)
        self.published_model.load_state_dict(d["model"])
        self.training_model.load_state_dict(d["model"])
        self.published_optimizer_state = copy.deepcopy(d["optimizer"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.normalizer = norm
        self.policy_version = d["policy_version"]
        self.environment_steps = d["environment_steps"]
        self.update_count = d["update_count"]
        self.seed = d["seed"]
        self.inference_rng.set_state(d["inference_rng"])
        self.update_rng.set_state(d["update_rng"])
        self.candidate_model = None
        self.normalizer_initialized = bool(d.get("normalizer_initialized", False))
        self._inference_started = True

    def load_checkpoint(self, path):
        self.restore_state(torch.load(path, map_location="cpu", weights_only=True))

    checkpoint = save_checkpoint
    restore = load_checkpoint


class TailPPOAgent(BasePPOAgent):
    action_dims = (7, 5)
    action_names = ("theta", "t")


class FinPPOAgent(BasePPOAgent):
    # b2 的合法值依赖 theta，使用联合分布避免独立分类头采出非法组合。
    action_dims = (18,)
    action_names = ("joint_action",)
    requires_action_mask = True

    def __init__(
        self,
        observation_dim=None,
        config=None,
        seed=None,
        history_length=30,
        normalized_clip=10.0,
    ):
        if config is None:
            config = PPOConfig(
                actor_hidden_sizes=(64, 64),
                critic_hidden_sizes=(64, 64),
                learning_rate=1e-4,
                minibatch_size=5,
                update_epochs=3,
                min_transitions_per_update=2,
                target_kl=0.03,
            )
        if observation_dim is None:
            observation_dim = 18 + 5 * history_length  # 15维实测状态 + 历史 + 3维约束状态
        super().__init__(observation_dim, config, seed, history_length, normalized_clip)

    def decode_action(self, action):
        if len(action) != 1 or any(
            isinstance(v, bool) or int(v) != v or not 0 <= v < count
            for v, count in zip(action, self.action_dims)
        ):
            raise ValueError("Fin 动作索引必须是 0～17 的联合分类索引")
        return FinActionSpace().decode_flat(action[0]).to_dict()

    def architecture(self):
        result = super().architecture()
        result['action_definition'] = {'joint_action': [a.to_dict() for a in FinActionSpace().all()]}
        result['sequence_rule'] = 'alternating_b1_effective_direction_v1'
        result['sequence_features'] = ['previous_theta/53', 'next_b1', 'last_effective_direction']
        return result

    def validate_transition_action(self, transition):
        super().validate_transition_action(transition)
        if self.observation_dim == 18 + 5 * self.history_length:
            # 实物 rollout 的掩码必须对应当时原始观测中的序列状态。
            theta, b1, direction = transition.observation[-3:]
            state = FinSequenceState(theta * 53., b1, direction)
            if transition.action_mask != state.action_mask():
                raise ValueError('action_mask 与采样时序列状态不一致')


def _atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint_pair(directory, tail_state, fin_state, metadata):
    """Commit two independent files via one manifest rename; old manifests stay valid."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    version = tail_state["policy_version"]
    if version != fin_state["policy_version"]:
        raise ValueError("checkpoint 必须保存同版本 Tail/Fin")
    generation = f"{version}_{uuid.uuid4().hex}"
    files = {}
    for name, state in (("tail", tail_state), ("fin", fin_state)):
        path = dest / f"{name}_ppo_{generation}_20260914.pt"
        _atomic_torch_save(state, path)
        files[name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "format_version": 1,
        "policy_version": version,
        "files": files,
        "metadata": metadata,
    }
    tmp = dest / "checkpoint_20260914.json.tmp"
    target = dest / "checkpoint_20260914.json"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return str(target)


def load_checkpoint_pair(directory, tail_agent, fin_agent):
    path = Path(directory)
    if path.is_dir():
        path = path / "checkpoint_20260914.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint manifest")
    states = {}
    for name, agent in (("tail", tail_agent), ("fin", fin_agent)):
        entry = manifest["files"][name]
        filename = entry["file"]
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("checkpoint 文件名非法")
        p = path.parent / filename
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("checkpoint 文件损坏")
        state = torch.load(p, map_location="cpu", weights_only=True)
        agent.validate_checkpoint(state)
        if state["policy_version"] != manifest["policy_version"]:
            raise ValueError("Tail/Fin checkpoint 版本不一致")
        states[name] = state
    # Both payloads validate before either policy is restored. Physical zero points
    # are intentionally absent: every launch must perform fresh water calibration.
    tail_agent.restore_state(states["tail"])
    fin_agent.restore_state(states["fin"])
    return manifest.get("metadata", {})


def save_fin_checkpoint(directory, agent, next_episode_index, metadata=None):
    """Commit only the published 18-action Fin policy with an atomic manifest."""
    if agent.action_dims != (18,) or agent.candidate_model is not None:
        raise RuntimeError("仅允许空闲时保存已发布的 Fin-only v3 checkpoint")
    if (
        not isinstance(next_episode_index, int)
        or isinstance(next_episode_index, bool)
        or next_episode_index < 0
    ):
        raise ValueError("next_episode_index 必须是非负整数")
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    context = dict(metadata or {})
    context["next_episode_index"] = next_episode_index
    payload = agent.checkpoint_state()
    payload["mode"] = "fin_only_v3"
    filename = f"fin_ppo_v{agent.policy_version}_{uuid.uuid4().hex}_20260914.pt"
    path = dest / filename
    _atomic_torch_save(payload, path)
    manifest = {
        "format_version": 3,
        "mode": "fin_only_v3",
        "policy_version": agent.policy_version,
        "files": {
            "fin": {
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        "metadata": context,
    }
    target = dest / "checkpoint_20260914.json"
    temp = dest / f"checkpoint_{uuid.uuid4().hex}.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    return str(target)


def load_fin_checkpoint(directory, agent):
    """Strict read-only resume; reject old pair/132/220-action snapshots."""
    path = Path(directory)
    if path.is_dir():
        path = path / "checkpoint_20260914.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "fin_only_v3" or manifest.get("format_version") != 3:
        raise ValueError(
            "仅接受 fin_only_v3 checkpoint；旧双 Agent / 132 / 220 动作模型不兼容，需重新训练"
        )
    if set(manifest.get("files", {})) != {"fin"}:
        raise ValueError("fin_only_v3 manifest 必须只包含 Fin")
    metadata = manifest.get("metadata", {})
    index = metadata.get("next_episode_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("checkpoint next_episode_index 非法")
    entry = manifest["files"]["fin"]
    filename = entry["file"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("checkpoint 文件名非法")
    file = path.parent / filename
    if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("checkpoint 文件损坏")
    payload = torch.load(file, map_location="cpu", weights_only=True)
    if (
        payload.get("mode") != "fin_only_v3"
        or payload.get("policy_version") != manifest["policy_version"]
    ):
        raise ValueError("checkpoint mode/policy 版本不一致")
    if payload.get("architecture", {}).get("action_dims") != [18]:
        raise ValueError("旧动作空间 checkpoint 不兼容 18 动作约束 Fin policy")
    agent.validate_checkpoint(payload)
    agent.restore_state(payload)
    return metadata
