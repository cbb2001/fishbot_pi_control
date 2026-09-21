from __future__ import annotations

import dataclasses
import tempfile
import unittest
import itertools
import threading
from pathlib import Path

import numpy as np
from control.runtime.rl_ppo_agent_20260914 import (
    torch,
    RunningMeanStd,
    TailPPOAgent,
    FinPPOAgent,
    load_checkpoint_pair,
    save_checkpoint_pair,
)
from control.runtime.rl_rollout_buffer_20260914 import FrozenRollout, Transition

from control.runtime.rl_actions_20260914 import FinSequenceState


def act(agent, observation, **kwargs):
    mask = FinSequenceState().action_mask() if agent.requires_action_mask else None
    return agent.act(observation, action_mask=mask, **kwargs)


SMALL = dict(
    actor_hidden_sizes=[8],
    critic_hidden_sizes=[8],
    update_epochs=2,
    minibatch_size=2,
    min_transitions_per_update=4,
)


def transitions_for(agent, n=4, episode=0):
    result = []
    for i in range(n):
        obs = tuple(float(j + i) for j in range(agent.observation_dim))
        decision = act(agent, obs)
        result.append(
            Transition(
                obs,
                decision.action,
                float(i + 1),
                obs,
                decision.log_prob,
                decision.value,
                agent.value(obs),
                decision.normalized_observation,
                action_mask=decision.action_mask,
                episode_index=episode,
                behavior_policy_version=agent.policy_version,
            )
        )
    result[-1] = dataclasses.replace(result[-1], truncated=True)
    return FrozenRollout(tuple(result), agent.policy_version)


class NormalizerTests(unittest.TestCase):
    def test_parallel_variance_matches_population_with_prior_and_is_immutable(self):
        rms = RunningMeanStd.create(2)
        first = rms.update([[1, 10], [3, 14]])
        second = first.update([[5, 18], [7, 22]])
        direct = rms.update([[1, 10], [3, 14], [5, 18], [7, 22]])
        np.testing.assert_allclose(second.mean, direct.mean)
        np.testing.assert_allclose(second.var, direct.var)
        np.testing.assert_allclose(second.mean, [4, 16], atol=0.001)
        np.testing.assert_allclose(second.var, [5, 20], atol=0.01)
        with self.assertRaises(ValueError):
            second.mean[0] = 9
        np.testing.assert_array_equal(rms.mean, [0, 0])


@unittest.skipIf(torch is None, "RL dependency missing: PyTorch is required")
class PPOAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_heads_log_probability_entropy_and_agent_independence(self):
        for cls, dimensions in ((TailPPOAgent, (7, 5)), (FinPPOAgent, (18,))):
            agent = cls(3, SMALL, seed=7)
            decision = act(agent, [1.0, 2.0, 3.0])
            obs = np.asarray([decision.normalized_observation])
            lp, entropy, values = agent.evaluate_actions(obs, [decision.action],
                action_masks=None if decision.action_mask is None else [decision.action_mask])
            self.assertAlmostEqual(float(lp[0].detach()), decision.log_prob, places=5)
            self.assertAlmostEqual(float(values[0].detach()), decision.value, places=5)
            logits = agent.published_model.actor(
                torch.as_tensor(obs, dtype=torch.float32)
            )
            self.assertEqual(logits.shape[-1], sum(dimensions))
            if decision.action_mask is not None:
                logits = logits.masked_fill(~torch.tensor([decision.action_mask]), -torch.inf)
            distributions = [
                torch.distributions.Categorical(logits=x)
                for x in torch.split(logits, dimensions, dim=-1)
            ]
            expected = sum(d.entropy().item() for d in distributions)
            self.assertAlmostEqual(entropy.item(), expected, places=5)
            self.assertEqual(len(decision.action), len(dimensions))
        tail, fin = TailPPOAgent(3, SMALL, seed=1), FinPPOAgent(3, SMALL, seed=1)
        self.assertNotEqual(
            next(tail.published_model.parameters()).data_ptr(),
            next(fin.published_model.parameters()).data_ptr(),
        )
        self.assertIsNot(tail.optimizer, fin.optimizer)

    def test_update_cannot_mutate_published_policy_or_normalizer(self):
        agent = TailPPOAgent(3, SMALL, seed=17)
        rollout = transitions_for(agent)
        before = act(agent, [1, 2, 3], deterministic=True)
        weights = {k: v.clone() for k, v in agent.published_model.state_dict().items()}
        normalizer = agent.normalizer
        metrics = agent.update(rollout)
        self.assertEqual(metrics["optimizer_steps"], 4)
        self.assertEqual(agent.policy_version, 0)
        self.assertIs(agent.normalizer, normalizer)
        self.assertEqual(act(agent, [1, 2, 3], deterministic=True), before)
        self.assertTrue(
            all(
                torch.equal(v, weights[k])
                for k, v in agent.published_model.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(v, weights[k])
                for k, v in agent.candidate_model.state_dict().items()
            )
        )
        agent.publish_candidate()
        self.assertEqual(agent.policy_version, 1)
        self.assertGreater(agent.normalizer.count, normalizer.count)

    def test_reject_nonfinite_and_mismatched_behavior_input(self):
        agent = TailPPOAgent(3, SMALL, seed=1)
        for obs in ([float("nan"), 1, 2], [1, float("inf"), 2], [1, 2]):
            with self.assertRaises(ValueError):
                act(agent, obs)
        rollout = transitions_for(agent)
        ts = list(rollout.transitions)
        ts[0] = dataclasses.replace(ts[0], normalized_observation=(9, 9, 9))
        with self.assertRaisesRegex(ValueError, "normalizer"):
            agent.update(FrozenRollout(tuple(ts), 0))

    def test_pair_checkpoint_restores_statistics_rng_and_rejects_architecture(self):
        tail, fin = TailPPOAgent(3, SMALL, seed=11), FinPPOAgent(3, SMALL, seed=12)
        for agent in (tail, fin):
            agent.update(transitions_for(agent))
            agent.publish_candidate()
            agent.environment_steps = 17
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint_pair(
                folder,
                tail.checkpoint_state(),
                fin.checkpoint_state(),
                {"next_episode_index": 5},
            )
            expected = [tail.act([3, 4, 5]), act(fin, [3, 4, 5])]
            t2, f2 = TailPPOAgent(3, SMALL, seed=90), FinPPOAgent(3, SMALL, seed=91)
            metadata = load_checkpoint_pair(folder, t2, f2)
            self.assertEqual(metadata["next_episode_index"], 5)
            self.assertEqual([t2.act([3, 4, 5]), act(f2, [3, 4, 5])], expected)
            self.assertEqual(t2.environment_steps, 17)
            np.testing.assert_array_equal(t2.normalizer.mean, tail.normalizer.mean)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_checkpoint_pair(folder, TailPPOAgent(4, SMALL), f2)
            # Manifest hashes detect a partial or corrupt agent checkpoint.
            import json

            manifest = json.loads(
                (Path(folder) / "checkpoint_20260914.json").read_text()
            )
            p = Path(folder) / manifest["files"]["fin"]["file"]
            p.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "损坏"):
                load_checkpoint_pair(folder, t2, f2)

    def test_fin_mode_head_covers_18_unique_canonical_actions(self):
        agent = FinPPOAgent(3, SMALL, seed=1)
        self.assertEqual(agent.action_dims, (18,))
        actions = [
            agent.decode_action(index)
            for index in itertools.product(range(18))
        ]
        self.assertEqual(len({tuple(a.values()) for a in actions}), 18)
        self.assertEqual({a["t"] for a in actions}, {0.3, 0.6})
        self.assertEqual(
            {(a["b1"], a["b2"]) for a in actions}, {(0, 0), (1, -1), (1, 1)}
        )
        with self.assertRaises(ValueError):
            agent.decode_action((0, 0, 0, 0))

    def test_normalizer_warmup_centers_servo_angles_and_cannot_reset_after_act(self):
        agent = FinPPOAgent(3, SMALL, seed=1)
        self.assertTrue(
            hasattr(agent, "initialize_normalizer"),
            "Explicit warmup normalizer required",
        )
        agent.initialize_normalizer([[85.0, 111.0, 143.0]])
        decision = act(agent, [85.0, 111.0, 143.0])
        np.testing.assert_allclose(
            decision.normalized_observation,
            [0.0, 0.0, 0.0], atol=1e-6
        )
        with self.assertRaises(RuntimeError):
            agent.initialize_normalizer([[90.0, 110.0, 145.0]])

    def test_candidate_normalizer_change_preserves_trained_logits_and_values(self):
        agent = FinPPOAgent(3, SMALL, seed=3, normalized_clip=2.0)
        self.assertTrue(
            hasattr(agent, "initialize_normalizer"),
            "Explicit warmup normalizer required",
        )
        agent.initialize_normalizer([[85.0, 111.0, 143.0]])
        rollout = transitions_for(agent)
        old_normalizer = agent.normalizer
        agent.update(rollout)
        raw = [50.0, 170.0, 250.0]
        old_input = torch.as_tensor(agent.normalize(raw, old_normalizer)).unsqueeze(0)
        new_input = torch.as_tensor(
            agent.normalize(raw, agent.candidate_normalizer)
        ).unsqueeze(0)
        with torch.no_grad():
            np.testing.assert_allclose(
                agent.training_model.actor(old_input),
                agent.candidate_model.actor(new_input),
                atol=2e-5,
            )
            np.testing.assert_allclose(
                agent.training_model.critic(old_input),
                agent.candidate_model.critic(new_input),
                atol=2e-5,
            )

    def test_cancelled_update_never_exposes_partial_candidate(self):
        agent = FinPPOAgent(3, SMALL, seed=3)
        rollout = transitions_for(agent)
        self.assertTrue(
            "cancel_event" in __import__("inspect").signature(agent.update).parameters,
            "PPO update must support cooperative cancellation",
        )
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(RuntimeError, "cancel|取消"):
            agent.update(rollout, cancel_event=cancel)
        self.assertIsNone(agent.candidate_model)
        self.assertEqual(agent.policy_version, 0)

    def test_target_kl_stops_reusing_rollout_after_excessive_drift(self):
        from control.runtime.rl_ppo_agent_20260914 import PPOConfig

        self.assertIn("target_kl", PPOConfig.__dataclass_fields__)
        agent = FinPPOAgent(
            3,
            {**SMALL, "learning_rate": 0.5, "update_epochs": 20, "target_kl": 1e-7},
            seed=2,
        )
        metrics = agent.update(transitions_for(agent))
        self.assertTrue(metrics["early_stopped"])
        self.assertLess(metrics["optimizer_steps"], 40)


if __name__ == "__main__":
    unittest.main()
