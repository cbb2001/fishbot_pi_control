from __future__ import annotations

import tempfile
import threading
import time
import unittest
import json
import dataclasses
from unittest.mock import patch
from pathlib import Path

from control.runtime.rl_ppo_agent_20260914 import torch, TailPPOAgent, FinPPOAgent
from control.runtime.rl_ppo_update_worker_20260914 import PPOUpdateWorker
from control.runtime.rl_training_coordinator_20260914 import RLTrainingCoordinator
from control.runtime.rl_rollout_buffer_20260914 import Transition
from control.runtime import rl_training_coordinator_20260914 as coordinator_module

from control.runtime.rl_actions_20260914 import FinSequenceState


def act(agent, observation, **kwargs):
    mask = FinSequenceState().action_mask() if agent.requires_action_mask else None
    return agent.act(observation, action_mask=mask, **kwargs)


SMALL = dict(
    actor_hidden_sizes=[8],
    critic_hidden_sizes=[8],
    update_epochs=1,
    minibatch_size=2,
    min_transitions_per_update=2,
)


@unittest.skipIf(torch is None, "RL dependency missing: PyTorch is required")
class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tail = TailPPOAgent(3, SMALL, seed=1)
        self.fin = FinPPOAgent(3, SMALL, seed=2)
        self.coordinator = RLTrainingCoordinator(self.tail, self.fin)
        self.addCleanup(self.coordinator.close)

    def collect(self, name, n=1):
        c = self.coordinator
        agent = self.tail if name == "tail" else self.fin
        for _ in range(n):
            observation = (1.0, 2.0, 3.0)
            decision = act(agent, observation)
            transition = Transition(
                observation,
                decision.action,
                1.0,
                observation,
                decision.log_prob,
                decision.value,
                agent.value(observation),
                decision.normalized_observation,
                action_mask=decision.action_mask,
                episode_index=c.episode_index,
                behavior_policy_version=decision.policy_version,
                episode_role=c.episode_role,
            )
            c.add_transition(name, transition)

    def wait_result(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.coordinator.process_update_results():
                return
            time.sleep(0.005)
        self.fail("background update did not finish")

    def test_both_thresholds_prevent_asymmetric_agent_starvation(self):
        c = self.coordinator
        c.begin_episode()
        self.collect("tail", 2)
        self.collect("fin", 1)
        self.assertFalse(c.end_episode().ppo_update_started)
        self.assertFalse(c.worker.busy)
        self.assertEqual(c.begin_episode().episode_role, "trainable")
        self.collect("tail", 2)
        self.collect("fin", 1)
        self.assertTrue(c.end_episode().ppo_update_started)
        self.assertEqual(c.begin_episode().episode_role, "bridge")
        self.wait_result()
        self.assertEqual(self.tail.policy_version, 0)
        self.assertEqual(self.fin.policy_version, 0)
        self.assertTrue(c.end_episode().policy_published_at_end)
        self.assertEqual((self.tail.policy_version, self.fin.policy_version), (1, 1))
        self.assertEqual(c.begin_episode().episode_role, "trainable")

    def test_slow_update_never_blocks_bridge_control_or_publishes_mid_episode(self):
        c = self.coordinator
        release = threading.Event()
        entered = threading.Event()
        self.addCleanup(release.set)
        update = self.tail.update

        def slow(rollout):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timeout")
            return update(rollout)

        self.tail.update = slow
        c.begin_episode()
        self.collect("tail", 2)
        self.collect("fin", 2)
        started = time.monotonic()
        self.assertTrue(c.end_episode().ppo_update_started)
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(entered.wait(1))
        # Multiple full episodes remain possible while backward has not returned.
        for _ in range(2):
            self.assertEqual(c.begin_episode().episode_role, "bridge")
            self.collect("tail")
            self.collect("fin")
            self.assertFalse(c.current["tail"][0].included_in_training)
            self.assertEqual(c.end_episode().tail_trainable_transitions, 0)
        c.begin_episode()
        release.set()
        self.wait_result()
        self.assertEqual(c.published_policy_version, 0)
        self.assertTrue(c.end_episode().policy_published_at_end)
        self.assertEqual(c.published_policy_version, 1)

    def test_checkpoint_is_written_by_worker_and_can_resume_pair(self):
        c = self.coordinator
        c.begin_episode()
        self.collect("tail")
        self.collect("fin")
        c.end_episode()
        with tempfile.TemporaryDirectory() as folder:
            c.request_checkpoint(folder)
            c.close()
            self.assertTrue((Path(folder) / "checkpoint_20260914.json").exists())
            restored = RLTrainingCoordinator(
                TailPPOAgent(3, SMALL), FinPPOAgent(3, SMALL)
            )
            try:
                restored.load_checkpoint(folder)
                self.assertEqual(restored.begin_episode().episode_index, 1)
                self.assertEqual(restored.tail_transition_count, 1)
            finally:
                restored.close()

    def test_background_exception_is_propagated_with_traceback(self):
        def fail(_):
            raise RuntimeError("optimizer exploded")

        self.tail.update = fail
        c = self.coordinator
        c.begin_episode()
        self.collect("tail", 2)
        self.collect("fin", 2)
        c.end_episode()
        deadline = time.monotonic() + 3
        while c.worker.results.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        with self.assertRaisesRegex(RuntimeError, "optimizer exploded"):
            c.process_update_results()


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None, "RL dependency missing: PyTorch required")
class FinOnlyCoordinatorTests(unittest.TestCase):
    def make_coordinator(self, timeout=2.0):
        self.assertTrue(
            hasattr(coordinator_module, "FinTrainingCoordinator"),
            "A Fin-only coordinator must not construct a Tail agent",
        )
        agent = FinPPOAgent(
            3, {**SMALL, "update_epochs": 3, "minibatch_size": 5}, seed=2
        )
        agent.initialize_normalizer([[85.0, 111.0, 143.0]])
        c = coordinator_module.FinTrainingCoordinator(
            agent,
            {
                "training": {"update_every_actions": 5},
                "runtime": {"update_timeout_s": timeout},
            },
        )
        self.addCleanup(c.close)
        c.begin_episode()
        return c

    def collect(self, c, count):
        for i in range(count):
            obs = (85.0 + i, 111.0, 143.0)
            decision = act(c.agent, obs)
            c.add_transition(
                Transition(
                    obs,
                    decision.action,
                    1.0 + i,
                    obs,
                    decision.log_prob,
                    decision.value,
                    c.agent.value(obs),
                    decision.normalized_observation,
                action_mask=decision.action_mask,
                    behavior_policy_version=decision.policy_version,
                    episode_index=c.episode_index,
                )
            )

    def wait_update(self, c):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = c.poll_update()
            if result is not None:
                return result
            time.sleep(0.005)
        self.fail("update did not finish")

    def test_five_actions_freeze_and_publish_before_episode_end(self):
        c = self.make_coordinator()
        self.collect(c, 4)
        self.assertFalse(c.ready)
        self.assertFalse(c.start_update())
        self.collect(c, 1)
        self.assertTrue(c.ready)
        self.assertTrue(c.start_update())
        self.assertFalse(c.start_update())
        self.assertTrue(c.last_rollout.transitions[-1].truncated)
        with self.assertRaises(RuntimeError):
            self.collect(c, 1)
        metrics = self.wait_update(c)
        self.assertEqual(metrics["rollout_size"], 5)
        self.assertEqual(metrics["optimizer_steps"], 3)
        self.assertEqual(metrics["rollout_boundary_reason"], "update_hold")
        self.assertEqual(c.agent.policy_version, 1)
        self.assertTrue(c.running)
        self.assertEqual(c.snapshot()["fin_rollout_size"], 0)

    def test_episode_tail_update_and_singleton_discard_never_cross_episode(self):
        c = self.make_coordinator()
        self.collect(c, 2)
        self.assertTrue(c.start_update(force=True))
        self.assertEqual(self.wait_update(c)["rollout_size"], 2)
        c.end_episode()
        c.begin_episode()
        self.collect(c, 1)
        self.assertFalse(c.start_update(force=True))
        summary = c.end_episode()
        self.assertEqual(summary["discarded_transitions"], 1)
        self.assertEqual(summary["discard_reason"], "insufficient_tail_batch")
        c.begin_episode()
        self.assertEqual(c.pending, [])

    def test_checkpoint_is_fin_only_and_restore_preserves_next_episode_rng(self):
        c = self.make_coordinator()
        self.collect(c, 2)
        c.start_update(force=True)
        self.wait_update(c)
        c.end_episode()
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = c.save_checkpoint(folder, next_episode_index=8)
            manifest = json.loads(Path(manifest_path).read_text())
            self.assertEqual(manifest["mode"], "fin_only_v3")
            self.assertEqual(set(manifest["files"]), {"fin"})
            expected = act(c.agent, [85.0, 111.0, 143.0])
            clone = coordinator_module.FinTrainingCoordinator(
                FinPPOAgent(3, c.agent.config, seed=9)
            )
            try:
                metadata = clone.load_checkpoint(folder)
                self.assertEqual(metadata["next_episode_index"], 8)
                self.assertEqual(clone.begin_episode().episode_index, 8)
                self.assertEqual(act(clone.agent, [85.0, 111.0, 143.0]), expected)
            finally:
                clone.close()
            manifest.pop("mode")
            Path(manifest_path).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fin_only_v3|旧|mode"):
                c.load_checkpoint(folder)

    def test_timeout_cancels_minibatch_and_forbids_candidate_publication(self):
        c = self.make_coordinator(timeout=0.02)
        self.collect(c, 5)
        original = c.agent.optimizer.step

        def slow_optimizer(*args, **kwargs):
            time.sleep(0.06)
            return original(*args, **kwargs)

        with patch.object(c.agent.optimizer, "step", slow_optimizer):
            c.start_update()
            time.sleep(0.04)
            with self.assertRaises(TimeoutError):
                c.poll_update()
            c.close()
        self.assertEqual(c.agent.policy_version, 0)
        self.assertIsNone(c.agent.candidate_model)
        self.assertFalse(c.worker.thread.is_alive())

    def test_optimizer_failure_is_propagated_and_no_candidate_is_published(self):
        c = self.make_coordinator()
        self.collect(c, 5)
        with patch.object(
            c.agent.optimizer, "step", side_effect=RuntimeError("optimizer failed")
        ):
            c.start_update()
            with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
                self.wait_update(c)
        self.assertEqual(c.agent.policy_version, 0)
        self.assertIsNone(c.agent.candidate_model)

    def test_fin_defaults_allow_five_action_updates_without_tail_configuration(self):
        agent = FinPPOAgent(seed=4)
        self.assertLessEqual(agent.config.min_transitions_per_update, 5)
        c = coordinator_module.FinTrainingCoordinator(agent)
        try:
            self.assertEqual(c.update_every_actions, 5)
        finally:
            c.close()

    def test_resume_old_220_payload_rejected_even_with_new_manifest(self):
        import hashlib

        c = self.make_coordinator()
        c.end_episode()
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(c.save_checkpoint(folder, next_episode_index=1))
            manifest = json.loads(manifest_path.read_text())
            payload_path = Path(folder) / manifest["files"]["fin"]["file"]
            payload = torch.load(payload_path, weights_only=True)
            payload["architecture"]["action_dims"] = [11, 5, 2, 2]
            torch.save(payload, payload_path)
            manifest["files"]["fin"]["sha256"] = hashlib.sha256(
                payload_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checkpoint.*18"):
                c.load_checkpoint(folder)
            self.assertEqual(c.published_policy_version, 0)

    def test_terminal_tail_does_not_bootstrap_and_successful_update_does_not_cross_episode(
        self,
    ):
        c = self.make_coordinator()
        self.collect(c, 2)
        c.pending[-1] = dataclasses.replace(
            c.pending[-1], terminated=True, next_value=1000.0
        )
        c.start_update(force=True)
        self.assertTrue(c.last_rollout.transitions[-1].terminated)
        self.assertFalse(c.last_rollout.transitions[-1].truncated)
        self.wait_update(c)
        c.end_episode(terminated=True)
        c.begin_episode()
        self.assertEqual(c.pending, [])
        self.assertEqual(c.agent.environment_steps, 2)

    def test_checkpoint_rejected_while_training_and_cancel_removes_ready_candidate(
        self,
    ):
        c = self.make_coordinator()
        self.collect(c, 5)
        c.start_update()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                c.save_checkpoint(folder)
            deadline = time.monotonic() + 2
            while c.worker.results.empty() and time.monotonic() < deadline:
                time.sleep(0.005)
            c.cancel_update("user_stop")
            result = c.poll_update()
            self.assertTrue(result["cancelled"])
            self.assertEqual(c.agent.policy_version, 0)
            self.assertIsNone(c.agent.candidate_model)
