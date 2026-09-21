"""约束动作的独立规则检查、执行确认及 masked PPO 回归测试；不访问硬件。"""
import dataclasses
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from control.runtime.rl_actions_20260914 import (
    ActionValidationError, FinAction, FinActionSpace, FinSequenceState, build_rl_trajectory,
)
from control.runtime.rl_action_history_20260914 import FinActionHistory
from control.runtime.rl_action_scheduler_20260914 import RLActionScheduler
from control.runtime.rl_ppo_agent_20260914 import FinPPOAgent, torch
from control.runtime.rl_rollout_buffer_20260914 import Transition


class FinConstraintTests(unittest.TestCase):
    def test_all_states_have_legal_actions_and_match_independent_rule(self):
        space = FinActionSpace()
        self.assertEqual(len(space.all()), 18)
        for theta, b1, last in itertools.product((-53, 0, 53), (0, 1), (-1, 0, 1)):
            state = FinSequenceState(theta, b1, last)
            mask = state.action_mask()
            self.assertEqual(sum(mask), 6 if b1 == 0 else (12 if last == 0 else 8))
            for index, action in enumerate(space.all()):
                self.assertEqual(space.decode_flat(index), action)
                direction = 0
                if action.b1 == 1 and theta != action.theta:
                    direction = action.b2 if action.theta > theta else -action.b2
                allowed = action.b1 == b1 and (not direction or not last or direction == -last)
                self.assertEqual(mask[index], allowed)
                if allowed:
                    following = state.after(action)
                    self.assertEqual(following.next_b1, 1 - b1)
                    self.assertEqual(following.last_effective_direction, direction or last)
                else:
                    with self.assertRaises(ActionValidationError):
                        state.after(action)

    def test_same_b2_can_mean_opposite_directions(self):
        state = FinSequenceState().after(FinAction(-53, .3, 0, 0))
        forward = FinAction(53, .3, 1, 1)
        self.assertEqual(state.direction(forward), 1)
        state = state.after(forward).after(FinAction(53, .6, 0, 0))
        backward = FinAction(-53, .6, 1, 1)
        self.assertEqual(state.direction(backward), -1)
        self.assertEqual(state.after(backward).last_effective_direction, -1)
        with self.assertRaises(ActionValidationError):
            state.after(FinAction(-53, .6, 1, -1))

    def test_empty_strokes_never_erase_direction_even_beyond_history(self):
        history = FinActionHistory(1)
        state = FinSequenceState(53, 0, 1)
        for i in range(80):
            action = FinAction(53, .3, i % 2, -1 if i % 2 else 0)
            history.append(action, completed=True, pwm_success=True)
            state = state.after(action)
            self.assertEqual(state.last_effective_direction, 1)
        state = state.after(FinAction(0, .3, 0, 0))
        self.assertFalse(state.allows(FinAction(53, .3, 1, 1)))
        self.assertTrue(state.allows(FinAction(53, .3, 1, -1)))

    def test_scheduler_advances_only_after_successful_endpoint(self):
        scheduler = RLActionScheduler(limits={i: (0.,270.) for i in range(1,8)})
        initial = scheduler.fin_sequence
        with self.assertRaises(ActionValidationError):
            scheduler.schedule('fin', FinAction(53, .3, 1, 1), 0)
        scheduler.schedule('fin', FinAction(-53, .3, 0, 0), 0)
        self.assertEqual(scheduler.fin_sequence, initial)
        with self.assertRaises(RuntimeError):
            scheduler.complete('fin', endpoint_written=True, completion_t_ns=1)
        with self.assertRaises(RuntimeError):
            scheduler.complete('fin', endpoint_written=False, completion_t_ns=300_000_000)
        self.assertEqual(scheduler.fin_sequence, initial)
        scheduler.complete('fin', endpoint_written=True, completion_t_ns=300_000_000)
        self.assertEqual(scheduler.fin_sequence, FinSequenceState(-53, 1, 0))
        scheduler.hold(scheduler.committed_angles)
        self.assertEqual(scheduler.fin_sequence, FinSequenceState(-53, 1, 0))
        with self.assertRaises(ActionValidationError):
            scheduler.schedule('fin', FinAction(0, .3, 0, 0), 400_000_000)

    def test_full_span_trajectory_still_mirrors_and_returns_tip(self):
        centers = {1:85, 2:95, 3:95, 4:111, 5:90, 6:143, 7:90}
        for b2 in (-1, 1):
            trajectory = build_rl_trajectory(FinAction(53, .6, 1, b2), -53, centers)
            self.assertEqual(trajectory.evaluate_all(.3), {4:111, 5:90+90*b2, 6:143, 7:90-90*b2})
            self.assertEqual(trajectory.evaluate_all(.6), {4:164, 5:90, 6:90, 7:90})


@unittest.skipIf(torch is None, 'PyTorch required')
class MaskedPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.agent = FinPPOAgent(history_length=0, seed=12, config={
            'actor_hidden_sizes': [8], 'critic_hidden_sizes': [8],
            'min_transitions_per_update': 5, 'minibatch_size': 5, 'update_epochs': 2,
        })

    def observation(self, state):
        return tuple([0.] * 15) + state.vector()

    def collect(self, state, n=5):
        rows = []
        for i in range(n):
            obs = self.observation(state)
            decision = self.agent.act(obs, action_mask=state.action_mask())
            action = FinAction(**self.agent.decode_action(decision.action))
            state = state.after(action)
            nxt = self.observation(state)
            rows.append(Transition(obs, decision.action, float(i), nxt,
                decision.log_prob, decision.value, self.agent.value(nxt),
                decision.normalized_observation, action_mask=decision.action_mask,
                duration_s=action.t, behavior_policy_version=decision.policy_version))
        return state, rows

    def test_forbidden_logits_cannot_win_and_logprob_entropy_are_masked(self):
        state = FinSequenceState(0, 1, 1)
        mask = state.action_mask()
        with torch.no_grad():
            for parameter in self.agent.published_model.actor.parameters():
                parameter.zero_()
            self.agent.published_model.actor[-1].bias[~torch.tensor(mask)] = 1e6
        for deterministic in (False, True):
            decision = self.agent.act(self.observation(state), deterministic, action_mask=mask)
            self.assertTrue(mask[decision.action[0]])
            self.assertAlmostEqual(decision.log_prob, -math.log(sum(mask)), places=5)
            lp, entropy, _ = self.agent.evaluate_actions([decision.normalized_observation],
                [decision.action], action_masks=[mask])
            self.assertAlmostEqual(lp.item(), decision.log_prob, places=5)
            self.assertAlmostEqual(entropy.item(), math.log(sum(mask)), places=5)

    def test_missing_empty_or_nonboolean_mask_rejected(self):
        obs = self.observation(FinSequenceState())
        for mask in (None, [False]*18, [True]*17, [1]*18):
            with self.assertRaises(ValueError):
                self.agent.act(obs, action_mask=mask)

    def test_training_reuses_each_behavior_mask_across_odd_updates(self):
        state = FinSequenceState()
        previous_b1 = None
        last_direction = 0
        for update in range(6):
            state, rows = self.collect(state)
            for row in rows:
                action = FinAction(**self.agent.decode_action(row.action))
                self.assertNotEqual(action.b1, previous_b1)
                previous_b1 = action.b1
                delta = action.theta - row.observation[-3]*53
                direction = (action.b2 if delta > 0 else -action.b2) if action.b1 and delta else 0
                if direction:
                    self.assertNotEqual(direction, last_direction)
                    last_direction = direction
            lp, _, _ = self.agent.evaluate_actions([r.normalized_observation for r in rows],
                [r.action for r in rows], action_masks=[r.action_mask for r in rows])
            np.testing.assert_allclose(lp.detach(), [r.old_log_prob for r in rows], atol=1e-6)
            model_type = type(self.agent.training_model)
            evaluate = model_type.evaluate
            seen = []
            def capture(model, o, a, masks):
                seen.extend(tuple(m) for m in masks.tolist())
                return evaluate(model, o, a, masks)
            with patch.object(model_type, 'evaluate', capture):
                self.agent.update(rows)
            self.assertCountEqual(seen, [r.action_mask for r in rows]*2)
            self.agent.publish_candidate()
            self.assertEqual(self.agent.policy_version, update+1)
            self.assertEqual(state.next_b1, (update+1) % 2)

    def test_training_rejects_missing_forbidden_or_wrong_context_mask(self):
        _, rows = self.collect(FinSequenceState())
        first = rows[0]
        mask = list(first.action_mask)
        mask[first.action[0]] = False
        for replacement in (None, tuple(mask), (True,)*18):
            corrupted = [dataclasses.replace(first, action_mask=replacement), *rows[1:]]
            with self.assertRaises(ValueError):
                self.agent.update(corrupted)
        self.assertIsNone(self.agent.candidate_model)


if __name__ == '__main__':
    unittest.main()
