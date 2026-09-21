from __future__ import annotations

import dataclasses
import unittest
import numpy as np
from control.runtime.rl_rollout_buffer_20260914 import (
    Transition,
    FrozenRollout,
    RolloutBuffer,
    compute_gae,
)


def tr(reward=1, value=0, next_value=0, duration=0.2, episode=0, **kw):
    return Transition(
        (0.0,),
        (0, 0),
        reward,
        (0.0,),
        0.0,
        value,
        next_value,
        (0.0,),
        duration_s=duration,
        episode_index=episode,
        **kw,
    )


class GAETests(unittest.TestCase):
    def test_gamma_and_lambda_use_physical_duration(self):
        data = [tr(reward=1, duration=0.4), tr(reward=2, duration=0.2)]
        returns, advantage = compute_gae(data, gamma=0.9, gae_lambda=0.8)
        np.testing.assert_allclose(advantage, [1 + 0.9**2 * 0.8**2 * 2, 2])
        np.testing.assert_allclose(returns, advantage)

    def test_truncation_bootstraps_without_leaking_next_episode_rewards(self):
        data = [
            tr(reward=1, value=3, next_value=10, truncated=True),
            tr(reward=999, episode=1),
        ]
        returns, advantage = compute_gae(data, gamma=0.9, gae_lambda=1)
        self.assertAlmostEqual(returns[0], 10)
        self.assertAlmostEqual(advantage[0], 7)
        # An explicit episode id boundary also cuts recursion even without a flag.
        data[0] = dataclasses.replace(data[0], truncated=False)
        self.assertAlmostEqual(compute_gae(data, gamma=0.9)[0][0], 10)

    def test_true_termination_removes_bootstrap(self):
        data = [tr(reward=2, value=3, next_value=1000, terminated=True)]
        returns, advantage = compute_gae(data)
        self.assertAlmostEqual(returns[0], 2)
        self.assertAlmostEqual(advantage[0], -1)

    def test_frozen_rollout_owns_values_and_rejects_mixed_policy(self):
        raw = [1.0]
        sample = Transition(raw, [0, 0], 1.0, raw, 0.0, 0.0, 0.0, raw)
        raw[0] = 900
        self.assertEqual(sample.observation, (1.0,))
        buffer = RolloutBuffer()
        buffer.add(sample)
        buffer.end_episode()
        frozen = buffer.freeze()
        buffer.clear()
        self.assertEqual(len(frozen), 1)
        self.assertTrue(frozen.transitions[0].truncated)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            frozen.transitions[0].reward = 2
        with self.assertRaises(ValueError):
            FrozenRollout(
                (sample, dataclasses.replace(sample, behavior_policy_version=2)), 0
            )


if __name__ == "__main__":
    unittest.main()
