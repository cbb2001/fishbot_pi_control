import unittest

from control.runtime.rl_episode_20260914 import EpisodeMonitor


def state(seconds, depth=.5, yaw=0., **extra):
    return {'t_ns': round(seconds * 1e9), 'depth_m': depth, 'yaw_deg': yaw, **extra}


class EpisodeTerminationTests(unittest.TestCase):
    def monitor(self, **overrides):
        monitor = EpisodeMonitor({'episode_duration_s': 60., **overrides},
                                 target_depth_m=.5, target_heading_deg=0.)
        monitor.start(3, 0)
        return monitor

    def test_both_targets_must_hold_for_continuous_real_time(self):
        monitor = self.monitor()
        for step in range(20):
            self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
        decision = monitor.observe(state(2.), action_count=1)
        self.assertEqual(decision.reason, 'success')
        self.assertTrue(decision.terminated)
        self.assertFalse(decision.truncated)
        self.assertEqual(decision.episode_index, 3)
        self.assertEqual(decision.elapsed_s, 2.)

    def test_single_crossing_and_either_out_of_bounds_reset_hold(self):
        for outside in (state(1.5, depth=.56), state(1.5, yaw=5.1)):
            monitor = self.monitor()
            for step in range(15):
                self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
            self.assertIsNone(monitor.observe(outside, action_count=1))
            for step in range(16, 36):
                self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
            self.assertEqual(monitor.observe(state(3.6), action_count=1).reason, 'success')

    def test_yaw_wrap_uses_shortest_distance_at_boundary(self):
        monitor = EpisodeMonitor({'success_hold_s': .2, 'min_episode_s': 0},
                                 target_depth_m=.5, target_heading_deg=179.)
        monitor.start(0, 0)
        self.assertIsNone(monitor.observe(state(0, yaw=-179), action_count=1))
        self.assertIsNone(monitor.observe(state(.1, yaw=174), action_count=1))
        self.assertEqual(monitor.observe(state(.2, yaw=-178), action_count=1).reason, 'success')

    def test_inclusive_depth_and_yaw_tolerances(self):
        monitor = self.monitor(success_hold_s=.2, min_episode_s=0)
        for moment in (0, .1):
            self.assertIsNone(monitor.observe(state(moment, depth=.55, yaw=5), action_count=1))
        self.assertEqual(monitor.observe(state(.2, depth=.45, yaw=-5), action_count=1).reason, 'success')

    def test_min_episode_and_completed_action_count_gate_success(self):
        monitor = self.monitor(success_hold_s=.2, min_episode_s=1., min_actions=2)
        for step in range(11):
            self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
        self.assertEqual(monitor.observe(state(1.1), action_count=2).reason, 'success')

    def test_observation_gap_restarts_success_window(self):
        monitor = self.monitor(success_hold_s=.3, min_episode_s=0)
        for moment in (0., .1, .2, 4., 4.1, 4.2):
            self.assertIsNone(monitor.observe(state(moment), action_count=1))
        self.assertEqual(monitor.observe(state(4.3), action_count=1).reason, 'success')

    def test_duplicate_and_backwards_state_cannot_complete_window(self):
        monitor = self.monitor(success_hold_s=.3, min_episode_s=0)
        for moment in (0, .1, .2, .2, .15, .3, .4, .5):
            self.assertIsNone(monitor.observe(state(moment), action_count=1))
        self.assertEqual(monitor.observe(state(.6), action_count=1).reason, 'success')

    def test_nonfinite_sample_resets_continuity(self):
        for field in ('depth_m', 'yaw_deg', 't_ns'):
            for invalid in (float('nan'), float('inf')):
                with self.subTest(field=field, invalid=invalid):
                    monitor = self.monitor(success_hold_s=.3, min_episode_s=0)
                    for moment in (0, .1, .2):
                        self.assertIsNone(monitor.observe(state(moment), action_count=1))
                    bad = state(.3)
                    bad[field] = invalid
                    self.assertIsNone(monitor.observe(bad, action_count=1))
                    for moment in (.4, .5, .6):
                        self.assertIsNone(monitor.observe(state(moment), action_count=1))
                    self.assertEqual(monitor.observe(state(.7), action_count=1).reason, 'success')

    def test_reused_sensor_evidence_cannot_extend_success_window(self):
        monitor = self.monitor(success_hold_s=.3, min_episode_s=0)
        for moment in (0, .1, .2, .3, .4, .5):
            stamps = {'imu': round(moment * 1e9), 'depth': 0}
            self.assertIsNone(monitor.observe(state(moment, sensor_sample_t_ns=stamps), action_count=1))
        for moment in (.6, .7, .8):
            stamps = {'imu': round(moment * 1e9), 'depth': round(moment * 1e9)}
            self.assertIsNone(monitor.observe(state(moment, sensor_sample_t_ns=stamps), action_count=1))
        stamps = {'imu': 900_000_000, 'depth': 900_000_000}
        self.assertEqual(monitor.observe(state(.9, sensor_sample_t_ns=stamps), action_count=1).reason, 'success')

    def test_time_limit_advances_without_observations_during_update(self):
        monitor = self.monitor(episode_duration_s=2.)
        self.assertIsNone(monitor.check_time(1_999_999_999))
        decision = monitor.check_time(2_000_000_000)
        self.assertEqual(decision.reason, 'time_limit')
        self.assertFalse(decision.terminated)
        self.assertTrue(decision.truncated)
        self.assertEqual(decision.end_ns, 2_000_000_000)

    def test_time_limit_precedes_success_at_same_deadline(self):
        monitor = self.monitor(episode_duration_s=2.)
        for step in range(20):
            self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
        self.assertEqual(monitor.observe(state(2.), action_count=1).reason, 'time_limit')

    def test_success_is_latched_until_next_episode(self):
        monitor = self.monitor(success_hold_s=.2, min_episode_s=0)
        for moment in (0, .1):
            self.assertIsNone(monitor.observe(state(moment), action_count=1))
        decision = monitor.observe(state(.2), action_count=1)
        self.assertIs(monitor.observe(state(.5, depth=8., yaw=80), action_count=1), decision)
        self.assertIs(monitor.check_time(100_000_000_000), decision)
        monitor.start(4, 100_000_000_000)
        self.assertIsNone(monitor.observe(state(100.), action_count=1))

    def test_disabled_success_only_expires_at_duration(self):
        monitor = self.monitor(success_enabled=False, episode_duration_s=3.)
        for step in range(30):
            self.assertIsNone(monitor.observe(state(step / 10), action_count=1))
        self.assertEqual(monitor.observe(state(3.), action_count=1).reason, 'time_limit')

    def test_nested_training_and_termination_config(self):
        monitor = EpisodeMonitor({'training': {'episode_duration_s': 1.,
                                 'termination': {'success_enabled': False}}})
        monitor.start(0, 1_000_000_000)
        self.assertIsNone(monitor.check_time(1_999_999_999))
        self.assertEqual(monitor.check_time(2_000_000_000).reason, 'time_limit')

    def test_invalid_thresholds_are_rejected(self):
        for override in ({'episode_duration_s': 0}, {'success_hold_s': -1},
                         {'depth_tolerance_m': float('nan')}, {'yaw_tolerance_deg': -1},
                         {'max_sample_gap_s': 0}, {'min_actions': 1.5}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.monitor(**override)


if __name__ == '__main__':
    unittest.main()
