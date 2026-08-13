from __future__ import annotations

import math
import unittest

from control.runtime.action_utils import (
    asymmetric_tail_offset,
    cosine_cycle_position,
    countdown_start_delay,
    cycle_state,
    safe_recenter_profile,
    smoothstep5,
    tip_envelope,
    validate_finite_range,
)


class MotionProfileTests(unittest.TestCase):
    def test_smoothstep_endpoints_and_clamping(self) -> None:
        self.assertEqual(smoothstep5(-1.0), 0.0)
        self.assertEqual(smoothstep5(0.0), 0.0)
        self.assertEqual(smoothstep5(1.0), 1.0)
        self.assertEqual(smoothstep5(2.0), 1.0)

    def test_shared_cycle_clock_invariant(self) -> None:
        for elapsed in (0.0, 0.1, 2.499, 2.5, 99.99):
            state = cycle_state(elapsed, 0.4)
            self.assertGreaterEqual(state.phase, 0.0)
            self.assertLess(state.phase, 1.0)
            self.assertAlmostEqual(state.phase, state.t_cycle_s / state.period_s)

    def test_root_cosine_key_points_with_opposite_numeric_directions(self) -> None:
        period = 4.0
        root4_bottom, root4_top = 164.0, 119.0
        root6_bottom, root6_top = 100.0, 145.0
        expected = (
            (0.0, root4_bottom, root6_bottom),
            (period / 2.0, root4_top, root6_top),
            (period, root4_bottom, root6_bottom),
        )
        for elapsed, expected4, expected6 in expected:
            q = cosine_cycle_position(elapsed, period)
            angle4 = root4_bottom + q * (root4_top - root4_bottom)
            angle6 = root6_bottom + q * (root6_top - root6_bottom)
            self.assertAlmostEqual(angle4, expected4)
            self.assertAlmostEqual(angle6, expected6)

    def test_tip_envelope_all_required_boundaries(self) -> None:
        period = 4.0
        transition = 0.4
        checkpoints = {
            0.0: 0.0,
            transition: 1.0,
            period / 2.0 - transition: 1.0,
            period / 2.0: 0.0,
            period: 0.0,
        }
        for elapsed, expected in checkpoints.items():
            self.assertAlmostEqual(tip_envelope(elapsed, period, transition), expected)

    def test_tip_angles_are_opposite_about_configured_centers(self) -> None:
        period = 4.0
        transition = 0.4
        amplitude = 25.0
        center5 = 91.0
        center7 = 88.0
        for elapsed in (0.0, 0.2, 0.4, 1.6, 1.8, 2.0, 4.0):
            h = tip_envelope(elapsed, period, transition)
            angle5 = center5 - amplitude * h
            angle7 = center7 + amplitude * h
            self.assertAlmostEqual(center5 - angle5, angle7 - center7)

    def test_tip_envelope_is_continuous_at_piece_boundaries(self) -> None:
        period = 4.0
        transition = 0.4
        epsilon = 1e-8
        for boundary in (transition, period / 2.0 - transition, period / 2.0):
            left = tip_envelope(boundary - epsilon, period, transition)
            at = tip_envelope(boundary, period, transition)
            right = tip_envelope(boundary + epsilon, period, transition)
            self.assertLess(abs(left - at), 1e-10)
            self.assertLess(abs(right - at), 1e-10)

    def test_asymmetric_tail_starts_center_and_reaches_both_amplitudes(self) -> None:
        frequency = 0.5
        period = 1.0 / frequency
        left = 12.0
        right = 7.0
        self.assertAlmostEqual(asymmetric_tail_offset(0.0, frequency, left, right), 0.0)
        self.assertAlmostEqual(asymmetric_tail_offset(period / 4.0, frequency, left, right), left)
        self.assertAlmostEqual(asymmetric_tail_offset(3.0 * period / 4.0, frequency, left, right), -right)
        self.assertAlmostEqual(asymmetric_tail_offset(period, frequency, left, right), 0.0)

    def test_asymmetric_tail_uses_direct_sine_not_sine_cubed(self) -> None:
        frequency = 0.5
        period = 1.0 / frequency
        # sin(pi/6) = 0.5, so direct sine produces exactly half the left amplitude.
        self.assertAlmostEqual(asymmetric_tail_offset(period / 12.0, frequency, 12.0, 7.0), 6.0)

    def test_tail_first_direction_reverses_first_excursion_without_prepositioning(self) -> None:
        frequency = 1.0
        epsilon = 1e-3
        self.assertGreater(asymmetric_tail_offset(epsilon, frequency, 10.0, 6.0, "left"), 0.0)
        self.assertLess(asymmetric_tail_offset(epsilon, frequency, 10.0, 6.0, "right"), 0.0)
        self.assertAlmostEqual(asymmetric_tail_offset(0.0, frequency, 10.0, 6.0, "right"), 0.0)

    def test_finite_range_validation(self) -> None:
        self.assertEqual(validate_finite_range("ratio", 1.0, minimum=0.0, maximum=1.0), 1.0)
        with self.assertRaises(ValueError):
            validate_finite_range("ratio", math.inf, minimum=0.0, maximum=1.0)

    def test_safe_recenter_profile_has_exact_endpoints(self) -> None:
        start = {0: 110.0, 1: 60.0}
        center = {0: 90.0, 1: 90.0}
        at_start, start_state = safe_recenter_profile(start, center, 0.0, 2.0)
        at_end, end_state = safe_recenter_profile(start, center, 2.0, 2.0)
        self.assertEqual(at_start, start)
        self.assertEqual(at_end, center)
        self.assertEqual(start_state["tail_amplitude_scale"], 1.0)
        self.assertEqual(end_state["tail_amplitude_scale"], 0.0)

    def test_countdown_has_no_external_dependency_and_prints_once_per_second(self) -> None:
        now = [100.0]
        messages: list[str] = []

        def clock() -> float:
            return now[0]

        def sleeper(seconds: float) -> None:
            now[0] += seconds

        countdown_start_delay(2.2, printer=messages.append, clock=clock, sleeper=sleeper)
        self.assertEqual(messages, ["Action starts in 3 s", "Action starts in 2 s", "Action starts in 1 s"])


if __name__ == "__main__":
    unittest.main()
