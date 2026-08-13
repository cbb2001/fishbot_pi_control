from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from control.runtime.action_utils import (
    asymmetric_tail_offset,
    cosine_cycle_position,
    cycle_state,
    tip_envelope,
    validate_finite_range,
)


@dataclass(frozen=True)
class RootPairTargets:
    """Physical-extreme start and ratio-derived targets for mirrored root servos."""

    roll_direction: str
    servo_4_start_deg: float
    servo_4_target_deg: float
    servo_6_start_deg: float
    servo_6_target_deg: float


@dataclass(frozen=True)
class TailServoMotion:
    center_deg: float
    direction_sign: float
    amplitude_scale: float = 1.0


@dataclass(frozen=True)
class RollTrajectorySample:
    action_elapsed_s: float
    pectoral_period_s: float
    pectoral_t_cycle_s: float
    pectoral_phase: float
    pectoral_cycle_index: int
    root_cosine_q: float
    tip_envelope_h: float
    tail_offset_deg: float
    commands_by_servo_id_deg: dict[int, float]


def build_roll_root_targets(
    *,
    roll_direction: str,
    root_amplitude_ratio: float,
    servo_4_top_deg: float,
    servo_4_bottom_deg: float,
    servo_6_top_deg: float,
    servo_6_bottom_deg: float,
) -> RootPairTargets:
    """Build equal-ratio, physically opposed root motion without guessing angle signs."""

    if roll_direction not in {"left", "right"}:
        raise ValueError("roll_direction must be 'left' or 'right'.")
    ratio = validate_finite_range(
        "root_amplitude_ratio",
        root_amplitude_ratio,
        minimum=0.0,
        maximum=1.0,
    )
    endpoints = {
        "servo_4_top_deg": servo_4_top_deg,
        "servo_4_bottom_deg": servo_4_bottom_deg,
        "servo_6_top_deg": servo_6_top_deg,
        "servo_6_bottom_deg": servo_6_bottom_deg,
    }
    finite_endpoints = {
        name: validate_finite_range(name, value)
        for name, value in endpoints.items()
    }

    if roll_direction == "right":
        start4 = finite_endpoints["servo_4_bottom_deg"]
        end4 = finite_endpoints["servo_4_top_deg"]
        start6 = finite_endpoints["servo_6_top_deg"]
        end6 = finite_endpoints["servo_6_bottom_deg"]
    else:
        start4 = finite_endpoints["servo_4_top_deg"]
        end4 = finite_endpoints["servo_4_bottom_deg"]
        start6 = finite_endpoints["servo_6_bottom_deg"]
        end6 = finite_endpoints["servo_6_top_deg"]

    target4 = _move_toward_physical_endpoint(4, start4, end4, ratio)
    target6 = _move_toward_physical_endpoint(6, start6, end6, ratio)
    return RootPairTargets(
        roll_direction=roll_direction,
        servo_4_start_deg=start4,
        servo_4_target_deg=target4,
        servo_6_start_deg=start6,
        servo_6_target_deg=target6,
    )


def tip_sign_from_direction(tip_direction: str) -> int:
    if tip_direction == "positive":
        return 1
    if tip_direction == "negative":
        return -1
    raise ValueError("tip_direction must be 'positive' or 'negative'.")


def evaluate_roll_trajectory(
    action_elapsed_s: float,
    *,
    pectoral_frequency_hz: float,
    roots: RootPairTargets,
    tip_centers_deg: Mapping[int, float],
    tip_amplitude_deg: float,
    tip_transition_s: float,
    tip_direction: str,
    tail_frequency_hz: float,
    tail_left_amplitude_deg: float,
    tail_right_amplitude_deg: float,
    tail_first_direction: str,
    tail_servos: Mapping[int, TailServoMotion],
) -> RollTrajectorySample:
    """Evaluate all seven target angles from one action elapsed-time value."""

    elapsed = validate_finite_range("action_elapsed_s", action_elapsed_s, minimum=0.0)
    tip_amplitude = validate_finite_range("tip_amplitude_deg", tip_amplitude_deg, minimum=0.0)
    for servo_id in (5, 7):
        if servo_id not in tip_centers_deg:
            raise ValueError(f"tip_centers_deg is missing servo_id={servo_id}.")
    for servo_id in (1, 2, 3):
        if servo_id not in tail_servos:
            raise ValueError(f"tail_servos is missing servo_id={servo_id}.")

    clock = cycle_state(elapsed, pectoral_frequency_hz)
    q = cosine_cycle_position(clock.t_cycle_s, clock.period_s)
    h = tip_envelope(clock.t_cycle_s, clock.period_s, tip_transition_s)
    tip_sign = tip_sign_from_direction(tip_direction)
    logical_tail_offset = asymmetric_tail_offset(
        elapsed,
        tail_frequency_hz,
        tail_left_amplitude_deg,
        tail_right_amplitude_deg,
        tail_first_direction,
    )

    commands = {
        4: roots.servo_4_start_deg
        + q * (roots.servo_4_target_deg - roots.servo_4_start_deg),
        5: float(tip_centers_deg[5]) + tip_sign * tip_amplitude * h,
        6: roots.servo_6_start_deg
        + q * (roots.servo_6_target_deg - roots.servo_6_start_deg),
        7: float(tip_centers_deg[7]) + tip_sign * tip_amplitude * h,
    }
    for servo_id in (1, 2, 3):
        motion = tail_servos[servo_id]
        commands[servo_id] = (
            float(motion.center_deg)
            + float(motion.direction_sign)
            * float(motion.amplitude_scale)
            * logical_tail_offset
        )

    return RollTrajectorySample(
        action_elapsed_s=elapsed,
        pectoral_period_s=clock.period_s,
        pectoral_t_cycle_s=clock.t_cycle_s,
        pectoral_phase=clock.phase,
        pectoral_cycle_index=clock.cycle_index,
        root_cosine_q=q,
        tip_envelope_h=h,
        tail_offset_deg=logical_tail_offset,
        commands_by_servo_id_deg=commands,
    )


def _move_toward_physical_endpoint(
    servo_id: int,
    start_deg: float,
    endpoint_deg: float,
    travel_ratio: float,
) -> float:
    delta = endpoint_deg - start_deg
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"servo_id={servo_id} physical top and bottom endpoints must be distinct."
        )
    return start_deg + travel_ratio * delta
