from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from control.safety import configured_servo_channels, servo_limits_from_config


@dataclass(frozen=True)
class CycleState:
    period_s: float
    t_cycle_s: float
    phase: float
    cycle_index: int


def validate_finite_range(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Validate one finite numeric parameter and return it as a float."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if minimum is not None:
        invalid = numeric < minimum if minimum_inclusive else numeric <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}, got {numeric}.")
    if maximum is not None:
        invalid = numeric > maximum if maximum_inclusive else numeric >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {numeric}.")
    return numeric


def smoothstep5(u: float) -> float:
    """Quintic smoothstep with zero first and second derivatives at both ends."""
    x = max(0.0, min(1.0, float(u)))
    return (10.0 * x**3) - (15.0 * x**4) + (6.0 * x**5)


def cycle_state(elapsed_s: float, frequency_hz: float) -> CycleState:
    frequency = validate_finite_range("frequency_hz", frequency_hz, minimum=0.0, minimum_inclusive=False)
    elapsed = validate_finite_range("elapsed_s", elapsed_s, minimum=0.0)
    period_s = 1.0 / frequency
    t_cycle_s = elapsed % period_s
    phase = t_cycle_s / period_s
    # Protect the documented invariant against a possible floating-point round-up.
    if phase >= 1.0:
        phase = 0.0
        t_cycle_s = 0.0
    return CycleState(
        period_s=period_s,
        t_cycle_s=t_cycle_s,
        phase=phase,
        cycle_index=int(math.floor(elapsed / period_s)),
    )


def cosine_cycle_position(t_cycle_s: float, period_s: float) -> float:
    period = validate_finite_range("period_s", period_s, minimum=0.0, minimum_inclusive=False)
    t_cycle = float(t_cycle_s) % period
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * t_cycle / period))


def tip_envelope(t_cycle_s: float, period_s: float, transition_s: float) -> float:
    period = validate_finite_range("period_s", period_s, minimum=0.0, minimum_inclusive=False)
    transition = validate_finite_range(
        "transition_s",
        transition_s,
        minimum=0.0,
        maximum=period / 4.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    t_cycle = float(t_cycle_s) % period
    half_period = period / 2.0
    if t_cycle < transition:
        return smoothstep5(t_cycle / transition)
    if t_cycle < half_period - transition:
        return 1.0
    if t_cycle < half_period:
        progress = (t_cycle - (half_period - transition)) / transition
        return 1.0 - smoothstep5(progress)
    return 0.0


def asymmetric_tail_offset(
    elapsed_s: float,
    frequency_hz: float,
    left_amplitude_deg: float,
    right_amplitude_deg: float,
    first_direction: str = "left",
) -> float:
    elapsed = validate_finite_range("elapsed_s", elapsed_s, minimum=0.0)
    frequency = validate_finite_range("frequency_hz", frequency_hz, minimum=0.0, minimum_inclusive=False)
    left = validate_finite_range("left_amplitude_deg", left_amplitude_deg, minimum=0.0)
    right = validate_finite_range("right_amplitude_deg", right_amplitude_deg, minimum=0.0)
    if first_direction not in {"left", "right"}:
        raise ValueError("first_direction must be 'left' or 'right'.")
    phase_sign = 1.0 if first_direction == "left" else -1.0
    sine = phase_sign * math.sin(2.0 * math.pi * frequency * elapsed)
    amplitude = left if sine >= 0.0 else right
    return amplitude * sine


def interpolate_angles(
    start_angles: dict[int, float],
    end_angles: dict[int, float],
    progress: float,
) -> dict[int, float]:
    envelope = smoothstep5(progress)
    return {
        int(channel): float(start_angles[channel])
        + envelope * (float(end_angle) - float(start_angles[channel]))
        for channel, end_angle in end_angles.items()
    }


def safe_recenter_profile(
    start_angles: dict[int, float],
    center_angles: dict[int, float],
    elapsed_s: float,
    duration_s: float,
) -> tuple[dict[int, float], dict[str, float]]:
    duration = validate_finite_range("duration_s", duration_s, minimum=0.0, minimum_inclusive=False)
    progress = max(0.0, min(1.0, float(elapsed_s) / duration))
    envelope = smoothstep5(progress)
    return (
        interpolate_angles(start_angles, center_angles, progress),
        {
            "transition_progress": progress,
            "tail_amplitude_scale": 1.0 - envelope,
        },
    )


def countdown_start_delay(
    delay_s: float,
    *,
    printer: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Run a local deterministic countdown without starting logging or sensors."""
    delay = validate_finite_range("start_delay_s", delay_s, minimum=0.0)
    if delay <= 0.0:
        return
    deadline = clock() + delay
    last_displayed: int | None = None
    while True:
        remaining = deadline - clock()
        if remaining <= 0.0:
            return
        displayed = max(1, int(math.ceil(remaining)))
        if displayed != last_displayed:
            printer(f"Action starts in {displayed} s")
            last_displayed = displayed
        sleeper(min(1.0, remaining))


def action_event(
    event_logger: Any,
    event_type: str,
    action: str,
    *,
    action_state: str | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    payload = {"action": action}
    if action_state is not None:
        payload["action_state"] = action_state
    if data:
        payload.update(data)
    return bool(event_logger.write(event_type, payload))


class DryRunServoController:
    """PCA9685-compatible validator that never imports or writes hardware drivers."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._raw_by_channel: dict[int, dict[str, Any]] = {}
        self._last_angles: dict[int, float] = {}
        for raw in configured_servo_channels(config):
            self._raw_by_channel[int(raw.get("channel", 0))] = raw

    def limits_for(self, channel: int) -> Any:
        raw = self._raw_by_channel.get(int(channel))
        if raw is None:
            raise ValueError(f"Servo channel {channel} is not configured.")
        return servo_limits_from_config(self._config, raw)

    def move_safely(self, channel: int, angle: float) -> None:
        self.write_angle(channel, angle)

    def write_angle(self, channel: int, angle: float) -> None:
        limits = self.limits_for(int(channel))
        self._last_angles[int(channel)] = limits.validate(float(angle))

    def write_angles(self, commands: dict[int, float]) -> None:
        validated: dict[int, float] = {}
        for channel, angle in commands.items():
            limits = self.limits_for(int(channel))
            validated[int(channel)] = limits.validate(float(angle))
        self._last_angles.update(validated)

    def stop_all(self, channels: Iterable[int] | None = None) -> None:
        if channels is None:
            self._last_angles.clear()
            return
        for channel in channels:
            self._last_angles.pop(int(channel), None)
