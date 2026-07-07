from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from control.safety import ServoLimits, servo_limits_from_config, sleep_safely


@dataclass(frozen=True)
class ServoChannelConfig:
    name: str
    channel: int
    center_angle: float
    min_angle: float
    max_angle: float
    actuation_range: float
    pulse_min_us: int
    pulse_max_us: int


class PCA9685ServoController:
    """ServoKit-based PCA9685 controller used on the Raspberry Pi."""

    def __init__(self, config: dict[str, Any]) -> None:
        from adafruit_servokit import ServoKit
        from gpiozero import OutputDevice

        self._config = config
        servo_config = config.get("servo", {})
        self._channels = self._load_channel_configs()
        self._channel_lookup = {item.channel: item for item in self._channels}
        self._last_angles: dict[int, float] = {}

        enable_pin = servo_config.get("enable_pin")
        self._enable_pin = OutputDevice(int(enable_pin)) if enable_pin is not None else None
        if self._enable_pin is not None:
            self._enable_pin.on()

        self._kit = ServoKit(channels=int(servo_config.get("pca9685_channels", 16)))
        for item in self._channels:
            servo = self._kit.servo[item.channel]
            servo.set_pulse_width_range(item.pulse_min_us, item.pulse_max_us)
            servo.actuation_range = item.actuation_range

    def _load_channel_configs(self) -> list[ServoChannelConfig]:
        servo_config = self._config.get("servo", {})
        default_pulse_min = int(servo_config.get("pulse_min_us", 500))
        default_pulse_max = int(servo_config.get("pulse_max_us", 2500))
        raw_channels = servo_config.get("channels", [])
        channels: list[ServoChannelConfig] = []

        for raw in raw_channels:
            if not isinstance(raw, dict):
                continue
            limits = servo_limits_from_config(self._config, raw)
            channels.append(
                ServoChannelConfig(
                    name=str(raw.get("name", f"channel_{raw.get('channel', 0)}")),
                    channel=int(raw.get("channel", 0)),
                    center_angle=limits.center_angle,
                    min_angle=limits.min_angle,
                    max_angle=limits.max_angle,
                    actuation_range=float(raw.get("actuation_range", 180)),
                    pulse_min_us=int(raw.get("pulse_min_us", default_pulse_min)),
                    pulse_max_us=int(raw.get("pulse_max_us", default_pulse_max)),
                )
            )

        if not channels:
            limits = servo_limits_from_config(self._config, None)
            channels.append(
                ServoChannelConfig(
                    name="tail",
                    channel=0,
                    center_angle=limits.center_angle,
                    min_angle=limits.min_angle,
                    max_angle=limits.max_angle,
                    actuation_range=180,
                    pulse_min_us=default_pulse_min,
                    pulse_max_us=default_pulse_max,
                )
            )

        channel_numbers = [item.channel for item in channels]
        if len(channel_numbers) != len(set(channel_numbers)):
            raise ValueError("Configured servo channels must be unique.")
        return channels

    def configured_channels(self) -> list[int]:
        return [item.channel for item in self._channels]

    def limits_for(self, channel: int) -> ServoLimits:
        channel_config = self._channel_lookup.get(channel)
        raw_config = None
        for raw in self._config.get("servo", {}).get("channels", []):
            if isinstance(raw, dict) and int(raw.get("channel", -1)) == channel:
                raw_config = raw
                break
        if channel_config is None:
            raise ValueError(f"Servo channel {channel} is not configured.")
        safety_limits = servo_limits_from_config(self._config, raw_config)
        return ServoLimits(
            center_angle=channel_config.center_angle,
            min_angle=channel_config.min_angle,
            max_angle=channel_config.max_angle,
            max_step_deg=safety_limits.max_step_deg,
            step_delay_seconds=safety_limits.step_delay_seconds,
        )

    def center(self, channel: int) -> None:
        limits = self.limits_for(channel)
        self.move_safely(channel, limits.center_angle)

    def center_all(self, channels: Iterable[int] | None = None) -> None:
        for channel in self._resolve_channels(channels):
            self.center(channel)

    def move_safely(self, channel: int, angle: float) -> None:
        limits = self.limits_for(channel)
        target = limits.validate(float(angle))
        start = self._last_angles.get(channel, limits.center_angle)
        step = max(0.1, abs(limits.max_step_deg))

        delta = target - start
        if abs(delta) <= step:
            self._write_angle(channel, target)
            return

        direction = 1.0 if delta > 0 else -1.0
        current = start
        while abs(target - current) > step:
            current += direction * step
            self._write_angle(channel, current)
            sleep_safely(limits.step_delay_seconds)
        self._write_angle(channel, target)

    def write_angle(self, channel: int, angle: float) -> None:
        limits = self.limits_for(channel)
        target = limits.validate(float(angle))
        self._write_angle(channel, target)

    def write_angles(self, commands: dict[int, float]) -> None:
        targets: dict[int, float] = {}
        for channel, angle in commands.items():
            limits = self.limits_for(int(channel))
            targets[int(channel)] = limits.validate(float(angle))
        for channel, angle in targets.items():
            self._write_angle(channel, angle)

    def stop(self, channel: int) -> None:
        self._kit.servo[int(channel)].angle = None
        self._last_angles.pop(int(channel), None)

    def stop_all(self, channels: Iterable[int] | None = None) -> None:
        for channel in self._resolve_channels(channels):
            self.stop(channel)

    def _write_angle(self, channel: int, angle: float) -> None:
        self._kit.servo[int(channel)].angle = float(angle)
        self._last_angles[int(channel)] = float(angle)

    def _resolve_channels(self, channels: Iterable[int] | None) -> list[int]:
        if channels is None:
            return self.configured_channels()
        return [int(channel) for channel in channels]
