from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from control.safety import SafetyError, ServoLimits, configured_servo_channels, servo_limits_from_config, sleep_safely


class ServoHardwareError(RuntimeError):
    """Raised when PCA9685/ServoKit hardware cannot be initialized."""


@dataclass(frozen=True)
class ServoChannel:
    name: str
    channel: int
    center_angle: float
    min_angle: float
    max_angle: float
    pulse_min_us: int = 500
    pulse_max_us: int = 2500


class PCA9685ServoController:
    def __init__(self, config: dict[str, Any]) -> None:
        servo_config = config.get("servo", {})
        i2c_config = config.get("i2c", {})

        self.frequency_hz = int(servo_config.get("frequency_hz", 50))
        self.address = int(i2c_config.get("pca9685_address", 0x40))
        self.channels = self._load_channels(config)
        base_limits = servo_limits_from_config(config)
        self.limits = {
            channel.channel: ServoLimits(
                center_angle=channel.center_angle,
                min_angle=channel.min_angle,
                max_angle=channel.max_angle,
                max_step_deg=base_limits.max_step_deg,
                step_delay_seconds=base_limits.step_delay_seconds,
            )
            for channel in self.channels.values()
        }
        self._last_angle: dict[int, float] = {}

        try:
            from adafruit_servokit import ServoKit
        except ImportError as exc:
            raise ServoHardwareError(
                "adafruit-circuitpython-servokit is required on the Raspberry Pi. "
                "Run codex_pi_workflow/setup_pi_venv.ps1 after syncing."
            ) from exc

        self.kit = ServoKit(channels=16, address=self.address, frequency=self.frequency_hz)
        for channel in self.channels.values():
            self.kit.servo[channel.channel].set_pulse_width_range(channel.pulse_min_us, channel.pulse_max_us)

    @staticmethod
    def _load_channels(config: dict[str, Any]) -> dict[int, ServoChannel]:
        result: dict[int, ServoChannel] = {}
        for raw in configured_servo_channels(config):
            limits = servo_limits_from_config(config, raw)
            channel = ServoChannel(
                name=str(raw.get("name", f"servo_{raw.get('channel', 0)}")),
                channel=int(raw.get("channel", 0)),
                center_angle=limits.center_angle,
                min_angle=limits.min_angle,
                max_angle=limits.max_angle,
                pulse_min_us=int(raw.get("pulse_min_us", 500)),
                pulse_max_us=int(raw.get("pulse_max_us", 2500)),
            )
            result[channel.channel] = channel
        return result

    def configured_channel_numbers(self) -> list[int]:
        return sorted(self.channels)

    def _limits_for(self, channel: int) -> ServoLimits:
        if channel not in self.limits:
            raise SafetyError(f"Servo channel {channel} is not configured.")
        return self.limits[channel]

    def set_angle(self, channel: int, angle: float) -> None:
        limits = self._limits_for(channel)
        safe_angle = limits.validate(float(angle))
        self.kit.servo[channel].angle = safe_angle
        self._last_angle[channel] = safe_angle
        print(f"servo channel {channel}: {safe_angle:.1f} deg")

    def move_safely(self, channel: int, target_angle: float) -> None:
        limits = self._limits_for(channel)
        target = limits.validate(float(target_angle))
        current = self._last_angle.get(channel)

        if current is None:
            self.set_angle(channel, target)
            sleep_safely(limits.step_delay_seconds)
            return

        step = max(0.1, abs(limits.max_step_deg))
        direction = 1 if target >= current else -1
        angle = current
        while abs(target - angle) > step:
            angle += direction * step
            self.set_angle(channel, angle)
            sleep_safely(limits.step_delay_seconds)
        self.set_angle(channel, target)
        sleep_safely(limits.step_delay_seconds)

    def center(self, channel: int) -> None:
        self.move_safely(channel, self._limits_for(channel).center_angle)

    def center_all(self, channels: Iterable[int] | None = None) -> None:
        for channel in channels or self.configured_channel_numbers():
            self.center(int(channel))

    def stop_all(self, channels: Iterable[int] | None = None) -> None:
        for channel in channels or self.configured_channel_numbers():
            try:
                self.kit.servo[int(channel)].angle = None
                print(f"servo channel {channel}: PWM released")
            except Exception as exc:  # noqa: BLE001
                print(f"servo channel {channel}: failed to release PWM: {exc}")
