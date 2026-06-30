from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "robot.yaml"


class SafetyError(RuntimeError):
    """Raised when a requested motion exceeds configured safety limits."""


@dataclass(frozen=True)
class ServoLimits:
    center_angle: float = 90.0
    min_angle: float = 70.0
    max_angle: float = 110.0
    max_step_deg: float = 1.0
    step_delay_seconds: float = 0.08

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, angle))

    def validate(self, angle: float) -> float:
        clamped = self.clamp(angle)
        if clamped != angle:
            raise SafetyError(
                f"Requested angle {angle:.1f} exceeds limits "
                f"{self.min_angle:.1f}..{self.max_angle:.1f}"
            )
        return angle


def ensure_not_windows_hardware_run() -> None:
    if platform.system().lower().startswith("win"):
        raise SystemExit(
            "Hardware scripts must be run on the Raspberry Pi via SSH, "
            "not on the Windows development machine."
        )


def load_robot_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Run setup_pi_venv.ps1 on the Raspberry Pi.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise SafetyError(f"Config file must contain a mapping: {config_path}")

    return data


def servo_limits_from_config(config: dict[str, Any], channel_config: dict[str, Any] | None = None) -> ServoLimits:
    channel_config = channel_config or {}
    safety = config.get("safety", {}).get("servo", {})

    return ServoLimits(
        center_angle=float(channel_config.get("center_angle", safety.get("default_center_angle", 90))),
        min_angle=float(channel_config.get("min_angle", safety.get("default_min_angle", 70))),
        max_angle=float(channel_config.get("max_angle", safety.get("default_max_angle", 110))),
        max_step_deg=float(safety.get("max_step_deg", 1)),
        step_delay_seconds=float(safety.get("step_delay_seconds", 0.08)),
    )


def configured_servo_channels(config: dict[str, Any]) -> list[dict[str, Any]]:
    channels = config.get("servo", {}).get("channels", [])
    if not channels:
        return [{"name": "tail", "channel": 0}]
    return channels


def sleep_safely(seconds: float) -> None:
    time.sleep(max(0.0, seconds))

