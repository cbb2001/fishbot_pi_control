from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from control.safety import configured_servo_channels, ensure_not_windows_hardware_run, load_robot_config, sleep_safely  # noqa: E402
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def parse_channels(raw: str | None, config: dict) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(item.get("channel", 0)) for item in configured_servo_channels(config)]


def main() -> None:
    ensure_not_windows_hardware_run()

    parser = argparse.ArgumentParser(description="Center configured servos conservatively.")
    parser.add_argument("--channels", default=None, help="Comma-separated channel list. Defaults to configured channels.")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="How long to hold center before releasing PWM.")
    parser.add_argument("--hold-pwm", action="store_true", help="Leave PWM active after centering.")
    args = parser.parse_args()

    config = load_robot_config()
    safety = config.get("safety", {}).get("servo", {})
    release_after = bool(safety.get("release_pwm_after_tests", True)) and not args.hold_pwm

    controller = PCA9685ServoController(config)
    channels = parse_channels(args.channels, config)

    try:
        print(f"Centering servo channels: {channels}")
        controller.center_all(channels)
        sleep_safely(args.hold_seconds)
    except Exception:
        controller.stop_all(channels)
        raise
    finally:
        if release_after:
            controller.stop_all(channels)


if __name__ == "__main__":
    main()

