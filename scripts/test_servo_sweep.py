from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from control.safety import (  # noqa: E402
    SafetyError,
    configured_servo_channels,
    ensure_not_windows_hardware_run,
    load_robot_config,
    servo_limits_from_config,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def main() -> None:
    ensure_not_windows_hardware_run()

    config = load_robot_config()
    first_channel = int(configured_servo_channels(config)[0].get("channel", 0))
    safety = config.get("safety", {}).get("servo", {})
    max_amplitude = float(safety.get("max_test_amplitude_deg", 15))

    parser = argparse.ArgumentParser(description="Conservative servo sweep. Requires explicit confirmation.")
    parser.add_argument("--confirm", action="store_true", help="Required to move the servo.")
    parser.add_argument("--channel", type=int, default=first_channel)
    parser.add_argument("--amplitude", type=float, default=10.0)
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing to move servo without --confirm.")
    if abs(args.amplitude) > max_amplitude:
        raise SafetyError(f"Sweep amplitude is limited to +/-{max_amplitude:.1f} degrees.")

    channel_config = next(
        (item for item in configured_servo_channels(config) if int(item.get("channel", 0)) == args.channel),
        None,
    )
    limits = servo_limits_from_config(config, channel_config)
    center = limits.center_angle
    points = [
        limits.validate(center),
        limits.validate(center + abs(args.amplitude)),
        limits.validate(center),
        limits.validate(center - abs(args.amplitude)),
        limits.validate(center),
    ]

    controller = PCA9685ServoController(config)
    try:
        for point in points:
            controller.move_safely(args.channel, point)
    except Exception:
        controller.stop_all([args.channel])
        raise
    finally:
        controller.stop_all([args.channel])


if __name__ == "__main__":
    main()

