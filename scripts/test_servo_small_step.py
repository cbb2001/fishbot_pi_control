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
    sleep_safely,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def main() -> None:
    ensure_not_windows_hardware_run()

    config = load_robot_config()
    first_channel = int(configured_servo_channels(config)[0].get("channel", 0))
    safety = config.get("safety", {}).get("servo", {})

    parser = argparse.ArgumentParser(description="Move one servo by a very small safe step.")
    parser.add_argument("--channel", type=int, default=first_channel, help="Servo channel to test.")
    parser.add_argument("--delta", type=float, default=float(safety.get("small_step_delta_deg", 5)), help="Degrees from center.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of small-step cycles.")
    parser.add_argument("--hold-pwm", action="store_true", help="Leave PWM active after the test.")
    args = parser.parse_args()

    max_delta = float(safety.get("small_step_delta_deg", 5))
    if abs(args.delta) > max_delta:
        raise SafetyError(f"Small-step delta is limited to +/-{max_delta:.1f} degrees.")

    channel_config = next(
        (item for item in configured_servo_channels(config) if int(item.get("channel", 0)) == args.channel),
        None,
    )
    limits = servo_limits_from_config(config, channel_config)
    center = limits.center_angle
    high = limits.validate(center + abs(args.delta))
    low = limits.validate(center - abs(args.delta))

    controller = PCA9685ServoController(config)
    release_after = bool(safety.get("release_pwm_after_tests", True)) and not args.hold_pwm

    try:
        print(f"Small-step test on channel {args.channel}: {low:.1f} -> {center:.1f} -> {high:.1f}")
        controller.center(args.channel)
        for _ in range(max(1, args.cycles)):
            controller.move_safely(args.channel, high)
            sleep_safely(0.3)
            controller.move_safely(args.channel, low)
            sleep_safely(0.3)
            controller.move_safely(args.channel, center)
    except Exception:
        controller.stop_all([args.channel])
        raise
    finally:
        if release_after:
            controller.stop_all([args.channel])


if __name__ == "__main__":
    main()

