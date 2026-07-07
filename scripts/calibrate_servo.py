from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from control.safety import configured_servo_channels, ensure_not_windows_hardware_run, load_robot_config, sleep_safely  # noqa: E402
from control.safety import SafetyError  # noqa: E402
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def _find_channel_config(config: dict, servo_id: int | None, channel: int | None) -> dict:
    channels = configured_servo_channels(config)
    for item in channels:
        if servo_id is not None and int(item.get("servo_id", -1)) == servo_id:
            return item
        if channel is not None and int(item.get("channel", -1)) == channel:
            return item
    raise SystemExit("No matching servo found in config.")


def _format_status(item: dict, angle: float, step: float) -> str:
    return (
        f"servo_id={item.get('servo_id')} name={item.get('name')} "
        f"joint={item.get('joint_name')} channel={item.get('channel')} "
        f"angle={angle:.1f} step={step:.1f}"
    )


def _print_help() -> None:
    print("")
    print("Commands:")
    print("  +        move up by current step")
    print("  -        move down by current step")
    print("  s <deg>  set step, recommended 1.0 or smaller near a limit")
    print("  a <deg>  move to an absolute angle, limited to one current step")
    print("  c        print current angle as candidate center")
    print("  min      print current angle as candidate minimum")
    print("  max      print current angle as candidate maximum")
    print("  show     print current status")
    print("  q        recenter, release PWM, and quit")
    print("")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively calibrate one servo with small manual steps.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--servo-id", type=int, help="Mechanical servo id, 1..7.")
    selector.add_argument("--channel", type=int, help="PCA9685 channel.")
    parser.add_argument("--confirm", default="", help="Must be MOVE to command hardware.")
    parser.add_argument("--step", type=float, default=1.0, help="Initial manual step in degrees.")
    parser.add_argument(
        "--max-command-step",
        type=float,
        default=1.0,
        help="Largest accepted interactive command step in degrees.",
    )
    parser.add_argument("--hold-pwm", action="store_true", help="Keep PWM active when exiting.")
    return parser.parse_args()


def main() -> None:
    ensure_not_windows_hardware_run()
    args = _parse_args()
    if args.confirm != "MOVE":
        raise SystemExit("Refusing to move servo without --confirm MOVE.")
    if args.step <= 0:
        raise SystemExit("--step must be > 0.")
    if args.max_command_step <= 0:
        raise SystemExit("--max-command-step must be > 0.")

    config = load_robot_config()
    item = _find_channel_config(config, args.servo_id, args.channel)
    channel = int(item["channel"])
    controller = PCA9685ServoController(config)
    limits = controller.limits_for(channel)
    angle = limits.center_angle
    command_step_limit = abs(args.max_command_step)
    step = min(abs(args.step), command_step_limit)

    def move_to(next_angle: float) -> None:
        nonlocal angle
        try:
            next_angle = limits.validate(next_angle)
        except SafetyError as exc:
            print(f"Refusing move: {exc}")
            return
        if abs(next_angle - angle) > step:
            print(f"Refusing a single command larger than current step ({step:.1f} deg).")
            return
        controller.move_safely(channel, next_angle)
        angle = next_angle
        print(_format_status(item, angle, step))

    print("Starting single-servo calibration.")
    print(_format_status(item, angle, step))
    print(f"Configured safe range: {limits.min_angle:.1f}..{limits.max_angle:.1f}")
    print(f"Interactive command step limit: {command_step_limit:.1f} deg")
    print("Moving to configured center first.")
    _print_help()

    try:
        controller.move_safely(channel, angle)
        while True:
            raw = input("calibrate> ").strip().lower()
            if raw in {"", "show"}:
                print(_format_status(item, angle, step))
                continue
            if raw == "q":
                break
            if raw == "+":
                move_to(angle + step)
                continue
            if raw == "-":
                move_to(angle - step)
                continue
            if raw.startswith("s "):
                next_step = float(raw.split(maxsplit=1)[1])
                if next_step <= 0:
                    print("step must be > 0")
                    continue
                step = min(next_step, command_step_limit)
                print(_format_status(item, angle, step))
                continue
            if raw.startswith("a "):
                move_to(float(raw.split(maxsplit=1)[1]))
                continue
            if raw in {"c", "center"}:
                print(f"CANDIDATE center_angle: {angle:.1f}")
                continue
            if raw == "min":
                print(f"CANDIDATE min_angle: {angle:.1f}")
                continue
            if raw == "max":
                print(f"CANDIDATE max_angle: {angle:.1f}")
                continue
            print("Unknown command.")
            _print_help()
    except KeyboardInterrupt:
        print("")
        print("Interrupted.")
    finally:
        if not args.hold_pwm:
            print("Recentering and releasing PWM.")
            controller.move_safely(channel, limits.center_angle)
            sleep_safely(0.5)
            controller.stop_all([channel])


if __name__ == "__main__":
    main()
