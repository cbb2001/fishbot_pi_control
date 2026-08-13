from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from control.safety import (  # noqa: E402
    SafetyError,
    configured_servo_channels,
    ensure_not_windows_hardware_run,
    load_robot_config,
    sleep_safely,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively calibrate three servos with independent target angles."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--servo-ids",
        type=int,
        nargs=3,
        metavar=("ID1", "ID2", "ID3"),
        help="Three mechanical servo ids, for example: --servo-ids 1 2 3.",
    )
    selector.add_argument(
        "--channels",
        type=int,
        nargs=3,
        metavar=("CH1", "CH2", "CH3"),
        help="Three PCA9685 channels, for example: --channels 0 1 2.",
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command hardware.")
    parser.add_argument("--hold-pwm", action="store_true", help="Keep all three PWM outputs active when exiting.")
    return parser.parse_args()


def _select_items(config: dict, servo_ids: list[int] | None, channels: list[int] | None) -> list[dict]:
    configured = configured_servo_channels(config)
    requested = servo_ids if servo_ids is not None else channels
    key = "servo_id" if servo_ids is not None else "channel"
    if requested is None or len(requested) != 3:
        raise SystemExit("Exactly three servos must be selected.")
    if len(set(requested)) != 3:
        raise SystemExit("The three selected servos must be different.")

    by_value = {int(item.get(key, -1)): item for item in configured}
    missing = [value for value in requested if value not in by_value]
    if missing:
        raise SystemExit(f"No configured servo found for {key}={missing[0]}.")
    return [by_value[value] for value in requested]


def _item_label(item: dict) -> str:
    return (
        f"servo_id={int(item['servo_id'])} name={item.get('name')} "
        f"joint={item.get('joint_name')} channel={int(item['channel'])}"
    )


def _print_status(items: list[dict], angles: dict[int, float], limits_by_channel: dict) -> None:
    print("")
    for item in items:
        channel = int(item["channel"])
        limits = limits_by_channel[channel]
        print(
            f"{_item_label(item)} angle={angles[channel]:.1f} "
            f"safe_range={limits.min_angle:.1f}..{limits.max_angle:.1f}"
        )
    print("")


def _print_help(items: list[dict]) -> None:
    examples = [
        f"  {int(item['servo_id'])} <deg>     set servo {int(item['servo_id'])} independently"
        for item in items
    ]
    print("")
    print("Commands:")
    for line in examples:
        print(line)
    print("  set <id> <deg>  same as '<id> <deg>'")
    print("  center <id>     move one selected servo to its configured center")
    print("  center all      move all selected servos to configured centers")
    print("  c <id>          print current angle as candidate center")
    print("  min <id>        print current angle as candidate minimum")
    print("  max <id>        print current angle as candidate maximum")
    print("  show             print all current angles and configured ranges")
    print("  help             print these commands")
    print("  q                recenter all, release PWM, and quit")
    print("")


def main() -> None:
    ensure_not_windows_hardware_run()
    args = _parse_args()
    if args.confirm != "MOVE":
        raise SystemExit("Refusing to move servos without --confirm MOVE.")

    config = load_robot_config()
    items = _select_items(config, args.servo_ids, args.channels)
    controller = PCA9685ServoController(config)
    channels = [int(item["channel"]) for item in items]
    limits_by_channel = {channel: controller.limits_for(channel) for channel in channels}
    items_by_servo_id = {int(item["servo_id"]): item for item in items}
    angles = {
        channel: float(limits_by_channel[channel].center_angle)
        for channel in channels
    }

    def move_item(item: dict, requested_angle: float) -> None:
        channel = int(item["channel"])
        limits = limits_by_channel[channel]
        try:
            target = limits.validate(float(requested_angle))
        except SafetyError as exc:
            print(f"Refusing move: {exc}")
            return
        controller.move_safely(channel, target)
        angles[channel] = target
        print(f"{_item_label(item)} angle={target:.1f}")

    def selected_item(text: str) -> dict | None:
        try:
            servo_id = int(text)
        except ValueError:
            print("Servo id must be an integer.")
            return None
        item = items_by_servo_id.get(servo_id)
        if item is None:
            selected = ", ".join(str(value) for value in items_by_servo_id)
            print(f"Servo {servo_id} is not selected. Selected servo ids: {selected}")
            return None
        return item

    print("Starting independent three-servo calibration.")
    print("Moving all selected servos to their configured centers first.")
    _print_status(items, angles, limits_by_channel)
    _print_help(items)

    try:
        for item in items:
            channel = int(item["channel"])
            controller.move_safely(channel, angles[channel])

        while True:
            raw = input("triple-calibrate> ").strip().lower()
            if raw in {"", "show"}:
                _print_status(items, angles, limits_by_channel)
                continue
            if raw in {"q", "quit", "exit"}:
                break
            if raw in {"help", "?"}:
                _print_help(items)
                continue

            parts = raw.split()
            try:
                if len(parts) == 2 and parts[0].lstrip("+-").isdigit():
                    item = selected_item(parts[0])
                    if item is not None:
                        move_item(item, float(parts[1]))
                    continue

                if len(parts) == 3 and parts[0] == "set":
                    item = selected_item(parts[1])
                    if item is not None:
                        move_item(item, float(parts[2]))
                    continue

                if len(parts) == 2 and parts[0] == "center":
                    if parts[1] == "all":
                        for item in items:
                            channel = int(item["channel"])
                            move_item(item, limits_by_channel[channel].center_angle)
                    else:
                        item = selected_item(parts[1])
                        if item is not None:
                            channel = int(item["channel"])
                            move_item(item, limits_by_channel[channel].center_angle)
                    continue

                if len(parts) == 2 and parts[0] in {"c", "min", "max"}:
                    item = selected_item(parts[1])
                    if item is not None:
                        channel = int(item["channel"])
                        field = {"c": "center_angle", "min": "min_angle", "max": "max_angle"}[parts[0]]
                        print(f"CANDIDATE servo_id={int(item['servo_id'])} {field}: {angles[channel]:.1f}")
                    continue
            except ValueError:
                print("Angle must be a number.")
                continue

            print("Unknown command.")
            _print_help(items)
    except KeyboardInterrupt:
        print("")
        print("Interrupted.")
    finally:
        if not args.hold_pwm:
            print("Recentering all selected servos and releasing PWM.")
            for item in items:
                channel = int(item["channel"])
                controller.move_safely(channel, limits_by_channel[channel].center_angle)
            sleep_safely(0.5)
            controller.stop_all(channels)


if __name__ == "__main__":
    main()
