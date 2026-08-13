from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()

from control.runtime.data_logger import (  # noqa: E402
    JsonlLogger,
    RawSensorLoggers,
    create_run_log_dir,
    prepare_raw_placeholders,
    write_metadata_yaml,
)
from control.runtime.event_logger import EventLogger  # noqa: E402
from control.runtime.link_manager import link_manager_from_config  # noqa: E402
from control.runtime.sensor_manager import SensorManager  # noqa: E402
from control.runtime.sensor_synchronizer import SensorSynchronizer  # noqa: E402
from control.safety import (  # noqa: E402
    configured_servo_channels,
    ensure_not_windows_hardware_run,
    load_robot_config,
    sleep_safely,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


ACTION_NAME = "pectoral_root_flap_tail_sweep_2"
TAIL_SERVO_IDS = (1, 2, 3)
PECTORAL_ROOT_SERVO_IDS = (4, 6)
PECTORAL_TIP_SERVO_IDS = (5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Action 2: pectoral roots 4/6 flap up-down periodically, pectoral tips "
            "5/7 switch during upstroke, and tail servos 1/2/3 sweep periodically. "
            "Sensors are recorded during the action."
        )
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command servos.")
    parser.add_argument("--duration", type=float, default=10.0, help="Action duration in seconds.")
    parser.add_argument("--pectoral-frequency", type=float, default=0.5, help="Pectoral flap frequency in Hz.")
    parser.add_argument(
        "--pectoral-root-scale",
        type=float,
        default=1.0,
        help="Scale root motion between top and bottom, 0..1. Default 1 means full calibrated range.",
    )
    parser.add_argument(
        "--tip-return-fraction",
        type=float,
        default=0.8,
        help="Upstroke progress where servo 5/7 start returning to 90. Must be in [0, 1).",
    )
    parser.add_argument("--tail-frequency", type=float, default=0.5, help="Tail sweep frequency in Hz.")
    parser.add_argument("--tail-amplitude", type=float, default=20.0, help="Tail sweep amplitude in degrees.")
    parser.add_argument("--command-hz", type=float, default=30.0, help="Servo command update rate.")
    parser.add_argument("--hold-pwm", action="store_true", help="Keep PWM active after the action.")
    parser.add_argument("--mock-sensors", action="store_true", help="Use mock sensor data; servos still move.")
    return parser.parse_args()


def main() -> int:
    ensure_not_windows_hardware_run()
    args = parse_args()
    _validate_args(args)

    config = load_robot_config()
    log_dir = create_run_log_dir(_resolve_log_base_dir(config), suffix=ACTION_NAME)
    prepare_raw_placeholders(log_dir)
    _write_action_metadata(config, args, log_dir)

    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    flush_interval_s = float(logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0)))
    queue_maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))

    sensor_manager = SensorManager(config, mock=args.mock_sensors, log_dir=log_dir)
    synchronizer = SensorSynchronizer(sensor_manager.buffers, config)
    sync_logger = JsonlLogger(
        log_dir / str(logging_cfg.get("synchronized_file", "synchronized_sensors.jsonl")),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    raw_loggers = RawSensorLoggers(log_dir, flush_interval_s=flush_interval_s, queue_maxsize=queue_maxsize)
    event_logger = EventLogger(
        log_dir / str(logging_cfg.get("events_file", "events.jsonl")),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    command_logger = JsonlLogger(
        log_dir / "commands.jsonl",
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    link_manager = link_manager_from_config(config, event_logger)

    controller = PCA9685ServoController(config)
    tail_items = [_servo_by_id(config, servo_id) for servo_id in TAIL_SERVO_IDS]
    root4 = _servo_by_id(config, 4)
    tip5 = _servo_by_id(config, 5)
    root6 = _servo_by_id(config, 6)
    tip7 = _servo_by_id(config, 7)

    tail_targets = _tail_targets(controller, tail_items, args.tail_amplitude)
    pectoral_targets = _pectoral_targets(controller, root4, tip5, root6, tip7, args.pectoral_root_scale)
    pectoral_channels = [int(values["channel"]) for values in pectoral_targets.values()]
    moving_channels = sorted(set(tail_targets) | set(pectoral_channels))
    release_after = bool(config.get("safety", {}).get("servo", {}).get("release_pwm_after_tests", True))
    release_after = release_after and not args.hold_pwm

    sync_logger.start()
    raw_loggers.start()
    event_logger.start()
    command_logger.start()
    sensor_manager.start_all()
    link_manager.start()

    event_logger.write(
        "action_started",
        {
            "action": ACTION_NAME,
            "duration_s": args.duration,
            "pectoral_frequency_hz": args.pectoral_frequency,
            "pectoral_root_scale": args.pectoral_root_scale,
            "tip_return_fraction": args.tip_return_fraction,
            "tail_frequency_hz": args.tail_frequency,
            "tail_amplitude_deg": args.tail_amplitude,
            "log_dir": str(log_dir),
        },
    )
    print(f"Action started. Logs: {log_dir}")
    print("Moving servos: pectoral roots 4/6, pectoral tips 5/7, tail ids 1/2/3.")

    try:
        _enter_initial_pose(controller, tail_targets, pectoral_targets)
        _run_action_loop(
            config,
            args,
            controller,
            tail_targets,
            pectoral_targets,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
        )
        event_logger.write("action_finished", {"action": ACTION_NAME, "reason": "duration_elapsed"})
    except KeyboardInterrupt:
        event_logger.write("ctrl_c_received", {"mode": "action", "action": ACTION_NAME})
        event_logger.write("action_finished", {"action": ACTION_NAME, "reason": "ctrl_c"})
        print("")
        print("Interrupted. Recentering commanded servos.")
    finally:
        link_manager.stop()
        raw_loggers.write_from_buffers(sensor_manager.buffers)
        _recenter_commanded_servos(controller, moving_channels)
        if release_after:
            sleep_safely(0.5)
            controller.stop_all(moving_channels)
        sensor_manager.stop_all()
        raw_loggers.write_from_buffers(sensor_manager.buffers)
        event_logger.write(
            "logger_stopped",
            {
                "mode": "action",
                "action": ACTION_NAME,
                "sync_dropped_count": sync_logger.dropped_count,
                "sync_last_error": sync_logger.last_error,
                "raw_logger_stats": raw_loggers.stats(),
                "command_dropped_count": command_logger.dropped_count,
                "command_last_error": command_logger.last_error,
            },
        )
        sync_logger.stop()
        raw_loggers.stop()
        command_logger.stop()
        event_logger.stop()
        print(f"Action stopped. Logs are in: {log_dir}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.confirm != "MOVE":
        raise SystemExit("Refusing to move servos without --confirm MOVE.")
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0.")
    if args.pectoral_frequency <= 0:
        raise SystemExit("--pectoral-frequency must be > 0.")
    if not 0.0 < args.pectoral_root_scale <= 1.0:
        raise SystemExit("--pectoral-root-scale must be in (0, 1].")
    if not 0.0 <= args.tip_return_fraction < 1.0:
        raise SystemExit("--tip-return-fraction must be in [0, 1).")
    if args.tail_frequency <= 0:
        raise SystemExit("--tail-frequency must be > 0.")
    if args.tail_amplitude < 0:
        raise SystemExit("--tail-amplitude must be >= 0.")
    if args.command_hz <= 0:
        raise SystemExit("--command-hz must be > 0.")


def _run_action_loop(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: PCA9685ServoController,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[str, dict[str, float]],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
) -> None:
    command_period_s = 1.0 / max(float(args.command_hz), 0.001)
    sample_hz = max(0.1, float(config.get("runtime", {}).get("synchronized_sample_hz", 30)))
    sample_period_s = 1.0 / sample_hz
    print_hz = max(0.0, float(config.get("runtime", {}).get("print_hz", 2)))
    print_period_s = 1.0 / print_hz if print_hz > 0 else None

    start_s = time.monotonic()
    next_command_s = start_s
    next_sample_s = start_s
    next_print_s = start_s

    while True:
        now_s = time.monotonic()
        elapsed_s = now_s - start_s
        if elapsed_s >= args.duration:
            break

        if now_s >= next_command_s:
            pectoral_state = _pectoral_commands(
                elapsed_s,
                args.pectoral_frequency,
                args.tip_return_fraction,
                pectoral_targets,
            )
            tail_offset = args.tail_amplitude * math.sin(
                (2.0 * math.pi * args.tail_frequency * elapsed_s) - (math.pi / 2.0)
            )
            commands = {
                channel: values["center"] + tail_offset
                for channel, values in tail_targets.items()
            }
            commands.update(pectoral_state["commands"])
            controller.write_angles(commands)
            command_logger.write(
                {
                    "t_ns": time.monotonic_ns(),
                    "action": ACTION_NAME,
                    "elapsed_s": elapsed_s,
                    "pectoral_phase": pectoral_state["phase"],
                    "pectoral_stroke": pectoral_state["stroke"],
                    "tail_offset_deg": tail_offset,
                    "commands_deg": {str(channel): angle for channel, angle in commands.items()},
                }
            )
            next_command_s += command_period_s
            if next_command_s < now_s - command_period_s:
                next_command_s = now_s + command_period_s

        if now_s >= next_sample_s:
            sample = synchronizer.build()
            raw_loggers.write_from_buffers(synchronizer.buffers)
            sync_logger.write(sample)
            if print_period_s is not None and now_s >= next_print_s:
                print(_summary(sample, elapsed_s))
                next_print_s = now_s + print_period_s
            next_sample_s += sample_period_s
            if next_sample_s < now_s - sample_period_s:
                next_sample_s = now_s + sample_period_s

        next_due_s = min(next_command_s, next_sample_s)
        if next_due_s > now_s:
            time.sleep(min(0.01, next_due_s - now_s))


def _pectoral_commands(
    elapsed_s: float,
    frequency_hz: float,
    tip_return_fraction: float,
    targets: dict[str, dict[str, float]],
) -> dict[str, Any]:
    phase = (elapsed_s * frequency_hz) % 1.0
    commands: dict[int, float] = {}

    if phase < 0.5:
        stroke = "downstroke"
        progress = phase / 0.5
        commands[targets["root4"]["channel"]] = _lerp(targets["root4"]["top"], targets["root4"]["bottom"], progress)
        commands[targets["root6"]["channel"]] = _lerp(targets["root6"]["top"], targets["root6"]["bottom"], progress)
        commands[targets["tip5"]["channel"]] = targets["tip5"]["center"]
        commands[targets["tip7"]["channel"]] = targets["tip7"]["center"]
    else:
        stroke = "upstroke"
        progress = (phase - 0.5) / 0.5
        commands[targets["root4"]["channel"]] = _lerp(targets["root4"]["bottom"], targets["root4"]["top"], progress)
        commands[targets["root6"]["channel"]] = _lerp(targets["root6"]["bottom"], targets["root6"]["top"], progress)
        if progress < tip_return_fraction:
            commands[targets["tip5"]["channel"]] = targets["tip5"]["upstroke_start"]
            commands[targets["tip7"]["channel"]] = targets["tip7"]["upstroke_start"]
        else:
            tip_progress = (progress - tip_return_fraction) / max(0.001, 1.0 - tip_return_fraction)
            commands[targets["tip5"]["channel"]] = _lerp(
                targets["tip5"]["upstroke_start"],
                targets["tip5"]["center"],
                tip_progress,
            )
            commands[targets["tip7"]["channel"]] = _lerp(
                targets["tip7"]["upstroke_start"],
                targets["tip7"]["center"],
                tip_progress,
            )

    return {"phase": phase, "stroke": stroke, "commands": commands}


def _enter_initial_pose(
    controller: PCA9685ServoController,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[int, dict[str, float]],
) -> None:
    for channel, values in tail_targets.items():
        controller.move_safely(channel, values["center"] - values["amplitude"])
    controller.move_safely(pectoral_targets["root4"]["channel"], pectoral_targets["root4"]["top"])
    controller.move_safely(pectoral_targets["root6"]["channel"], pectoral_targets["root6"]["top"])
    controller.move_safely(pectoral_targets["tip5"]["channel"], pectoral_targets["tip5"]["center"])
    controller.move_safely(pectoral_targets["tip7"]["channel"], pectoral_targets["tip7"]["center"])


def _recenter_commanded_servos(controller: PCA9685ServoController, channels: list[int]) -> None:
    for channel in channels:
        limits = controller.limits_for(channel)
        controller.move_safely(channel, limits.center_angle)


def _tail_targets(
    controller: PCA9685ServoController,
    tail_items: list[dict[str, Any]],
    amplitude: float,
) -> dict[int, dict[str, float]]:
    targets: dict[int, dict[str, float]] = {}
    for item in tail_items:
        channel = int(item["channel"])
        limits = controller.limits_for(channel)
        max_amplitude = min(limits.center_angle - limits.min_angle, limits.max_angle - limits.center_angle)
        if amplitude > max_amplitude:
            raise SystemExit(
                f"Tail amplitude {amplitude:.1f} exceeds servo_id={item.get('servo_id')} "
                f"safe symmetric amplitude {max_amplitude:.1f}."
            )
        targets[channel] = {"center": limits.center_angle, "amplitude": amplitude}
    return targets


def _pectoral_targets(
    controller: PCA9685ServoController,
    root4: dict[str, Any],
    tip5: dict[str, Any],
    root6: dict[str, Any],
    tip7: dict[str, Any],
    root_scale: float,
) -> dict[str, dict[str, float]]:
    channel4 = int(root4["channel"])
    channel5 = int(tip5["channel"])
    channel6 = int(root6["channel"])
    channel7 = int(tip7["channel"])
    limits4 = controller.limits_for(channel4)
    limits5 = controller.limits_for(channel5)
    limits6 = controller.limits_for(channel6)
    limits7 = controller.limits_for(channel7)

    root4_top = _direction_value(root4, "top_reference_angle", limits4.min_angle)
    root4_bottom_full = _direction_value(root4, "bottom_reference_angle", limits4.max_angle)
    root6_top = _direction_value(root6, "top_reference_angle", limits6.max_angle)
    root6_bottom_full = _direction_value(root6, "bottom_reference_angle", limits6.min_angle)
    root4_bottom = _lerp(root4_top, root4_bottom_full, root_scale)
    root6_bottom = _lerp(root6_top, root6_bottom_full, root_scale)

    return {
        "root4": {
            "channel": channel4,
            "top": limits4.validate(root4_top),
            "bottom": limits4.validate(root4_bottom),
            "center": limits4.center_angle,
        },
        "root6": {
            "channel": channel6,
            "top": limits6.validate(root6_top),
            "bottom": limits6.validate(root6_bottom),
            "center": limits6.center_angle,
        },
        "tip5": {
            "channel": channel5,
            "center": limits5.center_angle,
            "upstroke_start": limits5.validate(180.0),
        },
        "tip7": {
            "channel": channel7,
            "center": limits7.center_angle,
            "upstroke_start": limits7.validate(0.0),
        },
    }


def _direction_value(item: dict[str, Any], key: str, fallback: float) -> float:
    direction = item.get("direction", {})
    if isinstance(direction, dict) and key in direction:
        return float(direction[key])
    return float(fallback)


def _servo_by_id(config: dict[str, Any], servo_id: int) -> dict[str, Any]:
    for item in configured_servo_channels(config):
        if int(item.get("servo_id", -1)) == servo_id:
            return item
    raise SystemExit(f"Missing servo_id={servo_id} in config/robot.yaml")


def _resolve_log_base_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    base_dir = Path(str(logging_cfg.get("base_dir", runtime_cfg.get("log_dir", "logs"))))
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    return base_dir


def _write_action_metadata(config: dict[str, Any], args: argparse.Namespace, log_dir: Path) -> None:
    logging_cfg = config.get("logging", {})
    metadata_file = str(logging_cfg.get("metadata_file", "metadata.yaml"))
    metadata = {
        "started_wall_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "action",
        "action": ACTION_NAME,
        "duration_s": args.duration,
        "tail_servo_ids": list(TAIL_SERVO_IDS),
        "pectoral_root_servo_ids": list(PECTORAL_ROOT_SERVO_IDS),
        "pectoral_tip_servo_ids": list(PECTORAL_TIP_SERVO_IDS),
        "pectoral_frequency_hz": args.pectoral_frequency,
        "pectoral_root_scale": args.pectoral_root_scale,
        "tip_return_fraction": args.tip_return_fraction,
        "tail_frequency_hz": args.tail_frequency,
        "tail_amplitude_deg": args.tail_amplitude,
        "command_hz": args.command_hz,
        "mock_sensors": bool(args.mock_sensors),
        "project_root": str(PROJECT_ROOT),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "servo": config.get("servo", {}),
        "notes": [
            "Action 2 assumes the repeated user text means servo 7 starts upstroke at 0 degrees.",
            "At top: servo 4/6 are at top references and servo 5/7 are 90 degrees.",
            "Downstroke: servo 4/6 move toward bottom while servo 5/7 stay at 90 degrees.",
            "Upstroke: servo 4/6 return upward; servo 5 starts at 180 and servo 7 starts at 0.",
            "Near top, servo 5/7 return to 90 degrees.",
            "Tail servos 1/2/3 share the same sinusoidal phase as action 1.",
            "Sensors are recorded during the action.",
        ],
    }
    write_metadata_yaml(log_dir / metadata_file, metadata)


def _summary(sample: dict[str, Any], elapsed_s: float) -> str:
    status = sample.get("status", {})
    depth = sample.get("depth", {})
    vision = sample.get("vision", {})
    depth_text = "-"
    if depth.get("depth_m") is not None:
        depth_text = f"{depth.get('depth_m'):.3f}m"
    return (
        f"t={elapsed_s:.1f}s "
        f"imu={'ok' if sample.get('imu', {}).get('valid') else 'bad'} "
        f"depth={depth_text} "
        f"power={'ok' if sample.get('power', {}).get('valid') else 'bad'} "
        f"uwb={'ok' if sample.get('uwb', {}).get('valid') else sample.get('uwb', {}).get('error', 'bad')} "
        f"vision={'ok' if vision.get('valid') else vision.get('error', 'bad')} "
        f"missing={status.get('missing_sensors', [])}"
    )


def _lerp(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return float(start) + (float(end) - float(start)) * progress


if __name__ == "__main__":
    raise SystemExit(main())
