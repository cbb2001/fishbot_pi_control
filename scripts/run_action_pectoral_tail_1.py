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


ACTION_NAME = "pectoral_tip_mirror_tail_sweep_1"
TAIL_SERVO_IDS = (1, 2, 3)
PECTORAL_TIP_SERVO_IDS = (5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Action 1: servo 5/7 mirror tilt while tail servos 1/2/3 sweep "
            "periodically from right to left. Sensors are recorded during the action."
        )
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command servos.")
    parser.add_argument("--duration", type=float, default=10.0, help="Action duration in seconds.")
    parser.add_argument("--tail-frequency", type=float, default=0.5, help="Tail sweep frequency in Hz.")
    parser.add_argument("--tail-amplitude", type=float, default=20.0, help="Tail sweep amplitude in degrees.")
    parser.add_argument("--pectoral-tilt", type=float, default=30.0, help="Servo 5/7 mirror tilt in degrees.")
    parser.add_argument("--command-hz", type=float, default=30.0, help="Servo command update rate.")
    parser.add_argument("--hold-pwm", action="store_true", help="Keep PWM active after the action.")
    parser.add_argument("--mock-sensors", action="store_true", help="Use mock sensor data; servos still move.")
    return parser.parse_args()


def main() -> int:
    ensure_not_windows_hardware_run()
    args = parse_args()
    if args.confirm != "MOVE":
        raise SystemExit("Refusing to move servos without --confirm MOVE.")
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0.")
    if args.tail_frequency <= 0:
        raise SystemExit("--tail-frequency must be > 0.")
    if args.tail_amplitude < 0:
        raise SystemExit("--tail-amplitude must be >= 0.")
    if args.pectoral_tilt < 0:
        raise SystemExit("--pectoral-tilt must be >= 0.")
    if args.command_hz <= 0:
        raise SystemExit("--command-hz must be > 0.")

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
    pectoral5 = _servo_by_id(config, 5)
    pectoral7 = _servo_by_id(config, 7)
    moving_channels = [int(item["channel"]) for item in tail_items + [pectoral5, pectoral7]]
    tail_targets = _tail_targets(controller, tail_items, args.tail_amplitude)
    pectoral_targets = _pectoral_targets(controller, pectoral5, pectoral7, args.pectoral_tilt)
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
            "tail_frequency_hz": args.tail_frequency,
            "tail_amplitude_deg": args.tail_amplitude,
            "pectoral_tilt_deg": args.pectoral_tilt,
            "log_dir": str(log_dir),
        },
    )
    print(f"Action started. Logs: {log_dir}")
    print("Moving servos: tail ids 1/2/3, pectoral tip ids 5/7. Servos 4 and 6 are not commanded.")

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
        _recenter_commanded_servos(controller, tail_targets, pectoral_targets)
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


def _run_action_loop(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: PCA9685ServoController,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[int, float],
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
            offset = args.tail_amplitude * math.sin((2.0 * math.pi * args.tail_frequency * elapsed_s) - (math.pi / 2.0))
            commands = {
                channel: values["center"] + offset
                for channel, values in tail_targets.items()
            }
            commands.update(pectoral_targets)
            controller.write_angles(commands)
            command_logger.write(
                {
                    "t_ns": time.monotonic_ns(),
                    "action": ACTION_NAME,
                    "elapsed_s": elapsed_s,
                    "tail_offset_deg": offset,
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


def _enter_initial_pose(
    controller: PCA9685ServoController,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[int, float],
) -> None:
    for channel, values in tail_targets.items():
        controller.move_safely(channel, values["center"] - values["amplitude"])
    for channel, angle in pectoral_targets.items():
        controller.move_safely(channel, angle)


def _recenter_commanded_servos(
    controller: PCA9685ServoController,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[int, float],
) -> None:
    for channel, values in tail_targets.items():
        controller.move_safely(channel, values["center"])
    for channel in pectoral_targets:
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
    pectoral5: dict[str, Any],
    pectoral7: dict[str, Any],
    tilt: float,
) -> dict[int, float]:
    channel5 = int(pectoral5["channel"])
    channel7 = int(pectoral7["channel"])
    limits5 = controller.limits_for(channel5)
    limits7 = controller.limits_for(channel7)
    return {
        channel5: limits5.validate(limits5.center_angle + tilt),
        channel7: limits7.validate(limits7.center_angle - tilt),
    }


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
        "pectoral_tip_servo_ids": list(PECTORAL_TIP_SERVO_IDS),
        "fixed_pectoral_root_servo_ids": [4, 6],
        "tail_frequency_hz": args.tail_frequency,
        "tail_amplitude_deg": args.tail_amplitude,
        "pectoral_tilt_deg": args.pectoral_tilt,
        "command_hz": args.command_hz,
        "mock_sensors": bool(args.mock_sensors),
        "project_root": str(PROJECT_ROOT),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "servo": config.get("servo", {}),
        "notes": [
            "Action 1 commands servo 5 increasing and servo 7 decreasing for mirror pectoral tip tilt.",
            "Servo 4 and servo 6 are not commanded by this script.",
            "Tail servos 1/2/3 share one sinusoidal phase and start from the right side.",
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


if __name__ == "__main__":
    raise SystemExit(main())

