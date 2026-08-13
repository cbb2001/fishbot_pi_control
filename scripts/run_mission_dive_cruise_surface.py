from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
    servo_limits_from_config,
    sleep_safely,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


MISSION_NAME = "mission_dive_cruise_surface"
TAIL_SERVO_IDS = (1, 2, 3)
PECTORAL_ROOT_SERVO_IDS = (4, 6)
PECTORAL_TIP_SERVO_IDS = (5, 7)
ALL_MISSION_SERVO_IDS = TAIL_SERVO_IDS + PECTORAL_ROOT_SERVO_IDS + PECTORAL_TIP_SERVO_IDS

CommandBuilder = Callable[[float, float], tuple[dict[int, float], dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open-loop mission: dive with action-1 style pectoral attack angle, "
            "cruise with action-2 style pectoral flapping, then surface with the "
            "reverse attack angle. Sensors are recorded continuously."
        )
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command real servos.")
    parser.add_argument("--dry-run", action="store_true", help="Do not initialize PCA9685 or move servos.")
    parser.add_argument("--dive-duration", type=float, default=10.0, help="Dive phase duration in seconds.")
    parser.add_argument("--cruise-duration", type=float, default=20.0, help="Underwater cruise duration in seconds.")
    parser.add_argument("--surface-duration", type=float, default=10.0, help="Surface phase duration in seconds.")
    parser.add_argument("--dive-pectoral-tilt", type=float, default=20.0, help="Dive pectoral tip tilt in degrees.")
    parser.add_argument(
        "--surface-pectoral-tilt",
        type=float,
        default=20.0,
        help="Surface pectoral tip tilt in degrees; applied with the reverse sign of dive.",
    )
    parser.add_argument("--tail-frequency", type=float, default=0.5, help="Tail sweep frequency in Hz.")
    parser.add_argument("--tail-amplitude", type=float, default=20.0, help="Tail sweep amplitude in degrees.")
    parser.add_argument("--pectoral-frequency", type=float, default=0.5, help="Cruise pectoral flap frequency in Hz.")
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
    parser.add_argument("--command-hz", type=float, default=30.0, help="Servo command update rate.")
    parser.add_argument("--transition-s", type=float, default=1.0, help="Smooth pectoral transition time between phases.")
    parser.add_argument("--hold-pwm", action="store_true", help="Keep PWM active after the mission.")
    parser.add_argument("--mock-sensors", action="store_true", help="Use mock sensor data; real servos still move.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()
    _validate_args(args)

    config = load_robot_config()
    log_dir = create_run_log_dir(_resolve_log_base_dir(config), suffix=MISSION_NAME)
    prepare_raw_placeholders(log_dir)
    _write_mission_metadata(config, args, log_dir)

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

    controller: Any
    if args.dry_run:
        controller = DryRunServoController(config)
    else:
        controller = PCA9685ServoController(config)

    tail_items = [_servo_by_id(config, servo_id) for servo_id in TAIL_SERVO_IDS]
    root4 = _servo_by_id(config, 4)
    tip5 = _servo_by_id(config, 5)
    root6 = _servo_by_id(config, 6)
    tip7 = _servo_by_id(config, 7)

    tail_targets = _tail_targets(controller, tail_items, args.tail_amplitude)
    pectoral_targets = _pectoral_targets(controller, root4, tip5, root6, tip7, args.pectoral_root_scale)
    moving_channels = sorted(
        {
            int(channel)
            for channel in (
                list(tail_targets)
                + [int(values["channel"]) for values in pectoral_targets.values()]
            )
        }
    )
    pectoral_channels = sorted(int(values["channel"]) for values in pectoral_targets.values())
    release_after = bool(config.get("safety", {}).get("servo", {}).get("release_pwm_after_tests", True))
    release_after = release_after and not args.hold_pwm and not args.dry_run

    sync_logger.start()
    raw_loggers.start()
    event_logger.start()
    command_logger.start()
    sensor_manager.start_all()
    link_manager.start()

    event_logger.write(
        "mission_started",
        {
            "mission": MISSION_NAME,
            "dry_run": bool(args.dry_run),
            "dive_duration_s": args.dive_duration,
            "cruise_duration_s": args.cruise_duration,
            "surface_duration_s": args.surface_duration,
            "transition_s": args.transition_s,
            "log_dir": str(log_dir),
        },
    )
    print(f"Mission started. Logs: {log_dir}")
    if args.dry_run:
        print("Dry run: PCA9685 is not initialized and no servo PWM is written.")
    else:
        print("Moving servos: tail ids 1/2/3, pectoral roots 4/6, pectoral tips 5/7.")

    mission_reason = "duration_elapsed"
    try:
        mission_start_s = time.monotonic()
        initial_commands, _ = _fixed_tip_phase_commands(
            0.0,
            0.0,
            signed_tip_tilt=args.dive_pectoral_tilt,
            tail_targets=tail_targets,
            pectoral_targets=pectoral_targets,
            tail_frequency_hz=args.tail_frequency,
            tail_amplitude_deg=args.tail_amplitude,
        )
        _enter_initial_pose(controller, initial_commands)

        last_commands = _run_segment(
            config=config,
            args=args,
            controller=controller,
            segment_name="dive",
            segment_kind="phase",
            duration_s=args.dive_duration,
            mission_start_s=mission_start_s,
            command_builder=lambda phase_elapsed_s, mission_elapsed_s: _fixed_tip_phase_commands(
                phase_elapsed_s,
                mission_elapsed_s,
                signed_tip_tilt=args.dive_pectoral_tilt,
                tail_targets=tail_targets,
                pectoral_targets=pectoral_targets,
                tail_frequency_hz=args.tail_frequency,
                tail_amplitude_deg=args.tail_amplitude,
            ),
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
        )
        last_commands = _run_transition_if_needed(
            config,
            args,
            controller,
            "transition_dive_to_cruise",
            last_commands,
            _cruise_phase_commands(
                0.0,
                time.monotonic() - mission_start_s,
                tail_targets=tail_targets,
                pectoral_targets=pectoral_targets,
                tail_frequency_hz=args.tail_frequency,
                tail_amplitude_deg=args.tail_amplitude,
                pectoral_frequency_hz=args.pectoral_frequency,
                tip_return_fraction=args.tip_return_fraction,
            )[0],
            pectoral_channels,
            tail_targets,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
            event_logger,
            mission_start_s,
        )
        last_commands = _run_segment(
            config=config,
            args=args,
            controller=controller,
            segment_name="cruise_underwater",
            segment_kind="phase",
            duration_s=args.cruise_duration,
            mission_start_s=mission_start_s,
            command_builder=lambda phase_elapsed_s, mission_elapsed_s: _cruise_phase_commands(
                phase_elapsed_s,
                mission_elapsed_s,
                tail_targets=tail_targets,
                pectoral_targets=pectoral_targets,
                tail_frequency_hz=args.tail_frequency,
                tail_amplitude_deg=args.tail_amplitude,
                pectoral_frequency_hz=args.pectoral_frequency,
                tip_return_fraction=args.tip_return_fraction,
            ),
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
        )
        last_commands = _run_transition_if_needed(
            config,
            args,
            controller,
            "transition_cruise_to_surface",
            last_commands,
            _fixed_tip_phase_commands(
                0.0,
                time.monotonic() - mission_start_s,
                signed_tip_tilt=-args.surface_pectoral_tilt,
                tail_targets=tail_targets,
                pectoral_targets=pectoral_targets,
                tail_frequency_hz=args.tail_frequency,
                tail_amplitude_deg=args.tail_amplitude,
            )[0],
            pectoral_channels,
            tail_targets,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
            event_logger,
            mission_start_s,
        )
        _run_segment(
            config=config,
            args=args,
            controller=controller,
            segment_name="surface",
            segment_kind="phase",
            duration_s=args.surface_duration,
            mission_start_s=mission_start_s,
            command_builder=lambda phase_elapsed_s, mission_elapsed_s: _fixed_tip_phase_commands(
                phase_elapsed_s,
                mission_elapsed_s,
                signed_tip_tilt=-args.surface_pectoral_tilt,
                tail_targets=tail_targets,
                pectoral_targets=pectoral_targets,
                tail_frequency_hz=args.tail_frequency,
                tail_amplitude_deg=args.tail_amplitude,
            ),
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
        )
        event_logger.write("mission_finished", {"mission": MISSION_NAME, "reason": mission_reason})
    except KeyboardInterrupt:
        mission_reason = "ctrl_c"
        event_logger.write("ctrl_c_received", {"mode": "mission", "mission": MISSION_NAME})
        event_logger.write("mission_finished", {"mission": MISSION_NAME, "reason": mission_reason})
        print("")
        print("Interrupted. Recentering mission servos.")
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
                "mode": "mission",
                "mission": MISSION_NAME,
                "reason": mission_reason,
                "dry_run": bool(args.dry_run),
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
        print(f"Mission stopped. Logs are in: {log_dir}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if not args.dry_run and args.confirm != "MOVE":
        raise SystemExit("Refusing to move servos without --confirm MOVE. Use --dry-run for no-PWM validation.")
    for name in ("dive_duration", "cruise_duration", "surface_duration"):
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0.")
    if args.dive_pectoral_tilt < 0:
        raise SystemExit("--dive-pectoral-tilt must be >= 0.")
    if args.surface_pectoral_tilt < 0:
        raise SystemExit("--surface-pectoral-tilt must be >= 0.")
    if args.tail_frequency <= 0:
        raise SystemExit("--tail-frequency must be > 0.")
    if args.tail_amplitude < 0:
        raise SystemExit("--tail-amplitude must be >= 0.")
    if args.pectoral_frequency <= 0:
        raise SystemExit("--pectoral-frequency must be > 0.")
    if not 0.0 < args.pectoral_root_scale <= 1.0:
        raise SystemExit("--pectoral-root-scale must be in (0, 1].")
    if not 0.0 <= args.tip_return_fraction < 1.0:
        raise SystemExit("--tip-return-fraction must be in [0, 1).")
    if args.command_hz <= 0:
        raise SystemExit("--command-hz must be > 0.")
    if args.transition_s < 0:
        raise SystemExit("--transition-s must be >= 0.")


def _run_segment(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
    segment_name: str,
    segment_kind: str,
    duration_s: float,
    mission_start_s: float,
    command_builder: CommandBuilder,
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
) -> dict[int, float]:
    event_logger.write(
        "phase_started",
        {
            "mission": MISSION_NAME,
            "phase": segment_name,
            "segment_kind": segment_kind,
            "duration_s": duration_s,
        },
    )
    print(f"Phase started: {segment_name} ({duration_s:.1f}s)")

    command_period_s = 1.0 / max(float(args.command_hz), 0.001)
    sample_hz = max(0.1, float(config.get("runtime", {}).get("synchronized_sample_hz", 30)))
    sample_period_s = 1.0 / sample_hz
    print_hz = max(0.0, float(config.get("runtime", {}).get("print_hz", 2)))
    print_period_s = 1.0 / print_hz if print_hz > 0 else None

    phase_start_s = time.monotonic()
    next_command_s = phase_start_s
    next_sample_s = phase_start_s
    next_print_s = phase_start_s
    last_commands: dict[int, float] = {}

    while True:
        now_s = time.monotonic()
        phase_elapsed_s = now_s - phase_start_s
        mission_elapsed_s = now_s - mission_start_s
        if phase_elapsed_s >= duration_s:
            break

        if now_s >= next_command_s:
            commands, state = command_builder(phase_elapsed_s, mission_elapsed_s)
            controller.write_angles(commands)
            last_commands = dict(commands)
            _write_command(
                command_logger,
                segment_name,
                segment_kind,
                phase_elapsed_s,
                mission_elapsed_s,
                commands,
                state,
                dry_run=bool(args.dry_run),
            )
            next_command_s += command_period_s
            if next_command_s < now_s - command_period_s:
                next_command_s = now_s + command_period_s

        if now_s >= next_sample_s:
            sample = synchronizer.build()
            raw_loggers.write_from_buffers(synchronizer.buffers)
            sync_logger.write(sample)
            if print_period_s is not None and now_s >= next_print_s:
                print(_summary(sample, mission_elapsed_s, segment_name))
                next_print_s = now_s + print_period_s
            next_sample_s += sample_period_s
            if next_sample_s < now_s - sample_period_s:
                next_sample_s = now_s + sample_period_s

        next_due_s = min(next_command_s, next_sample_s)
        if next_due_s > now_s:
            time.sleep(min(0.01, next_due_s - now_s))

    event_logger.write(
        "phase_finished",
        {
            "mission": MISSION_NAME,
            "phase": segment_name,
            "segment_kind": segment_kind,
            "reason": "duration_elapsed",
        },
    )
    return last_commands


def _run_transition_if_needed(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
    segment_name: str,
    start_commands: dict[int, float],
    end_commands: dict[int, float],
    pectoral_channels: list[int],
    tail_targets: dict[int, dict[str, float]],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
    mission_start_s: float,
) -> dict[int, float]:
    if args.transition_s <= 0:
        return start_commands

    start_pectoral = {
        channel: start_commands[channel]
        for channel in pectoral_channels
        if channel in start_commands
    }
    end_pectoral = {
        channel: end_commands[channel]
        for channel in pectoral_channels
        if channel in end_commands
    }

    def command_builder(phase_elapsed_s: float, mission_elapsed_s: float) -> tuple[dict[int, float], dict[str, Any]]:
        progress = phase_elapsed_s / max(args.transition_s, 0.001)
        commands = _tail_commands(
            mission_elapsed_s,
            tail_targets,
            args.tail_frequency,
            args.tail_amplitude,
        )
        for channel, start_angle in start_pectoral.items():
            end_angle = end_pectoral.get(channel, start_angle)
            commands[channel] = _lerp(start_angle, end_angle, progress)
        return (
            commands,
            {
                "transition_progress": max(0.0, min(1.0, progress)),
                "tail_offset_deg": _tail_offset(mission_elapsed_s, args.tail_frequency, args.tail_amplitude),
            },
        )

    return _run_segment(
        config=config,
        args=args,
        controller=controller,
        segment_name=segment_name,
        segment_kind="transition",
        duration_s=args.transition_s,
        mission_start_s=mission_start_s,
        command_builder=command_builder,
        synchronizer=synchronizer,
        sync_logger=sync_logger,
        raw_loggers=raw_loggers,
        command_logger=command_logger,
        event_logger=event_logger,
    )


def _fixed_tip_phase_commands(
    phase_elapsed_s: float,
    mission_elapsed_s: float,
    *,
    signed_tip_tilt: float,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[str, dict[str, float]],
    tail_frequency_hz: float,
    tail_amplitude_deg: float,
) -> tuple[dict[int, float], dict[str, Any]]:
    del phase_elapsed_s
    commands = _tail_commands(mission_elapsed_s, tail_targets, tail_frequency_hz, tail_amplitude_deg)
    commands[pectoral_targets["root4"]["channel"]] = pectoral_targets["root4"]["center"]
    commands[pectoral_targets["root6"]["channel"]] = pectoral_targets["root6"]["center"]
    commands[pectoral_targets["tip5"]["channel"]] = pectoral_targets["tip5"]["limits"].validate(
        pectoral_targets["tip5"]["center"] + signed_tip_tilt
    )
    commands[pectoral_targets["tip7"]["channel"]] = pectoral_targets["tip7"]["limits"].validate(
        pectoral_targets["tip7"]["center"] - signed_tip_tilt
    )
    return (
        commands,
        {
            "pectoral_mode": "fixed_tip_attack_angle",
            "signed_tip_tilt_deg": signed_tip_tilt,
            "tail_offset_deg": _tail_offset(mission_elapsed_s, tail_frequency_hz, tail_amplitude_deg),
        },
    )


def _cruise_phase_commands(
    phase_elapsed_s: float,
    mission_elapsed_s: float,
    *,
    tail_targets: dict[int, dict[str, float]],
    pectoral_targets: dict[str, dict[str, float]],
    tail_frequency_hz: float,
    tail_amplitude_deg: float,
    pectoral_frequency_hz: float,
    tip_return_fraction: float,
) -> tuple[dict[int, float], dict[str, Any]]:
    pectoral_state = _pectoral_flap_commands(
        phase_elapsed_s,
        pectoral_frequency_hz,
        tip_return_fraction,
        pectoral_targets,
    )
    commands = _tail_commands(mission_elapsed_s, tail_targets, tail_frequency_hz, tail_amplitude_deg)
    commands.update(pectoral_state["commands"])
    return (
        commands,
        {
            "pectoral_mode": "root_flap_tip_switch",
            "pectoral_phase": pectoral_state["phase"],
            "pectoral_stroke": pectoral_state["stroke"],
            "tail_offset_deg": _tail_offset(mission_elapsed_s, tail_frequency_hz, tail_amplitude_deg),
        },
    )


def _pectoral_flap_commands(
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


def _tail_commands(
    mission_elapsed_s: float,
    tail_targets: dict[int, dict[str, float]],
    tail_frequency_hz: float,
    tail_amplitude_deg: float,
) -> dict[int, float]:
    offset = _tail_offset(mission_elapsed_s, tail_frequency_hz, tail_amplitude_deg)
    return {
        channel: values["center"] + offset
        for channel, values in tail_targets.items()
    }


def _tail_offset(mission_elapsed_s: float, tail_frequency_hz: float, tail_amplitude_deg: float) -> float:
    return tail_amplitude_deg * math.sin((2.0 * math.pi * tail_frequency_hz * mission_elapsed_s) - (math.pi / 2.0))


def _enter_initial_pose(controller: Any, commands: dict[int, float]) -> None:
    for channel in sorted(commands):
        controller.move_safely(channel, commands[channel])


def _recenter_commanded_servos(controller: Any, channels: list[int]) -> None:
    for channel in channels:
        limits = controller.limits_for(channel)
        controller.move_safely(channel, limits.center_angle)


def _tail_targets(
    controller: Any,
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
    controller: Any,
    root4: dict[str, Any],
    tip5: dict[str, Any],
    root6: dict[str, Any],
    tip7: dict[str, Any],
    root_scale: float,
) -> dict[str, dict[str, Any]]:
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
            "limits": limits4,
        },
        "root6": {
            "channel": channel6,
            "top": limits6.validate(root6_top),
            "bottom": limits6.validate(root6_bottom),
            "center": limits6.center_angle,
            "limits": limits6,
        },
        "tip5": {
            "channel": channel5,
            "center": limits5.center_angle,
            "upstroke_start": limits5.validate(180.0),
            "limits": limits5,
        },
        "tip7": {
            "channel": channel7,
            "center": limits7.center_angle,
            "upstroke_start": limits7.validate(0.0),
            "limits": limits7,
        },
    }


def _write_command(
    command_logger: JsonlLogger,
    segment_name: str,
    segment_kind: str,
    phase_elapsed_s: float,
    mission_elapsed_s: float,
    commands: dict[int, float],
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    command_logger.write(
        {
            "t_ns": time.monotonic_ns(),
            "mission": MISSION_NAME,
            "phase": segment_name,
            "segment_kind": segment_kind,
            "phase_elapsed_s": phase_elapsed_s,
            "mission_elapsed_s": mission_elapsed_s,
            "dry_run": dry_run,
            "tail_offset_deg": state.get("tail_offset_deg"),
            "pectoral_mode": state.get("pectoral_mode"),
            "pectoral_phase": state.get("pectoral_phase"),
            "pectoral_stroke": state.get("pectoral_stroke"),
            "signed_tip_tilt_deg": state.get("signed_tip_tilt_deg"),
            "transition_progress": state.get("transition_progress"),
            "commands_deg": {str(channel): angle for channel, angle in commands.items()},
        }
    )


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


def _write_mission_metadata(config: dict[str, Any], args: argparse.Namespace, log_dir: Path) -> None:
    logging_cfg = config.get("logging", {})
    metadata_file = str(logging_cfg.get("metadata_file", "metadata.yaml"))
    metadata = {
        "started_wall_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "mission",
        "mission": MISSION_NAME,
        "dry_run": bool(args.dry_run),
        "servo_ids": list(ALL_MISSION_SERVO_IDS),
        "phase_order": ["dive", "cruise_underwater", "surface"],
        "dive_duration_s": args.dive_duration,
        "cruise_duration_s": args.cruise_duration,
        "surface_duration_s": args.surface_duration,
        "transition_s": args.transition_s,
        "dive_pectoral_tilt_deg": args.dive_pectoral_tilt,
        "surface_pectoral_tilt_deg": args.surface_pectoral_tilt,
        "tail_frequency_hz": args.tail_frequency,
        "tail_amplitude_deg": args.tail_amplitude,
        "pectoral_frequency_hz": args.pectoral_frequency,
        "pectoral_root_scale": args.pectoral_root_scale,
        "tip_return_fraction": args.tip_return_fraction,
        "command_hz": args.command_hz,
        "mock_sensors": bool(args.mock_sensors),
        "project_root": str(PROJECT_ROOT),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "servo": config.get("servo", {}),
        "notes": [
            "This is an open-loop mission script, not closed-loop depth control.",
            "Dive uses servo 5 increasing and servo 7 decreasing for positive pectoral tip attack angle.",
            "Surface uses the reverse pectoral tip attack angle.",
            "Servos 4/6 are held at center during dive and surface to avoid carrying over cruise root flap positions.",
            "Cruise uses action-2 style root flapping and tip switching.",
            "Tail servos 1/2/3 share one sinusoidal phase across the whole mission.",
            "Transitions are added between main phases and keep sensor logging continuous.",
        ],
    }
    write_metadata_yaml(log_dir / metadata_file, metadata)


def _summary(sample: dict[str, Any], mission_elapsed_s: float, phase: str) -> str:
    status = sample.get("status", {})
    depth = sample.get("depth", {})
    vision = sample.get("vision", {})
    depth_text = "-"
    if depth.get("depth_m") is not None:
        depth_text = f"{depth.get('depth_m'):.3f}m"
    return (
        f"t={mission_elapsed_s:.1f}s "
        f"phase={phase} "
        f"imu={'ok' if sample.get('imu', {}).get('valid') else 'bad'} "
        f"depth={depth_text} "
        f"power={'ok' if sample.get('power', {}).get('valid') else 'bad'} "
        f"uwb={'ok' if sample.get('uwb', {}).get('valid') else sample.get('uwb', {}).get('error', 'bad')} "
        f"vision={'ok' if vision.get('valid') else vision.get('error', 'bad')} "
        f"missing={status.get('missing_sensors', [])}"
    )


def _direction_value(item: dict[str, Any], key: str, fallback: float) -> float:
    direction = item.get("direction", {})
    if isinstance(direction, dict) and key in direction:
        return float(direction[key])
    return float(fallback)


def _lerp(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return float(start) + (float(end) - float(start)) * progress


class DryRunServoController:
    """PCA9685-compatible controller that validates commands without touching hardware."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._raw_by_channel: dict[int, dict[str, Any]] = {}
        self._last_angles: dict[int, float] = {}
        for raw in configured_servo_channels(config):
            channel = int(raw.get("channel", 0))
            self._raw_by_channel[channel] = raw

    def limits_for(self, channel: int) -> Any:
        channel = int(channel)
        raw = self._raw_by_channel.get(channel)
        if raw is None:
            raise ValueError(f"Servo channel {channel} is not configured.")
        return servo_limits_from_config(self._config, raw)

    def move_safely(self, channel: int, angle: float) -> None:
        limits = self.limits_for(channel)
        self._last_angles[int(channel)] = limits.validate(float(angle))

    def write_angles(self, commands: dict[int, float]) -> None:
        targets: dict[int, float] = {}
        for channel, angle in commands.items():
            limits = self.limits_for(int(channel))
            targets[int(channel)] = limits.validate(float(angle))
        self._last_angles.update(targets)

    def stop_all(self, channels: list[int] | None = None) -> None:
        if channels is None:
            self._last_angles.clear()
            return
        for channel in channels:
            self._last_angles.pop(int(channel), None)


if __name__ == "__main__":
    raise SystemExit(main())
