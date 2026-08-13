from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()

from control.runtime.action_utils import (  # noqa: E402
    DryRunServoController,
    action_event,
    asymmetric_tail_offset,
    cosine_cycle_position,
    countdown_start_delay,
    cycle_state,
    interpolate_angles,
    safe_recenter_profile,
    tip_envelope,
    validate_finite_range,
)
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


ACTION_NAME = "smooth_pectoral_tail_20260713"
TAIL_SERVO_IDS = (1, 2, 3)
ROOT_SERVO_IDS = (4, 6)
TIP_SERVO_IDS = (5, 7)
ALL_SERVO_IDS = TAIL_SERVO_IDS + ROOT_SERVO_IDS + TIP_SERVO_IDS

CommandBuilder = Callable[[float], tuple[dict[int, float], dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smooth left/right pectoral test with one shared pectoral clock, "
            "quintic tip transitions, and an asymmetric sinusoidal tail waveform."
        )
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command real servos.")
    parser.add_argument("--dry-run", action="store_true", help="Do not initialize PCA9685 or write PWM.")
    parser.add_argument("--mock-sensors", action="store_true", help="Use mock sensors; real servos still move unless --dry-run is set.")
    parser.add_argument("--keep-pwm", action="store_true", help="Keep PWM active after safe recentering.")
    parser.add_argument("--duration", type=float, default=10.0, help="Periodic action duration in seconds.")
    parser.add_argument("--pectoral-frequency-hz", type=float, default=0.2, help="Shared pectoral frequency in Hz.")
    parser.add_argument("--root-amplitude-ratio", type=float, default=0.5, help="Fraction of each root servo's calibrated bottom-to-top travel, 0..1.")
    parser.add_argument("--tip-amplitude-deg", type=float, default=20.0, help="Opposite tip deflection magnitude in degrees.")
    parser.add_argument("--tip-transition-s", type=float, default=0.8, help="Quintic tip transition duration in seconds.")
    parser.add_argument("--tail-frequency-hz", type=float, default=0.3, help="Shared tail frequency in Hz.")
    parser.add_argument("--tail-left-amplitude-deg", type=float, default=10.0, help="Logical left tail amplitude in degrees.")
    parser.add_argument("--tail-right-amplitude-deg", type=float, default=10.0, help="Logical right tail amplitude in degrees.")
    parser.add_argument("--tail-first-direction", choices=("left", "right"), default="left", help="Direction of the first tail excursion from center.")
    parser.add_argument("--start-delay-s", type=float, default=20.0, help="Local pre-water countdown before logs, sensors, and servo control start.")
    parser.add_argument("--initial-hold-s", type=float, default=1.0, help="Baseline hold after entering the phase-zero pose.")
    parser.add_argument("--return-to-center-s", type=float, default=2.0, help="Quintic fade/recenter duration after the action.")
    parser.add_argument("--command-hz", type=float, default=50.0, help="Servo target update rate in Hz.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()
    _validate_scalar_args(args)

    config = load_robot_config()
    targets = _prepare_targets(config, args)
    if args.dry_run:
        _print_profile_checkpoints(args, targets)

    # Deliberately before log directory creation, sensors, network checks, or PCA9685.
    countdown_start_delay(args.start_delay_s)

    log_dir = create_run_log_dir(_resolve_log_base_dir(config), suffix=ACTION_NAME)
    prepare_raw_placeholders(log_dir)
    _write_metadata(config, args, targets, log_dir)

    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    flush_interval_s = float(logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0)))
    queue_maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))
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
    sensor_manager = SensorManager(config, mock=args.mock_sensors, log_dir=log_dir)
    synchronizer = SensorSynchronizer(sensor_manager.buffers, config)
    link_manager = link_manager_from_config(config, event_logger)

    sync_logger.start()
    raw_loggers.start()
    event_logger.start()
    command_logger.start()
    sensor_manager.start_all()
    link_manager.start()

    controller: Any | None = None
    centered = False
    reason = "duration_elapsed"
    last_commands = dict(targets["centers"])
    experiment_start_s = time.monotonic()
    release_after = bool(config.get("safety", {}).get("servo", {}).get("release_pwm_after_tests", True))
    release_after = release_after and not args.keep_pwm and not args.dry_run

    print(f"Logging and sensor acquisition started: {log_dir}")
    print("commands.jsonl records target servo angles, not feedback-measured physical angles.")
    try:
        controller = DryRunServoController(config) if args.dry_run else PCA9685ServoController(config)
        if args.dry_run:
            print("Dry run: PCA9685 is not initialized and no servo PWM is written.")

        # Establish only configured centers before entering the action pose.
        for channel in sorted(targets["centers"]):
            controller.move_safely(channel, targets["centers"][channel])
        _write_command(
            command_logger,
            targets,
            "safe_center_initialized",
            0.0,
            time.monotonic() - experiment_start_s,
            targets["centers"],
            {"dry_run": bool(args.dry_run)},
        )
        action_event(
            event_logger,
            "action_started",
            ACTION_NAME,
            action_state="safe_center_initialized",
            data={"dry_run": bool(args.dry_run), "log_dir": str(log_dir)},
        )

        entry_duration_s = args.return_to_center_s
        last_commands = _run_stage(
            config=config,
            args=args,
            stage_name="enter_initial_pose",
            duration_s=entry_duration_s,
            command_builder=lambda elapsed: (
                interpolate_angles(targets["centers"], targets["initial_pose"], elapsed / entry_duration_s),
                {"transition_progress": min(1.0, elapsed / entry_duration_s)},
            ),
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
            experiment_start_s=experiment_start_s,
        )

        last_commands = _run_stage(
            config=config,
            args=args,
            stage_name="initial_hold",
            duration_s=args.initial_hold_s,
            command_builder=lambda elapsed: (dict(targets["initial_pose"]), {"hold_elapsed_s": elapsed}),
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
            experiment_start_s=experiment_start_s,
        )

        action_start_s = time.monotonic()
        last_commands = _run_stage(
            config=config,
            args=args,
            stage_name="periodic_action",
            duration_s=args.duration,
            command_builder=lambda elapsed: _periodic_commands(elapsed, args, targets),
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            event_logger=event_logger,
            experiment_start_s=experiment_start_s,
            action_start_s=action_start_s,
        )

        last_commands = _run_return_to_center(
            config,
            args,
            controller,
            targets,
            last_commands,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
            event_logger,
            experiment_start_s,
        )
        centered = True
        action_event(
            event_logger,
            "action_finished",
            ACTION_NAME,
            action_state="safe_center",
            data={
                "reason": reason,
                "completed_pectoral_cycles": int(math.floor(args.duration * args.pectoral_frequency_hz)),
                "duration_on_pectoral_boundary": _duration_on_cycle_boundary(
                    args.duration, args.pectoral_frequency_hz
                ),
            },
        )
    except KeyboardInterrupt:
        reason = "ctrl_c"
        print("\nInterrupted. Smooth safe recentering will be attempted.")
        action_event(event_logger, "ctrl_c_received", ACTION_NAME, action_state="interrupted")
    finally:
        if controller is not None and not centered:
            try:
                last_commands = _run_return_to_center(
                    config,
                    args,
                    controller,
                    targets,
                    last_commands,
                    synchronizer,
                    sync_logger,
                    raw_loggers,
                    command_logger,
                    event_logger,
                    experiment_start_s,
                )
                centered = True
            except Exception as exc:
                action_event(
                    event_logger,
                    "safe_recenter_failed",
                    ACTION_NAME,
                    action_state="return_to_center",
                    data={"error": f"{type(exc).__name__}: {exc}"},
                )
                print(f"Safe recenter failed: {type(exc).__name__}: {exc}")

        if controller is not None and release_after:
            sleep_safely(0.5)
            controller.stop_all(targets["moving_channels"])

        link_manager.stop()
        raw_loggers.write_from_buffers(sensor_manager.buffers)
        sensor_manager.stop_all()
        raw_loggers.write_from_buffers(sensor_manager.buffers)
        action_event(
            event_logger,
            "logger_stopped",
            ACTION_NAME,
            action_state="stopped",
            data={
                "reason": reason,
                "centered": centered,
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
        print(f"Action stopped. Logs are in: {log_dir}")
    return 0


def _validate_scalar_args(args: argparse.Namespace) -> None:
    if not args.dry_run and args.confirm != "MOVE":
        raise SystemExit("Refusing to move servos without --confirm MOVE. Use --dry-run for no-PWM validation.")
    try:
        validate_finite_range("--duration", args.duration, minimum=0.0, minimum_inclusive=False)
        validate_finite_range("--pectoral-frequency-hz", args.pectoral_frequency_hz, minimum=0.0, minimum_inclusive=False)
        validate_finite_range("--tail-frequency-hz", args.tail_frequency_hz, minimum=0.0, minimum_inclusive=False)
        validate_finite_range("--root-amplitude-ratio", args.root_amplitude_ratio, minimum=0.0, maximum=1.0)
        validate_finite_range("--tip-amplitude-deg", args.tip_amplitude_deg, minimum=0.0)
        validate_finite_range("--tail-left-amplitude-deg", args.tail_left_amplitude_deg, minimum=0.0)
        validate_finite_range("--tail-right-amplitude-deg", args.tail_right_amplitude_deg, minimum=0.0)
        validate_finite_range("--start-delay-s", args.start_delay_s, minimum=0.0)
        validate_finite_range("--initial-hold-s", args.initial_hold_s, minimum=0.0)
        validate_finite_range("--return-to-center-s", args.return_to_center_s, minimum=0.0, minimum_inclusive=False)
        validate_finite_range("--command-hz", args.command_hz, minimum=0.0, minimum_inclusive=False)
        quarter_period = 1.0 / (4.0 * args.pectoral_frequency_hz)
        validate_finite_range(
            "--tip-transition-s",
            args.tip_transition_s,
            minimum=0.0,
            maximum=quarter_period,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _prepare_targets(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    items = {servo_id: _servo_by_id(config, servo_id) for servo_id in ALL_SERVO_IDS}
    limits = {servo_id: servo_limits_from_config(config, items[servo_id]) for servo_id in ALL_SERVO_IDS}
    for servo_id in ALL_SERVO_IDS:
        servo_limits = limits[servo_id]
        if not servo_limits.min_angle <= servo_limits.center_angle <= servo_limits.max_angle:
            raise SystemExit(
                f"servo_id={servo_id} configured center_angle={servo_limits.center_angle:.3f} "
                f"deg is outside configured mechanical range from config/robot.yaml: "
                f"{servo_limits.min_angle:.3f}..{servo_limits.max_angle:.3f} deg."
            )
    tail: dict[int, dict[str, float]] = {}
    for servo_id in TAIL_SERVO_IDS:
        item = items[servo_id]
        servo_limits = limits[servo_id]
        channel = int(item["channel"])
        sign = float(item.get("sign", 1.0))
        scale = float(item.get("amplitude_scale", 1.0))
        if not math.isfinite(sign) or sign == 0.0:
            raise SystemExit(f"servo_id={servo_id} tail sign must be finite and non-zero, got {sign!r}.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise SystemExit(f"servo_id={servo_id} amplitude_scale must be > 0, got {scale!r}.")
        left_target = servo_limits.center_angle + sign * scale * args.tail_left_amplitude_deg
        right_target = servo_limits.center_angle - sign * scale * args.tail_right_amplitude_deg
        _check_tail_target(
            servo_id, "left", args.tail_left_amplitude_deg, left_target, servo_limits
        )
        _check_tail_target(
            servo_id, "right", args.tail_right_amplitude_deg, right_target, servo_limits
        )
        tail[servo_id] = {
            "channel": channel,
            "center": servo_limits.center_angle,
            "sign": sign,
            "scale": scale,
            "left_target": left_target,
            "right_target": right_target,
        }

    roots: dict[int, dict[str, float]] = {}
    for servo_id in ROOT_SERVO_IDS:
        item = items[servo_id]
        servo_limits = limits[servo_id]
        channel = int(item["channel"])
        bottom = _physical_reference(item, "bottom_reference_angle", servo_id)
        upper_limit = _physical_reference(item, "top_reference_angle", servo_id)
        try:
            servo_limits.validate(bottom)
            servo_limits.validate(upper_limit)
        except Exception as exc:
            raise SystemExit(f"servo_id={servo_id} calibrated root reference is unsafe: {exc}") from exc
        top = bottom + args.root_amplitude_ratio * (upper_limit - bottom)
        try:
            servo_limits.validate(top)
        except Exception as exc:
            raise SystemExit(
                f"servo_id={servo_id} root target {top:.3f} from bottom={bottom:.3f}, "
                f"upper_limit={upper_limit:.3f}, ratio={args.root_amplitude_ratio:.3f} is unsafe: {exc}"
            ) from exc
        roots[servo_id] = {
            "channel": channel,
            "center": servo_limits.center_angle,
            "bottom": bottom,
            "upper_limit": upper_limit,
            "top": top,
        }

    tips: dict[int, dict[str, float]] = {}
    for servo_id, direction_sign in ((5, -1.0), (7, 1.0)):
        item = items[servo_id]
        servo_limits = limits[servo_id]
        channel = int(item["channel"])
        target = servo_limits.center_angle + direction_sign * args.tip_amplitude_deg
        try:
            servo_limits.validate(target)
        except Exception as exc:
            raise SystemExit(
                f"servo_id={servo_id} tip amplitude {args.tip_amplitude_deg:.3f} deg requests "
                f"target {target:.3f}; configured mechanical range from config/robot.yaml is "
                f"{servo_limits.min_angle:.3f}..{servo_limits.max_angle:.3f}."
            ) from exc
        tips[servo_id] = {
            "channel": channel,
            "center": servo_limits.center_angle,
            "direction_sign": direction_sign,
            "target": target,
        }

    centers = {int(items[i]["channel"]): limits[i].center_angle for i in ALL_SERVO_IDS}
    initial_pose = dict(centers)
    for servo_id in ROOT_SERVO_IDS:
        initial_pose[int(roots[servo_id]["channel"])] = roots[servo_id]["bottom"]

    return {
        "items": items,
        "tail": tail,
        "roots": roots,
        "tips": tips,
        "centers": centers,
        "initial_pose": initial_pose,
        "moving_channels": sorted(centers),
        "channel_to_servo_id": {int(items[i]["channel"]): i for i in ALL_SERVO_IDS},
    }


def _run_stage(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    stage_name: str,
    duration_s: float,
    command_builder: CommandBuilder,
    controller: Any,
    targets: dict[str, Any],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
    experiment_start_s: float,
    action_start_s: float | None = None,
) -> dict[int, float]:
    duration = max(0.0, float(duration_s))
    action_event(
        event_logger,
        "action_state_started",
        ACTION_NAME,
        action_state=stage_name,
        data={"duration_s": duration},
    )
    command_period_s = 1.0 / args.command_hz
    sample_hz = max(0.1, float(config.get("runtime", {}).get("synchronized_sample_hz", 30.0)))
    sample_period_s = 1.0 / sample_hz
    print_hz = max(0.0, float(config.get("runtime", {}).get("print_hz", 2.0)))
    print_period_s = 1.0 / print_hz if print_hz > 0.0 else None
    stage_start_s = time.monotonic()
    next_command_s = stage_start_s
    next_sample_s = stage_start_s
    next_print_s = stage_start_s
    last_commands: dict[int, float] = {}

    while True:
        now_s = time.monotonic()
        elapsed_s = now_s - stage_start_s
        if elapsed_s >= duration:
            break
        if now_s >= next_command_s:
            commands, state = command_builder(elapsed_s)
            controller.write_angles(commands)
            last_commands = dict(commands)
            _write_command(
                command_logger,
                targets,
                stage_name,
                elapsed_s,
                now_s - experiment_start_s,
                commands,
                state,
                action_elapsed_s=None if action_start_s is None else now_s - action_start_s,
            )
            next_command_s += command_period_s
            if next_command_s < now_s - command_period_s:
                next_command_s = now_s + command_period_s

        if now_s >= next_sample_s:
            sample = synchronizer.build()
            raw_loggers.write_from_buffers(synchronizer.buffers)
            sync_logger.write(sample)
            if print_period_s is not None and now_s >= next_print_s:
                print(_summary(sample, now_s - experiment_start_s, stage_name))
                next_print_s = now_s + print_period_s
            next_sample_s += sample_period_s
            if next_sample_s < now_s - sample_period_s:
                next_sample_s = now_s + sample_period_s

        next_due_s = min(next_command_s, next_sample_s)
        if next_due_s > now_s:
            time.sleep(min(0.01, next_due_s - now_s))

    # Explicit endpoint makes exact cycle-boundary and transition targets observable.
    commands, state = command_builder(duration)
    now_s = time.monotonic()
    controller.write_angles(commands)
    last_commands = dict(commands)
    _write_command(
        command_logger,
        targets,
        stage_name,
        duration,
        now_s - experiment_start_s,
        commands,
        state,
        action_elapsed_s=None if action_start_s is None else duration,
    )
    action_event(
        event_logger,
        "action_state_finished",
        ACTION_NAME,
        action_state=stage_name,
        data={"duration_s": duration},
    )
    return last_commands


def _run_return_to_center(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
    targets: dict[str, Any],
    start_commands: dict[int, float],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
    experiment_start_s: float,
) -> dict[int, float]:
    start = {
        channel: float(start_commands.get(channel, targets["centers"][channel]))
        for channel in targets["centers"]
    }
    return _run_stage(
        config=config,
        args=args,
        stage_name="return_to_center",
        duration_s=args.return_to_center_s,
        command_builder=lambda elapsed: safe_recenter_profile(
            start, targets["centers"], elapsed, args.return_to_center_s
        ),
        controller=controller,
        targets=targets,
        synchronizer=synchronizer,
        sync_logger=sync_logger,
        raw_loggers=raw_loggers,
        command_logger=command_logger,
        event_logger=event_logger,
        experiment_start_s=experiment_start_s,
    )


def _periodic_commands(
    elapsed_s: float,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    clock = cycle_state(elapsed_s, args.pectoral_frequency_hz)
    q = cosine_cycle_position(clock.t_cycle_s, clock.period_s)
    h = tip_envelope(clock.t_cycle_s, clock.period_s, args.tip_transition_s)
    logical_tail_offset = asymmetric_tail_offset(
        elapsed_s,
        args.tail_frequency_hz,
        args.tail_left_amplitude_deg,
        args.tail_right_amplitude_deg,
        args.tail_first_direction,
    )
    commands: dict[int, float] = {}
    for values in targets["tail"].values():
        commands[int(values["channel"])] = (
            values["center"] + values["sign"] * values["scale"] * logical_tail_offset
        )
    for values in targets["roots"].values():
        commands[int(values["channel"])] = values["bottom"] + q * (values["top"] - values["bottom"])
    for values in targets["tips"].values():
        commands[int(values["channel"])] = values["center"] + values["direction_sign"] * args.tip_amplitude_deg * h
    return (
        commands,
        {
            "pectoral_period_s": clock.period_s,
            "pectoral_t_cycle_s": clock.t_cycle_s,
            "pectoral_phase": clock.phase,
            "pectoral_cycle_index": clock.cycle_index,
            "root_cosine_q": q,
            "tip_envelope_h": h,
            "tail_offset_deg": logical_tail_offset,
        },
    )


def _write_command(
    command_logger: JsonlLogger,
    targets: dict[str, Any],
    action_state: str,
    state_elapsed_s: float,
    experiment_elapsed_s: float,
    commands: dict[int, float],
    state: dict[str, Any],
    *,
    action_elapsed_s: float | None = None,
) -> None:
    command_logger.write(
        {
            "t_ns": time.monotonic_ns(),
            "action": ACTION_NAME,
            "action_state": action_state,
            "state_elapsed_s": state_elapsed_s,
            "experiment_elapsed_s": experiment_elapsed_s,
            "action_elapsed_s": action_elapsed_s,
            "pectoral_period_s": state.get("pectoral_period_s"),
            "pectoral_t_cycle_s": state.get("pectoral_t_cycle_s"),
            "pectoral_phase": state.get("pectoral_phase"),
            "pectoral_cycle_index": state.get("pectoral_cycle_index"),
            "root_cosine_q": state.get("root_cosine_q"),
            "tip_envelope_h": state.get("tip_envelope_h"),
            "tail_offset_deg": state.get("tail_offset_deg"),
            "transition_progress": state.get("transition_progress"),
            "tail_amplitude_scale": state.get("tail_amplitude_scale"),
            "dry_run": state.get("dry_run"),
            "angle_semantics": "target_servo_angles_not_feedback_measurements",
            "commands_deg": {str(channel): angle for channel, angle in commands.items()},
            "commands_by_servo_id_deg": {
                str(targets["channel_to_servo_id"][channel]): angle
                for channel, angle in commands.items()
            },
        }
    )


def _print_profile_checkpoints(args: argparse.Namespace, targets: dict[str, Any]) -> None:
    period = 1.0 / args.pectoral_frequency_hz
    checkpoints = (
        0.0,
        args.tip_transition_s,
        period / 2.0 - args.tip_transition_s,
        period / 2.0,
        period,
    )
    print("Dry-run trajectory checkpoints (target angles by servo id):")
    for elapsed in checkpoints:
        commands, state = _periodic_commands(elapsed, args, targets)
        by_servo = {
            targets["channel_to_servo_id"][channel]: round(angle, 3)
            for channel, angle in commands.items()
        }
        print(
            f"  t={elapsed:.3f}s phase={state['pectoral_phase']:.3f} "
            f"q={state['root_cosine_q']:.3f} h={state['tip_envelope_h']:.3f} "
            f"targets={by_servo}"
        )


def _check_tail_target(
    servo_id: int,
    side: str,
    requested_amplitude: float,
    target: float,
    servo_limits: Any,
) -> None:
    if not servo_limits.min_angle <= target <= servo_limits.max_angle:
        raise SystemExit(
            f"servo_id={servo_id} requested {side} amplitude={requested_amplitude:.3f} deg "
            f"produces target={target:.3f} deg; configured mechanical range from "
            f"config/robot.yaml is {servo_limits.min_angle:.3f}.."
            f"{servo_limits.max_angle:.3f} deg."
        )


def _physical_reference(item: dict[str, Any], key: str, servo_id: int) -> float:
    direction = item.get("direction")
    if not isinstance(direction, dict) or key not in direction:
        raise SystemExit(f"servo_id={servo_id} is missing calibrated direction.{key} in config/robot.yaml.")
    value = float(direction[key])
    if not math.isfinite(value):
        raise SystemExit(f"servo_id={servo_id} direction.{key} must be finite.")
    return value


def _servo_by_id(config: dict[str, Any], servo_id: int) -> dict[str, Any]:
    for item in configured_servo_channels(config):
        if int(item.get("servo_id", -1)) == servo_id:
            return item
    raise SystemExit(f"Missing servo_id={servo_id} in config/robot.yaml")


def _duration_on_cycle_boundary(duration_s: float, frequency_hz: float) -> bool:
    cycles = duration_s * frequency_hz
    return math.isclose(cycles, round(cycles), rel_tol=0.0, abs_tol=1e-9)


def _resolve_log_base_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    base_dir = Path(str(logging_cfg.get("base_dir", runtime_cfg.get("log_dir", "logs"))))
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    return base_dir


def _write_metadata(
    config: dict[str, Any],
    args: argparse.Namespace,
    targets: dict[str, Any],
    log_dir: Path,
) -> None:
    metadata_file = str(config.get("logging", {}).get("metadata_file", "metadata.yaml"))
    root_targets = {
        str(servo_id): {
            "bottom_deg": values["bottom"],
            "top_deg": values["top"],
            "calibrated_upper_limit_deg": values["upper_limit"],
        }
        for servo_id, values in targets["roots"].items()
    }
    metadata = {
        "started_wall_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "action",
        "action": ACTION_NAME,
        "dry_run": bool(args.dry_run),
        "mock_sensors": bool(args.mock_sensors),
        "duration_s": args.duration,
        "pectoral_frequency_hz": args.pectoral_frequency_hz,
        "root_amplitude_ratio": args.root_amplitude_ratio,
        "root_targets": root_targets,
        "tip_amplitude_deg": args.tip_amplitude_deg,
        "tip_transition_s": args.tip_transition_s,
        "tail_frequency_hz": args.tail_frequency_hz,
        "tail_left_amplitude_deg": args.tail_left_amplitude_deg,
        "tail_right_amplitude_deg": args.tail_right_amplitude_deg,
        "tail_first_direction": args.tail_first_direction,
        "start_delay_s": args.start_delay_s,
        "initial_hold_s": args.initial_hold_s,
        "return_to_center_s": args.return_to_center_s,
        "command_hz": args.command_hz,
        "servo_ids": list(ALL_SERVO_IDS),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "servo": config.get("servo", {}),
        "notes": [
            "All pectoral servos use one elapsed-time clock and one pectoral phase.",
            "Root servos use a bottom-to-top-to-bottom cosine trajectory.",
            "Tip servos use opposite signed deflections with a quintic smoothstep envelope.",
            "Tail servos start at center and use one asymmetric sinusoidal logical waveform.",
            "Configured per-servo min_angle/max_angle already include the mechanical safety allowance; no extra margin is applied.",
            "commands.jsonl contains target servo angles, not feedback-measured actual angles.",
            "The start-delay countdown occurs before this log directory and sensor acquisition begin.",
        ],
    }
    write_metadata_yaml(log_dir / metadata_file, metadata)


def _summary(sample: dict[str, Any], elapsed_s: float, state: str) -> str:
    status = sample.get("status", {})
    depth = sample.get("depth", {})
    depth_text = "-" if depth.get("depth_m") is None else f"{depth.get('depth_m'):.3f}m"
    return (
        f"t={elapsed_s:.1f}s state={state} "
        f"imu={'ok' if sample.get('imu', {}).get('valid') else 'bad'} "
        f"depth={depth_text} "
        f"power={'ok' if sample.get('power', {}).get('valid') else 'bad'} "
        f"uwb={'ok' if sample.get('uwb', {}).get('valid') else sample.get('uwb', {}).get('error', 'bad')} "
        f"vision={'ok' if sample.get('vision', {}).get('valid') else sample.get('vision', {}).get('error', 'bad')} "
        f"missing={status.get('missing_sensors', [])}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
