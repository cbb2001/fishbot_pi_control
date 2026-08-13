from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()

from control.runtime.action_utils import (  # noqa: E402
    DryRunServoController,
    asymmetric_tail_offset,
    countdown_start_delay,
    interpolate_angles,
    smoothstep5,
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
from control.runtime.roll_motion import (  # noqa: E402
    RootPairTargets,
    TailServoMotion,
    build_roll_root_targets,
    evaluate_roll_trajectory,
    tip_sign_from_direction,
)
from control.runtime.sensor_manager import SensorManager  # noqa: E402
from control.runtime.sensor_synchronizer import SensorSynchronizer  # noqa: E402
from control.safety import (  # noqa: E402
    configured_servo_channels,
    ensure_not_windows_hardware_run,
    load_robot_config,
    servo_limits_from_config,
)
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


MISSION_NAME = "dive_roll_20260713"
TAIL_SERVO_IDS = (1, 2, 3)
ROOT_SERVO_IDS = (4, 6)
TIP_SERVO_IDS = (5, 7)
ALL_SERVO_IDS = TAIL_SERVO_IDS + ROOT_SERVO_IDS + TIP_SERVO_IDS
DRY_RUN_PROFILE_FILE = "dry_run_dive_roll_trajectory.jsonl"

CommandBuilder = Callable[[float], tuple[dict[int, float], dict[str, Any]]]


def parse_args(
    config: dict[str, Any] | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    del config  # Reserved for future configuration-backed defaults.
    parser = argparse.ArgumentParser(
        description=(
            "Open-loop combined mission: after the local start delay, dive with fixed mirrored "
            "pectoral-tip attack angle, transition smoothly to the roll initial pose, then run "
            "the roll_20260713 trajectory while recording all sensors continuously."
        )
    )
    parser.add_argument("--confirm", default="", help="Must be MOVE to command real servos.")
    parser.add_argument("--dry-run", action="store_true", help="Do not initialize PCA9685 or write real PWM.")
    parser.add_argument(
        "--mock-sensors",
        action="store_true",
        help="Use mock sensors; real servos still move unless --dry-run is also set.",
    )
    parser.add_argument("--keep-pwm", action="store_true", help="Keep PWM active after safe recentering.")

    parser.add_argument(
        "--dive-duration",
        type=float,
        default=10.0,
        help="Fixed-attack-angle dive action time in seconds, excluding transitions.",
    )
    parser.add_argument(
        "--dive-pectoral-tilt",
        type=float,
        default=20.0,
        help="Dive tip tilt: servo 5 center+tilt and servo 7 center-tilt.",
    )
    parser.add_argument(
        "--transition-s",
        type=float,
        default=1.0,
        help="Smooth center-to-dive and dive-to-roll transition duration.",
    )

    parser.add_argument("--roll-direction", choices=("left", "right"), default="right")
    parser.add_argument("--tip-direction", choices=("positive", "negative"), default="negative")
    parser.add_argument("--duration", type=float, default=10.0, help="Roll action time, excluding dive and transitions.")
    parser.add_argument("--pectoral-frequency-hz", type=float, default=0.2, help="Shared roll 4/5/6/7 frequency.")
    parser.add_argument(
        "--root-amplitude-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction 0..1 of each root servo's explicit physical top-to-bottom travel; "
            "roll motion starts at the direction-selected physical extreme."
        ),
    )
    parser.add_argument("--tip-amplitude-deg", type=float, default=10.0, help="Common roll 5/7 offset magnitude.")
    parser.add_argument("--tip-transition-s", type=float, default=0.8, help="Quintic roll 5/7 transition time.")
    parser.add_argument("--tail-frequency-hz", type=float, default=0.2, help="Shared tail frequency in dive and roll.")
    parser.add_argument("--tail-left-amplitude-deg", type=float, default=5.0)
    parser.add_argument("--tail-right-amplitude-deg", type=float, default=5.0)
    parser.add_argument(
        "--tail-first-direction",
        choices=("left", "right"),
        default="left",
        help="First tail excursion in both actions; each action starts its tail clock at center.",
    )
    parser.add_argument("--start-delay-s", type=float, default=20.0, help="Local countdown before logs and hardware.")
    parser.add_argument("--initial-hold-s", type=float, default=1.0, help="Hold after reaching the roll initial pose.")
    parser.add_argument(
        "--tail-ramp-down-s",
        type=float,
        default=1.0,
        help="Smooth tail-amplitude fade time after the roll action.",
    )
    parser.add_argument(
        "--return-to-center-s",
        type=float,
        default=2.0,
        help="Smooth final return-to-center duration.",
    )
    parser.add_argument("--command-hz", type=float, default=50.0, help="Target command update rate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    config = load_robot_config()
    args = parse_args(config, argv)
    _validate_scalar_args(args)
    targets = _prepare_targets(config, args)
    _validate_motion_authorization(args)
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()

    if args.dry_run:
        _print_profile_checkpoints(args, targets)

    # The delay is deliberately before formal logs, sensors, and servo initialization.
    start_delay_begin_ns = time.monotonic_ns()
    countdown_start_delay(args.start_delay_s)
    start_delay_end_ns = time.monotonic_ns()

    log_dir = create_run_log_dir(_resolve_log_base_dir(config), suffix=MISSION_NAME)
    prepare_raw_placeholders(log_dir)
    _write_metadata(config, args, targets, log_dir)
    if args.dry_run:
        preview_path = _write_dry_run_profile(log_dir, args, targets)
        print(f"Full combined dry-run profile: {preview_path}")

    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    flush_interval_s = float(
        logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0))
    )
    queue_maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))
    sync_logger = JsonlLogger(
        log_dir / str(logging_cfg.get("synchronized_file", "synchronized_sensors.jsonl")),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    raw_loggers = RawSensorLoggers(
        log_dir,
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
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

    controller: Any | None = None
    sensor_start_attempted = False
    link_start_attempted = False
    recording_began = False
    reason = "duration_elapsed"
    runtime_state: dict[str, Any] = {
        "last_commands": dict(targets["centers"]),
        "dive_elapsed_s": None,
        "roll_elapsed_s": None,
    }
    lifecycle = {
        "tail_ramp_attempted": False,
        "tail_ramp_done": False,
        "return_attempted": False,
        "centered": False,
    }
    release_after = (
        bool(config.get("safety", {}).get("servo", {}).get("release_pwm_after_tests", True))
        and not args.keep_pwm
        and not args.dry_run
    )

    sync_logger.start()
    raw_loggers.start()
    event_logger.start()
    command_logger.start()
    _write_event(
        event_logger,
        "start_delay_begin",
        args,
        data={"requested_delay_s": args.start_delay_s},
        t_ns=start_delay_begin_ns,
    )
    _write_event(
        event_logger,
        "start_delay_end",
        args,
        data={
            "requested_delay_s": args.start_delay_s,
            "actual_delay_s": (start_delay_end_ns - start_delay_begin_ns) / 1_000_000_000.0,
        },
        t_ns=start_delay_end_ns,
    )

    experiment_start_s = time.monotonic()
    print(f"Mission logs created: {log_dir}")
    print("IMU and depth are recorded only; they do not modify this open-loop trajectory.")
    try:
        sensor_start_attempted = True
        sensor_manager.start_all()
        recording_began = True
        _write_event(
            event_logger,
            "sensor_recording_begin",
            args,
            data={
                "mock_sensors": bool(args.mock_sensors),
                "synchronized_sample_hz": float(
                    config.get("runtime", {}).get("synchronized_sample_hz", 30.0)
                ),
            },
        )
        _record_synchronized_sample(synchronizer, sync_logger, raw_loggers)
        link_start_attempted = True
        link_manager.start()

        # Sensors are active before either the dry-run validator or PCA9685 is constructed.
        controller = DryRunServoController(config) if args.dry_run else PCA9685ServoController(config)
        if args.dry_run:
            print("Dry run: PCA9685 was not initialized and no real PWM is written.")

        _write_event(event_logger, "safe_center_begin", args)
        for channel in sorted(targets["centers"]):
            controller.move_safely(channel, targets["centers"][channel])
        _write_command(
            command_logger,
            args,
            targets,
            action_state="safe_center_initialized",
            state_elapsed_s=0.0,
            experiment_elapsed_s=time.monotonic() - experiment_start_s,
            commands=targets["centers"],
            state={"dry_run": bool(args.dry_run)},
        )
        runtime_state["last_commands"] = dict(targets["centers"])
        _write_event(event_logger, "safe_center_ready", args)

        _write_event(event_logger, "dive_initial_pose_begin", args)
        _run_stage(
            config=config,
            args=args,
            stage_name="dive_initial_pose",
            duration_s=args.transition_s,
            command_builder=lambda elapsed: (
                interpolate_angles(
                    targets["centers"],
                    targets["dive_pose"],
                    elapsed / args.transition_s,
                ),
                {"transition_progress": min(1.0, elapsed / args.transition_s)},
            ),
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            experiment_start_s=experiment_start_s,
            runtime_state=runtime_state,
        )
        _write_event(event_logger, "dive_initial_pose_ready", args)

        _write_event(
            event_logger,
            "dive_action_begin",
            args,
            data={
                "duration_s": args.dive_duration,
                "dive_pectoral_tilt_deg": args.dive_pectoral_tilt,
            },
        )
        try:
            _run_stage(
                config=config,
                args=args,
                stage_name="dive_action",
                duration_s=args.dive_duration,
                command_builder=lambda elapsed: _dive_commands(elapsed, args, targets),
                controller=controller,
                targets=targets,
                synchronizer=synchronizer,
                sync_logger=sync_logger,
                raw_loggers=raw_loggers,
                command_logger=command_logger,
                experiment_start_s=experiment_start_s,
                runtime_state=runtime_state,
                timeline="dive",
            )
        except BaseException as exc:
            _write_event(
                event_logger,
                "dive_action_end",
                args,
                data={"reason": _exception_reason(exc)},
            )
            raise
        else:
            _write_event(
                event_logger,
                "dive_action_end",
                args,
                data={"reason": "duration_elapsed"},
            )

        _write_event(
            event_logger,
            "dive_to_roll_transition_begin",
            args,
            data={"duration_s": args.transition_s},
        )
        _write_event(event_logger, "initial_pose_begin", args)
        try:
            _run_stage(
                config=config,
                args=args,
                stage_name="dive_to_roll_transition",
                duration_s=args.transition_s,
                command_builder=lambda elapsed: _dive_to_roll_commands(
                    elapsed,
                    args,
                    targets,
                ),
                controller=controller,
                targets=targets,
                synchronizer=synchronizer,
                sync_logger=sync_logger,
                raw_loggers=raw_loggers,
                command_logger=command_logger,
                experiment_start_s=experiment_start_s,
                runtime_state=runtime_state,
            )
        except BaseException as exc:
            _write_event(
                event_logger,
                "dive_to_roll_transition_end",
                args,
                data={"reason": _exception_reason(exc)},
            )
            raise
        else:
            _write_event(event_logger, "initial_pose_ready", args)
            _write_event(
                event_logger,
                "dive_to_roll_transition_end",
                args,
                data={"reason": "completed"},
            )

        _write_event(
            event_logger,
            "baseline_hold_begin",
            args,
            data={"initial_hold_s": args.initial_hold_s},
        )
        _run_stage(
            config=config,
            args=args,
            stage_name="initial_hold",
            duration_s=args.initial_hold_s,
            command_builder=lambda elapsed: (
                dict(targets["initial_pose"]),
                {"hold_elapsed_s": elapsed},
            ),
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            experiment_start_s=experiment_start_s,
            runtime_state=runtime_state,
        )
        _write_event(event_logger, "baseline_hold_end", args)

        _write_event(
            event_logger,
            "roll_action_begin",
            args,
            data={
                "duration_s": args.duration,
                "pectoral_frequency_hz": args.pectoral_frequency_hz,
            },
        )
        try:
            _run_stage(
                config=config,
                args=args,
                stage_name="roll_action",
                duration_s=args.duration,
                command_builder=lambda elapsed: _roll_commands(elapsed, args, targets),
                controller=controller,
                targets=targets,
                synchronizer=synchronizer,
                sync_logger=sync_logger,
                raw_loggers=raw_loggers,
                command_logger=command_logger,
                experiment_start_s=experiment_start_s,
                runtime_state=runtime_state,
                timeline="roll",
            )
        except BaseException as exc:
            _write_event(
                event_logger,
                "roll_action_end",
                args,
                data={"reason": _exception_reason(exc)},
            )
            raise
        else:
            _write_event(
                event_logger,
                "roll_action_end",
                args,
                data={
                    "reason": "duration_elapsed",
                    "duration_on_pectoral_boundary": _duration_on_cycle_boundary(
                        args.duration,
                        args.pectoral_frequency_hz,
                    ),
                },
            )

        lifecycle["tail_ramp_attempted"] = True
        _execute_tail_ramp_down(
            config,
            args,
            controller,
            targets,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
            event_logger,
            experiment_start_s,
            runtime_state,
        )
        lifecycle["tail_ramp_done"] = True

        lifecycle["return_attempted"] = True
        _execute_return_to_center(
            config,
            args,
            controller,
            targets,
            synchronizer,
            sync_logger,
            raw_loggers,
            command_logger,
            event_logger,
            experiment_start_s,
            runtime_state,
        )
        lifecycle["centered"] = True
    except KeyboardInterrupt:
        reason = "ctrl_c"
        print("\nInterrupted. Smooth tail fade and recentering will be attempted.")
        _write_event(event_logger, "ctrl_c_received", args, data={"action_state": "interrupted"})
    except Exception as exc:
        reason = "error"
        _write_event(
            event_logger,
            "action_error",
            args,
            data={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        if controller is not None:
            if not lifecycle["tail_ramp_attempted"]:
                lifecycle["tail_ramp_attempted"] = True
                try:
                    _execute_tail_ramp_down(
                        config,
                        args,
                        controller,
                        targets,
                        synchronizer,
                        sync_logger,
                        raw_loggers,
                        command_logger,
                        event_logger,
                        experiment_start_s,
                        runtime_state,
                    )
                    lifecycle["tail_ramp_done"] = True
                except BaseException as exc:
                    _cleanup_step(
                        "write tail-ramp failure event",
                        lambda: _write_event(
                            event_logger,
                            "tail_ramp_down_failed",
                            args,
                            data={"error": _exception_reason(exc)},
                        ),
                    )
                    print(f"Tail ramp-down failed: {type(exc).__name__}: {exc}")

            if not lifecycle["return_attempted"]:
                lifecycle["return_attempted"] = True
                try:
                    _execute_return_to_center(
                        config,
                        args,
                        controller,
                        targets,
                        synchronizer,
                        sync_logger,
                        raw_loggers,
                        command_logger,
                        event_logger,
                        experiment_start_s,
                        runtime_state,
                    )
                    lifecycle["centered"] = True
                except BaseException as exc:
                    _cleanup_step(
                        "write recenter failure event",
                        lambda: _write_event(
                            event_logger,
                            "return_to_center_failed",
                            args,
                            data={"error": _exception_reason(exc)},
                        ),
                    )
                    print(f"Return to center failed: {type(exc).__name__}: {exc}")

        if link_start_attempted:
            _cleanup_step("stop link manager", link_manager.stop)
        if sensor_start_attempted:
            if recording_began:
                _cleanup_step(
                    "write final synchronized sample",
                    lambda: _record_synchronized_sample(
                        synchronizer,
                        sync_logger,
                        raw_loggers,
                    ),
                )
            _cleanup_step(
                "flush raw sensor buffers before stop",
                lambda: raw_loggers.write_from_buffers(sensor_manager.buffers),
            )
            _cleanup_step("stop sensor manager", sensor_manager.stop_all)
            _cleanup_step(
                "flush raw sensor buffers after stop",
                lambda: raw_loggers.write_from_buffers(sensor_manager.buffers),
            )
        if recording_began:
            _cleanup_step(
                "write sensor-recording end event",
                lambda: _write_event(
                    event_logger,
                    "sensor_recording_end",
                    args,
                    data={"reason": reason, "centered": bool(lifecycle["centered"])},
                ),
            )
        _cleanup_step(
            "write logger-stopped event",
            lambda: _write_event(
                event_logger,
                "logger_stopped",
                args,
                data={
                    "reason": reason,
                    "centered": bool(lifecycle["centered"]),
                    "dry_run": bool(args.dry_run),
                    "sync_dropped_count": sync_logger.dropped_count,
                    "sync_last_error": sync_logger.last_error,
                    "raw_logger_stats": raw_loggers.stats(),
                    "command_dropped_count": command_logger.dropped_count,
                    "command_last_error": command_logger.last_error,
                },
            ),
        )
        _cleanup_step("stop synchronized logger", sync_logger.stop)
        _cleanup_step("stop raw sensor loggers", raw_loggers.stop)
        _cleanup_step("stop command logger", command_logger.stop)
        _cleanup_step("stop event logger", event_logger.stop)

        force_release_after_failed_recenter = (
            controller is not None
            and not args.dry_run
            and not lifecycle["centered"]
        )
        if controller is not None and (release_after or force_release_after_failed_recenter):
            pwm_disabled = _cleanup_step(
                "disable PWM",
                lambda: controller.stop_all(targets["moving_channels"]),
            )
            if pwm_disabled:
                if force_release_after_failed_recenter:
                    print("PWM forcibly disabled because safe recentering did not complete.")
                else:
                    print("PWM disabled after recentering.")
        elif controller is not None and not args.dry_run and lifecycle["centered"]:
            print("PWM remains active at the configured centers.")
        print(f"Mission stopped. Logs are in: {log_dir}")
    return 0


def _validate_scalar_args(args: argparse.Namespace) -> None:
    try:
        if args.roll_direction not in {"left", "right"}:
            raise ValueError("--roll-direction must be left or right.")
        if args.tip_direction not in {"positive", "negative"}:
            raise ValueError("--tip-direction must be positive or negative.")
        if args.tail_first_direction not in {"left", "right"}:
            raise ValueError("--tail-first-direction must be left or right.")
        validate_finite_range(
            "--dive-duration",
            args.dive_duration,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range(
            "--dive-pectoral-tilt",
            args.dive_pectoral_tilt,
            minimum=0.0,
        )
        validate_finite_range(
            "--transition-s",
            args.transition_s,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range("--duration", args.duration, minimum=0.0, minimum_inclusive=False)
        validate_finite_range(
            "--pectoral-frequency-hz",
            args.pectoral_frequency_hz,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range(
            "--root-amplitude-ratio",
            args.root_amplitude_ratio,
            minimum=0.0,
            maximum=1.0,
        )
        validate_finite_range("--tip-amplitude-deg", args.tip_amplitude_deg, minimum=0.0)
        validate_finite_range(
            "--tail-frequency-hz",
            args.tail_frequency_hz,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range(
            "--tail-left-amplitude-deg",
            args.tail_left_amplitude_deg,
            minimum=0.0,
        )
        validate_finite_range(
            "--tail-right-amplitude-deg",
            args.tail_right_amplitude_deg,
            minimum=0.0,
        )
        validate_finite_range("--start-delay-s", args.start_delay_s, minimum=0.0)
        validate_finite_range("--initial-hold-s", args.initial_hold_s, minimum=0.0)
        validate_finite_range(
            "--tail-ramp-down-s",
            args.tail_ramp_down_s,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range(
            "--return-to-center-s",
            args.return_to_center_s,
            minimum=0.0,
            minimum_inclusive=False,
        )
        validate_finite_range(
            "--command-hz",
            args.command_hz,
            minimum=0.0,
            minimum_inclusive=False,
        )
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


def _validate_motion_authorization(args: argparse.Namespace) -> None:
    if not args.dry_run and args.confirm != "MOVE":
        raise SystemExit(
            "Refusing to move real servos without --confirm MOVE. "
            "--mock-sensors only simulates sensors; use --dry-run to disable PCA9685/PWM."
        )


def _check_target(
    servo_id: int,
    request: str,
    target: float,
    servo_limits: Any,
) -> None:
    """Validate against robot.yaml's inclusive, safety-adjusted limits."""
    value = float(target)
    if (
        not math.isfinite(value)
        or not servo_limits.min_angle <= value <= servo_limits.max_angle
    ):
        raise SystemExit(
            f"servo_id={servo_id} request={request}; calculated target={value:.3f} deg; "
            f"configured mechanical range={servo_limits.min_angle:.3f}.."
            f"{servo_limits.max_angle:.3f} deg from config/robot.yaml."
        )


def _physical_reference(item: dict[str, Any], key: str, servo_id: int) -> float:
    direction = item.get("direction")
    if not isinstance(direction, dict) or key not in direction:
        raise SystemExit(
            f"servo_id={servo_id} is missing explicit direction.{key} in config/robot.yaml; "
            "physical top/bottom will not be guessed from min_angle/max_angle."
        )
    value = float(direction[key])
    if not math.isfinite(value):
        raise SystemExit(
            f"servo_id={servo_id} direction.{key} must be finite, got {value!r}."
        )
    return value


def _commands_by_channel(
    by_servo_id: dict[int, float],
    items: dict[int, dict[str, Any]],
) -> dict[int, float]:
    return {
        int(items[servo_id]["channel"]): float(angle)
        for servo_id, angle in by_servo_id.items()
    }


def _servo_by_id(config: dict[str, Any], servo_id: int) -> dict[str, Any]:
    for item in configured_servo_channels(config):
        if int(item.get("servo_id", -1)) == servo_id:
            return item
    raise SystemExit(f"Missing servo_id={servo_id} in config/robot.yaml")


def _duration_on_cycle_boundary(duration_s: float, frequency_hz: float) -> bool:
    cycles = duration_s * frequency_hz
    return math.isclose(cycles, round(cycles), rel_tol=0.0, abs_tol=1e-9)


def _prepare_targets(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    items = {servo_id: _servo_by_id(config, servo_id) for servo_id in ALL_SERVO_IDS}
    limits = {
        servo_id: servo_limits_from_config(config, items[servo_id])
        for servo_id in ALL_SERVO_IDS
    }
    mechanical_ranges: dict[int, tuple[float, float]] = {}
    for servo_id in ALL_SERVO_IDS:
        servo_limits = limits[servo_id]
        mechanical_ranges[servo_id] = (
            servo_limits.min_angle,
            servo_limits.max_angle,
        )
        _check_target(
            servo_id,
            "configured center_angle",
            servo_limits.center_angle,
            servo_limits,
        )

    physical_endpoints = {
        "servo_4_top_deg": _physical_reference(items[4], "top_reference_angle", 4),
        "servo_4_bottom_deg": _physical_reference(items[4], "bottom_reference_angle", 4),
        "servo_6_top_deg": _physical_reference(items[6], "top_reference_angle", 6),
        "servo_6_bottom_deg": _physical_reference(items[6], "bottom_reference_angle", 6),
    }
    for servo_id in ROOT_SERVO_IDS:
        servo_limits = limits[servo_id]
        for position in ("top", "bottom"):
            endpoint = physical_endpoints[f"servo_{servo_id}_{position}_deg"]
            _check_target(
                servo_id,
                f"configured physical {position} endpoint",
                endpoint,
                servo_limits,
            )

    try:
        roots = build_roll_root_targets(
            roll_direction=args.roll_direction,
            root_amplitude_ratio=args.root_amplitude_ratio,
            **physical_endpoints,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root_values = {
        4: (roots.servo_4_start_deg, roots.servo_4_target_deg),
        6: (roots.servo_6_start_deg, roots.servo_6_target_deg),
    }
    for servo_id, (start, target) in root_values.items():
        servo_limits = limits[servo_id]
        full_physical_travel = abs(
            physical_endpoints[f"servo_{servo_id}_top_deg"]
            - physical_endpoints[f"servo_{servo_id}_bottom_deg"]
        )
        requested_travel = args.root_amplitude_ratio * full_physical_travel
        _check_target(
            servo_id,
            f"roll_direction={args.roll_direction} root start",
            start,
            servo_limits,
        )
        _check_target(
            servo_id,
            f"--root-amplitude-ratio={args.root_amplitude_ratio:.6f} calculated root target",
            target,
            servo_limits,
        )
        if not math.isclose(abs(target - start), requested_travel, abs_tol=1e-9):
            raise SystemExit(
                f"servo_id={servo_id} internal root travel mismatch: requested="
                f"ratio {args.root_amplitude_ratio:.6f} of full physical travel "
                f"{full_physical_travel:.3f} deg = {requested_travel:.3f} deg, "
                f"calculated={abs(target - start):.3f} deg."
            )

    tip_sign = tip_sign_from_direction(args.tip_direction)
    tips: dict[int, dict[str, float]] = {}
    for servo_id in TIP_SERVO_IDS:
        servo_limits = limits[servo_id]
        target = servo_limits.center_angle + tip_sign * args.tip_amplitude_deg
        _check_target(
            servo_id,
            f"--tip-amplitude-deg={args.tip_amplitude_deg:.3f}, "
            f"--tip-direction={args.tip_direction}",
            target,
            servo_limits,
        )
        tips[servo_id] = {
            "channel": int(items[servo_id]["channel"]),
            "center": servo_limits.center_angle,
            "target": target,
        }

    tail: dict[int, dict[str, float]] = {}
    tail_motion: dict[int, TailServoMotion] = {}
    for servo_id in TAIL_SERVO_IDS:
        item = items[servo_id]
        servo_limits = limits[servo_id]
        direction_sign = float(item.get("sign", 1.0))
        amplitude_scale = float(item.get("amplitude_scale", 1.0))
        if not math.isfinite(direction_sign) or direction_sign == 0.0:
            raise SystemExit(
                f"servo_id={servo_id} configured sign must be finite and non-zero, "
                f"got {direction_sign!r}."
            )
        if not math.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
            raise SystemExit(
                f"servo_id={servo_id} configured amplitude_scale must be finite and > 0, "
                f"got {amplitude_scale!r}."
            )
        left_target = (
            servo_limits.center_angle
            + direction_sign * amplitude_scale * args.tail_left_amplitude_deg
        )
        right_target = (
            servo_limits.center_angle
            - direction_sign * amplitude_scale * args.tail_right_amplitude_deg
        )
        _check_target(
            servo_id,
            f"--tail-left-amplitude-deg={args.tail_left_amplitude_deg:.3f}",
            left_target,
            servo_limits,
        )
        _check_target(
            servo_id,
            f"--tail-right-amplitude-deg={args.tail_right_amplitude_deg:.3f}",
            right_target,
            servo_limits,
        )
        tail[servo_id] = {
            "channel": int(item["channel"]),
            "center": servo_limits.center_angle,
            "direction_sign": direction_sign,
            "amplitude_scale": amplitude_scale,
            "left_target": left_target,
            "right_target": right_target,
        }
        tail_motion[servo_id] = TailServoMotion(
            center_deg=servo_limits.center_angle,
            direction_sign=direction_sign,
            amplitude_scale=amplitude_scale,
        )

    centers_by_servo_id = {
        servo_id: limits[servo_id].center_angle
        for servo_id in ALL_SERVO_IDS
    }
    initial_by_servo_id = dict(centers_by_servo_id)
    initial_by_servo_id[4] = roots.servo_4_start_deg
    initial_by_servo_id[6] = roots.servo_6_start_deg
    dive_by_servo_id = dict(centers_by_servo_id)
    dive_by_servo_id[5] = limits[5].center_angle + args.dive_pectoral_tilt
    dive_by_servo_id[7] = limits[7].center_angle - args.dive_pectoral_tilt
    for servo_id in ALL_SERVO_IDS:
        _check_target(
            servo_id,
            (
                f"--dive-pectoral-tilt={args.dive_pectoral_tilt:.3f} dive pose"
                if servo_id in TIP_SERVO_IDS
                else "dive pose"
            ),
            dive_by_servo_id[servo_id],
            limits[servo_id],
        )
    centers = _commands_by_channel(centers_by_servo_id, items)
    initial_pose = _commands_by_channel(initial_by_servo_id, items)
    dive_pose = _commands_by_channel(dive_by_servo_id, items)
    channel_to_servo_id = {
        int(items[servo_id]["channel"]): servo_id
        for servo_id in ALL_SERVO_IDS
    }
    return {
        "items": items,
        "limits": limits,
        "mechanical_ranges": mechanical_ranges,
        "physical_endpoints": physical_endpoints,
        "roots": roots,
        "tips": tips,
        "tip_centers_deg": {servo_id: tips[servo_id]["center"] for servo_id in TIP_SERVO_IDS},
        "tail": tail,
        "tail_motion": tail_motion,
        "centers_by_servo_id": centers_by_servo_id,
        "initial_pose_by_servo_id": initial_by_servo_id,
        "dive_pose_by_servo_id": dive_by_servo_id,
        "centers": centers,
        "initial_pose": initial_pose,
        "dive_pose": dive_pose,
        "moving_channels": sorted(channel_to_servo_id),
        "channel_to_servo_id": channel_to_servo_id,
    }


def _dive_commands(
    elapsed_s: float,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    elapsed = validate_finite_range("dive_elapsed_s", elapsed_s, minimum=0.0)
    logical_tail_offset = asymmetric_tail_offset(
        elapsed,
        args.tail_frequency_hz,
        args.tail_left_amplitude_deg,
        args.tail_right_amplitude_deg,
        args.tail_first_direction,
    )
    by_servo_id = dict(targets["dive_pose_by_servo_id"])
    for servo_id, motion in targets["tail_motion"].items():
        by_servo_id[servo_id] = (
            float(motion.center_deg)
            + float(motion.direction_sign)
            * float(motion.amplitude_scale)
            * logical_tail_offset
        )
    commands = _commands_by_channel(by_servo_id, targets["items"])
    return (
        commands,
        {
            "dive_pectoral_tilt_deg": args.dive_pectoral_tilt,
            "tail_offset_deg": logical_tail_offset,
            "tail_phase_elapsed_s": elapsed,
        },
    )


def _dive_to_roll_commands(
    elapsed_s: float,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    elapsed = validate_finite_range("transition_elapsed_s", elapsed_s, minimum=0.0)
    progress = max(0.0, min(1.0, elapsed / args.transition_s))
    envelope = smoothstep5(progress)
    continued_tail_elapsed = args.dive_duration + elapsed
    unscaled_tail_offset = asymmetric_tail_offset(
        continued_tail_elapsed,
        args.tail_frequency_hz,
        args.tail_left_amplitude_deg,
        args.tail_right_amplitude_deg,
        args.tail_first_direction,
    )
    scaled_tail_offset = (1.0 - envelope) * unscaled_tail_offset

    by_servo_id = dict(targets["dive_pose_by_servo_id"])
    for servo_id, motion in targets["tail_motion"].items():
        by_servo_id[servo_id] = (
            float(motion.center_deg)
            + float(motion.direction_sign)
            * float(motion.amplitude_scale)
            * scaled_tail_offset
        )
    for servo_id in ROOT_SERVO_IDS + TIP_SERVO_IDS:
        start = float(targets["dive_pose_by_servo_id"][servo_id])
        end = float(targets["initial_pose_by_servo_id"][servo_id])
        by_servo_id[servo_id] = start + envelope * (end - start)

    return (
        _commands_by_channel(by_servo_id, targets["items"]),
        {
            "transition_progress": progress,
            "tail_amplitude_scale": 1.0 - envelope,
            "tail_offset_deg": scaled_tail_offset,
            "tail_unscaled_offset_deg": unscaled_tail_offset,
            "tail_phase_elapsed_s": continued_tail_elapsed,
        },
    )


def _roll_commands(
    elapsed_s: float,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    sample = evaluate_roll_trajectory(
        elapsed_s,
        pectoral_frequency_hz=args.pectoral_frequency_hz,
        roots=targets["roots"],
        tip_centers_deg=targets["tip_centers_deg"],
        tip_amplitude_deg=args.tip_amplitude_deg,
        tip_transition_s=args.tip_transition_s,
        tip_direction=args.tip_direction,
        tail_frequency_hz=args.tail_frequency_hz,
        tail_left_amplitude_deg=args.tail_left_amplitude_deg,
        tail_right_amplitude_deg=args.tail_right_amplitude_deg,
        tail_first_direction=args.tail_first_direction,
        tail_servos=targets["tail_motion"],
    )
    commands = _commands_by_channel(
        sample.commands_by_servo_id_deg,
        targets["items"],
    )
    return (
        commands,
        {
            "pectoral_period_s": sample.pectoral_period_s,
            "pectoral_t_cycle_s": sample.pectoral_t_cycle_s,
            "pectoral_phase": sample.pectoral_phase,
            "pectoral_cycle_index": sample.pectoral_cycle_index,
            "root_cosine_q": sample.root_cosine_q,
            "tip_envelope_h": sample.tip_envelope_h,
            "tail_offset_deg": sample.tail_offset_deg,
        },
    )


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
    experiment_start_s: float,
    runtime_state: dict[str, Any],
    timeline: str | None = None,
) -> dict[int, float]:
    duration = max(0.0, float(duration_s))
    command_period_s = 1.0 / args.command_hz
    sample_hz = max(
        0.1,
        float(config.get("runtime", {}).get("synchronized_sample_hz", 30.0)),
    )
    sample_period_s = 1.0 / sample_hz
    print_hz = max(0.0, float(config.get("runtime", {}).get("print_hz", 2.0)))
    print_period_s = 1.0 / print_hz if print_hz > 0.0 else None
    stage_start_s = time.monotonic()
    initial_commands, initial_state = command_builder(0.0)
    controller.write_angles(initial_commands)
    runtime_state["last_commands"] = dict(initial_commands)
    _update_timeline(runtime_state, timeline, 0.0)
    _write_command(
        command_logger,
        args,
        targets,
        action_state=stage_name,
        state_elapsed_s=0.0,
        experiment_elapsed_s=stage_start_s - experiment_start_s,
        commands=initial_commands,
        state=initial_state,
        dive_elapsed_s=0.0 if timeline == "dive" else None,
        roll_elapsed_s=0.0 if timeline == "roll" else None,
    )
    next_command_s = stage_start_s + command_period_s
    next_sample_s = stage_start_s
    next_print_s = stage_start_s

    while True:
        now_s = time.monotonic()
        elapsed_s = now_s - stage_start_s
        if elapsed_s >= duration:
            break
        if now_s >= next_command_s:
            commands, state = command_builder(elapsed_s)
            controller.write_angles(commands)
            runtime_state["last_commands"] = dict(commands)
            _update_timeline(runtime_state, timeline, elapsed_s)
            _write_command(
                command_logger,
                args,
                targets,
                action_state=stage_name,
                state_elapsed_s=elapsed_s,
                experiment_elapsed_s=now_s - experiment_start_s,
                commands=commands,
                state=state,
                dive_elapsed_s=elapsed_s if timeline == "dive" else None,
                roll_elapsed_s=elapsed_s if timeline == "roll" else None,
            )
            next_command_s += command_period_s
            if next_command_s < now_s - command_period_s:
                next_command_s = now_s + command_period_s

        if now_s >= next_sample_s:
            sample = _record_synchronized_sample(synchronizer, sync_logger, raw_loggers)
            if print_period_s is not None and now_s >= next_print_s:
                print(_summary(sample, now_s - experiment_start_s, stage_name))
                next_print_s = now_s + print_period_s
            next_sample_s += sample_period_s
            if next_sample_s < now_s - sample_period_s:
                next_sample_s = now_s + sample_period_s

        next_due_s = min(next_command_s, next_sample_s)
        if next_due_s > now_s:
            time.sleep(min(0.01, next_due_s - now_s))

    # Log and command the exact endpoint so every following stage starts continuously.
    commands, state = command_builder(duration)
    controller.write_angles(commands)
    runtime_state["last_commands"] = dict(commands)
    _update_timeline(runtime_state, timeline, duration)
    _write_command(
        command_logger,
        args,
        targets,
        action_state=stage_name,
        state_elapsed_s=duration,
        experiment_elapsed_s=time.monotonic() - experiment_start_s,
        commands=commands,
        state=state,
        dive_elapsed_s=duration if timeline == "dive" else None,
        roll_elapsed_s=duration if timeline == "roll" else None,
    )
    return dict(commands)


def _update_timeline(
    runtime_state: dict[str, Any],
    timeline: str | None,
    elapsed_s: float,
) -> None:
    if timeline == "dive":
        runtime_state["dive_elapsed_s"] = elapsed_s
    elif timeline == "roll":
        runtime_state["roll_elapsed_s"] = elapsed_s


def _record_synchronized_sample(
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
) -> dict[str, Any]:
    sample = synchronizer.build()
    raw_loggers.write_from_buffers(synchronizer.buffers)
    sync_logger.write(sample)
    return sample


def _execute_tail_ramp_down(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
    targets: dict[str, Any],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
    experiment_start_s: float,
    runtime_state: dict[str, Any],
) -> dict[int, float]:
    start_commands = {
        channel: float(runtime_state["last_commands"].get(channel, center))
        for channel, center in targets["centers"].items()
    }
    roll_elapsed_at_stop = runtime_state.get("roll_elapsed_s")
    continue_tail_phase = roll_elapsed_at_stop is not None
    _write_event(
        event_logger,
        "tail_ramp_down_begin",
        args,
        data={"duration_s": args.tail_ramp_down_s},
    )

    def builder(elapsed: float) -> tuple[dict[int, float], dict[str, Any]]:
        progress = max(0.0, min(1.0, elapsed / args.tail_ramp_down_s))
        amplitude_scale = 1.0 - smoothstep5(progress)
        commands = dict(start_commands)
        if continue_tail_phase:
            continued_elapsed: float | None = float(roll_elapsed_at_stop) + elapsed
            unscaled_offset: float | None = asymmetric_tail_offset(
                continued_elapsed,
                args.tail_frequency_hz,
                args.tail_left_amplitude_deg,
                args.tail_right_amplitude_deg,
                args.tail_first_direction,
            )
            scaled_offset: float | None = amplitude_scale * unscaled_offset
            for values in targets["tail"].values():
                commands[int(values["channel"])] = (
                    values["center"]
                    + values["direction_sign"] * values["amplitude_scale"] * scaled_offset
                )
        else:
            continued_elapsed = None
            unscaled_offset = None
            scaled_offset = None
            for values in targets["tail"].values():
                channel = int(values["channel"])
                commands[channel] = values["center"] + amplitude_scale * (
                    start_commands[channel] - values["center"]
                )
        return (
            commands,
            {
                "transition_progress": progress,
                "tail_amplitude_scale": amplitude_scale,
                "tail_offset_deg": scaled_offset,
                "tail_unscaled_offset_deg": unscaled_offset,
                "tail_phase_elapsed_s": continued_elapsed,
                "tail_phase_continued": continue_tail_phase,
            },
        )

    try:
        commands = _run_stage(
            config=config,
            args=args,
            stage_name="tail_ramp_down",
            duration_s=args.tail_ramp_down_s,
            command_builder=builder,
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            experiment_start_s=experiment_start_s,
            runtime_state=runtime_state,
        )
    except BaseException as exc:
        _write_event(
            event_logger,
            "tail_ramp_down_end",
            args,
            data={"reason": _exception_reason(exc)},
        )
        raise
    _write_event(event_logger, "tail_ramp_down_end", args, data={"reason": "completed"})
    return commands


def _execute_return_to_center(
    config: dict[str, Any],
    args: argparse.Namespace,
    controller: Any,
    targets: dict[str, Any],
    synchronizer: SensorSynchronizer,
    sync_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    command_logger: JsonlLogger,
    event_logger: EventLogger,
    experiment_start_s: float,
    runtime_state: dict[str, Any],
) -> dict[int, float]:
    start_commands = {
        channel: float(runtime_state["last_commands"].get(channel, center))
        for channel, center in targets["centers"].items()
    }
    _write_event(
        event_logger,
        "return_to_center_begin",
        args,
        data={"duration_s": args.return_to_center_s},
    )

    def builder(elapsed: float) -> tuple[dict[int, float], dict[str, Any]]:
        progress = max(0.0, min(1.0, elapsed / args.return_to_center_s))
        return (
            interpolate_angles(start_commands, targets["centers"], progress),
            {"transition_progress": progress, "tail_amplitude_scale": 0.0},
        )

    try:
        commands = _run_stage(
            config=config,
            args=args,
            stage_name="return_to_center",
            duration_s=args.return_to_center_s,
            command_builder=builder,
            controller=controller,
            targets=targets,
            synchronizer=synchronizer,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
            command_logger=command_logger,
            experiment_start_s=experiment_start_s,
            runtime_state=runtime_state,
        )
    except BaseException as exc:
        _write_event(
            event_logger,
            "return_to_center_end",
            args,
            data={"reason": _exception_reason(exc)},
        )
        raise
    _write_event(event_logger, "return_to_center_end", args, data={"reason": "completed"})
    return commands


def _write_command(
    command_logger: JsonlLogger,
    args: argparse.Namespace,
    targets: dict[str, Any],
    *,
    action_state: str,
    state_elapsed_s: float,
    experiment_elapsed_s: float,
    commands: dict[int, float],
    state: dict[str, Any],
    dive_elapsed_s: float | None = None,
    roll_elapsed_s: float | None = None,
) -> None:
    command_logger.write(
        {
            "t_ns": time.monotonic_ns(),
            "action": MISSION_NAME,
            "mission": MISSION_NAME,
            "action_state": action_state,
            "roll_direction": args.roll_direction,
            "tip_direction": args.tip_direction,
            "state_elapsed_s": state_elapsed_s,
            "experiment_elapsed_s": experiment_elapsed_s,
            "dive_elapsed_s": dive_elapsed_s,
            "action_elapsed_s": roll_elapsed_s,
            "roll_elapsed_s": roll_elapsed_s,
            "pectoral_period_s": state.get("pectoral_period_s"),
            "pectoral_t_cycle_s": state.get("pectoral_t_cycle_s"),
            "pectoral_phase": state.get("pectoral_phase"),
            "pectoral_cycle_index": state.get("pectoral_cycle_index"),
            "root_cosine_q": state.get("root_cosine_q"),
            "tip_envelope_h": state.get("tip_envelope_h"),
            "dive_pectoral_tilt_deg": state.get("dive_pectoral_tilt_deg"),
            "tail_offset_deg": state.get("tail_offset_deg"),
            "tail_unscaled_offset_deg": state.get("tail_unscaled_offset_deg"),
            "tail_phase_elapsed_s": state.get("tail_phase_elapsed_s"),
            "tail_amplitude_scale": state.get("tail_amplitude_scale"),
            "transition_progress": state.get("transition_progress"),
            "dry_run": bool(args.dry_run),
            "angle_semantics": "target_servo_angles_not_feedback_measurements",
            "commands_deg": {
                str(channel): angle
                for channel, angle in commands.items()
            },
            "commands_by_servo_id_deg": {
                str(targets["channel_to_servo_id"][channel]): angle
                for channel, angle in commands.items()
            },
        }
    )


def _write_event(
    event_logger: EventLogger,
    event_type: str,
    args: argparse.Namespace,
    *,
    data: dict[str, Any] | None = None,
    t_ns: int | None = None,
) -> bool:
    payload = {
        "action": MISSION_NAME,
        "mission": MISSION_NAME,
        "roll_direction": args.roll_direction,
        "tip_direction": args.tip_direction,
    }
    if data:
        payload.update(data)
    return event_logger.write(event_type, payload, t_ns=t_ns)


def _profile_row(
    *,
    phase: str,
    sequence_time_s: float,
    phase_elapsed_s: float,
    commands: dict[int, float],
    state: dict[str, Any],
    targets: dict[str, Any],
) -> dict[str, Any]:
    by_servo_id = {
        targets["channel_to_servo_id"][channel]: angle
        for channel, angle in commands.items()
    }
    return {
        "phase": phase,
        "sequence_time_s": sequence_time_s,
        "phase_elapsed_s": phase_elapsed_s,
        "tail_offset_deg": state.get("tail_offset_deg"),
        "transition_progress": state.get("transition_progress"),
        "pectoral_phase": state.get("pectoral_phase"),
        "root_cosine_q": state.get("root_cosine_q"),
        "tip_envelope_h": state.get("tip_envelope_h"),
        **{
            f"theta_{servo_id}": by_servo_id[servo_id]
            for servo_id in ALL_SERVO_IDS
        },
    }


def _build_dry_run_profile(
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence_time = 0.0

    transition_steps = max(20, min(200, int(math.ceil(args.transition_s * args.command_hz))))
    for index in range(transition_steps + 1):
        elapsed = args.transition_s * index / transition_steps
        commands = interpolate_angles(
            targets["centers"],
            targets["dive_pose"],
            elapsed / args.transition_s,
        )
        rows.append(
            _profile_row(
                phase="dive_initial_pose",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state={"transition_progress": elapsed / args.transition_s},
                targets=targets,
            )
        )
    sequence_time += args.transition_s

    dive_steps = max(100, min(2000, int(math.ceil(args.dive_duration * args.command_hz))))
    for index in range(dive_steps + 1):
        elapsed = args.dive_duration * index / dive_steps
        commands, state = _dive_commands(elapsed, args, targets)
        rows.append(
            _profile_row(
                phase="dive_action",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state=state,
                targets=targets,
            )
        )
    sequence_time += args.dive_duration

    for index in range(transition_steps + 1):
        elapsed = args.transition_s * index / transition_steps
        commands, state = _dive_to_roll_commands(
            elapsed,
            args,
            targets,
        )
        rows.append(
            _profile_row(
                phase="dive_to_roll_transition",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state=state,
                targets=targets,
            )
        )
    sequence_time += args.transition_s

    hold_steps = max(1, min(200, int(math.ceil(args.initial_hold_s * args.command_hz))))
    hold_indices = range(hold_steps + 1) if args.initial_hold_s > 0.0 else range(1)
    for index in hold_indices:
        elapsed = args.initial_hold_s * index / hold_steps
        rows.append(
            _profile_row(
                phase="initial_hold",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=dict(targets["initial_pose"]),
                state={"hold_elapsed_s": elapsed},
                targets=targets,
            )
        )
    sequence_time += args.initial_hold_s

    pectoral_period = 1.0 / args.pectoral_frequency_hz
    tail_period = 1.0 / args.tail_frequency_hz
    roll_profile_duration = args.duration
    roll_steps = max(
        100,
        min(2000, int(math.ceil(roll_profile_duration * args.command_hz))),
    )
    roll_times = {
        roll_profile_duration * index / roll_steps
        for index in range(roll_steps + 1)
    }
    roll_times.update(
        elapsed
        for elapsed in {
            0.0,
            args.tip_transition_s,
            pectoral_period / 2.0 - args.tip_transition_s,
            pectoral_period / 2.0,
            pectoral_period,
            tail_period / 4.0,
            tail_period / 2.0,
            3.0 * tail_period / 4.0,
            tail_period,
        }
        if 0.0 <= elapsed <= roll_profile_duration
    )
    for elapsed in sorted(roll_times):
        commands, state = _roll_commands(elapsed, args, targets)
        rows.append(
            _profile_row(
                phase="roll_action",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state=state,
                targets=targets,
            )
        )
    sequence_time += args.duration

    roll_end_commands, _ = _roll_commands(args.duration, args, targets)
    ramp_steps = max(
        20,
        min(500, int(math.ceil(args.tail_ramp_down_s * args.command_hz))),
    )
    ramp_end_commands = dict(roll_end_commands)
    for index in range(ramp_steps + 1):
        elapsed = args.tail_ramp_down_s * index / ramp_steps
        progress = elapsed / args.tail_ramp_down_s
        amplitude_scale = 1.0 - smoothstep5(progress)
        continued_elapsed = args.duration + elapsed
        unscaled_offset = asymmetric_tail_offset(
            continued_elapsed,
            args.tail_frequency_hz,
            args.tail_left_amplitude_deg,
            args.tail_right_amplitude_deg,
            args.tail_first_direction,
        )
        scaled_offset = amplitude_scale * unscaled_offset
        commands = dict(roll_end_commands)
        for values in targets["tail"].values():
            commands[int(values["channel"])] = (
                values["center"]
                + values["direction_sign"] * values["amplitude_scale"] * scaled_offset
            )
        ramp_end_commands = commands
        rows.append(
            _profile_row(
                phase="tail_ramp_down",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state={
                    "transition_progress": progress,
                    "tail_amplitude_scale": amplitude_scale,
                    "tail_offset_deg": scaled_offset,
                },
                targets=targets,
            )
        )
    sequence_time += args.tail_ramp_down_s

    return_steps = max(
        20,
        min(500, int(math.ceil(args.return_to_center_s * args.command_hz))),
    )
    for index in range(return_steps + 1):
        elapsed = args.return_to_center_s * index / return_steps
        progress = elapsed / args.return_to_center_s
        commands = interpolate_angles(
            ramp_end_commands,
            targets["centers"],
            progress,
        )
        rows.append(
            _profile_row(
                phase="return_to_center",
                sequence_time_s=sequence_time + elapsed,
                phase_elapsed_s=elapsed,
                commands=commands,
                state={"transition_progress": progress, "tail_amplitude_scale": 0.0},
                targets=targets,
            )
        )
    return rows


def _write_dry_run_profile(
    log_dir: Path,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> Path:
    path = log_dir / DRY_RUN_PROFILE_FILE
    with path.open("w", encoding="utf-8") as handle:
        for row in _build_dry_run_profile(args, targets):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _print_profile_checkpoints(args: argparse.Namespace, targets: dict[str, Any]) -> None:
    start, start_state = _dive_commands(0.0, args, targets)
    end, end_state = _dive_commands(args.dive_duration, args, targets)
    print(
        f"Dry-run dive profile: duration={args.dive_duration:.6f}s, "
        f"tip5=center+{args.dive_pectoral_tilt:.3f}, "
        f"tip7=center-{args.dive_pectoral_tilt:.3f}"
    )
    print("phase time_s theta_4 theta_5 theta_6 theta_7 tail_offset theta_1 theta_2 theta_3")
    for label, elapsed, commands, state in (
        ("start", 0.0, start, start_state),
        ("end", args.dive_duration, end, end_state),
    ):
        by_servo_id = {
            targets["channel_to_servo_id"][channel]: angle
            for channel, angle in commands.items()
        }
        print(
            f"{label} {elapsed:.6f} {by_servo_id[4]:.3f} {by_servo_id[5]:.3f} "
            f"{by_servo_id[6]:.3f} {by_servo_id[7]:.3f} "
            f"{float(state['tail_offset_deg']):.3f} {by_servo_id[1]:.3f} "
            f"{by_servo_id[2]:.3f} {by_servo_id[3]:.3f}"
        )
    pectoral_period = 1.0 / args.pectoral_frequency_hz
    tail_period = 1.0 / args.tail_frequency_hz
    checkpoints = sorted(
        {
            0.0,
            args.tip_transition_s,
            pectoral_period / 2.0 - args.tip_transition_s,
            pectoral_period / 2.0,
            pectoral_period,
            tail_period / 4.0,
            tail_period / 2.0,
            3.0 * tail_period / 4.0,
            tail_period,
        }
    )
    print(
        f"Dry-run roll profile: roll_direction={args.roll_direction}, "
        f"tip_direction={args.tip_direction}, T={pectoral_period:.6f}s"
    )
    print("time_s theta_4 theta_5 theta_6 theta_7 tail_offset theta_1 theta_2 theta_3")
    for elapsed in checkpoints:
        commands, state = _roll_commands(elapsed, args, targets)
        by_servo_id = {
            targets["channel_to_servo_id"][channel]: angle
            for channel, angle in commands.items()
        }
        print(
            f"{elapsed:.6f} {by_servo_id[4]:.3f} {by_servo_id[5]:.3f} "
            f"{by_servo_id[6]:.3f} {by_servo_id[7]:.3f} "
            f"{float(state['tail_offset_deg']):.3f} {by_servo_id[1]:.3f} "
            f"{by_servo_id[2]:.3f} {by_servo_id[3]:.3f}"
        )


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
    roots: RootPairTargets = targets["roots"]
    metadata = {
        "started_wall_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "mission",
        "mission": MISSION_NAME,
        "action": MISSION_NAME,
        "open_loop": True,
        "imu_feedback_control": False,
        "depth_feedback_control": False,
        "dry_run": bool(args.dry_run),
        "mock_sensors": bool(args.mock_sensors),
        "keep_pwm": bool(args.keep_pwm),
        "phase_order": [
            "dive_initial_pose",
            "dive_action",
            "dive_to_roll_transition",
            "initial_hold",
            "roll_action",
            "tail_ramp_down",
            "return_to_center",
        ],
        "dive_duration_s": args.dive_duration,
        "dive_pectoral_tilt_deg": args.dive_pectoral_tilt,
        "transition_s": args.transition_s,
        "roll_direction": args.roll_direction,
        "tip_direction": args.tip_direction,
        "roll_duration_s": args.duration,
        "pectoral_frequency_hz": args.pectoral_frequency_hz,
        "root_amplitude_ratio": args.root_amplitude_ratio,
        "tip_amplitude_deg": args.tip_amplitude_deg,
        "tip_transition_s": args.tip_transition_s,
        "tail_frequency_hz": args.tail_frequency_hz,
        "tail_left_amplitude_deg": args.tail_left_amplitude_deg,
        "tail_right_amplitude_deg": args.tail_right_amplitude_deg,
        "tail_first_direction": args.tail_first_direction,
        "start_delay_s": args.start_delay_s,
        "initial_hold_s": args.initial_hold_s,
        "tail_ramp_down_s": args.tail_ramp_down_s,
        "return_to_center_s": args.return_to_center_s,
        "command_hz": args.command_hz,
        "dive_pose_deg": {
            str(servo_id): angle
            for servo_id, angle in targets["dive_pose_by_servo_id"].items()
        },
        "physical_endpoints_deg": targets["physical_endpoints"],
        "root_motion_deg": {
            "servo_4_start_deg": roots.servo_4_start_deg,
            "servo_4_target_deg": roots.servo_4_target_deg,
            "servo_6_start_deg": roots.servo_6_start_deg,
            "servo_6_target_deg": roots.servo_6_target_deg,
        },
        "roll_tip_targets_deg": {
            str(servo_id): values["target"]
            for servo_id, values in targets["tips"].items()
        },
        "tail_extrema_deg": {
            str(servo_id): {
                "left": values["left_target"],
                "right": values["right_target"],
            }
            for servo_id, values in targets["tail"].items()
        },
        "dry_run_profile_file": DRY_RUN_PROFILE_FILE if args.dry_run else None,
        "servo_ids": list(ALL_SERVO_IDS),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "servo": config.get("servo", {}),
        "notes": [
            "This is open-loop dive and roll motion; IMU and depth are recorded but never fed back.",
            "Dive holds roots 4/6 at center, servo 5 at center+tilt, and servo 7 at center-tilt.",
            "Dive and roll reset their tail clocks independently, so both start at tail center.",
            "The dive-to-roll transition uses quintic smoothstep and reaches the exact roll phase-zero pose.",
            "All targets use configured per-servo min_angle/max_angle inclusively; no global test-amplitude cap or extra margin is applied.",
            "All event, command, raw-sensor, and synchronized timestamps use monotonic time.",
            "commands.jsonl contains targets, not position-feedback measurements.",
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


def _exception_reason(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "ctrl_c"
    return f"{type(exc).__name__}: {exc}"


def _cleanup_step(label: str, callback: Callable[[], Any]) -> bool:
    """Run one shutdown operation without preventing later safety cleanup."""
    try:
        callback()
    except BaseException as exc:
        print(f"Cleanup step failed ({label}): {_exception_reason(exc)}")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
