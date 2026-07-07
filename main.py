from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from control.runtime.data_logger import (
    JsonlLogger,
    RawSensorLoggers,
    create_run_log_dir,
    prepare_raw_placeholders,
    write_metadata_yaml,
)
from control.runtime.event_logger import EventLogger
from control.runtime.link_manager import link_manager_from_config
from control.runtime.sensor_manager import SENSOR_NAMES, SensorManager
from control.runtime.sensor_synchronizer import SensorSynchronizer
from control.safety import ensure_not_windows_hardware_run, load_robot_config


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "robot.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fishbot sensor collection and local logging entrypoint.")
    parser.add_argument("--mode", choices=("status", "observe", "record"), default="status")
    parser.add_argument("--mock", action="store_true", help="Generate fake sensor samples without touching hardware.")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds. Record may run until Ctrl+C.")
    parser.add_argument("--status-seconds", type=float, default=5.0, help="Duration for --mode status.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _runtime_config(load_robot_config(CONFIG_PATH), args)

    if not args.mock:
        ensure_not_windows_hardware_run()

    if args.mode == "status":
        run_status(config, args)
    elif args.mode == "observe":
        run_observe(config, args)
    elif args.mode == "record":
        run_record(config, args)
    return 0


def run_status(config: dict[str, Any], args: argparse.Namespace) -> None:
    manager = SensorManager(config, mock=args.mock)
    manager.start_all()
    print("Sensor status mode started. No servos, gait, or PCA9685 output are used.")
    deadline_s = time.monotonic() + max(0.1, float(args.status_seconds))
    try:
        while time.monotonic() < deadline_s:
            print_status(manager.get_status())
            time.sleep(0.5)
    finally:
        manager.stop_all()
        print("Sensor status mode stopped.")


def run_observe(config: dict[str, Any], args: argparse.Namespace) -> None:
    log_dir, event_logger = _start_event_logging(config, args, "observe")
    manager = SensorManager(config, mock=args.mock, log_dir=log_dir)
    synchronizer = SensorSynchronizer(manager.buffers, config)
    link_manager = link_manager_from_config(config, event_logger) if event_logger else None

    manager.start_all()
    if event_logger:
        event_logger.write("logger_started", {"mode": "observe", "log_dir": str(log_dir)})
    if link_manager:
        link_manager.start()

    print("Sensor observe mode started. No servos, gait, or PCA9685 output are used.")
    try:
        _run_sample_loop(
            config,
            synchronizer,
            duration_s=args.duration,
            print_enabled=True,
            sync_logger=None,
        )
    except KeyboardInterrupt:
        if event_logger:
            event_logger.write("ctrl_c_received", {"mode": "observe"})
    finally:
        if link_manager:
            link_manager.stop()
        manager.stop_all()
        if event_logger:
            event_logger.write("logger_stopped", {"mode": "observe"})
            event_logger.stop()
        print("Sensor observe mode stopped.")


def run_record(config: dict[str, Any], args: argparse.Namespace) -> None:
    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    base_dir = _resolve_log_base_dir(config)
    log_dir = create_run_log_dir(base_dir, suffix="sensor_test")
    prepare_raw_placeholders(log_dir)
    _write_metadata(config, args, log_dir, "record")

    flush_interval_s = float(logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0)))
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
    manager = SensorManager(config, mock=args.mock, log_dir=log_dir)
    synchronizer = SensorSynchronizer(manager.buffers, config)
    link_manager = link_manager_from_config(config, event_logger)

    sync_logger.start()
    raw_loggers.start()
    event_logger.start()
    event_logger.write("logger_started", {"mode": "record", "log_dir": str(log_dir)})
    event_logger.write("record_started", {"duration_s": args.duration, "mock": args.mock})
    manager.start_all()
    link_manager.start()

    print(f"Sensor record mode started. Writing logs to: {log_dir}")
    print("No servos, gait, or PCA9685 output are used.")
    try:
        _run_sample_loop(
            config,
            synchronizer,
            duration_s=args.duration,
            print_enabled=True,
            sync_logger=sync_logger,
            raw_loggers=raw_loggers,
        )
        event_logger.write("record_finished", {"reason": "duration_elapsed"})
    except KeyboardInterrupt:
        event_logger.write("ctrl_c_received", {"mode": "record"})
        event_logger.write("record_finished", {"reason": "ctrl_c"})
    finally:
        link_manager.stop()
        raw_loggers.write_from_buffers(manager.buffers)
        manager.stop_all()
        raw_loggers.write_from_buffers(manager.buffers)
        event_logger.write(
            "logger_stopped",
            {
                "mode": "record",
                "sync_dropped_count": sync_logger.dropped_count,
                "sync_last_error": sync_logger.last_error,
                "raw_logger_stats": raw_loggers.stats(),
            },
        )
        sync_logger.stop()
        raw_loggers.stop()
        event_logger.stop()
        print(f"Sensor record mode stopped. Logs are in: {log_dir}")


def _run_sample_loop(
    config: dict[str, Any],
    synchronizer: SensorSynchronizer,
    *,
    duration_s: float | None,
    print_enabled: bool,
    sync_logger: JsonlLogger | None,
    raw_loggers: RawSensorLoggers | None = None,
) -> None:
    runtime_cfg = config.get("runtime", {})
    sample_hz = max(0.1, float(runtime_cfg.get("synchronized_sample_hz", 30)))
    print_hz = max(0.0, float(runtime_cfg.get("print_hz", 2)))
    period_s = 1.0 / sample_hz
    print_period_s = 1.0 / print_hz if print_hz > 0 else None
    start_s = time.monotonic()
    next_sample_s = start_s
    next_print_s = start_s

    while True:
        now_s = time.monotonic()
        if duration_s is not None and now_s - start_s >= duration_s:
            break

        if now_s < next_sample_s:
            time.sleep(min(0.05, next_sample_s - now_s))
            continue

        sample = synchronizer.build()
        if raw_loggers is not None:
            raw_loggers.write_from_buffers(synchronizer.buffers)
        if sync_logger is not None:
            sync_logger.write(sample)

        if print_enabled and print_period_s is not None and now_s >= next_print_s:
            print(sample_summary(sample))
            next_print_s = now_s + print_period_s

        next_sample_s += period_s
        if next_sample_s < now_s - period_s:
            next_sample_s = now_s + period_s


def print_status(status: dict[str, Any]) -> None:
    parts = []
    for name in SENSOR_NAMES:
        item = status.get(name, {})
        age = item.get("age_ms")
        age_text = "none" if age is None else f"{age:.1f}ms"
        valid = "ok" if item.get("valid") else "bad"
        error = item.get("error") or "-"
        parts.append(f"{name}={valid} age={age_text} err={error} n={item.get('count', 0)}")
    print(" | ".join(parts))


def sample_summary(sample: dict[str, Any]) -> str:
    depth = sample.get("depth", {})
    power = sample.get("power", {})
    uwb = sample.get("uwb", {})
    vision = sample.get("vision", {})
    status = sample.get("status", {})
    depth_text = "-"
    if depth.get("depth_m") is not None:
        depth_text = f"{depth.get('depth_m'):.3f}m"
    voltage_text = "-"
    if power.get("voltage_v") is not None:
        voltage_text = f"{power.get('voltage_v'):.2f}V"
    return (
        f"t={sample.get('t_s', 0.0):.1f}s "
        f"imu={'ok' if sample.get('imu', {}).get('valid') else 'bad'} "
        f"depth={depth_text} "
        f"power={voltage_text} "
        f"uwb={'ok' if uwb.get('valid') else uwb.get('error', 'bad')} "
        f"vision={'ok' if vision.get('valid') else vision.get('error', 'bad')} "
        f"missing={status.get('missing_sensors', [])}"
    )


def _start_event_logging(
    config: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
) -> tuple[Path | None, EventLogger | None]:
    logging_cfg = config.get("logging", {})
    if not bool(logging_cfg.get("enabled", True)):
        return None, None
    log_dir = create_run_log_dir(_resolve_log_base_dir(config), suffix="sensor_test")
    _write_metadata(config, args, log_dir, mode)
    flush_interval_s = float(logging_cfg.get("flush_interval_s", 1.0))
    queue_maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))
    event_logger = EventLogger(
        log_dir / str(logging_cfg.get("events_file", "events.jsonl")),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    event_logger.start()
    return log_dir, event_logger


def _resolve_log_base_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    base_dir = Path(str(logging_cfg.get("base_dir", runtime_cfg.get("log_dir", "logs"))))
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    return base_dir


def _write_metadata(config: dict[str, Any], args: argparse.Namespace, log_dir: Path, mode: str) -> None:
    logging_cfg = config.get("logging", {})
    metadata_file = str(logging_cfg.get("metadata_file", "metadata.yaml"))
    metadata = {
        "started_wall_time": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "mock": bool(args.mock),
        "duration_s": args.duration,
        "project_root": str(PROJECT_ROOT),
        "robot": config.get("robot", {}),
        "runtime": config.get("runtime", {}),
        "sensors": config.get("sensors", {}),
        "vision": _vision_metadata(config),
        "safety": config.get("safety", {}),
        "notes": [
            "This stage records sensors only.",
            "No RL policy, gait, PCA9685, or servo output is started by this entrypoint.",
        ],
    }
    write_metadata_yaml(log_dir / metadata_file, metadata)


def _vision_metadata(config: dict[str, Any]) -> dict[str, Any]:
    vision = config.get("sensors", {}).get("vision", {})
    keys = (
        "enabled",
        "save_frames",
        "save_fps",
        "image_format",
        "jpeg_quality",
        "camera_index",
        "split_stereo",
        "save_combined_frame",
    )
    return {key: vision.get(key) for key in keys}


def _runtime_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = copy.deepcopy(config)
    runtime = runtime_config.setdefault("runtime", {})
    runtime["mode"] = args.mode
    runtime["mock"] = bool(args.mock)

    if args.mock:
        sensors = runtime_config.setdefault("sensors", {})
        for name in SENSOR_NAMES:
            sensor_cfg = sensors.setdefault(name, {})
            sensor_cfg["enabled"] = True
    return runtime_config


if __name__ == "__main__":
    raise SystemExit(main())
