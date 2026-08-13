from __future__ import annotations

import argparse
import math
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root


PROJECT_ROOT = add_project_root()

from control.runtime.action_scheduler import ActionScheduler  # noqa: E402
from control.runtime.data_logger import (  # noqa: E402
    JsonlLogger,
    RawSensorLoggers,
    create_run_log_dir,
    prepare_raw_placeholders,
    write_metadata_yaml,
)
from control.runtime.discrete_actions import (  # noqa: E402
    MissionValidationError,
    build_robot_calibration,
    validate_discrete_mission,
)
from control.runtime.event_logger import EventLogger  # noqa: E402
from control.runtime.manual_action_provider import ManualSequenceProvider  # noqa: E402
from control.runtime.sensor_manager import SensorManager  # noqa: E402
from control.runtime.sensor_sync_worker import SensorSyncWorker  # noqa: E402
from control.runtime.sensor_synchronizer import SensorSynchronizer  # noqa: E402
from control.runtime.servo_executor import (  # noqa: E402
    ServoExecutor,
    execution_config_from_robot,
)
from control.runtime.servo_state_tracker import ESTIMATION_MODE, ServoStateTracker  # noqa: E402
from control.safety import ensure_not_windows_hardware_run, load_robot_config  # noqa: E402


SCRIPT_VERSION = "20260716-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析人工任务路径和执行安全参数，不暴露旧周期运动参数。"""

    parser = argparse.ArgumentParser(description="运行三路并行的人工离散动作序列。")
    parser.add_argument("--mission", required=True, help="人工动作 mission YAML 路径。")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "robot.yaml"),
        help="机器人配置路径。",
    )
    parser.add_argument("--confirm", default="", help="真实舵机动作必须为 MOVE。")
    parser.add_argument("--dry-run", action="store_true", help="不导入或初始化真实 PCA9685。")
    parser.add_argument(
        "--mock-sensors",
        action="store_true",
        help="只模拟传感器；除非同时使用 --dry-run，否则舵机仍会真实运动。",
    )
    parser.add_argument("--start-delay-s", type=float, default=20.0, help="本地确定性启动倒计时。")
    parser.add_argument("--keep-pwm", action="store_true", help="成功回中后保持 PWM。")
    parser.add_argument(
        "--command-hz",
        type=float,
        default=None,
        help="覆盖 servo.discrete_executor.command_hz；不改变动作数学定义。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    boot_events: list[tuple[int, str, dict[str, Any]]] = []
    try:
        _validate_start_delay(args.start_delay_s)
        config = load_robot_config(args.config)
        provider = ManualSequenceProvider(_resolve_project_path(args.mission))
        mission = provider.load()
        loaded_t_ns = time.monotonic_ns()
        boot_events.append(
            (
                loaded_t_ns,
                "mission_loaded",
                {"mission_name": mission.name, "mission": mission.to_dict()},
            )
        )
        calibration = build_robot_calibration(config)
        validate_discrete_mission(mission, calibration)
        execution_config = execution_config_from_robot(
            config,
            command_hz_override=args.command_hz,
        )
        validated_t_ns = time.monotonic_ns()
        boot_events.append(
            (
                validated_t_ns,
                "mission_validated",
                {
                    "mission_name": mission.name,
                    "servo_ids": sorted(calibration.servos),
                    "validation_rule": "robot_yaml_min_max_only",
                },
            )
        )
    except (MissionValidationError, ValueError) as exc:
        print(f"离散动作任务验证失败：{exc}")
        return 2

    if not args.dry_run and args.confirm != "MOVE":
        print("拒绝真实舵机动作：必须显式提供 --confirm MOVE；无 PWM 验证请使用 --dry-run。")
        return 2
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()

    countdown_start_delay(args.start_delay_s)
    log_dir = create_run_log_dir(
        _resolve_log_base_dir(config),
        suffix=f"discrete_{_safe_suffix(mission.name)}",
    )
    prepare_raw_placeholders(log_dir)
    _write_metadata(config, mission.to_dict(), execution_config.to_dict(), args, log_dir)

    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    flush_interval_s = float(
        logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0))
    )
    queue_maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))
    shutdown_event = threading.Event()
    motion_stop_event = threading.Event()
    failure_queue: queue.Queue[dict[str, Any]] = queue.Queue()

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
    tracker = ServoStateTracker(calibration)
    scheduler = ActionScheduler(mission)
    sensor_manager = SensorManager(
        config,
        mock=args.mock_sensors,
        log_dir=log_dir,
        stop_event=shutdown_event,
    )
    synchronizer = SensorSynchronizer(sensor_manager.buffers, config)
    sync_worker = SensorSyncWorker(
        synchronizer=synchronizer,
        tracker=tracker,
        sync_logger=sync_logger,
        raw_loggers=raw_loggers,
        sample_hz=float(runtime_cfg.get("synchronized_sample_hz", 30.0)),
        shutdown_event=shutdown_event,
        failure_queue=failure_queue,
    )
    executor = ServoExecutor(
        config=config,
        calibration=calibration,
        scheduler=scheduler,
        tracker=tracker,
        command_logger=command_logger,
        event_logger=event_logger,
        execution_config=execution_config,
        dry_run=args.dry_run,
        keep_pwm=args.keep_pwm,
        shutdown_event=shutdown_event,
        motion_stop_event=motion_stop_event,
        failure_queue=failure_queue,
    )

    for logger in (sync_logger, command_logger):
        logger.start()
    raw_loggers.start()
    event_logger.start()
    for t_ns, event_type, data in boot_events:
        event_logger.write(event_type, data, t_ns=t_ns)
    sensor_manager.start_all()
    sync_worker.start()
    executor.start()

    print(f"日志和传感器已启动：{log_dir}")
    if args.dry_run:
        print("Dry-run：未导入或初始化真实 PCA9685，不会发送 PWM。")
    print("角度语义：无硬件位置反馈，estimated_angle_deg 仅为假设完全跟随的模型估计。")

    ctrl_c = False
    reported_failures: list[dict[str, Any]] = []
    seen_sensor_failures: set[tuple[str, str]] = set()
    try:
        while executor.is_alive():
            try:
                failure = failure_queue.get(timeout=0.1)
            except queue.Empty:
                failure = None
            if failure is not None:
                reported_failures.append(failure)
                if failure.get("source") != "ServoExecutor":
                    motion_stop_event.set()
            for failure in sensor_manager.fatal_errors():
                key = (failure["source"], failure["error"])
                if key not in seen_sensor_failures:
                    seen_sensor_failures.add(key)
                    reported_failures.append({**failure, "t_ns": time.monotonic_ns()})
                    motion_stop_event.set()
    except KeyboardInterrupt:
        ctrl_c = True
        motion_stop_event.set()
        print("\n收到 Ctrl+C，已停止动作推进，正在由 ServoExecutor 尝试安全回中。")
    finally:
        ctrl_c = _cleanup_step(
            "join ServoExecutor",
            lambda: executor.join(execution_config.join_timeout_s),
            reported_failures,
        ) or ctrl_c
        if executor.is_alive():
            motion_stop_event.set()
            ctrl_c = _cleanup_step(
                "retry join ServoExecutor",
                lambda: executor.join(execution_config.join_timeout_s),
                reported_failures,
            ) or ctrl_c
        executor_stuck = executor.is_alive()
        if executor_stuck:
            reported_failures.append(
                {
                    "source": "ServoExecutor",
                    "error": "join_timeout",
                    "t_ns": time.monotonic_ns(),
                }
            )

        # 先搬运最后一批 raw 样本和排队停止事件，再通知生产者退出。
        ctrl_c = _cleanup_step(
            "move final raw samples",
            lambda: raw_loggers.write_from_buffers(sensor_manager.buffers),
            reported_failures,
        ) or ctrl_c
        ctrl_c = _cleanup_step(
            "enqueue logger_stopped",
            lambda: event_logger.write(
                "logger_stopped",
                {
                    "reason": executor.result.reason,
                    "ctrl_c": ctrl_c,
                    "reported_background_failures": reported_failures,
                    "sync_dropped_count": sync_logger.dropped_count,
                    "command_dropped_count": command_logger.dropped_count,
                    "sync_last_error": sync_logger.last_error,
                    "command_last_error": command_logger.last_error,
                    "raw_logger_stats": raw_loggers.stats(),
                },
                t_ns=time.monotonic_ns(),
            ),
            reported_failures,
        ) or ctrl_c
        shutdown_event.set()
        for label, callback in (
            (
                "stop sensor workers",
                lambda: sensor_manager.stop_all(execution_config.join_timeout_s),
            ),
            (
                "join SensorSyncWorker",
                lambda: sync_worker.join(execution_config.join_timeout_s),
            ),
            (
                "move shutdown raw samples",
                lambda: raw_loggers.write_from_buffers(sensor_manager.buffers),
            ),
            (
                "stop synchronized logger",
                lambda: sync_logger.stop(execution_config.join_timeout_s),
            ),
            ("stop raw loggers", raw_loggers.stop),
            (
                "stop command logger",
                lambda: command_logger.stop(execution_config.join_timeout_s),
            ),
            ("stop event logger", event_logger.stop),
        ):
            ctrl_c = _cleanup_step(label, callback, reported_failures) or ctrl_c

    summary = {
        "mission": mission.name,
        "reason": executor.result.reason,
        "mission_finished": executor.result.mission_finished,
        "safe_recentered": executor.result.safe_recentered,
        "pwm_released": executor.result.pwm_released,
        "skipped_servo_ticks": executor.result.skipped_tick_count,
        "skipped_sync_ticks": sync_worker.skipped_tick_count,
        "background_failures": reported_failures,
        "alive_sensor_workers": sensor_manager.alive_worker_names(),
        "log_dir": str(log_dir),
    }
    print(f"实验摘要：{summary}")
    if ctrl_c:
        return 130
    if executor.is_alive() or not executor.result.safe_recentered:
        return 1
    return 0 if executor.result.mission_finished else 1


def countdown_start_delay(start_delay_s: float) -> None:
    """使用 monotonic_ns 的本地倒计时，不访问网络、日志、传感器或 PCA9685。"""

    duration_ns = int(round(max(0.0, float(start_delay_s)) * 1_000_000_000))
    if duration_ns <= 0:
        return
    deadline_ns = time.monotonic_ns() + duration_ns
    last_printed: int | None = None
    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        remaining_s = max(1, int(math.ceil(remaining_ns / 1_000_000_000.0)))
        if remaining_s != last_printed:
            print(f"Action starts in {remaining_s} s")
            last_printed = remaining_s
        time.sleep(min(0.1, remaining_ns / 1_000_000_000.0))


def _validate_start_delay(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise MissionValidationError("--start-delay-s 必须是有限非负数。")


def _write_metadata(
    config: dict[str, Any],
    mission: dict[str, Any],
    execution_config: dict[str, Any],
    args: argparse.Namespace,
    log_dir: Path,
) -> None:
    metadata_file = str(config.get("logging", {}).get("metadata_file", "metadata.yaml"))
    write_metadata_yaml(
        log_dir / metadata_file,
        {
            "mode": "discrete_action_sequence",
            "mission_name": mission["name"],
            "script_version": SCRIPT_VERSION,
            "manual_action_sequence": mission,
            "robot_yaml_snapshot": config,
            "feedback_available": False,
            "position_estimation_mode": ESTIMATION_MODE,
            "position_note": "该角度没有硬件反馈，只是模型估计值。",
            "execution_config": execution_config,
            "dry_run": bool(args.dry_run),
            "mock_sensors": bool(args.mock_sensors),
            "start_delay_s": args.start_delay_s,
            "keep_pwm": bool(args.keep_pwm),
            "safety_policy": {
                "angle_limits": "per_servo_robot_yaml_min_angle_max_angle_only",
                "clamp": False,
                "automatic_amplitude_reduction": False,
                "additional_amplitude_limit": False,
            },
        },
    )


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_log_base_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config.get("logging", {})
    runtime_cfg = config.get("runtime", {})
    path = Path(str(logging_cfg.get("base_dir", runtime_cfg.get("log_dir", "logs"))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_suffix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "mission"


def _cleanup_step(
    label: str,
    callback: Any,
    failures: list[dict[str, Any]],
) -> bool:
    """清理阶段吞掉重复 Ctrl+C 和后台错误，保证后续资源仍会关闭。"""

    try:
        callback()
        return False
    except BaseException as exc:
        failures.append(
            {
                "source": f"cleanup:{label}",
                "error": f"{type(exc).__name__}: {exc}",
                "t_ns": time.monotonic_ns(),
            }
        )
        return isinstance(exc, KeyboardInterrupt)


if __name__ == "__main__":
    raise SystemExit(main())
