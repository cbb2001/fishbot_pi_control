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

from control.runtime.absolute_action_scheduler import AbsoluteActionScheduler  # noqa: E402
from control.runtime.absolute_discrete_actions import (  # noqa: E402
    MissionValidationError, build_absolute_calibration, validate_absolute_mission,
)
from control.runtime.absolute_manual_action_provider import ManualSequenceProvider  # noqa: E402
from control.runtime.absolute_servo_executor import (  # noqa: E402
    ServoExecutor, execution_config_from_robot,
)
from control.runtime.absolute_servo_state_tracker import (  # noqa: E402
    ESTIMATION_MODE, ServoStateTracker,
)
from control.runtime.data_logger import (  # noqa: E402
    JsonlLogger, RawSensorLoggers, create_run_log_dir, prepare_raw_placeholders,
    write_metadata_yaml,
)
from control.runtime.event_logger import EventLogger  # noqa: E402
from control.runtime.sensor_manager import SensorManager  # noqa: E402
from control.runtime.sensor_sync_worker import SensorSyncWorker  # noqa: E402
from control.runtime.sensor_synchronizer import SensorSynchronizer  # noqa: E402
from control.safety import ensure_not_windows_hardware_run, load_robot_config  # noqa: E402


SCRIPT_VERSION = "20260717-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """只暴露绝对动作入口所需的执行参数，不接受旧周期/振幅参数。"""

    parser = argparse.ArgumentParser(description="运行绝对目标角度和名义速度离散动作序列。")
    parser.add_argument("--mission", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "robot.yaml"))
    parser.add_argument("--confirm", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-sensors", action="store_true",
                        help="只模拟传感器；不代表舵机 dry-run。")
    parser.add_argument("--start-delay-s", type=float, default=20.0)
    parser.add_argument("--keep-pwm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # 以下读取和完整模拟预检均发生在确认、倒计时、日志、传感器和硬件之前。
    args = parse_args(argv)
    boot_events: list[tuple[int, str, dict[str, Any]]] = []
    try:
        _validate_delay(args.start_delay_s)
        config = load_robot_config(args.config)
        mission = ManualSequenceProvider(_project_path(args.mission)).load()
        loaded_t_ns = time.monotonic_ns()
        boot_events.append((loaded_t_ns, "mission_loaded", {"mission_name": mission.name,
                                                             "mission": mission.to_dict()}))
        calibration = build_absolute_calibration(config)
        validate_absolute_mission(mission, calibration)
        execution_config = execution_config_from_robot(config)
        boot_events.append((time.monotonic_ns(), "mission_validated", {
            "mission_name": mission.name, "validation_rule": "robot_yaml_min_max_only",
            "tail_max_speed_deg_s": calibration.tail_max_speed_deg_s,
            "fin_max_speed_deg_s": calibration.fin_max_speed_deg_s}))
    except (MissionValidationError, ValueError) as exc:
        print(f"绝对离散动作任务验证失败：{exc}")
        return 2
    if not args.dry_run and args.confirm != "MOVE":
        print("拒绝真实舵机动作：必须显式提供 --confirm MOVE；无 PWM 验证请使用 --dry-run。")
        return 2
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()

    countdown_start_delay(args.start_delay_s)
    log_dir = create_run_log_dir(_log_base(config), suffix=f"absolute_{_safe_suffix(mission.name)}")
    prepare_raw_placeholders(log_dir)
    _write_metadata(config, mission.to_dict(), calibration, execution_config.to_dict(), args, log_dir)
    logging_cfg, runtime_cfg = config.get("logging", {}), config.get("runtime", {})
    flush = float(logging_cfg.get("flush_interval_s", runtime_cfg.get("flush_interval_s", 1.0)))
    maxsize = int(logging_cfg.get("write_queue_maxsize", 10000))
    shutdown_event, motion_stop_event = threading.Event(), threading.Event()
    failures: queue.Queue[dict[str, Any]] = queue.Queue()
    sync_logger = JsonlLogger(log_dir / str(logging_cfg.get("synchronized_file", "synchronized_sensors.jsonl")),
                              flush_interval_s=flush, queue_maxsize=maxsize)
    command_logger = JsonlLogger(log_dir / "commands.jsonl", flush_interval_s=flush,
                                 queue_maxsize=maxsize)
    raw_loggers = RawSensorLoggers(log_dir, flush_interval_s=flush, queue_maxsize=maxsize)
    event_logger = EventLogger(log_dir / str(logging_cfg.get("events_file", "events.jsonl")),
                               flush_interval_s=flush, queue_maxsize=maxsize)
    tracker = ServoStateTracker(calibration)
    scheduler = AbsoluteActionScheduler(mission, calibration)
    sensor_manager = SensorManager(config, mock=args.mock_sensors, log_dir=log_dir,
                                   stop_event=shutdown_event)
    synchronizer = SensorSynchronizer(sensor_manager.buffers, config)
    sync_worker = SensorSyncWorker(synchronizer=synchronizer, tracker=tracker,
        sync_logger=sync_logger, raw_loggers=raw_loggers,
        sample_hz=float(runtime_cfg.get("synchronized_sample_hz", 30)),
        shutdown_event=shutdown_event, failure_queue=failures)
    executor = ServoExecutor(config=config, calibration=calibration, scheduler=scheduler,
        tracker=tracker, command_logger=command_logger, event_logger=event_logger,
        execution_config=execution_config, dry_run=args.dry_run, keep_pwm=args.keep_pwm,
        shutdown_event=shutdown_event, motion_stop_event=motion_stop_event,
        failure_queue=failures)

    for logger in (sync_logger, command_logger):
        logger.start()
    raw_loggers.start()
    event_logger.start()
    for t_ns, event, data in boot_events:
        event_logger.write(event, data, t_ns=t_ns)
    sensor_manager.start_all()
    sync_worker.start()
    executor.start()
    print(f"日志和传感器已启动：{log_dir}")
    if args.dry_run:
        print("Dry-run：未导入或初始化真实 PCA9685，不会发送 PWM。")
    print("舵机无位置反馈：estimated 仅为 assumed_perfect_tracking 模型估计。")

    ctrl_c = False
    reported: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    try:
        while executor.is_alive():
            try:
                failure = failures.get(timeout=0.1)
            except queue.Empty:
                failure = None
            if failure:
                reported.append(failure)
                if failure.get("source") != "ServoExecutor":
                    motion_stop_event.set()
            for failure in sensor_manager.fatal_errors():
                key = (failure["source"], failure["error"])
                if key not in seen:
                    seen.add(key)
                    reported.append({**failure, "t_ns": time.monotonic_ns()})
                    motion_stop_event.set()
    except KeyboardInterrupt:
        ctrl_c = True
        motion_stop_event.set()
        print("\n收到 Ctrl+C，已停止动作推进，正在尝试安全回中。")
    finally:
        executor.join(execution_config.join_timeout_s)
        if executor.is_alive():
            motion_stop_event.set()
            executor.join(execution_config.join_timeout_s)
        try:
            raw_loggers.write_from_buffers(sensor_manager.buffers)
            event_logger.write("logger_stopped", {"reason": executor.result.reason,
                "ctrl_c": ctrl_c, "reported_background_failures": reported,
                "sync_dropped_count": sync_logger.dropped_count,
                "command_dropped_count": command_logger.dropped_count}, t_ns=time.monotonic_ns())
        except BaseException as exc:
            reported.append({"source": "cleanup:final_log", "error": repr(exc),
                             "t_ns": time.monotonic_ns()})
        shutdown_event.set()
        for callback in (lambda: sensor_manager.stop_all(execution_config.join_timeout_s),
                         lambda: sync_worker.join(execution_config.join_timeout_s),
                         lambda: sync_logger.stop(execution_config.join_timeout_s),
                         raw_loggers.stop,
                         lambda: command_logger.stop(execution_config.join_timeout_s),
                         event_logger.stop):
            try:
                callback()
            except BaseException as exc:
                reported.append({"source": "cleanup", "error": repr(exc),
                                 "t_ns": time.monotonic_ns()})

    print(f"实验摘要：{{'mission': {mission.name!r}, 'reason': {executor.result.reason!r}, "
          f"'mission_finished': {executor.result.mission_finished}, "
          f"'safe_recentered': {executor.result.safe_recentered}, 'log_dir': {str(log_dir)!r}, "
          f"'background_failures': {reported!r}}}")
    if ctrl_c:
        return 130
    return 0 if executor.result.mission_finished and executor.result.safe_recentered else 1


def countdown_start_delay(seconds: float) -> None:
    """仅依赖本地单调时钟；倒计时期间不访问日志、传感器、网络或舵机。"""

    deadline = time.monotonic_ns() + int(round(seconds * 1e9))
    last: int | None = None
    while True:
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0:
            return
        display = max(1, int(math.ceil(remaining / 1e9)))
        if display != last:
            print(f"Action starts in {display} s")
            last = display
        time.sleep(min(0.1, remaining / 1e9))


def _write_metadata(config: dict[str, Any], mission: dict[str, Any], calibration: Any,
                    execution: dict[str, Any], args: argparse.Namespace, log_dir: Path) -> None:
    coupling = {"left": calibration.left_coupling.__dict__,
                "right": calibration.right_coupling.__dict__}
    write_metadata_yaml(log_dir / str(config.get("logging", {}).get("metadata_file", "metadata.yaml")), {
        "mode": "discrete_absolute_actions", "mission_name": mission["name"],
        "script_version": SCRIPT_VERSION, "manual_action_sequence": mission,
        "robot_yaml_snapshot": config, "discrete_action_coupling": coupling,
        "speed_limits_deg_s": {"tail": calibration.tail_max_speed_deg_s,
                               "fin_root": calibration.fin_max_speed_deg_s},
        "feedback_available": False, "position_estimation_mode": ESTIMATION_MODE,
        "execution_config": execution, "dry_run": args.dry_run,
        "mock_sensors": args.mock_sensors, "start_delay_s": args.start_delay_s,
        "keep_pwm": args.keep_pwm})


def _validate_delay(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise MissionValidationError("--start-delay-s 必须是有限非负数。")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _log_base(config: dict[str, Any]) -> Path:
    path = Path(str(config.get("logging", {}).get("base_dir",
                    config.get("runtime", {}).get("log_dir", "logs"))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "mission"


if __name__ == "__main__":
    raise SystemExit(main())
