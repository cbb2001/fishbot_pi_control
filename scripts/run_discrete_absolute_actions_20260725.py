"""运行 20260725 人工定时绝对离散动作任务及其传感器、日志和安全流程。"""

from __future__ import annotations

import argparse
import math
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root


PROJECT_ROOT = add_project_root()

from control.runtime.action_scheduler_20260725 import ActionScheduler  # noqa: E402
from control.runtime.data_logger import (  # noqa: E402
    JsonlLogger,
    RawSensorLoggers,
    create_run_log_dir,
    prepare_raw_placeholders,
    write_metadata_yaml,
)
from control.runtime.discrete_absolute_actions_20260725 import (  # noqa: E402
    MissionValidationError,
    build_calibration,
    reference_config_to_dict,
    validate_mission,
)
from control.runtime.event_logger import EventLogger  # noqa: E402
from control.runtime.manual_action_provider_20260725 import (  # noqa: E402
    ManualSequenceProvider,
)
from control.runtime.sensor_manager import SensorManager  # noqa: E402
from control.runtime.sensor_sync_worker import SensorSyncWorker  # noqa: E402
from control.runtime.sensor_synchronizer import SensorSynchronizer  # noqa: E402
from control.runtime.servo_executor_20260725 import (  # noqa: E402
    ServoExecutor,
    execution_config_from_robot,
)
from control.runtime.servo_state_tracker_20260725 import (  # noqa: E402
    ESTIMATION_MODE,
    ServoStateTracker,
)
from control.safety import (  # noqa: E402
    SafetyError,
    ensure_not_windows_hardware_run,
    load_robot_config,
)


SCRIPT_VERSION = "20260725-v1"
SAVED_MISSION_FILENAME = "manual_action_sequence.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析新离散动作入口参数，不暴露频率、相位、速度或旧行程比例。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行 action1(theta,t)、action2/3(theta,t,b1,b2) "
            "人工绝对离散动作序列。"
        )
    )
    parser.add_argument("--mission", required=True, help="人工动作 YAML 路径。")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "robot.yaml"),
        help="机器人配置路径。",
    )
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不导入、不初始化真实 PCA9685，也不发送 PWM。",
    )
    parser.add_argument(
        "--mock-sensors",
        action="store_true",
        help="只模拟传感器；单独使用不代表舵机 dry-run。",
    )
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=5.0,
        help="日志、传感器和舵机初始化前的本地倒计时（默认 5 秒）。",
    )
    parser.add_argument(
        "--keep-pwm",
        action="store_true",
        help="安全回中后继续保持 PWM。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """完成预验证、并行任务、传感器记录、安全回中和有界线程清理。"""

    args = parse_args(argv)
    mission_path = _project_path(args.mission)
    boot_events: list[tuple[int, str, dict[str, Any]]] = []

    # 完整 mission 模拟和所有机械角检查必须先于 confirm、倒计时、日志、
    # 传感器及控制器初始化，保证非法动作不可能触碰硬件。
    try:
        _validate_delay(args.start_delay_s)
        config = load_robot_config(args.config)
        mission = ManualSequenceProvider(mission_path).load()
        loaded_t_ns = time.monotonic_ns()
        boot_events.append(
            (
                loaded_t_ns,
                "mission_loaded",
                {
                    "mission_name": mission.name,
                    "mission_source": str(mission_path),
                    "mission": mission.to_dict(),
                },
            )
        )
        calibration = build_calibration(config)
        validate_mission(mission, calibration)
        execution_config = execution_config_from_robot(config)
        boot_events.append(
            (
                time.monotonic_ns(),
                "mission_validated",
                {
                    "mission_name": mission.name,
                    "validation_rule": (
                        "20260725_reference_math_and_robot_yaml_mechanical_limits"
                    ),
                    "action_reference": reference_config_to_dict(calibration),
                },
            )
        )
    except (MissionValidationError, SafetyError, ValueError) as exc:
        print(f"20260725 绝对离散动作任务验证失败：{exc}")
        return 2

    if not args.dry_run and args.confirm != "MOVE":
        print(
            "拒绝真实舵机动作：必须显式提供 --confirm MOVE；"
            "无 PWM 验证请使用 --dry-run。"
        )
        return 2
    if not (args.dry_run and args.mock_sensors):
        ensure_not_windows_hardware_run()

    try:
        countdown_start_delay(args.start_delay_s)
    except KeyboardInterrupt:
        print("\n倒计时被 Ctrl+C 取消；尚未创建日志、启动传感器或初始化舵机。")
        return 130

    log_dir = create_run_log_dir(
        _log_base(config),
        suffix=f"discrete_absolute_20260725_{_safe_suffix(mission.name)}",
    )
    prepare_raw_placeholders(log_dir)
    # 保存原始人工动作文件，避免 YAML 格式、注释或字段顺序只存在于外部路径。
    shutil.copy2(mission_path, log_dir / SAVED_MISSION_FILENAME)
    _write_metadata(
        config,
        mission.to_dict(),
        mission_path,
        calibration,
        execution_config.to_dict(),
        args,
        log_dir,
    )

    logging_config = config.get("logging", {})
    runtime_config = config.get("runtime", {})
    flush_interval_s = float(
        logging_config.get(
            "flush_interval_s",
            runtime_config.get("flush_interval_s", 1.0),
        )
    )
    queue_maxsize = int(logging_config.get("write_queue_maxsize", 10_000))
    shutdown_event = threading.Event()
    motion_stop_event = threading.Event()
    failures: queue.Queue[dict[str, Any]] = queue.Queue()

    sync_logger = JsonlLogger(
        log_dir
        / str(
            logging_config.get(
                "synchronized_file",
                "synchronized_sensors.jsonl",
            )
        ),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    command_logger = JsonlLogger(
        log_dir / "commands.jsonl",
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    raw_loggers = RawSensorLoggers(
        log_dir,
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )
    event_logger = EventLogger(
        log_dir
        / str(logging_config.get("events_file", "events.jsonl")),
        flush_interval_s=flush_interval_s,
        queue_maxsize=queue_maxsize,
    )

    tracker = ServoStateTracker(calibration)
    scheduler = ActionScheduler(mission, calibration)
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
        sample_hz=float(runtime_config.get("synchronized_sample_hz", 30.0)),
        shutdown_event=shutdown_event,
        failure_queue=failures,
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
        failure_queue=failures,
    )

    ctrl_c = False
    reported_failures: list[dict[str, Any]] = []
    seen_failures: set[tuple[str, str]] = set()
    alive_logger_workers: list[str] = []
    camera_image_writers: list[Any] = []
    try:
        # 启动步骤也必须位于 finally 保护范围内。任何同步初始化异常都将
        # 转成主线程失败并进入同一套有界清理，避免已经启动的日志或传感器
        # 线程因后续组件启动失败而遗留。
        for logger in (sync_logger, command_logger):
            logger.start()
        raw_loggers.start()
        event_logger.start()
        for event_t_ns, event_type, data in boot_events:
            _write_required_final_event(
                event_logger,
                event_type,
                data,
                t_ns=event_t_ns,
            )
        sensor_manager.start_all()
        sync_worker.start()
        executor.start()

        print(f"日志、传感器和单一 ServoExecutor 已启动：{log_dir}")
        if args.dry_run:
            print("Dry-run：未导入或初始化真实 PCA9685，不会发送 PWM。")
        print(
            "舵机无位置反馈：estimated_angle 仅为 "
            "assumed_perfect_tracking 模型估计。"
        )

        while executor.is_alive():
            _drain_failure_queue(
                failures,
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
            for failure in sensor_manager.fatal_errors():
                _record_runtime_failure(
                    failure,
                    reported_failures,
                    seen_failures,
                    motion_stop_event,
                )
            for failure in _logger_and_camera_failures(
                sync_logger,
                command_logger,
                raw_loggers,
                event_logger,
                sensor_manager,
            ):
                _record_runtime_failure(
                    failure,
                    reported_failures,
                    seen_failures,
                    motion_stop_event,
                )
            time.sleep(0.05)
    except KeyboardInterrupt:
        ctrl_c = True
        motion_stop_event.set()
        print("\n收到 Ctrl+C，已停止动作推进，正在由 ServoExecutor 安全回中。")
    except BaseException as exc:
        failure = {
            "source": "main_startup_or_runtime",
            "error": f"{type(exc).__name__}: {exc}",
            "t_ns": time.monotonic_ns(),
        }
        _record_runtime_failure(
            failure,
            reported_failures,
            seen_failures,
            motion_stop_event,
        )
        print(f"启动或主循环异常：{failure['error']}")
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
        if executor.is_alive():
            reported_failures.append(
                {
                    "source": "ServoExecutor",
                    "error": "join_timeout",
                    "t_ns": time.monotonic_ns(),
                }
            )

        # 先停止所有生产者，再由主线程做最后一次 raw 搬运；这样不会与
        # SensorSyncWorker 竞争 RawSensorLoggers.last_t_ns 游标。
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
                "move final raw samples",
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
        ):
            ctrl_c = _cleanup_step(
                label,
                callback,
                reported_failures,
            ) or ctrl_c

        # CameraWorker 在 close() 中保留最终 writer 的诊断引用；等生产者停止
        # 后再采集，可覆盖执行器回中期间首次懒创建图片线程的情况。
        camera_image_writers = [
            image_writer
            for worker in sensor_manager.workers
            if (
                image_writer := (
                    getattr(worker, "_closed_image_writer", None)
                    or getattr(worker, "_image_writer", None)
                )
            )
            is not None
        ]
        _drain_failure_queue(
            failures,
            reported_failures,
            seen_failures,
            motion_stop_event,
        )
        # 短任务可能在主循环第一次轮询前结束，日志 flush 错误也可能只在
        # stop()/join() 阶段出现；清理后必须再次汇总，不能依赖运行中轮询。
        for failure in sensor_manager.fatal_errors():
            _record_runtime_failure(
                failure,
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
        for failure in _logger_and_camera_failures(
            sync_logger,
            command_logger,
            raw_loggers,
            event_logger,
            sensor_manager,
        ):
            _record_runtime_failure(
                failure,
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
        for worker_name in sensor_manager.alive_worker_names():
            _record_runtime_failure(
                {"source": worker_name, "error": "join_timeout"},
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
        if sync_worker.is_alive():
            _record_runtime_failure(
                {"source": "SensorSyncWorker", "error": "join_timeout"},
                reported_failures,
                seen_failures,
                motion_stop_event,
            )

        ctrl_c = _cleanup_step(
            "enqueue logger_stopped",
            lambda: _write_required_final_event(
                event_logger,
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
                    "event_dropped_count": event_logger.logger.dropped_count,
                    "event_last_error": event_logger.logger.last_error,
                },
                t_ns=time.monotonic_ns(),
            ),
            reported_failures,
        ) or ctrl_c
        ctrl_c = _cleanup_step(
            "stop event logger",
            event_logger.stop,
            reported_failures,
        ) or ctrl_c
        # event logger 自身只有 stop 后才能确认最终 flush 是否成功。
        for failure in _logger_and_camera_failures(
            sync_logger,
            command_logger,
            raw_loggers,
            event_logger,
            sensor_manager,
        ):
            _record_runtime_failure(
                failure,
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
        for failure in _captured_camera_failures(camera_image_writers):
            _record_runtime_failure(
                failure,
                reported_failures,
                seen_failures,
                motion_stop_event,
            )
        alive_logger_workers = _alive_logger_worker_names(
            sync_logger,
            command_logger,
            raw_loggers,
            event_logger,
            camera_image_writers,
        )
        for worker_name in alive_logger_workers:
            _record_runtime_failure(
                {"source": worker_name, "error": "join_timeout"},
                reported_failures,
                seen_failures,
                motion_stop_event,
            )

    alive_sensor_workers = sensor_manager.alive_worker_names()
    sync_worker_alive = sync_worker.is_alive()
    summary = {
        "mission": mission.name,
        "reason": executor.result.reason,
        "mission_finished": executor.result.mission_finished,
        "safe_recentered": executor.result.safe_recentered,
        "pwm_released": executor.result.pwm_released,
        "pwm_write_count": executor.result.pwm_write_count,
        "unchanged_skip_count": executor.result.unchanged_skip_count,
        "skipped_servo_ticks": executor.result.skipped_tick_count,
        "skipped_sync_ticks": sync_worker.skipped_tick_count,
        "background_failures": reported_failures,
        "alive_sensor_workers": alive_sensor_workers,
        "sync_worker_alive": sync_worker_alive,
        "alive_logger_workers": alive_logger_workers,
        "log_dir": str(log_dir),
        "saved_mission": str(log_dir / SAVED_MISSION_FILENAME),
    }
    print(f"实验摘要：{summary}")
    if ctrl_c:
        return 130
    if (
        executor.is_alive()
        or alive_sensor_workers
        or sync_worker_alive
        or alive_logger_workers
        or reported_failures
        or not executor.result.safe_recentered
    ):
        return 1
    return 0 if executor.result.mission_finished else 1


def countdown_start_delay(seconds: float) -> None:
    """仅使用本机 monotonic 时钟倒计时，不访问网络、日志或硬件。"""

    deadline_t_ns = time.monotonic_ns() + int(round(float(seconds) * 1e9))
    last_display: int | None = None
    while True:
        remaining_ns = deadline_t_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        display = max(1, int(math.ceil(remaining_ns / 1e9)))
        if display != last_display:
            print(f"Action starts in {display} s")
            last_display = display
        time.sleep(min(0.1, remaining_ns / 1e9))


def _write_metadata(
    config: dict[str, Any],
    mission: dict[str, Any],
    mission_path: Path,
    calibration: Any,
    execution: dict[str, Any],
    args: argparse.Namespace,
    log_dir: Path,
) -> None:
    """保存完整任务、robot 快照、动作参考和无反馈估计语义。"""

    reference = reference_config_to_dict(calibration)
    write_metadata_yaml(
        log_dir
        / str(
            config.get("logging", {}).get(
                "metadata_file",
                "metadata.yaml",
            )
        ),
        {
            "mode": "discrete_absolute_actions_20260725",
            "mission_name": mission["name"],
            "script_version": SCRIPT_VERSION,
            "manual_action_sequence": mission,
            "mission_source_path": str(mission_path),
            "saved_mission_file": SAVED_MISSION_FILENAME,
            "robot_yaml_snapshot": config,
            "action_reference": reference,
            "tail_theta_range_deg": [
                reference["tail"]["theta_min_deg"],
                reference["tail"]["theta_max_deg"],
            ],
            "root_reference_span_deg": {
                "left": reference["left_fin"]["root_reference_span_deg"],
                "right": reference["right_fin"]["root_reference_span_deg"],
            },
            "tip_reference_span_deg": {
                "left": reference["left_fin"]["tip_reference_span_deg"],
                "right": reference["right_fin"]["tip_reference_span_deg"],
            },
            "feedback_available": False,
            "position_estimation_mode": ESTIMATION_MODE,
            "execution_config": execution,
            "control_update_hz": execution["command_hz"],
            "dry_run": args.dry_run,
            "mock_sensors": args.mock_sensors,
            "start_delay_s": args.start_delay_s,
            "keep_pwm": args.keep_pwm,
        },
    )


def _logger_and_camera_failures(
    sync_logger: JsonlLogger,
    command_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    event_logger: EventLogger,
    sensor_manager: SensorManager,
) -> list[dict[str, str]]:
    """把原本只保存在 last_error 的后台写盘错误提升到主线程。"""

    failures: list[dict[str, str]] = []
    named_loggers = {
        "synchronized_logger": sync_logger,
        "command_logger": command_logger,
        "event_logger": event_logger.logger,
    }
    for name, logger in named_loggers.items():
        if logger.last_error:
            failures.append({"source": name, "error": logger.last_error})
        if logger.dropped_count:
            failures.append(
                {
                    "source": name,
                    "error": f"dropped_records={logger.dropped_count}",
                }
            )
    for name, stats in raw_loggers.stats().items():
        if stats.get("last_error"):
            failures.append(
                {
                    "source": f"raw_{name}_logger",
                    "error": str(stats["last_error"]),
                }
            )
        if int(stats.get("dropped_count", 0)) > 0:
            failures.append(
                {
                    "source": f"raw_{name}_logger",
                    "error": f"dropped_records={int(stats['dropped_count'])}",
                }
            )
    for worker in sensor_manager.workers:
        image_writer = getattr(worker, "_image_writer", None)
        if image_writer is not None and image_writer.last_error:
            failures.append(
                {
                    "source": "CameraImageWriter",
                    "error": str(image_writer.last_error),
                }
            )
        if (
            image_writer is not None
            and image_writer.index_logger.last_error
        ):
            failures.append(
                {
                    "source": "CameraIndexLogger",
                    "error": str(image_writer.index_logger.last_error),
                }
            )
        if (
            image_writer is not None
            and image_writer.index_logger.dropped_count
        ):
            failures.append(
                {
                    "source": "CameraIndexLogger",
                    "error": (
                        "dropped_records="
                        f"{image_writer.index_logger.dropped_count}"
                    ),
                }
            )
    return failures


def _captured_camera_failures(
    image_writers: list[Any],
) -> list[dict[str, str]]:
    """检查 CameraWorker.close() 后仍保留引用的图片和索引写盘错误。"""

    failures: list[dict[str, str]] = []
    for index, writer in enumerate(image_writers):
        if writer.last_error:
            failures.append(
                {
                    "source": f"CameraImageWriter[{index}]",
                    "error": str(writer.last_error),
                }
            )
        if writer.index_logger.last_error:
            failures.append(
                {
                    "source": f"CameraIndexLogger[{index}]",
                    "error": str(writer.index_logger.last_error),
                }
            )
        if writer.index_logger.dropped_count:
            failures.append(
                {
                    "source": f"CameraIndexLogger[{index}]",
                    "error": (
                        "dropped_records="
                        f"{writer.index_logger.dropped_count}"
                    ),
                }
            )
    return failures


def _alive_logger_worker_names(
    sync_logger: JsonlLogger,
    command_logger: JsonlLogger,
    raw_loggers: RawSensorLoggers,
    event_logger: EventLogger,
    image_writers: list[Any],
) -> list[str]:
    """返回清理后仍存活的全部异步写盘线程名称。"""

    alive: list[str] = []
    for name, logger in (
        ("synchronized_logger", sync_logger),
        ("command_logger", command_logger),
        ("event_logger", event_logger.logger),
    ):
        if logger.is_alive():
            alive.append(name)
    for name, logger in raw_loggers.loggers.items():
        if logger.is_alive():
            alive.append(f"raw_{name}_logger")
    for index, writer in enumerate(image_writers):
        thread = getattr(writer, "_thread", None)
        if thread is not None and thread.is_alive():
            alive.append(f"CameraImageWriter[{index}]")
        if writer.index_logger.is_alive():
            alive.append(f"CameraIndexLogger[{index}]")
    return alive


def _write_required_final_event(
    event_logger: EventLogger,
    event_type: str,
    data: dict[str, Any],
    *,
    t_ns: int,
) -> None:
    """最终事件必须成功入队；队列已满时让清理摘要明确失败。"""

    if not event_logger.write(event_type, data, t_ns=t_ns):
        raise RuntimeError(
            f"events.jsonl 异步队列已满，事件 {event_type} 未入队。"
        )


def _record_runtime_failure(
    failure: dict[str, Any],
    reported: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    motion_stop_event: threading.Event,
) -> None:
    """去重后台错误并请求停止动作；普通 invalid 传感器样本不会到此处。"""

    source = str(failure.get("source", "unknown"))
    error = str(failure.get("error", "unknown"))
    key = (source, error)
    if key in seen:
        return
    seen.add(key)
    reported.append(
        {
            **failure,
            "source": source,
            "error": error,
            "t_ns": int(failure.get("t_ns", time.monotonic_ns())),
        }
    )
    motion_stop_event.set()


def _drain_failure_queue(
    failures: queue.Queue[dict[str, Any]],
    reported: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    motion_stop_event: threading.Event,
) -> None:
    """非阻塞排空线程错误队列，避免主线程长期卡在 queue.get。"""

    while True:
        try:
            failure = failures.get_nowait()
        except queue.Empty:
            return
        _record_runtime_failure(
            failure,
            reported,
            seen,
            motion_stop_event,
        )


def _cleanup_step(
    label: str,
    callback: Any,
    failures: list[dict[str, Any]],
) -> bool:
    """清理时保留后续步骤；重复 Ctrl+C 和单项错误都写入摘要。"""

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


def _validate_delay(value: float) -> None:
    """验证启动倒计时为有限非负秒数。"""

    if not math.isfinite(value) or value < 0.0:
        raise MissionValidationError("--start-delay-s 必须是有限非负数。")


def _project_path(value: str) -> Path:
    """把相对命令行路径稳定解析到项目根目录。"""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _log_base(config: dict[str, Any]) -> Path:
    """从日志或运行配置读取日志根目录，并解析项目相对路径。"""

    path = Path(
        str(
            config.get("logging", {}).get(
                "base_dir",
                config.get("runtime", {}).get("log_dir", "logs"),
            )
        )
    )
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_suffix(value: str) -> str:
    """把任务名称转换为仅含安全 ASCII 字符的日志目录后缀。"""

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "mission"


if __name__ == "__main__":
    raise SystemExit(main())
