"""在独立线程中同步传感器，并按样本时间戳附加舵机模型姿态。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from control.runtime.data_logger import JsonlLogger, RawSensorLoggers
from control.runtime.sensor_manager import SENSOR_NAMES
from control.runtime.sensor_synchronizer import SensorSynchronizer
from control.runtime.servo_state_tracker import ServoStateTracker


class SensorSyncWorker:
    """独立生成同步样本并搬运 raw 日志，绝不运行在舵机控制循环中。"""

    def __init__(
        self,
        *,
        synchronizer: SensorSynchronizer,
        tracker: ServoStateTracker,
        sync_logger: JsonlLogger,
        raw_loggers: RawSensorLoggers,
        sample_hz: float,
        shutdown_event: threading.Event,
        failure_queue: queue.Queue[dict[str, Any]],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wait_fn: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        """注入同步器、日志器和统一停止/失败通道，并计算绝对采样周期。"""

        if float(sample_hz) <= 0.0:
            raise ValueError("同步采样频率必须大于 0。")
        self.synchronizer = synchronizer
        self.tracker = tracker
        self.sync_logger = sync_logger
        self.raw_loggers = raw_loggers
        self.sample_hz = float(sample_hz)
        self.period_ns = max(1, int(round(1_000_000_000 / self.sample_hz)))
        self.shutdown_event = shutdown_event
        self.failure_queue = failure_queue
        self.clock_ns = clock_ns
        self.wait_fn = wait_fn or (lambda event, seconds: event.wait(max(0.0, seconds)))
        self._thread: threading.Thread | None = None
        self.skipped_tick_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        """启动唯一同步采样后台线程；已经运行时保持幂等。"""

        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_guarded, name="sensor-synchronizer", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """有界等待同步采样线程结束。"""

        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        """报告同步采样线程是否仍在运行。"""

        return bool(self._thread and self._thread.is_alive())

    def _run_guarded(self) -> None:
        """捕获后台异常并通过共享失败队列传播给主线程。"""

        try:
            self._run()
        except BaseException as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failure_queue.put(
                {"source": "SensorSyncWorker", "error": self.last_error, "t_ns": self.clock_ns()}
            )

    def _run(self) -> None:
        """按绝对截止时间生成同步样本、附加舵机姿态并异步写日志。"""

        start_t_ns = self.clock_ns()
        tick_index = 0
        while not self.shutdown_event.is_set():
            scheduled_t_ns = start_t_ns + tick_index * self.period_ns
            now_ns = self.clock_ns()
            if now_ns < scheduled_t_ns:
                self.wait_fn(self.shutdown_event, (scheduled_t_ns - now_ns) / 1_000_000_000.0)
                if self.shutdown_event.is_set():
                    break
                now_ns = self.clock_ns()
            latest_due = max(0, (now_ns - start_t_ns) // self.period_ns)
            if latest_due > tick_index:
                self.skipped_tick_count += int(latest_due - tick_index)
                tick_index = int(latest_due)
                scheduled_t_ns = start_t_ns + tick_index * self.period_ns

            sample = self.synchronizer.build(scheduled_t_ns)
            attach_servo_state_by_sensor(sample, self.tracker)
            self.raw_loggers.write_from_buffers(self.synchronizer.buffers)
            self.sync_logger.write(sample)
            tick_index += 1


def attach_servo_state_by_sensor(
    synchronized_sample: dict[str, Any],
    tracker: ServoStateTracker,
) -> dict[str, Any]:
    """按各传感器自己的 sample_t_ns 添加集中式舵机状态映射。"""

    state_by_sensor: dict[str, Any] = {}
    for name in SENSOR_NAMES:
        sensor_field = synchronized_sample.get(name, {})
        sample_t_ns = sensor_field.get("sample_t_ns")
        state_by_sensor[name] = {
            "sample_t_ns": sample_t_ns,
            "servo_state": None
            if sample_t_ns is None
            else tracker.query(int(sample_t_ns)),
        }
    synchronized_sample["servo_state_by_sensor"] = state_by_sensor
    # 同步记录自身也附加其 scheduled/sample 时间对应的七路姿态，便于直接建模；
    # 各传感器的精确 sample_t_ns 查询仍保留在 servo_state_by_sensor 中。
    query_t_ns = synchronized_sample.get("t_ns")
    if query_t_ns is not None:
        pose = tracker.query(int(query_t_ns))
        synchronized_sample.update({
            "servo_pose_query_t_ns": int(query_t_ns),
            "servo_reference_angles_deg": pose["reference_angles_deg"],
            "servo_commanded_angles_deg": pose["commanded_angles_deg"],
            "servo_estimated_angles_deg": pose["estimated_angles_deg"],
            "servo_feedback_available": pose["feedback_available"],
            "servo_estimation_mode": pose["estimation_mode"],
            "tail_action_index": pose["tail_action_index"],
            "left_action_index": pose.get("left_action_index"),
            "right_action_index": pose.get("right_action_index"),
            "tail_action_progress": pose["tail_action_progress"],
            "left_action_progress": pose.get("left_action_progress"),
            "right_action_progress": pose.get("right_action_progress"),
        })
        # 20260725 绝对离散动作跟踪器还公开三个动作组各自的 previous/current
        # theta。使用 get 保持旧 tracker 和旧脚本的日志结构向后兼容；新版
        # tracker 提供字段时，同步记录会在传感器自己的 sample_t_ns 查询结果
        # 之外，再把同步 tick 对应的值提升到顶层，方便后续直接建模。
        optional_action_fields = (
            "previous_action1_theta",
            "previous_action2_theta",
            "previous_action3_theta",
            "current_action1_theta",
            "current_action2_theta",
            "current_action3_theta",
        )
        synchronized_sample.update({
            key: pose.get(key)
            for key in optional_action_fields
            if key in pose
        })
    return synchronized_sample
