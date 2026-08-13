from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

from control.runtime.camera_worker import CameraWorker
from control.runtime.i2c_worker import I2CWorker
from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.sensor_worker import BaseSensorWorker
from control.runtime.uart_workers import IMUWorker, UWBWorker


SENSOR_NAMES = ("imu", "uwb", "depth", "power", "vision")

DEFAULT_SENSOR_CONFIGS: dict[str, dict[str, Any]] = {
    "imu": {
        "enabled": True,
        "port": "/dev/ttyAMA4",
        "baudrate": 460800,
        "buffer_size": 1000,
        "timeout_ms": 100,
    },
    "uwb": {
        "enabled": True,
        "port": "/dev/ttyAMA0",
        "baudrate": 115200,
        "buffer_size": 300,
        "timeout_ms": 500,
        "optional_underwater": True,
    },
    "depth": {
        "enabled": True,
        "i2c_bus": 1,
        "address": 0x76,
        "rate_hz": 20,
        "buffer_size": 500,
        "timeout_ms": 300,
    },
    "power": {
        "enabled": True,
        "i2c_bus": 1,
        "address": 0x45,
        "rate_hz": 2,
        "buffer_size": 200,
        "timeout_ms": 2000,
    },
    "vision": {
        "enabled": False,
        "camera_index": 0,
        "width": 1280,
        "height": 480,
        "fourcc": "MJPG",
        "rate_hz": 15,
        "buffer_size": 100,
        "timeout_ms": 300,
        "save_frames": False,
        "save_fps": 5,
        "image_format": "jpg",
        "jpeg_quality": 85,
        "save_combined_frame": True,
        "split_stereo": False,
        "max_image_queue_size": 100,
    },
}


def sensor_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    merged = dict(DEFAULT_SENSOR_CONFIGS.get(name, {}))
    merged.update(config.get("sensors", {}).get(name, {}))
    return merged


class MockSensorWorker(BaseSensorWorker):
    def __init__(
        self,
        name: str,
        buffer: RingBuffer,
        *,
        rate_hz: float,
        environment: str,
        save_fps: float = 5.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__(
            name,
            buffer,
            stop_event=stop_event,
            loop_delay_s=1.0 / max(float(rate_hz), 0.001),
        )
        self.environment = environment
        self._start_s = time.monotonic()
        self.save_fps = float(save_fps)

    def read_once(self) -> SensorSample:
        elapsed_s = time.monotonic() - self._start_s
        if self.name == "imu":
            data = {
                "acc_mps2": [0.02 * math.sin(elapsed_s), 0.0, 9.80665],
                "gyro_radps": [0.0, 0.0, 0.02 * math.cos(elapsed_s)],
                "roll_deg": 1.0 * math.sin(elapsed_s),
                "pitch_deg": 0.5 * math.cos(elapsed_s),
                "yaw_deg": (elapsed_s * 5.0) % 360.0,
                "quat": [1.0, 0.0, 0.0, 0.0],
                "temperature_c": 25.0,
            }
            return self.make_sample(data)
        if self.name == "depth":
            depth_m = max(0.0, 0.4 + 0.02 * math.sin(elapsed_s * 0.5))
            data = {
                "depth_m": depth_m,
                "pressure_pa": 101325.0 + depth_m * 9806.65,
                "temperature_c": 22.0,
            }
            return self.make_sample(data)
        if self.name == "power":
            data = {
                "voltage_v": 12.1,
                "current_a": 0.8 + 0.05 * math.sin(elapsed_s),
                "power_w": 9.68,
            }
            return self.make_sample(data)
        if self.name == "uwb":
            if self.environment == "underwater":
                return self.make_sample({}, ok=False, error="no_signal_or_timeout")
            return self.make_sample(
                {
                    "x_m": 0.1 * elapsed_s,
                    "y_m": 0.0,
                    "z_m": 0.0,
                    "range_m": 1.0,
                    "quality": 1.0,
                    "anchor_id": "mock_anchor",
                    "tag_id": "mock_tag",
                    "raw_hex": "",
                }
            )
        if self.name == "vision":
            seq = self._seq + 1
            return self.make_sample(
                {
                    "frame_id": seq,
                    "width": 1280,
                    "height": 480,
                    "mean_brightness": 100.0 + 10.0 * math.sin(elapsed_s),
                    "timestamp": time.time(),
                    "image_file": None,
                    "image_t_ns": None,
                    "dropped_frames": 0,
                    "save_fps": self.save_fps,
                    "actual_save_fps": None,
                }
            )
        return self.make_sample({}, ok=False, error="unknown_mock_sensor")


class SensorManager:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        mock: bool | None = None,
        log_dir: str | Path | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        runtime_cfg = config.get("runtime", {})
        self.mock = bool(runtime_cfg.get("mock", False) if mock is None else mock)
        self.environment = str(runtime_cfg.get("environment", "underwater"))
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.stop_event = stop_event or threading.Event()
        self.buffers: dict[str, RingBuffer] = {
            name: RingBuffer(int(sensor_config(config, name).get("buffer_size", 100)))
            for name in SENSOR_NAMES
        }
        self.workers: list[Any] = []
        self._created = False

    def start_all(self) -> None:
        if not self._created:
            self._create_workers()
            self._created = True
        for worker in self.workers:
            worker.start()

    def stop_all(self, timeout: float | None = 10.0) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=timeout)

    def get_status(self) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        status: dict[str, Any] = {}
        for name in SENSOR_NAMES:
            cfg = sensor_config(self.config, name)
            enabled = bool(cfg.get("enabled", True))
            sample = self.buffers[name].latest()
            timeout_ns = int(float(cfg.get("timeout_ms", 1000)) * 1_000_000)
            if not enabled:
                status[name] = {
                    "enabled": False,
                    "valid": False,
                    "age_ms": None,
                    "error": "disabled",
                    "count": len(self.buffers[name]),
                    "latest": None,
                }
                continue
            if sample is None:
                status[name] = {
                    "enabled": True,
                    "valid": False,
                    "age_ms": None,
                    "error": "no_sample",
                    "count": len(self.buffers[name]),
                    "latest": None,
                }
                continue
            age_ns = max(0, now_ns - sample.t_ns)
            valid = bool(sample.ok) and age_ns <= timeout_ns
            error = sample.error
            if age_ns > timeout_ns:
                error = error or "stale"
            status[name] = {
                "enabled": True,
                "valid": valid,
                "age_ms": age_ns / 1_000_000.0,
                "error": error,
                "count": len(self.buffers[name]),
                "latest": sample.to_dict(),
            }
        return status

    def _create_workers(self) -> None:
        if self.mock:
            self._create_mock_workers()
            return

        imu_cfg = sensor_config(self.config, "imu")
        if bool(imu_cfg.get("enabled", True)):
            self.workers.append(
                IMUWorker(
                    self.buffers["imu"],
                    port=str(imu_cfg.get("port", "/dev/ttyAMA4")),
                    baudrate=int(imu_cfg.get("baudrate", 460800)),
                    timeout_s=float(imu_cfg.get("serial_timeout_s", 0.1)),
                    sample_rate_hz=float(imu_cfg["sample_rate_hz"]),
                    stop_event=self.stop_event,
                )
            )

        uwb_cfg = sensor_config(self.config, "uwb")
        if bool(uwb_cfg.get("enabled", True)):
            self.workers.append(
                UWBWorker(
                    self.buffers["uwb"],
                    port=str(uwb_cfg.get("port", "/dev/ttyAMA0")),
                    baudrate=int(uwb_cfg.get("baudrate", 115200)),
                    timeout_s=float(uwb_cfg.get("serial_timeout_s", 0.2)),
                    stop_event=self.stop_event,
                )
            )

        depth_enabled = bool(sensor_config(self.config, "depth").get("enabled", True))
        power_enabled = bool(sensor_config(self.config, "power").get("enabled", True))
        if depth_enabled or power_enabled:
            self.workers.append(I2CWorker(self.config, self.buffers, stop_event=self.stop_event))

        vision_cfg = sensor_config(self.config, "vision")
        if bool(vision_cfg.get("enabled", False)):
            self.workers.append(
                CameraWorker(
                    self.buffers["vision"],
                    camera_index=vision_cfg.get("camera_index", 0),
                    width=int(vision_cfg.get("width", 1280)),
                    height=int(vision_cfg.get("height", 480)),
                    fourcc=str(vision_cfg.get("fourcc", "MJPG")),
                    rate_hz=float(vision_cfg.get("rate_hz", 15)),
                    log_dir=self.log_dir,
                    save_frames=bool(vision_cfg.get("save_frames", False)),
                    save_fps=float(vision_cfg.get("save_fps", 5)),
                    image_format=str(vision_cfg.get("image_format", "jpg")),
                    jpeg_quality=int(vision_cfg.get("jpeg_quality", 85)),
                    save_combined_frame=bool(vision_cfg.get("save_combined_frame", True)),
                    split_stereo=bool(vision_cfg.get("split_stereo", False)),
                    max_image_queue_size=int(vision_cfg.get("max_image_queue_size", 100)),
                    camera_index_file=str(
                        self.config.get("logging", {}).get("camera_index_file", "camera_index.jsonl")
                    ),
                    flush_interval_s=float(self.config.get("logging", {}).get("flush_interval_s", 1.0)),
                    stop_event=self.stop_event,
                )
            )

    def _create_mock_workers(self) -> None:
        rates = {
            "imu": float(sensor_config(self.config, "imu")["sample_rate_hz"]),
            "uwb": 2.0,
            "depth": float(sensor_config(self.config, "depth").get("rate_hz", 20)),
            "power": float(sensor_config(self.config, "power").get("rate_hz", 2)),
            "vision": float(sensor_config(self.config, "vision").get("rate_hz", 15)),
        }
        for name in SENSOR_NAMES:
            cfg = sensor_config(self.config, name)
            if not bool(cfg.get("enabled", True)):
                continue
            self.workers.append(
                MockSensorWorker(
                    name,
                    self.buffers[name],
                    rate_hz=rates[name],
                    environment=self.environment,
                    save_fps=float(cfg.get("save_fps", 5)),
                    stop_event=self.stop_event,
                )
            )

    def fatal_errors(self) -> list[dict[str, str]]:
        """返回未被 invalid 样本机制吸收的线程级异常。"""

        errors: list[dict[str, str]] = []
        for worker in self.workers:
            error = getattr(worker, "fatal_error", None)
            if error:
                errors.append({"source": type(worker).__name__, "error": str(error)})
        return errors

    def alive_worker_names(self) -> list[str]:
        """返回 join 后仍存活的 worker，供清理摘要明确报告。"""

        return [
            type(worker).__name__
            for worker in self.workers
            if callable(getattr(worker, "is_alive", None)) and worker.is_alive()
        ]
