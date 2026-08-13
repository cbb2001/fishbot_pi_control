from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from control.runtime.sample import to_jsonable


RAW_PLACEHOLDER_FILES = (
    "raw_imu.jsonl",
    "raw_depth.jsonl",
    "raw_power.jsonl",
    "raw_uwb.jsonl",
    "raw_vision.jsonl",
)
RAW_SENSOR_FILES = {
    "imu": "raw_imu.jsonl",
    "depth": "raw_depth.jsonl",
    "power": "raw_power.jsonl",
    "uwb": "raw_uwb.jsonl",
    "vision": "raw_vision.jsonl",
}
CAMERA_DIR = "camera"
CAMERA_INDEX_FILE = "camera_index.jsonl"


class JsonlLogger:
    def __init__(
        self,
        path: Path,
        *,
        flush_interval_s: float = 1.0,
        queue_maxsize: int = 10000,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.path = Path(path)
        self.flush_interval_s = max(0.1, float(flush_interval_s))
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self.stop_event = stop_event or threading.Event()
        self._external_stop_event = stop_event is not None
        self._thread: threading.Thread | None = None
        self.dropped_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._external_stop_event:
            self.stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"jsonl-{self.path.name}", daemon=True)
        self._thread.start()

    def write(self, obj: Any) -> bool:
        if self._thread is None:
            self.start()
        try:
            self.queue.put_nowait(to_jsonable(obj))
            return True
        except queue.Full:
            self.dropped_count += 1
            return False

    def stop(self, timeout: float | None = 5.0) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                next_flush_s = time.monotonic() + self.flush_interval_s
                while not self.stop_event.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        item = None

                    if item is not None:
                        try:
                            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                        except Exception as exc:
                            self.last_error = f"{type(exc).__name__}: {exc}"

                    now_s = time.monotonic()
                    if now_s >= next_flush_s:
                        try:
                            handle.flush()
                        except Exception as exc:
                            self.last_error = f"{type(exc).__name__}: {exc}"
                        next_flush_s = now_s + self.flush_interval_s

                try:
                    handle.flush()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"


class RawSensorLoggers:
    def __init__(
        self,
        log_dir: Path,
        *,
        flush_interval_s: float = 1.0,
        queue_maxsize: int = 10000,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.loggers = {
            name: JsonlLogger(
                self.log_dir / filename,
                flush_interval_s=flush_interval_s,
                queue_maxsize=queue_maxsize,
                stop_event=stop_event,
            )
            for name, filename in RAW_SENSOR_FILES.items()
        }
        self.last_t_ns = {name: -1 for name in RAW_SENSOR_FILES}

    def start(self) -> None:
        for logger in self.loggers.values():
            logger.start()

    def write_from_buffers(self, buffers: dict[str, Any]) -> None:
        for name, logger in self.loggers.items():
            buffer = buffers.get(name)
            if buffer is None:
                continue
            last_t_ns = self.last_t_ns.get(name, -1)
            latest_t_ns = last_t_ns
            for sample in buffer.snapshot():
                sample_t_ns = int(getattr(sample, "t_ns", -1))
                if sample_t_ns <= last_t_ns:
                    continue
                data = sample.to_dict() if hasattr(sample, "to_dict") else to_jsonable(sample)
                if name == "vision":
                    data = self._strip_vision_binary(data)
                logger.write(data)
                latest_t_ns = max(latest_t_ns, sample_t_ns)
            self.last_t_ns[name] = latest_t_ns

    def stop(self) -> None:
        for logger in self.loggers.values():
            logger.stop()

    def stats(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "dropped_count": logger.dropped_count,
                "last_error": logger.last_error,
            }
            for name, logger in self.loggers.items()
        }

    @staticmethod
    def _strip_vision_binary(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        sample_data = data.get("data")
        if isinstance(sample_data, dict):
            for key in ("frame", "image", "image_bytes", "frame_bytes"):
                sample_data.pop(key, None)
        return data


def create_run_log_dir(base_dir: str | Path, *, suffix: str = "sensor_test") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_dir)
    for index in range(100):
        name = f"{timestamp}_{suffix}" if index == 0 else f"{timestamp}_{suffix}_{index:02d}"
        log_dir = base / name
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
            (log_dir / CAMERA_DIR).mkdir(parents=True, exist_ok=True)
            (log_dir / CAMERA_INDEX_FILE).touch(exist_ok=True)
            return log_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique log directory under {base}")


def write_metadata_yaml(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(to_jsonable(metadata), handle, sort_keys=False, allow_unicode=True)
    except ImportError:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(metadata), handle, ensure_ascii=False, indent=2)


def prepare_raw_placeholders(log_dir: Path) -> None:
    (log_dir / CAMERA_DIR).mkdir(parents=True, exist_ok=True)
    (log_dir / CAMERA_INDEX_FILE).touch(exist_ok=True)
    for filename in RAW_PLACEHOLDER_FILES:
        (log_dir / filename).touch(exist_ok=True)
