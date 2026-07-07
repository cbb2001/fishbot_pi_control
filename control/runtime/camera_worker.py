from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control.runtime.data_logger import JsonlLogger
from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.sensor_worker import BaseSensorWorker


@dataclass
class ImageSaveTask:
    frame: Any
    t_ns: int
    frame_id: int
    filename: str
    width: int
    height: int
    image_format: str
    jpeg_quality: int


class ImageWriter:
    def __init__(
        self,
        *,
        log_dir: Path,
        camera_dir: str = "camera",
        index_file: str = "camera_index.jsonl",
        max_queue_size: int = 100,
        flush_interval_s: float = 1.0,
        index_queue_maxsize: int = 10000,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.camera_dir = camera_dir
        self.camera_path = self.log_dir / camera_dir
        self.index_logger = JsonlLogger(
            self.log_dir / index_file,
            flush_interval_s=flush_interval_s,
            queue_maxsize=index_queue_maxsize,
        )
        self.queue: queue.Queue[ImageSaveTask] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped_frames = 0
        self.saved_frames = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.camera_path.mkdir(parents=True, exist_ok=True)
        self.index_logger.start()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-image-writer", daemon=True)
        self._thread.start()

    def enqueue(self, task: ImageSaveTask) -> bool:
        if self._thread is None:
            self.start()
        try:
            self.queue.put_nowait(task)
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def stop(self, timeout: float | None = 10.0) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout)
        self.index_logger.stop()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._write_task(task)

    def _write_task(self, task: ImageSaveTask) -> None:
        saved = False
        error = None
        try:
            cv2 = self._load_cv2()
            path = self.log_dir / task.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            if task.image_format.lower() in {"jpg", "jpeg"}:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(task.jpeg_quality)]
            saved = bool(cv2.imwrite(str(path), task.frame, params))
            if not saved:
                error = "cv2_imwrite_failed"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_error = error

        if saved:
            self.saved_frames += 1
        self.index_logger.write(
            {
                "t_ns": task.t_ns,
                "frame_id": task.frame_id,
                "filename": task.filename,
                "width": task.width,
                "height": task.height,
                "format": task.image_format,
                "jpeg_quality": task.jpeg_quality,
                "saved": saved,
                "error": error,
            }
        )

    @staticmethod
    def _load_cv2():
        import cv2

        return cv2


class CameraWorker(BaseSensorWorker):
    def __init__(
        self,
        buffer: RingBuffer,
        *,
        camera_index: int | str = 0,
        width: int = 1280,
        height: int = 480,
        fourcc: str = "MJPG",
        rate_hz: float = 15.0,
        log_dir: Path | None = None,
        save_frames: bool = False,
        save_fps: float = 5.0,
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        save_combined_frame: bool = True,
        split_stereo: bool = False,
        max_image_queue_size: int = 100,
        camera_index_file: str = "camera_index.jsonl",
        flush_interval_s: float = 1.0,
    ) -> None:
        period_s = 1.0 / max(float(rate_hz), 0.001)
        super().__init__("vision", buffer, loop_delay_s=period_s, error_backoff_s=1.0)
        self.camera_index = camera_index
        self.width = int(width)
        self.height = int(height)
        self.fourcc = fourcc.strip().upper()
        self.rate_hz = float(rate_hz)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.save_frames = bool(save_frames)
        self.save_fps = max(0.0, float(save_fps))
        self.image_format = image_format.lower().lstrip(".") or "jpg"
        if self.image_format == "jpeg":
            self.image_format = "jpg"
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.save_combined_frame = bool(save_combined_frame)
        self.split_stereo = bool(split_stereo)
        self.max_image_queue_size = int(max_image_queue_size)
        self.camera_index_file = camera_index_file
        self.flush_interval_s = float(flush_interval_s)
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._image_writer: ImageWriter | None = None
        self._frame_id = 0
        self._next_cv2_retry_s = 0.0
        self._next_save_s = 0.0
        self._first_save_s: float | None = None
        self._last_save_s: float | None = None
        self._latest_image_file: str | None = None
        self._latest_image_t_ns: int | None = None

    def read_once(self) -> SensorSample | None:
        now_s = time.monotonic()
        cv2 = self._ensure_cv2(now_s)
        if cv2 is None:
            return None
        capture = self._ensure_capture(cv2)
        ok, frame = capture.read()
        t_ns = time.monotonic_ns()
        if not ok or frame is None:
            return self.make_sample({}, ok=False, error="camera_read_failed", t_ns=t_ns)

        self._frame_id += 1
        height, width = frame.shape[:2]
        image_file = None
        image_t_ns = None
        image_save_error = None

        if self._should_save(now_s):
            image_file, image_t_ns, image_save_error = self._enqueue_frame(frame, t_ns, width, height)
            if image_file is not None:
                self._latest_image_file = image_file
                self._latest_image_t_ns = image_t_ns

        data = {
            "frame_id": self._frame_id,
            "width": int(width),
            "height": int(height),
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fourcc": self.fourcc,
            "requested_fps": self.rate_hz,
            "actual_fourcc": self._actual_fourcc(capture),
            "actual_fps": self._capture_prop(capture, self._cv2.CAP_PROP_FPS) if self._cv2 is not None else None,
            "mean_brightness": float(frame.mean()),
            "timestamp": time.time(),
            "image_file": self._latest_image_file,
            "image_t_ns": self._latest_image_t_ns,
            "saved_this_frame": image_file is not None,
            "dropped_frames": self._image_writer.dropped_frames if self._image_writer else 0,
            "save_fps": self.save_fps,
            "actual_save_fps": self._actual_save_fps(),
        }
        if image_save_error:
            data["image_save_error"] = image_save_error
        return self.make_sample(data, t_ns=t_ns)

    def on_error(self, exc: Exception) -> None:
        _ = exc
        self._close_capture()

    def close(self) -> None:
        self._close_capture()
        if self._image_writer is not None:
            self._image_writer.stop()
            self._image_writer = None

    def _ensure_cv2(self, now_s: float):
        if self._cv2 is not None:
            return self._cv2
        if now_s < self._next_cv2_retry_s:
            return None
        try:
            import cv2
        except ImportError:
            self._next_cv2_retry_s = now_s + 1.0
            self.buffer.append(self.make_sample({}, ok=False, error="opencv_not_available"))
            return None
        self._cv2 = cv2
        return cv2

    def _ensure_capture(self, cv2):
        if self._capture is not None:
            return self._capture
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.camera_index!r}")
        self._configure_capture(cv2, capture)
        self._capture = capture
        return capture

    def _configure_capture(self, cv2, capture) -> None:
        if self.fourcc:
            if len(self.fourcc) != 4:
                raise RuntimeError(f"Camera fourcc must be 4 characters, got {self.fourcc!r}")
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.rate_hz > 0:
            capture.set(cv2.CAP_PROP_FPS, self.rate_hz)

    def _actual_fourcc(self, capture) -> str | None:
        if self._cv2 is None:
            return None
        value = int(self._capture_prop(capture, self._cv2.CAP_PROP_FOURCC) or 0)
        if value <= 0:
            return None
        chars = []
        for shift in (0, 8, 16, 24):
            char = chr((value >> shift) & 0xFF)
            chars.append(char if char.isprintable() else "?")
        return "".join(chars)

    @staticmethod
    def _capture_prop(capture, prop_id: int) -> float | None:
        try:
            return float(capture.get(prop_id))
        except Exception:
            return None

    def _should_save(self, now_s: float) -> bool:
        if not self.save_frames or self.save_fps <= 0:
            return False
        if not self.save_combined_frame:
            return False
        if now_s < self._next_save_s:
            return False
        self._next_save_s = now_s + (1.0 / self.save_fps)
        return True

    def _enqueue_frame(
        self,
        frame: Any,
        t_ns: int,
        width: int,
        height: int,
    ) -> tuple[str | None, int | None, str | None]:
        if self.log_dir is None:
            return None, None, "log_dir_not_configured"
        writer = self._ensure_image_writer()
        filename = f"camera/frame_{self._frame_id:08d}.{self.image_format}"
        task = ImageSaveTask(
            frame=frame.copy(),
            t_ns=t_ns,
            frame_id=self._frame_id,
            filename=filename,
            width=int(width),
            height=int(height),
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
        )
        if not writer.enqueue(task):
            return None, None, "image_queue_full"
        if self._first_save_s is None:
            self._first_save_s = time.monotonic()
        self._last_save_s = time.monotonic()
        return filename, t_ns, None

    def _ensure_image_writer(self) -> ImageWriter:
        if self._image_writer is not None:
            return self._image_writer
        if self.log_dir is None:
            raise RuntimeError("Cannot create ImageWriter without a log directory.")
        self._image_writer = ImageWriter(
            log_dir=self.log_dir,
            index_file=self.camera_index_file,
            max_queue_size=self.max_image_queue_size,
            flush_interval_s=self.flush_interval_s,
        )
        self._image_writer.start()
        return self._image_writer

    def _actual_save_fps(self) -> float | None:
        writer = self._image_writer
        if writer is None or writer.saved_frames <= 1:
            return None
        if self._first_save_s is None or self._last_save_s is None:
            return None
        elapsed_s = max(0.001, self._last_save_s - self._first_save_s)
        return writer.saved_frames / elapsed_s

    def _close_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
