from __future__ import annotations

import threading
import time
from typing import Any

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample


class BaseSensorWorker:
    def __init__(
        self,
        name: str,
        buffer: RingBuffer,
        *,
        stop_event: threading.Event | None = None,
        loop_delay_s: float = 0.0,
        error_backoff_s: float = 1.0,
    ) -> None:
        self.name = name
        self.buffer = buffer
        self.stop_event = stop_event or threading.Event()
        self.loop_delay_s = max(0.0, float(loop_delay_s))
        self.error_backoff_s = max(0.0, float(error_backoff_s))
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.fatal_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_guarded, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def read_once(self) -> SensorSample | list[SensorSample] | None:
        raise NotImplementedError

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                samples = self.read_once()
                self._append_samples(samples)
                self._sleep_interruptible(self.loop_delay_s)
            except Exception as exc:
                self.buffer.append(self.make_sample({}, ok=False, error=f"{type(exc).__name__}: {exc}"))
                self.on_error(exc)
                self._sleep_interruptible(self.error_backoff_s)

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:
            self.fatal_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.close()
            except BaseException as exc:
                if self.fatal_error is None:
                    self.fatal_error = f"{type(exc).__name__}: {exc}"

    def on_error(self, exc: Exception) -> None:
        _ = exc

    def close(self) -> None:
        pass

    def make_sample(
        self,
        data: dict[str, Any],
        *,
        ok: bool = True,
        error: str | None = None,
        t_ns: int | None = None,
    ) -> SensorSample:
        self._seq += 1
        return SensorSample(
            name=self.name,
            t_ns=t_ns if t_ns is not None else time.monotonic_ns(),
            seq=self._seq,
            data=data,
            ok=ok,
            error=error,
        )

    def _append_samples(self, samples: SensorSample | list[SensorSample] | None) -> None:
        if samples is None:
            return
        if isinstance(samples, SensorSample):
            self.buffer.append(samples)
            return
        for sample in samples:
            self.buffer.append(sample)

    def _sleep_interruptible(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.stop_event.wait(seconds)
