from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Iterable


def _sample_t_ns(sample: Any) -> int:
    if hasattr(sample, "t_ns"):
        return int(sample.t_ns)
    if isinstance(sample, dict) and "t_ns" in sample:
        return int(sample["t_ns"])
    raise AttributeError("Sample must expose t_ns.")


class RingBuffer:
    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("RingBuffer maxlen must be positive.")
        self.maxlen = int(maxlen)
        self._items: deque[Any] = deque(maxlen=self.maxlen)
        self._lock = Lock()

    def append(self, sample: Any) -> None:
        _sample_t_ns(sample)
        with self._lock:
            self._items.append(sample)

    def latest(self) -> Any | None:
        with self._lock:
            if not self._items:
                return None
            return self._items[-1]

    def get_latest_before(self, t_ns: int) -> Any | None:
        with self._lock:
            for sample in reversed(self._items):
                if _sample_t_ns(sample) <= t_ns:
                    return sample
        return None

    def get_nearest(self, t_ns: int, max_age_ns: int | None = None) -> Any | None:
        nearest = None
        nearest_delta = None
        with self._lock:
            items: Iterable[Any] = tuple(self._items)

        for sample in items:
            delta = abs(_sample_t_ns(sample) - t_ns)
            if nearest_delta is None or delta < nearest_delta:
                nearest = sample
                nearest_delta = delta

        if nearest is None or nearest_delta is None:
            return None
        if max_age_ns is not None and nearest_delta > max_age_ns:
            return None
        return nearest

    def get_bracketing(self, t_ns: int) -> tuple[Any | None, Any | None]:
        before = None
        after = None
        with self._lock:
            items = tuple(self._items)

        for sample in items:
            sample_t = _sample_t_ns(sample)
            if sample_t <= t_ns:
                before = sample
            if sample_t >= t_ns:
                after = sample
                break
        return before, after

    def snapshot(self) -> list[Any]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

