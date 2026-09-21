"""Atomic live status publisher used by the SSH read-only watcher."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class LiveStatusPublisher:
    def __init__(self, path: str | Path, interval_s: float = 1.0) -> None:
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self._last_write = 0.0

    def publish(self, status: dict[str, Any], *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_write < self.interval_s:
            return False
        payload = dict(status)
        payload.setdefault("t_ns", time.monotonic_ns())
        payload.setdefault("updated_at", time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".rl-status-", dir=str(self.path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            self._last_write = now
            return True
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def close(self) -> None:
        return None
