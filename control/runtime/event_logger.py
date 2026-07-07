from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from control.runtime.data_logger import JsonlLogger


class EventLogger:
    def __init__(
        self,
        path: Path,
        *,
        flush_interval_s: float = 1.0,
        queue_maxsize: int = 10000,
    ) -> None:
        self.logger = JsonlLogger(
            path,
            flush_interval_s=flush_interval_s,
            queue_maxsize=queue_maxsize,
        )

    def start(self) -> None:
        self.logger.start()

    def write(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        t_ns: int | None = None,
    ) -> bool:
        return self.logger.write(
            {
                "t_ns": t_ns if t_ns is not None else time.monotonic_ns(),
                "event_type": event_type,
                "data": data or {},
            }
        )

    def stop(self) -> None:
        self.logger.stop()

