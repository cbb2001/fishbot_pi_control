from __future__ import annotations

import socket
import threading
from typing import Any

from control.runtime.event_logger import EventLogger


class LinkManager:
    def __init__(
        self,
        event_logger: EventLogger,
        *,
        host: str | None = None,
        port: int = 22,
        interval_s: float = 5.0,
        timeout_s: float = 1.0,
    ) -> None:
        self.event_logger = event_logger
        self.host = host
        self.port = int(port)
        self.interval_s = max(0.5, float(interval_s))
        self.timeout_s = max(0.1, float(timeout_s))
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_online: bool | None = None

    def start(self) -> None:
        if not self.host:
            return
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="link-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            online = self._check_once()
            if self._last_online is None:
                self._last_online = online
            elif online != self._last_online:
                self.event_logger.write(
                    "network_restored" if online else "network_lost",
                    {"host": self.host, "port": self.port},
                )
                self._last_online = online
            self.stop_event.wait(self.interval_s)

    def _check_once(self) -> bool:
        try:
            with socket.create_connection((str(self.host), self.port), timeout=self.timeout_s):
                return True
        except OSError:
            return False


def link_manager_from_config(config: dict[str, Any], event_logger: EventLogger) -> LinkManager:
    link_cfg = config.get("link", {})
    return LinkManager(
        event_logger,
        host=link_cfg.get("heartbeat_host"),
        port=int(link_cfg.get("heartbeat_port", 22)),
        interval_s=float(link_cfg.get("heartbeat_interval_s", 5.0)),
        timeout_s=float(link_cfg.get("heartbeat_timeout_s", 1.0)),
    )

