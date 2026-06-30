from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IMUConfig:
    port: str = "/dev/ttyAMA1"
    baudrate: int = 115200
    timeout: float = 1.0


class IMUSerial:
    def __init__(self, config: dict[str, Any]) -> None:
        uart = config.get("uart", {})
        self.config = IMUConfig(
            port=str(uart.get("imu_port", "/dev/ttyAMA1")),
            baudrate=int(uart.get("baudrate", 115200)),
            timeout=float(uart.get("read_timeout_seconds", 1.0)),
        )

    def read_line(self) -> bytes:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("IMU serial reading requires pyserial on the Raspberry Pi.") from exc

        with serial.Serial(self.config.port, self.config.baudrate, timeout=self.config.timeout) as ser:
            return ser.readline()

