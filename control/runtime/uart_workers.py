from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.sensor_worker import BaseSensorWorker


G_TO_MPS2 = 9.80665


def _yesense_default_output() -> dict[str, Any]:
    return {
        "tid": 0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "q0": 1.0,
        "q1": 0.0,
        "q2": 0.0,
        "q3": 0.0,
        "sensor_temp": 0.0,
        "acc_x": 0.0,
        "acc_y": 0.0,
        "acc_z": 0.0,
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "gyro_z": 0.0,
        "status": 0,
    }


class IMUWorker(BaseSensorWorker):
    def __init__(
        self,
        buffer: RingBuffer,
        *,
        port: str = "/dev/ttyAMA4",
        baudrate: int = 460800,
        timeout_s: float = 0.1,
    ) -> None:
        super().__init__("imu", buffer, loop_delay_s=0.0, error_backoff_s=1.0)
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self._serial = None
        self._decoder = None
        self._decode_buffer = bytearray()
        self._decoded = _yesense_default_output()

    def read_once(self) -> SensorSample | None:
        serial_port = self._ensure_serial()
        data = serial_port.read(256)
        if not data:
            return None

        decoder = self._ensure_decoder()
        if decoder is None:
            return self.make_sample(
                {"raw_hex": data.hex()},
                ok=False,
                error="yesense_decoder_not_available",
            )

        self._decode_buffer.extend(data)
        if len(self._decode_buffer) > 8192:
            self._decode_buffer = self._decode_buffer[-4096:]

        if decoder.proc_data(self._decode_buffer, len(self._decode_buffer), self._decoded, False):
            return self.make_sample(self._standard_imu_data(data))
        return None

    def on_error(self, exc: Exception) -> None:
        _ = exc
        self._close_serial()

    def close(self) -> None:
        self._close_serial()

    def _ensure_serial(self):
        if self._serial is not None:
            return self._serial
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for IMU UART reads.") from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        return self._serial

    def _ensure_decoder(self):
        if self._decoder is not None:
            return self._decoder
        decoder_dir = (
            Path(__file__).resolve().parents[2]
            / "drivers"
            / "Yesense-Decode-Python3-V2.0"
            / "Yesense-Decode-Python3-V2.0"
        )
        if str(decoder_dir) not in sys.path:
            sys.path.insert(0, str(decoder_dir))
        try:
            from yis_std_dec import std_decoder
        except ImportError:
            return None
        self._decoder = std_decoder()
        return self._decoder

    def _standard_imu_data(self, raw: bytes) -> dict[str, Any]:
        acc_g = [
            float(self._decoded.get("acc_x", 0.0)),
            float(self._decoded.get("acc_y", 0.0)),
            float(self._decoded.get("acc_z", 0.0)),
        ]
        gyro_dps = [
            float(self._decoded.get("gyro_x", 0.0)),
            float(self._decoded.get("gyro_y", 0.0)),
            float(self._decoded.get("gyro_z", 0.0)),
        ]
        return {
            "acc_mps2": [value * G_TO_MPS2 for value in acc_g],
            "gyro_radps": [math.radians(value) for value in gyro_dps],
            "roll_deg": float(self._decoded.get("roll", 0.0)),
            "pitch_deg": float(self._decoded.get("pitch", 0.0)),
            "yaw_deg": float(self._decoded.get("yaw", 0.0)),
            "quat": [
                float(self._decoded.get("q0", 1.0)),
                float(self._decoded.get("q1", 0.0)),
                float(self._decoded.get("q2", 0.0)),
                float(self._decoded.get("q3", 0.0)),
            ],
            "temperature_c": float(self._decoded.get("sensor_temp", 0.0)),
            "raw_hex": raw.hex(),
            "tid": int(self._decoded.get("tid", 0)),
            "status": int(self._decoded.get("status", 0)),
            "unit_note": "Yesense acc assumed g, gyro assumed deg/s; verify with hardware.",
        }

    def _close_serial(self) -> None:
        serial_port = self._serial
        self._serial = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass


class UWBWorker(BaseSensorWorker):
    def __init__(
        self,
        buffer: RingBuffer,
        *,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout_s: float = 0.2,
        invalid_interval_s: float = 0.5,
    ) -> None:
        super().__init__("uwb", buffer, loop_delay_s=0.0, error_backoff_s=1.0)
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.invalid_interval_s = float(invalid_interval_s)
        self._serial = None
        self._last_invalid_s = 0.0

    def read_once(self) -> SensorSample | None:
        serial_port = self._ensure_serial()
        data = serial_port.read(256)
        if not data:
            now_s = time.monotonic()
            if now_s - self._last_invalid_s < self.invalid_interval_s:
                return None
            self._last_invalid_s = now_s
            return self.make_sample({}, ok=False, error="no_signal_or_timeout")

        text = data.decode("utf-8", errors="replace").strip()
        return self.make_sample(
            {
                "raw_hex": data.hex(),
                "raw_text": text,
                "parsed": False,
            },
            ok=True,
        )

    def on_error(self, exc: Exception) -> None:
        _ = exc
        self._close_serial()

    def close(self) -> None:
        self._close_serial()

    def _ensure_serial(self):
        if self._serial is not None:
            return self._serial
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for UWB UART reads.") from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        return self._serial

    def _close_serial(self) -> None:
        serial_port = self._serial
        self._serial = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass

