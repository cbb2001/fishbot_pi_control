from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.sensor_worker import BaseSensorWorker


YESENSE_HEADER = b"\x59\x53"
YESENSE_PROTOCOL_OVERHEAD_BYTES = 7
YESENSE_PAYLOAD_LENGTH_OFFSET = 4
YESENSE_TID_MODULUS = 1 << 16
DEFAULT_DECODE_HARD_LIMIT_BYTES = 65536
DEFAULT_DECODE_HARD_LIMIT_STALL_READS = 8


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


def create_yesense_decoder() -> Any | None:
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
    return std_decoder()


@dataclass(frozen=True)
class DecodedYesenseFrame:
    raw: bytes
    values: dict[str, Any]
    backlog_bytes: int


class YesenseStreamDecoder:
    """Drain every complete Yesense frame from an arbitrary byte stream."""

    def __init__(
        self,
        decoder: Any,
        *,
        buffer: bytearray | None = None,
        hard_limit_bytes: int = DEFAULT_DECODE_HARD_LIMIT_BYTES,
        hard_limit_stall_reads: int = DEFAULT_DECODE_HARD_LIMIT_STALL_READS,
    ) -> None:
        self.decoder = decoder
        self.buffer = buffer if buffer is not None else bytearray()
        self.hard_limit_bytes = max(1024, int(hard_limit_bytes))
        self.hard_limit_stall_reads = max(1, int(hard_limit_stall_reads))
        self.decode_resync_count = 0
        self.discarded_bytes = 0
        self.crc_error_count = 0
        self._stalled_read_count = 0
        self._resync_active = False

    def feed(self, data: bytes | bytearray) -> list[DecodedYesenseFrame]:
        if data:
            self.buffer.extend(data)

        frames, made_progress = self._drain()
        if made_progress:
            self._stalled_read_count = 0
        elif data:
            self._stalled_read_count += 1

        if self._needs_hard_resync() and self._hard_resync():
            self._stalled_read_count = 0
            recovered_frames, _ = self._drain()
            frames.extend(recovered_frames)
        return frames

    def _drain(self) -> tuple[list[DecodedYesenseFrame], bool]:
        frames: list[DecodedYesenseFrame] = []
        made_progress = False
        decoded = _yesense_default_output()

        while self.buffer:
            before_len = len(self.buffer)
            frame_start, frame_raw = self._peek_complete_frame()
            crc_invalid = frame_raw is not None and not self._frame_crc_valid(frame_raw)
            success = bool(self.decoder.proc_data(self.buffer, before_len, decoded, False))
            after_len = len(self.buffer)

            if success:
                if frame_raw is None:
                    raise RuntimeError("Yesense decoder succeeded without a complete candidate frame.")
                leading_bytes = frame_start or 0
                if leading_bytes:
                    self._record_discard(leading_bytes)
                frames.append(
                    DecodedYesenseFrame(
                        raw=frame_raw,
                        values=dict(decoded),
                        backlog_bytes=after_len,
                    )
                )
                self._resync_active = False
                made_progress = True
                continue

            if after_len < before_len:
                self._record_discard(before_len - after_len)
                if crc_invalid:
                    self.crc_error_count += 1
                made_progress = True
                continue

            # No success and no bytes removed means only an incomplete tail remains.
            break

        return frames, made_progress

    def _peek_complete_frame(self) -> tuple[int | None, bytes | None]:
        frame_start = self.buffer.find(YESENSE_HEADER)
        if frame_start < 0:
            return None, None
        length_position = frame_start + YESENSE_PAYLOAD_LENGTH_OFFSET
        if length_position >= len(self.buffer):
            return frame_start, None
        frame_size = YESENSE_PROTOCOL_OVERHEAD_BYTES + int(self.buffer[length_position])
        frame_end = frame_start + frame_size
        if frame_end > len(self.buffer):
            return frame_start, None
        return frame_start, bytes(self.buffer[frame_start:frame_end])

    def _frame_crc_valid(self, frame: bytes) -> bool:
        if len(frame) < YESENSE_PROTOCOL_OVERHEAD_BYTES:
            return False
        payload_length = int(frame[YESENSE_PAYLOAD_LENGTH_OFFSET])
        if len(frame) != YESENSE_PROTOCOL_OVERHEAD_BYTES + payload_length:
            return False
        crc_data = frame[2:-2]
        expected = int.from_bytes(frame[-2:], byteorder="little", signed=False)
        actual = int(self.decoder.calc_crc16(crc_data, len(crc_data)))
        return actual == expected

    def _record_discard(self, count: int) -> None:
        if count <= 0:
            return
        self.discarded_bytes += int(count)
        if not self._resync_active:
            self.decode_resync_count += 1
            self._resync_active = True

    def _needs_hard_resync(self) -> bool:
        return (
            len(self.buffer) > self.hard_limit_bytes
            and self._stalled_read_count >= self.hard_limit_stall_reads
        )

    def _hard_resync(self) -> bool:
        """Discard only bytes before a protocol-derived recoverable boundary."""
        valid_start: int | None = None
        incomplete_start: int | None = None
        search_from = 0

        while True:
            frame_start = self.buffer.find(YESENSE_HEADER, search_from)
            if frame_start < 0:
                break
            length_position = frame_start + YESENSE_PAYLOAD_LENGTH_OFFSET
            if length_position >= len(self.buffer):
                incomplete_start = frame_start
                break
            frame_size = YESENSE_PROTOCOL_OVERHEAD_BYTES + int(self.buffer[length_position])
            frame_end = frame_start + frame_size
            if frame_end > len(self.buffer):
                incomplete_start = frame_start
                break
            if self._frame_crc_valid(bytes(self.buffer[frame_start:frame_end])):
                valid_start = frame_start
                break
            search_from = frame_start + 1

        if valid_start is not None:
            discard_count = valid_start
        elif incomplete_start is not None:
            discard_count = incomplete_start
        elif self.buffer.endswith(YESENSE_HEADER[:1]):
            discard_count = len(self.buffer) - 1
        else:
            discard_count = len(self.buffer)

        if discard_count <= 0:
            return False
        del self.buffer[:discard_count]
        self._record_discard(discard_count)
        return True


class IMUWorker(BaseSensorWorker):
    def __init__(
        self,
        buffer: RingBuffer,
        *,
        port: str = "/dev/ttyAMA4",
        baudrate: int = 460800,
        timeout_s: float = 0.1,
        sample_rate_hz: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__(
            "imu",
            buffer,
            stop_event=stop_event,
            loop_delay_s=0.0,
            error_backoff_s=1.0,
        )
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        if float(sample_rate_hz) <= 0:
            raise ValueError("IMU sample_rate_hz must be positive.")
        self.sample_rate_hz = float(sample_rate_hz)
        self._sample_period_ns = max(1, int(round(1_000_000_000 / self.sample_rate_hz)))
        self._serial = None
        self._decoder = None
        self._decode_buffer = bytearray()
        self._stream_decoder: YesenseStreamDecoder | None = None
        self._last_tid: int | None = None
        self._last_sample_t_ns: int | None = None
        self._clock_ns: Callable[[], int] = time.monotonic_ns

    def read_once(self) -> SensorSample | list[SensorSample] | None:
        serial_port = self._ensure_serial()
        data = serial_port.read(256)
        read_done_ns = self._clock_ns()
        if not data:
            return None

        decoder = self._ensure_decoder()
        if decoder is None:
            return self.make_sample(
                {"raw_hex": data.hex()},
                ok=False,
                error="yesense_decoder_not_available",
            )

        stream_decoder = self._ensure_stream_decoder(decoder)
        decoded_frames = stream_decoder.feed(data)
        if not decoded_frames:
            return None

        tids = [int(frame.values.get("tid", 0)) % YESENSE_TID_MODULUS for frame in decoded_frames]
        timestamps = self._reconstruct_timestamps(tids, read_done_ns)
        samples: list[SensorSample] = []
        for frame, tid, t_ns in zip(decoded_frames, tids, timestamps):
            tid_gap = self._tid_gap(tid)
            samples.append(
                self.make_sample(
                    self._standard_imu_data(
                        frame,
                        tid=tid,
                        tid_gap=tid_gap,
                        read_chunk_size_bytes=len(data),
                        stream_decoder=stream_decoder,
                    ),
                    t_ns=t_ns,
                )
            )
            self._last_tid = tid
            self._last_sample_t_ns = t_ns
        return samples

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
        self._decoder = create_yesense_decoder()
        return self._decoder

    def _ensure_stream_decoder(self, decoder: Any) -> YesenseStreamDecoder:
        if self._stream_decoder is None:
            self._stream_decoder = YesenseStreamDecoder(decoder, buffer=self._decode_buffer)
        return self._stream_decoder

    def _tid_gap(self, tid: int) -> int:
        if self._last_tid is None:
            return 0
        distance = (int(tid) - self._last_tid) % YESENSE_TID_MODULUS
        return max(0, distance - 1)

    def _reconstruct_timestamps(self, tids: list[int], read_done_ns: int) -> list[int]:
        if not tids:
            return []

        last_t_ns = self._last_sample_t_ns
        if last_t_ns is None:
            timestamps = [int(read_done_ns)] * len(tids)
            for index in range(len(tids) - 2, -1, -1):
                distance = (tids[index + 1] - tids[index]) % YESENSE_TID_MODULUS
                timestamps[index] = timestamps[index + 1] - max(1, distance) * self._sample_period_ns
            return timestamps

        timestamps: list[int] = []
        previous_tid = self._last_tid
        current_t_ns = last_t_ns
        for tid in tids:
            if previous_tid is None:
                distance = 1
            else:
                distance = (tid - previous_tid) % YESENSE_TID_MODULUS
            current_t_ns += max(1, distance) * self._sample_period_ns
            timestamps.append(current_t_ns)
            previous_tid = tid

        if timestamps[-1] > int(read_done_ns):
            shift_ns = timestamps[-1] - int(read_done_ns)
            shifted = [t_ns - shift_ns for t_ns in timestamps]
            if shifted[0] > last_t_ns:
                return shifted

            available_ns = int(read_done_ns) - last_t_ns
            if available_ns < len(tids):
                raise RuntimeError("monotonic clock did not advance enough for strictly ordered IMU samples.")
            distances: list[int] = []
            previous_tid = self._last_tid
            for tid in tids:
                if previous_tid is None:
                    distance = 1
                else:
                    distance = (tid - previous_tid) % YESENSE_TID_MODULUS
                distances.append(max(1, distance))
                previous_tid = tid
            total_distance = sum(distances)
            cumulative = 0
            for index, distance in enumerate(distances):
                cumulative += distance
                t_ns = last_t_ns + (available_ns * cumulative) // total_distance
                minimum = last_t_ns + index + 1
                maximum = int(read_done_ns) - (len(tids) - index - 1)
                timestamps[index] = min(max(t_ns, minimum), maximum)

        return timestamps

    def _standard_imu_data(
        self,
        frame: DecodedYesenseFrame,
        *,
        tid: int,
        tid_gap: int,
        read_chunk_size_bytes: int,
        stream_decoder: YesenseStreamDecoder,
    ) -> dict[str, Any]:
        decoded = frame.values
        acc_mps2 = [
            float(decoded.get("acc_x", 0.0)),
            float(decoded.get("acc_y", 0.0)),
            float(decoded.get("acc_z", 0.0)),
        ]
        gyro_dps = [
            float(decoded.get("gyro_x", 0.0)),
            float(decoded.get("gyro_y", 0.0)),
            float(decoded.get("gyro_z", 0.0)),
        ]
        return {
            "acc_mps2": acc_mps2,
            "gyro_radps": [math.radians(value) for value in gyro_dps],
            "roll_deg": float(decoded.get("roll", 0.0)),
            "pitch_deg": float(decoded.get("pitch", 0.0)),
            "yaw_deg": float(decoded.get("yaw", 0.0)),
            "quat": [
                float(decoded.get("q0", 1.0)),
                float(decoded.get("q1", 0.0)),
                float(decoded.get("q2", 0.0)),
                float(decoded.get("q3", 0.0)),
            ],
            "temperature_c": float(decoded.get("sensor_temp", 0.0)),
            "raw_hex": frame.raw.hex(),
            "frame_size_bytes": len(frame.raw),
            "read_chunk_size_bytes": int(read_chunk_size_bytes),
            "decode_backlog_bytes": int(frame.backlog_bytes),
            "decode_resync_count": stream_decoder.decode_resync_count,
            "discarded_bytes": stream_decoder.discarded_bytes,
            "crc_error_count": stream_decoder.crc_error_count,
            "tid": int(tid),
            "tid_gap": int(tid_gap),
            "status": int(decoded.get("status", 0)),
            "unit_note": "Yesense acceleration used directly as m/s^2; gyro deg/s converted to rad/s.",
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
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__(
            "uwb",
            buffer,
            stop_event=stop_event,
            loop_delay_s=0.0,
            error_backoff_s=1.0,
        )
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
