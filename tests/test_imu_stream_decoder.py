from __future__ import annotations

import math
import random
import struct
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.runtime.ring_buffer import RingBuffer
from control.runtime.data_logger import RawSensorLoggers
from control.runtime.uart_workers import (
    IMUWorker,
    YESENSE_TID_MODULUS,
    YesenseStreamDecoder,
    create_yesense_decoder,
)


def _crc16(data: bytes) -> int:
    check_a = 0
    check_b = 0
    for value in data:
        check_a += value
        check_b += check_a
    return ((check_b % 256) << 8) + (check_a % 256)


def build_frame(
    tid: int,
    *,
    acc: tuple[float, float, float] = (0.0, 0.0, 9.80665),
    corrupt_crc: bool = False,
) -> bytes:
    factor = 1_000_000
    payload = b"".join(
        (
            bytes((0x10, 0x0C)) + struct.pack("<iii", *(round(value * factor) for value in acc)),
            bytes((0x20, 0x0C)) + struct.pack("<iii", 0, 0, 0),
            bytes((0x40, 0x0C)) + struct.pack("<iii", 0, 0, 0),
            bytes((0x41, 0x10)) + struct.pack("<iiii", factor, 0, 0, 0),
        )
    )
    prefix = b"YS" + struct.pack("<H", tid % YESENSE_TID_MODULUS) + bytes((len(payload),)) + payload
    crc = _crc16(prefix[2:])
    if corrupt_crc:
        crc ^= 0xFFFF
    return prefix + struct.pack("<H", crc)


def chunk_bytes(data: bytes, size: int) -> list[bytes]:
    return [data[index : index + size] for index in range(0, len(data), size)]


class FakeSerial:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = deque(chunks)
        self.read_calls = 0

    def read(self, size: int) -> bytes:
        if size != 256:
            raise AssertionError(f"unexpected read size: {size}")
        self.read_calls += 1
        return self.chunks.popleft() if self.chunks else b""

    def close(self) -> None:
        pass


class AdvancingClock:
    def __init__(self, start_ns: int = 1_000_000_000, step_ns: int = 20_000_000) -> None:
        self.now_ns = start_ns - step_ns
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.now_ns += self.step_ns
        return self.now_ns


def make_stream() -> YesenseStreamDecoder:
    decoder = create_yesense_decoder()
    if decoder is None:
        raise AssertionError("vendor decoder unavailable")
    return YesenseStreamDecoder(decoder)


def decode_chunks(chunks: list[bytes]) -> tuple[list, YesenseStreamDecoder]:
    stream = make_stream()
    frames = []
    for chunk in chunks:
        frames.extend(stream.feed(chunk))
    return frames, stream


def make_worker(chunks: list[bytes], *, clock_step_ns: int = 20_000_000) -> tuple[IMUWorker, FakeSerial]:
    decoder = create_yesense_decoder()
    if decoder is None:
        raise AssertionError("vendor decoder unavailable")
    worker = IMUWorker(RingBuffer(4000), sample_rate_hz=200)
    serial_port = FakeSerial(chunks)
    worker._serial = serial_port
    worker._decoder = decoder
    worker._clock_ns = AdvancingClock(step_ns=clock_step_ns)
    return worker, serial_port


class YesenseStreamDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = [build_frame(tid) for tid in range(100, 140)]
        self.stream_bytes = b"".join(self.frames)

    def assert_clean(self, stream: YesenseStreamDecoder) -> None:
        self.assertEqual(stream.decode_resync_count, 0)
        self.assertEqual(stream.discarded_bytes, 0)
        self.assertEqual(stream.crc_error_count, 0)

    def test_one_complete_frame_in_one_input(self) -> None:
        frames, stream = decode_chunks([self.frames[0]])
        self.assertEqual([frame.values["tid"] for frame in frames], [100])
        self.assertEqual(frames[0].raw, self.frames[0])
        self.assertEqual(len(stream.buffer), 0)
        self.assert_clean(stream)

    def test_one_complete_frame_byte_by_byte(self) -> None:
        frames, stream = decode_chunks(chunk_bytes(self.frames[0], 1))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].raw, self.frames[0])
        self.assert_clean(stream)

    def test_fixed_chunk_sizes(self) -> None:
        for size in (7, 32, 67, 128, 256):
            with self.subTest(size=size):
                frames, stream = decode_chunks(chunk_bytes(self.stream_bytes, size))
                self.assertEqual([frame.values["tid"] for frame in frames], list(range(100, 140)))
                self.assertEqual(len(stream.buffer), 0)
                self.assert_clean(stream)

    def test_random_chunk_sizes(self) -> None:
        rng = random.Random(20260713)
        chunks = []
        index = 0
        while index < len(self.stream_bytes):
            size = rng.randint(1, 300)
            chunks.append(self.stream_bytes[index : index + size])
            index += size
        frames, stream = decode_chunks(chunks)
        self.assertEqual([frame.values["tid"] for frame in frames], list(range(100, 140)))
        self.assertEqual(len(stream.buffer), 0)
        self.assert_clean(stream)

    def test_multiple_frames_in_one_input(self) -> None:
        frames, stream = decode_chunks([self.stream_bytes])
        self.assertEqual(len(frames), len(self.frames))
        self.assertEqual([len(frame.raw) for frame in frames], [67] * len(self.frames))
        self.assert_clean(stream)

    def test_half_frame_multiple_frames_and_half_frame(self) -> None:
        first = build_frame(10)
        middle = [build_frame(tid) for tid in (11, 12, 13)]
        last = build_frame(14)
        stream = make_stream()
        self.assertEqual(stream.feed(first[:20]), [])
        decoded = stream.feed(first[20:] + b"".join(middle) + last[:30])
        self.assertEqual([frame.values["tid"] for frame in decoded], [10, 11, 12, 13])
        self.assertEqual(bytes(stream.buffer), last[:30])
        decoded = stream.feed(last[30:])
        self.assertEqual([frame.values["tid"] for frame in decoded], [14])
        self.assertEqual(len(stream.buffer), 0)

    def test_garbage_before_header_is_discarded_and_valid_frame_survives(self) -> None:
        garbage = b"garbage-before-header"
        frames, stream = decode_chunks([garbage + self.frames[0]])
        self.assertEqual([frame.values["tid"] for frame in frames], [100])
        self.assertEqual(stream.discarded_bytes, len(garbage))
        self.assertEqual(stream.decode_resync_count, 1)

    def test_crc_error_followed_by_valid_frame(self) -> None:
        invalid = build_frame(200, corrupt_crc=True)
        valid = build_frame(201)
        frames, stream = decode_chunks([invalid + valid])
        self.assertEqual([frame.values["tid"] for frame in frames], [201])
        self.assertEqual(stream.crc_error_count, 1)
        self.assertGreaterEqual(stream.discarded_bytes, len(invalid))
        self.assertEqual(stream.decode_resync_count, 1)

    def test_incomplete_tail_is_the_only_remaining_buffer_data(self) -> None:
        tail = build_frame(141)[:19]
        frames, stream = decode_chunks([self.stream_bytes + tail])
        self.assertEqual(len(frames), len(self.frames))
        self.assertEqual(bytes(stream.buffer), tail)
        self.assert_clean(stream)

    def test_no_progress_does_not_loop_and_hard_resync_uses_boundary_scan(self) -> None:
        class NoProgressDecoder:
            @staticmethod
            def proc_data(data, data_len, result, dbg_flg):
                return False

            @staticmethod
            def calc_crc16(data, data_len):
                return _crc16(bytes(data[:data_len]))

        stream = YesenseStreamDecoder(
            NoProgressDecoder(), hard_limit_bytes=1024, hard_limit_stall_reads=2
        )
        self.assertEqual(stream.feed(b"x" * 1100), [])
        self.assertEqual(stream.feed(b"y"), [])
        self.assertEqual(len(stream.buffer), 0)
        self.assertEqual(stream.decode_resync_count, 1)
        self.assertEqual(stream.discarded_bytes, 1101)

    def test_every_raw_hex_independently_decodes_to_same_frame(self) -> None:
        decoded, stream = decode_chunks([self.stream_bytes])
        self.assert_clean(stream)
        for expected_tid, frame in zip(range(100, 140), decoded):
            replayed, replay_stream = decode_chunks([bytes.fromhex(frame.raw.hex())])
            self.assertEqual(len(replayed), 1)
            self.assertEqual(replayed[0].values["tid"], expected_tid)
            self.assertEqual(replayed[0].raw, frame.raw)
            self.assert_clean(replay_stream)


class IMUWorkerTests(unittest.TestCase):
    def test_one_256_byte_read_with_prior_tail_creates_four_samples(self) -> None:
        frames = [build_frame(tid) for tid in range(20, 24)]
        worker, serial_port = make_worker([frames[0][12:] + b"".join(frames[1:])])
        worker._decode_buffer.extend(frames[0][:12])
        samples = worker.read_once()
        self.assertIsInstance(samples, list)
        assert isinstance(samples, list)
        self.assertEqual(serial_port.read_calls, 1)
        self.assertEqual(len(samples), 4)
        self.assertEqual([sample.data["tid"] for sample in samples], [20, 21, 22, 23])
        self.assertEqual([sample.data["tid_gap"] for sample in samples], [0, 0, 0, 0])
        self.assertEqual([sample.data["frame_size_bytes"] for sample in samples], [67] * 4)
        self.assertEqual([sample.data["read_chunk_size_bytes"] for sample in samples], [256] * 4)
        self.assertTrue(all(len(bytes.fromhex(sample.data["raw_hex"])) == 67 for sample in samples))
        self.assertEqual(len(worker._decode_buffer), 0)

    def test_incomplete_input_returns_none(self) -> None:
        frame = build_frame(1)
        worker, _ = make_worker([frame[:30], frame[30:]])
        self.assertIsNone(worker.read_once())
        samples = worker.read_once()
        self.assertIsInstance(samples, list)
        assert isinstance(samples, list)
        self.assertEqual(len(samples), 1)

    def test_tid_wrap_and_missing_frame_gap(self) -> None:
        frames = [build_frame(65535), build_frame(0), build_frame(3)]
        worker, _ = make_worker([b"".join(frames)])
        samples = worker.read_once()
        assert isinstance(samples, list)
        self.assertEqual([sample.data["tid"] for sample in samples], [65535, 0, 3])
        self.assertEqual([sample.data["tid_gap"] for sample in samples], [0, 0, 2])

    def test_timestamps_are_strict_and_follow_200_hz_tid_spacing(self) -> None:
        first_batch = b"".join(build_frame(tid) for tid in range(1000, 1004))
        second_batch = b"".join(build_frame(tid) for tid in range(1004, 1008))
        worker, _ = make_worker([first_batch, second_batch], clock_step_ns=20_000_000)
        first = worker.read_once()
        second = worker.read_once()
        assert isinstance(first, list) and isinstance(second, list)
        samples = first + second
        timestamps = [sample.t_ns for sample in samples]
        self.assertTrue(all(current > previous for previous, current in zip(timestamps, timestamps[1:])))
        self.assertEqual([b - a for a, b in zip(timestamps, timestamps[1:])], [5_000_000] * 7)
        self.assertLessEqual(first[-1].t_ns, 1_000_000_000)
        self.assertLessEqual(second[-1].t_ns, 1_020_000_000)

    def test_static_acceleration_norm_is_mps2_not_double_converted(self) -> None:
        worker, _ = make_worker([build_frame(1, acc=(0.1, -0.2, 9.80))])
        samples = worker.read_once()
        assert isinstance(samples, list)
        norm = math.sqrt(sum(value * value for value in samples[0].data["acc_mps2"]))
        self.assertAlmostEqual(norm, math.sqrt(0.1**2 + 0.2**2 + 9.8**2), places=6)
        self.assertLess(norm, 11.0)

    def test_ten_seconds_at_200_hz_produces_2000_unique_samples(self) -> None:
        all_bytes = b"".join(build_frame(tid) for tid in range(2000))
        chunks = chunk_bytes(all_bytes, 256)
        worker, _ = make_worker(chunks, clock_step_ns=19_100_000)
        samples = []
        for _ in chunks:
            batch = worker.read_once()
            if batch:
                assert isinstance(batch, list)
                samples.extend(batch)
        self.assertEqual(len(samples), 2000)
        self.assertEqual([sample.seq for sample in samples], list(range(1, 2001)))
        self.assertEqual([sample.data["tid_gap"] for sample in samples], [0] * 2000)
        self.assertLess(len(worker._decode_buffer), 67)
        assert worker._stream_decoder is not None
        self.assertEqual(worker._stream_decoder.decode_resync_count, 0)
        self.assertEqual(worker._stream_decoder.discarded_bytes, 0)
        self.assertEqual(worker._stream_decoder.crc_error_count, 0)
        timestamps = [sample.t_ns for sample in samples]
        self.assertTrue(all(current > previous for previous, current in zip(timestamps, timestamps[1:])))

    def test_sample_list_reaches_ring_buffer_and_raw_logger(self) -> None:
        worker, _ = make_worker([b"".join(build_frame(tid) for tid in range(50, 54))])
        samples = worker.read_once()
        assert isinstance(samples, list)
        worker._append_samples(samples)
        self.assertEqual(len(worker.buffer), 4)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_loggers = RawSensorLoggers(Path(temporary_directory), queue_maxsize=100)
            raw_loggers.start()
            raw_loggers.write_from_buffers({"imu": worker.buffer})
            raw_loggers.stop()
            output = Path(temporary_directory) / "raw_imu.jsonl"
            records = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(records), 4)
            self.assertEqual(raw_loggers.stats()["imu"]["dropped_count"], 0)


if __name__ == "__main__":
    unittest.main()
