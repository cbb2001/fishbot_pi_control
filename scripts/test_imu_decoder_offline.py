from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.runtime.uart_workers import (  # noqa: E402
    YESENSE_TID_MODULUS,
    YesenseStreamDecoder,
    create_yesense_decoder,
)


def replay_raw_imu(path: Path) -> dict[str, Any]:
    decoder = create_yesense_decoder()
    if decoder is None:
        raise RuntimeError("Yesense decoder is not available.")
    stream = YesenseStreamDecoder(decoder)

    input_log_lines = 0
    input_raw_bytes = 0
    input_t_ns: list[int] = []
    frames = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_log_lines += 1
            try:
                record = json.loads(line)
                raw_hex = record.get("data", {}).get("raw_hex", "")
                raw = bytes.fromhex(raw_hex) if raw_hex else b""
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid raw IMU JSONL at line {line_number}: {exc}") from exc
            input_raw_bytes += len(raw)
            if "t_ns" in record:
                input_t_ns.append(int(record["t_ns"]))
            frames.extend(stream.feed(raw))

    tids = [int(frame.values.get("tid", 0)) % YESENSE_TID_MODULUS for frame in frames]
    tid_discontinuities = 0
    total_missing_frames = 0
    for previous, current in zip(tids, tids[1:]):
        distance = (current - previous) % YESENSE_TID_MODULUS
        if distance != 1:
            tid_discontinuities += 1
            if distance > 1:
                total_missing_frames += distance - 1

    acc_norms = [
        math.sqrt(
            float(frame.values.get("acc_x", 0.0)) ** 2
            + float(frame.values.get("acc_y", 0.0)) ** 2
            + float(frame.values.get("acc_z", 0.0)) ** 2
        )
        for frame in frames
    ]
    frame_sizes = [len(frame.raw) for frame in frames]

    estimated_sample_rate_hz: float | None = None
    if len(input_t_ns) >= 2 and input_t_ns[-1] > input_t_ns[0] and len(frames) >= 2:
        duration_s = (input_t_ns[-1] - input_t_ns[0]) / 1_000_000_000.0
        estimated_sample_rate_hz = (len(frames) - 1) / duration_s

    return {
        "input_log_lines": input_log_lines,
        "input_raw_bytes": input_raw_bytes,
        "decoded_frames": len(frames),
        "remaining_buffer_bytes": len(stream.buffer),
        "first_tid": tids[0] if tids else None,
        "last_tid": tids[-1] if tids else None,
        "tid_discontinuities": tid_discontinuities,
        "total_missing_frames": total_missing_frames,
        "estimated_sample_rate_hz": estimated_sample_rate_hz,
        "mean_acc_norm_mps2": statistics.fmean(acc_norms) if acc_norms else None,
        "std_acc_norm_mps2": statistics.pstdev(acc_norms) if acc_norms else None,
        "min_frame_size": min(frame_sizes) if frame_sizes else None,
        "max_frame_size": max(frame_sizes) if frame_sizes else None,
        "discarded_bytes": stream.discarded_bytes,
        "resync_count": stream.decode_resync_count,
        "crc_error_count": stream.crc_error_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay raw_hex chunks from an existing raw_imu.jsonl without modifying it."
    )
    parser.add_argument("raw_imu_jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.raw_imu_jsonl.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"raw IMU log does not exist: {path}")
    print(json.dumps(replay_raw_imu(path), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
