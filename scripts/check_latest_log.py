from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the newest fishbot sensor log directory.")
    parser.add_argument("--logs", default="logs", help="Base logs directory. Default: logs.")
    parser.add_argument("--lines", type=int, default=5, help="Number of recent JSONL lines to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.logs)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir

    latest = latest_log_dir(base_dir)
    if latest is None:
        print(f"No *_sensor_test log directories found under: {base_dir}")
        return 1

    sync_file = latest / "synchronized_sensors.jsonl"
    camera_index_file = latest / "camera_index.jsonl"
    camera_dir = latest / "camera"
    camera_count = count_camera_images(camera_dir)
    camera_rows = read_jsonl(camera_index_file)

    print(f"Latest log directory: {latest}")
    print(f"synchronized_sensors.jsonl: {'present' if sync_file.exists() else 'missing'}")
    print(f"camera_index.jsonl: {'present' if camera_index_file.exists() else 'missing'}")
    print(f"camera jpg count: {camera_count}")
    print_raw_log_summary(latest)
    fps = estimate_camera_fps(camera_rows)
    if fps is None:
        print("estimated camera save FPS: unavailable")
    else:
        print(f"estimated camera save FPS: {fps:.2f}")
    print("30s at save_fps=5 should be about 150 images; small drops are acceptable.")
    print()

    print_file(latest / "metadata.yaml", "metadata.yaml", args.lines)
    print_file(sync_file, "synchronized_sensors.jsonl", args.lines)
    print_file(latest / "events.jsonl", "events.jsonl", args.lines)
    print_recent_camera_index(camera_rows, args.lines)
    return 0


def latest_log_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [path for path in base_dir.glob("*_sensor_test*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def count_camera_images(camera_dir: Path) -> int:
    if not camera_dir.exists():
        return 0
    return len(list(camera_dir.glob("*.jpg"))) + len(list(camera_dir.glob("*.jpeg")))


def print_raw_log_summary(log_dir: Path) -> None:
    raw_files = (
        "raw_imu.jsonl",
        "raw_depth.jsonl",
        "raw_power.jsonl",
        "raw_uwb.jsonl",
        "raw_vision.jsonl",
    )
    print("raw sensor logs:")
    for filename in raw_files:
        path = log_dir / filename
        if not path.exists():
            print(f"  {filename}: missing")
            continue
        print(f"  {filename}: {path.stat().st_size} bytes, {count_lines(path)} lines")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def estimate_camera_fps(rows: list[dict]) -> float | None:
    saved = [row for row in rows if row.get("saved") and isinstance(row.get("t_ns"), int)]
    if len(saved) < 2:
        return None
    elapsed_s = (int(saved[-1]["t_ns"]) - int(saved[0]["t_ns"])) / 1_000_000_000.0
    if elapsed_s <= 0:
        return None
    return (len(saved) - 1) / elapsed_s


def print_file(path: Path, title: str, lines: int) -> None:
    print(f"== {title} ==")
    if not path.exists():
        print("missing")
        print()
        return
    for line in tail_lines(path, lines):
        print(line.rstrip())
    print()


def print_recent_camera_index(rows: list[dict], lines: int) -> None:
    print("== camera_index.jsonl recent ==")
    if not rows:
        print("missing or empty")
        print()
        return
    for row in rows[-max(1, int(lines)):]:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    print()


def tail_lines(path: Path, count: int) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return list(deque(handle, maxlen=max(1, int(count))))


if __name__ == "__main__":
    raise SystemExit(main())
