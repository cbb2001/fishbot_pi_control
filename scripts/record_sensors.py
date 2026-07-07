from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from main import main as fishbot_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record synchronized sensor samples to local JSONL logs.")
    parser.add_argument("--duration", type=float, default=None, help="Record duration in seconds.")
    parser.add_argument("--mock", action="store_true", help="Use fake sensor data without touching hardware.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    argv = ["--mode", "record"]
    if args.duration is not None:
        argv.extend(["--duration", str(args.duration)])
    if args.mock:
        argv.append("--mock")
    return fishbot_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

