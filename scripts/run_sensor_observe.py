from __future__ import annotations

import argparse

from _bootstrap import add_project_root

add_project_root()

from main import main as fishbot_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live synchronized sensor observe mode.")
    parser.add_argument("--mock", action="store_true", help="Use fake sensor data without touching hardware.")
    parser.add_argument("--duration", type=float, default=None, help="Optional observe duration in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    argv = ["--mode", "observe"]
    if args.duration is not None:
        argv.extend(["--duration", str(args.duration)])
    if args.mock:
        argv.append("--mock")
    return fishbot_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

