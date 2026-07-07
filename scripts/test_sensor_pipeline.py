from __future__ import annotations

import argparse
import sys

from _bootstrap import add_project_root

add_project_root()

from main import main as fishbot_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short sensor pipeline status test.")
    parser.add_argument("--mock", action="store_true", help="Use fake sensor data without touching hardware.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Status test duration. Default: 5 seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    argv = ["--mode", "status", "--status-seconds", str(args.seconds)]
    if args.mock:
        argv.append("--mock")
    return fishbot_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

