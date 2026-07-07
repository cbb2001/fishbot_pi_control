from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()

from control.safety import ensure_not_windows_hardware_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a detached local sensor recording process on the Pi.")
    parser.add_argument("--duration", type=float, default=300.0, help="Recording duration in seconds. Default: 300.")
    parser.add_argument("--mock", action="store_true", help="Use fake sensor data without touching hardware.")
    parser.add_argument("--stdout", default="logs/last_run.out", help="Detached process stdout/stderr file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mock:
        ensure_not_windows_hardware_run()

    stdout_path = Path(args.stdout)
    if not stdout_path.is_absolute():
        stdout_path = PROJECT_ROOT / stdout_path
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "record_sensors.py"),
        "--duration",
        str(args.duration),
    ]
    if args.mock:
        command.append("--mock")

    stdout_handle = stdout_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stdout_handle.close()

    print("Started detached sensor recorder.")
    print(f"PID: {process.pid}")
    print(f"Output: {stdout_path}")
    print("Logs will be written under the project logs directory.")
    print("Equivalent manual nohup command:")
    print(
        "  nohup python3 scripts/record_sensors.py "
        f"--duration {args.duration:g} > logs/last_run.out 2>&1 &"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

