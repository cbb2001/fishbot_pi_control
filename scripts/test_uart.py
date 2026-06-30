from __future__ import annotations

import argparse
import glob
import time

from _bootstrap import add_project_root

add_project_root()

from control.safety import ensure_not_windows_hardware_run, load_robot_config  # noqa: E402


def list_uart_devices() -> list[str]:
    patterns = ["/dev/ttyAMA*", "/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial*"]
    devices: list[str] = []
    for pattern in patterns:
        devices.extend(glob.glob(pattern))
    return sorted(set(devices))


def main() -> None:
    ensure_not_windows_hardware_run()

    parser = argparse.ArgumentParser(description="List UART devices and optionally read one port.")
    parser.add_argument("--port", default=None, help="Serial port to open for a read-only smoke test.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Read duration when --port is provided.")
    args = parser.parse_args()

    config = load_robot_config()
    uart = config.get("uart", {})
    baudrate = int(uart.get("baudrate", 115200))
    timeout = float(uart.get("read_timeout_seconds", 1.0))

    devices = list_uart_devices()
    if devices:
        print("UART-like devices:")
        for device in devices:
            print(f"  {device}")
    else:
        print("No UART-like devices found.")

    if not args.port:
        print("No --port specified; not opening a serial device.")
        return

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for UART read tests.") from exc

    print(f"Opening {args.port} at {baudrate} baud for read-only test...")
    deadline = time.monotonic() + max(0.1, args.seconds)
    with serial.Serial(args.port, baudrate, timeout=timeout) as ser:
        while time.monotonic() < deadline:
            data = ser.readline()
            if data:
                print(data.hex(" "))


if __name__ == "__main__":
    main()

