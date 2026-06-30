from __future__ import annotations

import argparse
from datetime import datetime

from _bootstrap import add_project_root

add_project_root()

from control.safety import ensure_not_windows_hardware_run, load_robot_config, sleep_safely  # noqa: E402
from drivers.depth_sensor import DepthSensor  # noqa: E402


def main() -> None:
    ensure_not_windows_hardware_run()

    parser = argparse.ArgumentParser(description="Read MS5837 depth sensor data over I2C.")
    parser.add_argument("--samples", type=int, default=5, help="Number of readings to print. Use 0 to run until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings.")
    parser.add_argument("--zero", action="store_true", help="Use the first reading as surface pressure.")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be 0 or a positive integer.")

    config = load_robot_config()
    sensor = DepthSensor(config)

    try:
        print(f"Depth sensor I2C address: 0x{sensor.address:02X} on bus {sensor.bus}")
        if not sensor.begin():
            raise SystemExit("Depth sensor init failed. Check I2C wiring/address.")

        if args.zero:
            sensor.zero()
            print(f"Zeroed surface pressure: {sensor.surface_pressure_mbar:.2f} mbar")

        index = 0
        try:
            while args.samples == 0 or index < args.samples:
                reading = sensor.read()
                timestamp = datetime.now().isoformat(timespec="seconds")
                print(
                    f"time={timestamp} "
                    f"sample={index + 1} "
                    f"temperature={reading.temperature_c:.2f} C "
                    f"pressure={reading.pressure_mbar:.2f} mbar "
                    f"depth={reading.depth_cm:.2f} cm"
                )
                index += 1
                if args.samples == 0 or index < args.samples:
                    sleep_safely(args.interval)
        except KeyboardInterrupt:
            print("Depth sensor sampling stopped by user.")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
