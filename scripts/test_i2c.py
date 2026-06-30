from __future__ import annotations

import subprocess

from _bootstrap import add_project_root

add_project_root()

from control.safety import ensure_not_windows_hardware_run, load_robot_config  # noqa: E402


def scan_with_smbus(bus_number: int) -> list[int]:
    from smbus2 import SMBus

    found: list[int] = []
    with SMBus(bus_number) as bus:
        for address in range(0x03, 0x78):
            try:
                bus.write_quick(address)
                found.append(address)
            except OSError:
                pass
    return found


def main() -> None:
    ensure_not_windows_hardware_run()
    config = load_robot_config()
    bus_number = int(config.get("i2c", {}).get("bus", 1))

    print(f"Scanning I2C bus {bus_number}...")
    try:
        found = scan_with_smbus(bus_number)
    except Exception as exc:  # noqa: BLE001
        print(f"smbus2 scan failed: {exc}")
        print("Falling back to i2cdetect if available...")
        subprocess.run(["i2cdetect", "-y", str(bus_number)], check=False)
        return

    if not found:
        print("No I2C devices detected.")
        return

    print("I2C devices:")
    for address in found:
        print(f"  0x{address:02X}")


if __name__ == "__main__":
    main()

