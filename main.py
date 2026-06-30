from __future__ import annotations

from pathlib import Path

from control.safety import load_robot_config


def main() -> None:
    config_path = Path(__file__).resolve().parent / "config" / "robot.yaml"
    config = load_robot_config(config_path)
    robot_name = config.get("robot", {}).get("name", "fishbot_pi_control")

    print(f"{robot_name}: project loaded.")
    print("No motion is started by main.py.")
    print("Use scripts/test_i2c.py and scripts/test_uart.py before any servo test.")


if __name__ == "__main__":
    main()

