from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from control.safety import ensure_not_windows_hardware_run, load_robot_config  # noqa: E402
from drivers.pca9685_servo import PCA9685ServoController  # noqa: E402


def main() -> None:
    ensure_not_windows_hardware_run()

    config = load_robot_config()
    controller = PCA9685ServoController(config)
    controller.stop_all()
    print("Emergency stop complete: PWM released for configured channels.")


if __name__ == "__main__":
    main()

