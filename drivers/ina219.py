from __future__ import annotations

from typing import Any


class INA219Reader:
    def __init__(self, config: dict[str, Any]) -> None:
        i2c_config = config.get("i2c", {})
        self.address = int(i2c_config.get("ina219_address", 0x40))

    def read(self) -> dict[str, float]:
        try:
            import board
            from adafruit_ina219 import INA219
        except ImportError as exc:
            raise RuntimeError("INA219 support requires adafruit-circuitpython-ina219 on the Pi.") from exc

        sensor = INA219(board.I2C(), addr=self.address)
        return {
            "bus_voltage_v": float(sensor.bus_voltage),
            "shunt_voltage_v": float(sensor.shunt_voltage),
            "current_ma": float(sensor.current),
            "power_w": float(sensor.power),
        }

