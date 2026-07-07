from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PowerReading:
    voltage_v: float
    current_a: float
    power_w: float


class INA219Sensor:
    def __init__(
        self,
        *,
        bus: int = 1,
        address: int = 0x45,
        linear_cal_reading_ma: float = 1000.0,
        linear_cal_actual_ma: float = 1000.0,
    ) -> None:
        self.bus = int(bus)
        self.address = int(address)
        self.linear_cal_reading_ma = float(linear_cal_reading_ma)
        self.linear_cal_actual_ma = float(linear_cal_actual_ma)
        self._sensor = None

    def begin(self) -> bool:
        vendor_class = self._load_vendor_class()
        sensor = vendor_class(self.bus, self.address)
        if not sensor.begin():
            self._close_vendor(sensor)
            return False
        sensor.linear_cal(self.linear_cal_reading_ma, self.linear_cal_actual_ma)
        self._sensor = sensor
        return True

    def read(self) -> PowerReading:
        sensor = self._require_started()
        voltage_v = float(sensor.get_bus_voltage_V())
        current_a = float(sensor.get_current_mA()) / 1000.0
        power_w = float(sensor.get_power_mW()) / 1000.0
        return PowerReading(voltage_v=voltage_v, current_a=current_a, power_w=power_w)

    def close(self) -> None:
        if self._sensor is not None:
            self._close_vendor(self._sensor)
            self._sensor = None

    def _require_started(self):
        if self._sensor is None:
            raise RuntimeError("INA219Sensor.begin() must succeed before reading data.")
        return self._sensor

    def _load_vendor_class(self):
        try:
            import smbus  # noqa: F401
        except ImportError:
            try:
                import smbus2
            except ImportError as exc:
                raise RuntimeError("INA219 requires smbus or smbus2 on the Raspberry Pi.") from exc
            sys.modules.setdefault("smbus", smbus2)

        vendor_dir = (
            Path(__file__).resolve().parent
            / "DFRobot_INA219"
            / "Python"
            / "RespberryPi"
        )
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

        try:
            from DFRobot_INA219 import INA219
        except ImportError as exc:
            raise RuntimeError(f"Could not import DFRobot INA219 driver from {vendor_dir}.") from exc
        return INA219

    @staticmethod
    def _close_vendor(sensor) -> None:
        bus = getattr(sensor, "i2cbus", None)
        close = getattr(bus, "close", None)
        if callable(close):
            close()

