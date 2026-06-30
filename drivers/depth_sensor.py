from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DepthReading:
    temperature_c: float
    pressure_mbar: float
    depth_cm: float
    depth_m: float


class DepthSensor:
    def __init__(self, config: dict[str, Any]) -> None:
        i2c_config = config.get("i2c", {})
        depth_config = config.get("depth_sensor", {})

        self.model = str(depth_config.get("model", "MS5837")).upper()
        self.bus = int(i2c_config.get("bus", 1))
        self.address = int(depth_config.get("i2c_address_7bit", i2c_config.get("depth_sensor_address", 0x76)))
        self.surface_pressure_mbar = float(depth_config.get("surface_pressure_mbar", 1144.0))
        self._sensor = None

    def probe(self) -> bool:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise RuntimeError("Depth sensor probing requires smbus2 on the Raspberry Pi.") from exc

        with SMBus(self.bus) as bus:
            try:
                bus.read_byte(self.address)
                return True
            except OSError:
                return False

    def begin(self) -> bool:
        if self.model != "MS5837":
            raise RuntimeError(f"Unsupported depth sensor model: {self.model}")

        from drivers.DFRobot_MS5837 import MS5837

        sensor = MS5837(self.bus, self.address)
        if not sensor.begin():
            sensor.close()
            return False

        sensor.surface_pressure_mbar = self.surface_pressure_mbar
        self._sensor = sensor
        return True

    def zero(self) -> None:
        sensor = self._require_started()
        sensor.set_zero()
        self.surface_pressure_mbar = sensor.surface_pressure_mbar

    def read(self) -> DepthReading:
        sensor = self._require_started()
        data = sensor.get_data()
        return DepthReading(
            temperature_c=float(data["temperature_C"]),
            pressure_mbar=float(data["pressure_mbar"]),
            depth_cm=float(data["depth_cm"]),
            depth_m=float(data["depth_m"]),
        )

    def close(self) -> None:
        if self._sensor is not None:
            self._sensor.close()
            self._sensor = None

    def _require_started(self):
        if self._sensor is None:
            raise RuntimeError("DepthSensor.begin() must succeed before reading data.")
        return self._sensor
