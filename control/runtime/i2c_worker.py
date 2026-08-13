from __future__ import annotations

import threading
import time
from typing import Any

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from drivers.depth_sensor import DepthSensor
from drivers.ina219 import INA219Sensor


class I2CWorker:
    def __init__(
        self,
        config: dict[str, Any],
        buffers: dict[str, RingBuffer],
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.buffers = buffers
        self.stop_event = stop_event or threading.Event()
        self._thread: threading.Thread | None = None
        self._seq: dict[str, int] = {"depth": 0, "power": 0}
        self._depth_sensor: DepthSensor | None = None
        self._power_sensor: INA219Sensor | None = None
        self._next_depth_init_try_s = 0.0
        self._next_power_init_try_s = 0.0
        self.fatal_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_guarded, name="i2c-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:
            self.fatal_error = f"{type(exc).__name__}: {exc}"

    def run(self) -> None:
        sensors = self.config.get("sensors", {})
        depth_cfg = sensors.get("depth", {})
        power_cfg = sensors.get("power", {})
        depth_enabled = bool(depth_cfg.get("enabled", True))
        power_enabled = bool(power_cfg.get("enabled", True))
        depth_period = 1.0 / max(float(depth_cfg.get("rate_hz", 20)), 0.001)
        power_period = 1.0 / max(float(power_cfg.get("rate_hz", 2)), 0.001)
        next_depth_s = time.monotonic()
        next_power_s = time.monotonic()

        try:
            while not self.stop_event.is_set():
                now_s = time.monotonic()
                did_work = False

                if depth_enabled and now_s >= next_depth_s:
                    did_work = True
                    self._read_depth(now_s)
                    next_depth_s = max(next_depth_s + depth_period, time.monotonic())

                if power_enabled and now_s >= next_power_s:
                    did_work = True
                    self._read_power(now_s)
                    next_power_s = max(next_power_s + power_period, time.monotonic())

                if not did_work:
                    next_due = min(
                        next_depth_s if depth_enabled else now_s + 1.0,
                        next_power_s if power_enabled else now_s + 1.0,
                    )
                    self.stop_event.wait(max(0.001, min(0.05, next_due - now_s)))
        finally:
            self.close()

    def close(self) -> None:
        if self._depth_sensor is not None:
            self._depth_sensor.close()
            self._depth_sensor = None
        if self._power_sensor is not None:
            self._power_sensor.close()
            self._power_sensor = None

    def _read_depth(self, now_s: float) -> None:
        buffer = self.buffers.get("depth")
        if buffer is None:
            return
        try:
            sensor = self._ensure_depth(now_s)
            if sensor is None:
                return
            reading = sensor.read()
            data = {
                "depth_m": reading.depth_m,
                "pressure_pa": reading.pressure_mbar * 100.0,
                "temperature_c": reading.temperature_c,
            }
            buffer.append(self._sample("depth", data))
        except Exception as exc:
            self._depth_sensor = None
            buffer.append(self._sample("depth", {}, ok=False, error=f"{type(exc).__name__}: {exc}"))

    def _read_power(self, now_s: float) -> None:
        buffer = self.buffers.get("power")
        if buffer is None:
            return
        try:
            sensor = self._ensure_power(now_s)
            if sensor is None:
                return
            reading = sensor.read()
            data = {
                "voltage_v": reading.voltage_v,
                "current_a": reading.current_a,
                "power_w": reading.power_w,
            }
            buffer.append(self._sample("power", data))
        except Exception as exc:
            self._power_sensor = None
            buffer.append(self._sample("power", {}, ok=False, error=f"{type(exc).__name__}: {exc}"))

    def _ensure_depth(self, now_s: float) -> DepthSensor | None:
        if self._depth_sensor is not None:
            return self._depth_sensor
        if now_s < self._next_depth_init_try_s:
            return None

        cfg = self._depth_driver_config()
        sensor = DepthSensor(cfg)
        if not sensor.begin():
            self._next_depth_init_try_s = now_s + 1.0
            self.buffers["depth"].append(self._sample("depth", {}, ok=False, error="depth_begin_failed"))
            return None
        self._depth_sensor = sensor
        return sensor

    def _ensure_power(self, now_s: float) -> INA219Sensor | None:
        if self._power_sensor is not None:
            return self._power_sensor
        if now_s < self._next_power_init_try_s:
            return None

        cfg = self.config.get("sensors", {}).get("power", {})
        sensor = INA219Sensor(
            bus=int(cfg.get("i2c_bus", self.config.get("i2c", {}).get("bus", 1))),
            address=int(cfg.get("address", self.config.get("i2c", {}).get("ina219_address", 0x45))),
            linear_cal_reading_ma=float(cfg.get("linear_cal_reading_ma", 1000.0)),
            linear_cal_actual_ma=float(cfg.get("linear_cal_actual_ma", 1000.0)),
        )
        if not sensor.begin():
            self._next_power_init_try_s = now_s + 1.0
            self.buffers["power"].append(self._sample("power", {}, ok=False, error="ina219_begin_failed"))
            return None
        self._power_sensor = sensor
        return sensor

    def _depth_driver_config(self) -> dict[str, Any]:
        cfg = dict(self.config)
        depth_cfg = cfg.get("sensors", {}).get("depth", {})
        i2c_cfg = dict(cfg.get("i2c", {}))
        legacy_depth_cfg = dict(cfg.get("depth_sensor", {}))

        i2c_cfg["bus"] = int(depth_cfg.get("i2c_bus", i2c_cfg.get("bus", 1)))
        i2c_cfg["depth_sensor_address"] = int(depth_cfg.get("address", i2c_cfg.get("depth_sensor_address", 0x76)))
        legacy_depth_cfg["i2c_address_7bit"] = int(
            depth_cfg.get("address", legacy_depth_cfg.get("i2c_address_7bit", 0x76))
        )

        cfg["i2c"] = i2c_cfg
        cfg["depth_sensor"] = legacy_depth_cfg
        return cfg

    def _sample(
        self,
        name: str,
        data: dict[str, Any],
        *,
        ok: bool = True,
        error: str | None = None,
    ) -> SensorSample:
        self._seq[name] = self._seq.get(name, 0) + 1
        return SensorSample(
            name=name,
            t_ns=time.monotonic_ns(),
            seq=self._seq[name],
            data=data,
            ok=ok,
            error=error,
        )
