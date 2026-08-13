from __future__ import annotations

import time
from typing import Any

from control.runtime.ring_buffer import RingBuffer
from control.runtime.sample import SensorSample
from control.runtime.sensor_manager import SENSOR_NAMES, sensor_config


class SensorSynchronizer:
    def __init__(self, buffers: dict[str, RingBuffer], config: dict[str, Any]) -> None:
        self.buffers = buffers
        self.config = config
        self.start_ns = time.monotonic_ns()

    def build(self, t_ns: int | None = None) -> dict[str, Any]:
        now_ns = t_ns if t_ns is not None else time.monotonic_ns()
        sample: dict[str, Any] = {
            "t_ns": now_ns,
            "t_s": (now_ns - self.start_ns) / 1_000_000_000.0,
        }

        for name in SENSOR_NAMES:
            field = self._sensor_field(name, now_ns)
            if name == "depth":
                self._add_depth_fields(field, now_ns)
            if name == "power":
                self._add_power_fields(field)
            sample[name] = field

        sample["status"] = self._status(sample)
        return sample

    def _sensor_field(self, name: str, now_ns: int) -> dict[str, Any]:
        cfg = sensor_config(self.config, name)
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return {
                "valid": False,
                "age_ms": None,
                "error": "disabled",
                "interpolated": False,
                "interpolation_gap_ms": None,
                "data": {},
            }

        buffer = self.buffers.get(name)
        # 按同步截止时刻取最近样本，避免把未来重建时间戳的样本错误挂到更早记录。
        sample = buffer.get_latest_before(now_ns) if buffer is not None else None
        if sample is None:
            return {
                "valid": False,
                "age_ms": None,
                "error": "no_sample",
                "interpolated": False,
                "interpolation_gap_ms": None,
                "data": {},
            }

        age_ns = max(0, now_ns - sample.t_ns)
        timeout_ns = int(float(cfg.get("timeout_ms", 1000)) * 1_000_000)
        error = sample.error
        valid = bool(sample.ok) and age_ns <= timeout_ns
        if age_ns > timeout_ns:
            valid = False
            error = error or "stale"

        data = dict(sample.data)
        if name == "vision":
            for binary_key in ("frame", "image", "image_bytes", "frame_bytes"):
                data.pop(binary_key, None)

        return {
            "valid": valid,
            "age_ms": age_ns / 1_000_000.0,
            "error": error,
            "interpolated": False,
            "interpolation_gap_ms": None,
            "seq": sample.seq,
            "sample_t_ns": sample.t_ns,
            "data": data,
        }

    def _add_depth_fields(self, field: dict[str, Any], now_ns: int) -> None:
        data = field.get("data", {})
        if "depth_m" in data:
            field["depth_m"] = data["depth_m"]
        rate = self._depth_rate_mps(now_ns)
        field["depth_rate_mps"] = rate

    def _add_power_fields(self, field: dict[str, Any]) -> None:
        data = field.get("data", {})
        for key in ("voltage_v", "current_a", "power_w"):
            if key in data:
                field[key] = data[key]

    def _depth_rate_mps(self, now_ns: int) -> float | None:
        buffer = self.buffers.get("depth")
        if buffer is None:
            return None
        samples = [
            sample for sample in buffer.snapshot()
            if (
                isinstance(sample, SensorSample)
                and sample.ok
                and sample.t_ns <= now_ns
                and "depth_m" in sample.data
            )
        ]
        if len(samples) < 2:
            return None
        latest = samples[-1]
        previous = samples[-2]
        if latest.t_ns > now_ns:
            return None
        dt_s = (latest.t_ns - previous.t_ns) / 1_000_000_000.0
        if dt_s <= 0:
            return None
        return (float(latest.data["depth_m"]) - float(previous.data["depth_m"])) / dt_s

    def _status(self, sample: dict[str, Any]) -> dict[str, Any]:
        environment = str(self.config.get("runtime", {}).get("environment", "underwater"))
        missing = []
        warnings = []
        for name in SENSOR_NAMES:
            field = sample.get(name, {})
            if not field.get("valid", False):
                missing.append(name)
                error = field.get("error")
                if error and error != "disabled":
                    warnings.append(f"{name}: {error}")

        return {
            "environment": environment,
            "all_sensors_ok": len(missing) == 0,
            "missing_sensors": missing,
            "warnings": warnings,
        }
