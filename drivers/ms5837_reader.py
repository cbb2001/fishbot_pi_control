#!/usr/bin/env python3
"""
MS5837 pressure/depth reader for Raspberry Pi / Linux I2C.

Converted from the supplied STM32 example, with two source bugs corrected:
1) C1..C6 PROM coefficients are 16-bit and are read as TWO bytes.
2) Second-order temperature compensation compares TEMP (0.01 degC units)
   directly with 2000 / -1500, rather than TEMP/100 with those values.

Important: this script performs NO cross-sample software filtering.
Each output sample is one D1 pressure conversion + one D2 temperature conversion,
followed by the sensor compensation calculation.

Dependency:
    python3 -m pip install smbus2

Examples:
    python3 ms5837_reader.py --rate-hz 20
    python3 ms5837_reader.py --rate-hz 10 --duration-s 60 --output raw_depth.jsonl
    python3 ms5837_reader.py --rate-hz 20 --surface-pressure-pa 103313.91
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    SMBus = None
    i2c_msg = None


# The STM32 code uses 8-bit HAL addresses 0xEC/0xED.
# Linux/smbus uses the corresponding 7-bit address: 0x76.
DEFAULT_I2C_ADDRESS = 0x76

CMD_RESET = 0x1E
CMD_ADC_READ = 0x00
CMD_CONVERT_D1_OSR8192 = 0x4A
CMD_CONVERT_D2_OSR8192 = 0x5A
PROM_C1 = 0xA2

# The supplied example waits 20 ms for each OSR=8192 conversion.
DEFAULT_CONVERSION_DELAY_S = 0.020


@dataclass(frozen=True)
class Sample:
    d1: int
    d2: int
    temperature_c: float
    pressure_mbar: float
    pressure_pa: float
    depth_m: Optional[float]


class MS5837:
    def __init__(
        self,
        bus_number: int = 1,
        address: int = DEFAULT_I2C_ADDRESS,
        conversion_delay_s: float = DEFAULT_CONVERSION_DELAY_S,
    ) -> None:
        if SMBus is None or i2c_msg is None:
            raise RuntimeError(
                "Missing dependency 'smbus2'. Install it with: python3 -m pip install smbus2"
            )
        self.bus_number = bus_number
        self.address = address
        self.conversion_delay_s = conversion_delay_s
        self.bus = SMBus(bus_number)
        self.C = [0] * 7  # C[1]..C[6]
        self.reset()
        self.read_calibration_coefficients()

    def close(self) -> None:
        self.bus.close()

    def _write_command(self, command: int) -> None:
        msg = i2c_msg.write(self.address, [command & 0xFF])
        self.bus.i2c_rdwr(msg)

    def _command_read(self, command: int, length: int) -> bytes:
        write_msg = i2c_msg.write(self.address, [command & 0xFF])
        read_msg = i2c_msg.read(self.address, length)
        self.bus.i2c_rdwr(write_msg, read_msg)
        return bytes(read_msg)

    def reset(self) -> None:
        self._write_command(CMD_RESET)
        time.sleep(0.001)  # same reset delay as supplied STM32 example

    def read_calibration_coefficients(self) -> None:
        """Read C1..C6 from PROM; every coefficient is 16-bit big-endian."""
        for i in range(1, 7):
            command = PROM_C1 + 2 * (i - 1)
            raw = self._command_read(command, 2)
            self.C[i] = (raw[0] << 8) | raw[1]

        # A completely empty/invalid PROM is an immediate wiring/read red flag.
        if all(v in (0x0000, 0xFFFF) for v in self.C[1:]):
            raise RuntimeError(
                f"PROM coefficients look invalid: {self.C[1:]}. "
                "Check I2C address, wiring, and sensor power."
            )

    def _read_adc(self, conversion_command: int) -> int:
        self._write_command(conversion_command)
        time.sleep(self.conversion_delay_s)
        raw = self._command_read(CMD_ADC_READ, 3)
        return (raw[0] << 16) | (raw[1] << 8) | raw[2]

    def read_raw(self) -> tuple[int, int]:
        """One raw pressure conversion D1 and one raw temperature conversion D2."""
        d1 = self._read_adc(CMD_CONVERT_D1_OSR8192)
        d2 = self._read_adc(CMD_CONVERT_D2_OSR8192)
        return d1, d2

    def compensate(self, d1: int, d2: int) -> tuple[float, float]:
        """
        Apply the same compensation equations used by the supplied C example,
        with corrected second-order temperature branch comparisons.

        Returns:
            (temperature_c, pressure_mbar)
        """
        c1, c2, c3, c4, c5, c6 = self.C[1:7]

        d_t = d2 - c5 * 256
        temp = 2000.0 + d_t * c6 / 8388608.0  # 2^23, units 0.01 degC
        off = c2 * 65536.0 + c4 * d_t / 128.0
        sens = c1 * 32768.0 + c3 * d_t / 256.0

        # Second-order compensation, translated from the supplied source.
        if temp < 2000.0:
            ti = 3.0 * d_t * d_t / 8589934592.0       # 2^33
            offi = 3.0 * (temp - 2000.0) ** 2 / 2.0
            sensi = 5.0 * (temp - 2000.0) ** 2 / 8.0
            if temp < -1500.0:
                offi += 7.0 * (temp + 1500.0) ** 2
                sensi += 4.0 * (temp + 1500.0) ** 2
        else:
            ti = 2.0 * d_t * d_t / 137438953472.0     # 2^37
            offi = (temp - 2000.0) ** 2 / 16.0
            sensi = 0.0

        off2 = off - offi
        sens2 = sens - sensi
        temperature_c = (temp - ti) / 100.0

        # Same scaling as P2 in the supplied C source: result in mbar.
        pressure_mbar = (((d1 * sens2) / 2097152.0 - off2) / 8192.0) / 10.0
        return temperature_c, pressure_mbar

    def read_sample(
        self,
        surface_pressure_pa: Optional[float] = None,
        water_density_kg_m3: float = 1000.0,
        gravity_m_s2: float = 9.80665,
    ) -> Sample:
        d1, d2 = self.read_raw()
        temperature_c, pressure_mbar = self.compensate(d1, d2)
        pressure_pa = pressure_mbar * 100.0

        depth_m: Optional[float] = None
        if surface_pressure_pa is not None:
            # Positive depth means pressure above the calibrated surface pressure.
            depth_m = (
                (pressure_pa - surface_pressure_pa)
                / (water_density_kg_m3 * gravity_m_s2)
            )

        return Sample(
            d1=d1,
            d2=d2,
            temperature_c=temperature_c,
            pressure_mbar=pressure_mbar,
            pressure_pa=pressure_pa,
            depth_m=depth_m,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read MS5837 pressure at a user-selectable output sampling rate."
    )
    p.add_argument("--bus", type=int, default=1, help="Linux I2C bus number (default: 1)")
    p.add_argument(
        "--address",
        type=lambda x: int(x, 0),
        default=DEFAULT_I2C_ADDRESS,
        help="7-bit I2C address (default: 0x76)",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=20.0,
        help="Requested output sample rate in Hz (default: 20)",
    )
    p.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Stop after this many seconds; omit to run until Ctrl+C",
    )
    p.add_argument(
        "--conversion-delay-ms",
        type=float,
        default=20.0,
        help="Wait after each OSR8192 conversion command (default: 20 ms, matching source)",
    )
    p.add_argument(
        "--surface-pressure-pa",
        type=float,
        default=None,
        help="Optional calibrated surface pressure. If given, output physical depth_m.",
    )
    p.add_argument(
        "--water-density",
        type=float,
        default=1000.0,
        help="Water density kg/m^3 for depth conversion (default: 1000)",
    )
    p.add_argument(
        "--gravity",
        type=float,
        default=9.80665,
        help="Gravity m/s^2 for depth conversion (default: 9.80665)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSONL output path",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print every sample to terminal",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be > 0")
    if args.duration_s is not None and args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")
    if args.conversion_delay_ms <= 0:
        raise SystemExit("--conversion-delay-ms must be > 0")

    conversion_delay_s = args.conversion_delay_ms / 1000.0

    # One output sample requires TWO conversions (D1 and D2). The source uses
    # 20 ms for each, so 20 Hz is feasible, while 25 Hz leaves essentially no
    # time for I2C + Python overhead. Reject obviously impossible requests.
    acquisition_floor_s = 2.0 * conversion_delay_s
    period_s = 1.0 / args.rate_hz
    if period_s <= acquisition_floor_s:
        max_theoretical_hz = 1.0 / acquisition_floor_s
        raise SystemExit(
            f"Requested {args.rate_hz:g} Hz is too high for "
            f"{args.conversion_delay_ms:g} ms x 2 conversions/sample. "
            f"The theoretical ceiling is {max_theoretical_hz:.2f} Hz before I2C/Python overhead. "
            "Use a lower --rate-hz or a validated shorter conversion delay/OSR."
        )

    stop = False

    def _stop_handler(signum, frame):  # noqa: ARG001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    sensor = MS5837(
        bus_number=args.bus,
        address=args.address,
        conversion_delay_s=conversion_delay_s,
    )

    out_f = open(args.output, "a", encoding="utf-8", buffering=1) if args.output else None

    print(
        "MS5837 ready | "
        f"bus={args.bus} addr=0x{args.address:02X} rate={args.rate_hz:g} Hz | "
        f"C1..C6={sensor.C[1:]} | software_filter=none",
        file=sys.stderr,
    )

    t0_mono = time.perf_counter()
    next_deadline = t0_mono
    sample_index = 0
    overruns = 0

    try:
        while not stop:
            now = time.perf_counter()
            if args.duration_s is not None and (now - t0_mono) >= args.duration_s:
                break

            sample_start_mono = time.perf_counter()
            sample = sensor.read_sample(
                surface_pressure_pa=args.surface_pressure_pa,
                water_density_kg_m3=args.water_density,
                gravity_m_s2=args.gravity,
            )
            t_ns = time.time_ns()
            elapsed_s = sample_start_mono - t0_mono

            record = {
                "seq": sample_index,
                "t_ns": t_ns,
                "elapsed_s": elapsed_s,
                "ok": True,
                "data": {
                    "d1_raw": sample.d1,
                    "d2_raw": sample.d2,
                    "temperature_c": sample.temperature_c,
                    "pressure_mbar": sample.pressure_mbar,
                    "pressure_pa": sample.pressure_pa,
                    "depth_m": sample.depth_m,
                    "software_filter": "none",
                },
            }

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            if out_f:
                out_f.write(line + "\n")
            if not args.quiet:
                if sample.depth_m is None:
                    print(
                        f"{sample_index:06d}  "
                        f"P={sample.pressure_pa:10.2f} Pa  "
                        f"T={sample.temperature_c:7.3f} C  "
                        f"D1={sample.d1:8d} D2={sample.d2:8d}"
                    )
                else:
                    print(
                        f"{sample_index:06d}  "
                        f"P={sample.pressure_pa:10.2f} Pa  "
                        f"depth={sample.depth_m:8.4f} m  "
                        f"T={sample.temperature_c:7.3f} C"
                    )

            sample_index += 1
            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Do not add extra delay; keep running and re-anchor only after
                # a full-period overrun to prevent unbounded scheduler drift.
                overruns += 1
                if sleep_s < -period_s:
                    next_deadline = time.perf_counter()

    finally:
        if out_f:
            out_f.close()
        sensor.close()

    elapsed_total = time.perf_counter() - t0_mono
    actual_rate = sample_index / elapsed_total if elapsed_total > 0 else float("nan")
    print(
        f"Stopped | samples={sample_index} elapsed={elapsed_total:.3f}s "
        f"actual_rate={actual_rate:.3f}Hz overruns={overruns}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
