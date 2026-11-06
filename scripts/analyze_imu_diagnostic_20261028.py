#!/usr/bin/env python3
"""Offline, standard-library-only analysis of fixed-body --no-servo logs.

Time zero is calibration_completed. Each measurement window starts at zero
position and velocity and integrates logged acceleration WITHOUT UWB updates.
These are diagnostic drift measurements, not reconstructed physical tracks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_no}: invalid JSON; incomplete log?") from exc


def describe(samples, key):
    values = [row[key][:2] for row in samples]
    if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in values):
        raise ValueError(f"Invalid acceleration in {key}")
    velocity, position, integral = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    intervals = []
    for previous, current, a0, a1 in zip(samples, samples[1:], values, values[1:]):
        dt = (current["sample_t_ns"] - previous["sample_t_ns"]) / 1e9
        if dt <= 0:
            raise ValueError("IMU timestamps are not strictly increasing")
        intervals.append(dt)
        for axis in range(2):
            dv = (a0[axis] + a1[axis]) * 0.5 * dt
            position[axis] += (velocity[axis] + 0.5 * dv) * dt
            velocity[axis] += dv
            integral[axis] += dv
    span = sum(intervals)
    mean = [v / span for v in integral]
    return {
        "mean_xy_mps2": mean,
        "mean_horizontal_norm_mps2": math.hypot(*mean),
        "sample_std_xy_mps2": [statistics.pstdev(v[i] for v in values) for i in range(2)],
        "zero_start_velocity_xy_mps": velocity,
        "zero_start_displacement_xy_m": position,
        "zero_start_displacement_norm_m": math.hypot(*position),
        "max_gap_s": max(intervals),
        "span_s": span,
    }


def analyze(session, protocol):
    result_path = session / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if result.get("executor_started") is not False:
        raise ValueError("Use fixed-body --no-servo logs; this run started the actuator or lacks its status")
    events = list(read_jsonl(session / "events.jsonl"))
    completed = [event for event in events if event.get("event") == "calibration_completed"]
    if len(completed) != 1:
        raise ValueError("Expected one completed calibration; inspect result.json")
    origin = completed[0]["t_ns"]
    rows = list(read_jsonl(session / "imu_filtered.jsonl"))
    if any(b["sample_t_ns"] <= a["sample_t_ns"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Duplicate or backward IMU timestamps")
    if protocol == "static":
        windows = [("static_10_30", 10, 30), ("static_30_50", 30, 50)]
    else:
        labels = ["baseline", "nose_up", "nose_down", "left_roll", "right_roll", "yaw_90", "return_baseline"]
        windows = [(label, 30 * index + 15, 30 * (index + 1)) for index, label in enumerate(labels)]
    report = {
        "session": str(session), "protocol": protocol,
        "time_origin": "calibration_completed event", "result": result,
        "assumption": "Body fixed within each window; p0=v0=0; no UWB corrections. Pose labels follow operator schedule, not automatic detection.",
        "windows": [],
    }
    lines = ["IMU fixed-body diagnostic (no UWB correction)", str(session),
             "All windows start with p0=v0=0. Pose labels assume the prescribed schedule.",
             "Mean acceleration: m/s^2; displacement: m; velocity: m/s.",
             "0.005 m/s^2 corresponds to 1 m / 20 s under constant bias and zero initial velocity error.",
             "This is an error-budget reference, NOT a sensor specification or a navigation pass/fail test.", ""]
    warnings = []
    if result.get("reason") != "time_limit" or result.get("fault") or result.get("cleanup_errors"):
        warnings.append("Run did not finish cleanly: inspect result.json before interpretation.")
    for label, start, end in windows:
        samples = [row for row in rows if start <= (row["sample_t_ns"] - origin) / 1e9 < end]
        window = {"label": label, "start_s": start, "end_s": end, "count": len(samples)}
        report["windows"].append(window)
        if len(samples) < 2:
            window["valid"] = False
            warnings.append(f"{label}: missing data")
            lines.append(f"{label}: MISSING DATA")
            continue
        for key in ("raw_linear_acc", "filtered_acc"):
            window[key] = describe(samples, key)
        stats = window["filtered_acc"]
        window["valid"] = stats["span_s"] >= 0.95 * (end - start) and stats["max_gap_s"] <= 0.05
        if not window["valid"]:
            warnings.append(f"{label}: incomplete coverage or IMU gap > 0.05 s; drift figures are unreliable")
        lines.append(f"{label}: {start}-{end}s, N={len(samples)}, span={stats['span_s']:.3f}s, max_gap={stats['max_gap_s']:.4f}s, coverage_ok={window['valid']}")
        for key in ("raw_linear_acc", "filtered_acc"):
            stat = window[key]
            mean, std = stat["mean_xy_mps2"], stat["sample_std_xy_mps2"]
            pos, vel = stat["zero_start_displacement_xy_m"], stat["zero_start_velocity_xy_mps"]
            lines.append(f"  {key}: mean=({mean[0]:+.6f}, {mean[1]:+.6f}), |mean|={stat['mean_horizontal_norm_mps2']:.6f}, std=({std[0]:.6f}, {std[1]:.6f})")
            lines.append(f"    independent integral: p=({pos[0]:+.4f}, {pos[1]:+.4f}), |p|={stat['zero_start_displacement_norm_m']:.4f}, v=({vel[0]:+.4f}, {vel[1]:+.4f})")
        lines.append("")
    report["warnings"] = warnings
    lines += ["WARNINGS:"] + (warnings or ["None from file/coverage checks. Physical stillness must be checked by the operator."])
    return report, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--protocol", choices=("static", "attitude"), required=True)
    args = parser.parse_args()
    try:
        report, text = analyze(args.session, args.protocol)
        prefix = args.session / ("imu_diagnostic_" + args.protocol)
        prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        prefix.with_suffix(".txt").write_text(text, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(text)
    print(f"Saved: {prefix.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
