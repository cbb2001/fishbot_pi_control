"""Offline calibration artifact generator; does not alter the robot controller.

Run with Python + numpy + scipy + matplotlib. Paths resolve from repository root.
Every run is referenced to its first valid pressure sample, with elapsed time zero
at that sample. Five-second means reduce noise; each recording has equal total
least-squares weight, regardless of duration. No recording is excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = ["2", "3", "4", "5", "6", "2026092802", "2026092803",
            "2026092804", "2026092903", "2026092902", "2026092901", "2026093002"]
OUTPUT = ROOT / "logs/analysis_drift_20261001"
CONFIG = ROOT / "config/pressure_drift_10hz.json"


def curve(t, amplitude_fast_pa, tau_fast_s, amplitude_slow_pa, tau_slow_s):
    return (amplitude_fast_pa * np.expm1(-np.asarray(t) / tau_fast_s)
            - amplitude_slow_pa * np.expm1(-np.asarray(t) / tau_slow_s))


def linear_tail(t, amplitude_pa, tau_s, slope_pa_s):
    return amplitude_pa * np.expm1(-np.asarray(t) / tau_s) + slope_pa_s * t


def fit(runs, model="bounded"):
    fun, start, bounds = (
        (curve, [2500, 100, 1200, 2500], ([0, 1, 0, 300], [10000, 1000, 10000, 30000]))
        if model == "bounded" else
        (linear_tail, [2300, 100, .2], ([0, 1, -2], [10000, 1000, 2]))
    )
    result = least_squares(
        lambda p: np.concatenate([(fun(x, *p) - y) / np.sqrt(len(x)) for x, y in runs]),
        start, bounds=bounds, max_nfev=3000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs, raw_runs, sources = [], [], []
    for suffix in SUFFIXES:
        path = ROOT / f"logs/raw_depth_0_10_{suffix}.jsonl"
        blob = path.read_bytes()
        rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
        if not all(row.get("ok") is True for row in rows):
            raise ValueError(f"Unsuccessful samples in {path}")
        t = np.array([row["elapsed_s"] for row in rows], dtype=float)
        t -= t[0]
        pressure = np.array([row["data"]["pressure_pa"] for row in rows], dtype=float)
        if not (np.isfinite(t).all() and np.isfinite(pressure).all()
                and (np.diff(t) > 0).all()
                and abs(np.median(np.diff(t)) - .1) < .005):
            raise ValueError(f"Invalid time or pressure in {path}")
        if [row["seq"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Sequence discontinuity in {path}")
        y = pressure - pressure[0]
        # Actual time bins, so small scheduling jitter does not shift the fit clock.
        bins = np.floor(t / 5).astype(int)
        count = np.bincount(bins)
        nonempty = count > 0
        x_bin = np.bincount(bins, weights=t)[nonempty] / count[nonempty]
        y_bin = np.bincount(bins, weights=y)[nonempty] / count[nonempty]
        runs.append((x_bin, y_bin))
        raw_runs.append((t, y))
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(blob).hexdigest(), "samples": len(rows),
                        "duration_s": float(t[-1]), "initial_pressure_pa": float(pressure[0])})

    parameters = fit(runs)
    names = ["amplitude_fast_pa", "tau_fast_s", "amplitude_slow_pa", "tau_slow_s"]
    metrics = []
    for i, ((x, y), (t, raw)) in enumerate(zip(runs, raw_runs)):
        error = y - curve(x, *parameters)
        other_parameters = fit(runs[:i] + runs[i + 1:])
        metrics.append({"source": sources[i]["path"],
                        "fit_5s_mean_rmse_pa": rmse(error),
                        "fit_raw_rmse_pa": rmse(raw - curve(t, *parameters)),
                        "fit_5s_mean_max_abs_error_pa": float(max(abs(error))),
                        "leave_one_run_out_5s_mean_rmse_pa": rmse(y - curve(x, *other_parameters))})

    temporal = {}
    for model, fun in [("bounded", curve), ("linear_tail", linear_tail)]:
        cropped = [(x[x < 1800], y[x < 1800]) for x, y in runs]
        p = fit(cropped, model)
        temporal[model] = {
            "training_cutoff_s": 1800, "parameters": p.tolist(),
            "held_out_tail_rmse_pa": {
                sources[i]["path"]: rmse(y[x >= 1800] - fun(x[x >= 1800], *p))
                for i, (x, y) in enumerate(runs) if (x >= 1800).any()},
            "note": "Temporal holdout uses each run's initial pressure; not a 2h validation.",
        }

    grid = np.arange(72001, dtype=float) / 10
    values = np.round(curve(grid, *parameters), 6)
    config = {
        "schema_version": 1,
        "calibration_id": "pressure_drift_10hz_20261001",
        "description": "Relative pressure drift for surface-floating fish, pressure port fully submerged.",
        "sample_rate_hz": 10.0, "pressure_unit": "Pa", "time_unit": "s",
        "reference": {
            "time_origin": "First valid sample of uninterrupted 10Hz sensor acquisition, not motor start.",
            "pressure_origin": "First valid pressure sample of each recording; each run starts at 0 Pa drift.",
            "runtime_p0": "Pressure measured at the same surface posture and port immersion at t=0.",
            "multi_sample_p0": "Optional noise reduction: mean(P_i - f(t_i)) over initial surface samples.",
            "depth_origin": "Depth change relative to this surface posture; port immersion is included in P0.",
        },
        "model": {
            "type": "fast_decay_plus_saturating_recovery",
            "formula": "f(t)=-amplitude_fast_pa*(1-exp(-t/tau_fast_s))+amplitude_slow_pa*(1-exp(-t/tau_slow_s))",
            "parameters": dict(zip(names, map(float, parameters))),
            "asymptote_pa": float(-parameters[0] + parameters[2]),
        },
        "lookup": {"start_s": 0.0, "step_s": 0.1, "end_s": 7200.0, "count": len(values),
                   "index_rule": "t_i = start_s + i * step_s; values are relative drift, not absolute pressure.",
                   "interpolation": "linear for timestamps between entries",
                   "out_of_range": "Outside [0,7200] is not covered by this calibration table.",
                   "pressure_drift_pa": values.tolist()},
        "usage": {
            "surface_reference_pa": "P0 + f(elapsed_s)",
            "corrected_gauge_pressure_pa": "measured_pressure_pa - P0 - f(elapsed_s)",
            "relative_depth_m": "corrected_gauge_pressure_pa / (water_density_kg_m3 * gravity_m_s2)",
            "water_density_kg_m3_default": 1000.0, "gravity_m_s2_default": 9.80665,
            "clock": "Use monotonic sensor acquisition elapsed time. Do not reset when motors stop if acquisition continues.",
            "restart": "A pause/reset/power cycle may change drift; reestablish the acquisition conditions and surface P0.",
        },
        "fit": {"offset_removal": "subtract first sample, independently per recording",
                "bin_s": 5, "weighting": "equal total squared-error weight per recording",
                "sources": sources, "per_run_metrics": metrics,
                "temporal_validation": temporal},
        "validity": {
            "longest_observed_duration_s": max(s["duration_s"] for s in sources),
            "prediction_end_s": 7200, "two_hour_accuracy_validated": False,
            "notes": ["After approximately 3600 s the curve is extrapolated, not measured.",
                      "Saturating recovery is an empirical extrapolation assumption, not an established physical law.",
                      "Only one recording reaches 60 min; per-run drift shape differs even after zeroing.",
                      "Identical sensor, 10Hz conversion settings, start conditions and immersion are required for transfer.",
                      "Dynamic flow pressure, water temperature changes and real depth changes are not modeled."]},
    }
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "fit_metrics.json").write_text(json.dumps({"parameters": config["model"],
        "per_run_metrics": metrics, "temporal_validation": temporal}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for suffix, (x, y) in zip(SUFFIXES, runs):
        axes[0].plot(x / 60, y, label=suffix, alpha=.7, linewidth=1)
        axes[2].plot(x / 60, y - curve(x, *parameters), alpha=.7, label=suffix)
    axes[0].plot(grid[:36001] / 60, values[:36001], color="black", lw=2.5, label="Shared model")
    axes[0].set(title="12 recordings, each minus its first pressure sample", ylabel="Relative pressure (Pa)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].plot(grid / 60, values, color="black", lw=2)
    axes[1].axvspan(60, 120, color="orange", alpha=.18, label="Extrapolation: no measured 60-120 min data")
    axes[1].set(title="Calibration lookup: 0-120 min, 0.1 s interval", ylabel="f(t) (Pa)")
    axes[1].legend()
    axes[2].set(title="Residual after shared compensation (5 s means)", ylabel="Residual (Pa)")
    for ax in axes:
        ax.set_xlabel("Time since acquisition start (min)")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "pressure_drift_fit.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"config": str(CONFIG), "parameters": config["model"]["parameters"],
        "checkpoints_pa": {str(t): float(curve(t, *parameters)) for t in [0, 600, 1800, 3600, 7200]},
        "fit_rmse_pa_range": [min(m["fit_5s_mean_rmse_pa"] for m in metrics), max(m["fit_5s_mean_rmse_pa"] for m in metrics)],
        "loo_rmse_pa_range": [min(m["leave_one_run_out_5s_mean_rmse_pa"] for m in metrics), max(m["leave_one_run_out_5s_mean_rmse_pa"] for m in metrics)]}, indent=2))


if __name__ == "__main__":
    main()
