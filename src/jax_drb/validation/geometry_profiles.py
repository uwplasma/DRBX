from __future__ import annotations

import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


def build_diagnostic_profile_report(
    *,
    diagnostic_positions: dict[str, tuple[int, np.ndarray]],
    derived_histories: dict[str, tuple[str, np.ndarray]],
    time_points: np.ndarray,
    normalization: dict[str, object],
) -> dict[str, object]:
    diagnostics: dict[str, dict[str, dict[str, object]]] = {}
    for diagnostic_name, (y_index, positions) in diagnostic_positions.items():
        diagnostics[diagnostic_name] = {}
        for observable_name, (units, history_4d) in derived_histories.items():
            tx = np.asarray(history_4d[:, :, y_index, :], dtype=np.float64)
            mean = np.mean(tx, axis=(0, -1))
            std = np.std(tx, axis=(0, -1))
            diagnostics[diagnostic_name][observable_name] = {
                "units": units,
                "position_units": "cm",
                "positions": np.asarray(positions, dtype=np.float64).tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "minimum": float(np.min(tx)),
                "maximum": float(np.max(tx)),
            }

    time_values = np.asarray(time_points, dtype=np.float64)
    return {
        "available": True,
        "parse_status": "ok",
        "normalization": normalization,
        "time_window": {
            "tmin": float(time_values[0]),
            "tmax": float(time_values[-1]),
            "stored_states": int(time_values.size),
        },
        "diagnostics": diagnostics,
    }


def write_diagnostic_profile_arrays_npz(profile_report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    diagnostics = profile_report.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for diagnostic_name, observables in diagnostics.items():
            if not isinstance(observables, dict):
                continue
            for observable_name, observable in observables.items():
                if not isinstance(observable, dict):
                    continue
                key_prefix = f"{diagnostic_name}:{observable_name}"
                payload[f"{key_prefix}:positions"] = np.asarray(observable.get("positions", []), dtype=np.float64)
                payload[f"{key_prefix}:mean"] = np.asarray(observable.get("mean", []), dtype=np.float64)
                payload[f"{key_prefix}:std"] = np.asarray(observable.get("std", []), dtype=np.float64)
    payload["__metadata__"] = np.asarray(json.dumps(profile_report, sort_keys=True), dtype=np.str_)
    np.savez_compressed(target, **payload)
    return target


def save_diagnostic_profile_summary_plot(
    profile_report: dict[str, object],
    path: str | Path,
    *,
    diagnostic_order: tuple[str, ...],
    observable_order: tuple[tuple[str, str], ...],
    title: str,
    line_color: str = "#005f73",
    band_color: str = "#94d2bd",
    separatrix_color: str = "#ae2012",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = profile_report.get("diagnostics", {})
    figure, axes = plt.subplots(
        len(observable_order),
        len(diagnostic_order),
        figsize=(13.5, 14.5),
        constrained_layout=True,
        sharex="col",
    )
    for column, diagnostic_name in enumerate(diagnostic_order):
        diagnostic = diagnostics.get(diagnostic_name, {}) if isinstance(diagnostics, dict) else {}
        for row, (observable_name, label) in enumerate(observable_order):
            axis = axes[row, column]
            observable = diagnostic.get(observable_name) if isinstance(diagnostic, dict) else None
            if not isinstance(observable, dict):
                axis.set_visible(False)
                continue
            positions = np.asarray(observable.get("positions", []), dtype=np.float64)
            mean = np.asarray(observable.get("mean", []), dtype=np.float64)
            std = np.asarray(observable.get("std", []), dtype=np.float64)
            axis.plot(positions, mean, color=line_color, linewidth=2.0)
            axis.fill_between(positions, mean - std, mean + std, color=band_color, alpha=0.35)
            axis.axvline(0.0, color=separatrix_color, linestyle="--", linewidth=1.0, alpha=0.8)
            axis.grid(alpha=0.25, linewidth=0.5)
            axis.set_title(f"{diagnostic_name} · {label}", fontsize=10)
            axis.set_ylabel(str(observable.get("units", "")))
            if row == len(observable_order) - 1:
                axis.set_xlabel("R - R_sep [cm]")
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
