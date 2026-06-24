from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


@dataclass(frozen=True)
class LineoutSpec:
    name: str
    axis: int
    coordinate_name: str
    coordinate_values: np.ndarray
    fixed_indices: tuple[int, int]


def build_lineout_report(
    *,
    fields: dict[str, np.ndarray],
    specs: tuple[LineoutSpec, ...],
) -> dict[str, object]:
    diagnostics: dict[str, dict[str, object]] = {}
    for spec in specs:
        diagnostics[spec.name] = {}
        for field_name, values in fields.items():
            line = _extract_line(values, spec)
            diagnostics[spec.name][field_name] = {
                "coordinate_name": spec.coordinate_name,
                "coordinate_values": np.asarray(spec.coordinate_values, dtype=np.float64).tolist(),
                "mean": line.tolist(),
                "minimum": float(np.min(line)),
                "maximum": float(np.max(line)),
            }
    return {"available": True, "parse_status": "ok", "diagnostics": diagnostics}


def write_lineout_arrays_npz(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    diagnostics = report.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for diagnostic_name, fields in diagnostics.items():
            if not isinstance(fields, dict):
                continue
            for field_name, field_report in fields.items():
                if not isinstance(field_report, dict):
                    continue
                key_prefix = f"{diagnostic_name}:{field_name}"
                payload[f"{key_prefix}:coords"] = np.asarray(field_report.get("coordinate_values", []), dtype=np.float64)
                payload[f"{key_prefix}:mean"] = np.asarray(field_report.get("mean", []), dtype=np.float64)
    payload["__metadata__"] = np.asarray(json.dumps(report, sort_keys=True), dtype=np.str_)
    np.savez_compressed(target, **payload)
    return target


def save_lineout_summary_plot(
    report: dict[str, object],
    path: str | Path,
    *,
    field_names: tuple[str, ...],
    title: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = report.get("diagnostics", {})
    diagnostic_names = tuple(diagnostics.keys()) if isinstance(diagnostics, dict) else ()
    figure, axes = plt.subplots(
        len(field_names),
        max(1, len(diagnostic_names)),
        figsize=(13.5, max(6.0, 2.35 * len(field_names))),
        constrained_layout=True,
        squeeze=False,
    )
    for col, diagnostic_name in enumerate(diagnostic_names):
        diagnostic = diagnostics.get(diagnostic_name, {})
        for row, field_name in enumerate(field_names):
            axis = axes[row, col]
            field_report = diagnostic.get(field_name) if isinstance(diagnostic, dict) else None
            if not isinstance(field_report, dict):
                axis.set_visible(False)
                continue
            coords = np.asarray(field_report.get("coordinate_values", []), dtype=np.float64)
            line = np.asarray(field_report.get("mean", []), dtype=np.float64)
            axis.plot(coords, line, color="#005f73", linewidth=2.2)
            axis.grid(alpha=0.25)
            axis.set_title(f"{_display_label(diagnostic_name)} · {_display_label(field_name)}", fontsize=10)
            axis.set_xlabel(str(field_report.get("coordinate_name", "coord")))
            axis.set_ylabel(_display_label(field_name))
            span = float(np.ptp(line))
            scale = max(float(np.max(np.abs(line))), 1.0e-12)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-2, 2))
            if span / scale < 1.0e-3:
                formatter.set_scientific(False)
            axis.yaxis.set_major_formatter(formatter)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _extract_line(values: np.ndarray, spec: LineoutSpec) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array for lineout extraction, got shape {array.shape}")
    if spec.axis == 0:
        return array[:, spec.fixed_indices[0], spec.fixed_indices[1]]
    if spec.axis == 1:
        return array[spec.fixed_indices[0], :, spec.fixed_indices[1]]
    if spec.axis == 2:
        return array[spec.fixed_indices[0], spec.fixed_indices[1], :]
    raise ValueError(f"Unsupported axis {spec.axis}")


def _display_label(name: str) -> str:
    labels = {
        "radial_midplane": "Radial midplane",
        "toroidal_cut": "Toroidal cut",
        "poloidal_cut": "Poloidal cut",
        "Bmag": "|B|",
        "J": "Jacobian",
        "jacobian": "Jacobian",
        "g11": "g11",
        "g22": "g22",
        "g33": "g33",
        "g_11": "g^11",
        "g_22": "g^22",
        "g_33": "g^33",
    }
    return labels.get(name, name.replace("_", " "))
