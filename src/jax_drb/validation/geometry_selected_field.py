from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GeometrySelectedFieldVariableError:
    name: str
    max_abs_error: float
    rms_error: float
    relative_l2_error: float


@dataclass(frozen=True)
class GeometrySelectedFieldParityResult:
    field_names: tuple[str, ...]
    variable_errors: dict[str, GeometrySelectedFieldVariableError]


def compare_geometry_selected_fields(
    *,
    reference_fields: dict[str, np.ndarray],
    candidate_fields: dict[str, np.ndarray],
    field_names: tuple[str, ...],
) -> GeometrySelectedFieldParityResult:
    variable_errors: dict[str, GeometrySelectedFieldVariableError] = {}
    for field_name in field_names:
        if field_name not in reference_fields or field_name not in candidate_fields:
            raise KeyError(f"Missing selected field {field_name!r} in parity comparison.")
        reference = np.asarray(reference_fields[field_name], dtype=np.float64)
        candidate = np.asarray(candidate_fields[field_name], dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"Field {field_name!r} shape mismatch: {reference.shape} vs {candidate.shape}")
        diff = candidate - reference
        reference_norm = float(np.linalg.norm(reference.ravel()))
        variable_errors[field_name] = GeometrySelectedFieldVariableError(
            name=field_name,
            max_abs_error=float(np.max(np.abs(diff))),
            rms_error=float(np.sqrt(np.mean(np.square(diff)))),
            relative_l2_error=float(np.linalg.norm(diff.ravel()) / max(reference_norm, np.finfo(np.float64).tiny)),
        )
    return GeometrySelectedFieldParityResult(field_names=field_names, variable_errors=variable_errors)


def write_geometry_selected_field_parity_json(
    result: GeometrySelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "field_names": list(result.field_names),
        "variable_errors": {
            name: {
                "name": error.name,
                "max_abs_error": error.max_abs_error,
                "rms_error": error.rms_error,
                "relative_l2_error": error.relative_l2_error,
            }
            for name, error in result.variable_errors.items()
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_geometry_selected_field_parity_arrays(
    result: GeometrySelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, error in result.variable_errors.items():
        payload[f"{name}:max_abs_error"] = np.asarray(error.max_abs_error, dtype=np.float64)
        payload[f"{name}:rms_error"] = np.asarray(error.rms_error, dtype=np.float64)
        payload[f"{name}:relative_l2_error"] = np.asarray(error.relative_l2_error, dtype=np.float64)
    np.savez_compressed(target, **payload)
    return target


def save_geometry_selected_field_parity_plot(
    result: GeometrySelectedFieldParityResult,
    path: str | Path,
    *,
    title: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(result.field_names)
    max_abs = [result.variable_errors[name].max_abs_error for name in field_names]
    rms = [result.variable_errors[name].rms_error for name in field_names]
    rel_l2 = [result.variable_errors[name].relative_l2_error for name in field_names]
    x = np.arange(len(field_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    axis.bar(x - width, max_abs, width=width, color="#bb3e03", label="max|Δ|")
    axis.bar(x, rms, width=width, color="#0a9396", label="RMS")
    axis.bar(x + width, rel_l2, width=width, color="#3a86ff", label="rel L2")
    axis.set_xticks(x, field_names)
    axis.set_ylabel("error metric")
    axis.set_title(title)
    axis.grid(alpha=0.25, axis="y")
    axis.legend(frameon=False, ncol=3)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
