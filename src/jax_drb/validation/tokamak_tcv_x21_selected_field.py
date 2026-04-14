from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset

from .diverted_tokamak_movie import assemble_tokamak_rank_history
from .tokamak_tcv_x21_scaffold import _write_synthetic_preview_workdir


@dataclass(frozen=True)
class TcvX21SelectedFieldVariableError:
    name: str
    max_abs_error: float
    rms_error: float
    relative_l2_error: float
    max_abs_error_history: np.ndarray
    rms_error_history: np.ndarray


@dataclass(frozen=True)
class TcvX21SelectedFieldParityResult:
    field_names: tuple[str, ...]
    time_points: np.ndarray
    variable_errors: dict[str, TcvX21SelectedFieldVariableError]


@dataclass(frozen=True)
class TcvX21SelectedFieldParityArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path


def compare_tcv_x21_selected_field_workdirs(
    *,
    reference_workdir: str | Path,
    candidate_workdir: str | Path,
    field_names: tuple[str, ...] = ("Ne", "Pe", "phi"),
) -> TcvX21SelectedFieldParityResult:
    reference = _load_selected_field_histories(reference_workdir, field_names=field_names)
    candidate = _load_selected_field_histories(candidate_workdir, field_names=field_names)
    reference_time = reference["time_points"]
    candidate_time = candidate["time_points"]
    if reference_time.shape != candidate_time.shape or not np.allclose(reference_time, candidate_time, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Reference and candidate time points do not match for selected-field parity.")

    variable_errors: dict[str, TcvX21SelectedFieldVariableError] = {}
    for field_name in field_names:
        reference_history = np.asarray(reference["fields"][field_name], dtype=np.float64)
        candidate_history = np.asarray(candidate["fields"][field_name], dtype=np.float64)
        if reference_history.shape != candidate_history.shape:
            raise ValueError(f"Field {field_name!r} shape mismatch: {reference_history.shape} vs {candidate_history.shape}")
        diff = candidate_history - reference_history
        axes = tuple(range(1, diff.ndim))
        max_abs_error_history = np.max(np.abs(diff), axis=axes)
        rms_error_history = np.sqrt(np.mean(np.square(diff), axis=axes))
        reference_norm = float(np.linalg.norm(reference_history.ravel()))
        relative_l2_error = float(np.linalg.norm(diff.ravel()) / max(reference_norm, np.finfo(np.float64).tiny))
        variable_errors[field_name] = TcvX21SelectedFieldVariableError(
            name=field_name,
            max_abs_error=float(np.max(np.abs(diff))),
            rms_error=float(np.sqrt(np.mean(np.square(diff)))),
            relative_l2_error=relative_l2_error,
            max_abs_error_history=np.asarray(max_abs_error_history, dtype=np.float64),
            rms_error_history=np.asarray(rms_error_history, dtype=np.float64),
        )

    return TcvX21SelectedFieldParityResult(
        field_names=tuple(field_names),
        time_points=np.asarray(reference_time, dtype=np.float64),
        variable_errors=variable_errors,
    )


def create_tcv_x21_selected_field_parity_package(
    *,
    reference_workdir: str | Path | None,
    candidate_workdir: str | Path | None,
    output_root: str | Path,
    case_label: str = "tokamak_tcv_x21_selected_field_parity",
    field_names: tuple[str, ...] = ("Ne", "Pe", "phi"),
) -> TcvX21SelectedFieldParityArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if reference_workdir is None or candidate_workdir is None:
        with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_") as temp_dir:
            temp_root = Path(temp_dir)
            reference_paths = _write_synthetic_preview_workdir(temp_root / "reference", field_name="phi")
            candidate_paths = _write_synthetic_preview_workdir(temp_root / "candidate", field_name="phi")
            result = compare_tcv_x21_selected_field_workdirs(
                reference_workdir=reference_paths.workdir,
                candidate_workdir=candidate_paths.workdir,
                field_names=field_names,
            )
    else:
        result = compare_tcv_x21_selected_field_workdirs(
            reference_workdir=reference_workdir,
            candidate_workdir=candidate_workdir,
            field_names=field_names,
        )

    parity_json_path = write_tcv_x21_selected_field_parity_json(
        result,
        data_dir / f"{case_label}.json",
    )
    parity_arrays_npz_path = write_tcv_x21_selected_field_parity_arrays(
        result,
        data_dir / f"{case_label}.npz",
    )
    parity_plot_png_path = save_tcv_x21_selected_field_parity_plot(
        result,
        images_dir / f"{case_label}.png",
    )
    return TcvX21SelectedFieldParityArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
    )


def write_tcv_x21_selected_field_parity_json(
    result: TcvX21SelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "field_names": list(result.field_names),
        "time_points": result.time_points.tolist(),
        "variable_errors": {
            name: {
                "name": error.name,
                "max_abs_error": error.max_abs_error,
                "rms_error": error.rms_error,
                "relative_l2_error": error.relative_l2_error,
                "max_abs_error_history": error.max_abs_error_history.tolist(),
                "rms_error_history": error.rms_error_history.tolist(),
            }
            for name, error in result.variable_errors.items()
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_tcv_x21_selected_field_parity_arrays(
    result: TcvX21SelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "time_points": np.asarray(result.time_points, dtype=np.float64),
    }
    for name, error in result.variable_errors.items():
        payload[f"{name}:max_abs_error_history"] = np.asarray(error.max_abs_error_history, dtype=np.float64)
        payload[f"{name}:rms_error_history"] = np.asarray(error.rms_error_history, dtype=np.float64)
    np.savez_compressed(target, **payload)
    return target


def save_tcv_x21_selected_field_parity_plot(
    result: TcvX21SelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), constrained_layout=True)
    time_points = np.asarray(result.time_points, dtype=np.float64)
    colors = ("#005f73", "#ca6702", "#bb3e03", "#3a86ff")

    for color, field_name in zip(colors, result.field_names, strict=False):
        error = result.variable_errors[field_name]
        axes[0].plot(time_points, error.rms_error_history, label=f"{field_name} RMS", linewidth=2.0, color=color)
        axes[1].plot(
            time_points,
            error.max_abs_error_history,
            label=f"{field_name} max|Δ|",
            linewidth=2.0,
            color=color,
        )
    axes[0].set_title("TCV-X21 Reduced Selected-Field Parity")
    axes[0].set_ylabel("RMS error")
    axes[0].grid(alpha=0.25, linewidth=0.5)
    axes[1].set_ylabel("Max abs error")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].legend(frameon=False, ncol=2)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _load_selected_field_histories(
    workdir: str | Path,
    *,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(workdir)
    time_points: np.ndarray | None = None
    field_histories: dict[str, np.ndarray] = {}
    for field_name in field_names:
        history = assemble_tokamak_rank_history(root, field_name=field_name)
        z_mean = np.asarray(history.history_4d.mean(axis=-1), dtype=np.float64)
        field_histories[field_name] = z_mean
        if time_points is None:
            time_points = np.asarray(history.time_points, dtype=np.float64)
        elif time_points.shape != history.time_points.shape or not np.allclose(time_points, history.time_points, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Selected-field histories do not share a common time grid.")
    return {
        "time_points": np.asarray(time_points, dtype=np.float64),
        "fields": field_histories,
    }
