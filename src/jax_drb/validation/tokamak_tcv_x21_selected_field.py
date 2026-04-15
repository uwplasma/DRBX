from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report
from .diverted_tokamak_movie import assemble_tokamak_rank_history
from .tokamak_tcv_x21_scaffold import (
    _build_public_benchmark_data_report,
    _load_public_benchmark_history,
    _write_synthetic_preview_workdir,
)


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
    observable_report_json_path: Path
    benchmark_data_report_json_path: Path | None


def compare_tcv_x21_selected_field_workdirs(
    *,
    reference_workdir: str | Path,
    candidate_workdir: str | Path,
    field_names: tuple[str, ...] = ("Ne", "Pe", "phi"),
) -> TcvX21SelectedFieldParityResult:
    return _compare_selected_field_histories(
        reference=_load_selected_field_histories(reference_workdir, field_names=field_names),
        candidate=_load_selected_field_histories(candidate_workdir, field_names=field_names),
        field_names=field_names,
    )


def create_tcv_x21_selected_field_parity_package(
    *,
    reference_workdir: str | Path | None,
    candidate_workdir: str | Path | None,
    output_root: str | Path,
    benchmark_data_root: str | Path | None = None,
    case_label: str = "tokamak_tcv_x21_selected_field_parity",
    field_names: tuple[str, ...] = ("Ne", "Pe", "phi"),
) -> TcvX21SelectedFieldParityArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data_report_json_path: Path | None = None
    source_mode = "explicit_workdir_pair"

    if benchmark_data_root is not None:
        source_mode = (
            "external_benchmark_reference_and_candidate_workdir"
            if candidate_workdir is not None
            else "external_benchmark_reference_derived_candidate"
        )
        benchmark_root = Path(benchmark_data_root)
        benchmark_data_report_json_path = data_dir / f"{case_label}_benchmark_data_report.json"
        benchmark_data_report = _build_public_benchmark_data_report(benchmark_root, requested_field_name=field_names[0])
        benchmark_data_report_json_path.write_text(
            json.dumps(benchmark_data_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        reference_histories = _load_public_selected_field_histories(
            benchmark_root=benchmark_root,
            field_names=field_names,
        )
        if candidate_workdir is None:
            candidate_histories = _derive_candidate_selected_field_histories(reference_histories)
        else:
            candidate_histories = _load_selected_field_histories(candidate_workdir, field_names=field_names)
        result = _compare_selected_field_histories(
            reference=reference_histories,
            candidate=candidate_histories,
            field_names=field_names,
        )
    elif reference_workdir is None or candidate_workdir is None:
        source_mode = "synthetic_preview"
        with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_") as temp_dir:
            temp_root = Path(temp_dir)
            reference_paths = _write_synthetic_preview_workdir(temp_root / "reference", field_name="phi")
            candidate_paths = _write_synthetic_preview_workdir(temp_root / "candidate", field_name="phi")
            result = _compare_selected_field_histories(
                reference=_load_selected_field_histories(reference_paths.workdir, field_names=field_names),
                candidate=_load_selected_field_histories(candidate_paths.workdir, field_names=field_names),
                field_names=field_names,
            )
    else:
        result = _compare_selected_field_histories(
            reference=_load_selected_field_histories(reference_workdir, field_names=field_names),
            candidate=_load_selected_field_histories(candidate_workdir, field_names=field_names),
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
    observable_report = build_geometry_observable_report(
        geometry_family="tokamak_3d",
        benchmark_adapter="tcv_x21_selected_field",
        observable_groups=(
            {
                "name": "selected_field_parity",
                "description": "Compact selected-field parity surface for the 3D tokamak lane.",
                "families": [
                    {
                        "name": "selected_fields",
                        "kind": "selected_field_parity",
                        "coordinate_name": "time",
                        "field_names": list(result.field_names),
                    }
                ],
            },
        ),
        metadata={
            "compare_surface": "z_mean_selected_field_history",
            "source_mode": source_mode,
            "reference_source": "public_tcv_x21_benchmark_bundle" if benchmark_data_root is not None else "tokamak_workdir",
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    return TcvX21SelectedFieldParityArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        observable_report_json_path=observable_report_json_path,
        benchmark_data_report_json_path=benchmark_data_report_json_path,
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


def _load_public_selected_field_histories(
    *,
    benchmark_root: Path,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    time_points: np.ndarray | None = None
    field_histories: dict[str, np.ndarray] = {}
    resolved_field_names: dict[str, str] = {}
    for field_name in field_names:
        history, candidate_time_points, resolved_name = _load_public_benchmark_history(
            benchmark_root=benchmark_root,
            field_name=field_name,
        )
        field_histories[field_name] = np.asarray(history, dtype=np.float64)
        resolved_field_names[field_name] = resolved_name
        if time_points is None:
            time_points = np.asarray(candidate_time_points, dtype=np.float64)
        elif time_points.shape != candidate_time_points.shape or not np.allclose(
            time_points,
            candidate_time_points,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Public benchmark selected-field histories do not share a common time grid.")
    return {
        "time_points": np.asarray(time_points, dtype=np.float64),
        "fields": field_histories,
        "resolved_field_names": resolved_field_names,
    }


def _derive_candidate_selected_field_histories(reference: dict[str, Any]) -> dict[str, Any]:
    derived_fields: dict[str, np.ndarray] = {}
    for field_name, values in reference["fields"].items():
        reference_values = np.asarray(values, dtype=np.float64)
        scale = {
            "Ne": 1.008,
            "Pe": 0.994,
            "phi": 1.015,
        }.get(field_name, 1.005)
        bias = {
            "Ne": 2.0e-3,
            "Pe": -1.5e-3,
            "phi": 5.0e-4,
        }.get(field_name, 0.0)
        derived_fields[field_name] = reference_values * scale + bias
    return {
        "time_points": np.asarray(reference["time_points"], dtype=np.float64),
        "fields": derived_fields,
        "resolved_field_names": dict(reference.get("resolved_field_names", {})),
    }


def _compare_selected_field_histories(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    field_names: tuple[str, ...],
) -> TcvX21SelectedFieldParityResult:
    reference_time = np.asarray(reference["time_points"], dtype=np.float64)
    candidate_time = np.asarray(candidate["time_points"], dtype=np.float64)
    if reference_time.shape != candidate_time.shape or not np.allclose(
        reference_time,
        candidate_time,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
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
        time_points=reference_time,
        variable_errors=variable_errors,
    )
