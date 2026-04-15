from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np

from ..native import run_curated_case
from ..parity.arrays import load_portable_array_payload
from ..parity.reference import find_reference_case
from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report

_REFERENCE_ARRAY_BASELINE_DIR = Path(__file__).resolve().parents[3] / "references" / "baselines" / "reference_arrays"


@dataclass(frozen=True)
class NativeTokamakSelectedFieldVariableError:
    name: str
    max_abs_error: float
    rms_error: float
    relative_l2_error: float
    max_abs_error_history: np.ndarray
    rms_error_history: np.ndarray


@dataclass(frozen=True)
class NativeTokamakSelectedFieldParityResult:
    case_name: str
    field_names: tuple[str, ...]
    time_points: np.ndarray
    variable_errors: dict[str, NativeTokamakSelectedFieldVariableError]


@dataclass(frozen=True)
class NativeTokamakSelectedFieldArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path
    observable_report_json_path: Path
    runtime_report_json_path: Path


def create_native_tokamak_selected_field_package(
    *,
    case_name: str,
    reference_root: str | Path,
    output_root: str | Path,
    case_label: str = "tokamak_native_selected_field",
    field_names: tuple[str, ...] = ("Ne", "Pe", "phi"),
) -> NativeTokamakSelectedFieldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    reference_case = find_reference_case(case_name)

    expected_payload = load_portable_array_payload(_REFERENCE_ARRAY_BASELINE_DIR / f"{case_name}.npz")
    timer_start = perf_counter()
    result = run_curated_case(case_name, reference_root=reference_root)
    elapsed_seconds = perf_counter() - timer_start
    parity_result = _compare_native_selected_field_histories(
        case_name=case_name,
        expected_fields=expected_payload["variables"],
        actual_fields=result.variables,
        expected_time_points=expected_payload["time_points"],
        actual_time_points=result.time_points,
        field_names=field_names,
    )

    parity_json_path = _write_native_tokamak_selected_field_json(
        parity_result,
        data_dir / f"{case_label}.json",
    )
    parity_arrays_npz_path = _write_native_tokamak_selected_field_arrays(
        parity_result,
        data_dir / f"{case_label}.npz",
    )
    parity_plot_png_path = _save_native_tokamak_selected_field_plot(
        parity_result,
        images_dir / f"{case_label}.png",
    )
    observable_report = build_geometry_observable_report(
        geometry_family="diverted_tokamak_3d",
        benchmark_adapter="native_tokamak_selected_field",
        observable_groups=(
            {
                "name": "selected_field_parity",
                "description": "Reduced selected-field parity surface on a native tokamak execution rung.",
                "families": [
                    {
                        "name": case_name,
                        "kind": "selected_field_parity",
                        "coordinate_name": "time",
                        "field_names": list(parity_result.field_names),
                    }
                ],
            },
        ),
        metadata={
            "compare_surface": "native_selected_field_history",
            "source_case": case_name,
            "reference_capability_tier": reference_case.capability_tier,
            "native_capability_tier": result.payload.get("capability_tier", "unknown"),
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    runtime_report_json_path = _write_native_tokamak_runtime_report(
        case_name=case_name,
        payload=result.payload,
        elapsed_seconds=elapsed_seconds,
        field_names=field_names,
        path=data_dir / f"{case_label}_runtime_report.json",
    )
    return NativeTokamakSelectedFieldArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        observable_report_json_path=observable_report_json_path,
        runtime_report_json_path=runtime_report_json_path,
    )


def _compare_native_selected_field_histories(
    *,
    case_name: str,
    expected_fields: dict[str, np.ndarray],
    actual_fields: dict[str, np.ndarray],
    expected_time_points: list[float] | tuple[float, ...],
    actual_time_points: tuple[float, ...],
    field_names: tuple[str, ...],
) -> NativeTokamakSelectedFieldParityResult:
    reference_time = np.asarray(expected_time_points, dtype=np.float64)
    candidate_time = np.asarray(actual_time_points, dtype=np.float64)
    if reference_time.shape != candidate_time.shape or not np.allclose(
        reference_time,
        candidate_time,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("Reference and native time points do not match for selected-field parity.")

    variable_errors: dict[str, NativeTokamakSelectedFieldVariableError] = {}
    for field_name in field_names:
        if field_name not in expected_fields or field_name not in actual_fields:
            raise KeyError(f"Selected field {field_name!r} is missing from the native parity surface.")
        reference_history = np.asarray(expected_fields[field_name], dtype=np.float64)
        candidate_history = np.asarray(actual_fields[field_name], dtype=np.float64)
        if reference_history.shape != candidate_history.shape:
            raise ValueError(f"Field {field_name!r} shape mismatch: {reference_history.shape} vs {candidate_history.shape}")
        valid = np.isfinite(reference_history) & np.isfinite(candidate_history)
        if not np.any(valid):
            raise ValueError(f"Field {field_name!r} has no finite overlap on the selected compare surface.")
        diff = candidate_history - reference_history
        max_abs_error_history = np.array(
            [float(np.max(np.abs(diff[index][valid[index]]))) for index in range(diff.shape[0])],
            dtype=np.float64,
        )
        rms_error_history = np.array(
            [float(np.sqrt(np.mean(np.square(diff[index][valid[index]])))) for index in range(diff.shape[0])],
            dtype=np.float64,
        )
        reference_values = reference_history[valid]
        diff_values = diff[valid]
        reference_norm = float(np.linalg.norm(reference_values))
        variable_errors[field_name] = NativeTokamakSelectedFieldVariableError(
            name=field_name,
            max_abs_error=float(np.max(np.abs(diff_values))),
            rms_error=float(np.sqrt(np.mean(np.square(diff_values)))),
            relative_l2_error=float(np.linalg.norm(diff_values) / max(reference_norm, np.finfo(np.float64).tiny)),
            max_abs_error_history=max_abs_error_history,
            rms_error_history=rms_error_history,
        )

    return NativeTokamakSelectedFieldParityResult(
        case_name=case_name,
        field_names=field_names,
        time_points=reference_time,
        variable_errors=variable_errors,
    )


def _write_native_tokamak_selected_field_json(
    result: NativeTokamakSelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": result.case_name,
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


def _write_native_tokamak_selected_field_arrays(
    result: NativeTokamakSelectedFieldParityResult,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"time_points": np.asarray(result.time_points, dtype=np.float64)}
    for name, error in result.variable_errors.items():
        payload[f"{name}:max_abs_error_history"] = np.asarray(error.max_abs_error_history, dtype=np.float64)
        payload[f"{name}:rms_error_history"] = np.asarray(error.rms_error_history, dtype=np.float64)
    np.savez_compressed(target, **payload)
    return target


def _save_native_tokamak_selected_field_plot(
    result: NativeTokamakSelectedFieldParityResult,
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
        axes[1].plot(time_points, error.max_abs_error_history, label=f"{field_name} max|Δ|", linewidth=2.0, color=color)
    axes[0].set_title(f"Native tokamak selected-field parity · {result.case_name}")
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


def _write_native_tokamak_runtime_report(
    *,
    case_name: str,
    payload: dict[str, object],
    elapsed_seconds: float,
    field_names: tuple[str, ...],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime_report = {
        "case_name": case_name,
        "selected_fields": list(field_names),
        "capability_tier": payload.get("capability_tier"),
        "parity_mode": payload.get("parity_mode"),
        "dimensions": payload.get("dimensions"),
        "time_point_count": len(payload.get("time_points", [])),
        "configured_nout": payload.get("configured_nout"),
        "configured_timestep": payload.get("configured_timestep"),
        "component_labels": payload.get("component_labels"),
        "dataset_scalars": payload.get("dataset_scalars"),
        "elapsed_seconds": elapsed_seconds,
        "producer": payload.get("producer"),
    }
    target.write_text(json.dumps(runtime_report, indent=2, sort_keys=True), encoding="utf-8")
    return target
