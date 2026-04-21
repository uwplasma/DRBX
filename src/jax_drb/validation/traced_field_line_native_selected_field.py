from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report
from .traced_field_line_selected_field import _load_metric_fields, _resolve_field_alias


@dataclass(frozen=True)
class NativeTracedFieldLineSelectedFieldVariableError:
    name: str
    max_abs_error: float
    rms_error: float
    relative_l2_error: float


@dataclass(frozen=True)
class NativeTracedFieldLineSelectedFieldParityResult:
    field_names: tuple[str, ...]
    variable_errors: dict[str, NativeTracedFieldLineSelectedFieldVariableError]


@dataclass(frozen=True)
class NativeTracedFieldLineSelectedFieldArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path
    comparison_json_path: Path
    comparison_plot_png_path: Path
    observable_report_json_path: Path
    runtime_report_json_path: Path


def create_native_traced_field_line_selected_field_package(
    *,
    reference_mesh_spec: str | Path,
    candidate_mesh_spec: str | Path,
    output_root: str | Path,
    case_label: str = "traced_field_line_native_selected_field",
    field_names: tuple[str, ...] = ("g11", "g33"),
) -> NativeTracedFieldLineSelectedFieldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    timer_start = perf_counter()
    result, reference_profiles, candidate_profiles = compare_native_traced_field_line_selected_fields(
        reference_mesh_spec=reference_mesh_spec,
        candidate_mesh_spec=candidate_mesh_spec,
        field_names=field_names,
    )
    elapsed_seconds = perf_counter() - timer_start

    parity_json_path = _write_parity_json(result, data_dir / f"{case_label}.json")
    parity_arrays_npz_path = _write_parity_arrays(result, data_dir / f"{case_label}.npz")
    parity_plot_png_path = _save_parity_plot(result, images_dir / f"{case_label}.png")
    comparison_json_path = _write_comparison_json(
        result,
        reference_profiles=reference_profiles,
        candidate_profiles=candidate_profiles,
        path=data_dir / f"{case_label}_comparison.json",
    )
    comparison_plot_png_path = _save_comparison_plot(
        result,
        reference_profiles=reference_profiles,
        candidate_profiles=candidate_profiles,
        path=images_dir / f"{case_label}_comparison.png",
    )
    observable_report = build_geometry_observable_report(
        geometry_family="traced_field_line_3d",
        benchmark_adapter="native_traced_field_line_selected_field",
        observable_groups=(
            {
                "name": "native_reduced_selected_field",
                "description": "JAX-native reduced selected-field parity surface on traced-field-line geometry data.",
                "families": [
                    {
                        "name": "reduced_radial_profiles",
                        "kind": "selected_field_parity",
                        "coordinate_name": "radial_index",
                        "field_names": list(result.field_names),
                    }
                ],
            },
        ),
        metadata={
            "compare_surface": "jax_native_reduced_radial_profile",
            "native_capability_tier": "native_exact_reduced",
            "selected_fields": list(result.field_names),
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    runtime_report_json_path = _write_runtime_report(
        field_names=result.field_names,
        elapsed_seconds=elapsed_seconds,
        reference_mesh_spec=Path(reference_mesh_spec),
        candidate_mesh_spec=Path(candidate_mesh_spec),
        path=data_dir / f"{case_label}_runtime_report.json",
    )
    return NativeTracedFieldLineSelectedFieldArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        comparison_json_path=comparison_json_path,
        comparison_plot_png_path=comparison_plot_png_path,
        observable_report_json_path=observable_report_json_path,
        runtime_report_json_path=runtime_report_json_path,
    )


def compare_native_traced_field_line_selected_fields(
    *,
    reference_mesh_spec: str | Path,
    candidate_mesh_spec: str | Path,
    field_names: tuple[str, ...] = ("g11", "g33"),
) -> tuple[
    NativeTracedFieldLineSelectedFieldParityResult,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    reference_fields = _load_metric_fields(reference_mesh_spec)
    candidate_fields = _load_metric_fields(candidate_mesh_spec)
    resolved_field_names = tuple(_resolve_field_alias(reference_fields, candidate_fields, name) for name in field_names)
    reference_profiles, candidate_profiles = _native_reduce_metric_profile_pair(
        reference_fields,
        candidate_fields,
        resolved_field_names,
    )
    variable_errors: dict[str, NativeTracedFieldLineSelectedFieldVariableError] = {}
    for field_name in resolved_field_names:
        reference = np.asarray(reference_profiles[field_name], dtype=np.float64)
        candidate = np.asarray(candidate_profiles[field_name], dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"Reduced profile shape mismatch for {field_name!r}: {reference.shape} vs {candidate.shape}")
        diff = candidate - reference
        reference_norm = float(np.linalg.norm(reference.ravel()))
        variable_errors[field_name] = NativeTracedFieldLineSelectedFieldVariableError(
            name=field_name,
            max_abs_error=float(np.max(np.abs(diff))),
            rms_error=float(np.sqrt(np.mean(np.square(diff)))),
            relative_l2_error=float(np.linalg.norm(diff.ravel()) / max(reference_norm, np.finfo(np.float64).tiny)),
        )
    return (
        NativeTracedFieldLineSelectedFieldParityResult(
            field_names=resolved_field_names,
            variable_errors=variable_errors,
        ),
        reference_profiles,
        candidate_profiles,
    )


@jax.jit
def _native_radial_profile_batch(values: jax.Array) -> jax.Array:
    if values.ndim <= 2:
        return values
    axes = tuple(range(2, values.ndim))
    return jnp.mean(values, axis=axes)


@jax.jit
def _native_radial_profile_pair_batch(values: jax.Array) -> jax.Array:
    if values.ndim <= 3:
        return values
    axes = tuple(range(3, values.ndim))
    return jnp.mean(values, axis=axes)


def _native_reduce_metric_profiles(
    fields: dict[str, np.ndarray],
    field_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    stacked = jnp.stack([jnp.asarray(fields[field_name], dtype=jnp.float64) for field_name in field_names], axis=0)
    reduced_values = np.asarray(_native_radial_profile_batch(stacked), dtype=np.float64)
    return {
        field_name: reduced_values[index]
        for index, field_name in enumerate(field_names)
    }


def _native_reduce_metric_profile_pair(
    reference_fields: dict[str, np.ndarray],
    candidate_fields: dict[str, np.ndarray],
    field_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference_stacked = jnp.stack([jnp.asarray(reference_fields[field_name], dtype=jnp.float64) for field_name in field_names], axis=0)
    candidate_stacked = jnp.stack([jnp.asarray(candidate_fields[field_name], dtype=jnp.float64) for field_name in field_names], axis=0)
    combined = jnp.stack((reference_stacked, candidate_stacked), axis=0)
    reduced_values = np.asarray(_native_radial_profile_pair_batch(combined), dtype=np.float64)
    return (
        {field_name: reduced_values[0, index] for index, field_name in enumerate(field_names)},
        {field_name: reduced_values[1, index] for index, field_name in enumerate(field_names)},
    )


def _write_parity_json(result: NativeTracedFieldLineSelectedFieldParityResult, path: Path) -> Path:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_parity_arrays(result: NativeTracedFieldLineSelectedFieldParityResult, path: Path) -> Path:
    payload: dict[str, np.ndarray] = {}
    for name, error in result.variable_errors.items():
        payload[f"{name}:max_abs_error"] = np.asarray(error.max_abs_error, dtype=np.float64)
        payload[f"{name}:rms_error"] = np.asarray(error.rms_error, dtype=np.float64)
        payload[f"{name}:relative_l2_error"] = np.asarray(error.relative_l2_error, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def _save_parity_plot(result: NativeTracedFieldLineSelectedFieldParityResult, path: Path) -> Path:
    field_names = list(result.field_names)
    x = np.arange(len(field_names))
    width = 0.24
    max_abs = [result.variable_errors[name].max_abs_error for name in field_names]
    rms = [result.variable_errors[name].rms_error for name in field_names]
    rel_l2 = [result.variable_errors[name].relative_l2_error for name in field_names]
    figure, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    axis.bar(x - width, max_abs, width=width, color="#bb3e03", label="max|Δ|")
    axis.bar(x, rms, width=width, color="#0a9396", label="RMS")
    axis.bar(x + width, rel_l2, width=width, color="#3a86ff", label="rel L2")
    axis.set_xticks(x, field_names)
    axis.set_ylabel("error metric")
    axis.set_title("Native traced-field-line reduced selected-field parity")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(frameon=False, ncol=3)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_comparison_json(
    result: NativeTracedFieldLineSelectedFieldParityResult,
    *,
    reference_profiles: dict[str, np.ndarray],
    candidate_profiles: dict[str, np.ndarray],
    path: Path,
) -> Path:
    payload = {
        "field_names": list(result.field_names),
        "comparison_profiles": {
            field_name: {
                "reference_radial_profile": np.asarray(reference_profiles[field_name], dtype=np.float64).tolist(),
                "native_radial_profile": np.asarray(candidate_profiles[field_name], dtype=np.float64).tolist(),
            }
            for field_name in result.field_names
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _save_comparison_plot(
    result: NativeTracedFieldLineSelectedFieldParityResult,
    *,
    reference_profiles: dict[str, np.ndarray],
    candidate_profiles: dict[str, np.ndarray],
    path: Path,
) -> Path:
    figure, axes = plt.subplots(len(result.field_names), 1, figsize=(10.5, 3.5 * len(result.field_names)), constrained_layout=True)
    if len(result.field_names) == 1:
        axes = [axes]
    for axis, field_name in zip(axes, result.field_names, strict=False):
        radial_index = np.arange(np.asarray(reference_profiles[field_name]).shape[0], dtype=np.float64)
        axis.plot(radial_index, reference_profiles[field_name], linewidth=2.0, color="#005f73", label="reference")
        axis.plot(radial_index, candidate_profiles[field_name], linewidth=2.0, color="#ca6702", label="native")
        axis.set_ylabel(field_name)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("radial index")
    axes[0].set_title("Native traced-field-line reduced radial profiles")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_runtime_report(
    *,
    field_names: tuple[str, ...],
    elapsed_seconds: float,
    reference_mesh_spec: Path,
    candidate_mesh_spec: Path,
    path: Path,
) -> Path:
    payload = {
        "geometry_family": "traced_field_line_3d",
        "benchmark_adapter": "native_traced_field_line_selected_field",
        "native_capability_tier": "native_exact_reduced",
        "selected_fields": list(field_names),
        "elapsed_seconds": float(elapsed_seconds),
        "jax_backend": jax.default_backend(),
        "reference_input_name": reference_mesh_spec.name,
        "candidate_input_name": candidate_mesh_spec.name,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
