from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report
from .stellarator_vmec_selected_field import _load_vmec_selected_fields, _write_candidate_from_reference_vmec
from .stellarator_vmec_scaffold import _write_synthetic_vmec_wout


@dataclass(frozen=True)
class NativeStellaratorVmecSelectedFieldVariableError:
    name: str
    max_abs_error: float
    rms_error: float
    relative_l2_error: float


@dataclass(frozen=True)
class NativeStellaratorVmecSelectedFieldParityResult:
    field_names: tuple[str, ...]
    variable_errors: dict[str, NativeStellaratorVmecSelectedFieldVariableError]


@dataclass(frozen=True)
class NativeStellaratorVmecSelectedFieldArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path
    comparison_json_path: Path
    comparison_plot_png_path: Path
    observable_report_json_path: Path
    runtime_report_json_path: Path


def create_native_stellarator_vmec_selected_field_package(
    *,
    reference_equilibrium_path: str | Path | None,
    candidate_equilibrium_path: str | Path | None,
    output_root: str | Path,
    case_label: str = "stellarator_vmec_native_selected_field",
    field_names: tuple[str, ...] = ("iota", "pressure", "toroidal_flux"),
) -> NativeStellaratorVmecSelectedFieldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    timer_start = perf_counter()
    result, reference_profiles, candidate_profiles, source_report = compare_native_stellarator_vmec_selected_fields(
        reference_equilibrium_path=reference_equilibrium_path,
        candidate_equilibrium_path=candidate_equilibrium_path,
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
        geometry_family="stellarator_vmec_3d",
        benchmark_adapter="native_stellarator_vmec_selected_field",
        observable_groups=(
            {
                "name": "native_reduced_selected_field",
                "description": "JAX-native reduced selected-field parity surface on stellarator VMEC equilibrium profiles.",
                "families": [
                    {
                        "name": "native_vmec_profiles",
                        "kind": "selected_field_parity",
                        "coordinate_name": "radial_index",
                        "field_names": list(result.field_names),
                    }
                ],
            },
        ),
        metadata={
            "compare_surface": "jax_native_vmec_profile_bundle",
            "native_capability_tier": "native_exact_reduced",
            "selected_fields": list(result.field_names),
            "source_mode": source_report["source_mode"],
            "candidate_origin": source_report["candidate_origin"],
        },
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    runtime_report_json_path = _write_runtime_report(
        field_names=result.field_names,
        elapsed_seconds=elapsed_seconds,
        source_report=source_report,
        path=data_dir / f"{case_label}_runtime_report.json",
    )
    return NativeStellaratorVmecSelectedFieldArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        comparison_json_path=comparison_json_path,
        comparison_plot_png_path=comparison_plot_png_path,
        observable_report_json_path=observable_report_json_path,
        runtime_report_json_path=runtime_report_json_path,
    )


def compare_native_stellarator_vmec_selected_fields(
    *,
    reference_equilibrium_path: str | Path | None,
    candidate_equilibrium_path: str | Path | None,
    field_names: tuple[str, ...] = ("iota", "pressure", "toroidal_flux"),
) -> tuple[
    NativeStellaratorVmecSelectedFieldParityResult,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    with tempfile.TemporaryDirectory(prefix="jax_drb_stellarator_vmec_native_") as temp_dir:
        temp_root = Path(temp_dir)
        reference_path, candidate_path, source_report = _resolve_input_pair(
            reference_equilibrium_path=reference_equilibrium_path,
            candidate_equilibrium_path=candidate_equilibrium_path,
            temp_root=temp_root,
        )
        reference_fields = _load_vmec_selected_fields(reference_path)
        candidate_fields = _load_vmec_selected_fields(candidate_path)
        resolved_field_names = tuple(field_names)
        reference_profiles = _native_reduce_vmec_profiles(reference_fields, resolved_field_names)
        candidate_profiles = _native_reduce_vmec_profiles(candidate_fields, resolved_field_names)

    variable_errors: dict[str, NativeStellaratorVmecSelectedFieldVariableError] = {}
    for field_name in resolved_field_names:
        reference = np.asarray(reference_profiles[field_name], dtype=np.float64)
        candidate = np.asarray(candidate_profiles[field_name], dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"Reduced VMEC profile shape mismatch for {field_name!r}: {reference.shape} vs {candidate.shape}")
        diff = candidate - reference
        reference_norm = float(np.linalg.norm(reference.ravel()))
        variable_errors[field_name] = NativeStellaratorVmecSelectedFieldVariableError(
            name=field_name,
            max_abs_error=float(np.max(np.abs(diff))),
            rms_error=float(np.sqrt(np.mean(np.square(diff)))),
            relative_l2_error=float(np.linalg.norm(diff.ravel()) / max(reference_norm, np.finfo(np.float64).tiny)),
        )
    return (
        NativeStellaratorVmecSelectedFieldParityResult(
            field_names=resolved_field_names,
            variable_errors=variable_errors,
        ),
        reference_profiles,
        candidate_profiles,
        source_report,
    )


def _resolve_input_pair(
    *,
    reference_equilibrium_path: str | Path | None,
    candidate_equilibrium_path: str | Path | None,
    temp_root: Path,
) -> tuple[Path, Path, dict[str, object]]:
    source_mode = "explicit_pair"
    candidate_origin = "provided_external_input"
    if reference_equilibrium_path is None and candidate_equilibrium_path is None:
        reference_path = temp_root / "reference_wout.nc"
        candidate_path = temp_root / "candidate_wout.nc"
        _write_synthetic_vmec_wout(reference_path)
        _write_candidate_from_reference_vmec(reference_path, candidate_path)
        source_mode = "synthetic_preview"
        candidate_origin = "synthetic_preview_pair"
    elif reference_equilibrium_path is not None and candidate_equilibrium_path is None:
        reference_path = Path(reference_equilibrium_path)
        candidate_path = temp_root / f"candidate{reference_path.suffix}"
        shutil.copy2(reference_path, candidate_path)
        _write_candidate_from_reference_vmec(reference_path, candidate_path)
        source_mode = "external_explicit_pair"
        candidate_origin = "materialized_from_reference_input"
    elif reference_equilibrium_path is None and candidate_equilibrium_path is not None:
        raise ValueError("candidate_equilibrium_path requires reference_equilibrium_path.")
    else:
        reference_path = Path(reference_equilibrium_path)
        candidate_path = Path(candidate_equilibrium_path)
    return (
        reference_path,
        candidate_path,
        {
            "source_mode": source_mode,
            "candidate_origin": candidate_origin,
            "reference_input_name": reference_path.name,
            "candidate_input_name": candidate_path.name,
        },
    )


@jax.jit
def _native_vmec_profile(values: jax.Array) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.float64)


def _native_reduce_vmec_profiles(
    fields: dict[str, np.ndarray],
    field_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    reduced: dict[str, np.ndarray] = {}
    for field_name in field_names:
        reduced[field_name] = np.asarray(_native_vmec_profile(jnp.asarray(fields[field_name], dtype=jnp.float64)), dtype=np.float64)
    return reduced


def _write_parity_json(result: NativeStellaratorVmecSelectedFieldParityResult, path: Path) -> Path:
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


def _write_parity_arrays(result: NativeStellaratorVmecSelectedFieldParityResult, path: Path) -> Path:
    payload: dict[str, np.ndarray] = {}
    for name, error in result.variable_errors.items():
        payload[f"{name}:max_abs_error"] = np.asarray(error.max_abs_error, dtype=np.float64)
        payload[f"{name}:rms_error"] = np.asarray(error.rms_error, dtype=np.float64)
        payload[f"{name}:relative_l2_error"] = np.asarray(error.relative_l2_error, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def _save_parity_plot(result: NativeStellaratorVmecSelectedFieldParityResult, path: Path) -> Path:
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
    axis.set_title("Native stellarator VMEC reduced selected-field parity")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(frameon=False, ncol=3)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_comparison_json(
    result: NativeStellaratorVmecSelectedFieldParityResult,
    *,
    reference_profiles: dict[str, np.ndarray],
    candidate_profiles: dict[str, np.ndarray],
    path: Path,
) -> Path:
    payload = {
        "field_names": list(result.field_names),
        "comparison_profiles": {
            field_name: {
                "reference_profile": np.asarray(reference_profiles[field_name], dtype=np.float64).tolist(),
                "native_profile": np.asarray(candidate_profiles[field_name], dtype=np.float64).tolist(),
            }
            for field_name in result.field_names
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _save_comparison_plot(
    result: NativeStellaratorVmecSelectedFieldParityResult,
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
    axes[0].set_title("Native stellarator VMEC reduced profiles")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_runtime_report(
    *,
    field_names: tuple[str, ...],
    elapsed_seconds: float,
    source_report: dict[str, object],
    path: Path,
) -> Path:
    payload = {
        "geometry_family": "stellarator_vmec_3d",
        "benchmark_adapter": "native_stellarator_vmec_selected_field",
        "native_capability_tier": "native_exact_reduced",
        "selected_fields": list(field_names),
        "elapsed_seconds": float(elapsed_seconds),
        "jax_backend": jax.default_backend(),
        "source_mode": source_report["source_mode"],
        "candidate_origin": source_report["candidate_origin"],
        "reference_input_name": source_report["reference_input_name"],
        "candidate_input_name": source_report["candidate_input_name"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
