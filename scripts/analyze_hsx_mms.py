#!/usr/bin/env python3
"""Analyze Stage-7 HSX/RLP MMS artifacts without running a simulation.

The production MMS driver writes one compact aggregate ``.npz`` per campaign.
This utility deliberately imports neither JAX nor the production driver: it is
safe to run on a login node after copying results from a compute node.  It
merges aggregate files, checks the hard reproducibility gates, and reports
spatial, temporal, and short-leg diagnostics in a small JSON document.

Examples
--------
``python scripts/analyze_hsx_mms.py work/stage7_mms/hsx_mms_frozen.npz``

``python scripts/analyze_hsx_mms.py \
    work/stage7_mms/hsx_mms_frozen.npz --require-spatial``

``python scripts/analyze_hsx_mms.py \
    work/stage7_mms/hsx_mms_frozen.npz \
    work/stage7_mms/hsx_mms_evolved_N64_t2e-5_dt5e-7.npz \
    --output work/stage7_mms/analysis.json``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_CONFIGURATION: dict[str, Any] = {
    "flux_framework": "production-split",
    "parallel_operator_scheme": "fci",
    "parallel_velocity_layout": "cell-centered",
    "parallel_flux_pairing": "support-core",
    "parallel_boundary_pairing": "characteristic-sat",
    "parallel_characteristic_wall_law": "energy-absorbing",
    "characteristic_sat_affine_current_lift": "enabled",
    "parallel_current_phi_pair": "enabled",
    "parallel_inflow_closure": "central",
    "parallel_short_leg_treatment": "local-backward-euler",
    "parallel_short_leg_selection": "all-physical-walls",
    "parallel_short_leg_cfl_limit": 2.5,
    "parallel_short_leg_implicit_terms": [
        "selected-characteristic-material-action",
        "selected-mu-tau-grad-parallel-Ti",
    ],
    "parallel_short_leg_explicit_energy_pair": (
        "mu-grad-parallel-phi<->weighted-adjoint-current-divergence"
    ),
    "fci_parallel_leg_scheme": "centered",
    "time_integrator": "imex-ssp222",
    "poisson_bracket_scheme": "material-scalar-third-order-upwind",
    "parallel_material_scheme": "production-path",
    "curvature_scheme": "conservative",
    "curvature_operator": "production-characteristic-owner-face",
    "curvature_rlp_face_scheme": "projected-fine",
    "curvature_wall_flux_closure": (
        "bc-characteristic-operator-trace-canonical-face-state"
    ),
    "curvature_equations": ["density", "Te", "Ti", "vorticity"],
    "curvature_evolution_component": "full",
    "vorticity_current_inflow_trace": "operator",
    "angular_rlp": "automatic-radius-dependent-projected-fine",
    "neumann_ghost_scheme": "physical",
    "physical_wall_model": "legacy-velocity-trace",
    "parallel_velocity_wall_bc": "neumann",
    "fci_trace_substeps": 4,
    "halo_width": 2,
    "fit_sample_shape": [64, 64, 64],
    "toroidal_modes": 10,
    "metric_reference_resolution": [64, 64, 64],
    "metric_radial_degree": 17,
    "metric_poloidal_modes": 15,
    "metric_toroidal_modes": 3,
    "eta_projection_iterations": 0,
    "axis_core_radius": 0.03,
    "makegrid_currents": [
        10722.0, 10722.0, 10722.0, 10722.0, 10722.0, 10722.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    "gmres_target_tolerance": 1.0e-8,
    "gmres_acceptance_tolerance": 5.0e-5,
    "gmres_max_iterations": 500,
    "gmres_restart": 100,
    "gmres_preconditioner": "line-u",
    "gmres_residual_correction_steps": 1,
    "evolved_initial_phi": "analytic-manufactured",
    "frozen_phi_audit": "exact-and-reconstructed",
    "reference_projection_method": "jacobian-weighted-cell-midpoint",
    "reference_projection_order": 2,
}

# Keep this list in the same spelling as FciDrbEBRhsParameters.  The MMS
# deliberately exercises perpendicular diffusion while leaving parallel
# diffusion/viscosity to the production wall/FCI machinery.
EXPECTED_PHYSICAL_PARAMETERS: dict[str, float] = {
    "tau": 1.0,
    "mi_over_me": 1836.0,
    "rho_star": 1.0,
    "density_D_perp": 1.0e-5,
    "density_D_parallel": 0.0,
    "electron_temperature_chi_parallel": 0.0,
    "electron_temperature_D_perp": 1.0e-5,
    "ion_temperature_chi_parallel": 0.0,
    "ion_temperature_D_perp": 1.0e-5,
    "Ve_nu": 0.0,
    "Ve_D_perp": 1.0e-5,
    "Ve_parallel_viscosity": 0.0,
    "Vi_D_perp": 1.0e-5,
    "Vi_parallel_viscosity": 0.0,
    "vorticity_D_perp": 1.0e-5,
    "vorticity_D_parallel": 0.0,
}
LEGACY_PARAMETER_METADATA_BASENAMES = {"hsx_mms_frozen8_final.npz"}

DEFAULT_REGIONS = (
    "ordinary_bulk",
    "rlp_rings",
    "rlp_transition_rings",
    "physical_wall",
    "short_leg_topology_transition",
    "double_hit",
)
DEFAULT_FIELDS = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")

GLOBAL_NORM_KEYS = (
    "exact_phi_residual",
    "forced_residual",
    "source_increment",
    "representation_error",
    "phi_reconstruction_difference",
    "reconstructed_phi_residual",
    "phi_reconstruction_rhs_difference",
    "integration_error",
    "integration_error_by_field",
    "continuum_total_error",
    "continuum_total_error_by_field",
)
REGIONAL_NORM_KEYS = (
    "partitioned_exact_phi_residual",
    "partitioned_forced_residual",
    "partitioned_rhs_term_norms",
    "partitioned_rhs_term_error_norms",
)
SPATIAL_KEYS = (
    "exact_phi_residual",
    "integration_error_by_field",
    "forced_residual",
    "source_increment",
    "representation_error",
    "reconstructed_phi_residual",
    "phi_reconstruction_rhs_difference",
    "rhs_term_error_norms",
    "partitioned_exact_phi_residual",
    "partitioned_forced_residual",
    "partitioned_rhs_term_error_norms",
)

CANONICAL_SPATIAL_RESOLUTIONS = (32, 48, 64)
MINIMUM_FINEST_PAIR_L2_ORDER = 1.8
REMOTE_SPATIAL_SHARD_COUNTS = (1, 1, 4)
REMOTE_SPATIAL_EXECUTION = "eta-sharded"
REFERENCE_DERIVATIVE_METHOD = "structured-nonuniform-five-point-finite-difference"
REFERENCE_PROJECTION_METHOD = "jacobian-weighted-cell-midpoint"
REFERENCE_PROJECTION_ORDER = 2
EVOLVED_BASELINE_START_TIME = 0.0
EVOLVED_BASELINE_FINAL_TIME = 2.0e-5
EVOLVED_BASELINE_TIMESTEP = 1.0e-6
EVOLVED_BASELINE_NUM_STEPS = 20
TEMPORAL_REFINEMENT_TIMESTEPS = (1.0e-6, 5.0e-7, 2.5e-7)
TEMPORAL_REFINEMENT_NUM_STEPS = (20, 40, 80)
KNOWN_SHORT_LEG_CLASSIFICATIONS = {
    "positive-growth",
    "bounded-or-decaying-closure-layer",
    "insufficient-time-samples",
}
DIAGNOSTIC_SPATIAL_KEYS = {
    "source_increment",
    "rhs_term_error_norms",
    "partitioned_exact_phi_residual",
    "partitioned_forced_residual",
    "partitioned_rhs_term_error_norms",
}


@dataclass(frozen=True)
class Artifact:
    path: Path
    arrays: Mapping[str, np.ndarray]
    configuration: Mapping[str, Any] | None
    fields: tuple[str, ...]
    regions: tuple[str, ...]
    rhs_terms: tuple[str, ...]
    command: tuple[str, ...]
    dt: float | None
    initial_time: float | None
    final_time: float | None
    dt_values: tuple[float, ...] | None = None
    initial_time_values: tuple[float, ...] | None = None
    final_time_values: tuple[float, ...] | None = None


def _scalar(value: Any) -> Any:
    """Convert a NumPy scalar to a Python value without losing strings."""

    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _json_scalar(arrays: Mapping[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in arrays:
        return default
    value = _scalar(arrays[key])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _tuple_of_strings(value: Any, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return tuple(default)


def _number(value: Any) -> float | None:
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        number = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text_scalar(value: Any) -> str | None:
    """Return one decoded text scalar, rejecting non-scalar metadata arrays."""

    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        scalar = array.reshape(-1)[0].item()
    except (AttributeError, TypeError, ValueError):
        return None
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return scalar if isinstance(scalar, str) else None


def _metadata_series(arrays: Mapping[str, np.ndarray], key: str) -> tuple[float, ...] | None:
    """Return finite per-row metadata, or ``None`` for scalar/malformed data."""

    if key not in arrays or np.asarray(arrays[key]).ndim == 0:
        return None
    try:
        values = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return tuple(float(value) for value in values)


def _command_float(command: Sequence[str], *names: str) -> float | None:
    for index, token in enumerate(command):
        for name in names:
            if token == name and index + 1 < len(command):
                try:
                    return float(command[index + 1])
                except ValueError:
                    continue
            if token.startswith(name + "="):
                try:
                    return float(token.split("=", 1)[1])
                except ValueError:
                    continue
    return None


_DT_IN_NAME = re.compile(r"(?:^|[_-])dt(?P<value>[0-9]+(?:p[0-9]+)?(?:e[+-]?[0-9]+)?)")


def _path_dt(path: Path) -> float | None:
    match = _DT_IN_NAME.search(path.stem.lower())
    if match is None:
        return None
    text = match.group("value").replace("p", ".")
    try:
        return float(text)
    except ValueError:
        return None


def load_artifact(path: str | Path) -> Artifact:
    """Load one aggregate MMS file using ``allow_pickle=False``."""

    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    config = _json_scalar(arrays, "production_configuration_json")
    if not isinstance(config, Mapping):
        config = None
    command_value = _json_scalar(arrays, "command_json", ())
    command = tuple(str(item) for item in command_value) if isinstance(command_value, (list, tuple)) else ()
    # Prefer explicit run metadata written by the driver.  Command lines and
    # filename conventions are compatibility fallbacks for older artifacts.
    direct_dt = _number(arrays.get("actual_timestep"))
    direct_initial = _number(arrays.get("start_time"))
    direct_final = _number(arrays.get("final_time"))
    direct_dt_values = _metadata_series(arrays, "actual_timestep")
    direct_initial_values = _metadata_series(arrays, "start_time")
    direct_final_values = _metadata_series(arrays, "final_time")
    dt = (
        direct_dt if direct_dt is not None
        else (direct_dt_values[0] if direct_dt_values else (_command_float(command, "--dt") or _path_dt(resolved)))
    )
    initial_time = (
        direct_initial if direct_initial is not None
        else (direct_initial_values[0] if direct_initial_values else _command_float(command, "--time", "--start-time"))
    )
    final_time = (
        direct_final if direct_final is not None
        else (direct_final_values[0] if direct_final_values else _command_float(command, "--final-time", "--end-time"))
    )
    return Artifact(
        path=resolved,
        arrays=arrays,
        configuration=config,
        fields=_tuple_of_strings(_json_scalar(arrays, "field_names_json"), DEFAULT_FIELDS),
        regions=_tuple_of_strings(_json_scalar(arrays, "region_names_json"), DEFAULT_REGIONS),
        rhs_terms=_tuple_of_strings(_json_scalar(arrays, "rhs_term_names_json")),
        command=command,
        dt=dt,
        initial_time=initial_time,
        final_time=final_time,
        dt_values=direct_dt_values,
        initial_time_values=direct_initial_values,
        final_time_values=direct_final_values,
    )


def _status(name: str, status: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **details}


def _array_finite(values: Any) -> bool:
    try:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))
    except (TypeError, ValueError):
        return False


def _resolution_rows(artifact: Artifact) -> list[dict[str, Any]]:
    values = artifact.arrays.get("resolutions")
    if values is None:
        return []
    resolutions = np.asarray(values).reshape(-1)
    rows: list[dict[str, Any]] = []
    for index, resolution in enumerate(resolutions):
        try:
            n = int(resolution)
        except (TypeError, ValueError):
            continue
        def row_metadata(series: tuple[float, ...] | None, fallback: float | None) -> float | None:
            return (
                series[index]
                if series is not None and len(series) == len(resolutions)
                else fallback
            )

        rows.append({
            "artifact": artifact,
            "index": index,
            "resolution": n,
            "dt": row_metadata(artifact.dt_values, artifact.dt),
            "initial_time": row_metadata(artifact.initial_time_values, artifact.initial_time),
            "final_time": row_metadata(artifact.final_time_values, artifact.final_time),
        })
    return rows


def _row_value(row: Mapping[str, Any], key: str) -> np.ndarray | None:
    artifact = row["artifact"]
    values = artifact.arrays.get(key)
    if values is None:
        return None
    index = int(row["index"])
    values = np.asarray(values)
    if values.ndim == 0:
        return values.copy()
    if index >= values.shape[0]:
        return None
    return np.asarray(values[index])


def _unique_spatial_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one row per resolution from a coherent spatial campaign.

    A multi-resolution frozen artifact is stronger spatial evidence than a
    one-resolution temporal refinement artifact.  Prefer the artifact with
    the most resolution rows before using the path as a deterministic tie
    breaker, so an N64 evolved file cannot silently replace the N64 row from a
    32/48/64 campaign.
    """

    selected: dict[int, dict[str, Any]] = {}
    candidates = list(rows)
    campaign_sizes: dict[int, int] = {}
    for row in candidates:
        artifact = row["artifact"]
        campaign_sizes[id(artifact)] = len(_resolution_rows(artifact))
    for row in sorted(
        candidates,
        key=lambda item: (
            int(item["resolution"]),
            -campaign_sizes.get(id(item["artifact"]), 0),
            str(item["artifact"].path),
            int(item["index"]),
        ),
    ):
        selected.setdefault(int(row["resolution"]), dict(row))
    return [selected[n] for n in sorted(selected)]


def _orders(resolutions: Sequence[int], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(resolutions) < 2:
        return np.empty((0,) + values.shape[1:], dtype=np.float64)
    orders = np.full((len(resolutions) - 1,) + values.shape[1:], np.nan, dtype=np.float64)
    scale = np.log(np.asarray(resolutions[1:], dtype=np.float64) / np.asarray(resolutions[:-1], dtype=np.float64))
    numerator = np.log(np.maximum(values[:-1], np.finfo(float).tiny) / np.maximum(values[1:], np.finfo(float).tiny))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        orders[...] = numerator / scale.reshape((-1,) + (1,) * (values.ndim - 1))
    valid = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (values[:-1] > 0.0) & (values[1:] > 0.0)
    return np.where(valid, orders, np.nan)


def _order_summary(resolutions: Sequence[int], values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    orders = _orders(resolutions, values)
    finite = orders[np.isfinite(orders)]
    decreasing = 0.0
    pairs = values[:-1] if len(resolutions) > 1 else np.empty(0)
    if pairs.size:
        valid = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (values[:-1] > 0.0) & (values[1:] > 0.0)
        if np.any(valid):
            decreasing = float(np.count_nonzero((values[:-1] > values[1:]) & valid) / np.count_nonzero(valid))
    return {
        "resolutions": [int(n) for n in resolutions],
        "values": values.tolist(),
        "orders": orders.tolist(),
        "finite_order_count": int(finite.size),
        "min_order": float(np.min(finite)) if finite.size else None,
        "median_order": float(np.median(finite)) if finite.size else None,
        "max_order": float(np.max(finite)) if finite.size else None,
        "decreasing_pair_fraction": decreasing,
    }


def _temporal_order_summary(timesteps: Sequence[float], values: np.ndarray) -> dict[str, Any]:
    """Observed order for decreasing ``dt`` (coarse timestep first)."""

    timesteps = [float(dt) for dt in timesteps]
    values = np.asarray(values, dtype=np.float64)
    # _orders assumes an increasing refinement coordinate.  1/dt is exactly
    # that coordinate while the public report keeps the more useful dt list.
    summary = _order_summary([1.0 / dt for dt in timesteps], values)
    summary["resolutions"] = timesteps
    return summary


def _configuration_check(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    metric_values: list[tuple[int, ...]] = []
    b0_values: list[float] = []
    nfp_values: list[int] = []
    trace_values: list[int] = []
    for artifact in artifacts:
        arrays = artifact.arrays
        if artifact.configuration is None:
            failures.append(f"{artifact.path}: missing production_configuration_json")
        else:
            for key, expected in EXPECTED_CONFIGURATION.items():
                actual = artifact.configuration.get(key)
                if actual != expected:
                    failures.append(f"{artifact.path}: {key}={actual!r}, expected {expected!r}")
        metric = arrays.get("metric_reference_resolution")
        if metric is None:
            failures.append(f"{artifact.path}: missing metric_reference_resolution")
        else:
            try:
                metric_values.append(tuple(int(item) for item in np.asarray(metric).reshape(-1)))
            except (TypeError, ValueError):
                failures.append(f"{artifact.path}: invalid metric_reference_resolution")
        b0 = _number(arrays.get("reference_magnetic_field")) if "reference_magnetic_field" in arrays else None
        if b0 is None or b0 <= 0.0:
            failures.append(f"{artifact.path}: invalid reference_magnetic_field")
        else:
            b0_values.append(b0)
        for key, target, values in (("nfp", None, nfp_values), ("fci_trace_substeps", 4, trace_values)):
            if key not in arrays:
                failures.append(f"{artifact.path}: missing {key}")
                continue
            try:
                value = int(_scalar(arrays[key]))
            except (TypeError, ValueError):
                failures.append(f"{artifact.path}: invalid {key}")
                continue
            values.append(value)
            if target is not None and value != target:
                failures.append(f"{artifact.path}: {key}={value}, expected {target}")
    if metric_values and any(item != (64, 64, 64) for item in metric_values):
        failures.append(f"metric reference resolutions are not all (64, 64, 64): {metric_values}")
    if len(set(metric_values)) > 1:
        failures.append(f"metric reference differs between artifacts: {metric_values}")
    if len(b0_values) > 1 and not np.allclose(b0_values, b0_values[0], rtol=1.0e-10, atol=1.0e-12):
        failures.append(f"reference magnetic field differs between artifacts: {b0_values}")
    if len(set(nfp_values)) > 1:
        failures.append(f"nfp differs between artifacts: {nfp_values}")
    if len(set(trace_values)) > 1:
        failures.append(f"fci_trace_substeps differs between artifacts: {trace_values}")
    names = [(artifact.fields, artifact.regions, artifact.rhs_terms) for artifact in artifacts]
    if names and any(item != names[0] for item in names[1:]):
        failures.append("field, region, or RHS-term names differ between artifacts")
    status = "fail" if failures else ("warning" if warnings else "pass")
    return _status(
        "production_configuration_and_fixed_metric",
        status,
        failures=failures,
        warnings=warnings,
        metric_reference_resolution=list(metric_values[0]) if metric_values else None,
        reference_magnetic_field=b0_values[0] if b0_values else None,
        nfp=nfp_values[0] if nfp_values else None,
        fci_trace_substeps=trace_values[0] if trace_values else None,
    )


def _physical_parameters_check(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Require the intended MMS physical-parameter scalar and compare exactly."""

    failures: list[str] = []
    warnings: list[str] = []
    metadata: list[dict[str, Any]] = []
    canonical_values: list[str] = []
    for artifact in artifacts:
        value = _json_scalar(artifact.arrays, "physical_parameters_json")
        if value is None:
            if artifact.path.name in LEGACY_PARAMETER_METADATA_BASENAMES:
                warnings.append(
                    f"{artifact.path}: missing physical_parameters_json (recognized legacy artifact)"
                )
                metadata.append({"path": str(artifact.path), "status": "legacy-missing"})
            else:
                failures.append(f"{artifact.path}: missing physical_parameters_json")
                metadata.append({"path": str(artifact.path), "status": "missing"})
            continue
        if not isinstance(value, Mapping):
            failures.append(f"{artifact.path}: physical_parameters_json is not an object")
            metadata.append({"path": str(artifact.path), "status": "invalid"})
            continue
        mismatches = {
            key: {"actual": value.get(key), "expected": expected}
            for key, expected in EXPECTED_PHYSICAL_PARAMETERS.items()
            if key not in value or value.get(key) != expected
        }
        if mismatches:
            failures.append(
                f"{artifact.path}: physical-parameter mismatch {json.dumps(mismatches, sort_keys=True)}"
            )
        # Canonical JSON makes equality a deliberate exact comparison rather
        # than a floating-point allclose check.  Extra metadata fields are
        # retained and must also agree across artifacts.
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        canonical_values.append(canonical)
        metadata.append({"path": str(artifact.path), "status": "present", "values": value})
    if canonical_values and any(item != canonical_values[0] for item in canonical_values[1:]):
        failures.append("physical_parameters_json differs exactly between artifacts")
    status = "fail" if failures else ("warning" if warnings else "pass")
    return _status(
        "physical_parameters",
        status,
        expected=EXPECTED_PHYSICAL_PARAMETERS,
        artifacts=metadata,
        warnings=warnings,
        failures=failures,
    )


def _finite_norm_check(artifacts: Sequence[Artifact], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    checked: list[str] = []
    for artifact in artifacts:
        resolutions = artifact.arrays.get("resolutions")
        if resolutions is None or np.asarray(resolutions).size == 0:
            failures.append(f"{artifact.path}: missing resolutions")
        elif np.any(np.asarray(resolutions) <= 0):
            failures.append(f"{artifact.path}: resolutions must be positive")
        for key in GLOBAL_NORM_KEYS:
            values = artifact.arrays.get(key)
            if values is None:
                continue
            # Frozen-only artifacts intentionally use NaN for the evolved
            # integration ledger.  That is an unavailable quantity, not a
            # non-finite result in a populated spatial region.
            try:
                numeric_values = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                failures.append(f"{artifact.path}: non-numeric global norm {key}")
                continue
            if numeric_values.size and np.all(np.isnan(numeric_values)):
                continue
            checked.append(f"{artifact.path.name}:{key}")
            if not _array_finite(numeric_values):
                failures.append(f"{artifact.path}: non-finite global norm {key}")
        counts = artifact.arrays.get("region_cell_counts")
        if counts is None:
            continue
        counts = np.asarray(counts)
        if counts.ndim != 2:
            failures.append(f"{artifact.path}: region_cell_counts must be 2-D")
            continue
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            failures.append(f"{artifact.path}: invalid region_cell_counts")
            continue
        for key in REGIONAL_NORM_KEYS:
            values = artifact.arrays.get(key)
            if values is None:
                continue
            checked.append(f"{artifact.path.name}:{key}")
            values = np.asarray(values)
            if values.ndim < 2 or values.shape[:2] != counts.shape:
                failures.append(f"{artifact.path}: {key} does not align with region_cell_counts")
                continue
            populated = counts > 0
            if np.any(populated) and not np.all(np.isfinite(values[populated])):
                failures.append(f"{artifact.path}: non-finite {key} in populated regions")
    status = "fail" if failures else ("pass" if checked else "warning")
    return _status("finite_populated_region_norms", status, failures=failures, checked=checked)


def _source_pairing_check(artifacts: Sequence[Artifact], tolerance: float) -> dict[str, Any]:
    failures: list[str] = []
    maxima: list[float] = []
    for artifact in artifacts:
        values = artifact.arrays.get("source_increment")
        if values is None:
            failures.append(f"{artifact.path}: missing source_increment")
            continue
        values = np.asarray(values, dtype=np.float64)
        if not _array_finite(values):
            failures.append(f"{artifact.path}: source_increment is non-finite")
            continue
        maxima.append(float(np.max(np.abs(values))) if values.size else 0.0)
        if values.size and np.max(np.abs(values)) > tolerance:
            failures.append(f"{artifact.path}: source pairing {np.max(np.abs(values)):.3e} exceeds {tolerance:.3e}")
    status = "fail" if failures else ("pass" if maxima else "warning")
    return _status(
        "independent_source_pairing_roundoff",
        status,
        tolerance=float(tolerance),
        maximum=float(max(maxima)) if maxima else None,
        per_artifact_maximum=maxima,
        failures=failures,
    )


def _matrix_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[list[int], np.ndarray] | None:
    selected = [row for row in rows if _row_value(row, key) is not None]
    finite_selected = []
    for row in selected:
        try:
            if np.any(np.isfinite(np.asarray(_row_value(row, key), dtype=np.float64))):
                finite_selected.append(row)
        except (TypeError, ValueError):
            pass
    # When frozen and evolved campaign files are merged, prefer the rows that
    # actually contain the evolved ledger instead of resolving an all-NaN
    # frozen placeholder by filename order.
    if finite_selected:
        selected = finite_selected
    if len(selected) < 2:
        return None
    selected = _unique_spatial_rows(selected)
    if len(selected) < 2:
        return None
    values = [_row_value(row, key) for row in selected]
    if any(value is None for value in values):
        return None
    arrays = [np.asarray(value, dtype=np.float64) for value in values if value is not None]
    try:
        stacked = np.stack(arrays, axis=0)
    except ValueError:
        return None
    return [int(row["resolution"]) for row in selected], stacked


def _spatial_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in SPATIAL_KEYS:
        matrix = _matrix_from_rows(rows, key)
        if matrix is None:
            result[key] = {"status": "unavailable"}
            continue
        resolutions, values = matrix
        summary = _order_summary(resolutions, values)
        finite_positive = np.all(np.isfinite(values) & (values > 0.0), axis=0)
        exact_zero = np.all(np.isfinite(values) & (values == 0.0), axis=0)
        unavailable = ~(finite_positive | exact_zero)
        if np.any(finite_positive):
            positive_values = values[:, finite_positive]
            decreasing = np.all(positive_values[:-1] > positive_values[1:], axis=0)
            assessed_count = int(np.count_nonzero(finite_positive))
            decreasing_count = int(np.count_nonzero(decreasing))
        else:
            assessed_count = 0
            decreasing_count = 0
        details = {
            **summary,
            "assessed_positive_component_count": assessed_count,
            "strictly_decreasing_component_count": decreasing_count,
            "exact_zero_or_disabled_component_count": int(np.count_nonzero(exact_zero)),
            "unavailable_component_count": int(np.count_nonzero(unavailable)),
        }
        if key in DIAGNOSTIC_SPATIAL_KEYS:
            # Term-by-term and wall-partition convergence is useful evidence,
            # but an exact-zero disabled lane or a closure with no populated
            # cells is not a passed order estimate.  Keep these ledgers
            # explicitly diagnostic rather than promoting them to a gate.
            result[key] = {"status": "diagnostic" if assessed_count else "unavailable", **details}
        elif assessed_count == 0:
            result[key] = {"status": "unavailable", **details}
        else:
            result[key] = {
                "status": "pass" if decreasing_count == assessed_count else "fail",
                **details,
            }
    assessed = [
        entry for entry in result.values()
        if entry.get("status") not in {"unavailable", "diagnostic"}
    ]
    status = (
        "fail" if any(entry["status"] == "fail" for entry in assessed)
        else ("pass" if assessed else "warning")
    )
    return {"status": status, "quantities": result}


def _canonical_spatial_matrix(
    rows: Sequence[Mapping[str, Any]], key: str
) -> tuple[np.ndarray | None, list[int]]:
    """Return the canonical 32/48/64 rows and report any missing resolutions."""

    matrix = _matrix_from_rows(rows, key)
    if matrix is None:
        return None, list(CANONICAL_SPATIAL_RESOLUTIONS)
    resolutions, values = matrix
    by_resolution = {int(n): index for index, n in enumerate(resolutions)}
    missing = [n for n in CANONICAL_SPATIAL_RESOLUTIONS if n not in by_resolution]
    if missing:
        return None, missing
    indices = [by_resolution[n] for n in CANONICAL_SPATIAL_RESOLUTIONS]
    return np.asarray(values, dtype=np.float64)[indices], []


def _primary_field_gate(
    values: np.ndarray,
    fields: Sequence[str],
    *,
    quantity: str,
    minimum_order: float | None,
) -> dict[str, Any]:
    """Apply the Stage-7 all-fields L2 decrease and optional order requirement."""

    values = np.asarray(values, dtype=np.float64)
    failures: list[str] = []
    if values.shape != (len(CANONICAL_SPATIAL_RESOLUTIONS), len(fields)):
        return {
            "quantity": quantity,
            "status": "fail",
            "failures": [
                f"expected shape {(len(CANONICAL_SPATIAL_RESOLUTIONS), len(fields))}, "
                f"got {values.shape}"
            ],
        }
    finite_positive = np.all(np.isfinite(values) & (values > 0.0), axis=0)
    pair_decrease = values[:-1] > values[1:]
    orders = _orders(CANONICAL_SPATIAL_RESOLUTIONS, values)
    finest_orders = orders[-1]
    field_reports: list[dict[str, Any]] = []
    for index, field in enumerate(fields):
        field_failures: list[str] = []
        if not finite_positive[index]:
            field_failures.append("L2 errors must be finite and positive at 32/48/64")
        else:
            if not bool(np.all(pair_decrease[:, index])):
                field_failures.append("L2 error does not strictly decrease on both refinement pairs")
            if minimum_order is not None and (
                not math.isfinite(float(finest_orders[index]))
                or finest_orders[index] < minimum_order
            ):
                field_failures.append(
                    f"48-to-64 L2 order {float(finest_orders[index]):.6g} is below {minimum_order:.6g}"
                )
        failures.extend(f"{field}: {message}" for message in field_failures)
        field_reports.append({
            "field": str(field),
            "values": values[:, index].tolist(),
            "pair_orders": orders[:, index].tolist(),
            "strictly_decreasing": bool(
                finite_positive[index] and np.all(pair_decrease[:, index])
            ),
            "finest_pair_order": (
                float(finest_orders[index]) if math.isfinite(float(finest_orders[index])) else None
            ),
            "status": "fail" if field_failures else "pass",
            "failures": field_failures,
        })
    return {
        "quantity": quantity,
        "status": "fail" if failures else "pass",
        "norm": "owner-volume-weighted-L2",
        "fields": field_reports,
        "failures": failures,
    }


def _spatial_gate(
    rows: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Artifact],
    *,
    minimum_order: float = MINIMUM_FINEST_PAIR_L2_ORDER,
) -> dict[str, Any]:
    """Hard Stage-7 spatial gate, with a valid frozen-only scope."""

    fields = tuple(rows[0]["artifact"].fields) if rows else DEFAULT_FIELDS
    contract_failures: list[str] = []
    contract: dict[str, Any] = {}
    if len(artifacts) != 1:
        contract_failures.append(
            f"remote spatial gate requires exactly one aggregate artifact, got {len(artifacts)}"
        )
    else:
        artifact = artifacts[0]
        arrays = artifact.arrays
        raw_resolutions = np.asarray(arrays.get("resolutions", ()))
        try:
            resolution_numbers = np.asarray(raw_resolutions, dtype=np.float64).reshape(-1)
            resolutions = tuple(int(value) for value in resolution_numbers)
        except (TypeError, ValueError, OverflowError):
            resolution_numbers = np.empty(0, dtype=np.float64)
            resolutions = ()
        contract["artifact"] = str(artifact.path)
        contract["ordered_resolutions"] = list(resolutions)
        if (
            raw_resolutions.ndim != 1
            or resolution_numbers.shape != (3,)
            or not np.array_equal(
                resolution_numbers,
                np.asarray(CANONICAL_SPATIAL_RESOLUTIONS, dtype=np.float64),
            )
        ):
            contract_failures.append(
                "aggregate resolutions must be the ordered one-dimensional sequence "
                f"{CANONICAL_SPATIAL_RESOLUTIONS}, got {resolutions}"
            )

        raw_shard_counts = np.asarray(arrays.get("shard_counts", ()))
        try:
            shard_numbers = np.asarray(raw_shard_counts, dtype=np.float64).reshape(-1)
            shard_counts = tuple(int(value) for value in shard_numbers)
        except (TypeError, ValueError, OverflowError):
            shard_numbers = np.empty(0, dtype=np.float64)
            shard_counts = ()
        contract["shard_counts"] = list(shard_counts)
        if (
            raw_shard_counts.ndim != 1
            or shard_numbers.shape != (3,)
            or not np.array_equal(
                shard_numbers,
                np.asarray(REMOTE_SPATIAL_SHARD_COUNTS, dtype=np.float64),
            )
        ):
            contract_failures.append(
                f"shard_counts must be {REMOTE_SPATIAL_SHARD_COUNTS}, got {shard_counts}"
            )

        device_count_value = _number(arrays.get("device_count"))
        device_count = (
            int(device_count_value)
            if device_count_value is not None and device_count_value.is_integer()
            else None
        )
        contract["device_count"] = device_count
        if device_count is None or device_count < 4:
            contract_failures.append(f"device_count must be at least 4, got {device_count!r}")

        for key in ("frozen_execution", "evolved_execution"):
            value = _text_scalar(arrays[key]) if key in arrays else None
            contract[key] = value
            if value != REMOTE_SPATIAL_EXECUTION:
                contract_failures.append(
                    f"{key} must be {REMOTE_SPATIAL_EXECUTION!r}, got {value!r}"
                )

        generalized = (
            _scalar(arrays["generalized_potential_enabled"])
            if "generalized_potential_enabled" in arrays else None
        )
        generalized_valid = isinstance(generalized, (bool, np.bool_)) and bool(generalized)
        contract["generalized_potential_enabled"] = (
            bool(generalized) if isinstance(generalized, (bool, np.bool_)) else None
        )
        if not generalized_valid:
            contract_failures.append(
                "generalized_potential_enabled must be present and boolean true"
            )

        projection_method = _text_scalar(
            arrays.get("reference_projection_method")
        )
        contract["reference_projection_method"] = projection_method
        if projection_method != REFERENCE_PROJECTION_METHOD:
            contract_failures.append(
                "reference_projection_method must be "
                f"{REFERENCE_PROJECTION_METHOD!r}, got {projection_method!r}"
            )
        projection_order_value = _number(
            arrays.get("reference_projection_order")
        )
        projection_order = (
            int(projection_order_value)
            if projection_order_value is not None
            and projection_order_value.is_integer()
            else None
        )
        contract["reference_projection_order"] = projection_order
        if projection_order != REFERENCE_PROJECTION_ORDER:
            contract_failures.append(
                "reference_projection_order must be "
                f"{REFERENCE_PROJECTION_ORDER}, got {projection_order!r}"
            )

        derivative_method = (
            _text_scalar(arrays["reference_derivative_method"])
            if "reference_derivative_method" in arrays else None
        )
        contract["reference_derivative_method"] = derivative_method
        if derivative_method != REFERENCE_DERIVATIVE_METHOD:
            contract_failures.append(
                f"reference_derivative_method must be {REFERENCE_DERIVATIVE_METHOD!r}, "
                f"got {derivative_method!r}"
            )

        derivative_order_value = _number(arrays.get("reference_derivative_order"))
        derivative_order = (
            int(derivative_order_value)
            if derivative_order_value is not None and derivative_order_value.is_integer()
            else None
        )
        contract["reference_derivative_order"] = derivative_order
        if derivative_order != 4:
            contract_failures.append(
                f"reference_derivative_order must be 4, got {derivative_order!r}"
            )

        periodic_axes = _json_scalar(arrays, "reference_periodic_axes_json")
        contract["reference_periodic_axes"] = periodic_axes
        if not isinstance(periodic_axes, Mapping) or any(
            periodic_axes.get(axis) != "geometry-domain" for axis in ("theta", "eta")
        ):
            contract_failures.append(
                "reference_periodic_axes_json must map both theta and eta to "
                "'geometry-domain'"
            )

    failures: list[str] = list(contract_failures)
    exact, missing_exact = _canonical_spatial_matrix(rows, "exact_phi_residual")
    if exact is None:
        exact_report = {
            "quantity": "exact_phi_residual",
            "status": "fail",
            "missing_resolutions": missing_exact,
            "failures": [f"missing canonical exact-phi rows: {missing_exact}"],
        }
    else:
        exact_report = _primary_field_gate(
            exact, fields, quantity="exact_phi_residual", minimum_order=None
        )
    failures.extend(exact_report.get("failures", ()))

    # Frozen artifacts deliberately store an all-NaN evolved ledger.  That is
    # a valid frozen-only gate, not evidence that evolved convergence passed.
    all_integration_values = [
        np.asarray(value, dtype=np.float64)
        for row in rows
        if (value := _row_value(row, "integration_error_by_field")) is not None
    ]
    has_evolved_data = any(np.any(np.isfinite(value)) for value in all_integration_values)
    if not has_evolved_data:
        evolved_report = {
            "quantity": "integration_error_by_field",
            "status": "not-run",
            "reason": "frozen-only artifact has no evolved-field error ledger",
        }
        scope = "frozen-only"
    else:
        evolved, missing_evolved = _canonical_spatial_matrix(rows, "integration_error_by_field")
        if evolved is None:
            evolved_report = {
                "quantity": "integration_error_by_field",
                "status": "fail",
                "missing_resolutions": missing_evolved,
                "failures": [f"missing canonical evolved-error rows: {missing_evolved}"],
            }
        else:
            evolved_report = _primary_field_gate(
                evolved,
                fields,
                quantity="integration_error_by_field",
                minimum_order=minimum_order,
            )
        failures.extend(evolved_report.get("failures", ()))
        scope = "frozen-and-evolved"
    return _status(
        "spatial_convergence_gate",
        "fail" if failures else "pass",
        scope=scope,
        full_stage7_spatial_gate=(scope == "frozen-and-evolved" and not failures),
        required_resolutions=list(CANONICAL_SPATIAL_RESOLUTIONS),
        remote_contract={
            "status": "fail" if contract_failures else "pass",
            **contract,
            "failures": contract_failures,
        },
        minimum_finest_pair_l2_order=float(minimum_order),
        exact_phi_owner_residual=exact_report,
        evolved_field_error=evolved_report,
        failures=failures,
    )


def _phi_flag_failures(artifacts: Sequence[Artifact]) -> tuple[list[str], dict[str, list[bool]]]:
    failures: list[str] = []
    flags: dict[str, list[bool]] = {"phi_converged": [], "phi_failed": []}
    for artifact in artifacts:
        for key in flags:
            if key not in artifact.arrays:
                continue
            values = np.asarray(artifact.arrays[key])
            try:
                if values.dtype.kind not in "bi":
                    numeric = np.asarray(values, dtype=np.float64)
                    if not np.all(np.isfinite(numeric)):
                        failures.append(f"{artifact.path}: non-finite {key}")
                    booleans = numeric.astype(bool)
                else:
                    booleans = values.astype(bool)
            except (TypeError, ValueError):
                failures.append(f"{artifact.path}: invalid {key}")
                continue
            flags[key].extend(bool(value) for value in booleans.reshape(-1))
            if key == "phi_failed" and np.any(booleans):
                failures.append(f"{artifact.path}: phi_failed contains true")
            if key == "phi_converged" and np.any(~booleans):
                failures.append(f"{artifact.path}: phi_converged contains false")
    return failures, flags


def _phi_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    artifacts = list({id(row["artifact"]): row["artifact"] for row in rows}.values())
    flag_failures, flags = _phi_flag_failures(artifacts)
    matrix = _matrix_from_rows(rows, "phi_reconstruction_difference")
    if matrix is None:
        return _status(
            "phi_reconstruction_evidence",
            "fail" if flag_failures else "unavailable",
            reason="phi_reconstruction_difference not present at two resolutions",
            flag_failures=flag_failures,
            flags=flags,
        )
    resolutions, values = matrix
    summary = _order_summary(resolutions, values)
    valid_pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (values[:-1] > 0) & (values[1:] > 0)
    improving = bool(np.any((values[:-1] > values[1:]) & valid_pairs))
    status = "fail" if flag_failures else ("pass" if improving else "warning")
    return _status(
        "phi_reconstruction_evidence",
        status,
        improving=improving,
        flag_failures=flag_failures,
        flags=flags,
        **summary,
    )


def _owner_representation_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    owner = _matrix_from_rows(rows, "exact_phi_residual")
    representation = _matrix_from_rows(rows, "representation_error")
    if owner is None or representation is None:
        return _status("owner_vs_fine_rlp_representation_error", "unavailable")
    owner_resolutions, owner_values = owner
    representation_resolutions, representation_values = representation
    if owner_resolutions != representation_resolutions or owner_values.shape != representation_values.shape:
        return _status("owner_vs_fine_rlp_representation_error", "fail", reason="error ledgers do not align")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = representation_values / owner_values
    return _status(
        "owner_vs_fine_rlp_representation_error",
        "pass" if np.any(np.isfinite(ratio)) else "warning",
        resolutions=owner_resolutions,
        owner_error=owner_values.tolist(),
        representation_error=representation_values.tolist(),
        representation_to_owner_ratio=ratio.tolist(),
    )


def _continuum_total_error_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report total continuum error without mislabeling it temporal error.

    At fixed spatial resolution ``||U_h,dt-u||`` contains the common spatial
    truncation/reconstruction error as well as time-integration error.  It is
    useful diagnostic context, but its timestep ratios are not a temporal
    order estimate.
    """

    groups: dict[tuple[int, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        values = _row_value(row, "continuum_total_error_by_field")
        if values is None:
            values = _row_value(row, "integration_error_by_field")
        if values is None:
            scalar = _row_value(row, "continuum_total_error")
            if scalar is None:
                scalar = _row_value(row, "integration_error")
            values = scalar
        if values is None or not np.any(np.isfinite(np.asarray(values, dtype=np.float64))):
            continue
        key = (int(row["resolution"]), row.get("final_time"))
        groups.setdefault(key, []).append(
            dict(row, continuum_total_error=np.asarray(values, dtype=np.float64))
        )
    reports: list[dict[str, Any]] = []
    for (resolution, final_time), members in sorted(groups.items()):
        by_dt: dict[float, dict[str, Any]] = {}
        for row in members:
            dt = row.get("dt")
            if dt is None or not math.isfinite(float(dt)) or float(dt) <= 0:
                continue
            else:
                by_dt.setdefault(float(dt), row)
        if not by_dt:
            continue
        ordered = [by_dt[dt] for dt in sorted(by_dt, reverse=True)]
        values = np.stack([
            np.asarray(row["continuum_total_error"], dtype=np.float64)
            for row in ordered
        ])
        dts = [float(row["dt"]) for row in ordered]
        reports.append({
            "resolution": resolution,
            "final_time": final_time,
            "timesteps": dts,
            "values": values.tolist(),
        })
    return _status(
        "continuum_total_error_by_timestep",
        "diagnostic" if reports else "unavailable",
        groups=reports,
        interpretation=(
            "total continuum error includes spatial/reconstruction and temporal "
            "components; no temporal order is inferred from these values"
        ),
    )


def _resolve_related_path(artifact: Artifact, raw: Any) -> Path | None:
    """Resolve a companion artifact, including after a campaign is copied."""

    if not raw or not isinstance(raw, str):
        return None
    candidate = Path(raw).expanduser()
    candidates = (
        [candidate, artifact.path.parent / candidate.name, Path.cwd() / candidate.name]
        if candidate.is_absolute()
        else [artifact.path.parent / candidate, Path.cwd() / candidate]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _resolve_diagnostic_path(artifact: Artifact, raw: Any) -> Path | None:
    return _resolve_related_path(artifact, raw)


def _validate_short_leg_companion(
    artifact: Artifact, *, index: int, resolution: int
) -> tuple[dict[str, Any], list[str], int]:
    """Validate one evolved campaign's durable short-leg diagnostic."""

    failures: list[str] = []
    paths = _json_scalar(artifact.arrays, "short_leg_diagnostics_paths_json", [])
    fallback = _json_scalar(artifact.arrays, "short_leg_classification_json", [])
    raw_path = paths[index] if isinstance(paths, list) and index < len(paths) else None
    path = _resolve_diagnostic_path(artifact, raw_path)
    record: dict[str, Any] = {
        "artifact": str(artifact.path),
        "resolution": int(resolution),
        "path": str(path) if path is not None else None,
    }
    if path is None:
        failures.append(
            f"{artifact.path}: unresolved short-leg diagnostic for N={resolution}: {raw_path!r}"
        )
        return record, failures, 0
    try:
        with np.load(path, allow_pickle=False) as source:
            diagnostic = {key: np.asarray(source[key]) for key in source.files}
    except (OSError, ValueError, KeyError) as error:
        failures.append(f"{path}: unreadable short-leg diagnostic ({error})")
        return record, failures, 0

    required_series = (
        "times",
        "high_mode_rms",
        "maximum_poloidal_jump",
    )
    missing = [key for key in required_series if key not in diagnostic]
    if missing:
        failures.append(f"{path}: missing short-leg series {missing}")
    try:
        times = np.asarray(diagnostic.get("times", ()), dtype=np.float64)
    except (TypeError, ValueError):
        times = np.empty((0,), dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        failures.append(f"{path}: times must be a nonempty finite 1-D series")
    for key in required_series[1:]:
        try:
            values = np.asarray(diagnostic.get(key, ()), dtype=np.float64)
        except (TypeError, ValueError):
            values = np.empty((0,), dtype=np.float64)
        if (
            values.ndim < 1
            or values.shape[0] != times.size
            or not np.all(np.isfinite(values))
        ):
            failures.append(f"{path}: {key} must be finite and align with times")

    row_fallback = (
        fallback[index]
        if isinstance(fallback, list) and index < len(fallback)
        else []
    )
    classifications = _json_scalar(diagnostic, "classification_json", row_fallback)
    if not isinstance(classifications, list):
        classifications = []
    unknown = [
        str(value)
        for value in classifications
        if value not in KNOWN_SHORT_LEG_CLASSIFICATIONS
    ]
    if len(classifications) != len(artifact.fields):
        failures.append(
            f"{path}: expected {len(artifact.fields)} short-leg classifications, "
            f"got {len(classifications)}"
        )
    if unknown:
        failures.append(f"{path}: unknown short-leg classifications {unknown}")
    positive = sum(value == "positive-growth" for value in classifications)
    record.update(
        sample_count=int(times.size),
        classifications=[str(value) for value in classifications],
        unknown_classifications=unknown,
    )
    return record, failures, int(positive)


def _history_path_for_resolution(
    artifact: Artifact, resolution: int
) -> tuple[Path | None, str | None]:
    paths = _json_scalar(artifact.arrays, "history_paths_json", [])
    resolutions = _exact_resolutions(artifact)
    if not isinstance(paths, list) or resolutions is None:
        return None, "missing history_paths_json or resolutions"
    try:
        index = resolutions.index(int(resolution))
    except ValueError:
        return None, f"artifact has no N={resolution} row"
    if index >= len(paths):
        return None, f"history_paths_json has no entry for N={resolution}"
    path = _resolve_related_path(artifact, paths[index])
    if path is None:
        return None, f"unresolved N={resolution} production history: {paths[index]!r}"
    return path, None


def _load_temporal_endpoint(
    artifact: Artifact,
    *,
    resolution: int,
    expected_final_time: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    """Load one final production state and its exact owner-volume measure."""

    path, path_error = _history_path_for_resolution(artifact, resolution)
    record: dict[str, Any] = {
        "artifact": str(artifact.path),
        "resolution": int(resolution),
        "history": str(path) if path is not None else None,
    }
    failures: list[str] = []
    if path_error is not None or path is None:
        failures.append(f"{artifact.path}: {path_error}")
        return None, record, failures
    try:
        with np.load(path, allow_pickle=False) as source:
            history_dtype = _text_scalar(
                source["history_dtype"] if "history_dtype" in source.files else None
            )
            try:
                times = np.array(source["times"], dtype=np.float64, copy=True)
            except (KeyError, TypeError, ValueError):
                times = np.empty((0,), dtype=np.float64)
            try:
                active = np.array(source["owner_active"], dtype=bool, copy=True)
                volume = np.array(
                    source["owner_aggregate_volume"], dtype=np.float64, copy=True
                )
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    f"{path}: missing/invalid owner-volume history metadata ({error})"
                )
                return None, record, failures

            final_fields: dict[str, np.ndarray] = {}
            for field in artifact.fields:
                try:
                    values = np.asarray(source[field], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as error:
                    failures.append(
                        f"{path}: missing/invalid history field {field!r} ({error})"
                    )
                    continue
                expected_shape = (times.size,) + active.shape
                if values.shape != expected_shape:
                    failures.append(
                        f"{path}: field {field!r} shape {values.shape}, "
                        f"expected {expected_shape}"
                    )
                elif not np.all(np.isfinite(values[-1])):
                    failures.append(f"{path}: non-finite final {field!r} state")
                else:
                    # Copy only the final frame.  Retaining a view would keep
                    # the full 101/201/401-frame N64 history alive while the
                    # other timestep histories are loaded.
                    final_fields[field] = np.array(values[-1], copy=True)
    except (OSError, ValueError, KeyError) as error:
        failures.append(f"{path}: unreadable production history ({error})")
        return None, record, failures

    record["history_dtype"] = history_dtype
    if history_dtype != "float64":
        failures.append(f"{path}: history_dtype={history_dtype!r}, expected 'float64'")
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or not np.isclose(times[-1], expected_final_time, rtol=0.0, atol=1.0e-14)
    ):
        failures.append(
            f"{path}: times must be finite, include initial/final frames, and end "
            f"at {expected_final_time:.16g}"
        )
    record["saved_frame_count"] = int(times.size)
    record["final_time"] = float(times[-1]) if times.size else None

    if active.ndim != 3 or volume.shape != active.shape:
        failures.append(
            f"{path}: owner_active/owner_aggregate_volume shapes must match in 3-D"
        )
    elif (
        not np.any(active)
        or not np.all(np.isfinite(volume))
        or np.any(volume < 0.0)
        or np.any(volume[active] <= 0.0)
        or not float(np.sum(volume[active])) > 0.0
    ):
        failures.append(f"{path}: owner aggregate-volume measure is invalid")

    if failures:
        return None, record, failures
    return {
        "path": path,
        "active": active,
        "volume": volume,
        "fields": final_fields,
    }, record, []


def _complete_stage7_report(
    artifacts: Sequence[Artifact],
    *,
    required: bool,
    minimum_order: float = MINIMUM_FINEST_PAIR_L2_ORDER,
) -> dict[str, Any]:
    """Gate the exact four-campaign chain using temporal self-convergence.

    The total continuum ledger is intentionally not used here.  At fixed N64
    the differences between three production final states cancel their common
    spatial truncation to leading order, leaving a genuine timestep
    self-convergence estimate.
    """

    if len(artifacts) != 4:
        status = "fail" if required else "unavailable"
        return _status(
            "complete_stage7_acceptance",
            status,
            required=bool(required),
            failures=(
                [f"complete Stage-7 acceptance requires exactly four artifacts, got {len(artifacts)}"]
                if required else []
            ),
            interpretation=(
                "expected frozen, evolved dt=1e-6, N64 dt=5e-7, N64 dt=2.5e-7"
            ),
        )

    failures: list[str] = []
    evolved_gate = _evolved_prerequisite_gate(artifacts[:2])
    if evolved_gate["status"] == "fail":
        failures.append("canonical frozen/evolved prerequisite gate failed")
        failures.extend(
            f"evolved prerequisite: {message}"
            for message in evolved_gate.get("failures", ())
        )
    evolved_spatial_gate = _spatial_gate(
        _resolution_rows(artifacts[1]), (artifacts[1],)
    )
    if (
        evolved_spatial_gate["status"] != "pass"
        or evolved_spatial_gate.get("scope") != "frozen-and-evolved"
        or not evolved_spatial_gate.get("full_stage7_spatial_gate", False)
    ):
        failures.append(
            "canonical evolved 32/48/64 spatial convergence gate failed"
        )
        failures.extend(
            f"evolved spatial gate: {message}"
            for message in evolved_spatial_gate.get("failures", ())
        )

    # The frozen prerequisite already applies this contract through
    # _spatial_gate.  Require the same production execution/reference
    # identity on the baseline and both temporal artifacts rather than
    # accepting a single-device or legacy refinement into the final gate.
    for artifact in artifacts[1:]:
        arrays = artifact.arrays
        raw_shard_counts = np.asarray(arrays.get("shard_counts", ()))
        try:
            shard_numbers = np.asarray(raw_shard_counts, dtype=np.float64).reshape(-1)
            shard_counts = tuple(int(value) for value in shard_numbers)
        except (TypeError, ValueError, OverflowError):
            shard_numbers = np.empty((0,), dtype=np.float64)
            shard_counts = ()
        if (
            raw_shard_counts.ndim != 1
            or shard_numbers.shape != (3,)
            or not np.array_equal(
                shard_numbers,
                np.asarray(REMOTE_SPATIAL_SHARD_COUNTS, dtype=np.float64),
            )
        ):
            failures.append(
                f"{artifact.path}: shard_counts={shard_counts}, expected {REMOTE_SPATIAL_SHARD_COUNTS}"
            )
        device_value = _number(arrays.get("device_count"))
        if device_value is None or not device_value.is_integer() or int(device_value) < 4:
            failures.append(
                f"{artifact.path}: device_count must be an integer at least 4"
            )
        for key in ("frozen_execution", "evolved_execution"):
            value = _text_scalar(arrays.get(key))
            if value != REMOTE_SPATIAL_EXECUTION:
                failures.append(
                    f"{artifact.path}: {key}={value!r}, expected {REMOTE_SPATIAL_EXECUTION!r}"
                )
        generalized = _scalar(arrays["generalized_potential_enabled"]) \
            if "generalized_potential_enabled" in arrays else None
        if not isinstance(generalized, (bool, np.bool_)) or not bool(generalized):
            failures.append(
                f"{artifact.path}: generalized_potential_enabled must be boolean true"
            )
        projection_method = _text_scalar(
            arrays.get("reference_projection_method")
        )
        if projection_method != REFERENCE_PROJECTION_METHOD:
            failures.append(
                f"{artifact.path}: reference_projection_method="
                f"{projection_method!r}, expected {REFERENCE_PROJECTION_METHOD!r}"
            )
        projection_order = _number(arrays.get("reference_projection_order"))
        if projection_order != float(REFERENCE_PROJECTION_ORDER):
            failures.append(
                f"{artifact.path}: reference_projection_order="
                f"{projection_order!r}, expected {REFERENCE_PROJECTION_ORDER}"
            )
        derivative_method = _text_scalar(arrays.get("reference_derivative_method"))
        if derivative_method != REFERENCE_DERIVATIVE_METHOD:
            failures.append(
                f"{artifact.path}: reference_derivative_method={derivative_method!r}, "
                f"expected {REFERENCE_DERIVATIVE_METHOD!r}"
            )
        derivative_order = _number(arrays.get("reference_derivative_order"))
        if derivative_order != 4.0:
            failures.append(
                f"{artifact.path}: reference_derivative_order={derivative_order!r}, expected 4"
            )
        periodic_axes = _json_scalar(arrays, "reference_periodic_axes_json")
        if not isinstance(periodic_axes, Mapping) or any(
            periodic_axes.get(axis) != "geometry-domain" for axis in ("theta", "eta")
        ):
            failures.append(
                f"{artifact.path}: reference_periodic_axes_json must use geometry-domain theta/eta"
            )

    expected_resolutions = (
        CANONICAL_SPATIAL_RESOLUTIONS,
        CANONICAL_SPATIAL_RESOLUTIONS,
        (64,),
        (64,),
    )
    for artifact, expected in zip(artifacts, expected_resolutions, strict=True):
        actual = _exact_resolutions(artifact)
        if actual != tuple(expected):
            failures.append(
                f"{artifact.path}: resolutions={actual}, expected exactly {tuple(expected)}"
            )
    reference_fields = artifacts[1].fields
    fields_compatible = all(
        artifact.fields == reference_fields for artifact in artifacts[1:]
    )
    if not fields_compatible:
        failures.append(
            "evolved/temporal artifact field-name tuples must be identical"
        )

    for artifact, timestep, num_steps in zip(
        artifacts[1:],
        TEMPORAL_REFINEMENT_TIMESTEPS,
        TEMPORAL_REFINEMENT_NUM_STEPS,
        strict=True,
    ):
        expected = {
            "start_time": EVOLVED_BASELINE_START_TIME,
            "final_time": EVOLVED_BASELINE_FINAL_TIME,
            "actual_timestep": timestep,
            "num_steps": float(num_steps),
        }
        for key, target in expected.items():
            value = _number(artifact.arrays.get(key))
            tolerance = 0.0 if key == "num_steps" else max(1.0e-15, abs(target) * 1.0e-12)
            if value is None or not np.isclose(value, target, rtol=0.0, atol=tolerance):
                failures.append(
                    f"{artifact.path}: {key}={value!r}, expected {target:.16g}"
                )

    endpoints: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    for artifact in artifacts[1:]:
        endpoint, record, endpoint_failures = _load_temporal_endpoint(
            artifact,
            resolution=64,
            expected_final_time=EVOLVED_BASELINE_FINAL_TIME,
        )
        history_records.append(record)
        failures.extend(endpoint_failures)
        if endpoint is not None:
            endpoints.append(endpoint)

    temporal_diagnostic_records: list[dict[str, Any]] = []
    temporal_positive_growth_count = 0
    for artifact in artifacts[2:]:
        record, diagnostic_failures, positive_count = _validate_short_leg_companion(
            artifact, index=0, resolution=64
        )
        temporal_diagnostic_records.append(record)
        failures.extend(diagnostic_failures)
        temporal_positive_growth_count += positive_count

    field_reports: list[dict[str, Any]] = []
    if len(endpoints) == 3:
        reference_active = endpoints[0]["active"]
        reference_volume = endpoints[0]["volume"]
        compatible_measures = True
        for endpoint in endpoints[1:]:
            if (
                endpoint["active"].shape != reference_active.shape
                or endpoint["volume"].shape != reference_volume.shape
            ):
                failures.append(
                    f"{endpoint['path']}: owner measure shape differs from the dt=1e-6 history"
                )
                compatible_measures = False
                continue
            if not np.array_equal(endpoint["active"], reference_active):
                failures.append(
                    f"{endpoint['path']}: owner_active differs from the dt=1e-6 history"
                )
                compatible_measures = False
            if not np.allclose(
                endpoint["volume"], reference_volume, rtol=1.0e-12, atol=1.0e-14
            ):
                failures.append(
                    f"{endpoint['path']}: owner_aggregate_volume differs from the dt=1e-6 history"
                )
                compatible_measures = False
        if compatible_measures and fields_compatible:
            total_volume = float(np.sum(reference_volume[reference_active]))
            for field in artifacts[1].fields:
                coarse_difference = (
                    endpoints[0]["fields"][field] - endpoints[1]["fields"][field]
                )
                fine_difference = (
                    endpoints[1]["fields"][field] - endpoints[2]["fields"][field]
                )
                coarse_norm = float(np.sqrt(
                    np.sum(reference_volume[reference_active] * coarse_difference[reference_active] ** 2)
                    / total_volume
                ))
                fine_norm = float(np.sqrt(
                    np.sum(reference_volume[reference_active] * fine_difference[reference_active] ** 2)
                    / total_volume
                ))
                order = (
                    float(math.log(coarse_norm / fine_norm, 2.0))
                    if coarse_norm > 0.0 and fine_norm > 0.0
                    else None
                )
                field_failures: list[str] = []
                if order is None or not math.isfinite(order):
                    field_failures.append("temporal self-differences must be finite and positive")
                elif order < minimum_order:
                    field_failures.append(
                        f"temporal self-convergence order {order:.6g} is below {minimum_order:.6g}"
                    )
                failures.extend(f"{field}: {message}" for message in field_failures)
                field_reports.append({
                    "field": field,
                    "coarse_difference_norm": coarse_norm,
                    "fine_difference_norm": fine_norm,
                    "order": order,
                    "status": "fail" if field_failures else "pass",
                    "failures": field_failures,
                })

    if failures:
        status = "fail" if required else "warning"
    elif evolved_gate["status"] == "warning" or temporal_positive_growth_count:
        status = "warning"
    else:
        status = "pass"
    return _status(
        "complete_stage7_acceptance",
        status,
        required=bool(required),
        artifact_order=(
            "frozen-32/48/64",
            "evolved-32/48/64-dt1e-6",
            "evolved-N64-dt5e-7",
            "evolved-N64-dt2.5e-7",
        ),
        timesteps=list(TEMPORAL_REFINEMENT_TIMESTEPS),
        minimum_temporal_order=float(minimum_order),
        temporal_error_definition=(
            "owner-volume-weighted N64 final-state self-convergence; "
            "common spatial truncation cancels from pairwise differences"
        ),
        total_continuum_error_is_not_temporal_error=True,
        evolved_prerequisite_gate=evolved_gate,
        evolved_spatial_convergence_gate=evolved_spatial_gate,
        history_records=history_records,
        temporal_short_leg_diagnostic_records=temporal_diagnostic_records,
        temporal_positive_growth_count=int(temporal_positive_growth_count),
        temporal_self_convergence_by_field=field_reports,
        failures=failures,
    )


def _short_leg_report(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    known = KNOWN_SHORT_LEG_CLASSIFICATIONS
    for artifact in artifacts:
        paths = _json_scalar(artifact.arrays, "short_leg_diagnostics_paths_json", [])
        classifications = _json_scalar(artifact.arrays, "short_leg_classification_json", [])
        if not isinstance(paths, list):
            paths = []
        if not isinstance(classifications, list):
            classifications = []
        resolutions = np.asarray(artifact.arrays.get("resolutions", ())).reshape(-1)
        for index, resolution in enumerate(resolutions):
            raw_diag_path = paths[index] if index < len(paths) else None
            diag_path = _resolve_diagnostic_path(artifact, raw_diag_path)
            row_classes = classifications[index] if index < len(classifications) else None
            diag: dict[str, Any] = {}
            if diag_path is not None:
                try:
                    with np.load(diag_path, allow_pickle=False) as payload:
                        diag_arrays = {key: np.asarray(payload[key]) for key in payload.files}
                    classes = _json_scalar(diag_arrays, "classification_json", row_classes)
                    if isinstance(classes, list):
                        row_classes = classes
                    times = np.asarray(diag_arrays.get("times", ()), dtype=np.float64)
                    diag = {
                        "path": str(diag_path),
                        "sample_count": int(times.size),
                        "late_log_growth_rate": np.asarray(diag_arrays.get("late_log_growth_rate", ()), dtype=float).tolist(),
                        "late_growth_factor": np.asarray(diag_arrays.get("late_growth_factor", ()), dtype=float).tolist(),
                        "late_growth_r_squared": np.asarray(diag_arrays.get("late_growth_r_squared", ()), dtype=float).tolist(),
                    }
                    finite_series = all(
                        _array_finite(diag_arrays[key])
                        for key in ("times", "high_mode_rms", "maximum_poloidal_jump")
                        if key in diag_arrays
                    )
                    diag["finite_series"] = finite_series
                except (OSError, ValueError, KeyError) as exc:
                    missing.append(f"{diag_path}: unreadable short-leg diagnostics ({exc})")
            elif paths and index < len(paths) and paths[index]:
                missing.append(f"{artifact.path}: missing short-leg diagnostics {paths[index]}")
            if not isinstance(row_classes, list):
                row_classes = []
            # Frozen-only rows encode absent optional diagnostics as null/[];
            # do not turn those placeholders into misleading empty records.
            if not diag_path and not raw_diag_path and not row_classes:
                continue
            bad = [str(item) for item in row_classes if item not in known]
            records.append({
                "artifact": str(artifact.path),
                "resolution": int(resolution),
                "classifications": [str(item) for item in row_classes],
                "unknown_classifications": bad,
                **diag,
            })
    if not records:
        return _status("short_leg_growth_classification", "unavailable", records=[])
    positive = [record for record in records if "positive-growth" in record["classifications"]]
    bad = [record for record in records if record["unknown_classifications"]]
    status = "fail" if bad else ("warning" if positive or missing else "pass")
    return _status(
        "short_leg_growth_classification",
        status,
        records=records,
        positive_growth_count=len(positive),
        missing_diagnostics=missing,
        interpretation=("localized high-mode growth requires investigation" if positive else "no sustained positive-growth classification"),
    )


def _exact_resolutions(artifact: Artifact) -> tuple[int, ...] | None:
    """Return the artifact resolution sequence without merging other files."""

    try:
        values = np.asarray(artifact.arrays["resolutions"]).reshape(-1)
        return tuple(int(value) for value in values)
    except (KeyError, TypeError, ValueError):
        return None


def _evolved_prerequisite_gate(artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Validate the canonical frozen artifact and 20-step evolved baseline.

    This is deliberately a structural prerequisite gate.  Positive localized
    growth remains scientific evidence to investigate during temporal
    refinement; malformed or incomplete diagnostics prevent that refinement
    from being launched.
    """

    failures: list[str] = []
    if len(artifacts) != 2:
        return _status(
            "evolved_prerequisite_gate",
            "fail",
            failures=[
                "--require-evolved expects exactly two artifacts in order: "
                "frozen 32/48/64, then evolved 32/48/64 baseline"
            ],
        )

    frozen, baseline = artifacts
    expected_resolutions = tuple(CANONICAL_SPATIAL_RESOLUTIONS)
    frozen_resolutions = _exact_resolutions(frozen)
    baseline_resolutions = _exact_resolutions(baseline)
    if frozen_resolutions != expected_resolutions:
        failures.append(
            f"{frozen.path}: frozen resolutions are {frozen_resolutions}, "
            f"expected exactly {expected_resolutions}"
        )
    if baseline_resolutions != expected_resolutions:
        failures.append(
            f"{baseline.path}: evolved resolutions are {baseline_resolutions}, "
            f"expected exactly {expected_resolutions}"
        )

    frozen_spatial = _spatial_gate(_resolution_rows(frozen), (frozen,))
    if frozen_spatial["status"] != "pass" or frozen_spatial.get("scope") != "frozen-only":
        failures.append(
            f"{frozen.path}: canonical frozen spatial gate did not pass as frozen-only"
        )
        failures.extend(
            f"frozen spatial gate: {message}"
            for message in frozen_spatial.get("failures", ())
        )

    frozen_steps = _number(frozen.arrays.get("num_steps"))
    frozen_dt = _number(frozen.arrays.get("actual_timestep"))
    frozen_start = _number(frozen.arrays.get("start_time"))
    frozen_final = _number(frozen.arrays.get("final_time"))
    if frozen_steps != 0.0:
        failures.append(f"{frozen.path}: frozen num_steps={frozen_steps!r}, expected 0")
    if frozen_dt != 0.0:
        failures.append(f"{frozen.path}: frozen actual_timestep={frozen_dt!r}, expected 0")
    if frozen_start is None or frozen_final is None or not np.isclose(
        frozen_start, frozen_final, rtol=0.0, atol=1.0e-15
    ):
        failures.append(
            f"{frozen.path}: frozen start/final metadata must be finite and equal"
        )

    expected_scalars = {
        "start_time": EVOLVED_BASELINE_START_TIME,
        "final_time": EVOLVED_BASELINE_FINAL_TIME,
        "actual_timestep": EVOLVED_BASELINE_TIMESTEP,
        "num_steps": float(EVOLVED_BASELINE_NUM_STEPS),
    }
    actual_scalars: dict[str, float | None] = {}
    for key, expected in expected_scalars.items():
        actual = _number(baseline.arrays.get(key))
        actual_scalars[key] = actual
        tolerance = 0.0 if key == "num_steps" else max(1.0e-15, abs(expected) * 1.0e-12)
        if actual is None or not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
            failures.append(
                f"{baseline.path}: {key}={actual!r}, expected {expected:.16g}"
            )

    integration = baseline.arrays.get("integration_error_by_field")
    expected_shape = (len(expected_resolutions), len(baseline.fields))
    if integration is None:
        failures.append(f"{baseline.path}: missing integration_error_by_field")
    else:
        try:
            integration_values = np.asarray(integration, dtype=np.float64)
        except (TypeError, ValueError):
            integration_values = np.empty((0,), dtype=np.float64)
        if integration_values.shape != expected_shape:
            failures.append(
                f"{baseline.path}: integration_error_by_field shape "
                f"{integration_values.shape}, expected {expected_shape}"
            )
        elif not np.all(np.isfinite(integration_values)):
            failures.append(
                f"{baseline.path}: integration_error_by_field must be finite "
                "for every resolution and field"
            )

    paths = _json_scalar(baseline.arrays, "short_leg_diagnostics_paths_json", [])
    fallback_classes = _json_scalar(
        baseline.arrays, "short_leg_classification_json", []
    )
    if not isinstance(paths, list) or len(paths) != len(expected_resolutions):
        failures.append(
            f"{baseline.path}: short-leg diagnostic paths must contain one entry "
            "for each of 32/48/64"
        )
        paths = []
    if not isinstance(fallback_classes, list):
        fallback_classes = []

    diagnostic_records: list[dict[str, Any]] = []
    positive_growth_count = 0
    required_series = (
        "times",
        "high_mode_rms",
        "maximum_poloidal_jump",
    )
    for index, resolution in enumerate(expected_resolutions):
        raw_path = paths[index] if index < len(paths) else None
        resolved = _resolve_diagnostic_path(baseline, raw_path)
        record: dict[str, Any] = {
            "resolution": int(resolution),
            "path": str(resolved) if resolved is not None else None,
        }
        if resolved is None:
            failures.append(
                f"{baseline.path}: unresolved short-leg diagnostic for N={resolution}: "
                f"{raw_path!r}"
            )
            diagnostic_records.append(record)
            continue
        try:
            with np.load(resolved, allow_pickle=False) as payload:
                diagnostic = {key: np.asarray(payload[key]) for key in payload.files}
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"{resolved}: unreadable short-leg diagnostic ({error})")
            diagnostic_records.append(record)
            continue

        missing_series = [key for key in required_series if key not in diagnostic]
        if missing_series:
            failures.append(f"{resolved}: missing short-leg series {missing_series}")
        times = np.asarray(diagnostic.get("times", ()), dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            failures.append(f"{resolved}: times must be a nonempty finite 1-D series")
        for key in required_series[1:]:
            try:
                values = np.asarray(diagnostic.get(key, ()), dtype=np.float64)
            except (TypeError, ValueError):
                values = np.empty((0,), dtype=np.float64)
            if (
                values.ndim < 1
                or values.shape[0] != times.size
                or not np.all(np.isfinite(values))
            ):
                failures.append(
                    f"{resolved}: {key} must be finite and align with times"
                )

        row_fallback = (
            fallback_classes[index]
            if index < len(fallback_classes)
            else []
        )
        classifications = _json_scalar(
            diagnostic, "classification_json", row_fallback
        )
        if not isinstance(classifications, list):
            classifications = []
        unknown = [
            str(value)
            for value in classifications
            if value not in KNOWN_SHORT_LEG_CLASSIFICATIONS
        ]
        if len(classifications) != len(baseline.fields):
            failures.append(
                f"{resolved}: expected {len(baseline.fields)} short-leg "
                f"classifications, got {len(classifications)}"
            )
        if unknown:
            failures.append(f"{resolved}: unknown short-leg classifications {unknown}")
        positive_growth_count += sum(
            value == "positive-growth" for value in classifications
        )
        record.update(
            sample_count=int(times.size),
            classifications=[str(value) for value in classifications],
            unknown_classifications=unknown,
        )
        diagnostic_records.append(record)

    status = "fail" if failures else ("warning" if positive_growth_count else "pass")
    return _status(
        "evolved_prerequisite_gate",
        status,
        required_resolutions=list(expected_resolutions),
        expected_baseline={
            "start_time": EVOLVED_BASELINE_START_TIME,
            "final_time": EVOLVED_BASELINE_FINAL_TIME,
            "actual_timestep": EVOLVED_BASELINE_TIMESTEP,
            "num_steps": EVOLVED_BASELINE_NUM_STEPS,
        },
        actual_baseline=actual_scalars,
        frozen_spatial_gate=frozen_spatial,
        diagnostic_records=diagnostic_records,
        positive_growth_count=int(positive_growth_count),
        interpretation=(
            "localized positive growth remains valid evidence for temporal investigation"
            if positive_growth_count
            else "baseline diagnostics are structurally complete"
        ),
        failures=failures,
    )


def analyze(
    paths: Sequence[str | Path],
    *,
    source_tolerance: float = 1.0e-10,
    require_spatial: bool = False,
    require_evolved: bool = False,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Merge and analyze aggregate MMS artifacts.

    ``ok`` reflects hard reproducibility/numerical-integrity gates.  A
    ``warning`` (for example, a positive short-leg growth classification or
    an unavailable temporal refinement) remains visible without pretending
    that a missing campaign is a failed numerical result.
    """

    artifacts = [load_artifact(path) for path in paths]
    rows = [row for artifact in artifacts for row in _resolution_rows(artifact)]
    checks = [
        _configuration_check(artifacts),
        _physical_parameters_check(artifacts),
        _finite_norm_check(artifacts, rows),
        _source_pairing_check(artifacts, source_tolerance),
    ]
    spatial = _spatial_report(rows)
    if require_spatial:
        checks.append(_spatial_gate(rows, artifacts))
    if require_evolved:
        checks.append(_evolved_prerequisite_gate(artifacts))
    checks.append(_phi_report(rows))
    checks.append(_owner_representation_report(rows))
    checks.append(_continuum_total_error_report(rows))
    checks.append(_complete_stage7_report(artifacts, required=require_complete))
    checks.append(_short_leg_report(artifacts))
    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warning"]
    optional_names = {"complete_stage7_acceptance", "short_leg_growth_classification"}
    incomplete = [
        check for check in checks
        if check["name"] in optional_names and check["status"] == "unavailable"
    ]
    return {
        "schema": "hsx-rlp-stage7-mms-analysis-v1",
        "ok": not failures,
        "artifacts": [
            {
                "path": str(artifact.path),
                "resolutions": np.asarray(artifact.arrays.get("resolutions", ())).reshape(-1).astype(int).tolist(),
                "dt": artifact.dt,
                "actual_timestep_values": list(artifact.dt_values) if artifact.dt_values is not None else None,
                "initial_time": artifact.initial_time,
                "start_time_values": list(artifact.initial_time_values) if artifact.initial_time_values is not None else None,
                "final_time": artifact.final_time,
                "final_time_values": list(artifact.final_time_values) if artifact.final_time_values is not None else None,
                "fields": list(artifact.fields),
                "regions": list(artifact.regions),
                "rhs_terms": list(artifact.rhs_terms),
            }
            for artifact in artifacts
        ],
        "merged_resolutions": sorted({int(row["resolution"]) for row in rows}),
        "spatial_gate_required": bool(require_spatial),
        "evolved_gate_required": bool(require_evolved),
        "complete_gate_required": bool(require_complete),
        "spatial": spatial,
        "checks": checks,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "incomplete_count": len(incomplete),
        "incomplete_checks": [check["name"] for check in incomplete],
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"HSX/RLP Stage-7 MMS analysis: {'PASS' if report['ok'] else 'FAIL'}")
    print(f"resolutions: {report['merged_resolutions']}")
    for check in report["checks"]:
        print(f"  {check['status'].upper():9s} {check['name']}")
        if check["name"] == "independent_source_pairing_roundoff":
            print(f"             max={check.get('maximum')!r} tolerance={check.get('tolerance'):.3e}")
        if check["name"] == "spatial_convergence_gate":
            print(
                f"             scope={check.get('scope')} "
                f"minimum evolved-field finest-pair order="
                f"{check.get('minimum_finest_pair_l2_order')!r}"
            )
            for failure in check.get("failures", [])[:3]:
                print(f"             failure: {failure}")
        if check["name"] == "evolved_prerequisite_gate":
            print(
                f"             positive-growth classifications="
                f"{check.get('positive_growth_count', 0)}"
            )
            for failure in check.get("failures", [])[:3]:
                print(f"             failure: {failure}")
        if check["name"] == "complete_stage7_acceptance":
            for field in check.get("temporal_self_convergence_by_field", []):
                print(
                    f"             {field['field']} temporal-order="
                    f"{field.get('order')!r}"
                )
        if check["name"] == "short_leg_growth_classification":
            print(f"             positive-growth groups={check.get('positive_growth_count', 0)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path, help="aggregate MMS .npz artifacts")
    parser.add_argument("--output", type=Path, help="write the merged JSON report")
    parser.add_argument("--source-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--require-spatial",
        action="store_true",
        help=(
            "require 32/48/64 exact-phi spatial convergence (and evolved-field "
            "convergence when that ledger is present), without requiring optional "
            "temporal or short-leg campaigns"
        ),
    )
    parser.add_argument(
        "--require-evolved",
        action="store_true",
        help=(
            "require exactly two artifacts (canonical frozen spatial result, "
            "then the complete 32/48/64 dt=1e-6 evolved baseline) before a "
            "temporal-refinement campaign"
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "require the exact four Stage-7 artifacts and owner-volume-weighted "
            "N64 final-state temporal self-convergence order >= 1.8 for every field"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when optional temporal/short-leg checks are unavailable or any check is a warning",
    )
    args = parser.parse_args(argv)
    report = analyze(
        args.artifacts,
        source_tolerance=args.source_tolerance,
        require_spatial=args.require_spatial,
        require_evolved=args.require_evolved,
        require_complete=args.require_complete,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(report)
    return 2 if (
        not report["ok"]
        or (args.strict and (report["warning_count"] or report["incomplete_count"]))
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
