from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset

from .compare import ComparisonIssue, ComparisonResult


def build_portable_array_payload(
    *,
    case_name: str,
    parity_mode: str,
    capability_tier: str,
    compare_variables: tuple[str, ...],
    component_labels: tuple[str, ...],
    dimensions: Mapping[str, int],
    time_points: tuple[float, ...],
    dataset_scalars: Mapping[str, float],
    variables: Mapping[str, Any],
    variable_dimensions: Mapping[str, tuple[str, ...]] | None = None,
    overrides: tuple[str, ...] = (),
    configured_nout: int | None = None,
    configured_timestep: float | None = None,
    producer: str = "jax-drb",
) -> dict[str, Any]:
    summary_dimensions = tuple(dimensions)
    payload_variables: dict[str, np.ndarray] = {}
    payload_variable_dimensions: dict[str, list[str]] = {}
    for name in compare_variables:
        if name not in variables:
            continue
        array = np.asarray(variables[name], dtype=np.float64)
        payload_variables[name] = array
        if variable_dimensions is not None and name in variable_dimensions:
            dims = variable_dimensions[name]
        elif len(summary_dimensions) == array.ndim:
            dims = summary_dimensions
        else:
            dims = tuple(["t", *[f"dim_{index}" for index in range(1, array.ndim)]])
        payload_variable_dimensions[name] = list(dims)

    payload: dict[str, Any] = {
        "case_name": case_name,
        "parity_mode": parity_mode,
        "capability_tier": capability_tier,
        "producer": producer,
        "overrides": list(overrides),
        "compare_variables": list(compare_variables),
        "component_labels": list(component_labels),
        "dimensions": dict(dimensions),
        "time_points": list(time_points),
        "dataset_scalars": {key: float(value) for key, value in dataset_scalars.items()},
        "variable_dimensions": payload_variable_dimensions,
        "variables": payload_variables,
        "effective_output_points": len(time_points),
    }
    if configured_nout is not None:
        payload["configured_nout"] = configured_nout
    if configured_timestep is not None:
        payload["configured_timestep"] = configured_timestep
    return payload


def build_dataset_array_payload(
    dataset_path: str | Path,
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...],
    component_labels: tuple[str, ...],
    overrides: tuple[str, ...] = (),
    trim_x_guards: bool = False,
    x_guards: int = 0,
    trim_y_guards: bool = False,
    y_guards: int = 0,
    configured_nout: int | None = None,
    configured_timestep: float | None = None,
    producer: str = "external-reference",
) -> dict[str, Any]:
    with Dataset(Path(dataset_path)) as dataset:
        dimensions = {name: len(dimension) for name, dimension in dataset.dimensions.items()}
        time_points = tuple(float(value) for value in dataset.variables["t_array"][:]) if "t_array" in dataset.variables else ()
        dataset_scalars = {
            name: float(dataset.variables[name][...].item())
            for name in ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
            if name in dataset.variables
        }
        variables = {
            name: _maybe_trim_guards(
                np.asarray(dataset.variables[name][:], dtype=np.float64),
                dimensions=tuple(dataset.variables[name].dimensions),
                trim_x_guards=trim_x_guards,
                x_guards=x_guards,
                trim_y_guards=trim_y_guards,
                y_guards=y_guards,
            )
            for name in compare_variables
            if name in dataset.variables
        }
        variable_dimensions = {
            name: tuple(dataset.variables[name].dimensions)
            for name in compare_variables
            if name in dataset.variables
        }
    return build_portable_array_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        capability_tier="scaffolded_reference_backed",
        compare_variables=compare_variables,
        component_labels=component_labels,
        dimensions=dimensions,
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        variables=variables,
        variable_dimensions=variable_dimensions,
        overrides=overrides,
        configured_nout=configured_nout,
        configured_timestep=configured_timestep,
        producer=producer,
    )


def build_array_payload_from_summary_payload(
    summary_payload: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    variable_dimensions: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    return build_portable_array_payload(
        case_name=str(summary_payload["case_name"]),
        parity_mode=str(summary_payload["parity_mode"]),
        capability_tier=str(summary_payload.get("capability_tier", "native_exact")),
        compare_variables=tuple(summary_payload.get("compare_variables", [])),
        component_labels=tuple(summary_payload.get("component_labels", [])),
        dimensions=summary_payload.get("dimensions", {}),
        time_points=tuple(float(value) for value in summary_payload.get("time_points", [])),
        dataset_scalars=summary_payload.get("dataset_scalars", {}),
        variables=variables,
        variable_dimensions=variable_dimensions,
        overrides=tuple(summary_payload.get("overrides", [])),
        configured_nout=summary_payload.get("configured_nout"),
        configured_timestep=summary_payload.get("configured_timestep"),
        producer=str(summary_payload.get("producer", "jax-drb")),
    )


def write_portable_array_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in payload.items() if key != "variables"}
    arrays = {
        f"var__{name}": np.asarray(value, dtype=np.float64)
        for name, value in payload.get("variables", {}).items()
    }
    np.savez_compressed(target, __metadata__=json.dumps(metadata, sort_keys=True), **arrays)
    return target


def load_portable_array_payload(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata__"]))
        variables = {
            key.removeprefix("var__"): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key.startswith("var__")
        }
    metadata["variables"] = variables
    return metadata


def compare_array_payloads(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    scalar_rtol: float = 1e-10,
    scalar_atol: float = 1e-12,
    array_rtol: float = 1e-10,
    array_atol: float = 1e-12,
    time_rtol: float = 1e-12,
    time_atol: float = 1e-12,
) -> ComparisonResult:
    issues: list[ComparisonIssue] = []

    for field in ("case_name", "parity_mode", "compare_variables", "component_labels", "dimensions", "variable_dimensions"):
        if actual.get(field) != expected.get(field):
            issues.append(
                ComparisonIssue(field=field, message=f"expected {expected.get(field)!r}, got {actual.get(field)!r}")
            )
    if "capability_tier" in expected and "capability_tier" in actual:
        if actual.get("capability_tier") != expected.get("capability_tier"):
            issues.append(
                ComparisonIssue(
                    field="capability_tier",
                    message=f"expected {expected.get('capability_tier')!r}, got {actual.get('capability_tier')!r}",
                )
            )

    _compare_float_sequences(
        issues,
        field="time_points",
        expected=expected.get("time_points", []),
        actual=actual.get("time_points", []),
        rtol=time_rtol,
        atol=time_atol,
    )
    _compare_float_mapping(
        issues,
        field="dataset_scalars",
        expected=expected.get("dataset_scalars", {}),
        actual=actual.get("dataset_scalars", {}),
        rtol=scalar_rtol,
        atol=scalar_atol,
    )

    expected_variables = expected.get("variables", {})
    actual_variables = actual.get("variables", {})
    for name in sorted(set(expected_variables) | set(actual_variables)):
        if name not in expected_variables:
            issues.append(ComparisonIssue(field=f"variables.{name}", message="unexpected variable"))
            continue
        if name not in actual_variables:
            issues.append(ComparisonIssue(field=f"variables.{name}", message="missing variable"))
            continue
        expected_array = np.asarray(expected_variables[name], dtype=np.float64)
        actual_array = np.asarray(actual_variables[name], dtype=np.float64)
        if expected_array.shape != actual_array.shape:
            issues.append(
                ComparisonIssue(
                    field=f"variables.{name}.shape",
                    message=f"expected {expected_array.shape!r}, got {actual_array.shape!r}",
                )
            )
            continue
        if not np.allclose(actual_array, expected_array, rtol=array_rtol, atol=array_atol):
            delta = float(np.max(np.abs(actual_array - expected_array)))
            issues.append(
                ComparisonIssue(
                    field=f"variables.{name}",
                    message=f"arrays differ (max_abs_diff={delta:.8e}, rtol={array_rtol:g}, atol={array_atol:g})",
                )
            )

    return ComparisonResult(ok=not issues, issues=tuple(issues))


def _maybe_trim_guards(
    array: np.ndarray,
    *,
    dimensions: tuple[str, ...],
    trim_x_guards: bool,
    x_guards: int,
    trim_y_guards: bool,
    y_guards: int,
) -> np.ndarray:
    result = array
    if trim_x_guards and x_guards > 0 and "x" in dimensions:
        axis = dimensions.index("x")
        if result.shape[axis] > 2 * x_guards:
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(x_guards, -x_guards)
            result = result[tuple(slicer)]
    if trim_y_guards and y_guards > 0 and "y" in dimensions:
        axis = dimensions.index("y")
        if result.shape[axis] > 2 * y_guards:
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(y_guards, -y_guards)
            result = result[tuple(slicer)]
    return result


def _compare_float_sequences(
    issues: list[ComparisonIssue],
    *,
    field: str,
    expected: list[Any],
    actual: list[Any],
    rtol: float,
    atol: float,
) -> None:
    if len(expected) != len(actual):
        issues.append(ComparisonIssue(field=field, message=f"expected length {len(expected)}, got {len(actual)}"))
        return
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
        if not math.isclose(float(actual_value), float(expected_value), rel_tol=rtol, abs_tol=atol):
            issues.append(
                ComparisonIssue(
                    field=f"{field}[{index}]",
                    message=f"expected {expected_value!r}, got {actual_value!r}",
                )
            )


def _compare_float_mapping(
    issues: list[ComparisonIssue],
    *,
    field: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    rtol: float,
    atol: float,
) -> None:
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            issues.append(ComparisonIssue(field=f"{field}.{key}", message="unexpected field"))
            continue
        if key not in actual:
            issues.append(ComparisonIssue(field=f"{field}.{key}", message="missing field"))
            continue
        if not math.isclose(float(actual[key]), float(expected[key]), rel_tol=rtol, abs_tol=atol):
            issues.append(
                ComparisonIssue(
                    field=f"{field}.{key}",
                    message=f"expected {expected[key]!r}, got {actual[key]!r}",
                )
            )
