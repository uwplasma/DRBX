from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..parity.arrays import load_portable_array_payload


@dataclass(frozen=True)
class NeutralMixedAnalysisResult:
    density_variable: str
    pressure_variable: str
    momentum_variable: str
    center_index_x: int
    center_index_y: int
    center_index_z: int
    time_points: np.ndarray
    center_density_history: np.ndarray
    center_pressure_history: np.ndarray
    center_momentum_history: np.ndarray
    center_temperature_history: np.ndarray
    total_density_history: np.ndarray
    total_pressure_history: np.ndarray
    momentum_rms_history: np.ndarray


@dataclass(frozen=True)
class NeutralMixedSeriesError:
    name: str
    max_abs_error: float
    rms_error: float
    error_history: np.ndarray


@dataclass(frozen=True)
class NeutralMixedParityResult:
    expected: NeutralMixedAnalysisResult
    actual: NeutralMixedAnalysisResult
    series_errors: dict[str, NeutralMixedSeriesError]


def load_neutral_mixed_analysis_json(path: str | Path) -> NeutralMixedAnalysisResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NeutralMixedAnalysisResult(
        density_variable=str(payload["density_variable"]),
        pressure_variable=str(payload["pressure_variable"]),
        momentum_variable=str(payload["momentum_variable"]),
        center_index_x=int(payload["center_index_x"]),
        center_index_y=int(payload["center_index_y"]),
        center_index_z=int(payload["center_index_z"]),
        time_points=np.asarray(payload["time_points"], dtype=np.float64),
        center_density_history=np.asarray(payload["center_density_history"], dtype=np.float64),
        center_pressure_history=np.asarray(payload["center_pressure_history"], dtype=np.float64),
        center_momentum_history=np.asarray(payload["center_momentum_history"], dtype=np.float64),
        center_temperature_history=np.asarray(payload["center_temperature_history"], dtype=np.float64),
        total_density_history=np.asarray(payload["total_density_history"], dtype=np.float64),
        total_pressure_history=np.asarray(payload["total_pressure_history"], dtype=np.float64),
        momentum_rms_history=np.asarray(payload["momentum_rms_history"], dtype=np.float64),
    )


def analyze_neutral_mixed_npz(
    arrays_npz: str | Path,
    *,
    density_variable: str = "Nh",
    pressure_variable: str = "Ph",
    momentum_variable: str = "NVh",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
) -> NeutralMixedAnalysisResult:
    payload = load_portable_array_payload(arrays_npz)
    return analyze_neutral_mixed_array_payload(
        payload,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )


def analyze_neutral_mixed_array_payload(
    payload: Mapping[str, Any],
    *,
    density_variable: str = "Nh",
    pressure_variable: str = "Ph",
    momentum_variable: str = "NVh",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
) -> NeutralMixedAnalysisResult:
    variables = payload.get("variables", {})
    density = _require_history(variables, density_variable)
    pressure = _require_history(variables, pressure_variable)
    momentum = _require_history(variables, momentum_variable)
    if pressure.shape != density.shape or momentum.shape != density.shape:
        raise ValueError("Neutral mixed analysis requires matching density, pressure, and momentum shapes.")

    nt, nx, ny, nz = density.shape
    center_index_x = _resolve_index(x_index, size=nx, label="x")
    center_index_y = _resolve_index(y_index, size=ny, label="y")
    center_index_z = _resolve_index(z_index, size=nz, label="z")
    time_points = np.asarray(payload.get("time_points", []), dtype=np.float64)
    if time_points.size != nt:
        raise ValueError("Neutral mixed analysis requires one time point per stored output.")

    center_density_history = density[:, center_index_x, center_index_y, center_index_z]
    center_pressure_history = pressure[:, center_index_x, center_index_y, center_index_z]
    center_momentum_history = momentum[:, center_index_x, center_index_y, center_index_z]
    center_temperature_history = center_pressure_history / np.maximum(center_density_history, np.finfo(np.float64).tiny)

    axes = (1, 2, 3)
    total_density_history = np.sum(density, axis=axes)
    total_pressure_history = np.sum(pressure, axis=axes)
    momentum_rms_history = np.sqrt(np.mean(np.square(momentum), axis=axes))

    return NeutralMixedAnalysisResult(
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        center_index_x=center_index_x,
        center_index_y=center_index_y,
        center_index_z=center_index_z,
        time_points=time_points,
        center_density_history=np.asarray(center_density_history, dtype=np.float64),
        center_pressure_history=np.asarray(center_pressure_history, dtype=np.float64),
        center_momentum_history=np.asarray(center_momentum_history, dtype=np.float64),
        center_temperature_history=np.asarray(center_temperature_history, dtype=np.float64),
        total_density_history=np.asarray(total_density_history, dtype=np.float64),
        total_pressure_history=np.asarray(total_pressure_history, dtype=np.float64),
        momentum_rms_history=np.asarray(momentum_rms_history, dtype=np.float64),
    )


def compare_neutral_mixed_npz(
    expected_npz: str | Path,
    actual_npz: str | Path,
    *,
    density_variable: str = "Nh",
    pressure_variable: str = "Ph",
    momentum_variable: str = "NVh",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
) -> NeutralMixedParityResult:
    return compare_neutral_mixed_artifacts(
        expected_npz,
        actual_npz,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )


def compare_neutral_mixed_artifacts(
    expected_artifact: str | Path,
    actual_artifact: str | Path,
    *,
    density_variable: str = "Nh",
    pressure_variable: str = "Ph",
    momentum_variable: str = "NVh",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
) -> NeutralMixedParityResult:
    expected = _load_neutral_mixed_analysis_artifact(
        expected_artifact,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )
    actual = _load_neutral_mixed_analysis_artifact(
        actual_artifact,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )
    return compare_neutral_mixed_analysis_results(expected, actual)


def compare_neutral_mixed_array_payloads(
    expected_payload: Mapping[str, Any],
    actual_payload: Mapping[str, Any],
    *,
    density_variable: str = "Nh",
    pressure_variable: str = "Ph",
    momentum_variable: str = "NVh",
    x_index: int | None = None,
    y_index: int | None = None,
    z_index: int | None = None,
) -> NeutralMixedParityResult:
    expected = analyze_neutral_mixed_array_payload(
        expected_payload,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )
    actual = analyze_neutral_mixed_array_payload(
        actual_payload,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )
    return compare_neutral_mixed_analysis_results(expected, actual)


def compare_neutral_mixed_analysis_results(
    expected: NeutralMixedAnalysisResult,
    actual: NeutralMixedAnalysisResult,
) -> NeutralMixedParityResult:
    _require_matching_layout(expected, actual)
    series_errors = {
        "center_density": _build_series_error(
            "center_density",
            expected.center_density_history,
            actual.center_density_history,
        ),
        "center_pressure": _build_series_error(
            "center_pressure",
            expected.center_pressure_history,
            actual.center_pressure_history,
        ),
        "center_momentum": _build_series_error(
            "center_momentum",
            expected.center_momentum_history,
            actual.center_momentum_history,
        ),
        "center_temperature": _build_series_error(
            "center_temperature",
            expected.center_temperature_history,
            actual.center_temperature_history,
        ),
        "total_density": _build_series_error(
            "total_density",
            expected.total_density_history,
            actual.total_density_history,
        ),
        "total_pressure": _build_series_error(
            "total_pressure",
            expected.total_pressure_history,
            actual.total_pressure_history,
        ),
        "momentum_rms": _build_series_error(
            "momentum_rms",
            expected.momentum_rms_history,
            actual.momentum_rms_history,
        ),
    }
    return NeutralMixedParityResult(expected=expected, actual=actual, series_errors=series_errors)


def write_neutral_mixed_analysis_json(result: NeutralMixedAnalysisResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_serialize_analysis(result), indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_neutral_mixed_parity_json(result: NeutralMixedParityResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected": _serialize_analysis(result.expected),
        "actual": _serialize_analysis(result.actual),
        "series_errors": {
            name: {
                "name": value.name,
                "max_abs_error": value.max_abs_error,
                "rms_error": value.rms_error,
                "error_history": value.error_history.tolist(),
            }
            for name, value in sorted(result.series_errors.items())
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def save_neutral_mixed_diagnostic_plot(result: NeutralMixedAnalysisResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 1, figsize=(8.5, 9.0), sharex=True)
    time_points = result.time_points

    axes[0].plot(time_points, result.center_density_history, linewidth=2.0)
    axes[0].set_ylabel(f"{result.density_variable} center")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_points, result.center_pressure_history, linewidth=2.0)
    axes[1].set_ylabel(f"{result.pressure_variable} center")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_points, result.center_momentum_history, linewidth=2.0)
    axes[2].set_ylabel(f"{result.momentum_variable} center")
    axes[2].grid(alpha=0.25)

    axes[3].plot(time_points, result.center_temperature_history, linewidth=2.0, label="center temperature")
    axes[3].plot(time_points, result.momentum_rms_history, linewidth=1.8, linestyle="--", label="momentum RMS")
    axes[3].set_ylabel("derived")
    axes[3].set_xlabel("Output time")
    axes[3].legend(frameon=False)
    axes[3].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def save_neutral_mixed_parity_plot(result: NeutralMixedParityResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 1, figsize=(8.5, 9.0), sharex=True)
    time_points = result.expected.time_points

    axes[0].plot(time_points, result.expected.center_density_history, label="Reference", linewidth=2.0)
    axes[0].plot(time_points, result.actual.center_density_history, label="Native", linewidth=1.8, linestyle="--")
    axes[0].set_ylabel(f"{result.expected.density_variable} center")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_points, result.expected.center_pressure_history, linewidth=2.0)
    axes[1].plot(time_points, result.actual.center_pressure_history, linewidth=1.8, linestyle="--")
    axes[1].set_ylabel(f"{result.expected.pressure_variable} center")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_points, result.expected.center_momentum_history, linewidth=2.0)
    axes[2].plot(time_points, result.actual.center_momentum_history, linewidth=1.8, linestyle="--")
    axes[2].set_ylabel(f"{result.expected.momentum_variable} center")
    axes[2].grid(alpha=0.25)

    axes[3].plot(time_points, result.expected.center_temperature_history, linewidth=2.0)
    axes[3].plot(time_points, result.actual.center_temperature_history, linewidth=1.8, linestyle="--")
    axes[3].set_ylabel("center T")
    axes[3].set_xlabel("Output time")
    axes[3].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _load_neutral_mixed_analysis_artifact(
    artifact: str | Path,
    *,
    density_variable: str,
    pressure_variable: str,
    momentum_variable: str,
    x_index: int | None,
    y_index: int | None,
    z_index: int | None,
) -> NeutralMixedAnalysisResult:
    path = Path(artifact)
    if path.suffix.lower() == ".json":
        return load_neutral_mixed_analysis_json(path)
    return analyze_neutral_mixed_npz(
        path,
        density_variable=density_variable,
        pressure_variable=pressure_variable,
        momentum_variable=momentum_variable,
        x_index=x_index,
        y_index=y_index,
        z_index=z_index,
    )


def _require_matching_layout(expected: NeutralMixedAnalysisResult, actual: NeutralMixedAnalysisResult) -> None:
    if (
        expected.density_variable != actual.density_variable
        or expected.pressure_variable != actual.pressure_variable
        or expected.momentum_variable != actual.momentum_variable
    ):
        raise ValueError("Neutral mixed parity comparison requires matching variable names.")
    if (
        expected.center_index_x != actual.center_index_x
        or expected.center_index_y != actual.center_index_y
        or expected.center_index_z != actual.center_index_z
    ):
        raise ValueError("Neutral mixed parity comparison requires matching probe indices.")
    if expected.time_points.shape != actual.time_points.shape or not np.allclose(
        expected.time_points,
        actual.time_points,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("Neutral mixed parity comparison requires matching output times.")


def _build_series_error(name: str, expected: np.ndarray, actual: np.ndarray) -> NeutralMixedSeriesError:
    error_history = np.asarray(actual - expected, dtype=np.float64)
    return NeutralMixedSeriesError(
        name=name,
        max_abs_error=float(np.max(np.abs(error_history))),
        rms_error=float(np.sqrt(np.mean(np.square(error_history)))),
        error_history=error_history,
    )


def _serialize_analysis(result: NeutralMixedAnalysisResult) -> dict[str, Any]:
    return {
        "density_variable": result.density_variable,
        "pressure_variable": result.pressure_variable,
        "momentum_variable": result.momentum_variable,
        "center_index_x": result.center_index_x,
        "center_index_y": result.center_index_y,
        "center_index_z": result.center_index_z,
        "time_points": result.time_points.tolist(),
        "center_density_history": result.center_density_history.tolist(),
        "center_pressure_history": result.center_pressure_history.tolist(),
        "center_momentum_history": result.center_momentum_history.tolist(),
        "center_temperature_history": result.center_temperature_history.tolist(),
        "total_density_history": result.total_density_history.tolist(),
        "total_pressure_history": result.total_pressure_history.tolist(),
        "momentum_rms_history": result.momentum_rms_history.tolist(),
    }


def _require_history(variables: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing neutral mixed variable {name!r}. Available variables: {available}")
    array = np.asarray(variables[name], dtype=np.float64)
    if array.ndim != 4:
        raise ValueError("Neutral mixed analysis expects a (t, x, y, z) history.")
    return array


def _resolve_index(index: int | None, *, size: int, label: str) -> int:
    if size <= 0:
        raise ValueError(f"Neutral mixed analysis requires a positive {label} extent.")
    resolved = size // 2 if index is None else int(index)
    if resolved < 0 or resolved >= size:
        raise IndexError(f"Neutral mixed {label}-index {resolved} is outside [0, {size}).")
    return resolved
