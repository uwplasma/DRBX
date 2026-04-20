from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..parity.arrays import load_portable_array_payload


@dataclass(frozen=True)
class Blob2DAnalysisResult:
    density_variable: str
    background_density: float
    time_points: np.ndarray
    peak_excess_history: np.ndarray
    center_of_mass_x_history: np.ndarray
    center_of_mass_z_history: np.ndarray


@dataclass(frozen=True)
class Blob2DParityResult:
    expected: Blob2DAnalysisResult
    actual: Blob2DAnalysisResult
    peak_max_abs_error: float
    peak_rms_error: float
    center_of_mass_x_max_abs_error: float
    center_of_mass_z_max_abs_error: float


def load_blob2d_analysis_json(path: str | Path) -> Blob2DAnalysisResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Blob2DAnalysisResult(
        density_variable=str(payload["density_variable"]),
        background_density=float(payload["background_density"]),
        time_points=np.asarray(payload["time_points"], dtype=np.float64),
        peak_excess_history=np.asarray(payload["peak_excess_history"], dtype=np.float64),
        center_of_mass_x_history=np.asarray(payload["center_of_mass_x_history"], dtype=np.float64),
        center_of_mass_z_history=np.asarray(payload["center_of_mass_z_history"], dtype=np.float64),
    )


def analyze_blob2d_npz(
    arrays_npz: str | Path,
    *,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Blob2DAnalysisResult:
    payload = load_portable_array_payload(arrays_npz)
    return analyze_blob2d_array_payload(
        payload,
        density_variable=density_variable,
        background_density=background_density,
    )


def analyze_blob2d_array_payload(
    payload: Mapping[str, Any],
    *,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Blob2DAnalysisResult:
    variables = payload.get("variables", {})
    if density_variable not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing density variable {density_variable!r}. Available variables: {available}")

    density_history = np.asarray(variables[density_variable], dtype=np.float64)
    if density_history.ndim != 4:
        raise ValueError("Blob2D analysis expects a (t, x, y, z) density history.")
    if density_history.shape[2] != 1:
        raise ValueError("Blob2D analysis currently expects a single active y plane.")

    excess_history = np.maximum(density_history[:, :, 0, :] - background_density, 0.0)
    time_points = np.asarray(payload.get("time_points", []), dtype=np.float64)
    if time_points.size != density_history.shape[0]:
        raise ValueError("Blob2D analysis requires one time point per stored output.")

    peak_excess_history = np.max(excess_history, axis=(1, 2))
    x_coordinates = np.arange(excess_history.shape[1], dtype=np.float64)
    z_coordinates = np.arange(excess_history.shape[2], dtype=np.float64)
    mass_history = np.sum(excess_history, axis=(1, 2))
    safe_mass = np.maximum(mass_history, np.finfo(np.float64).tiny)
    center_of_mass_x_history = np.sum(excess_history * x_coordinates[None, :, None], axis=(1, 2)) / safe_mass
    center_of_mass_z_history = np.sum(excess_history * z_coordinates[None, None, :], axis=(1, 2)) / safe_mass

    return Blob2DAnalysisResult(
        density_variable=density_variable,
        background_density=background_density,
        time_points=time_points,
        peak_excess_history=np.asarray(peak_excess_history, dtype=np.float64),
        center_of_mass_x_history=np.asarray(center_of_mass_x_history, dtype=np.float64),
        center_of_mass_z_history=np.asarray(center_of_mass_z_history, dtype=np.float64),
    )


def compare_blob2d_npz(
    expected_npz: str | Path,
    actual_npz: str | Path,
    *,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Blob2DParityResult:
    return compare_blob2d_artifacts(
        expected_npz,
        actual_npz,
        density_variable=density_variable,
        background_density=background_density,
    )


def compare_blob2d_artifacts(
    expected_artifact: str | Path,
    actual_artifact: str | Path,
    *,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Blob2DParityResult:
    expected = _load_blob2d_analysis_artifact(
        expected_artifact,
        density_variable=density_variable,
        background_density=background_density,
    )
    actual = _load_blob2d_analysis_artifact(
        actual_artifact,
        density_variable=density_variable,
        background_density=background_density,
    )
    return compare_blob2d_analysis_results(expected, actual)


def compare_blob2d_array_payloads(
    expected_payload: Mapping[str, Any],
    actual_payload: Mapping[str, Any],
    *,
    density_variable: str = "Ne",
    background_density: float = 1.0,
) -> Blob2DParityResult:
    expected = analyze_blob2d_array_payload(
        expected_payload,
        density_variable=density_variable,
        background_density=background_density,
    )
    actual = analyze_blob2d_array_payload(
        actual_payload,
        density_variable=density_variable,
        background_density=background_density,
    )
    return compare_blob2d_analysis_results(expected, actual)


def compare_blob2d_analysis_results(
    expected: Blob2DAnalysisResult,
    actual: Blob2DAnalysisResult,
) -> Blob2DParityResult:
    _require_matching_time_points(expected, actual)

    peak_diff = actual.peak_excess_history - expected.peak_excess_history
    com_x_diff = actual.center_of_mass_x_history - expected.center_of_mass_x_history
    com_z_diff = actual.center_of_mass_z_history - expected.center_of_mass_z_history
    return Blob2DParityResult(
        expected=expected,
        actual=actual,
        peak_max_abs_error=float(np.max(np.abs(peak_diff))),
        peak_rms_error=float(np.sqrt(np.mean(np.square(peak_diff)))),
        center_of_mass_x_max_abs_error=float(np.max(np.abs(com_x_diff))),
        center_of_mass_z_max_abs_error=float(np.max(np.abs(com_z_diff))),
    )


def write_blob2d_analysis_json(result: Blob2DAnalysisResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_serialize_analysis(result), indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_blob2d_parity_json(result: Blob2DParityResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected": _serialize_analysis(result.expected),
        "actual": _serialize_analysis(result.actual),
        "peak_max_abs_error": result.peak_max_abs_error,
        "peak_rms_error": result.peak_rms_error,
        "center_of_mass_x_max_abs_error": result.center_of_mass_x_max_abs_error,
        "center_of_mass_z_max_abs_error": result.center_of_mass_z_max_abs_error,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def save_blob2d_parity_plot(result: Blob2DParityResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    time_points = result.expected.time_points

    axes[0].plot(time_points, result.expected.peak_excess_history, label="Reference", linewidth=2.0)
    axes[0].plot(time_points, result.actual.peak_excess_history, label="Native", linewidth=1.8, linestyle="--")
    axes[0].set_ylabel("Peak excess")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_points, result.expected.center_of_mass_x_history, linewidth=2.0)
    axes[1].plot(time_points, result.actual.center_of_mass_x_history, linewidth=1.8, linestyle="--")
    axes[1].set_ylabel("COM x")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_points, result.expected.center_of_mass_z_history, linewidth=2.0)
    axes[2].plot(time_points, result.actual.center_of_mass_z_history, linewidth=1.8, linestyle="--")
    axes[2].set_ylabel("COM z")
    axes[2].set_xlabel("Output time")
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _require_matching_time_points(expected: Blob2DAnalysisResult, actual: Blob2DAnalysisResult) -> None:
    if expected.time_points.shape != actual.time_points.shape or not np.allclose(
        expected.time_points,
        actual.time_points,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Blob2D parity comparison requires matching output times.")


def _load_blob2d_analysis_artifact(
    artifact: str | Path,
    *,
    density_variable: str,
    background_density: float,
) -> Blob2DAnalysisResult:
    path = Path(artifact)
    if path.suffix.lower() == ".json":
        return load_blob2d_analysis_json(path)
    return analyze_blob2d_npz(
        path,
        density_variable=density_variable,
        background_density=background_density,
    )


def _serialize_analysis(result: Blob2DAnalysisResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["time_points"] = result.time_points.tolist()
    payload["peak_excess_history"] = result.peak_excess_history.tolist()
    payload["center_of_mass_x_history"] = result.center_of_mass_x_history.tolist()
    payload["center_of_mass_z_history"] = result.center_of_mass_z_history.tolist()
    return payload
