from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver, load_bout_input
from ..config.normalization import ELEMENTARY_CHARGE, PROTON_MASS
from ..native.units import resolved_dataset_scalars
from ..parity.arrays import load_portable_array_payload
from ..runtime.run_config import RunConfiguration

VACUUM_PERMEABILITY = 4.0e-7 * np.pi


@dataclass(frozen=True)
class AlfvenWaveBenchmarkScalars:
    kpar: float
    kperp: float
    alfven_speed: float
    electron_skin_depth: float
    analytic_phase_speed: float
    analytic_omega: float


@dataclass(frozen=True)
class AlfvenWaveAnalysisResult:
    field_variable: str
    x_index: int
    benchmark: AlfvenWaveBenchmarkScalars
    measured_omega: float
    measured_phase_speed: float
    relative_phase_speed_error: float
    time_seconds: np.ndarray
    mean_square_history: np.ndarray


@dataclass(frozen=True)
class AlfvenWaveParityResult:
    expected: AlfvenWaveAnalysisResult
    actual: AlfvenWaveAnalysisResult
    phase_speed_error: float
    omega_error: float
    mean_square_max_abs_error: float
    mean_square_rms_error: float


def analyze_alfven_wave_npz(
    arrays_npz: str | Path,
    *,
    input_file: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
) -> AlfvenWaveAnalysisResult:
    payload = load_portable_array_payload(arrays_npz)
    config = load_bout_input(input_file)
    run_config = RunConfiguration.from_config(config)
    return analyze_alfven_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=resolved_dataset_scalars(run_config),
        field_variable=field_variable,
        x_index=x_index,
    )


def analyze_alfven_wave_array_payload(
    payload: Mapping[str, Any],
    *,
    config: BoutConfig,
    dataset_scalars: Mapping[str, float],
    field_variable: str = "phi",
    x_index: int = 2,
) -> AlfvenWaveAnalysisResult:
    variables = payload.get("variables", {})
    if field_variable not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing field variable {field_variable!r}. Available variables: {available}")

    field_history = np.asarray(variables[field_variable], dtype=np.float64)
    if field_history.ndim != 4:
        raise ValueError(f"{field_variable} must have shape (t, x, y, z).")
    if not 0 <= x_index < field_history.shape[1]:
        raise IndexError(f"x-index {x_index} is out of bounds for shape {field_history.shape}.")

    time_seconds = np.asarray(payload.get("time_points", []), dtype=np.float64) / float(dataset_scalars["Omega_ci"])
    if time_seconds.size < 4:
        raise ValueError("Alfven-wave analysis requires at least four output times.")

    plane_history = field_history[:, x_index, :, :]
    mean_square_history = np.mean(np.square(plane_history), axis=(1, 2))
    measured_omega = _measure_alfven_frequency(mean_square_history, time_seconds)
    benchmark = compute_alfven_wave_benchmark_scalars(config, dataset_scalars=dataset_scalars)
    measured_phase_speed = measured_omega / benchmark.kpar
    return AlfvenWaveAnalysisResult(
        field_variable=field_variable,
        x_index=x_index,
        benchmark=benchmark,
        measured_omega=measured_omega,
        measured_phase_speed=measured_phase_speed,
        relative_phase_speed_error=abs(measured_phase_speed - benchmark.analytic_phase_speed)
        / benchmark.analytic_phase_speed,
        time_seconds=time_seconds,
        mean_square_history=mean_square_history,
    )


def compare_alfven_wave_npz(
    expected_npz: str | Path,
    actual_npz: str | Path,
    *,
    input_file: str | Path,
    field_variable: str = "phi",
    x_index: int = 2,
) -> AlfvenWaveParityResult:
    expected_payload = load_portable_array_payload(expected_npz)
    actual_payload = load_portable_array_payload(actual_npz)
    config = load_bout_input(input_file)
    run_config = RunConfiguration.from_config(config)
    return compare_alfven_wave_array_payloads(
        expected_payload,
        actual_payload,
        config=config,
        dataset_scalars=resolved_dataset_scalars(run_config),
        field_variable=field_variable,
        x_index=x_index,
    )


def compare_alfven_wave_array_payloads(
    expected_payload: Mapping[str, Any],
    actual_payload: Mapping[str, Any],
    *,
    config: BoutConfig,
    dataset_scalars: Mapping[str, float],
    field_variable: str = "phi",
    x_index: int = 2,
) -> AlfvenWaveParityResult:
    expected = analyze_alfven_wave_array_payload(
        expected_payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable=field_variable,
        x_index=x_index,
    )
    actual = analyze_alfven_wave_array_payload(
        actual_payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable=field_variable,
        x_index=x_index,
    )
    diff = actual.mean_square_history - expected.mean_square_history
    return AlfvenWaveParityResult(
        expected=expected,
        actual=actual,
        phase_speed_error=actual.measured_phase_speed - expected.measured_phase_speed,
        omega_error=actual.measured_omega - expected.measured_omega,
        mean_square_max_abs_error=float(np.max(np.abs(diff))),
        mean_square_rms_error=float(np.sqrt(np.mean(np.square(diff)))),
    )


def compute_alfven_wave_benchmark_scalars(
    config: BoutConfig,
    *,
    dataset_scalars: Mapping[str, float],
) -> AlfvenWaveBenchmarkScalars:
    resolver = NumericResolver(config)
    magnetic_field = resolver.resolve("mesh", "B")
    density_si = resolver.resolve("i", "density")
    ion_mass = resolver.resolve("i", "AA") * PROTON_MASS
    electron_mass = resolver.resolve("e", "AA") * PROTON_MASS
    kpar = (2.0 * np.pi) / resolver.resolve("mesh", "Ly")
    kperp = (2.0 * np.pi) / resolver.resolve("mesh", "Lz")
    alfven_speed = magnetic_field / np.sqrt(VACUUM_PERMEABILITY * density_si * ion_mass)
    electron_skin_depth = np.sqrt(
        electron_mass / (VACUUM_PERMEABILITY * density_si * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE)
    )
    analytic_phase_speed = alfven_speed / np.sqrt(1.0 + (kperp * electron_skin_depth) ** 2)
    analytic_omega = kpar * analytic_phase_speed
    return AlfvenWaveBenchmarkScalars(
        kpar=kpar,
        kperp=kperp,
        alfven_speed=alfven_speed,
        electron_skin_depth=electron_skin_depth,
        analytic_phase_speed=analytic_phase_speed,
        analytic_omega=analytic_omega,
    )


def write_alfven_wave_analysis_json(result: AlfvenWaveAnalysisResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["time_seconds"] = result.time_seconds.tolist()
    payload["mean_square_history"] = result.mean_square_history.tolist()
    payload["benchmark"] = asdict(result.benchmark)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_alfven_wave_parity_json(result: AlfvenWaveParityResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected": _analysis_to_dict(result.expected),
        "actual": _analysis_to_dict(result.actual),
        "phase_speed_error": result.phase_speed_error,
        "omega_error": result.omega_error,
        "mean_square_max_abs_error": result.mean_square_max_abs_error,
        "mean_square_rms_error": result.mean_square_rms_error,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def save_alfven_wave_diagnostic_plot(result: AlfvenWaveAnalysisResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.time_seconds * 1.0e6, result.mean_square_history, linewidth=2.0)
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel(f"Mean square {result.field_variable}")
    ax.set_title(
        "Alfven-wave benchmark\n"
        f"measured vp={result.measured_phase_speed:.3e} m/s, analytic vp={result.benchmark.analytic_phase_speed:.3e} m/s"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_alfven_wave_parity_plot(result: AlfvenWaveParityResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        result.expected.time_seconds * 1.0e6,
        result.expected.mean_square_history,
        linewidth=2.0,
        label="expected",
    )
    ax.plot(
        result.actual.time_seconds * 1.0e6,
        result.actual.mean_square_history,
        linewidth=2.0,
        linestyle="--",
        label="actual",
    )
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel(f"Mean square {result.expected.field_variable}")
    ax.set_title(
        "Alfven-wave parity\n"
        f"|domega|={abs(result.omega_error):.3e}, |dvp|={abs(result.phase_speed_error):.3e}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _measure_alfven_frequency(mean_square_history: np.ndarray, time_seconds: np.ndarray) -> float:
    if time_seconds.size < 4:
        raise ValueError("Need at least four time points to estimate Alfven-wave frequency.")
    dt = float(np.mean(np.diff(time_seconds)))
    derivative = np.gradient(mean_square_history[1:])
    if derivative.size < 3:
        raise ValueError("Need at least three derivative samples to estimate Alfven-wave frequency.")
    crossings_mask = derivative[1:] * derivative[:-1] < 0.0
    indices = np.where(crossings_mask)[0]
    if indices.size < 2:
        raise ValueError("Need at least two zero crossings in d/dt(field^2) to estimate the Alfven-wave period.")
    crossings = (
        indices * np.abs(derivative[indices + 1]) + (indices + 1) * np.abs(derivative[indices])
    ) / np.abs(derivative[indices + 1] - derivative[indices])
    period = 4.0 * float(np.mean(np.diff(crossings))) * dt
    return 2.0 * np.pi / period


def _analysis_to_dict(result: AlfvenWaveAnalysisResult) -> dict[str, Any]:
    return {
        "field_variable": result.field_variable,
        "x_index": result.x_index,
        "benchmark": asdict(result.benchmark),
        "measured_omega": result.measured_omega,
        "measured_phase_speed": result.measured_phase_speed,
        "relative_phase_speed_error": result.relative_phase_speed_error,
        "time_seconds": result.time_seconds.tolist(),
        "mean_square_history": result.mean_square_history.tolist(),
    }
