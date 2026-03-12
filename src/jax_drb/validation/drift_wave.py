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

VACUUM_PERMITTIVITY = 8.8541878128e-12


@dataclass(frozen=True)
class DriftWaveBenchmarkScalars:
    wstar: float
    sigmapar: float
    sigmapar_over_wstar: float
    coulomb_log: float
    nu_ei: float
    wci: float
    wce: float
    analytic_gamma_over_wstar: float
    analytic_omega_over_wstar: float
    simple_analytic_gamma_over_wstar: float
    simple_analytic_omega_over_wstar: float


@dataclass(frozen=True)
class DriftWaveAnalysisResult:
    density_variable: str
    trace_x_index: int
    trace_y_index: int
    fit_points: int
    benchmark: DriftWaveBenchmarkScalars
    measured_gamma_over_wstar: float
    measured_omega_over_wstar: float
    equilibrium_density: float
    time_seconds: np.ndarray
    time_wstar: np.ndarray
    rms_history: np.ndarray
    log_rms_history: np.ndarray
    peak_history: np.ndarray


def analyze_drift_wave_npz(
    arrays_npz: str | Path,
    *,
    input_file: str | Path,
    density_variable: str = "Ni",
    x_index: int = 0,
    y_index: int = 0,
    fit_points: int = 10,
) -> DriftWaveAnalysisResult:
    payload = load_portable_array_payload(arrays_npz)
    config = load_bout_input(input_file)
    run_config = RunConfiguration.from_config(config)
    return analyze_drift_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=resolved_dataset_scalars(run_config),
        density_variable=density_variable,
        x_index=x_index,
        y_index=y_index,
        fit_points=fit_points,
    )


def analyze_drift_wave_array_payload(
    payload: Mapping[str, Any],
    *,
    config: BoutConfig,
    dataset_scalars: Mapping[str, float],
    density_variable: str = "Ni",
    x_index: int = 0,
    y_index: int = 0,
    fit_points: int = 10,
) -> DriftWaveAnalysisResult:
    variables = payload.get("variables", {})
    if density_variable not in variables:
        available = ", ".join(sorted(variables))
        raise KeyError(f"Missing density variable {density_variable!r}. Available variables: {available}")

    density_history = np.asarray(variables[density_variable], dtype=np.float64)
    density_trace = _select_density_trace(density_history, x_index=x_index, y_index=y_index)
    time_seconds = np.asarray(payload.get("time_points", []), dtype=np.float64) / float(dataset_scalars["Omega_ci"])
    if time_seconds.size < 2:
        raise ValueError("Drift-wave analysis requires at least two output times.")
    fit_points = _resolve_fit_points(fit_points, total_points=time_seconds.size)
    benchmark = compute_drift_wave_benchmark_scalars(config, dataset_scalars=dataset_scalars)
    equilibrium_density = float(np.mean(density_trace[0]))
    perturbation = density_trace - equilibrium_density
    rms_history = np.sqrt(np.mean(np.square(perturbation), axis=-1))
    log_rms_history = np.log(np.maximum(rms_history, np.finfo(np.float64).tiny))
    peak_history = _track_peak_positions(perturbation)
    measured_gamma = _measure_growth_rate(log_rms_history, time_seconds=time_seconds, fit_points=fit_points)
    measured_omega = _measure_frequency(
        peak_history,
        time_seconds=time_seconds,
        nz=density_trace.shape[-1],
        fit_points=fit_points,
    )

    return DriftWaveAnalysisResult(
        density_variable=density_variable,
        trace_x_index=x_index,
        trace_y_index=y_index,
        fit_points=fit_points,
        benchmark=benchmark,
        measured_gamma_over_wstar=measured_gamma / benchmark.wstar,
        measured_omega_over_wstar=measured_omega / benchmark.wstar,
        equilibrium_density=equilibrium_density,
        time_seconds=time_seconds,
        time_wstar=time_seconds * benchmark.wstar,
        rms_history=rms_history,
        log_rms_history=log_rms_history,
        peak_history=peak_history,
    )


def compute_drift_wave_benchmark_scalars(
    config: BoutConfig,
    *,
    dataset_scalars: Mapping[str, float],
) -> DriftWaveBenchmarkScalars:
    resolver = NumericResolver(config)
    qe = ELEMENTARY_CHARGE
    mp = PROTON_MASS
    electron_temperature = resolver.resolve("e", "temperature")
    ion_temperature = resolver.resolve("i", "temperature")
    ion_charge = resolver.resolve("i", "charge")
    electron_density = float(dataset_scalars["Nnorm"])
    ion_density = electron_density / ion_charge
    ion_mass = resolver.resolve("i", "AA") * mp
    electron_mass = resolver.resolve("e", "AA") * mp
    k_z = (2.0 * np.pi) / resolver.resolve("mesh", "Lz")
    k_y = (2.0 * np.pi) / resolver.resolve("mesh", "Ly")
    inv_ln = resolver.resolve("mesh", "inv_Ln")
    magnetic_field = resolver.resolve("mesh", "B")

    coulomb_log = 31.0 - 0.5 * np.log(electron_density) + np.log(electron_temperature)
    nu_ei = (
        qe**4
        * ion_charge**2
        * ion_density
        * coulomb_log
        * (1.0 + electron_mass / ion_mass)
        / (
            3.0
            * np.pi ** 1.5
            * VACUUM_PERMITTIVITY**2
            * np.power(2.0 * qe * (electron_temperature / electron_mass + ion_temperature / ion_mass), 1.5)
            * electron_mass**2
        )
    )
    wci = qe * magnetic_field / ion_mass
    wce = qe * magnetic_field / electron_mass
    wstar = k_z * electron_temperature * inv_ln / magnetic_field
    sigmapar = (k_y / k_z) ** 2 * wci * wce / (0.51 * nu_ei)
    sigmapar_over_wstar = sigmapar / wstar

    simple_term = 0.5 * (np.sqrt(sigmapar_over_wstar**4 + 16.0 * sigmapar_over_wstar**2) - sigmapar_over_wstar**2)
    simple_analytic_omega_over_wstar = 0.5 * np.sqrt(simple_term)
    simple_analytic_gamma_over_wstar = sigmapar_over_wstar / np.sqrt(simple_term) - 0.5 * sigmapar_over_wstar

    roots = np.roots(
        [
            wstar / (0.51 * nu_ei),
            1j,
            -sigmapar_over_wstar,
            sigmapar_over_wstar,
        ]
    )
    fastest_growth_root = roots[np.argmax(np.imag(roots))]

    return DriftWaveBenchmarkScalars(
        wstar=float(wstar),
        sigmapar=float(sigmapar),
        sigmapar_over_wstar=float(sigmapar_over_wstar),
        coulomb_log=float(coulomb_log),
        nu_ei=float(nu_ei),
        wci=float(wci),
        wce=float(wce),
        analytic_gamma_over_wstar=float(np.imag(fastest_growth_root)),
        analytic_omega_over_wstar=float(np.real(fastest_growth_root)),
        simple_analytic_gamma_over_wstar=float(simple_analytic_gamma_over_wstar),
        simple_analytic_omega_over_wstar=float(simple_analytic_omega_over_wstar),
    )


def write_drift_wave_analysis_json(result: DriftWaveAnalysisResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "density_variable": result.density_variable,
        "trace_x_index": result.trace_x_index,
        "trace_y_index": result.trace_y_index,
        "fit_points": result.fit_points,
        "benchmark": asdict(result.benchmark),
        "measured_gamma_over_wstar": result.measured_gamma_over_wstar,
        "measured_omega_over_wstar": result.measured_omega_over_wstar,
        "equilibrium_density": result.equilibrium_density,
        "time_seconds": result.time_seconds.tolist(),
        "time_wstar": result.time_wstar.tolist(),
        "rms_history": result.rms_history.tolist(),
        "log_rms_history": result.log_rms_history.tolist(),
        "peak_history": result.peak_history.tolist(),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def save_drift_wave_diagnostic_plot(result: DriftWaveAnalysisResult, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fit_slice = slice(max(0, len(result.time_wstar) - result.fit_points), len(result.time_wstar))
    growth_fit = np.polyfit(result.time_wstar[fit_slice], result.log_rms_history[fit_slice], deg=1)
    phase_history = result.peak_history - result.peak_history[0]
    phase_fit = np.polyfit(result.time_wstar[fit_slice], phase_history[fit_slice], deg=1)

    figure, axes = plt.subplots(nrows=2, figsize=(8.0, 6.0), constrained_layout=True)

    axes[0].plot(result.time_wstar, result.log_rms_history, color="#0b7285", linewidth=2.0, label="Measured")
    axes[0].plot(
        result.time_wstar[fit_slice],
        np.polyval(growth_fit, result.time_wstar[fit_slice]),
        color="#d9480f",
        linestyle="--",
        linewidth=2.0,
        label="Tail fit",
    )
    axes[0].set_ylabel(r"$\log(n_\mathrm{rms})$")
    axes[0].set_title(
        "Drift-wave benchmark diagnostics\n"
        + rf"$\gamma/\omega_*={result.measured_gamma_over_wstar:.3f}$, "
        + rf"analytic $={result.benchmark.analytic_gamma_over_wstar:.3f}$"
    )
    axes[0].legend(loc="best")

    axes[1].plot(result.time_wstar, phase_history, color="#495057", linewidth=2.0, label="Tracked peak")
    axes[1].plot(
        result.time_wstar[fit_slice],
        np.polyval(phase_fit, result.time_wstar[fit_slice]),
        color="#d9480f",
        linestyle="--",
        linewidth=2.0,
        label="Tail fit",
    )
    axes[1].set_xlabel(r"$\omega_* t$")
    axes[1].set_ylabel("Peak offset")
    axes[1].set_title(
        rf"$\omega/\omega_*={result.measured_omega_over_wstar:.3f}$, "
        + rf"analytic $={result.benchmark.analytic_omega_over_wstar:.3f}$"
    )
    axes[1].legend(loc="best")

    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _select_density_trace(
    density_history: np.ndarray,
    *,
    x_index: int,
    y_index: int,
) -> np.ndarray:
    if density_history.ndim != 4:
        raise ValueError(f"Expected a 4D density history with shape (t, x, y, z); got {density_history.shape!r}")
    return np.asarray(density_history[:, x_index, y_index, :], dtype=np.float64)


def _track_peak_positions(perturbation: np.ndarray) -> np.ndarray:
    nt, nz = perturbation.shape
    peak = np.zeros(nt, dtype=np.float64)
    for time_index in range(nt):
        index = int(np.argmax(perturbation[time_index]))
        center = perturbation[time_index, index]
        minus = perturbation[time_index, (index - 1) % nz]
        plus = perturbation[time_index, (index + 1) % nz]
        linear = 0.5 * (plus - minus)
        quadratic = plus - (center + linear)
        if abs(quadratic) < 1.0e-14:
            peak[time_index] = float(index)
            continue
        peak[time_index] = float(index) - 0.5 * linear / quadratic

    if peak[-1] > peak[-2]:
        for index in range(1, nt):
            if peak[index] < peak[index - 1]:
                peak[index:] += nz
    else:
        for index in range(1, nt):
            if peak[index] > peak[index - 1]:
                peak[index:] -= nz
    return peak


def _resolve_fit_points(fit_points: int, *, total_points: int) -> int:
    if total_points < 2:
        raise ValueError("At least two time points are required to fit growth and frequency.")
    return min(max(int(fit_points), 2), total_points)


def _measure_growth_rate(
    log_rms_history: np.ndarray,
    *,
    time_seconds: np.ndarray,
    fit_points: int,
) -> float:
    return float(np.mean(np.gradient(log_rms_history, time_seconds)[-fit_points:]))


def _measure_frequency(
    peak_history: np.ndarray,
    *,
    time_seconds: np.ndarray,
    nz: int,
    fit_points: int,
) -> float:
    return float(2.0 * np.pi * np.mean(np.gradient(peak_history, time_seconds)[-fit_points:]) / float(nz))
