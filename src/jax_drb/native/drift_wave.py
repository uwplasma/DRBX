from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from ..config.normalization import ELEMENTARY_CHARGE, PROTON_MASS
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics

ELECTRON_MASS = 9.1093837015e-31
VACUUM_PERMITTIVITY = 8.8541878128e-12
_NEUMANN_PATTERN = re.compile(r"^\s*neumann\((.*)\)\s*$", re.IGNORECASE)
_UNITS_REFERENCE_PATTERN = re.compile(r"\bunits:([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True)
class DriftWaveBenchmark:
    ion_temperature: float
    ion_atomic_mass: float
    electron_temperature: float
    electron_atomic_mass: float
    electron_charge: float
    average_atomic_mass: float
    density_gradient_inner: np.ndarray
    density_gradient_outer: np.ndarray
    rho_s0: float
    dy: np.ndarray
    dz: np.ndarray
    J: np.ndarray
    g22: np.ndarray
    g33: np.ndarray
    Bxy: np.ndarray
    dx: float
    right_face_j: float
    left_face_j: float
    y_slice: slice
    Nnorm: float
    Tnorm: float
    Omega_ci: float
    sound_speed: float
    fastest_wave: float
    density_floor: float = 1.0e-7

    @property
    def momentum_coefficient(self) -> float:
        return 0.51


@dataclass(frozen=True)
class DriftWaveState:
    ion_density: np.ndarray
    electron_momentum: np.ndarray
    vorticity: np.ndarray


@dataclass(frozen=True)
class DriftWaveRhsResult:
    density: np.ndarray
    momentum: np.ndarray
    vorticity: np.ndarray
    potential: np.ndarray
    electron_density: np.ndarray
    electron_pressure: np.ndarray


@dataclass(frozen=True)
class DriftWaveHistoryResult:
    ion_density_history: np.ndarray
    electron_momentum_history: np.ndarray
    vorticity_history: np.ndarray
    potential_history: np.ndarray


def build_drift_wave_benchmark(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: Mapping[str, float],
) -> DriftWaveBenchmark:
    resolver = NumericResolver(config)
    ion_temperature = resolver.resolve("i", "temperature") / float(dataset_scalars["Tnorm"])
    ion_atomic_mass = resolver.resolve("i", "AA")
    electron_temperature = resolver.resolve("e", "temperature") / float(dataset_scalars["Tnorm"])
    electron_atomic_mass = resolver.resolve("e", "AA")
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    x_index = mesh.xstart
    metric_bxy = np.asarray(metrics.Bxy[x_index, y_slice, :], dtype=np.float64)
    if np.allclose(metric_bxy, 1.0, rtol=1e-12, atol=1e-12) and config.has_option("mesh", "B"):
        normalized_bxy = resolver.resolve("mesh", "B") / float(dataset_scalars["Bnorm"])
        metric_bxy = np.full_like(metric_bxy, normalized_bxy, dtype=np.float64)
    units = {
        "inv_meters_cubed": float(dataset_scalars["Nnorm"]),
        "eV": float(dataset_scalars["Tnorm"]),
        "Tesla": float(dataset_scalars["Bnorm"]),
        "seconds": 1.0 / float(dataset_scalars["Omega_ci"]),
        "meters": float(dataset_scalars["rho_s0"]),
    }
    return DriftWaveBenchmark(
        ion_temperature=ion_temperature,
        ion_atomic_mass=ion_atomic_mass,
        electron_temperature=electron_temperature,
        electron_atomic_mass=electron_atomic_mass,
        electron_charge=resolver.resolve("e", "charge"),
        average_atomic_mass=resolver.resolve("vorticity", "average_atomic_mass")
        if config.has_option("vorticity", "average_atomic_mass")
        else 1.0,
        density_gradient_inner=_evaluate_neumann_increment(config, "Ni", "bndry_xin", mesh=mesh, units=units),
        density_gradient_outer=_evaluate_neumann_increment(config, "Ni", "bndry_xout", mesh=mesh, units=units),
        rho_s0=float(dataset_scalars["rho_s0"]),
        dy=np.asarray(metrics.dy[x_index, y_slice, :], dtype=np.float64),
        dz=np.asarray(metrics.dz[x_index, y_slice, :], dtype=np.float64),
        J=np.asarray(metrics.J[x_index, y_slice, :], dtype=np.float64),
        g22=np.asarray(metrics.g22[x_index, y_slice, :], dtype=np.float64),
        g33=np.asarray(metrics.g33[x_index, y_slice, :], dtype=np.float64),
        Bxy=metric_bxy,
        dx=float(metrics.dx[x_index, mesh.ystart, 0]),
        right_face_j=0.5 * float(metrics.J[x_index, mesh.ystart, 0] + metrics.J[x_index + 1, mesh.ystart, 0]),
        left_face_j=0.5 * float(metrics.J[x_index, mesh.ystart, 0] + metrics.J[x_index - 1, mesh.ystart, 0]),
        y_slice=y_slice,
        Nnorm=float(dataset_scalars["Nnorm"]),
        Tnorm=float(dataset_scalars["Tnorm"]),
        Omega_ci=float(dataset_scalars["Omega_ci"]),
        sound_speed=_compute_collective_sound_speed(
            ion_temperature,
            ion_atomic_mass,
            electron_temperature,
            electron_atomic_mass,
        ),
        fastest_wave=_compute_fastest_wave(
            ion_temperature,
            ion_atomic_mass,
            electron_temperature,
            electron_atomic_mass,
        ),
    )


def initialize_drift_wave_state(config: BoutConfig, *, mesh: StructuredMesh) -> DriftWaveState:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    density = broadcast_to_field_shape(evaluator.resolve_option("Ni", "function"), mesh)
    density = np.asarray(density[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    zeros = np.zeros_like(density, dtype=np.float64)
    return DriftWaveState(ion_density=density, electron_momentum=zeros, vorticity=zeros)


def compute_drift_wave_rhs(
    state: DriftWaveState,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    include_parallel_transport: bool = False,
    include_phi_dissipation: bool = False,
) -> DriftWaveRhsResult:
    density_full = _assemble_density_field(state.ion_density, benchmark=benchmark, mesh=mesh)
    momentum_full = _assemble_zero_dirichlet_field(state.electron_momentum, mesh=mesh)
    vorticity_full = _assemble_zero_dirichlet_field(state.vorticity, mesh=mesh)

    potential = solve_drift_wave_potential(state.vorticity, benchmark=benchmark)
    potential_full = _assemble_neumann_potential_field(potential, mesh=mesh)

    density_rhs_full = -_compute_xz_exb_divergence(density_full, potential_full, mesh=mesh, benchmark=benchmark)
    momentum_rhs_full = -_compute_xz_exb_divergence(momentum_full, potential_full, mesh=mesh, benchmark=benchmark)
    vorticity_rhs_full = -_compute_xz_exb_divergence(vorticity_full, potential_full, mesh=mesh, benchmark=benchmark)

    electron_density = np.asarray(state.ion_density, dtype=np.float64)
    electron_density_limited = np.maximum(electron_density, benchmark.density_floor)
    electron_pressure = electron_density * benchmark.electron_temperature
    collision_frequency = _electron_ion_collision_frequency(electron_density, benchmark=benchmark)
    electron_velocity = state.electron_momentum / (benchmark.electron_atomic_mass * electron_density_limited)

    momentum_rhs = np.asarray(momentum_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    momentum_rhs += electron_density * _grad_par_periodic(potential, benchmark=benchmark)
    momentum_rhs -= _grad_par_periodic(electron_pressure, benchmark=benchmark)
    if include_parallel_transport:
        momentum_rhs -= benchmark.electron_atomic_mass * _div_par_fvv_periodic(
            electron_density_limited,
            electron_velocity,
            benchmark.fastest_wave,
            benchmark=benchmark,
        )
    momentum_rhs -= benchmark.momentum_coefficient * collision_frequency * state.electron_momentum

    parallel_current = (benchmark.electron_charge / benchmark.electron_atomic_mass) * state.electron_momentum
    vorticity_rhs = np.asarray(vorticity_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    vorticity_rhs += _div_par_periodic(parallel_current, benchmark=benchmark)
    if include_phi_dissipation:
        vorticity_rhs -= _div_par_scalar_periodic(-potential, benchmark.sound_speed, benchmark=benchmark)

    return DriftWaveRhsResult(
        density=np.asarray(density_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64),
        momentum=momentum_rhs,
        vorticity=vorticity_rhs,
        potential=potential,
        electron_density=electron_density,
        electron_pressure=electron_pressure,
    )


def advance_drift_wave_history(
    initial_state: DriftWaveState,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    timestep: float,
    steps: int,
    substeps: int,
    include_parallel_transport: bool = False,
    include_phi_dissipation: bool = False,
) -> DriftWaveHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if substeps <= 0:
        raise ValueError("substeps must be positive")

    state = initial_state
    density_history = [_assemble_density_field(state.ion_density, benchmark=benchmark, mesh=mesh)]
    momentum_history = [_assemble_zero_dirichlet_field(state.electron_momentum, mesh=mesh)]
    vorticity_history = [_assemble_zero_dirichlet_field(state.vorticity, mesh=mesh)]
    potential_history = [np.zeros_like(density_history[0], dtype=np.float64)]

    sub_timestep = timestep / float(substeps)
    for _ in range(steps):
        for _ in range(substeps):
            state = _rk4_step(
                state,
                mesh=mesh,
                benchmark=benchmark,
                timestep=sub_timestep,
                include_parallel_transport=include_parallel_transport,
                include_phi_dissipation=include_phi_dissipation,
            )
        density_history.append(_assemble_density_field(state.ion_density, benchmark=benchmark, mesh=mesh))
        momentum_history.append(_assemble_zero_dirichlet_field(state.electron_momentum, mesh=mesh))
        vorticity_history.append(_assemble_zero_dirichlet_field(state.vorticity, mesh=mesh))
        potential_history.append(
            _assemble_neumann_potential_field(
                solve_drift_wave_potential(state.vorticity, benchmark=benchmark),
                mesh=mesh,
            )
        )

    return DriftWaveHistoryResult(
        ion_density_history=np.stack(density_history, axis=0),
        electron_momentum_history=np.stack(momentum_history, axis=0),
        vorticity_history=np.stack(vorticity_history, axis=0),
        potential_history=np.stack(potential_history, axis=0),
    )


def solve_drift_wave_potential(vorticity: np.ndarray, *, benchmark: DriftWaveBenchmark) -> np.ndarray:
    hat = np.fft.rfft(np.asarray(vorticity, dtype=np.float64), axis=-1)
    output = np.zeros_like(hat)
    zlength = float(benchmark.dz[0, 0]) * float(vorticity.shape[-1])
    bxy = benchmark.Bxy[:, 0]
    g33 = benchmark.g33[:, 0]
    for mode in range(1, hat.shape[-1]):
        wave_number = (2.0 * np.pi * mode) / zlength
        output[:, mode] = -(bxy * bxy) * hat[:, mode] / (
            benchmark.average_atomic_mass * g33 * wave_number * wave_number
        )
    return np.asarray(np.fft.irfft(output, n=vorticity.shape[-1], axis=-1), dtype=np.float64)


def _rk4_step(
    state: DriftWaveState,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    timestep: float,
    include_parallel_transport: bool,
    include_phi_dissipation: bool,
) -> DriftWaveState:
    k1 = compute_drift_wave_rhs(
        state,
        mesh=mesh,
        benchmark=benchmark,
        include_parallel_transport=include_parallel_transport,
        include_phi_dissipation=include_phi_dissipation,
    )
    k2 = compute_drift_wave_rhs(
        _add_state(state, k1, scale=0.5 * timestep),
        mesh=mesh,
        benchmark=benchmark,
        include_parallel_transport=include_parallel_transport,
        include_phi_dissipation=include_phi_dissipation,
    )
    k3 = compute_drift_wave_rhs(
        _add_state(state, k2, scale=0.5 * timestep),
        mesh=mesh,
        benchmark=benchmark,
        include_parallel_transport=include_parallel_transport,
        include_phi_dissipation=include_phi_dissipation,
    )
    k4 = compute_drift_wave_rhs(
        _add_state(state, k3, scale=timestep),
        mesh=mesh,
        benchmark=benchmark,
        include_parallel_transport=include_parallel_transport,
        include_phi_dissipation=include_phi_dissipation,
    )
    return DriftWaveState(
        ion_density=state.ion_density + (timestep / 6.0) * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density),
        electron_momentum=state.electron_momentum
        + (timestep / 6.0) * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum),
        vorticity=state.vorticity + (timestep / 6.0) * (k1.vorticity + 2.0 * k2.vorticity + 2.0 * k3.vorticity + k4.vorticity),
    )


def _add_state(state: DriftWaveState, rhs: DriftWaveRhsResult, *, scale: float) -> DriftWaveState:
    return DriftWaveState(
        ion_density=state.ion_density + scale * rhs.density,
        electron_momentum=state.electron_momentum + scale * rhs.momentum,
        vorticity=state.vorticity + scale * rhs.vorticity,
    )


def _assemble_density_field(interior: np.ndarray, *, benchmark: DriftWaveBenchmark, mesh: StructuredMesh) -> np.ndarray:
    field = _assemble_interior_field(interior, mesh=mesh)
    for offset in range(1, mesh.mxg + 1):
        field[mesh.xstart - offset, benchmark.y_slice, :] = field[mesh.xstart - offset + 1, benchmark.y_slice, :] - benchmark.density_gradient_inner
        field[mesh.xend + offset, benchmark.y_slice, :] = field[mesh.xend + offset - 1, benchmark.y_slice, :] + benchmark.density_gradient_outer
    return field


def _assemble_zero_dirichlet_field(interior: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    field = _assemble_interior_field(interior, mesh=mesh)
    if mesh.mxg > 0:
        field[mesh.xstart - 1, mesh.ystart : mesh.yend + 1, :] = -field[mesh.xstart, mesh.ystart : mesh.yend + 1, :]
        field[mesh.xend + 1, mesh.ystart : mesh.yend + 1, :] = -field[mesh.xend, mesh.ystart : mesh.yend + 1, :]
        for offset in range(2, mesh.mxg + 1):
            field[mesh.xstart - offset, mesh.ystart : mesh.yend + 1, :] = 0.0
            field[mesh.xend + offset, mesh.ystart : mesh.yend + 1, :] = 0.0
    return field


def _assemble_neumann_potential_field(interior: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    field = _assemble_interior_field(interior, mesh=mesh)
    for offset in range(1, mesh.mxg + 1):
        field[mesh.xstart - offset, mesh.ystart : mesh.yend + 1, :] = field[mesh.xstart - offset + 1, mesh.ystart : mesh.yend + 1, :]
        field[mesh.xend + offset, mesh.ystart : mesh.yend + 1, :] = field[mesh.xend + offset - 1, mesh.ystart : mesh.yend + 1, :]
    return field


def _assemble_interior_field(interior: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    field[mesh.xstart, mesh.ystart : mesh.yend + 1, :] = np.asarray(interior, dtype=np.float64)
    for offset in range(mesh.myg):
        field[:, mesh.ystart - 1 - offset, :] = field[:, mesh.yend - offset, :]
        field[:, mesh.yend + 1 + offset, :] = field[:, mesh.ystart + offset, :]
    return field


def _compute_xz_exb_divergence(
    field: np.ndarray,
    potential: np.ndarray,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    for j in range(mesh.ystart, mesh.yend + 1):
        for i in range(mesh.xstart, mesh.xend + 1):
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                km = (k - 1 + mesh.nz) % mesh.nz
                fmm = 0.25 * (potential[i, j, k] + potential[i - 1, j, k] + potential[i, j, km] + potential[i - 1, j, km])
                fmp = 0.25 * (potential[i, j, k] + potential[i, j, kp] + potential[i - 1, j, k] + potential[i - 1, j, kp])
                fpp = 0.25 * (potential[i, j, k] + potential[i, j, kp] + potential[i + 1, j, k] + potential[i + 1, j, kp])
                fpm = 0.25 * (potential[i, j, k] + potential[i + 1, j, k] + potential[i, j, km] + potential[i + 1, j, km])
                v_up = benchmark.J[j - mesh.ystart, k] * (fmp - fpp) / benchmark.dx
                v_down = benchmark.J[j - mesh.ystart, k] * (fmm - fpm) / benchmark.dx
                v_right = benchmark.right_face_j * (fpp - fpm) / benchmark.dz[j - mesh.ystart, k]
                v_left = benchmark.left_face_j * (fmp - fmm) / benchmark.dz[j - mesh.ystart, k]
                center = field[i, j, k]
                x_left_face, x_right_face = _mc_cell_edges(center, field[i - 1, j, k], field[i + 1, j, k])
                if v_right > 0.0:
                    result[i, j, k] += v_right * x_right_face / (benchmark.dx * benchmark.J[j - mesh.ystart, k])
                if v_left < 0.0:
                    result[i, j, k] += -v_left * x_left_face / (benchmark.dx * benchmark.J[j - mesh.ystart, k])
                z_left_face, z_right_face = _mc_cell_edges(center, field[i, j, km], field[i, j, kp])
                if v_up > 0.0:
                    result[i, j, k] += v_up * z_right_face / (benchmark.J[j - mesh.ystart, k] * benchmark.dz[j - mesh.ystart, k])
                if v_down < 0.0:
                    result[i, j, k] += -v_down * z_left_face / (benchmark.J[j - mesh.ystart, k] * benchmark.dz[j - mesh.ystart, k])
    return result


def _mc_cell_edges(center: np.ndarray, minus: np.ndarray, plus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def _grad_par_periodic(field: np.ndarray, *, benchmark: DriftWaveBenchmark) -> np.ndarray:
    return benchmark.rho_s0 * (np.roll(field, shift=-1, axis=0) - np.roll(field, shift=1, axis=0)) / (
        2.0 * benchmark.dy * np.sqrt(benchmark.g22)
    )


def _div_par_periodic(field: np.ndarray, *, benchmark: DriftWaveBenchmark) -> np.ndarray:
    return benchmark.Bxy * _grad_par_periodic(field / benchmark.Bxy, benchmark=benchmark)


def _electron_ion_collision_frequency(density: np.ndarray, *, benchmark: DriftWaveBenchmark) -> np.ndarray:
    electron_temperature = benchmark.electron_temperature * benchmark.Tnorm
    ion_temperature = benchmark.ion_temperature * benchmark.Tnorm
    electron_density = np.maximum(density * benchmark.Nnorm, 1.0e10)
    ion_density = np.maximum(density * benchmark.Nnorm, 1.0e10)
    me_over_mi = ELECTRON_MASS / PROTON_MASS
    coulomb_log = 31.0 - 0.5 * np.log(electron_density) + np.log(electron_temperature)
    electron_speed_sq = 2.0 * electron_temperature * ELEMENTARY_CHARGE / ELECTRON_MASS
    ion_speed_sq = 2.0 * ion_temperature * ELEMENTARY_CHARGE / PROTON_MASS
    numerator = (ELEMENTARY_CHARGE**4) * ion_density * np.maximum(coulomb_log, 1.0) * (1.0 + me_over_mi)
    denominator = 3.0 * np.power(np.pi * (electron_speed_sq + ion_speed_sq), 1.5) * (VACUUM_PERMITTIVITY * ELECTRON_MASS) ** 2
    return numerator / denominator / benchmark.Omega_ci


def _evaluate_neumann_increment(
    config: BoutConfig,
    variable_name: str,
    option_name: str,
    *,
    mesh: StructuredMesh,
    units: Mapping[str, float],
) -> np.ndarray:
    if not config.has_option(variable_name, option_name):
        return np.zeros((mesh.ny, mesh.nz), dtype=np.float64)
    raw = str(config.raw(variable_name, option_name)).strip()
    match = _NEUMANN_PATTERN.match(raw)
    if match is None:
        raise NotImplementedError(f"Expected {variable_name}:{option_name} to be a neumann(...) specification.")
    expression = _substitute_unit_references(match.group(1), units)
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    value = broadcast_to_field_shape(evaluator.evaluate(expression, current_section=variable_name), mesh)
    return np.asarray(value[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)


def _substitute_unit_references(expression: str, units: Mapping[str, float]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in units:
            raise KeyError(f"Unsupported unit reference units:{key}")
        return repr(float(units[key]))

    return _UNITS_REFERENCE_PATTERN.sub(replace, expression)


def _compute_collective_sound_speed(
    ion_temperature: float,
    ion_atomic_mass: float,
    electron_temperature: float,
    electron_atomic_mass: float,
) -> float:
    return float(np.sqrt((ion_temperature + electron_temperature) / (ion_atomic_mass + electron_atomic_mass)))


def _compute_fastest_wave(
    ion_temperature: float,
    ion_atomic_mass: float,
    electron_temperature: float,
    electron_atomic_mass: float,
) -> float:
    ion_wave = np.sqrt(ion_temperature / ion_atomic_mass)
    electron_wave = np.sqrt(electron_temperature / electron_atomic_mass)
    sound_speed = _compute_collective_sound_speed(
        ion_temperature,
        ion_atomic_mass,
        electron_temperature,
        electron_atomic_mass,
    )
    return float(max(ion_wave, electron_wave, sound_speed))


def _div_par_scalar_periodic(
    field: np.ndarray,
    wave_speed: float | np.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> np.ndarray:
    left_cell, right_cell = _mc_field_edges(field)
    wave = _broadcast_wave_speed(wave_speed, field)
    amax_right = np.maximum(wave, np.roll(wave, shift=-1, axis=0))
    amax_left = np.maximum(wave, np.roll(wave, shift=1, axis=0))
    flux_right = 0.5 * right_cell * amax_right
    flux_left = -0.5 * left_cell * amax_left
    return _scatter_face_divergence(flux_right, flux_left, benchmark=benchmark)


def _div_par_fvv_periodic(
    density: np.ndarray,
    velocity: np.ndarray,
    wave_speed: float | np.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> np.ndarray:
    density_left_cell, density_right_cell = _mc_field_edges(density)
    velocity_left_cell, velocity_right_cell = _mc_field_edges(velocity)
    wave = _broadcast_wave_speed(wave_speed, velocity)
    velocity_center = np.asarray(velocity, dtype=np.float64)
    amax_right = np.maximum(
        np.maximum(wave, np.roll(wave, shift=-1, axis=0)),
        np.maximum(np.abs(velocity_center), np.abs(np.roll(velocity_center, shift=-1, axis=0))),
    )
    amax_left = np.maximum(
        np.maximum(wave, np.roll(wave, shift=1, axis=0)),
        np.maximum(np.abs(velocity_center), np.abs(np.roll(velocity_center, shift=1, axis=0))),
    )
    flux_right = density_right_cell * 0.5 * (velocity_right_cell + amax_right) * velocity_right_cell
    flux_left = density_left_cell * 0.5 * (velocity_left_cell - amax_left) * velocity_left_cell
    return _scatter_face_divergence(flux_right, flux_left, benchmark=benchmark)


def _mc_field_edges(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(field, dtype=np.float64)
    minus = np.roll(center, shift=1, axis=0)
    plus = np.roll(center, shift=-1, axis=0)
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _broadcast_wave_speed(wave_speed: float | np.ndarray, field: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave_speed, dtype=np.float64)
    if wave.ndim == 0:
        wave = np.full_like(field, float(wave), dtype=np.float64)
    return wave


def _scatter_face_divergence(
    flux_right: np.ndarray,
    flux_left: np.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> np.ndarray:
    common_right = (benchmark.J + np.roll(benchmark.J, shift=-1, axis=0)) / (
        np.sqrt(benchmark.g22) + np.sqrt(np.roll(benchmark.g22, shift=-1, axis=0))
    )
    common_left = (benchmark.J + np.roll(benchmark.J, shift=1, axis=0)) / (
        np.sqrt(benchmark.g22) + np.sqrt(np.roll(benchmark.g22, shift=1, axis=0))
    )
    result_right = benchmark.rho_s0 * flux_right * common_right / (benchmark.dy * benchmark.J)
    result_left = benchmark.rho_s0 * flux_left * common_left / (benchmark.dy * benchmark.J)
    return result_right - result_left
