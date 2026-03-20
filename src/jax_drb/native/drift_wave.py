from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
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
    density_bndry_flux: bool
    momentum_bndry_flux: bool
    vorticity_bndry_flux: bool
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
        g22=np.asarray(metrics.g22[x_index, y_slice, :], dtype=np.float64) / float(dataset_scalars["rho_s0"]) ** 2,
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
        density_bndry_flux=bool(config.parsed("i", "bndry_flux")) if config.has_option("i", "bndry_flux") else True,
        momentum_bndry_flux=bool(config.parsed("e", "bndry_flux")) if config.has_option("e", "bndry_flux") else True,
        vorticity_bndry_flux=bool(config.parsed("vorticity", "bndry_flux"))
        if config.has_option("vorticity", "bndry_flux")
        else False,
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

    density_rhs_full = -_compute_xz_exb_divergence(
        density_full,
        potential_full,
        mesh=mesh,
        benchmark=benchmark,
        bndry_flux=benchmark.density_bndry_flux,
    )
    momentum_rhs_full = -_compute_xz_exb_divergence(
        momentum_full,
        potential_full,
        mesh=mesh,
        benchmark=benchmark,
        bndry_flux=benchmark.momentum_bndry_flux,
    )
    vorticity_rhs_full = -_compute_xz_exb_divergence(
        vorticity_full,
        potential_full,
        mesh=mesh,
        benchmark=benchmark,
        bndry_flux=benchmark.vorticity_bndry_flux,
    )

    electron_density = jnp.asarray(state.ion_density, dtype=jnp.float64)
    electron_density_limited = jnp.maximum(electron_density, benchmark.density_floor)
    electron_pressure = electron_density * benchmark.electron_temperature
    collision_frequency = _electron_ion_collision_frequency(electron_density, benchmark=benchmark)
    electron_velocity = state.electron_momentum / (benchmark.electron_atomic_mass * electron_density_limited)

    momentum_rhs = jnp.asarray(momentum_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64)
    momentum_rhs = momentum_rhs + electron_density * _grad_par_periodic(potential, benchmark=benchmark)
    momentum_rhs = momentum_rhs - _grad_par_periodic(electron_pressure, benchmark=benchmark)
    if include_parallel_transport:
        momentum_rhs = momentum_rhs - benchmark.electron_atomic_mass * _div_par_fvv_periodic(
            electron_density_limited,
            electron_velocity,
            benchmark.fastest_wave,
            benchmark=benchmark,
        )
    momentum_rhs = momentum_rhs - benchmark.momentum_coefficient * collision_frequency * state.electron_momentum

    parallel_current = (benchmark.electron_charge / benchmark.electron_atomic_mass) * state.electron_momentum
    vorticity_rhs = jnp.asarray(vorticity_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64)
    vorticity_rhs = vorticity_rhs + _div_par_periodic(parallel_current, benchmark=benchmark)
    if include_phi_dissipation:
        vorticity_rhs = vorticity_rhs - _div_par_scalar_periodic(-potential, benchmark.sound_speed, benchmark=benchmark)

    return DriftWaveRhsResult(
        density=jnp.asarray(density_rhs_full[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64),
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


def advance_drift_wave_history_adaptive(
    initial_state: DriftWaveState,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    timestep: float,
    steps: int,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    max_step: float | None = None,
    initial_step: float | None = None,
    include_parallel_transport: bool = False,
    include_phi_dissipation: bool = False,
) -> DriftWaveHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if timestep <= 0.0 and steps > 0:
        raise ValueError("timestep must be positive when steps > 0")

    target_times = np.asarray([timestep * index for index in range(steps + 1)], dtype=np.float64)
    state_shape = initial_state.ion_density.shape
    state_flat = _flatten_state(initial_state)
    current_time = 0.0
    step_size = float(initial_step if initial_step is not None else min(timestep, 1.0))
    max_step_size = float(max_step if max_step is not None else timestep)

    density_history = [_assemble_density_field(initial_state.ion_density, benchmark=benchmark, mesh=mesh)]
    momentum_history = [_assemble_zero_dirichlet_field(initial_state.electron_momentum, mesh=mesh)]
    vorticity_history = [_assemble_zero_dirichlet_field(initial_state.vorticity, mesh=mesh)]
    potential_history = [np.zeros_like(density_history[0], dtype=np.float64)]

    for target_time in target_times[1:]:
        while current_time + 1.0e-14 < float(target_time):
            trial_step = min(step_size, max_step_size, float(target_time) - current_time)
            next_state, error_estimate = _rk23_step_flat(
                state_flat,
                timestep=trial_step,
                state_shape=state_shape,
                mesh=mesh,
                benchmark=benchmark,
                include_parallel_transport=include_parallel_transport,
                include_phi_dissipation=include_phi_dissipation,
            )
            scale = atol + rtol * np.maximum(np.abs(state_flat), np.abs(next_state))
            error_norm = float(np.sqrt(np.mean(np.square(error_estimate / scale))))
            if error_norm <= 1.0 or trial_step <= 1.0e-12:
                state_flat = next_state
                current_time += trial_step
                step_size = _next_adaptive_step(trial_step, error_norm)
                continue
            step_size = _next_adaptive_step(trial_step, error_norm)

        state = _unflatten_state(state_flat, state_shape)
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


def _flatten_state(state: DriftWaveState) -> np.ndarray:
    return np.concatenate(
        [
            np.ravel(np.asarray(state.ion_density, dtype=np.float64)),
            np.ravel(np.asarray(state.electron_momentum, dtype=np.float64)),
            np.ravel(np.asarray(state.vorticity, dtype=np.float64)),
        ]
    )


def _unflatten_state(state: np.ndarray, shape: tuple[int, int]) -> DriftWaveState:
    block = int(np.prod(shape))
    return DriftWaveState(
        ion_density=np.asarray(state[:block], dtype=np.float64).reshape(shape),
        electron_momentum=np.asarray(state[block : 2 * block], dtype=np.float64).reshape(shape),
        vorticity=np.asarray(state[2 * block :], dtype=np.float64).reshape(shape),
    )


def _rk23_step_flat(
    state_flat: np.ndarray,
    *,
    timestep: float,
    state_shape: tuple[int, int],
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    include_parallel_transport: bool,
    include_phi_dissipation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    state = _unflatten_state(state_flat, state_shape)
    k1 = _flatten_rhs(
        compute_drift_wave_rhs(
            state,
            mesh=mesh,
            benchmark=benchmark,
            include_parallel_transport=include_parallel_transport,
            include_phi_dissipation=include_phi_dissipation,
        )
    )
    k2_state = state_flat + timestep * (0.5 * k1)
    k2 = _flatten_rhs(
        compute_drift_wave_rhs(
            _unflatten_state(k2_state, state_shape),
            mesh=mesh,
            benchmark=benchmark,
            include_parallel_transport=include_parallel_transport,
            include_phi_dissipation=include_phi_dissipation,
        )
    )
    k3_state = state_flat + timestep * (0.75 * k2)
    k3 = _flatten_rhs(
        compute_drift_wave_rhs(
            _unflatten_state(k3_state, state_shape),
            mesh=mesh,
            benchmark=benchmark,
            include_parallel_transport=include_parallel_transport,
            include_phi_dissipation=include_phi_dissipation,
        )
    )
    third_order = state_flat + timestep * ((2.0 / 9.0) * k1 + (1.0 / 3.0) * k2 + (4.0 / 9.0) * k3)
    k4 = _flatten_rhs(
        compute_drift_wave_rhs(
            _unflatten_state(third_order, state_shape),
            mesh=mesh,
            benchmark=benchmark,
            include_parallel_transport=include_parallel_transport,
            include_phi_dissipation=include_phi_dissipation,
        )
    )
    second_order = state_flat + timestep * (
        (7.0 / 24.0) * k1 + 0.25 * k2 + (1.0 / 3.0) * k3 + 0.125 * k4
    )
    return third_order, third_order - second_order


def _flatten_rhs(rhs: DriftWaveRhsResult) -> np.ndarray:
    return np.concatenate(
        [
            np.ravel(np.asarray(rhs.density, dtype=np.float64)),
            np.ravel(np.asarray(rhs.momentum, dtype=np.float64)),
            np.ravel(np.asarray(rhs.vorticity, dtype=np.float64)),
        ]
    )


def _next_adaptive_step(current_step: float, error_norm: float) -> float:
    if error_norm <= 0.0:
        return current_step * 2.0
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    factor = safety * error_norm ** (-1.0 / 3.0)
    factor = min(max(factor, min_factor), max_factor)
    return current_step * factor


def _assemble_density_field(interior: np.ndarray, *, benchmark: DriftWaveBenchmark, mesh: StructuredMesh) -> np.ndarray:
    field = _assemble_interior_field(interior, mesh=mesh)
    inner_increment = benchmark.density_gradient_inner * benchmark.dx
    outer_increment = benchmark.density_gradient_outer * benchmark.dx
    for offset in range(1, mesh.mxg + 1):
        field[mesh.xstart - offset, benchmark.y_slice, :] = (
            field[mesh.xstart - offset + 1, benchmark.y_slice, :] - inner_increment
        )
        field[mesh.xend + offset, benchmark.y_slice, :] = (
            field[mesh.xend + offset - 1, benchmark.y_slice, :] + outer_increment
        )
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
    field: jnp.ndarray,
    potential: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    benchmark: DriftWaveBenchmark,
    bndry_flux: bool,
) -> jnp.ndarray:
    result = jnp.zeros_like(field, dtype=jnp.float64)
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_slice = slice(mesh.ystart, mesh.yend + 1)

    field_center = jnp.asarray(field[x_slice, y_slice, :], dtype=jnp.float64)
    field_left = jnp.asarray(field[mesh.xstart - 1 : mesh.xend, y_slice, :], dtype=jnp.float64)
    field_right = jnp.asarray(field[mesh.xstart + 1 : mesh.xend + 2, y_slice, :], dtype=jnp.float64)

    potential_center = jnp.asarray(potential[x_slice, y_slice, :], dtype=jnp.float64)
    potential_left = jnp.asarray(potential[mesh.xstart - 1 : mesh.xend, y_slice, :], dtype=jnp.float64)
    potential_right = jnp.asarray(potential[mesh.xstart + 1 : mesh.xend + 2, y_slice, :], dtype=jnp.float64)

    potential_center_km = jnp.roll(potential_center, shift=1, axis=-1)
    potential_center_kp = jnp.roll(potential_center, shift=-1, axis=-1)
    potential_left_km = jnp.roll(potential_left, shift=1, axis=-1)
    potential_left_kp = jnp.roll(potential_left, shift=-1, axis=-1)
    potential_right_km = jnp.roll(potential_right, shift=1, axis=-1)
    potential_right_kp = jnp.roll(potential_right, shift=-1, axis=-1)

    fmm = 0.25 * (potential_center + potential_left + potential_center_km + potential_left_km)
    fmp = 0.25 * (potential_center + potential_center_kp + potential_left + potential_left_kp)
    fpp = 0.25 * (potential_center + potential_center_kp + potential_right + potential_right_kp)
    fpm = 0.25 * (potential_center + potential_right + potential_center_km + potential_right_km)

    J = jnp.asarray(benchmark.J, dtype=jnp.float64)[None, :, :]
    dz = jnp.asarray(benchmark.dz, dtype=jnp.float64)[None, :, :]
    inv_cell = 1.0 / (benchmark.dx * J)
    inv_z_cell = 1.0 / (J * dz)

    v_up = J * (fmp - fpp) / benchmark.dx
    v_down = J * (fmm - fpm) / benchmark.dx
    v_right = benchmark.right_face_j * (fpp - fpm) / dz
    v_left = benchmark.left_face_j * (fmp - fmm) / dz

    x_left_face, x_right_face = _mc_cell_edges(field_center, field_left, field_right)
    z_left_face, z_right_face = _mc_cell_edges(
        field_center,
        jnp.roll(field_center, shift=1, axis=-1),
        jnp.roll(field_center, shift=-1, axis=-1),
    )

    active_result = jnp.zeros_like(field_center, dtype=jnp.float64)

    if field_center.shape[0] > 1:
        right_flux = jnp.where(v_right[:-1] > 0.0, v_right[:-1] * x_right_face[:-1], 0.0)
        active_result = active_result.at[:-1].add(right_flux * inv_cell)
        active_result = active_result.at[1:].add(-right_flux * inv_cell)

        left_flux = jnp.where(v_left[1:] < 0.0, v_left[1:] * x_left_face[1:], 0.0)
        active_result = active_result.at[1:].add(-left_flux * inv_cell)
        active_result = active_result.at[:-1].add(left_flux * inv_cell)

    if bndry_flux:
        right_boundary_flux = jnp.where(
            v_right[-1] > 0.0,
            v_right[-1] * x_right_face[-1],
            v_right[-1] * 0.5 * (field_right[-1] + field_center[-1]),
        )
        active_result = active_result.at[-1].add(right_boundary_flux * inv_cell[0])
        result = result.at[mesh.xend + 1, y_slice, :].add(-right_boundary_flux * inv_cell[0])

        left_boundary_flux = jnp.where(
            v_left[0] < 0.0,
            v_left[0] * x_left_face[0],
            v_left[0] * 0.5 * (field_left[0] + field_center[0]),
        )
        active_result = active_result.at[0].add(-left_boundary_flux * inv_cell[0])
        result = result.at[mesh.xstart - 1, y_slice, :].add(left_boundary_flux * inv_cell[0])

    up_flux = jnp.where(v_up > 0.0, v_up * z_right_face * inv_z_cell, 0.0)
    active_result += up_flux
    active_result -= jnp.roll(up_flux, shift=1, axis=-1)

    down_flux = jnp.where(v_down < 0.0, v_down * z_left_face * inv_z_cell, 0.0)
    active_result -= down_flux
    active_result += jnp.roll(down_flux, shift=-1, axis=-1)

    return result.at[x_slice, y_slice, :].set(active_result)


def _mc_cell_edges(center: jnp.ndarray, minus: jnp.ndarray, plus: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _minmod3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
    return jnp.where(same_sign, jnp.sign(a) * magnitude, 0.0)


def _grad_par_periodic(field: jnp.ndarray, *, benchmark: DriftWaveBenchmark) -> jnp.ndarray:
    return benchmark.rho_s0 * (jnp.roll(field, shift=-1, axis=0) - jnp.roll(field, shift=1, axis=0)) / (
        2.0 * benchmark.dy * jnp.sqrt(benchmark.g22)
    )


def _div_par_periodic(field: jnp.ndarray, *, benchmark: DriftWaveBenchmark) -> jnp.ndarray:
    return benchmark.Bxy * _grad_par_periodic(field / benchmark.Bxy, benchmark=benchmark)


def _electron_ion_collision_frequency(density: jnp.ndarray, *, benchmark: DriftWaveBenchmark) -> jnp.ndarray:
    electron_temperature = benchmark.electron_temperature * benchmark.Tnorm
    ion_temperature = benchmark.ion_temperature * benchmark.Tnorm
    electron_density = jnp.maximum(density * benchmark.Nnorm, 1.0e10)
    ion_density = jnp.maximum(density * benchmark.Nnorm, 1.0e10)
    me_over_mi = ELECTRON_MASS / PROTON_MASS
    coulomb_log = 31.0 - 0.5 * jnp.log(electron_density) + jnp.log(electron_temperature)
    electron_speed_sq = 2.0 * electron_temperature * ELEMENTARY_CHARGE / ELECTRON_MASS
    ion_speed_sq = 2.0 * ion_temperature * ELEMENTARY_CHARGE / PROTON_MASS
    numerator = (ELEMENTARY_CHARGE**4) * ion_density * jnp.maximum(coulomb_log, 1.0) * (1.0 + me_over_mi)
    denominator = 3.0 * jnp.power(jnp.pi * (electron_speed_sq + ion_speed_sq), 1.5) * (VACUUM_PERMITTIVITY * ELECTRON_MASS) ** 2
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
    field: jnp.ndarray,
    wave_speed: float | jnp.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> jnp.ndarray:
    left_cell, right_cell = _mc_field_edges(field)
    wave = _broadcast_wave_speed(wave_speed, field)
    amax_right = jnp.maximum(wave, jnp.roll(wave, shift=-1, axis=0))
    amax_left = jnp.maximum(wave, jnp.roll(wave, shift=1, axis=0))
    midpoint_right = 0.5 * (jnp.asarray(field, dtype=jnp.float64) + jnp.roll(field, shift=-1, axis=0))
    midpoint_left = 0.5 * (jnp.asarray(field, dtype=jnp.float64) + jnp.roll(field, shift=1, axis=0))
    flux_right = amax_right * (right_cell - midpoint_right)
    flux_left = -amax_left * (left_cell - midpoint_left)
    return _scatter_face_divergence(flux_right, flux_left, benchmark=benchmark)


def _div_par_fvv_periodic(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    wave_speed: float | jnp.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> jnp.ndarray:
    density_left_cell, density_right_cell = _mc_field_edges(density)
    velocity_left_cell, velocity_right_cell = _mc_field_edges(velocity)
    wave = _broadcast_wave_speed(wave_speed, velocity)
    velocity_center = jnp.asarray(velocity, dtype=jnp.float64)
    density_center = jnp.asarray(density, dtype=jnp.float64)
    amax_right = jnp.maximum(
        jnp.maximum(wave, jnp.roll(wave, shift=-1, axis=0)),
        jnp.maximum(jnp.abs(velocity_center), jnp.abs(jnp.roll(velocity_center, shift=-1, axis=0))),
    )
    amax_left = jnp.maximum(
        jnp.maximum(wave, jnp.roll(wave, shift=1, axis=0)),
        jnp.maximum(jnp.abs(velocity_center), jnp.abs(jnp.roll(velocity_center, shift=1, axis=0))),
    )
    velocity_mid_right = 0.5 * (velocity_center + jnp.roll(velocity_center, shift=-1, axis=0))
    velocity_mid_left = 0.5 * (velocity_center + jnp.roll(velocity_center, shift=1, axis=0))
    density_mid_right = 0.5 * (density_center + jnp.roll(density_center, shift=-1, axis=0))
    density_mid_left = 0.5 * (density_center + jnp.roll(density_center, shift=1, axis=0))
    flux_right = density_right_cell * velocity_right_cell * velocity_right_cell + amax_right * density_mid_right * (
        velocity_right_cell - velocity_mid_right
    )
    flux_left = density_left_cell * velocity_left_cell * velocity_left_cell - amax_left * density_mid_left * (
        velocity_left_cell - velocity_mid_left
    )
    return _scatter_face_divergence(flux_right, flux_left, benchmark=benchmark)


def _mc_field_edges(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    center = jnp.asarray(field, dtype=jnp.float64)
    minus = jnp.roll(center, shift=1, axis=0)
    plus = jnp.roll(center, shift=-1, axis=0)
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _broadcast_wave_speed(wave_speed: float | jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
    wave = jnp.asarray(wave_speed, dtype=jnp.float64)
    if wave.ndim == 0:
        wave = jnp.full_like(field, float(wave), dtype=jnp.float64)
    return wave


def _scatter_face_divergence(
    flux_right: jnp.ndarray,
    flux_left: jnp.ndarray,
    *,
    benchmark: DriftWaveBenchmark,
) -> jnp.ndarray:
    common_right = (benchmark.J + jnp.roll(benchmark.J, shift=-1, axis=0)) / (
        jnp.sqrt(benchmark.g22) + jnp.sqrt(jnp.roll(benchmark.g22, shift=-1, axis=0))
    )
    common_left = (benchmark.J + jnp.roll(benchmark.J, shift=1, axis=0)) / (
        jnp.sqrt(benchmark.g22) + jnp.sqrt(jnp.roll(benchmark.g22, shift=1, axis=0))
    )
    result_right = benchmark.rho_s0 * flux_right * common_right / (benchmark.dy * benchmark.J)
    result_left = benchmark.rho_s0 * flux_left * common_left / (benchmark.dy * benchmark.J)
    return result_right - result_left
