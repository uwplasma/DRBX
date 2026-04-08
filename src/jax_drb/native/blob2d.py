from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from ..solver import FourierHelmholtzOperator, build_fourier_helmholtz_operator, solve_fourier_helmholtz
from .drift_wave import _compute_xz_exb_divergence
from .expression import ArrayExpressionEvaluator
from .metrics import StructuredMetrics
from .mesh import StructuredMesh, apply_zero_dirichlet_x_guards, broadcast_to_field_shape


@dataclass(frozen=True)
class Blob2DBenchmark:
    electron_temperature: float
    connection_length: float
    curvature_z: np.ndarray
    dx: float
    J: np.ndarray
    dz: np.ndarray
    right_face_j: float
    left_face_j: float
    density_bndry_flux: bool
    vorticity_bndry_flux: bool


@dataclass(frozen=True)
class Blob2DState:
    electron_density: np.ndarray
    vorticity: np.ndarray


@dataclass(frozen=True)
class BlobPotentialOperator:
    lower_diagonals: jnp.ndarray
    diagonals: jnp.ndarray
    upper_diagonals: jnp.ndarray
    rhs_scale: jnp.ndarray
    nz: int
    zlength: float


@dataclass(frozen=True)
class Blob2DRhsResult:
    electron_density: np.ndarray
    electron_pressure: np.ndarray
    potential: np.ndarray
    density_rhs: np.ndarray
    vorticity_rhs: np.ndarray


@dataclass(frozen=True)
class Blob2DHistoryResult:
    electron_density_history: np.ndarray
    electron_pressure_history: np.ndarray
    vorticity_history: np.ndarray
    potential_history: np.ndarray


def build_blob2d_benchmark(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: Mapping[str, float],
) -> Blob2DBenchmark:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    resolver = NumericResolver(config)
    electron_temperature = resolver.resolve("e", "temperature") / float(dataset_scalars["Tnorm"])
    curvature_raw = (
        broadcast_to_field_shape(evaluator.resolve_option("mesh", "bxcvz"), mesh)
        if config.has_option("mesh", "bxcvz")
        else jnp.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=jnp.float64)
    )
    connection_length = resolver.resolve("sheath_closure", "connection_length") / float(dataset_scalars["rho_s0"])
    density_bndry_flux = bool(config.parsed("e", "bndry_flux")) if config.has_option("e", "bndry_flux") else True
    vorticity_bndry_flux = bool(config.parsed("vorticity", "bndry_flux")) if config.has_option("vorticity", "bndry_flux") else False
    return Blob2DBenchmark(
        electron_temperature=electron_temperature,
        connection_length=connection_length,
        curvature_z=2.0 * float(dataset_scalars["rho_s0"]) ** 2 * jnp.asarray(curvature_raw, dtype=jnp.float64),
        dx=float(metrics.dx[mesh.xstart, mesh.ystart, 0]),
        J=jnp.asarray(metrics.J[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64),
        dz=jnp.asarray(metrics.dz[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64),
        right_face_j=0.5 * float(metrics.J[mesh.xstart, mesh.ystart, 0] + metrics.J[mesh.xstart + 1, mesh.ystart, 0]),
        left_face_j=0.5 * float(metrics.J[mesh.xstart, mesh.ystart, 0] + metrics.J[mesh.xstart - 1, mesh.ystart, 0]),
        density_bndry_flux=density_bndry_flux,
        vorticity_bndry_flux=vorticity_bndry_flux,
    )


def initialize_blob2d_state(config: BoutConfig, *, mesh: StructuredMesh) -> Blob2DState:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    density = jnp.asarray(broadcast_to_field_shape(evaluator.resolve_option("Ne", "function"), mesh), dtype=jnp.float64)
    density = jnp.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=jnp.float64)
    vorticity = jnp.zeros_like(density, dtype=jnp.float64)
    return Blob2DState(electron_density=density, vorticity=vorticity)


def build_blob2d_potential_operator(
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    average_atomic_mass: float,
) -> BlobPotentialOperator:
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_index = mesh.ystart

    operator = build_fourier_helmholtz_operator(
        dx=jnp.asarray(metrics.dx[x_slice, y_index, 0], dtype=jnp.float64),
        dz=jnp.asarray(metrics.dz[x_slice, y_index, 0], dtype=jnp.float64),
        g11=jnp.asarray(metrics.g11[x_slice, y_index, 0], dtype=jnp.float64),
        g33=jnp.asarray(metrics.g33[x_slice, y_index, 0], dtype=jnp.float64),
        rhs_scale=(jnp.asarray(metrics.Bxy[x_slice, y_index, 0], dtype=jnp.float64) ** 2) / float(average_atomic_mass),
        nz=mesh.nz,
    )
    return BlobPotentialOperator(**operator.__dict__)


def compute_blob2d_rhs(
    state: Blob2DState,
    *,
    mesh: StructuredMesh,
    benchmark: Blob2DBenchmark,
    operator: BlobPotentialOperator | None = None,
) -> Blob2DRhsResult:
    density = jnp.asarray(apply_zero_dirichlet_x_guards(state.electron_density, mesh), dtype=jnp.float64)
    vorticity = jnp.asarray(apply_zero_dirichlet_x_guards(state.vorticity, mesh), dtype=jnp.float64)
    pressure = density * benchmark.electron_temperature
    if operator is None:
        potential = jnp.zeros_like(density, dtype=jnp.float64)
        density_rhs = jnp.zeros_like(density, dtype=jnp.float64)
        vorticity_rhs = _curvature_drive(pressure, benchmark=benchmark)
    else:
        potential = solve_blob2d_potential(vorticity, mesh=mesh, operator=operator)
        density_rhs = -_compute_xz_exb_divergence(
            density,
            potential,
            mesh=mesh,
            benchmark=benchmark,
            bndry_flux=benchmark.density_bndry_flux,
        )
        density_rhs += density * potential / benchmark.connection_length
        vorticity_rhs = _curvature_drive(pressure, benchmark=benchmark)
        vorticity_rhs -= _compute_xz_exb_divergence(
            vorticity,
            potential,
            mesh=mesh,
            benchmark=benchmark,
            bndry_flux=benchmark.vorticity_bndry_flux,
        )
        vorticity_rhs += density * potential / benchmark.connection_length
    density_rhs = jnp.asarray(apply_zero_dirichlet_x_guards(density_rhs, mesh), dtype=jnp.float64)
    vorticity_rhs = jnp.asarray(apply_zero_dirichlet_x_guards(vorticity_rhs, mesh), dtype=jnp.float64)
    return Blob2DRhsResult(
        electron_density=density,
        electron_pressure=pressure,
        potential=potential,
        density_rhs=density_rhs,
        vorticity_rhs=vorticity_rhs,
    )


def advance_blob2d_history(
    initial_state: Blob2DState,
    *,
    mesh: StructuredMesh,
    benchmark: Blob2DBenchmark,
    operator: BlobPotentialOperator,
    timestep: float,
    steps: int,
    substeps: int,
    start_time: float = 0.0,
) -> Blob2DHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if substeps <= 0:
        raise ValueError("substeps must be positive")
    del start_time

    state = Blob2DState(
        electron_density=jnp.asarray(apply_zero_dirichlet_x_guards(initial_state.electron_density, mesh), dtype=jnp.float64),
        vorticity=jnp.asarray(apply_zero_dirichlet_x_guards(initial_state.vorticity, mesh), dtype=jnp.float64),
    )
    density_history = [jnp.asarray(state.electron_density, dtype=jnp.float64)]
    pressure_history = [jnp.asarray(state.electron_density * benchmark.electron_temperature, dtype=jnp.float64)]
    vorticity_history = [jnp.asarray(state.vorticity, dtype=jnp.float64)]
    potential_history = [jnp.zeros_like(state.electron_density, dtype=jnp.float64)]

    sub_timestep = timestep / float(substeps)
    for _ in range(steps):
        for _ in range(substeps):
            state = _rk4_step(state, mesh=mesh, benchmark=benchmark, operator=operator, timestep=sub_timestep)
        density = jnp.asarray(apply_zero_dirichlet_x_guards(state.electron_density, mesh), dtype=jnp.float64)
        vorticity = jnp.asarray(apply_zero_dirichlet_x_guards(state.vorticity, mesh), dtype=jnp.float64)
        potential = solve_blob2d_potential(vorticity, mesh=mesh, operator=operator)
        density_history.append(density)
        pressure_history.append(density * benchmark.electron_temperature)
        vorticity_history.append(vorticity)
        potential_history.append(potential)
        state = Blob2DState(electron_density=density, vorticity=vorticity)

    return Blob2DHistoryResult(
        electron_density_history=jnp.stack(density_history, axis=0),
        electron_pressure_history=jnp.stack(pressure_history, axis=0),
        vorticity_history=jnp.stack(vorticity_history, axis=0),
        potential_history=jnp.stack(potential_history, axis=0),
    )


def _curvature_drive(pressure: jnp.ndarray, *, benchmark: Blob2DBenchmark) -> jnp.ndarray:
    return benchmark.curvature_z * (
        jnp.roll(pressure, shift=-1, axis=-1) - jnp.roll(pressure, shift=1, axis=-1)
    ) / (2.0 * benchmark.dz)


def _rk4_step(
    state: Blob2DState,
    *,
    mesh: StructuredMesh,
    benchmark: Blob2DBenchmark,
    operator: BlobPotentialOperator,
    timestep: float,
) -> Blob2DState:
    k1 = compute_blob2d_rhs(state, mesh=mesh, benchmark=benchmark, operator=operator)
    k2 = compute_blob2d_rhs(_add_state(state, k1, scale=0.5 * timestep, mesh=mesh), mesh=mesh, benchmark=benchmark, operator=operator)
    k3 = compute_blob2d_rhs(_add_state(state, k2, scale=0.5 * timestep, mesh=mesh), mesh=mesh, benchmark=benchmark, operator=operator)
    k4 = compute_blob2d_rhs(_add_state(state, k3, scale=timestep, mesh=mesh), mesh=mesh, benchmark=benchmark, operator=operator)
    density = state.electron_density + (timestep / 6.0) * (
        k1.density_rhs + 2.0 * k2.density_rhs + 2.0 * k3.density_rhs + k4.density_rhs
    )
    vorticity = state.vorticity + (timestep / 6.0) * (
        k1.vorticity_rhs + 2.0 * k2.vorticity_rhs + 2.0 * k3.vorticity_rhs + k4.vorticity_rhs
    )
    return Blob2DState(
        electron_density=jnp.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=jnp.float64),
        vorticity=jnp.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=jnp.float64),
    )


def _add_state(state: Blob2DState, rhs: Blob2DRhsResult, *, scale: float, mesh: StructuredMesh) -> Blob2DState:
    density = state.electron_density + scale * rhs.density_rhs
    vorticity = state.vorticity + scale * rhs.vorticity_rhs
    return Blob2DState(
        electron_density=jnp.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=jnp.float64),
        vorticity=jnp.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=jnp.float64),
    )


def solve_blob2d_potential(
    vorticity: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    operator: BlobPotentialOperator,
) -> jnp.ndarray:
    guarded = jnp.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=jnp.float64)
    interior = jnp.asarray(guarded[mesh.xstart : mesh.xend + 1, mesh.ystart, :], dtype=jnp.float64)
    interior_phi = solve_fourier_helmholtz(interior, operator=operator)

    potential = jnp.zeros_like(guarded, dtype=jnp.float64)
    potential = potential.at[mesh.xstart : mesh.xend + 1, mesh.ystart, :].set(interior_phi)
    return _apply_blob_potential_boundaries(potential, mesh)


def _apply_blob_potential_boundaries(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    if mesh.mxg != 2:
        raise NotImplementedError("Native blob electrostatic potential boundaries currently require MXG = 2.")

    result = jnp.asarray(field, dtype=jnp.float64)
    for j in range(mesh.ystart, mesh.yend + 1):
        result = result.at[mesh.xstart - 1, j, :].set(-result[mesh.xstart, j, :])
        result = result.at[mesh.xend + 1, j, :].set(-result[mesh.xend, j, :])
        result = result.at[mesh.xstart - 2, j, :].set(result[mesh.xstart - 1, j, :])
        result = result.at[mesh.xend + 2, j, :].set(result[mesh.xend + 1, j, :])
    return result
