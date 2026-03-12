from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
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
    lower_diagonals: tuple[np.ndarray, ...]
    diagonals: tuple[np.ndarray, ...]
    upper_diagonals: tuple[np.ndarray, ...]
    rhs_scale: np.ndarray
    nz: int


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
        else np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    )
    connection_length = resolver.resolve("sheath_closure", "connection_length") / float(dataset_scalars["rho_s0"])
    density_bndry_flux = bool(config.parsed("e", "bndry_flux")) if config.has_option("e", "bndry_flux") else True
    vorticity_bndry_flux = bool(config.parsed("vorticity", "bndry_flux")) if config.has_option("vorticity", "bndry_flux") else False
    return Blob2DBenchmark(
        electron_temperature=electron_temperature,
        connection_length=connection_length,
        curvature_z=2.0 * float(dataset_scalars["rho_s0"]) ** 2 * np.asarray(curvature_raw, dtype=np.float64),
        dx=float(metrics.dx[mesh.xstart, mesh.ystart, 0]),
        J=np.asarray(metrics.J[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64),
        dz=np.asarray(metrics.dz[mesh.xstart, mesh.ystart : mesh.yend + 1, :], dtype=np.float64),
        right_face_j=0.5 * float(metrics.J[mesh.xstart, mesh.ystart, 0] + metrics.J[mesh.xstart + 1, mesh.ystart, 0]),
        left_face_j=0.5 * float(metrics.J[mesh.xstart, mesh.ystart, 0] + metrics.J[mesh.xstart - 1, mesh.ystart, 0]),
        density_bndry_flux=density_bndry_flux,
        vorticity_bndry_flux=vorticity_bndry_flux,
    )


def initialize_blob2d_state(config: BoutConfig, *, mesh: StructuredMesh) -> Blob2DState:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    density = np.asarray(broadcast_to_field_shape(evaluator.resolve_option("Ne", "function"), mesh), dtype=np.float64)
    density = np.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=np.float64)
    vorticity = np.zeros_like(density, dtype=np.float64)
    return Blob2DState(electron_density=density, vorticity=vorticity)


def build_blob2d_potential_operator(
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    average_atomic_mass: float,
) -> BlobPotentialOperator:
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_index = mesh.ystart

    dx = np.asarray(metrics.dx[x_slice, y_index, 0], dtype=np.float64)
    dz = np.asarray(metrics.dz[x_slice, y_index, 0], dtype=np.float64)
    g11 = np.asarray(metrics.g11[x_slice, y_index, 0], dtype=np.float64)
    g33 = np.asarray(metrics.g33[x_slice, y_index, 0], dtype=np.float64)
    rhs_scale = np.asarray(metrics.Bxy[x_slice, y_index, 0], dtype=np.float64)
    rhs_scale = (rhs_scale * rhs_scale) / float(average_atomic_mass)

    zlength = float(dz[0]) * float(mesh.nz)
    x_coef = g11 / (dx * dx)

    lower_diagonals: list[np.ndarray] = []
    diagonals: list[np.ndarray] = []
    upper_diagonals: list[np.ndarray] = []
    for kz in range(mesh.nz // 2 + 1):
        wave_number = (2.0 * np.pi * kz) / zlength
        diagonal = -2.0 * x_coef - (wave_number * wave_number) * g33
        diagonal = np.asarray(diagonal, dtype=np.complex128)
        diagonal[0] -= x_coef[0]
        diagonal[-1] -= x_coef[-1]
        lower_diagonals.append(np.asarray(x_coef[1:], dtype=np.complex128))
        diagonals.append(diagonal)
        upper_diagonals.append(np.asarray(x_coef[:-1], dtype=np.complex128))

    return BlobPotentialOperator(
        lower_diagonals=tuple(lower_diagonals),
        diagonals=tuple(diagonals),
        upper_diagonals=tuple(upper_diagonals),
        rhs_scale=rhs_scale,
        nz=mesh.nz,
    )


def compute_blob2d_rhs(
    state: Blob2DState,
    *,
    mesh: StructuredMesh,
    benchmark: Blob2DBenchmark,
    operator: BlobPotentialOperator | None = None,
) -> Blob2DRhsResult:
    density = np.asarray(apply_zero_dirichlet_x_guards(state.electron_density, mesh), dtype=np.float64)
    vorticity = np.asarray(apply_zero_dirichlet_x_guards(state.vorticity, mesh), dtype=np.float64)
    pressure = density * benchmark.electron_temperature
    if operator is None:
        potential = np.zeros_like(density, dtype=np.float64)
        density_rhs = np.zeros_like(density, dtype=np.float64)
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
    density_rhs = np.asarray(apply_zero_dirichlet_x_guards(density_rhs, mesh), dtype=np.float64)
    vorticity_rhs = np.asarray(apply_zero_dirichlet_x_guards(vorticity_rhs, mesh), dtype=np.float64)
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
) -> Blob2DHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if substeps <= 0:
        raise ValueError("substeps must be positive")

    state = Blob2DState(
        electron_density=np.asarray(apply_zero_dirichlet_x_guards(initial_state.electron_density, mesh), dtype=np.float64),
        vorticity=np.asarray(apply_zero_dirichlet_x_guards(initial_state.vorticity, mesh), dtype=np.float64),
    )
    density_history = [np.asarray(state.electron_density, dtype=np.float64)]
    pressure_history = [np.asarray(state.electron_density * benchmark.electron_temperature, dtype=np.float64)]
    vorticity_history = [np.asarray(state.vorticity, dtype=np.float64)]
    potential_history = [np.zeros_like(state.electron_density, dtype=np.float64)]

    sub_timestep = timestep / float(substeps)
    for _ in range(steps):
        for _ in range(substeps):
            state = _rk4_step(state, mesh=mesh, benchmark=benchmark, operator=operator, timestep=sub_timestep)
        density = np.asarray(apply_zero_dirichlet_x_guards(state.electron_density, mesh), dtype=np.float64)
        vorticity = np.asarray(apply_zero_dirichlet_x_guards(state.vorticity, mesh), dtype=np.float64)
        potential = solve_blob2d_potential(vorticity, mesh=mesh, operator=operator)
        density_history.append(density)
        pressure_history.append(density * benchmark.electron_temperature)
        vorticity_history.append(vorticity)
        potential_history.append(potential)
        state = Blob2DState(electron_density=density, vorticity=vorticity)

    return Blob2DHistoryResult(
        electron_density_history=np.stack(density_history, axis=0),
        electron_pressure_history=np.stack(pressure_history, axis=0),
        vorticity_history=np.stack(vorticity_history, axis=0),
        potential_history=np.stack(potential_history, axis=0),
    )


def _curvature_drive(pressure: np.ndarray, *, benchmark: Blob2DBenchmark) -> np.ndarray:
    return benchmark.curvature_z * (
        np.roll(pressure, shift=-1, axis=-1) - np.roll(pressure, shift=1, axis=-1)
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
        electron_density=np.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=np.float64),
        vorticity=np.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=np.float64),
    )


def _add_state(state: Blob2DState, rhs: Blob2DRhsResult, *, scale: float, mesh: StructuredMesh) -> Blob2DState:
    density = state.electron_density + scale * rhs.density_rhs
    vorticity = state.vorticity + scale * rhs.vorticity_rhs
    return Blob2DState(
        electron_density=np.asarray(apply_zero_dirichlet_x_guards(density, mesh), dtype=np.float64),
        vorticity=np.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=np.float64),
    )


def solve_blob2d_potential(
    vorticity: np.ndarray,
    *,
    mesh: StructuredMesh,
    operator: BlobPotentialOperator,
) -> np.ndarray:
    guarded = np.asarray(apply_zero_dirichlet_x_guards(vorticity, mesh), dtype=np.float64)
    interior = np.asarray(guarded[mesh.xstart : mesh.xend + 1, mesh.ystart, :], dtype=np.float64)
    rhs_hat = np.fft.rfft(interior * operator.rhs_scale[:, None], axis=-1)

    interior_modes: list[np.ndarray] = []
    for kz, rhs_mode in enumerate(np.moveaxis(rhs_hat, -1, 0)):
        interior_modes.append(
            _solve_tridiagonal_complex(
                operator.lower_diagonals[kz],
                operator.diagonals[kz],
                operator.upper_diagonals[kz],
                rhs_mode,
            )
        )

    interior_hat = np.stack(interior_modes, axis=-1)
    interior_phi = np.asarray(np.fft.irfft(interior_hat, n=operator.nz, axis=-1), dtype=np.float64)

    potential = np.zeros_like(guarded, dtype=np.float64)
    potential[mesh.xstart : mesh.xend + 1, mesh.ystart, :] = interior_phi
    return _apply_blob_potential_boundaries(potential, mesh)


def _solve_tridiagonal_complex(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    diagonal_work = np.array(diagonal, dtype=np.complex128, copy=True)
    rhs_work = np.array(rhs, dtype=np.complex128, copy=True)
    upper_work = np.asarray(upper, dtype=np.complex128)
    lower_work = np.asarray(lower, dtype=np.complex128)

    for index in range(1, diagonal_work.shape[0]):
        factor = lower_work[index - 1] / diagonal_work[index - 1]
        diagonal_work[index] -= factor * upper_work[index - 1]
        rhs_work[index] -= factor * rhs_work[index - 1]

    solution = np.zeros_like(rhs_work, dtype=np.complex128)
    solution[-1] = rhs_work[-1] / diagonal_work[-1]
    for index in range(diagonal_work.shape[0] - 2, -1, -1):
        solution[index] = (rhs_work[index] - upper_work[index] * solution[index + 1]) / diagonal_work[index]
    return solution


def _apply_blob_potential_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    if mesh.mxg != 2:
        raise NotImplementedError("Native blob electrostatic potential boundaries currently require MXG = 2.")

    result = np.asarray(field, dtype=np.float64).copy()
    for j in range(mesh.ystart, mesh.yend + 1):
        result[mesh.xstart - 1, j, :] = -result[mesh.xstart, j, :]
        result[mesh.xend + 1, j, :] = -result[mesh.xend, j, :]
        result[mesh.xstart - 2, j, :] = result[mesh.xstart - 1, j, :]
        result[mesh.xend + 2, j, :] = result[mesh.xend + 1, j, :]
    return result
