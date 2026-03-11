from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import expm

from .mesh import StructuredMesh, apply_field_boundaries
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class OneStepDiffusionResult:
    density: jnp.ndarray
    pressure: jnp.ndarray


@dataclass(frozen=True)
class DiffusionHistoryResult:
    density_history: jnp.ndarray
    pressure_history: jnp.ndarray


def advance_anomalous_diffusion_one_step(
    density: jnp.ndarray,
    pressure: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    anomalous_D: float,
    density_boundary: str,
    pressure_boundary: str,
    timestep: float,
) -> OneStepDiffusionResult:
    if mesh.nz != 1:
        raise NotImplementedError("Native one-step anomalous diffusion currently supports nz = 1 only.")
    if density_boundary.strip().lower() != "neumann" or pressure_boundary.strip().lower() != "neumann":
        raise NotImplementedError("Native one-step anomalous diffusion currently supports Neumann X boundaries only.")
    if not jnp.allclose(density, pressure, rtol=1e-12, atol=1e-12):
        raise NotImplementedError(
            "Native one-step anomalous diffusion currently requires identical density and pressure initial states."
        )
    if not jnp.allclose(metrics.g23, 0.0, rtol=1e-12, atol=1e-12):
        raise NotImplementedError("Native one-step anomalous diffusion currently supports g23 = 0 structured metrics only.")

    operator = _build_radial_diffusion_operator(mesh, metrics, anomalous_D)
    propagator = expm(operator * timestep)
    density_next = _advance_field_with_operator(density, propagator, mesh, boundary_kind=density_boundary)
    pressure_next = _advance_field_with_operator(pressure, propagator, mesh, boundary_kind=pressure_boundary)
    return OneStepDiffusionResult(density=density_next, pressure=pressure_next)


def advance_anomalous_diffusion_history(
    density: jnp.ndarray,
    pressure: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    anomalous_D: float,
    density_boundary: str,
    pressure_boundary: str,
    timestep: float,
    steps: int,
) -> DiffusionHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if mesh.nz != 1:
        raise NotImplementedError("Native anomalous diffusion history currently supports nz = 1 only.")
    if density_boundary.strip().lower() != "neumann" or pressure_boundary.strip().lower() != "neumann":
        raise NotImplementedError("Native anomalous diffusion history currently supports Neumann X boundaries only.")

    operator = _build_radial_diffusion_operator(mesh, metrics, anomalous_D)
    propagator = expm(operator * timestep)
    density_history = [jnp.asarray(density, dtype=jnp.float64)]
    pressure_history = [jnp.asarray(pressure, dtype=jnp.float64)]
    current_density = density_history[0]
    current_pressure = pressure_history[0]
    for _ in range(steps):
        current_density = _advance_field_with_operator(current_density, propagator, mesh, boundary_kind=density_boundary)
        current_pressure = _advance_field_with_operator(current_pressure, propagator, mesh, boundary_kind=pressure_boundary)
        density_history.append(current_density)
        pressure_history.append(current_pressure)
    return DiffusionHistoryResult(
        density_history=jnp.stack(density_history, axis=0),
        pressure_history=jnp.stack(pressure_history, axis=0),
    )


def _build_radial_diffusion_operator(
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    anomalous_D: float,
) -> jnp.ndarray:
    y_index = mesh.ystart
    z_index = 0
    dx = metrics.dx[:, y_index, z_index]
    J = metrics.J[:, y_index, z_index]
    g11 = metrics.g11[:, y_index, z_index]

    interior_nx = mesh.xend - mesh.xstart + 1
    matrix = jnp.zeros((interior_nx, interior_nx), dtype=jnp.float64)

    for global_index in range(mesh.xstart, mesh.xend):
        left = global_index - mesh.xstart
        right = left + 1
        face_coef = anomalous_D * (J[global_index] * g11[global_index] + J[global_index + 1] * g11[global_index + 1])
        face_coef /= dx[global_index] + dx[global_index + 1]

        matrix = matrix.at[left, left].add(-face_coef / (dx[global_index] * J[global_index]))
        matrix = matrix.at[left, right].add(face_coef / (dx[global_index] * J[global_index]))
        matrix = matrix.at[right, left].add(face_coef / (dx[global_index + 1] * J[global_index + 1]))
        matrix = matrix.at[right, right].add(-face_coef / (dx[global_index + 1] * J[global_index + 1]))

    return matrix


def _advance_field_with_operator(
    field: jnp.ndarray,
    propagator: jnp.ndarray,
    mesh: StructuredMesh,
    *,
    boundary_kind: str,
) -> jnp.ndarray:
    result = jnp.asarray(field, dtype=jnp.float64)
    interior = result[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, 0]
    updated = propagator @ interior
    result = result.at[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, 0].set(updated)
    result = apply_field_boundaries(result, mesh, x_boundary=boundary_kind)
    return result
