from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from ..config.boutinp import BoutConfig
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class NeutralMixedState:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray


@dataclass(frozen=True)
class NeutralMixedRhsResult:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    diffusion: np.ndarray


def initialize_neutral_mixed_state(
    config: BoutConfig,
    *,
    section: str,
    mesh: StructuredMesh,
) -> NeutralMixedState:
    density = _evaluate_field_option(config, f"N{section}", mesh=mesh)
    pressure = _evaluate_field_option(config, f"P{section}", mesh=mesh)
    if config.has_section(f"NV{section}"):
        momentum = _evaluate_field_option(config, f"NV{section}", mesh=mesh)
    else:
        momentum = np.zeros_like(density, dtype=np.float64)
    return NeutralMixedState(
        density=_apply_scalar_boundaries(density, mesh),
        pressure=_apply_scalar_boundaries(pressure, mesh),
        momentum=_apply_momentum_boundaries(momentum, mesh),
    )


def compute_neutral_mixed_rhs(
    config: BoutConfig,
    state: NeutralMixedState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    meters_scale: float,
    tnorm: float,
) -> NeutralMixedRhsResult:
    atomic_mass = _section_scalar(config, section, "AA", default=1.0)
    flux_limit = _section_scalar(config, section, "flux_limit", default=0.2)

    density = _apply_scalar_boundaries(state.density, mesh)
    pressure = _apply_scalar_boundaries(state.pressure, mesh)
    momentum = _apply_momentum_boundaries(state.momentum, mesh)

    density_floor = _section_scalar(config, section, "density_floor", default=1.0e-8)
    temperature_floor = _section_scalar(config, section, "temperature_floor", default=0.1) / tnorm
    pressure_floor = density_floor * temperature_floor

    density_limited = np.maximum(density, density_floor)
    pressure_limited = np.maximum(pressure, pressure_floor)
    log_pressure = np.log(pressure_limited)

    diffusion = compute_neutral_mixed_diffusion(
        density_limited,
        pressure_limited,
        log_pressure,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        meters_scale=meters_scale,
        flux_limit=flux_limit,
    )

    density_rhs = _div_a_grad_perp_flows(diffusion * density_limited, log_pressure, mesh=mesh, metrics=metrics)
    pressure_rhs = (5.0 / 3.0) * _div_a_grad_perp_flows(
        diffusion * pressure_limited,
        log_pressure,
        mesh=mesh,
        metrics=metrics,
    )
    momentum_rhs = -_grad_par_wall(pressure, mesh=mesh, metrics=metrics)

    return NeutralMixedRhsResult(
        density=np.asarray(density_rhs, dtype=np.float64),
        pressure=np.asarray(pressure_rhs, dtype=np.float64),
        momentum=np.asarray(momentum_rhs, dtype=np.float64),
        diffusion=np.asarray(diffusion, dtype=np.float64),
    )


def compute_neutral_mixed_diffusion(
    density: np.ndarray,
    pressure: np.ndarray,
    log_pressure: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    meters_scale: float,
    flux_limit: float,
) -> np.ndarray:
    temperature = pressure / density
    thermal_speed = np.sqrt(temperature / atomic_mass)
    neutral_lmax = 0.1 / meters_scale
    raw_diffusion = thermal_speed * neutral_lmax

    if flux_limit > 0.0:
        grad_magnitude = _gradient_magnitude(log_pressure, mesh=mesh, metrics=metrics)
        diffusion_max = flux_limit * thermal_speed / (grad_magnitude + (1.0 / neutral_lmax))
        diffusion = raw_diffusion * diffusion_max / (raw_diffusion + diffusion_max)
    else:
        diffusion = raw_diffusion

    return _apply_dirichlet_like_scalar_boundaries(diffusion, mesh)


def _evaluate_field_option(
    config: BoutConfig,
    variable_name: str,
    *,
    mesh: StructuredMesh,
) -> np.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    if config.has_option(variable_name, "function"):
        value = evaluator.resolve_option(variable_name, "function")
    elif config.has_option(variable_name, "solution"):
        value = evaluator.resolve_option(variable_name, "solution")
    else:
        raise KeyError(f"Missing function or solution for {variable_name}.")
    return np.asarray(broadcast_to_field_shape(value, mesh), dtype=np.float64)


def _section_scalar(config: BoutConfig, section: str, name: str, *, default: float) -> float:
    if not config.has_option(section, name):
        return default
    return float(config.parsed(section, name))


def _apply_scalar_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    for offset in range(1, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = result[mesh.xstart - 1 + offset, y_slice, :]
        result[mesh.xend + offset, y_slice, :] = result[mesh.xend + 1 - offset, y_slice, :]
    return _apply_y_mirror_guards(result, mesh, sign=1.0)


def _apply_dirichlet_like_scalar_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[mesh.xstart - 1, y_slice, :] = -result[mesh.xstart, y_slice, :]
    result[mesh.xend + 1, y_slice, :] = -result[mesh.xend, y_slice, :]
    for offset in range(2, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = 0.0
        result[mesh.xend + offset, y_slice, :] = 0.0
    return _apply_y_mirror_guards(result, mesh, sign=1.0)


def _apply_momentum_boundaries(field: np.ndarray, mesh: StructuredMesh) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[mesh.xstart - 1, y_slice, :] = -result[mesh.xstart, y_slice, :]
    result[mesh.xend + 1, y_slice, :] = -result[mesh.xend, y_slice, :]
    for offset in range(2, mesh.mxg + 1):
        result[mesh.xstart - offset, y_slice, :] = 0.0
        result[mesh.xend + offset, y_slice, :] = 0.0
    return _apply_y_mirror_guards(result, mesh, sign=-1.0)


def _apply_y_mirror_guards(field: np.ndarray, mesh: StructuredMesh, *, sign: float) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    interior = result[:, mesh.ystart : mesh.yend + 1, :]
    for offset in range(mesh.myg):
        result[:, mesh.ystart - 1 - offset, :] = sign * interior[:, offset, :]
        result[:, mesh.yend + 1 + offset, :] = sign * interior[:, -(offset + 1), :]
    return result


def _gradient_magnitude(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                km = (k - 1 + mesh.nz) % mesh.nz
                kp = (k + 1) % mesh.nz
                dfdx = (field[i + 1, j, k] - field[i - 1, j, k]) / (dx[i, j, k] + dx[i - 1, j, k])
                dfdy = (field[i, j + 1, k] - field[i, j - 1, k]) / (dy[i, j, k] + dy[i, j - 1, k])
                dfdz = (field[i, j, kp] - field[i, j, km]) / (2.0 * dz[i, j, k])
                result[i, j, k] = np.sqrt(
                    g11[i, j, k] * dfdx * dfdx
                    + g33[i, j, k] * dfdz * dfdz
                    + (dfdy / J[i, j, k]) * (dfdy / J[i, j, k])
                )
    return result


def _div_a_grad_perp_flows(
    coefficient: np.ndarray,
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g23 = np.asarray(metrics.g23, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)

    if not np.allclose(g23, 0.0, rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native neutral mixed transport currently requires g23 = 0.")

    for i in range(mesh.xstart - 1, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                face_flux = (
                    0.5
                    * (coefficient[i, j, k] + coefficient[i + 1, j, k])
                    * (J[i, j, k] * g11[i, j, k] + J[i + 1, j, k] * g11[i + 1, j, k])
                    * (field[i + 1, j, k] - field[i, j, k])
                    / (dx[i, j, k] + dx[i + 1, j, k])
                )
                result[i, j, k] += face_flux / (dx[i, j, k] * J[i, j, k])
                result[i + 1, j, k] -= face_flux / (dx[i + 1, j, k] * J[i + 1, j, k])

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                face_flux = (
                    0.25
                    * (coefficient[i, j, k] + coefficient[i, j, kp])
                    * (J[i, j, k] * g33[i, j, k] + J[i, j, kp] * g33[i, j, kp])
                    * ((field[i, j, kp] - field[i, j, k]) / dz[i, j, k])
                )
                result[i, j, k] += face_flux / (J[i, j, k] * dz[i, j, k])
                result[i, j, kp] -= face_flux / (J[i, j, kp] * dz[i, j, kp])

    return result


def _grad_par_wall(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                spacing = dy[i, j, k] + dy[i, j - 1, k]
                result[i, j, k] = (field[i, j + 1, k] - field[i, j - 1, k]) / (spacing * J[i, j, k])
    return result
