from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics


@dataclass(frozen=True)
class Fluid1DState:
    density: jnp.ndarray
    pressure: jnp.ndarray
    momentum: jnp.ndarray


@dataclass(frozen=True)
class Fluid1DRhsResult:
    density: jnp.ndarray
    pressure: jnp.ndarray
    momentum: jnp.ndarray


@dataclass(frozen=True)
class Fluid1DHistoryResult:
    density_history: jnp.ndarray
    pressure_history: jnp.ndarray
    momentum_history: jnp.ndarray


def initialize_mms_state(config: BoutConfig, *, section: str, mesh: StructuredMesh) -> Fluid1DState:
    return Fluid1DState(
        density=evaluate_field_option(config, f"N{section}", "solution", mesh=mesh, time=0.0),
        pressure=evaluate_field_option(config, f"P{section}", "solution", mesh=mesh, time=0.0),
        momentum=evaluate_field_option(config, f"NV{section}", "solution", mesh=mesh, time=0.0),
    )


def evaluate_field_option(
    config: BoutConfig,
    variable_name: str,
    option: str,
    *,
    mesh: StructuredMesh,
    time: float,
) -> jnp.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context(time=time))
    field = broadcast_to_field_shape(evaluator.resolve_option(variable_name, option), mesh)
    return apply_periodic_y_guards(field, mesh)


def compute_mms_rhs(
    config: BoutConfig,
    state: Fluid1DState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    time: float,
) -> Fluid1DRhsResult:
    guarded = Fluid1DState(
        density=apply_periodic_y_guards(state.density, mesh),
        pressure=apply_periodic_y_guards(state.pressure, mesh),
        momentum=apply_periodic_y_guards(state.momentum, mesh),
    )

    density = guarded.density
    pressure = guarded.pressure
    momentum = guarded.momentum
    velocity = momentum / (atomic_mass * density)
    temperature = pressure / density
    fastest_wave = jnp.sqrt(temperature / atomic_mass)

    density_rhs = -div_par_mod_periodic(density, velocity, fastest_wave, mesh=mesh, metrics=metrics)
    density_rhs = density_rhs + evaluate_field_option(config, f"N{section}", "source", mesh=mesh, time=time)

    pressure_rhs = -(5.0 / 3.0) * div_par_mod_periodic(pressure, velocity, fastest_wave, mesh=mesh, metrics=metrics)
    pressure_rhs = pressure_rhs + (2.0 / 3.0) * velocity * grad_par_periodic(pressure, mesh=mesh, metrics=metrics)
    pressure_rhs = pressure_rhs + evaluate_field_option(config, f"P{section}", "source", mesh=mesh, time=time)

    momentum_rhs = -atomic_mass * div_par_fvv_periodic(density, velocity, fastest_wave, mesh=mesh, metrics=metrics)
    momentum_rhs = momentum_rhs - grad_par_periodic(pressure, mesh=mesh, metrics=metrics)
    momentum_rhs = momentum_rhs + evaluate_field_option(config, f"NV{section}", "source", mesh=mesh, time=time)

    return Fluid1DRhsResult(
        density=density_rhs,
        pressure=pressure_rhs,
        momentum=momentum_rhs,
    )


def advance_mms_history(
    config: BoutConfig,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    timestep: float,
    steps: int,
    substeps: int,
    initial_state: Fluid1DState | None = None,
    start_time: float = 0.0,
) -> Fluid1DHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if substeps <= 0:
        raise ValueError("substeps must be positive")

    state = initial_state if initial_state is not None else initialize_mms_state(config, section=section, mesh=mesh)
    density_history = [state.density]
    pressure_history = [state.pressure]
    momentum_history = [state.momentum]
    current_time = float(start_time)
    sub_timestep = timestep / float(substeps)

    for _ in range(steps):
        for _ in range(substeps):
            state = _rk4_step(
                config,
                state,
                section=section,
                mesh=mesh,
                metrics=metrics,
                atomic_mass=atomic_mass,
                time=current_time,
                timestep=sub_timestep,
            )
            current_time += sub_timestep
        density_history.append(state.density)
        pressure_history.append(state.pressure)
        momentum_history.append(state.momentum)

    return Fluid1DHistoryResult(
        density_history=jnp.stack(density_history, axis=0),
        pressure_history=jnp.stack(pressure_history, axis=0),
        momentum_history=jnp.stack(momentum_history, axis=0),
    )


def apply_periodic_y_guards(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    result = jnp.asarray(field, dtype=jnp.float64)
    if mesh.myg <= 0:
        return result

    interior = result[:, mesh.ystart : mesh.yend + 1, :]
    for offset in range(mesh.myg):
        result = result.at[:, mesh.ystart - 1 - offset, :].set(interior[:, -(offset + 1), :])
        result = result.at[:, mesh.yend + 1 + offset, :].set(interior[:, offset, :])
    return result


def div_par_mod_periodic(
    field: jnp.ndarray,
    velocity: jnp.ndarray,
    wave_speed: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> jnp.ndarray:
    field_interior = _interior_y(field, mesh)
    velocity_interior = _interior_y(velocity, mesh)
    wave_interior = _interior_y(wave_speed, mesh)
    dy = _interior_y(metrics.dy, mesh)
    J = _interior_y(metrics.J, mesh)
    common = _face_common_factor(metrics, mesh)

    field_left_cell, field_right_cell = _mc_cell_edges(field_interior)
    velocity_left_cell, velocity_right_cell = _mc_cell_edges(velocity_interior)

    field_left = field_right_cell
    field_right = jnp.roll(field_left_cell, shift=-1, axis=1)
    velocity_left = velocity_right_cell
    velocity_right = jnp.roll(velocity_left_cell, shift=-1, axis=1)
    amax = _face_wave_speed(velocity_interior, wave_interior)

    flux = 0.5 * (
        field_left * velocity_left
        + field_right * velocity_right
        + amax * (field_left - field_right)
    )
    return _scatter_face_divergence(flux, dy=dy, J=J, common=common, mesh=mesh)


def div_par_fvv_periodic(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    wave_speed: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> jnp.ndarray:
    density_interior = _interior_y(density, mesh)
    velocity_interior = _interior_y(velocity, mesh)
    wave_interior = _interior_y(wave_speed, mesh)
    dy = _interior_y(metrics.dy, mesh)
    J = _interior_y(metrics.J, mesh)
    common = _face_common_factor(metrics, mesh)

    density_left_cell, density_right_cell = _mc_cell_edges(density_interior)
    velocity_left_cell, velocity_right_cell = _mc_cell_edges(velocity_interior)

    density_left = density_right_cell
    density_right = jnp.roll(density_left_cell, shift=-1, axis=1)
    velocity_left = velocity_right_cell
    velocity_right = jnp.roll(velocity_left_cell, shift=-1, axis=1)
    amax = _face_wave_speed(velocity_interior, wave_interior)

    flux = 0.5 * (
        density_left * velocity_left * velocity_left
        + density_right * velocity_right * velocity_right
        + amax * (density_left * velocity_left - density_right * velocity_right)
    )
    return _scatter_face_divergence(flux, dy=dy, J=J, common=common, mesh=mesh)


def grad_par_periodic(field: jnp.ndarray, *, mesh: StructuredMesh, metrics: StructuredMetrics) -> jnp.ndarray:
    dy = _interior_y(metrics.dy, mesh)
    field_interior = _interior_y(field, mesh)
    if not jnp.allclose(dy, dy[:, :1, :], rtol=1e-12, atol=1e-12):
        raise NotImplementedError("Native periodic fluid gradient currently requires uniform dy.")

    spacing = dy[:, :1, :]
    gradient = (jnp.roll(field_interior, shift=-1, axis=1) - jnp.roll(field_interior, shift=1, axis=1)) / (2.0 * spacing)
    result = jnp.zeros_like(field, dtype=jnp.float64)
    result = result.at[:, mesh.ystart : mesh.yend + 1, :].set(gradient)
    return apply_periodic_y_guards(result, mesh)


def _rk4_step(
    config: BoutConfig,
    state: Fluid1DState,
    *,
    section: str,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    atomic_mass: float,
    time: float,
    timestep: float,
) -> Fluid1DState:
    k1 = compute_mms_rhs(config, state, section=section, mesh=mesh, metrics=metrics, atomic_mass=atomic_mass, time=time)
    k2 = compute_mms_rhs(
        config,
        _add_state(state, k1, scale=0.5 * timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        time=time + 0.5 * timestep,
    )
    k3 = compute_mms_rhs(
        config,
        _add_state(state, k2, scale=0.5 * timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        time=time + 0.5 * timestep,
    )
    k4 = compute_mms_rhs(
        config,
        _add_state(state, k3, scale=timestep),
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        time=time + timestep,
    )

    next_state = Fluid1DState(
        density=state.density + (timestep / 6.0) * (k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density),
        pressure=state.pressure
        + (timestep / 6.0) * (k1.pressure + 2.0 * k2.pressure + 2.0 * k3.pressure + k4.pressure),
        momentum=state.momentum
        + (timestep / 6.0) * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum),
    )
    return Fluid1DState(
        density=apply_periodic_y_guards(next_state.density, mesh),
        pressure=apply_periodic_y_guards(next_state.pressure, mesh),
        momentum=apply_periodic_y_guards(next_state.momentum, mesh),
    )


def _add_state(state: Fluid1DState, rhs: Fluid1DRhsResult, *, scale: float) -> Fluid1DState:
    return Fluid1DState(
        density=state.density + scale * rhs.density,
        pressure=state.pressure + scale * rhs.pressure,
        momentum=state.momentum + scale * rhs.momentum,
    )


def _interior_y(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    return jnp.asarray(field, dtype=jnp.float64)[:, mesh.ystart : mesh.yend + 1, :]


def _mc_cell_edges(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    center = field
    minus = jnp.roll(center, shift=1, axis=1)
    plus = jnp.roll(center, shift=-1, axis=1)
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _minmod3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
    return jnp.where(same_sign, jnp.sign(a) * magnitude, 0.0)


def _face_common_factor(metrics: StructuredMetrics, mesh: StructuredMesh) -> jnp.ndarray:
    J = _interior_y(metrics.J, mesh)
    g22 = _interior_y(metrics.g_22, mesh)
    return (J + jnp.roll(J, shift=-1, axis=1)) / (jnp.sqrt(g22) + jnp.sqrt(jnp.roll(g22, shift=-1, axis=1)))


def _face_wave_speed(velocity: jnp.ndarray, wave_speed: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(
        jnp.maximum(wave_speed, jnp.roll(wave_speed, shift=-1, axis=1)),
        jnp.maximum(jnp.abs(velocity), jnp.abs(jnp.roll(velocity, shift=-1, axis=1))),
    )


def _scatter_face_divergence(
    flux: jnp.ndarray,
    *,
    dy: jnp.ndarray,
    J: jnp.ndarray,
    common: jnp.ndarray,
    mesh: StructuredMesh,
) -> jnp.ndarray:
    flux_right = flux * common / (dy * J)
    flux_left = jnp.roll(flux, shift=1, axis=1) * jnp.roll(common, shift=1, axis=1) / (dy * J)
    result = jnp.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=jnp.float64)
    result = result.at[:, mesh.ystart : mesh.yend + 1, :].set(flux_right - flux_left)
    return apply_periodic_y_guards(result, mesh)
