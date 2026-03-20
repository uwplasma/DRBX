from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from ..solver import FourierHelmholtzOperator, build_fourier_helmholtz_operator, solve_fourier_helmholtz
from .mesh import StructuredMesh, apply_zero_dirichlet_x_guards, communicate_y_guards
from .metrics import StructuredMetrics


VorticityOperator = FourierHelmholtzOperator


@dataclass(frozen=True)
class VorticityRhsResult:
    vorticity: jnp.ndarray
    potential: jnp.ndarray


@dataclass(frozen=True)
class VorticityHistoryResult:
    vorticity_history: jnp.ndarray
    potential_history: jnp.ndarray


def build_vorticity_operator(
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    average_atomic_mass: float,
) -> VorticityOperator:
    _require_supported_vorticity_mesh(mesh, metrics=metrics)

    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_index = mesh.ystart

    dx = jnp.asarray(metrics.dx[x_slice, y_index, 0], dtype=jnp.float64)
    dz = jnp.asarray(metrics.dz[x_slice, y_index, 0], dtype=jnp.float64)
    g11 = jnp.asarray(metrics.g11[x_slice, y_index, 0], dtype=jnp.float64)
    g33 = jnp.asarray(metrics.g33[x_slice, y_index, 0], dtype=jnp.float64)
    rhs_scale = jnp.asarray(metrics.Bxy[x_slice, y_index, 0], dtype=jnp.float64)
    rhs_scale = (rhs_scale * rhs_scale) / jnp.asarray(average_atomic_mass, dtype=jnp.float64)

    zlength = float(dz[0]) * float(mesh.nz)
    x_coef = g11 / (dx * dx)

    return VorticityOperator(
        **build_fourier_helmholtz_operator(
            dx=dx,
            dz=dz,
            g11=g11,
            g33=g33,
            rhs_scale=rhs_scale,
            nz=mesh.nz,
        ).__dict__,
    )


def apply_vorticity_boundaries(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    result = apply_zero_dirichlet_x_guards(field, mesh)
    if mesh.myg > 0:
        result = communicate_y_guards(result, mesh)
    return result


def solve_potential(
    vorticity: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    operator: VorticityOperator,
) -> jnp.ndarray:
    guarded = apply_vorticity_boundaries(vorticity, mesh)
    y_index = mesh.ystart
    interior = jnp.asarray(guarded[mesh.xstart : mesh.xend + 1, y_index, :], dtype=jnp.float64)
    interior_phi = solve_fourier_helmholtz(interior, operator=operator)

    potential = jnp.zeros_like(guarded, dtype=jnp.float64)
    potential = potential.at[mesh.xstart : mesh.xend + 1, y_index, :].set(interior_phi)
    potential = _apply_potential_boundaries(potential, mesh)
    if mesh.myg > 0:
        potential = communicate_y_guards(potential, mesh)
    return potential


def compute_vorticity_rhs(
    vorticity: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    operator: VorticityOperator,
) -> VorticityRhsResult:
    guarded = apply_vorticity_boundaries(vorticity, mesh)
    potential = solve_potential(guarded, mesh=mesh, operator=operator)
    result = jnp.zeros_like(guarded, dtype=jnp.float64)

    for j in range(mesh.ystart, mesh.yend + 1):
        for i in range(mesh.xstart, mesh.xend + 1):
            dx_i = jnp.asarray(metrics.dx[i, j, 0], dtype=jnp.float64)
            J_i = jnp.asarray(metrics.J[i, j, 0], dtype=jnp.float64)
            for k in range(mesh.nz):
                kp = (k + 1) % mesh.nz
                kpp = (kp + 1) % mesh.nz
                km = (k - 1 + mesh.nz) % mesh.nz
                kmm = (km - 1 + mesh.nz) % mesh.nz

                fmm = 0.25 * (
                    potential[i, j, k]
                    + potential[i - 1, j, k]
                    + potential[i, j, km]
                    + potential[i - 1, j, km]
                )
                fmp = 0.25 * (
                    potential[i, j, k]
                    + potential[i, j, kp]
                    + potential[i - 1, j, k]
                    + potential[i - 1, j, kp]
                )
                fpp = 0.25 * (
                    potential[i, j, k]
                    + potential[i, j, kp]
                    + potential[i + 1, j, k]
                    + potential[i + 1, j, kp]
                )
                fpm = 0.25 * (
                    potential[i, j, k]
                    + potential[i + 1, j, k]
                    + potential[i, j, km]
                    + potential[i + 1, j, km]
                )

                v_up = J_i * (fmp - fpp) / dx_i
                v_down = J_i * (fmm - fpm) / dx_i

                J_right = 0.5 * (
                    jnp.asarray(metrics.J[i, j, 0], dtype=jnp.float64)
                    + jnp.asarray(metrics.J[i + 1, j, 0], dtype=jnp.float64)
                )
                J_left = 0.5 * (
                    jnp.asarray(metrics.J[i, j, 0], dtype=jnp.float64)
                    + jnp.asarray(metrics.J[i - 1, j, 0], dtype=jnp.float64)
                )
                dz_i = jnp.asarray(metrics.dz[i, j, 0], dtype=jnp.float64)
                v_right = J_right * (fpp - fpm) / dz_i
                v_left = J_left * (fmp - fmm) / dz_i

                center = guarded[i, j, k]
                left = guarded[i - 1, j, k]
                right = guarded[i + 1, j, k]
                x_left_face, x_right_face = _mc_cell_edges(center, left, right)

                if i != mesh.xend:
                    dx_right = jnp.asarray(metrics.dx[i + 1, j, 0], dtype=jnp.float64)
                    J_right_cell = jnp.asarray(metrics.J[i + 1, j, 0], dtype=jnp.float64)
                    flux_right = jnp.where(v_right > 0.0, v_right * x_right_face, 0.0)
                    result = result.at[i, j, k].add(flux_right / (dx_i * J_i))
                    result = result.at[i + 1, j, k].add(-flux_right / (dx_right * J_right_cell))

                if i != mesh.xstart:
                    dx_left = jnp.asarray(metrics.dx[i - 1, j, 0], dtype=jnp.float64)
                    J_left_cell = jnp.asarray(metrics.J[i - 1, j, 0], dtype=jnp.float64)
                    flux_left = jnp.where(v_left < 0.0, v_left * x_left_face, 0.0)
                    result = result.at[i, j, k].add(-flux_left / (dx_i * J_i))
                    result = result.at[i - 1, j, k].add(flux_left / (dx_left * J_left_cell))

                z_left_face, z_right_face = _mc_cell_edges(center, guarded[i, j, km], guarded[i, j, kp])
                flux_up = jnp.where(v_up > 0.0, v_up * z_right_face / (J_i * dz_i), 0.0)
                result = result.at[i, j, k].add(flux_up)
                result = result.at[i, j, kp].add(-flux_up)

                flux_down = jnp.where(v_down < 0.0, v_down * z_left_face / (J_i * dz_i), 0.0)
                result = result.at[i, j, k].add(-flux_down)
                result = result.at[i, j, km].add(flux_down)

    return VorticityRhsResult(
        vorticity=-result,
        potential=potential,
    )


def advance_vorticity_history(
    initial_vorticity: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    operator: VorticityOperator,
    timestep: float,
    steps: int,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    mxstep: int = 20000,
) -> VorticityHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative")

    y0 = jnp.ravel(jnp.asarray(initial_vorticity[mesh.xstart : mesh.xend + 1, mesh.ystart, :], dtype=jnp.float64))
    times = jnp.asarray([timestep * index for index in range(steps + 1)], dtype=jnp.float64)

    def rhs_flat(interior_state: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        del time
        full_state = _assemble_state(interior_state, mesh)
        rhs = compute_vorticity_rhs(full_state, mesh=mesh, metrics=metrics, operator=operator).vorticity
        return jnp.ravel(rhs[mesh.xstart : mesh.xend + 1, mesh.ystart, :])

    interior_history = odeint(rhs_flat, y0, times, rtol=rtol, atol=atol, mxstep=mxstep)

    vorticity_history = []
    potential_history = []
    for index, interior_state in enumerate(interior_history):
        full_state = _assemble_state(interior_state, mesh)
        vorticity_history.append(full_state)
        if index == 0:
            potential_history.append(jnp.zeros_like(full_state, dtype=jnp.float64))
        else:
            potential_history.append(solve_potential(full_state, mesh=mesh, operator=operator))

    return VorticityHistoryResult(
        vorticity_history=jnp.stack(vorticity_history, axis=0),
        potential_history=jnp.stack(potential_history, axis=0),
    )


def _mc_cell_edges(center: jnp.ndarray, minus: jnp.ndarray, plus: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    slope = _minmod3(2.0 * (plus - center), 0.5 * (plus - minus), 2.0 * (center - minus))
    return center - 0.5 * slope, center + 0.5 * slope


def _minmod3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    magnitude = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
    return jnp.where(same_sign, jnp.sign(a) * magnitude, 0.0)


def _apply_potential_boundaries(field: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    if mesh.mxg != 2:
        raise NotImplementedError("Native electrostatic potential boundaries currently require MXG = 2.")

    result = jnp.asarray(field, dtype=jnp.float64)
    for j in range(mesh.ystart, mesh.yend + 1):
        result = result.at[mesh.xstart - 1, j, :].set(-result[mesh.xstart, j, :])
        result = result.at[mesh.xend + 1, j, :].set(-result[mesh.xend, j, :])
        result = result.at[mesh.xstart - 2, j, :].set(result[mesh.xstart - 1, j, :])
        result = result.at[mesh.xend + 2, j, :].set(result[mesh.xend + 1, j, :])
    return result


def _assemble_state(interior_state: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    interior = jnp.reshape(interior_state, (mesh.xend - mesh.xstart + 1, mesh.nz))
    full = jnp.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=jnp.float64)
    full = full.at[mesh.xstart : mesh.xend + 1, mesh.ystart, :].set(interior)
    return apply_vorticity_boundaries(full, mesh)


def _require_supported_vorticity_mesh(mesh: StructuredMesh, *, metrics: StructuredMetrics) -> None:
    if mesh.mxg != 2:
        raise NotImplementedError("Native electrostatic vorticity currently requires MXG = 2.")
    if mesh.ny != 1 or mesh.myg != 0:
        raise NotImplementedError("Native electrostatic vorticity currently requires ny = 1 with MYG = 0.")
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        raise NotImplementedError("Native electrostatic vorticity currently requires g23 = 0.")

    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_index = mesh.ystart
    for name, array in {
        "dx": metrics.dx[x_slice, y_index, 0],
        "dz": metrics.dz[x_slice, y_index, 0],
        "J": metrics.J[x_slice, y_index, 0],
        "g11": metrics.g11[x_slice, y_index, 0],
        "g33": metrics.g33[x_slice, y_index, 0],
        "Bxy": metrics.Bxy[x_slice, y_index, 0],
    }.items():
        values = np.asarray(array, dtype=np.float64)
        if not np.allclose(values, values[:1], rtol=1e-12, atol=1e-12):
            raise NotImplementedError(f"Native electrostatic vorticity currently requires uniform {name}.")
