from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .array_backend import use_jax_backend
from .limiters import (
    monotonic_centered_edges_jax as _mc_edges_jax,
    monotonic_centered_edges_numpy as _mc_edges,
)
from .mesh import StructuredMesh
from .metrics import StructuredMetrics


_last_parallel_flow = np.zeros((1, 1, 1), dtype=np.float64)


def last_parallel_flow() -> np.ndarray:
    return np.asarray(_last_parallel_flow, dtype=np.float64)


def gradient_components(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if use_jax_backend(field):
        template = jnp.asarray(field, dtype=jnp.float64)
        dfdx_result = jnp.zeros_like(template, dtype=jnp.float64)
        dfdy_result = jnp.zeros_like(template, dtype=jnp.float64)
        dfdz_result = jnp.zeros_like(template, dtype=jnp.float64)
        dx = jnp.asarray(metrics.dx, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        dz = jnp.asarray(metrics.dz, dtype=jnp.float64)

        ix = slice(mesh.xstart, mesh.xend + 1)
        jy = slice(mesh.ystart, mesh.yend + 1)
        active_field = template[ix, jy, :]
        dfdx = (
            template[
                mesh.xstart + 1 : mesh.xend + 2,
                mesh.ystart : mesh.yend + 1,
                :,
            ]
            - template[
                mesh.xstart - 1 : mesh.xend,
                mesh.ystart : mesh.yend + 1,
                :,
            ]
        ) / (
            dx[ix, jy, :]
            + dx[mesh.xstart - 1 : mesh.xend, mesh.ystart : mesh.yend + 1, :]
        )
        dfdy = (
            template[ix, mesh.ystart + 1 : mesh.yend + 2, :]
            - template[ix, mesh.ystart - 1 : mesh.yend, :]
        ) / (
            dy[ix, jy, :] + dy[ix, mesh.ystart - 1 : mesh.yend, :]
        )
        dfdz = (
            jnp.roll(active_field, -1, axis=2) - jnp.roll(active_field, 1, axis=2)
        ) / (2.0 * dz[ix, jy, :])
        return (
            dfdx_result.at[ix, jy, :].set(dfdx),
            dfdy_result.at[ix, jy, :].set(dfdy),
            dfdz_result.at[ix, jy, :].set(dfdz),
        )

    dfdx_result = np.zeros_like(field, dtype=np.float64)
    dfdy_result = np.zeros_like(field, dtype=np.float64)
    dfdz_result = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)

    ix = slice(mesh.xstart, mesh.xend + 1)
    jy = slice(mesh.ystart, mesh.yend + 1)
    active_field = np.asarray(field[ix, jy, :], dtype=np.float64)
    dfdx = (
        np.asarray(
            field[
                mesh.xstart + 1 : mesh.xend + 2,
                mesh.ystart : mesh.yend + 1,
                :,
            ],
            dtype=np.float64,
        )
        - np.asarray(
            field[
                mesh.xstart - 1 : mesh.xend,
                mesh.ystart : mesh.yend + 1,
                :,
            ],
            dtype=np.float64,
        )
    ) / (
        np.asarray(dx[ix, jy, :], dtype=np.float64)
        + np.asarray(
            dx[mesh.xstart - 1 : mesh.xend, mesh.ystart : mesh.yend + 1, :],
            dtype=np.float64,
        )
    )
    dfdy = (
        np.asarray(field[ix, mesh.ystart + 1 : mesh.yend + 2, :], dtype=np.float64)
        - np.asarray(field[ix, mesh.ystart - 1 : mesh.yend, :], dtype=np.float64)
    ) / (
        np.asarray(dy[ix, jy, :], dtype=np.float64)
        + np.asarray(dy[ix, mesh.ystart - 1 : mesh.yend, :], dtype=np.float64)
    )
    dfdz = (np.roll(active_field, -1, axis=2) - np.roll(active_field, 1, axis=2)) / (
        2.0 * np.asarray(dz[ix, jy, :], dtype=np.float64)
    )
    dfdx_result[ix, jy, :] = dfdx
    dfdy_result[ix, jy, :] = dfdy
    dfdz_result[ix, jy, :] = dfdz
    return dfdx_result, dfdy_result, dfdz_result


def gradient_magnitude(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    dfdx, dfdy, dfdz = gradient_components(field, mesh=mesh, metrics=metrics)
    if use_jax_backend(dfdx, dfdy, dfdz):
        g11 = jnp.asarray(metrics.g11, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g22, dtype=jnp.float64)
        g33 = jnp.asarray(metrics.g33, dtype=jnp.float64)
        g23 = jnp.asarray(metrics.g23, dtype=jnp.float64)
        return jnp.sqrt(
            g11 * dfdx * dfdx
            + g22 * dfdy * dfdy
            + g33 * dfdz * dfdz
            + 2.0 * g23 * dfdy * dfdz
        )

    g11 = np.asarray(metrics.g11, dtype=np.float64)
    g22 = np.asarray(metrics.g22, dtype=np.float64)
    g33 = np.asarray(metrics.g33, dtype=np.float64)
    g23 = np.asarray(metrics.g23, dtype=np.float64)
    return np.sqrt(
        g11 * dfdx * dfdx
        + g22 * dfdy * dfdy
        + g33 * dfdz * dfdz
        + 2.0 * g23 * dfdy * dfdz
    )


def div_par_mod_open(
    field: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    global _last_parallel_flow
    if not fix_flux:
        raise NotImplementedError("Native neutral mixed advection currently supports fix_flux=True only.")
    if use_jax_backend(field, velocity, wave_speed):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        velocity_array = jnp.asarray(velocity, dtype=jnp.float64)
        wave_array = jnp.asarray(wave_speed, dtype=jnp.float64)
        result = jnp.zeros_like(field_array, dtype=jnp.float64)
        flow_ylow = jnp.zeros_like(field_array, dtype=jnp.float64)
        dx = jnp.asarray(metrics.dx, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        dz = jnp.asarray(metrics.dz, dtype=jnp.float64)
        J = jnp.asarray(metrics.J, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)

        ix = slice(mesh.xstart, mesh.xend + 1)
        jy = slice(mesh.ystart, mesh.yend + 1)
        jminus = slice(mesh.ystart - 1, mesh.yend)
        jplus = slice(mesh.ystart + 1, mesh.yend + 2)

        center = field_array[ix, jy, :]
        minus = field_array[ix, jminus, :]
        plus = field_array[ix, jplus, :]
        s_left, s_right = _mc_edges_jax(center, minus, plus)

        velocity_center = velocity_array[ix, jy, :]
        velocity_minus = velocity_array[ix, jminus, :]
        velocity_plus = velocity_array[ix, jplus, :]
        v_left, v_right = _mc_edges_jax(velocity_center, velocity_minus, velocity_plus)

        J_center = J[ix, jy, :]
        J_minus = J[ix, jminus, :]
        J_plus = J[ix, jplus, :]
        dy_center = dy[ix, jy, :]
        dy_minus = dy[ix, jminus, :]
        dy_plus = dy[ix, jplus, :]
        dx_center = dx[ix, jy, :]
        dx_plus = dx[ix, jplus, :]
        dz_center = dz[ix, jy, :]
        dz_plus = dz[ix, jplus, :]
        g22_center = g22[ix, jy, :]
        g22_minus = g22[ix, jminus, :]
        g22_plus = g22[ix, jplus, :]
        wave_center = wave_array[ix, jy, :]
        wave_minus = wave_array[ix, jminus, :]
        wave_plus = wave_array[ix, jplus, :]

        right_common = (J_center + J_plus) / (jnp.sqrt(g22_center) + jnp.sqrt(g22_plus))
        flux_factor_rc = right_common / (dy_center * J_center)
        flux_factor_rp = right_common / (dy_plus * J_plus)
        area_rp = right_common * dx_plus * dz_plus

        left_common = (J_center + J_minus) / (jnp.sqrt(g22_center) + jnp.sqrt(g22_minus))
        flux_factor_lc = left_common / (dy_center * J_center)
        flux_factor_lm = left_common / (dy_minus * J_minus)
        area_lc = left_common * dx_center * dz_center

        amax_right = jnp.maximum(
            jnp.maximum(wave_center, wave_plus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_plus)),
        )
        boundary_right = 0.5 * (center + plus) * 0.5 * (velocity_center + velocity_plus)
        interior_right = s_right * 0.5 * (v_right + amax_right)
        right_boundary_mask = jnp.zeros((1, center.shape[1], 1), dtype=bool).at[:, -1, :].set(True)
        right_flux = jnp.where(right_boundary_mask, boundary_right, interior_right)

        amax_left = jnp.maximum(
            jnp.maximum(wave_center, wave_minus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_minus)),
        )
        boundary_left = 0.5 * (center + minus) * 0.5 * (velocity_center + velocity_minus)
        interior_left = s_left * 0.5 * (v_left - amax_left)
        left_boundary_mask = jnp.zeros((1, center.shape[1], 1), dtype=bool).at[:, 0, :].set(True)
        left_flux = jnp.where(left_boundary_mask, boundary_left, interior_left)

        result = result.at[ix, jy, :].add(right_flux * flux_factor_rc)
        result = result.at[ix, jplus, :].add(-right_flux * flux_factor_rp)
        flow_ylow = flow_ylow.at[ix, jplus, :].add(right_flux * area_rp)

        result = result.at[ix, jy, :].add(-left_flux * flux_factor_lc)
        result = result.at[ix, jminus, :].add(left_flux * flux_factor_lm)
        flow_ylow = flow_ylow.at[ix, jy, :].add(left_flux * area_lc)
        try:
            # Preserve the legacy diagnostic side channel in eager mode; traced
            # JAX calls cannot host-copy this buffer.
            _last_parallel_flow = np.asarray(flow_ylow, dtype=np.float64)
        except Exception:
            pass
        return result

    result = np.zeros_like(field, dtype=np.float64)
    flow_ylow = np.zeros_like(field, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    ix = slice(mesh.xstart, mesh.xend + 1)
    jy = slice(mesh.ystart, mesh.yend + 1)
    jminus = slice(mesh.ystart - 1, mesh.yend)
    jplus = slice(mesh.ystart + 1, mesh.yend + 2)

    center = np.asarray(field[ix, jy, :], dtype=np.float64)
    minus = np.asarray(field[ix, jminus, :], dtype=np.float64)
    plus = np.asarray(field[ix, jplus, :], dtype=np.float64)
    s_left, s_right = _mc_edges(center, minus, plus)

    velocity_center = np.asarray(velocity[ix, jy, :], dtype=np.float64)
    velocity_minus = np.asarray(velocity[ix, jminus, :], dtype=np.float64)
    velocity_plus = np.asarray(velocity[ix, jplus, :], dtype=np.float64)
    v_left, v_right = _mc_edges(velocity_center, velocity_minus, velocity_plus)

    J_center = np.asarray(J[ix, jy, :], dtype=np.float64)
    J_minus = np.asarray(J[ix, jminus, :], dtype=np.float64)
    J_plus = np.asarray(J[ix, jplus, :], dtype=np.float64)
    dy_center = np.asarray(dy[ix, jy, :], dtype=np.float64)
    dy_minus = np.asarray(dy[ix, jminus, :], dtype=np.float64)
    dy_plus = np.asarray(dy[ix, jplus, :], dtype=np.float64)
    dx_center = np.asarray(dx[ix, jy, :], dtype=np.float64)
    dx_plus = np.asarray(dx[ix, jplus, :], dtype=np.float64)
    dz_center = np.asarray(dz[ix, jy, :], dtype=np.float64)
    dz_plus = np.asarray(dz[ix, jplus, :], dtype=np.float64)
    g22_center = np.asarray(g22[ix, jy, :], dtype=np.float64)
    g22_minus = np.asarray(g22[ix, jminus, :], dtype=np.float64)
    g22_plus = np.asarray(g22[ix, jplus, :], dtype=np.float64)
    wave_center = np.asarray(wave_speed[ix, jy, :], dtype=np.float64)
    wave_minus = np.asarray(wave_speed[ix, jminus, :], dtype=np.float64)
    wave_plus = np.asarray(wave_speed[ix, jplus, :], dtype=np.float64)

    right_common = (J_center + J_plus) / (np.sqrt(g22_center) + np.sqrt(g22_plus))
    flux_factor_rc = right_common / (dy_center * J_center)
    flux_factor_rp = right_common / (dy_plus * J_plus)
    area_rp = right_common * dx_plus * dz_plus

    left_common = (J_center + J_minus) / (np.sqrt(g22_center) + np.sqrt(g22_minus))
    flux_factor_lc = left_common / (dy_center * J_center)
    flux_factor_lm = left_common / (dy_minus * J_minus)
    area_lc = left_common * dx_center * dz_center

    amax_right = np.maximum.reduce((wave_center, wave_plus, np.abs(velocity_center), np.abs(velocity_plus)))
    boundary_right = 0.5 * (center + plus) * 0.5 * (velocity_center + velocity_plus)
    interior_right = s_right * 0.5 * (v_right + amax_right)
    right_boundary_mask = np.zeros((1, center.shape[1], 1), dtype=bool)
    right_boundary_mask[:, -1, :] = True
    right_flux = np.where(right_boundary_mask, boundary_right, interior_right)

    amax_left = np.maximum.reduce((wave_center, wave_minus, np.abs(velocity_center), np.abs(velocity_minus)))
    boundary_left = 0.5 * (center + minus) * 0.5 * (velocity_center + velocity_minus)
    interior_left = s_left * 0.5 * (v_left - amax_left)
    left_boundary_mask = np.zeros((1, center.shape[1], 1), dtype=bool)
    left_boundary_mask[:, 0, :] = True
    left_flux = np.where(left_boundary_mask, boundary_left, interior_left)

    result[ix, jy, :] += right_flux * flux_factor_rc
    result[ix, jplus, :] -= right_flux * flux_factor_rp
    flow_ylow[ix, jplus, :] += right_flux * area_rp

    result[ix, jy, :] -= left_flux * flux_factor_lc
    result[ix, jminus, :] += left_flux * flux_factor_lm
    flow_ylow[ix, jy, :] += left_flux * area_lc

    _last_parallel_flow = np.asarray(flow_ylow, dtype=np.float64)
    return result


def div_par_mod_open_active(
    field: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    """Return the active-domain slice of :func:`div_par_mod_open` directly."""

    if not fix_flux:
        raise NotImplementedError(
            "Native neutral mixed advection currently supports fix_flux=True only."
        )
    if use_jax_backend(field, velocity, wave_speed):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        velocity_array = jnp.asarray(velocity, dtype=jnp.float64)
        wave_array = jnp.asarray(wave_speed, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        J = jnp.asarray(metrics.J, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)

        ix = slice(mesh.xstart, mesh.xend + 1)
        jy = slice(mesh.ystart, mesh.yend + 1)
        jminus = slice(mesh.ystart - 1, mesh.yend)
        jplus = slice(mesh.ystart + 1, mesh.yend + 2)

        center = field_array[ix, jy, :]
        minus = field_array[ix, jminus, :]
        plus = field_array[ix, jplus, :]
        s_left, s_right = _mc_edges_jax(center, minus, plus)

        velocity_center = velocity_array[ix, jy, :]
        velocity_minus = velocity_array[ix, jminus, :]
        velocity_plus = velocity_array[ix, jplus, :]
        v_left, v_right = _mc_edges_jax(
            velocity_center,
            velocity_minus,
            velocity_plus,
        )

        J_center = J[ix, jy, :]
        J_minus = J[ix, jminus, :]
        J_plus = J[ix, jplus, :]
        dy_center = dy[ix, jy, :]
        dy_minus = dy[ix, jminus, :]
        dy_plus = dy[ix, jplus, :]
        g22_center = g22[ix, jy, :]
        g22_minus = g22[ix, jminus, :]
        g22_plus = g22[ix, jplus, :]
        wave_center = wave_array[ix, jy, :]
        wave_minus = wave_array[ix, jminus, :]
        wave_plus = wave_array[ix, jplus, :]

        right_common = (J_center + J_plus) / (
            jnp.sqrt(g22_center) + jnp.sqrt(g22_plus)
        )
        flux_factor_rc = right_common / (dy_center * J_center)
        flux_factor_rp = right_common / (dy_plus * J_plus)

        left_common = (J_center + J_minus) / (
            jnp.sqrt(g22_center) + jnp.sqrt(g22_minus)
        )
        flux_factor_lc = left_common / (dy_center * J_center)
        flux_factor_lm = left_common / (dy_minus * J_minus)

        amax_right = jnp.maximum(
            jnp.maximum(wave_center, wave_plus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_plus)),
        )
        boundary_right = 0.5 * (center + plus) * 0.5 * (
            velocity_center + velocity_plus
        )
        interior_right = s_right * 0.5 * (v_right + amax_right)
        right_boundary_mask = (
            jnp.zeros((1, center.shape[1], 1), dtype=bool)
            .at[:, -1, :]
            .set(True)
        )
        right_flux = jnp.where(right_boundary_mask, boundary_right, interior_right)

        amax_left = jnp.maximum(
            jnp.maximum(wave_center, wave_minus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_minus)),
        )
        boundary_left = 0.5 * (center + minus) * 0.5 * (
            velocity_center + velocity_minus
        )
        interior_left = s_left * 0.5 * (v_left - amax_left)
        left_boundary_mask = (
            jnp.zeros((1, center.shape[1], 1), dtype=bool)
            .at[:, 0, :]
            .set(True)
        )
        left_flux = jnp.where(left_boundary_mask, boundary_left, interior_left)

        result = right_flux * flux_factor_rc - left_flux * flux_factor_lc
        result = result.at[:, 1:, :].add(
            -right_flux[:, :-1, :] * flux_factor_rp[:, :-1, :]
        )
        result = result.at[:, :-1, :].add(
            left_flux[:, 1:, :] * flux_factor_lm[:, 1:, :]
        )
        return result

    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    ix = slice(mesh.xstart, mesh.xend + 1)
    jy = slice(mesh.ystart, mesh.yend + 1)
    jminus = slice(mesh.ystart - 1, mesh.yend)
    jplus = slice(mesh.ystart + 1, mesh.yend + 2)

    center = np.asarray(field[ix, jy, :], dtype=np.float64)
    minus = np.asarray(field[ix, jminus, :], dtype=np.float64)
    plus = np.asarray(field[ix, jplus, :], dtype=np.float64)
    s_left, s_right = _mc_edges(center, minus, plus)

    velocity_center = np.asarray(velocity[ix, jy, :], dtype=np.float64)
    velocity_minus = np.asarray(velocity[ix, jminus, :], dtype=np.float64)
    velocity_plus = np.asarray(velocity[ix, jplus, :], dtype=np.float64)
    v_left, v_right = _mc_edges(velocity_center, velocity_minus, velocity_plus)

    J_center = np.asarray(J[ix, jy, :], dtype=np.float64)
    J_minus = np.asarray(J[ix, jminus, :], dtype=np.float64)
    J_plus = np.asarray(J[ix, jplus, :], dtype=np.float64)
    dy_center = np.asarray(dy[ix, jy, :], dtype=np.float64)
    dy_minus = np.asarray(dy[ix, jminus, :], dtype=np.float64)
    dy_plus = np.asarray(dy[ix, jplus, :], dtype=np.float64)
    g22_center = np.asarray(g22[ix, jy, :], dtype=np.float64)
    g22_minus = np.asarray(g22[ix, jminus, :], dtype=np.float64)
    g22_plus = np.asarray(g22[ix, jplus, :], dtype=np.float64)
    wave_center = np.asarray(wave_speed[ix, jy, :], dtype=np.float64)
    wave_minus = np.asarray(wave_speed[ix, jminus, :], dtype=np.float64)
    wave_plus = np.asarray(wave_speed[ix, jplus, :], dtype=np.float64)

    right_common = (J_center + J_plus) / (np.sqrt(g22_center) + np.sqrt(g22_plus))
    flux_factor_rc = right_common / (dy_center * J_center)
    flux_factor_rp = right_common / (dy_plus * J_plus)

    left_common = (J_center + J_minus) / (np.sqrt(g22_center) + np.sqrt(g22_minus))
    flux_factor_lc = left_common / (dy_center * J_center)
    flux_factor_lm = left_common / (dy_minus * J_minus)

    amax_right = np.maximum.reduce(
        (wave_center, wave_plus, np.abs(velocity_center), np.abs(velocity_plus))
    )
    boundary_right = 0.5 * (center + plus) * 0.5 * (
        velocity_center + velocity_plus
    )
    interior_right = s_right * 0.5 * (v_right + amax_right)
    right_boundary_mask = np.zeros((1, center.shape[1], 1), dtype=bool)
    right_boundary_mask[:, -1, :] = True
    right_flux = np.where(right_boundary_mask, boundary_right, interior_right)

    amax_left = np.maximum.reduce(
        (wave_center, wave_minus, np.abs(velocity_center), np.abs(velocity_minus))
    )
    boundary_left = 0.5 * (center + minus) * 0.5 * (
        velocity_center + velocity_minus
    )
    interior_left = s_left * 0.5 * (v_left - amax_left)
    left_boundary_mask = np.zeros((1, center.shape[1], 1), dtype=bool)
    left_boundary_mask[:, 0, :] = True
    left_flux = np.where(left_boundary_mask, boundary_left, interior_left)

    result = right_flux * flux_factor_rc - left_flux * flux_factor_lc
    if result.shape[1] > 1:
        result[:, 1:, :] -= right_flux[:, :-1, :] * flux_factor_rp[:, :-1, :]
        result[:, :-1, :] += left_flux[:, 1:, :] * flux_factor_lm[:, 1:, :]
    return result


def div_par_fvv_open(
    density: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    if use_jax_backend(density, velocity, wave_speed):
        density_array = jnp.asarray(density, dtype=jnp.float64)
        velocity_array = jnp.asarray(velocity, dtype=jnp.float64)
        wave_array = jnp.asarray(wave_speed, dtype=jnp.float64)
        result = jnp.zeros_like(density_array, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        J = jnp.asarray(metrics.J, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)

        ix = slice(mesh.xstart, mesh.xend + 1)
        jy = slice(mesh.ystart, mesh.yend + 1)
        jminus = slice(mesh.ystart - 1, mesh.yend)
        jplus = slice(mesh.ystart + 1, mesh.yend + 2)

        density_center = density_array[ix, jy, :]
        density_minus = density_array[ix, jminus, :]
        density_plus = density_array[ix, jplus, :]
        velocity_center = velocity_array[ix, jy, :]
        velocity_minus = velocity_array[ix, jminus, :]
        velocity_plus = velocity_array[ix, jplus, :]
        wave_center = wave_array[ix, jy, :]
        wave_minus = wave_array[ix, jminus, :]
        wave_plus = wave_array[ix, jplus, :]

        s_left, s_right = _mc_edges_jax(density_center, density_minus, density_plus)
        v_left, v_right = _mc_edges_jax(velocity_center, velocity_minus, velocity_plus)

        J_center = J[ix, jy, :]
        J_minus = J[ix, jminus, :]
        J_plus = J[ix, jplus, :]
        dy_center = dy[ix, jy, :]
        dy_minus = dy[ix, jminus, :]
        dy_plus = dy[ix, jplus, :]
        g22_center = g22[ix, jy, :]
        g22_minus = g22[ix, jminus, :]
        g22_plus = g22[ix, jplus, :]

        right_common = (J_center + J_plus) / (jnp.sqrt(g22_center) + jnp.sqrt(g22_plus))
        flux_factor_rc = right_common / (dy_center * J_center)
        flux_factor_rp = right_common / (dy_plus * J_plus)

        left_common = (J_center + J_minus) / (jnp.sqrt(g22_center) + jnp.sqrt(g22_minus))
        flux_factor_lc = left_common / (dy_center * J_center)
        flux_factor_lm = left_common / (dy_minus * J_minus)

        v_mid_right = 0.5 * (velocity_center + velocity_plus)
        n_mid_right = 0.5 * (density_center + density_plus)
        amax_right = jnp.maximum(
            jnp.maximum(wave_center, wave_plus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_plus)),
        )
        boundary_right = n_mid_right * v_mid_right * v_mid_right
        interior_right = s_right * 0.5 * (v_right + amax_right) * v_right
        if not fix_flux:
            boundary_right = s_right * v_right * v_right + amax_right * n_mid_right * (v_right - v_mid_right)
        right_boundary_mask = jnp.zeros((1, density_center.shape[1], 1), dtype=bool).at[:, -1, :].set(True)
        right_flux = jnp.where(right_boundary_mask, boundary_right, interior_right)

        v_mid_left = 0.5 * (velocity_center + velocity_minus)
        n_mid_left = 0.5 * (density_center + density_minus)
        amax_left = jnp.maximum(
            jnp.maximum(wave_center, wave_minus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_minus)),
        )
        boundary_left = n_mid_left * v_mid_left * v_mid_left
        interior_left = s_left * 0.5 * (v_left - amax_left) * v_left
        if not fix_flux:
            boundary_left = s_left * v_left * v_left - amax_left * n_mid_left * (v_left - v_mid_left)
        left_boundary_mask = jnp.zeros((1, density_center.shape[1], 1), dtype=bool).at[:, 0, :].set(True)
        left_flux = jnp.where(left_boundary_mask, boundary_left, interior_left)

        result = result.at[ix, jy, :].add(right_flux * flux_factor_rc)
        result = result.at[ix, jplus, :].add(-right_flux * flux_factor_rp)
        result = result.at[ix, jy, :].add(-left_flux * flux_factor_lc)
        result = result.at[ix, jminus, :].add(left_flux * flux_factor_lm)
        return result

    result = np.zeros_like(density, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    ix = slice(mesh.xstart, mesh.xend + 1)
    jy = slice(mesh.ystart, mesh.yend + 1)
    jminus = slice(mesh.ystart - 1, mesh.yend)
    jplus = slice(mesh.ystart + 1, mesh.yend + 2)

    density_center = np.asarray(density[ix, jy, :], dtype=np.float64)
    density_minus = np.asarray(density[ix, jminus, :], dtype=np.float64)
    density_plus = np.asarray(density[ix, jplus, :], dtype=np.float64)
    velocity_center = np.asarray(velocity[ix, jy, :], dtype=np.float64)
    velocity_minus = np.asarray(velocity[ix, jminus, :], dtype=np.float64)
    velocity_plus = np.asarray(velocity[ix, jplus, :], dtype=np.float64)
    wave_center = np.asarray(wave_speed[ix, jy, :], dtype=np.float64)
    wave_minus = np.asarray(wave_speed[ix, jminus, :], dtype=np.float64)
    wave_plus = np.asarray(wave_speed[ix, jplus, :], dtype=np.float64)

    s_left, s_right = _mc_edges(density_center, density_minus, density_plus)
    v_left, v_right = _mc_edges(velocity_center, velocity_minus, velocity_plus)

    J_center = np.asarray(J[ix, jy, :], dtype=np.float64)
    J_minus = np.asarray(J[ix, jminus, :], dtype=np.float64)
    J_plus = np.asarray(J[ix, jplus, :], dtype=np.float64)
    dy_center = np.asarray(dy[ix, jy, :], dtype=np.float64)
    dy_minus = np.asarray(dy[ix, jminus, :], dtype=np.float64)
    dy_plus = np.asarray(dy[ix, jplus, :], dtype=np.float64)
    g22_center = np.asarray(g22[ix, jy, :], dtype=np.float64)
    g22_minus = np.asarray(g22[ix, jminus, :], dtype=np.float64)
    g22_plus = np.asarray(g22[ix, jplus, :], dtype=np.float64)

    right_common = (J_center + J_plus) / (np.sqrt(g22_center) + np.sqrt(g22_plus))
    flux_factor_rc = right_common / (dy_center * J_center)
    flux_factor_rp = right_common / (dy_plus * J_plus)

    left_common = (J_center + J_minus) / (np.sqrt(g22_center) + np.sqrt(g22_minus))
    flux_factor_lc = left_common / (dy_center * J_center)
    flux_factor_lm = left_common / (dy_minus * J_minus)

    v_mid_right = 0.5 * (velocity_center + velocity_plus)
    n_mid_right = 0.5 * (density_center + density_plus)
    amax_right = np.maximum.reduce((wave_center, wave_plus, np.abs(velocity_center), np.abs(velocity_plus)))
    boundary_right = n_mid_right * v_mid_right * v_mid_right
    interior_right = s_right * 0.5 * (v_right + amax_right) * v_right
    if not fix_flux:
        boundary_right = s_right * v_right * v_right + amax_right * n_mid_right * (v_right - v_mid_right)
    right_boundary_mask = np.zeros((1, density_center.shape[1], 1), dtype=bool)
    right_boundary_mask[:, -1, :] = True
    right_flux = np.where(right_boundary_mask, boundary_right, interior_right)

    v_mid_left = 0.5 * (velocity_center + velocity_minus)
    n_mid_left = 0.5 * (density_center + density_minus)
    amax_left = np.maximum.reduce((wave_center, wave_minus, np.abs(velocity_center), np.abs(velocity_minus)))
    boundary_left = n_mid_left * v_mid_left * v_mid_left
    interior_left = s_left * 0.5 * (v_left - amax_left) * v_left
    if not fix_flux:
        boundary_left = s_left * v_left * v_left - amax_left * n_mid_left * (v_left - v_mid_left)
    left_boundary_mask = np.zeros((1, density_center.shape[1], 1), dtype=bool)
    left_boundary_mask[:, 0, :] = True
    left_flux = np.where(left_boundary_mask, boundary_left, interior_left)

    result[ix, jy, :] += right_flux * flux_factor_rc
    result[ix, jplus, :] -= right_flux * flux_factor_rp
    result[ix, jy, :] -= left_flux * flux_factor_lc
    result[ix, jminus, :] += left_flux * flux_factor_lm
    return result


def div_par_fvv_open_active(
    density: np.ndarray,
    velocity: np.ndarray,
    wave_speed: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    fix_flux: bool = True,
) -> np.ndarray:
    """Return the active-domain slice of :func:`div_par_fvv_open` directly."""

    if use_jax_backend(density, velocity, wave_speed):
        density_array = jnp.asarray(density, dtype=jnp.float64)
        velocity_array = jnp.asarray(velocity, dtype=jnp.float64)
        wave_array = jnp.asarray(wave_speed, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        J = jnp.asarray(metrics.J, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)

        ix = slice(mesh.xstart, mesh.xend + 1)
        jy = slice(mesh.ystart, mesh.yend + 1)
        jminus = slice(mesh.ystart - 1, mesh.yend)
        jplus = slice(mesh.ystart + 1, mesh.yend + 2)

        density_center = density_array[ix, jy, :]
        density_minus = density_array[ix, jminus, :]
        density_plus = density_array[ix, jplus, :]
        velocity_center = velocity_array[ix, jy, :]
        velocity_minus = velocity_array[ix, jminus, :]
        velocity_plus = velocity_array[ix, jplus, :]
        wave_center = wave_array[ix, jy, :]
        wave_minus = wave_array[ix, jminus, :]
        wave_plus = wave_array[ix, jplus, :]

        s_left, s_right = _mc_edges_jax(density_center, density_minus, density_plus)
        v_left, v_right = _mc_edges_jax(
            velocity_center,
            velocity_minus,
            velocity_plus,
        )

        J_center = J[ix, jy, :]
        J_minus = J[ix, jminus, :]
        J_plus = J[ix, jplus, :]
        dy_center = dy[ix, jy, :]
        dy_minus = dy[ix, jminus, :]
        dy_plus = dy[ix, jplus, :]
        g22_center = g22[ix, jy, :]
        g22_minus = g22[ix, jminus, :]
        g22_plus = g22[ix, jplus, :]

        right_common = (J_center + J_plus) / (
            jnp.sqrt(g22_center) + jnp.sqrt(g22_plus)
        )
        flux_factor_rc = right_common / (dy_center * J_center)
        flux_factor_rp = right_common / (dy_plus * J_plus)

        left_common = (J_center + J_minus) / (
            jnp.sqrt(g22_center) + jnp.sqrt(g22_minus)
        )
        flux_factor_lc = left_common / (dy_center * J_center)
        flux_factor_lm = left_common / (dy_minus * J_minus)

        v_mid_right = 0.5 * (velocity_center + velocity_plus)
        n_mid_right = 0.5 * (density_center + density_plus)
        amax_right = jnp.maximum(
            jnp.maximum(wave_center, wave_plus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_plus)),
        )
        boundary_right = n_mid_right * v_mid_right * v_mid_right
        interior_right = s_right * 0.5 * (v_right + amax_right) * v_right
        if not fix_flux:
            boundary_right = (
                s_right * v_right * v_right
                + amax_right * n_mid_right * (v_right - v_mid_right)
            )
        right_boundary_mask = (
            jnp.zeros((1, density_center.shape[1], 1), dtype=bool)
            .at[:, -1, :]
            .set(True)
        )
        right_flux = jnp.where(right_boundary_mask, boundary_right, interior_right)

        v_mid_left = 0.5 * (velocity_center + velocity_minus)
        n_mid_left = 0.5 * (density_center + density_minus)
        amax_left = jnp.maximum(
            jnp.maximum(wave_center, wave_minus),
            jnp.maximum(jnp.abs(velocity_center), jnp.abs(velocity_minus)),
        )
        boundary_left = n_mid_left * v_mid_left * v_mid_left
        interior_left = s_left * 0.5 * (v_left - amax_left) * v_left
        if not fix_flux:
            boundary_left = (
                s_left * v_left * v_left
                - amax_left * n_mid_left * (v_left - v_mid_left)
            )
        left_boundary_mask = (
            jnp.zeros((1, density_center.shape[1], 1), dtype=bool)
            .at[:, 0, :]
            .set(True)
        )
        left_flux = jnp.where(left_boundary_mask, boundary_left, interior_left)

        result = right_flux * flux_factor_rc - left_flux * flux_factor_lc
        result = result.at[:, 1:, :].add(
            -right_flux[:, :-1, :] * flux_factor_rp[:, :-1, :]
        )
        result = result.at[:, :-1, :].add(
            left_flux[:, 1:, :] * flux_factor_lm[:, 1:, :]
        )
        return result

    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)

    ix = slice(mesh.xstart, mesh.xend + 1)
    jy = slice(mesh.ystart, mesh.yend + 1)
    jminus = slice(mesh.ystart - 1, mesh.yend)
    jplus = slice(mesh.ystart + 1, mesh.yend + 2)

    density_center = np.asarray(density[ix, jy, :], dtype=np.float64)
    density_minus = np.asarray(density[ix, jminus, :], dtype=np.float64)
    density_plus = np.asarray(density[ix, jplus, :], dtype=np.float64)
    velocity_center = np.asarray(velocity[ix, jy, :], dtype=np.float64)
    velocity_minus = np.asarray(velocity[ix, jminus, :], dtype=np.float64)
    velocity_plus = np.asarray(velocity[ix, jplus, :], dtype=np.float64)
    wave_center = np.asarray(wave_speed[ix, jy, :], dtype=np.float64)
    wave_minus = np.asarray(wave_speed[ix, jminus, :], dtype=np.float64)
    wave_plus = np.asarray(wave_speed[ix, jplus, :], dtype=np.float64)

    s_left, s_right = _mc_edges(density_center, density_minus, density_plus)
    v_left, v_right = _mc_edges(velocity_center, velocity_minus, velocity_plus)

    J_center = np.asarray(J[ix, jy, :], dtype=np.float64)
    J_minus = np.asarray(J[ix, jminus, :], dtype=np.float64)
    J_plus = np.asarray(J[ix, jplus, :], dtype=np.float64)
    dy_center = np.asarray(dy[ix, jy, :], dtype=np.float64)
    dy_minus = np.asarray(dy[ix, jminus, :], dtype=np.float64)
    dy_plus = np.asarray(dy[ix, jplus, :], dtype=np.float64)
    g22_center = np.asarray(g22[ix, jy, :], dtype=np.float64)
    g22_minus = np.asarray(g22[ix, jminus, :], dtype=np.float64)
    g22_plus = np.asarray(g22[ix, jplus, :], dtype=np.float64)

    right_common = (J_center + J_plus) / (np.sqrt(g22_center) + np.sqrt(g22_plus))
    flux_factor_rc = right_common / (dy_center * J_center)
    flux_factor_rp = right_common / (dy_plus * J_plus)

    left_common = (J_center + J_minus) / (np.sqrt(g22_center) + np.sqrt(g22_minus))
    flux_factor_lc = left_common / (dy_center * J_center)
    flux_factor_lm = left_common / (dy_minus * J_minus)

    v_mid_right = 0.5 * (velocity_center + velocity_plus)
    n_mid_right = 0.5 * (density_center + density_plus)
    amax_right = np.maximum.reduce(
        (wave_center, wave_plus, np.abs(velocity_center), np.abs(velocity_plus))
    )
    boundary_right = n_mid_right * v_mid_right * v_mid_right
    interior_right = s_right * 0.5 * (v_right + amax_right) * v_right
    if not fix_flux:
        boundary_right = (
            s_right * v_right * v_right
            + amax_right * n_mid_right * (v_right - v_mid_right)
        )
    right_boundary_mask = np.zeros((1, density_center.shape[1], 1), dtype=bool)
    right_boundary_mask[:, -1, :] = True
    right_flux = np.where(right_boundary_mask, boundary_right, interior_right)

    v_mid_left = 0.5 * (velocity_center + velocity_minus)
    n_mid_left = 0.5 * (density_center + density_minus)
    amax_left = np.maximum.reduce(
        (wave_center, wave_minus, np.abs(velocity_center), np.abs(velocity_minus))
    )
    boundary_left = n_mid_left * v_mid_left * v_mid_left
    interior_left = s_left * 0.5 * (v_left - amax_left) * v_left
    if not fix_flux:
        boundary_left = (
            s_left * v_left * v_left
            - amax_left * n_mid_left * (v_left - v_mid_left)
        )
    left_boundary_mask = np.zeros((1, density_center.shape[1], 1), dtype=bool)
    left_boundary_mask[:, 0, :] = True
    left_flux = np.where(left_boundary_mask, boundary_left, interior_left)

    result = right_flux * flux_factor_rc - left_flux * flux_factor_lc
    if result.shape[1] > 1:
        result[:, 1:, :] -= right_flux[:, :-1, :] * flux_factor_rp[:, :-1, :]
        result[:, :-1, :] += left_flux[:, 1:, :] * flux_factor_lm[:, 1:, :]
    return result


def div_par_k_grad_par_open(
    coefficient: np.ndarray,
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    boundary_flux: bool,
) -> np.ndarray:
    if use_jax_backend(coefficient, field):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        coefficient_array = jnp.asarray(coefficient, dtype=jnp.float64)
        result = jnp.zeros_like(field_array, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        J = jnp.asarray(metrics.J, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)
        xs = slice(mesh.xstart, mesh.xend + 1)

        if boundary_flux:
            up_cells = slice(mesh.ystart, mesh.yend + 1)
            down_cells = slice(mesh.ystart, mesh.yend + 1)
        else:
            up_cells = slice(mesh.ystart, mesh.yend)
            down_cells = slice(mesh.ystart + 1, mesh.yend + 1)

        if up_cells.start < up_cells.stop:
            coefficient_up = 0.5 * (
                coefficient_array[xs, up_cells, :]
                + coefficient_array[xs, up_cells.start + 1 : up_cells.stop + 1, :]
            )
            jacobian_up = 0.5 * (J[xs, up_cells, :] + J[xs, up_cells.start + 1 : up_cells.stop + 1, :])
            metric_up = 0.5 * (g22[xs, up_cells, :] + g22[xs, up_cells.start + 1 : up_cells.stop + 1, :])
            gradient_up = 2.0 * (
                field_array[xs, up_cells.start + 1 : up_cells.stop + 1, :] - field_array[xs, up_cells, :]
            ) / (dy[xs, up_cells, :] + dy[xs, up_cells.start + 1 : up_cells.stop + 1, :])
            flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
            result = result.at[xs, up_cells, :].add(flux_up / (dy[xs, up_cells, :] * J[xs, up_cells, :]))

        if down_cells.start < down_cells.stop:
            coefficient_down = 0.5 * (
                coefficient_array[xs, down_cells, :]
                + coefficient_array[xs, down_cells.start - 1 : down_cells.stop - 1, :]
            )
            jacobian_down = 0.5 * (J[xs, down_cells, :] + J[xs, down_cells.start - 1 : down_cells.stop - 1, :])
            metric_down = 0.5 * (g22[xs, down_cells, :] + g22[xs, down_cells.start - 1 : down_cells.stop - 1, :])
            gradient_down = 2.0 * (
                field_array[xs, down_cells, :] - field_array[xs, down_cells.start - 1 : down_cells.stop - 1, :]
            ) / (dy[xs, down_cells, :] + dy[xs, down_cells.start - 1 : down_cells.stop - 1, :])
            flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
            result = result.at[xs, down_cells, :].add(-flux_down / (dy[xs, down_cells, :] * J[xs, down_cells, :]))

        # A non-target lower end is a no-flow boundary on open-field recycling
        # decks. Only wrap the parallel operator when both Y ends are connected.
        has_connected_y_ends = not mesh.has_lower_y_target and not mesh.has_upper_y_target
        if not boundary_flux and has_connected_y_ends:
            lower = mesh.ystart
            connected = mesh.yend
            coefficient_down = 0.5 * (coefficient_array[xs, lower, :] + coefficient_array[xs, connected, :])
            jacobian_down = 0.5 * (J[xs, lower, :] + J[xs, connected, :])
            metric_down = 0.5 * (g22[xs, lower, :] + g22[xs, connected, :])
            gradient_down = 2.0 * (field_array[xs, lower, :] - field_array[xs, connected, :]) / (
                dy[xs, lower, :] + dy[xs, connected, :]
            )
            flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
            result = result.at[xs, lower, :].add(-flux_down / (dy[xs, lower, :] * J[xs, lower, :]))

        if not boundary_flux and has_connected_y_ends:
            upper = mesh.yend
            connected = mesh.ystart
            coefficient_up = 0.5 * (coefficient_array[xs, upper, :] + coefficient_array[xs, connected, :])
            jacobian_up = 0.5 * (J[xs, upper, :] + J[xs, connected, :])
            metric_up = 0.5 * (g22[xs, upper, :] + g22[xs, connected, :])
            gradient_up = 2.0 * (field_array[xs, connected, :] - field_array[xs, upper, :]) / (
                dy[xs, upper, :] + dy[xs, connected, :]
            )
            flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
            result = result.at[xs, upper, :].add(flux_up / (dy[xs, upper, :] * J[xs, upper, :]))

        return result

    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    xs = slice(mesh.xstart, mesh.xend + 1)

    if boundary_flux:
        up_cells = slice(mesh.ystart, mesh.yend + 1)
        down_cells = slice(mesh.ystart, mesh.yend + 1)
    else:
        up_cells = slice(mesh.ystart, mesh.yend)
        down_cells = slice(mesh.ystart + 1, mesh.yend + 1)

    if up_cells.start < up_cells.stop:
        coefficient_up = 0.5 * (coefficient[xs, up_cells, :] + coefficient[xs, up_cells.start + 1 : up_cells.stop + 1, :])
        jacobian_up = 0.5 * (J[xs, up_cells, :] + J[xs, up_cells.start + 1 : up_cells.stop + 1, :])
        metric_up = 0.5 * (g22[xs, up_cells, :] + g22[xs, up_cells.start + 1 : up_cells.stop + 1, :])
        gradient_up = 2.0 * (
            field[xs, up_cells.start + 1 : up_cells.stop + 1, :] - field[xs, up_cells, :]
        ) / (
            dy[xs, up_cells, :] + dy[xs, up_cells.start + 1 : up_cells.stop + 1, :]
        )
        flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
        result[xs, up_cells, :] += flux_up / (dy[xs, up_cells, :] * J[xs, up_cells, :])

    if down_cells.start < down_cells.stop:
        coefficient_down = 0.5 * (coefficient[xs, down_cells, :] + coefficient[xs, down_cells.start - 1 : down_cells.stop - 1, :])
        jacobian_down = 0.5 * (J[xs, down_cells, :] + J[xs, down_cells.start - 1 : down_cells.stop - 1, :])
        metric_down = 0.5 * (g22[xs, down_cells, :] + g22[xs, down_cells.start - 1 : down_cells.stop - 1, :])
        gradient_down = 2.0 * (
            field[xs, down_cells, :] - field[xs, down_cells.start - 1 : down_cells.stop - 1, :]
        ) / (
            dy[xs, down_cells, :] + dy[xs, down_cells.start - 1 : down_cells.stop - 1, :]
        )
        flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
        result[xs, down_cells, :] -= flux_down / (dy[xs, down_cells, :] * J[xs, down_cells, :])

    # A non-target lower end is a no-flow boundary on open-field recycling
    # decks. Only wrap the parallel operator when both Y ends are connected.
    has_connected_y_ends = not mesh.has_lower_y_target and not mesh.has_upper_y_target
    if not boundary_flux and has_connected_y_ends:
        lower = mesh.ystart
        connected = mesh.yend
        coefficient_down = 0.5 * (coefficient[xs, lower, :] + coefficient[xs, connected, :])
        jacobian_down = 0.5 * (J[xs, lower, :] + J[xs, connected, :])
        metric_down = 0.5 * (g22[xs, lower, :] + g22[xs, connected, :])
        gradient_down = 2.0 * (field[xs, lower, :] - field[xs, connected, :]) / (
            dy[xs, lower, :] + dy[xs, connected, :]
        )
        flux_down = coefficient_down * jacobian_down * gradient_down / metric_down
        result[xs, lower, :] -= flux_down / (dy[xs, lower, :] * J[xs, lower, :])

    if not boundary_flux and has_connected_y_ends:
        upper = mesh.yend
        connected = mesh.ystart
        coefficient_up = 0.5 * (coefficient[xs, upper, :] + coefficient[xs, connected, :])
        jacobian_up = 0.5 * (J[xs, upper, :] + J[xs, connected, :])
        metric_up = 0.5 * (g22[xs, upper, :] + g22[xs, connected, :])
        gradient_up = 2.0 * (field[xs, connected, :] - field[xs, upper, :]) / (
            dy[xs, upper, :] + dy[xs, connected, :]
        )
        flux_up = coefficient_up * jacobian_up * gradient_up / metric_up
        result[xs, upper, :] += flux_up / (dy[xs, upper, :] * J[xs, upper, :])

    return result


def div_a_grad_perp_flows(
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

    y_active = slice(mesh.ystart, mesh.yend + 1)

    x_left = slice(mesh.xstart - 1, mesh.xend + 1)
    x_right = slice(mesh.xstart, mesh.xend + 2)
    x_face_flux = (
        0.5
        * (coefficient[x_left, y_active, :] + coefficient[x_right, y_active, :])
        * (
            J[x_left, y_active, :] * g11[x_left, y_active, :]
            + J[x_right, y_active, :] * g11[x_right, y_active, :]
        )
        * (field[x_right, y_active, :] - field[x_left, y_active, :])
        / (dx[x_left, y_active, :] + dx[x_right, y_active, :])
    )
    result[x_left, y_active, :] += x_face_flux / (dx[x_left, y_active, :] * J[x_left, y_active, :])
    result[x_right, y_active, :] -= x_face_flux / (dx[x_right, y_active, :] * J[x_right, y_active, :])

    x_active = slice(mesh.xstart, mesh.xend + 1)
    coefficient_active = coefficient[x_active, y_active, :]
    field_active = field[x_active, y_active, :]
    J_active = J[x_active, y_active, :]
    g33_active = g33[x_active, y_active, :]
    dz_active = dz[x_active, y_active, :]

    coefficient_kp = np.roll(coefficient_active, -1, axis=2)
    field_kp = np.roll(field_active, -1, axis=2)
    J_kp = np.roll(J_active, -1, axis=2)
    g33_kp = np.roll(g33_active, -1, axis=2)
    z_face_flux = (
        0.25
        * (coefficient_active + coefficient_kp)
        * (J_active * g33_active + J_kp * g33_kp)
        * ((field_kp - field_active) / dz_active)
    )
    result[x_active, y_active, :] += z_face_flux / (J_active * dz_active)
    result[x_active, y_active, :] -= np.roll(z_face_flux, 1, axis=2) / (J_active * dz_active)

    return result


def grad_par_open(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    if use_jax_backend(field):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        result = jnp.zeros_like(field_array, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)
        x_slice = slice(mesh.xstart, mesh.xend + 1)
        y_slice = slice(mesh.ystart, mesh.yend + 1)
        active_gradient = (
            0.5
            * (
                field_array[x_slice, mesh.ystart + 1 : mesh.yend + 2, :]
                - field_array[x_slice, mesh.ystart - 1 : mesh.yend, :]
            )
            / (dy[x_slice, y_slice, :] * jnp.sqrt(g22[x_slice, y_slice, :]))
        )
        return result.at[x_slice, y_slice, :].set(active_gradient)

    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[x_slice, y_slice, :] = (
        0.5
        * (
            np.asarray(field[x_slice, mesh.ystart + 1 : mesh.yend + 2, :], dtype=np.float64)
            - np.asarray(field[x_slice, mesh.ystart - 1 : mesh.yend, :], dtype=np.float64)
        )
        / (
            np.asarray(dy[x_slice, y_slice, :], dtype=np.float64)
            * np.sqrt(np.asarray(g22[x_slice, y_slice, :], dtype=np.float64))
        )
    )
    return result


def grad_par_open_active(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    """Return the active-domain slice of :func:`grad_par_open` directly."""

    if use_jax_backend(field):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        g22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)
        x_slice = slice(mesh.xstart, mesh.xend + 1)
        y_slice = slice(mesh.ystart, mesh.yend + 1)
        return (
            0.5
            * (
                field_array[x_slice, mesh.ystart + 1 : mesh.yend + 2, :]
                - field_array[x_slice, mesh.ystart - 1 : mesh.yend, :]
            )
            / (dy[x_slice, y_slice, :] * jnp.sqrt(g22[x_slice, y_slice, :]))
        )

    dy = np.asarray(metrics.dy, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    return (
        0.5
        * (
            np.asarray(
                field[x_slice, mesh.ystart + 1 : mesh.yend + 2, :],
                dtype=np.float64,
            )
            - np.asarray(
                field[x_slice, mesh.ystart - 1 : mesh.yend, :],
                dtype=np.float64,
            )
        )
        / (
            np.asarray(dy[x_slice, y_slice, :], dtype=np.float64)
            * np.sqrt(np.asarray(g22[x_slice, y_slice, :], dtype=np.float64))
        )
    )
