from __future__ import annotations

from dataclasses import dataclass

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .mesh import StructuredMesh


@dataclass(frozen=True)
class ElectronForceBalanceResult:
    epar: jnp.ndarray
    force_density: jnp.ndarray


@dataclass(frozen=True)
class RecyclingSourceResult:
    density_source: jnp.ndarray
    energy_source: jnp.ndarray
    target_density_source: jnp.ndarray
    target_energy_source: jnp.ndarray


def _use_numpy_backend(*values: object) -> bool:
    return any(isinstance(value, np.ndarray) for value in values if value is not None)


def limit_free(fm: jnp.ndarray, fc: jnp.ndarray, mode: int | float) -> jnp.ndarray:
    mode_value = float(mode)
    if _use_numpy_backend(fm, fc):
        fm_np = np.asarray(fm, dtype=np.float64)
        fc_np = np.asarray(fc, dtype=np.float64)
        extrapolated = np.divide(
            fc_np * fc_np,
            fm_np,
            out=np.array(fc_np, dtype=np.float64, copy=True),
            where=fm_np >= 1.0e-10,
        )
        if mode_value == 0.0:
            return np.where(fm_np < fc_np, fc_np, extrapolated)
        if mode_value == 1.0:
            return extrapolated
        if mode_value == 2.0:
            return 2.0 * fc_np - fm_np
        raise ValueError(f"Unsupported boundary mode {mode!r}")
    fm = jnp.asarray(fm, dtype=jnp.float64)
    fc = jnp.asarray(fc, dtype=jnp.float64)
    if mode_value == 0.0:
        extrapolated = jnp.where(fm < 1.0e-10, fc, (fc * fc) / fm)
        return jnp.where(fm < fc, fc, extrapolated)
    if mode_value == 1.0:
        return jnp.where(fm < 1.0e-10, fc, (fc * fc) / fm)
    if mode_value == 2.0:
        return 2.0 * fc - fm
    raise ValueError(f"Unsupported boundary mode {mode!r}")


def apply_noflow_scalar_guards(
    field: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> jnp.ndarray:
    if _use_numpy_backend(field):
        result = np.array(field, dtype=np.float64, copy=True)
        if mesh.myg <= 0:
            return result
        if lower_y:
            result[:, mesh.ystart - 1, :] = result[:, mesh.ystart, :]
        if upper_y:
            result[:, mesh.yend + 1, :] = result[:, mesh.yend, :]
        return result
    result = jnp.asarray(field, dtype=jnp.float64)
    if mesh.myg <= 0:
        return result
    if lower_y:
        result = result.at[:, mesh.ystart - 1, :].set(result[:, mesh.ystart, :])
    if upper_y:
        result = result.at[:, mesh.yend + 1, :].set(result[:, mesh.yend, :])
    return result


def apply_noflow_flow_guards(
    field: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> jnp.ndarray:
    if _use_numpy_backend(field):
        result = np.array(field, dtype=np.float64, copy=True)
        if mesh.myg <= 0:
            return result
        if lower_y:
            result[:, mesh.ystart - 1, :] = -result[:, mesh.ystart, :]
        if upper_y:
            result[:, mesh.yend + 1, :] = -result[:, mesh.yend, :]
        return result
    result = jnp.asarray(field, dtype=jnp.float64)
    if mesh.myg <= 0:
        return result
    if lower_y:
        result = result.at[:, mesh.ystart - 1, :].set(-result[:, mesh.ystart, :])
    if upper_y:
        result = result.at[:, mesh.yend + 1, :].set(-result[:, mesh.yend, :])
    return result


def grad_par_y(field: jnp.ndarray, *, mesh: StructuredMesh, dy: jnp.ndarray) -> jnp.ndarray:
    if _use_numpy_backend(field, dy):
        field_np = np.asarray(field, dtype=np.float64)
        dy_np = np.asarray(dy, dtype=np.float64)
        result = np.zeros_like(field_np, dtype=np.float64)
        interior = field_np[:, mesh.ystart : mesh.yend + 1, :]
        if interior.shape[1] == 1:
            return result
        left = field_np[:, mesh.ystart - 1 : mesh.yend, :]
        right = field_np[:, mesh.ystart + 1 : mesh.yend + 2, :]
        dy_interior = dy_np[:, mesh.ystart : mesh.yend + 1, :]
        result[:, mesh.ystart : mesh.yend + 1, :] = (right - left) / (2.0 * dy_interior)
        return result
    result = jnp.zeros_like(field, dtype=jnp.float64)
    interior = jnp.asarray(field[:, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64)
    dy_interior = jnp.asarray(dy[:, mesh.ystart : mesh.yend + 1, :], dtype=jnp.float64)
    if interior.shape[1] == 1:
        return result
    left = jnp.asarray(field[:, mesh.ystart - 1 : mesh.yend, :], dtype=jnp.float64)
    right = jnp.asarray(field[:, mesh.ystart + 1 : mesh.yend + 2, :], dtype=jnp.float64)
    gradient = (right - left) / (2.0 * dy_interior)
    result = result.at[:, mesh.ystart : mesh.yend + 1, :].set(gradient)
    return result


def compute_electron_force_balance(
    electron_pressure: jnp.ndarray,
    electron_density: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    dy: jnp.ndarray,
    electron_momentum_source: jnp.ndarray | None = None,
    density_floor: float = 1.0e-5,
) -> ElectronForceBalanceResult:
    if _use_numpy_backend(electron_pressure, electron_density, dy, electron_momentum_source):
        pressure = np.asarray(electron_pressure, dtype=np.float64)
        density = np.asarray(electron_density, dtype=np.float64)
        force_density = -np.asarray(grad_par_y(pressure, mesh=mesh, dy=np.asarray(dy, dtype=np.float64)), dtype=np.float64)
        if electron_momentum_source is not None:
            force_density = force_density + np.asarray(electron_momentum_source, dtype=np.float64)
        epar = force_density / np.maximum(density, float(density_floor))
        return ElectronForceBalanceResult(epar=epar, force_density=force_density)
    pressure = jnp.asarray(electron_pressure, dtype=jnp.float64)
    density = jnp.asarray(electron_density, dtype=jnp.float64)
    force_density = -grad_par_y(pressure, mesh=mesh, dy=jnp.asarray(dy, dtype=jnp.float64))
    if electron_momentum_source is not None:
        force_density = force_density + jnp.asarray(electron_momentum_source, dtype=jnp.float64)
    epar = force_density / jnp.maximum(density, float(density_floor))
    return ElectronForceBalanceResult(epar=epar, force_density=force_density)


def apply_parallel_electric_force(
    density: jnp.ndarray,
    *,
    charge: float,
    epar: jnp.ndarray,
    existing_source: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if _use_numpy_backend(density, epar, existing_source):
        source = float(charge) * np.asarray(density, dtype=np.float64) * np.asarray(epar, dtype=np.float64)
        if existing_source is not None:
            source = source + np.asarray(existing_source, dtype=np.float64)
        return source
    source = charge * jnp.asarray(density, dtype=jnp.float64) * jnp.asarray(epar, dtype=jnp.float64)
    if existing_source is not None:
        source = source + jnp.asarray(existing_source, dtype=jnp.float64)
    return source


def compute_target_recycling_sources(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    temperature: jnp.ndarray,
    *,
    mesh: StructuredMesh,
    J: jnp.ndarray,
    dy: jnp.ndarray,
    dx: jnp.ndarray,
    dz: jnp.ndarray,
    g_22: jnp.ndarray,
    target_multiplier: float,
    target_energy: float,
    gamma_i: float,
    target_fast_recycle_fraction: float = 0.0,
    target_fast_recycle_energy_factor: float = 0.0,
    lower_y: bool = True,
    upper_y: bool = True,
) -> RecyclingSourceResult:
    if _use_numpy_backend(density, velocity, temperature, J, dy, dx, dz, g_22):
        density_np = np.asarray(density, dtype=np.float64)
        velocity_np = np.asarray(velocity, dtype=np.float64)
        temperature_np = np.asarray(temperature, dtype=np.float64)
        j_np = np.asarray(J, dtype=np.float64)
        dy_np = np.asarray(dy, dtype=np.float64)
        dx_np = np.asarray(dx, dtype=np.float64)
        dz_np = np.asarray(dz, dtype=np.float64)
        g22_np = np.asarray(g_22, dtype=np.float64)

        density_source = np.zeros_like(density_np, dtype=np.float64)
        energy_source = np.zeros_like(density_np, dtype=np.float64)

        if lower_y and mesh.myg > 0:
            lower_density_source, lower_energy_source = _target_boundary_sources(
                density_np,
                velocity_np,
                temperature_np,
                J=j_np,
                dy=dy_np,
                dx=dx_np,
                dz=dz_np,
                g_22=g22_np,
                y_index=mesh.ystart,
                guard_index=mesh.ystart - 1,
                sign=-1.0,
                target_multiplier=target_multiplier,
                target_energy=target_energy,
                gamma_i=gamma_i,
                fast_recycle_fraction=target_fast_recycle_fraction,
                fast_recycle_energy_factor=target_fast_recycle_energy_factor,
            )
            density_source[:, mesh.ystart, :] = lower_density_source
            energy_source[:, mesh.ystart, :] = lower_energy_source

        if upper_y and mesh.myg > 0:
            upper_density_source, upper_energy_source = _target_boundary_sources(
                density_np,
                velocity_np,
                temperature_np,
                J=j_np,
                dy=dy_np,
                dx=dx_np,
                dz=dz_np,
                g_22=g22_np,
                y_index=mesh.yend,
                guard_index=mesh.yend + 1,
                sign=1.0,
                target_multiplier=target_multiplier,
                target_energy=target_energy,
                gamma_i=gamma_i,
                fast_recycle_fraction=target_fast_recycle_fraction,
                fast_recycle_energy_factor=target_fast_recycle_energy_factor,
            )
            density_source[:, mesh.yend, :] += upper_density_source
            energy_source[:, mesh.yend, :] += upper_energy_source

        return RecyclingSourceResult(
            density_source=density_source,
            energy_source=energy_source,
            target_density_source=density_source,
            target_energy_source=energy_source,
        )
    density = jnp.asarray(density, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    J = jnp.asarray(J, dtype=jnp.float64)
    dy = jnp.asarray(dy, dtype=jnp.float64)
    dx = jnp.asarray(dx, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    g_22 = jnp.asarray(g_22, dtype=jnp.float64)

    density_source = jnp.zeros_like(density, dtype=jnp.float64)
    energy_source = jnp.zeros_like(density, dtype=jnp.float64)

    if lower_y and mesh.myg > 0:
        lower_density_source, lower_energy_source = _target_boundary_sources(
            density,
            velocity,
            temperature,
            J=J,
            dy=dy,
            dx=dx,
            dz=dz,
            g_22=g_22,
            y_index=mesh.ystart,
            guard_index=mesh.ystart - 1,
            sign=-1.0,
            target_multiplier=target_multiplier,
            target_energy=target_energy,
            gamma_i=gamma_i,
            fast_recycle_fraction=target_fast_recycle_fraction,
            fast_recycle_energy_factor=target_fast_recycle_energy_factor,
        )
        density_source = density_source.at[:, mesh.ystart, :].set(lower_density_source)
        energy_source = energy_source.at[:, mesh.ystart, :].set(lower_energy_source)

    if upper_y and mesh.myg > 0:
        upper_density_source, upper_energy_source = _target_boundary_sources(
            density,
            velocity,
            temperature,
            J=J,
            dy=dy,
            dx=dx,
            dz=dz,
            g_22=g_22,
            y_index=mesh.yend,
            guard_index=mesh.yend + 1,
            sign=1.0,
            target_multiplier=target_multiplier,
            target_energy=target_energy,
            gamma_i=gamma_i,
            fast_recycle_fraction=target_fast_recycle_fraction,
            fast_recycle_energy_factor=target_fast_recycle_energy_factor,
        )
        density_source = density_source.at[:, mesh.yend, :].add(upper_density_source)
        energy_source = energy_source.at[:, mesh.yend, :].add(upper_energy_source)

    return RecyclingSourceResult(
        density_source=density_source,
        energy_source=energy_source,
        target_density_source=density_source,
        target_energy_source=energy_source,
    )


def _target_boundary_sources(
    density: jnp.ndarray,
    velocity: jnp.ndarray,
    temperature: jnp.ndarray,
    *,
    J: jnp.ndarray,
    dy: jnp.ndarray,
    dx: jnp.ndarray,
    dz: jnp.ndarray,
    g_22: jnp.ndarray,
    y_index: int,
    guard_index: int,
    sign: float,
    target_multiplier: float,
    target_energy: float,
    gamma_i: float,
    fast_recycle_fraction: float,
    fast_recycle_energy_factor: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_i = density[:, y_index, :]
    n_g = density[:, guard_index, :]
    v_i = velocity[:, y_index, :]
    v_g = velocity[:, guard_index, :]
    t_i = temperature[:, y_index, :]
    t_g = temperature[:, guard_index, :]
    j_i = J[:, y_index, :]
    j_g = J[:, guard_index, :]
    dy_i = dy[:, y_index, :]
    dx_i = dx[:, y_index, :]
    dx_g = dx[:, guard_index, :]
    dz_i = dz[:, y_index, :]
    dz_g = dz[:, guard_index, :]
    g_i = g_22[:, y_index, :]
    g_g = g_22[:, guard_index, :]

    flux = sign * 0.25 * (n_i + n_g) * (v_i + v_g)
    flux = jnp.maximum(flux, 0.0)
    daparsheath = 0.25 * (j_i + j_g) / (jnp.sqrt(g_i) + jnp.sqrt(g_g)) * (dx_i + dx_g) * (dz_i + dz_g)
    volume = j_i * dx_i * dy_i * dz_i
    flow = float(target_multiplier) * flux * daparsheath

    nisheath = 0.5 * (n_i + n_g)
    tisheath = 0.5 * (t_i + t_g)
    visheath = 0.5 * (v_i + v_g)
    sheath_ion_heat_flow = jnp.abs(float(gamma_i) * nisheath * tisheath * visheath * daparsheath / volume)
    recycle_energy_flow = (
        sheath_ion_heat_flow
        * float(target_multiplier)
        * float(fast_recycle_energy_factor)
        * float(fast_recycle_fraction)
        + flow * (1.0 - float(fast_recycle_fraction)) * float(target_energy)
    )
    return flow / volume, recycle_energy_flow / volume
