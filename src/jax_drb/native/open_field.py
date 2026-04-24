from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend
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


@dataclass(frozen=True)
class TargetBoundaryGeometry:
    source_scale: jnp.ndarray


@dataclass(frozen=True)
class SimpleIonSheathResult:
    sheath_velocity: jnp.ndarray
    guard_velocity: jnp.ndarray
    guard_momentum: jnp.ndarray
    energy_source_delta: jnp.ndarray


def _use_numpy_backend(*values: object) -> bool:
    if use_jax_backend(*values):
        return False
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


def compute_simple_ion_sheath_boundary(
    *,
    sheath_density: jnp.ndarray,
    sheath_temperature: jnp.ndarray,
    electron_sheath_temperature: jnp.ndarray,
    interior_velocity: jnp.ndarray,
    interior_momentum: jnp.ndarray,
    atomic_mass: float,
    charge: float,
    gamma_i: float,
    sheath_ion_polytropic: float,
    direction: float,
    no_flow: bool = False,
    source_scale: jnp.ndarray | None = None,
) -> SimpleIonSheathResult:
    """Evaluate the simple Bohm-sheath ion guard and energy-source formula.

    ``direction`` is ``+1`` for the upper target and ``-1`` for the lower
    target. The returned ``energy_source_delta`` is the additive contribution to
    the cell-centered ion energy source, so it carries the same sign convention
    as the existing recycling RHS.
    """

    if _use_numpy_backend(
        sheath_density,
        sheath_temperature,
        electron_sheath_temperature,
        interior_velocity,
        interior_momentum,
        source_scale,
    ):
        density = np.asarray(sheath_density, dtype=np.float64)
        temperature = np.asarray(sheath_temperature, dtype=np.float64)
        electron_temperature = np.asarray(electron_sheath_temperature, dtype=np.float64)
        velocity = np.asarray(interior_velocity, dtype=np.float64)
        momentum = np.asarray(interior_momentum, dtype=np.float64)
        c_i_sq = np.maximum(
            (float(sheath_ion_polytropic) * temperature + float(charge) * electron_temperature)
            / float(atomic_mass),
            0.0,
        )
        sonic_speed = np.sqrt(c_i_sq)
        if float(direction) >= 0.0:
            sheath_velocity = np.maximum(velocity, sonic_speed)
        else:
            sheath_velocity = np.minimum(velocity, -sonic_speed)
        if no_flow:
            sheath_velocity = np.zeros_like(sheath_velocity)
        guard_velocity = 2.0 * sheath_velocity - velocity
        guard_momentum = 2.0 * float(atomic_mass) * density * sheath_velocity - momentum
        q = float(gamma_i) * temperature * density * sheath_velocity
        q = q - (2.5 * temperature + 0.5 * float(atomic_mass) * np.square(sheath_velocity)) * density * sheath_velocity
        scale = 1.0 if source_scale is None else np.asarray(source_scale, dtype=np.float64)
        energy_source_delta = -float(direction) * q * scale
        return SimpleIonSheathResult(
            sheath_velocity=sheath_velocity,
            guard_velocity=guard_velocity,
            guard_momentum=guard_momentum,
            energy_source_delta=energy_source_delta,
        )

    density = jnp.asarray(sheath_density, dtype=jnp.float64)
    temperature = jnp.asarray(sheath_temperature, dtype=jnp.float64)
    electron_temperature = jnp.asarray(electron_sheath_temperature, dtype=jnp.float64)
    velocity = jnp.asarray(interior_velocity, dtype=jnp.float64)
    momentum = jnp.asarray(interior_momentum, dtype=jnp.float64)
    c_i_sq = jnp.maximum(
        (float(sheath_ion_polytropic) * temperature + float(charge) * electron_temperature) / float(atomic_mass),
        0.0,
    )
    sonic_speed = jnp.sqrt(c_i_sq)
    sheath_velocity = jnp.where(
        float(direction) >= 0.0,
        jnp.maximum(velocity, sonic_speed),
        jnp.minimum(velocity, -sonic_speed),
    )
    if no_flow:
        sheath_velocity = jnp.zeros_like(sheath_velocity)
    guard_velocity = 2.0 * sheath_velocity - velocity
    guard_momentum = 2.0 * float(atomic_mass) * density * sheath_velocity - momentum
    q = float(gamma_i) * temperature * density * sheath_velocity
    q = q - (2.5 * temperature + 0.5 * float(atomic_mass) * jnp.square(sheath_velocity)) * density * sheath_velocity
    scale = 1.0 if source_scale is None else jnp.asarray(source_scale, dtype=jnp.float64)
    energy_source_delta = -float(direction) * q * scale
    return SimpleIonSheathResult(
        sheath_velocity=sheath_velocity,
        guard_velocity=guard_velocity,
        guard_momentum=guard_momentum,
        energy_source_delta=energy_source_delta,
    )


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
    lower_geometry: TargetBoundaryGeometry | None = None,
    upper_geometry: TargetBoundaryGeometry | None = None,
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
                source_scale=(
                    None
                    if lower_geometry is None
                    else np.asarray(lower_geometry.source_scale, dtype=np.float64)
                ),
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
                source_scale=(
                    None
                    if upper_geometry is None
                    else np.asarray(upper_geometry.source_scale, dtype=np.float64)
                ),
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
            source_scale=(None if lower_geometry is None else lower_geometry.source_scale),
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
            source_scale=(None if upper_geometry is None else upper_geometry.source_scale),
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
    source_scale: jnp.ndarray | None = None,
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
    if source_scale is None:
        daparsheath = 0.25 * (j_i + j_g) / (jnp.sqrt(g_i) + jnp.sqrt(g_g)) * (dx_i + dx_g) * (dz_i + dz_g)
        volume = j_i * dx_i * dy_i * dz_i
        source_scale = daparsheath / volume
    else:
        source_scale = jnp.asarray(source_scale, dtype=jnp.float64)
    flow_per_volume = float(target_multiplier) * flux * source_scale

    # The current Hermès reference output for target recycling records the
    # fixed returning-neutral energy in Ed_target_recycle and the neutral
    # pressure source, even when fast-recycle options are present in the deck.
    recycle_energy_flow = flow_per_volume * (1.0 - float(fast_recycle_fraction)) * float(target_energy)
    return flow_per_volume, recycle_energy_flow


def build_target_boundary_geometry(
    *,
    J: jnp.ndarray,
    dy: jnp.ndarray,
    dx: jnp.ndarray,
    dz: jnp.ndarray,
    g_22: jnp.ndarray,
    y_index: int,
    guard_index: int,
) -> TargetBoundaryGeometry:
    if _use_numpy_backend(J, dy, dx, dz, g_22):
        j_i = np.asarray(J[:, y_index, :], dtype=np.float64)
        j_g = np.asarray(J[:, guard_index, :], dtype=np.float64)
        dy_i = np.asarray(dy[:, y_index, :], dtype=np.float64)
        dx_i = np.asarray(dx[:, y_index, :], dtype=np.float64)
        dx_g = np.asarray(dx[:, guard_index, :], dtype=np.float64)
        dz_i = np.asarray(dz[:, y_index, :], dtype=np.float64)
        dz_g = np.asarray(dz[:, guard_index, :], dtype=np.float64)
        g_i = np.asarray(g_22[:, y_index, :], dtype=np.float64)
        g_g = np.asarray(g_22[:, guard_index, :], dtype=np.float64)
        daparsheath = 0.25 * (j_i + j_g) / (np.sqrt(g_i) + np.sqrt(g_g)) * (dx_i + dx_g) * (dz_i + dz_g)
        volume = j_i * dx_i * dy_i * dz_i
        return TargetBoundaryGeometry(source_scale=daparsheath / volume)
    j_i = jnp.asarray(J[:, y_index, :], dtype=jnp.float64)
    j_g = jnp.asarray(J[:, guard_index, :], dtype=jnp.float64)
    dy_i = jnp.asarray(dy[:, y_index, :], dtype=jnp.float64)
    dx_i = jnp.asarray(dx[:, y_index, :], dtype=jnp.float64)
    dx_g = jnp.asarray(dx[:, guard_index, :], dtype=jnp.float64)
    dz_i = jnp.asarray(dz[:, y_index, :], dtype=jnp.float64)
    dz_g = jnp.asarray(dz[:, guard_index, :], dtype=jnp.float64)
    g_i = jnp.asarray(g_22[:, y_index, :], dtype=jnp.float64)
    g_g = jnp.asarray(g_22[:, guard_index, :], dtype=jnp.float64)
    daparsheath = 0.25 * (j_i + j_g) / (jnp.sqrt(g_i) + jnp.sqrt(g_g)) * (dx_i + dx_g) * (dz_i + dz_g)
    volume = j_i * dx_i * dy_i * dz_i
    return TargetBoundaryGeometry(source_scale=daparsheath / volume)
