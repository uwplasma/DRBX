from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .neutral_mixed import _div_par_fvv_open, _div_par_mod_open, _grad_par_open
from .neutral_mixed_operators import (
    div_par_fvv_open_active as _div_par_fvv_open_active,
    div_par_mod_open_active as _div_par_mod_open_active,
    grad_par_open_active as _grad_par_open_active,
)
from .open_field import apply_parallel_electric_force
from .recycling_targets import (
    grad_par_electron_force_balance_open,
    grad_par_electron_force_balance_open_active,
)


@dataclass(frozen=True)
class ElectronPressureRhsTerms:
    explicit_pressure_source: np.ndarray
    parallel_divergence: np.ndarray
    parallel_advection: np.ndarray
    energy_source: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class IonRhsTerms:
    density_source: np.ndarray
    density_transport: np.ndarray
    density_total: np.ndarray
    explicit_pressure_source: np.ndarray
    parallel_divergence: np.ndarray
    parallel_advection: np.ndarray
    energy_source: np.ndarray
    pressure_total: np.ndarray
    momentum_advection: np.ndarray
    pressure_gradient: np.ndarray
    momentum_source: np.ndarray
    momentum_error: np.ndarray
    momentum_total: np.ndarray


@dataclass(frozen=True)
class NeutralRhsTerms:
    density_source: np.ndarray
    density_transport: np.ndarray
    density_total: np.ndarray
    explicit_pressure_source: np.ndarray
    parallel_divergence: np.ndarray
    parallel_advection: np.ndarray
    energy_source: np.ndarray
    pressure_total: np.ndarray
    momentum_advection: np.ndarray
    pressure_gradient: np.ndarray
    momentum_source: np.ndarray
    momentum_error: np.ndarray
    momentum_total: np.ndarray


@dataclass(frozen=True)
class ElectronParallelForceTerms:
    force_density: np.ndarray
    epar: np.ndarray
    ion_momentum_source: dict[str, np.ndarray]


def soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    minimum_value = float(minimum)
    if use_jax_backend(value):
        value_array = jnp.maximum(jnp.asarray(value, dtype=jnp.float64), 0.0)
        return value_array + minimum_value * jnp.exp(-value_array / minimum_value)
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


def assemble_electron_pressure_rhs_terms(
    *,
    explicit_pressure_source: np.ndarray | None,
    electron_pressure: np.ndarray,
    electron_velocity: np.ndarray,
    electron_fastest_wave: np.ndarray,
    electron_energy_source: np.ndarray | None,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> ElectronPressureRhsTerms:
    use_jax = use_jax_backend(
        explicit_pressure_source,
        electron_pressure,
        electron_velocity,
        electron_fastest_wave,
        electron_energy_source,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    explicit = _source_or_zero(explicit_pressure_source, use_jax=use_jax)
    pressure = array(electron_pressure, dtype=dtype)
    velocity = array(electron_velocity, dtype=dtype)
    fastest_wave = array(electron_fastest_wave, dtype=dtype)
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        pressure,
        velocity,
        fastest_wave,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * velocity * _grad_par_open(
        pressure,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source = (2.0 / 3.0) * _source_or_zero(
        electron_energy_source,
        use_jax=use_jax,
    )
    total = explicit + parallel_divergence + parallel_advection + energy_source
    return ElectronPressureRhsTerms(
        explicit_pressure_source=explicit,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source,
        total=total,
    )


def assemble_electron_pressure_active_rhs_terms(
    *,
    explicit_pressure_source: np.ndarray | None,
    electron_pressure: np.ndarray,
    electron_velocity: np.ndarray,
    electron_fastest_wave: np.ndarray,
    electron_energy_source: np.ndarray | None,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> ElectronPressureRhsTerms:
    """Assemble electron pressure terms directly on the active domain."""

    use_jax = use_jax_backend(
        explicit_pressure_source,
        electron_pressure,
        electron_velocity,
        electron_fastest_wave,
        electron_energy_source,
    )
    active = _active_slices(mesh)
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    pressure = array(electron_pressure, dtype=dtype)
    velocity = array(electron_velocity, dtype=dtype)
    explicit = _source_active_or_zero(
        explicit_pressure_source,
        active_slices=active,
        use_jax=use_jax,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open_active(
        pressure,
        velocity,
        array(electron_fastest_wave, dtype=dtype),
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * velocity[active] * _grad_par_open_active(
        pressure,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source = (2.0 / 3.0) * _source_active_or_zero(
        electron_energy_source,
        active_slices=active,
        use_jax=use_jax,
    )
    total = explicit + parallel_divergence + parallel_advection + energy_source
    return ElectronPressureRhsTerms(
        explicit_pressure_source=explicit,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source,
        total=total,
    )


def assemble_electron_parallel_force_terms(
    *,
    electron_pressure: np.ndarray,
    electron_density: np.ndarray,
    electron_momentum_source: np.ndarray | None,
    ion_density: Mapping[str, np.ndarray],
    ion_charge: Mapping[str, float],
    ion_momentum_source: Mapping[str, np.ndarray | None],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    density_floor: float = 1.0e-5,
) -> ElectronParallelForceTerms:
    """Return electron force balance and ion electric-force source updates."""

    use_jax = use_jax_backend(
        electron_pressure,
        electron_density,
        electron_momentum_source,
        *(ion_density.values()),
        *(ion_momentum_source.values()),
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    maximum = jnp.maximum if use_jax else np.maximum
    force_density = -grad_par_electron_force_balance_open(
        array(electron_pressure, dtype=dtype),
        mesh=mesh,
        metrics=metrics,
    )
    force_density = force_density + _source_or_zero(
        electron_momentum_source,
        use_jax=use_jax,
    )
    epar = force_density / maximum(array(electron_density, dtype=dtype), float(density_floor))
    updated_ion_momentum_source: dict[str, np.ndarray] = {}
    for name, density in ion_density.items():
        updated_ion_momentum_source[name] = apply_parallel_electric_force(
            array(density, dtype=dtype),
            charge=float(ion_charge[name]),
            epar=epar,
            existing_source=_source_or_zero(
                ion_momentum_source.get(name),
                use_jax=use_jax,
            ),
        )
    return ElectronParallelForceTerms(
        force_density=force_density,
        epar=epar,
        ion_momentum_source=updated_ion_momentum_source,
    )


def assemble_electron_parallel_force_active_terms(
    *,
    electron_pressure: np.ndarray,
    electron_density: np.ndarray,
    electron_momentum_source: np.ndarray | None,
    ion_density: Mapping[str, np.ndarray],
    ion_charge: Mapping[str, float],
    ion_momentum_source: Mapping[str, np.ndarray | None],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    density_floor: float = 1.0e-5,
) -> ElectronParallelForceTerms:
    """Return active-domain electron force balance and ion force sources."""

    use_jax = use_jax_backend(
        electron_pressure,
        electron_density,
        electron_momentum_source,
        *(ion_density.values()),
        *(ion_momentum_source.values()),
    )
    active = _active_slices(mesh)
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    maximum = jnp.maximum if use_jax else np.maximum
    force_density = -grad_par_electron_force_balance_open_active(
        array(electron_pressure, dtype=dtype),
        mesh=mesh,
        metrics=metrics,
    )
    force_density = force_density + _source_active_or_zero(
        electron_momentum_source,
        active_slices=active,
        use_jax=use_jax,
    )
    density_active = array(electron_density, dtype=dtype)[active]
    epar = force_density / maximum(density_active, float(density_floor))
    updated_ion_momentum_source: dict[str, np.ndarray] = {}
    for name, density in ion_density.items():
        existing = _source_active_or_zero(
            ion_momentum_source.get(name),
            active_slices=active,
            use_jax=use_jax,
        )
        updated_ion_momentum_source[name] = (
            float(ion_charge[name]) * array(density, dtype=dtype)[active] * epar
            + existing
        )
    return ElectronParallelForceTerms(
        force_density=force_density,
        epar=epar,
        ion_momentum_source=updated_ion_momentum_source,
    )


def assemble_ion_rhs_terms(
    *,
    density_source: np.ndarray | None,
    explicit_pressure_source: np.ndarray | None,
    momentum_source: np.ndarray | None,
    atomic_mass: float,
    density_floor: float,
    ion_state: Any,
    ion_velocity: np.ndarray,
    fastest_wave: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    energy_source: np.ndarray | None,
) -> IonRhsTerms:
    use_jax = use_jax_backend(
        density_source,
        explicit_pressure_source,
        momentum_source,
        ion_state.density,
        ion_state.pressure,
        ion_state.momentum_error,
        ion_velocity,
        fastest_wave,
        energy_source,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    density_source_array = _source_or_zero(density_source, use_jax=use_jax)
    ion_velocity_array = array(ion_velocity, dtype=dtype)
    fastest_wave_array = array(fastest_wave, dtype=dtype)
    explicit_pressure_source_array = _source_or_zero(
        explicit_pressure_source,
        use_jax=use_jax,
    )
    momentum_source_array = _source_or_zero(momentum_source, use_jax=use_jax)
    density_array = array(ion_state.density, dtype=dtype)
    pressure_array = array(ion_state.pressure, dtype=dtype)
    momentum_error_array = array(ion_state.momentum_error, dtype=dtype)
    pressure_gradient = -_grad_par_open(pressure_array, mesh=mesh, metrics=metrics)
    density_transport = -_div_par_mod_open(
        density_array,
        ion_velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        pressure_array,
        ion_velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * ion_velocity_array * _grad_par_open(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source_term = (2.0 / 3.0) * _source_or_zero(
        energy_source,
        use_jax=use_jax,
    )
    momentum_advection = -float(atomic_mass) * _div_par_fvv_open(
        soft_floor(density_array, density_floor),
        ion_velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
        fix_flux=False,
    )
    density_total = density_source_array + density_transport
    pressure_total = explicit_pressure_source_array + parallel_divergence + parallel_advection + energy_source_term
    momentum_total = (
        momentum_advection
        + pressure_gradient
        + momentum_source_array
        + momentum_error_array
    )
    return IonRhsTerms(
        density_source=density_source_array,
        density_transport=density_transport,
        density_total=density_total,
        explicit_pressure_source=explicit_pressure_source_array,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source_term,
        pressure_total=pressure_total,
        momentum_advection=momentum_advection,
        pressure_gradient=pressure_gradient,
        momentum_source=momentum_source_array,
        momentum_error=momentum_error_array,
        momentum_total=momentum_total,
    )


def assemble_ion_active_rhs_terms(
    *,
    density_source: np.ndarray | None,
    explicit_pressure_source: np.ndarray | None,
    momentum_source: np.ndarray | None,
    atomic_mass: float,
    density_floor: float,
    ion_state: Any,
    ion_velocity: np.ndarray,
    fastest_wave: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    energy_source: np.ndarray | None,
) -> IonRhsTerms:
    """Assemble ion RHS terms directly on the active domain."""

    use_jax = use_jax_backend(
        density_source,
        explicit_pressure_source,
        momentum_source,
        ion_state.density,
        ion_state.pressure,
        ion_state.momentum_error,
        ion_velocity,
        fastest_wave,
        energy_source,
    )
    active = _active_slices(mesh)
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    density_array = array(ion_state.density, dtype=dtype)
    pressure_array = array(ion_state.pressure, dtype=dtype)
    velocity_array = array(ion_velocity, dtype=dtype)
    fastest_wave_array = array(fastest_wave, dtype=dtype)
    density_source_array = _source_active_or_zero(
        density_source,
        active_slices=active,
        use_jax=use_jax,
    )
    explicit_pressure_source_array = _source_active_or_zero(
        explicit_pressure_source,
        active_slices=active,
        use_jax=use_jax,
    )
    momentum_source_array = _source_active_or_zero(
        momentum_source,
        active_slices=active,
        use_jax=use_jax,
    )
    pressure_gradient = -_grad_par_open_active(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    density_transport = -_div_par_mod_open_active(
        density_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open_active(
        pressure_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * velocity_array[active] * _grad_par_open_active(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source_term = (2.0 / 3.0) * _source_active_or_zero(
        energy_source,
        active_slices=active,
        use_jax=use_jax,
    )
    momentum_advection = -float(atomic_mass) * _div_par_fvv_open_active(
        soft_floor(density_array, density_floor),
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
        fix_flux=False,
    )
    momentum_error_array = array(ion_state.momentum_error, dtype=dtype)[active]
    density_total = density_source_array + density_transport
    pressure_total = (
        explicit_pressure_source_array
        + parallel_divergence
        + parallel_advection
        + energy_source_term
    )
    momentum_total = (
        momentum_advection
        + pressure_gradient
        + momentum_source_array
        + momentum_error_array
    )
    return IonRhsTerms(
        density_source=density_source_array,
        density_transport=density_transport,
        density_total=density_total,
        explicit_pressure_source=explicit_pressure_source_array,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source_term,
        pressure_total=pressure_total,
        momentum_advection=momentum_advection,
        pressure_gradient=pressure_gradient,
        momentum_source=momentum_source_array,
        momentum_error=momentum_error_array,
        momentum_total=momentum_total,
    )


def assemble_neutral_rhs_terms(
    *,
    density_source: np.ndarray | None,
    explicit_pressure_source: np.ndarray | None,
    momentum_source: np.ndarray | None,
    atomic_mass: float,
    density_floor: float,
    neutral_state: Any,
    neutral_velocity: np.ndarray,
    fastest_wave: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    energy_source: np.ndarray | None,
    include_energy_source: bool = True,
) -> NeutralRhsTerms:
    use_jax = use_jax_backend(
        density_source,
        explicit_pressure_source,
        momentum_source,
        neutral_state.density,
        neutral_state.pressure,
        neutral_state.momentum_error,
        neutral_velocity,
        fastest_wave,
        energy_source,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    density_source_array = _source_or_zero(density_source, use_jax=use_jax)
    explicit_pressure_source_array = _source_or_zero(
        explicit_pressure_source,
        use_jax=use_jax,
    )
    momentum_source_array = _source_or_zero(momentum_source, use_jax=use_jax)
    density_array = array(neutral_state.density, dtype=dtype)
    pressure_array = array(neutral_state.pressure, dtype=dtype)
    momentum_error_array = array(neutral_state.momentum_error, dtype=dtype)
    velocity_array = array(neutral_velocity, dtype=dtype)
    fastest_wave_array = array(fastest_wave, dtype=dtype)

    density_transport = -_div_par_mod_open(
        density_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        pressure_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * velocity_array * _grad_par_open(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    if include_energy_source:
        energy_source_term = (2.0 / 3.0) * _source_or_zero(
            energy_source,
            use_jax=use_jax,
        )
    else:
        energy_source_term = _source_or_zero(None, use_jax=use_jax)
    momentum_advection = -float(atomic_mass) * _div_par_fvv_open(
        soft_floor(density_array, density_floor),
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
        fix_flux=False,
    )
    pressure_gradient = -_grad_par_open(pressure_array, mesh=mesh, metrics=metrics)
    density_total = density_source_array + density_transport
    pressure_total = explicit_pressure_source_array + parallel_divergence + parallel_advection + energy_source_term
    momentum_total = momentum_advection + pressure_gradient + momentum_source_array + momentum_error_array
    return NeutralRhsTerms(
        density_source=density_source_array,
        density_transport=density_transport,
        density_total=density_total,
        explicit_pressure_source=explicit_pressure_source_array,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source_term,
        pressure_total=pressure_total,
        momentum_advection=momentum_advection,
        pressure_gradient=pressure_gradient,
        momentum_source=momentum_source_array,
        momentum_error=momentum_error_array,
        momentum_total=momentum_total,
    )


def assemble_neutral_active_rhs_terms(
    *,
    density_source: np.ndarray | None,
    explicit_pressure_source: np.ndarray | None,
    momentum_source: np.ndarray | None,
    atomic_mass: float,
    density_floor: float,
    neutral_state: Any,
    neutral_velocity: np.ndarray,
    fastest_wave: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    energy_source: np.ndarray | None,
    include_energy_source: bool = True,
) -> NeutralRhsTerms:
    """Assemble neutral RHS terms directly on the active domain."""

    use_jax = use_jax_backend(
        density_source,
        explicit_pressure_source,
        momentum_source,
        neutral_state.density,
        neutral_state.pressure,
        neutral_state.momentum_error,
        neutral_velocity,
        fastest_wave,
        energy_source,
    )
    active = _active_slices(mesh)
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    density_array = array(neutral_state.density, dtype=dtype)
    pressure_array = array(neutral_state.pressure, dtype=dtype)
    velocity_array = array(neutral_velocity, dtype=dtype)
    fastest_wave_array = array(fastest_wave, dtype=dtype)
    density_source_array = _source_active_or_zero(
        density_source,
        active_slices=active,
        use_jax=use_jax,
    )
    explicit_pressure_source_array = _source_active_or_zero(
        explicit_pressure_source,
        active_slices=active,
        use_jax=use_jax,
    )
    momentum_source_array = _source_active_or_zero(
        momentum_source,
        active_slices=active,
        use_jax=use_jax,
    )

    density_transport = -_div_par_mod_open_active(
        density_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open_active(
        pressure_array,
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * velocity_array[active] * _grad_par_open_active(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    if include_energy_source:
        energy_source_term = (2.0 / 3.0) * _source_active_or_zero(
            energy_source,
            active_slices=active,
            use_jax=use_jax,
        )
    else:
        energy_source_term = _source_or_zero(None, use_jax=use_jax)
    momentum_advection = -float(atomic_mass) * _div_par_fvv_open_active(
        soft_floor(density_array, density_floor),
        velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
        fix_flux=False,
    )
    pressure_gradient = -_grad_par_open_active(
        pressure_array,
        mesh=mesh,
        metrics=metrics,
    )
    momentum_error_array = array(neutral_state.momentum_error, dtype=dtype)[active]
    density_total = density_source_array + density_transport
    pressure_total = (
        explicit_pressure_source_array
        + parallel_divergence
        + parallel_advection
        + energy_source_term
    )
    momentum_total = (
        momentum_advection
        + pressure_gradient
        + momentum_source_array
        + momentum_error_array
    )
    return NeutralRhsTerms(
        density_source=density_source_array,
        density_transport=density_transport,
        density_total=density_total,
        explicit_pressure_source=explicit_pressure_source_array,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source_term,
        pressure_total=pressure_total,
        momentum_advection=momentum_advection,
        pressure_gradient=pressure_gradient,
        momentum_source=momentum_source_array,
        momentum_error=momentum_error_array,
        momentum_total=momentum_total,
    )


def _source_or_zero(value: np.ndarray | None, *, use_jax: bool):
    """Return an additive source, using scalar zero when the source is absent."""

    if value is None:
        return jnp.asarray(0.0, dtype=jnp.float64) if use_jax else np.asarray(0.0, dtype=np.float64)
    return jnp.asarray(value, dtype=jnp.float64) if use_jax else np.asarray(value, dtype=np.float64)


def _source_active_or_zero(
    value: np.ndarray | None,
    *,
    active_slices: tuple[slice, slice, slice],
    use_jax: bool,
) -> np.ndarray:
    if value is None:
        return _source_or_zero(None, use_jax=use_jax)
    source = _source_or_zero(value, use_jax=use_jax)
    if source.ndim == 0:
        return source
    active_xy_shape = (
        active_slices[0].stop - active_slices[0].start,
        active_slices[1].stop - active_slices[1].start,
    )
    if source.ndim == 3 and tuple(source.shape[:2]) == active_xy_shape:
        return source
    return source[active_slices]


def _active_slices(mesh: StructuredMesh) -> tuple[slice, slice, slice]:
    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )
