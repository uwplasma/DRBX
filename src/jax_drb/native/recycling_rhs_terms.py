from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .array_backend import use_jax_backend
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .neutral_mixed import _div_par_fvv_open, _div_par_mod_open, _grad_par_open


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


def soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    minimum_value = float(minimum)
    if use_jax_backend(value):
        value_array = jnp.maximum(jnp.asarray(value, dtype=jnp.float64), 0.0)
        return value_array + minimum_value * jnp.exp(-value_array / minimum_value)
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


def assemble_electron_pressure_rhs_terms(
    *,
    explicit_pressure_source: np.ndarray,
    electron_pressure: np.ndarray,
    electron_velocity: np.ndarray,
    electron_fastest_wave: np.ndarray,
    electron_energy_source: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> ElectronPressureRhsTerms:
    use_jax = use_jax_backend(
        explicit_pressure_source,
        electron_pressure,
        electron_velocity,
        electron_fastest_wave,
        electron_energy_source,
        metrics.dy,
        metrics.J,
        metrics.g_22,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    explicit = array(explicit_pressure_source, dtype=dtype)
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
    energy_source = (2.0 / 3.0) * array(electron_energy_source, dtype=dtype)
    total = explicit + parallel_divergence + parallel_advection + energy_source
    return ElectronPressureRhsTerms(
        explicit_pressure_source=explicit,
        parallel_divergence=parallel_divergence,
        parallel_advection=parallel_advection,
        energy_source=energy_source,
        total=total,
    )


def assemble_ion_rhs_terms(
    *,
    density_source: np.ndarray,
    explicit_pressure_source: np.ndarray,
    momentum_source: np.ndarray,
    atomic_mass: float,
    density_floor: float,
    ion_state: Any,
    ion_velocity: np.ndarray,
    fastest_wave: np.ndarray,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    energy_source: np.ndarray,
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
        metrics.dy,
        metrics.J,
        metrics.g_22,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    density_source_array = array(density_source, dtype=dtype)
    ion_velocity_array = array(ion_velocity, dtype=dtype)
    fastest_wave_array = array(fastest_wave, dtype=dtype)
    explicit_pressure_source_array = array(explicit_pressure_source, dtype=dtype)
    momentum_source_array = array(momentum_source, dtype=dtype)
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
    energy_source_term = (2.0 / 3.0) * array(energy_source, dtype=dtype)
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
