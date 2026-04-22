from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    minimum_value = float(minimum)
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
    explicit = np.asarray(explicit_pressure_source, dtype=np.float64)
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        electron_pressure,
        electron_velocity,
        electron_fastest_wave,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * electron_velocity * _grad_par_open(
        electron_pressure,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source = (2.0 / 3.0) * np.asarray(electron_energy_source, dtype=np.float64)
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
    density_source_array = np.asarray(density_source, dtype=np.float64)
    ion_velocity_array = np.asarray(ion_velocity, dtype=np.float64)
    fastest_wave_array = np.asarray(fastest_wave, dtype=np.float64)
    explicit_pressure_source_array = np.asarray(explicit_pressure_source, dtype=np.float64)
    momentum_source_array = np.asarray(momentum_source, dtype=np.float64)
    pressure_gradient = -_grad_par_open(ion_state.pressure, mesh=mesh, metrics=metrics)
    density_transport = -_div_par_mod_open(
        ion_state.density,
        ion_velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_divergence = -(5.0 / 3.0) * _div_par_mod_open(
        ion_state.pressure,
        ion_velocity_array,
        fastest_wave_array,
        mesh=mesh,
        metrics=metrics,
    )
    parallel_advection = (2.0 / 3.0) * ion_velocity_array * _grad_par_open(
        ion_state.pressure,
        mesh=mesh,
        metrics=metrics,
    )
    energy_source_term = (2.0 / 3.0) * np.asarray(energy_source, dtype=np.float64)
    momentum_advection = -float(atomic_mass) * _div_par_fvv_open(
        soft_floor(ion_state.density, density_floor),
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
        + np.asarray(ion_state.momentum_error, dtype=np.float64)
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
        momentum_error=np.asarray(ion_state.momentum_error, dtype=np.float64),
        momentum_total=momentum_total,
    )
