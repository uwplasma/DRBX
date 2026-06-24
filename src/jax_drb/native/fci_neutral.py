from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..geometry import FciGeometry3D
from .fci import conservative_parallel_diffusion_fci, conservative_perp_diffusion_xy


@dataclass(frozen=True)
class FciNeutralReactionDiffusionResult:
    neutral_density_source: jnp.ndarray
    ion_density_source: jnp.ndarray
    electron_density_source: jnp.ndarray
    neutral_pressure_source: jnp.ndarray
    ion_pressure_source: jnp.ndarray
    electron_pressure_source: jnp.ndarray
    neutral_momentum_source: jnp.ndarray
    ion_momentum_source: jnp.ndarray
    neutral_diffusion_source: jnp.ndarray
    neutral_pressure_diffusion_source: jnp.ndarray
    ionisation_rate: jnp.ndarray
    recombination_rate: jnp.ndarray
    charge_exchange_rate: jnp.ndarray
    total_particle_residual: jnp.ndarray
    total_momentum_residual: jnp.ndarray
    total_charge_exchange_particle_residual: jnp.ndarray


def compute_fci_neutral_reaction_diffusion(
    *,
    neutral_density: jnp.ndarray,
    neutral_pressure: jnp.ndarray,
    neutral_momentum: jnp.ndarray,
    ion_density: jnp.ndarray,
    ion_pressure: jnp.ndarray,
    ion_momentum: jnp.ndarray,
    electron_density: jnp.ndarray,
    electron_pressure: jnp.ndarray,
    geometry: FciGeometry3D,
    neutral_parallel_diffusivity: float = 0.03,
    neutral_perp_diffusivity: float = 4.0e-4,
    ionisation_coefficient: float = 0.08,
    recombination_coefficient: float = 0.015,
    charge_exchange_coefficient: float = 0.04,
    ionisation_energy: float = 0.035,
    recombination_energy: float = 0.012,
    neutral_mass: float = 1.0,
    ion_mass: float = 1.0,
    density_floor: float = 1.0e-12,
) -> FciNeutralReactionDiffusionResult:
    """Compute compact FCI neutral diffusion, ionisation, recombination, and CX gates."""

    nn = jnp.maximum(jnp.asarray(neutral_density, dtype=jnp.float64), float(density_floor))
    pn = jnp.maximum(jnp.asarray(neutral_pressure, dtype=jnp.float64), 0.0)
    mn = jnp.asarray(neutral_momentum, dtype=jnp.float64)
    ni = jnp.maximum(jnp.asarray(ion_density, dtype=jnp.float64), float(density_floor))
    pi = jnp.maximum(jnp.asarray(ion_pressure, dtype=jnp.float64), 0.0)
    mi = jnp.asarray(ion_momentum, dtype=jnp.float64)
    ne = jnp.maximum(jnp.asarray(electron_density, dtype=jnp.float64), float(density_floor))
    pe = jnp.maximum(jnp.asarray(electron_pressure, dtype=jnp.float64), 0.0)
    tn = pn / nn
    ti = pi / ni
    te = pe / ne
    un = mn / jnp.maximum(float(neutral_mass) * nn, float(density_floor))
    ui = mi / jnp.maximum(float(ion_mass) * ni, float(density_floor))

    neutral_diffusion_coefficient = neutral_parallel_diffusivity * jnp.sqrt(jnp.maximum(tn, 1.0e-12))
    neutral_perp_coefficient = neutral_perp_diffusivity * jnp.sqrt(jnp.maximum(tn, 1.0e-12))
    neutral_diffusion_source = conservative_parallel_diffusion_fci(
        nn,
        neutral_diffusion_coefficient,
        geometry,
    ) + conservative_perp_diffusion_xy(nn, neutral_perp_coefficient, geometry)
    neutral_pressure_diffusion_source = conservative_parallel_diffusion_fci(
        pn,
        neutral_diffusion_coefficient,
        geometry,
    ) + conservative_perp_diffusion_xy(pn, neutral_perp_coefficient, geometry)

    ionisation_rate = ionisation_coefficient * nn * ne * jnp.sqrt(jnp.maximum(te, 1.0e-12))
    recombination_rate = recombination_coefficient * ni * ne / jnp.sqrt(jnp.maximum(te, 1.0e-12))
    charge_exchange_rate = charge_exchange_coefficient * nn * ni * jnp.sqrt(jnp.maximum(tn + ti, 1.0e-12))

    neutral_density_source = neutral_diffusion_source - ionisation_rate + recombination_rate
    ion_density_source = ionisation_rate - recombination_rate
    electron_density_source = ionisation_rate - recombination_rate

    ionisation_neutral_energy = 1.5 * ionisation_rate * tn
    recombination_ion_energy = 1.5 * recombination_rate * ti
    cx_neutral_energy = 1.5 * charge_exchange_rate * tn
    cx_ion_energy = 1.5 * charge_exchange_rate * ti
    relative_velocity = ui - un
    cx_kinetic = 0.5 * charge_exchange_rate * jnp.square(relative_velocity)

    neutral_pressure_source = (
        neutral_pressure_diffusion_source
        - ionisation_neutral_energy
        + recombination_ion_energy
        - cx_neutral_energy
        + cx_ion_energy
        + cx_kinetic
    )
    ion_pressure_source = ionisation_neutral_energy - recombination_ion_energy + cx_neutral_energy - cx_ion_energy + cx_kinetic
    electron_pressure_source = -ionisation_energy * ionisation_rate - recombination_energy * recombination_rate

    ionisation_momentum = ionisation_rate * float(neutral_mass) * un
    recombination_momentum = recombination_rate * float(ion_mass) * ui
    cx_neutral_momentum = charge_exchange_rate * float(neutral_mass) * un
    cx_ion_momentum = charge_exchange_rate * float(ion_mass) * ui
    neutral_momentum_source = -ionisation_momentum + recombination_momentum - cx_neutral_momentum + cx_ion_momentum
    ion_momentum_source = ionisation_momentum - recombination_momentum + cx_neutral_momentum - cx_ion_momentum

    total_particle_residual = jnp.sum((neutral_density_source - neutral_diffusion_source) + ion_density_source)
    total_momentum_residual = jnp.sum(neutral_momentum_source + ion_momentum_source)
    total_charge_exchange_particle_residual = jnp.sum(charge_exchange_rate - charge_exchange_rate)

    return FciNeutralReactionDiffusionResult(
        neutral_density_source=neutral_density_source,
        ion_density_source=ion_density_source,
        electron_density_source=electron_density_source,
        neutral_pressure_source=neutral_pressure_source,
        ion_pressure_source=ion_pressure_source,
        electron_pressure_source=electron_pressure_source,
        neutral_momentum_source=neutral_momentum_source,
        ion_momentum_source=ion_momentum_source,
        neutral_diffusion_source=neutral_diffusion_source,
        neutral_pressure_diffusion_source=neutral_pressure_diffusion_source,
        ionisation_rate=ionisation_rate,
        recombination_rate=recombination_rate,
        charge_exchange_rate=charge_exchange_rate,
        total_particle_residual=total_particle_residual,
        total_momentum_residual=total_momentum_residual,
        total_charge_exchange_particle_residual=total_charge_exchange_particle_residual,
    )
