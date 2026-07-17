"""Plasma <-> neutral atomic reaction sources for the 1D SOL model.

Assembles the ionization / recombination / charge-exchange source channels that
couple a hydrogenic plasma fluid (ion density ``Ni``, parallel momentum, ion
temperature ``Ti``, electron temperature ``Te``) to a neutral fluid (``Nn``,
momentum, ``Tn``), following the hermes-3 hydrogen reaction closure. Particle,
momentum, and (thermal) energy transfers use the Galilean-invariant form: each
particle transfer carries the source species' momentum ``m V`` and thermal
energy ``1.5 T``, and charge exchange adds a frictional heating ``0.5 m R dV^2``
from the ion-atom velocity difference. The electron channel is the ionization
cost / recombination radiation from the AMJUEL energy-loss fits.

All fields are in hermes-3 normalized units (density / ``Nnorm``, temperature /
``Tnorm`` [eV], velocity / ``Cs0``); the returned source rates are normalized to
``Nnorm * Omega_ci`` (particle / momentum) and ``Nnorm * Tnorm * Omega_ci``
(energy). Everything is pure ``jax.numpy`` and differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .atomic_rates import (
    charge_exchange_rate_coefficient,
    energy_loss_coefficient,
    rate_coefficient,
)

__all__ = ["PlasmaNormalization", "HydrogenReactionSources", "compute_hydrogen_reaction_sources"]

_ELEMENTARY_CHARGE = 1.602176634e-19
_PROTON_MASS = 1.67262192369e-27


@dataclass(frozen=True)
class PlasmaNormalization:
    """hermes-3 normalization constants (deck defaults: 1e19 m^-3, 100 eV, 1 T)."""

    Nnorm: float = 1.0e19
    Tnorm: float = 100.0
    Bnorm: float = 1.0
    ion_mass: float = 2.0  # deuterium, in proton masses

    @property
    def sound_speed(self) -> float:
        return float((_ELEMENTARY_CHARGE * self.Tnorm / _PROTON_MASS) ** 0.5)

    @property
    def cyclotron_frequency(self) -> float:
        return float(_ELEMENTARY_CHARGE * self.Bnorm / _PROTON_MASS)


@dataclass(frozen=True)
class HydrogenReactionSources:
    """Normalized reaction source channels for the ion, neutral, and electron fluids."""

    ion_density: jnp.ndarray
    neutral_density: jnp.ndarray
    ion_momentum: jnp.ndarray
    neutral_momentum: jnp.ndarray
    ion_energy: jnp.ndarray
    neutral_energy: jnp.ndarray
    electron_energy: jnp.ndarray
    ionization_rate: jnp.ndarray
    recombination_rate: jnp.ndarray
    charge_exchange_rate: jnp.ndarray


def compute_hydrogen_reaction_sources(
    ion_density: jnp.ndarray,
    ion_velocity: jnp.ndarray,
    ion_temperature: jnp.ndarray,
    electron_temperature: jnp.ndarray,
    neutral_density: jnp.ndarray,
    neutral_velocity: jnp.ndarray,
    neutral_temperature: jnp.ndarray,
    *,
    normalization: PlasmaNormalization = PlasmaNormalization(),
    charge_exchange_multiplier: float = 1.0,
) -> HydrogenReactionSources:
    """Return the normalized ion/neutral/electron reaction source channels."""

    mass = float(normalization.ion_mass)
    nnorm = float(normalization.Nnorm)
    tnorm = float(normalization.Tnorm)
    omega = normalization.cyclotron_frequency
    rate_scale = nnorm / omega
    energy_scale = nnorm / (tnorm * omega)

    # Physical inputs for the rate fits.
    electron_temperature_ev = electron_temperature * tnorm
    electron_density_m3 = ion_density * nnorm  # quasineutral, Z = 1

    ionization_sigmav = rate_coefficient("d", "iz", electron_temperature_ev, electron_density_m3)
    recombination_sigmav = rate_coefficient("d", "rec", electron_temperature_ev, electron_density_m3)
    ionization_energy = energy_loss_coefficient("d", "iz", electron_temperature_ev, electron_density_m3)
    recombination_energy = energy_loss_coefficient("d", "rec", electron_temperature_ev, electron_density_m3)

    effective_temperature = jnp.clip(
        (neutral_temperature / mass + ion_temperature / mass) * tnorm, 0.01, 1.0e4
    )
    cx_sigmav = charge_exchange_rate_coefficient(effective_temperature)

    # Normalized volumetric reaction rates.
    ionization_rate = neutral_density * ion_density * ionization_sigmav * rate_scale
    recombination_rate = ion_density * ion_density * recombination_sigmav * rate_scale
    charge_exchange_rate = (
        ion_density * neutral_density * cx_sigmav * rate_scale * float(charge_exchange_multiplier)
    )
    ionization_radiation = neutral_density * ion_density * ionization_energy * energy_scale
    recombination_radiation = ion_density * ion_density * recombination_energy * energy_scale

    # Momentum carried by each particle transfer (source species' m V).
    ionization_momentum = ionization_rate * mass * neutral_velocity
    recombination_momentum = recombination_rate * mass * ion_velocity
    cx_neutral_momentum = charge_exchange_rate * mass * neutral_velocity
    cx_ion_momentum = charge_exchange_rate * mass * ion_velocity

    # Thermal energy carried by each particle transfer, plus CX frictional heating.
    ionization_energy_transfer = 1.5 * ionization_rate * neutral_temperature
    recombination_energy_transfer = 1.5 * recombination_rate * ion_temperature
    cx_neutral_thermal = 1.5 * charge_exchange_rate * neutral_temperature
    cx_ion_thermal = 1.5 * charge_exchange_rate * ion_temperature
    velocity_delta = ion_velocity - neutral_velocity
    cx_frictional_heat = 0.5 * mass * charge_exchange_rate * velocity_delta**2

    ion_density_source = ionization_rate - recombination_rate
    neutral_density_source = -ionization_rate + recombination_rate
    ion_momentum_source = (
        ionization_momentum - recombination_momentum + cx_neutral_momentum - cx_ion_momentum
    )
    neutral_momentum_source = (
        -ionization_momentum + recombination_momentum - cx_neutral_momentum + cx_ion_momentum
    )
    ion_energy_source = (
        ionization_energy_transfer
        - recombination_energy_transfer
        + cx_neutral_thermal
        - cx_ion_thermal
        + cx_frictional_heat
    )
    neutral_energy_source = (
        -ionization_energy_transfer
        + recombination_energy_transfer
        - cx_neutral_thermal
        + cx_ion_thermal
        + cx_frictional_heat
    )
    electron_energy_source = -ionization_radiation - recombination_radiation

    return HydrogenReactionSources(
        ion_density=ion_density_source,
        neutral_density=neutral_density_source,
        ion_momentum=ion_momentum_source,
        neutral_momentum=neutral_momentum_source,
        ion_energy=ion_energy_source,
        neutral_energy=neutral_energy_source,
        electron_energy=electron_energy_source,
        ionization_rate=ionization_rate,
        recombination_rate=recombination_rate,
        charge_exchange_rate=charge_exchange_rate,
    )
