"""Self-consistent detaching 1D SOL model (evolved temperature).

Extends the recycling SOL to an *evolved* plasma pressure so the target
temperature responds to the plasma conditions -- the ingredient a detachment
study needs. Along the field it evolves ion density, parallel momentum, plasma
pressure, and a diffusive neutral density, with:

- **Spitzer parallel conduction** ``kappa ~ T^{5/2}`` solved implicitly (a solvax
  tridiagonal), so the stiff parabolic heat transport is unconditionally stable;
- **radiative / ionization energy loss** from the AMJUEL fits, applied as a
  *self-limiting* semi-implicit sink (``P <- P / (1 + dt * loss_rate)``) so the
  loss cannot drive the pressure negative and switches off as the plasma cools;
- a **Bohm sheath heat sink** at the target and an **upstream power source**; and
- the neutral **recycling**, parallel diffusion, and ionization/recombination /
  charge-exchange coupling of the recycling model.

As the upstream density rises at fixed upstream power the target cools, crosses
into the recombining regime below ~1 eV, and the target ion flux rolls over --
the classic SD1D detachment signature. Everything is pure ``jax.numpy`` (with the
solvax tridiagonal solves) and differentiable. Fields are hermes-3 normalized;
time is normalized to the parallel transit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from solvax import tridiagonal_solve

from .atomic_rates import (
    charge_exchange_rate_coefficient,
    energy_loss_coefficient,
    rate_coefficient,
)
from .reactions import PlasmaNormalization

__all__ = [
    "DetachmentSolParameters",
    "DetachmentSolState",
    "detachment_sol_step",
    "detachment_sol_run",
    "detachment_diagnostics",
]


@dataclass(frozen=True)
class DetachmentSolParameters:
    """Parameters of the self-consistent detaching SOL model."""

    parallel_length: float = 30.0
    upstream_density: float = 4.0
    upstream_power: float = 6.0              # normalized upstream power source
    power_width: float = 0.2                 # parallel width of the power source
    conduction_coefficient: float = 2.0      # Spitzer kappa0 (normalized)
    sheath_transmission: float = 7.0         # gamma (Te + Ti)
    neutral_diffusion: float = 8.0
    recycling_fraction: float = 0.95
    ion_mass: float = 2.0
    density_floor: float = 1.0e-4
    pressure_floor: float = 1.0e-5
    temperature_floor: float = 1.0e-3
    # A scrape-off-layer reference temperature (upstream ~ tens of eV); the
    # detachment transition and target-flux rollover live in this regime.
    normalization: PlasmaNormalization = PlasmaNormalization(Tnorm=50.0)


class DetachmentSolState(NamedTuple):
    """Evolved fields: ion density, ion momentum, plasma pressure, neutral density."""

    ion_density: jnp.ndarray
    ion_momentum: jnp.ndarray
    plasma_pressure: jnp.ndarray
    neutral_density: jnp.ndarray


def _temperature(density, pressure, params):
    return jnp.maximum(pressure / (2.0 * jnp.maximum(density, params.density_floor)), params.temperature_floor)


def _sound_speed(density, pressure, params):
    return jnp.sqrt(jnp.maximum(pressure / (params.ion_mass * jnp.maximum(density, params.density_floor)), 0.0))


def _hyperbolic_rhs(density, momentum, pressure, params):
    mass = params.ion_mass
    dz = 1.0 / density.shape[0]
    velocity = momentum / (mass * jnp.maximum(density, params.density_floor))
    speed = jnp.abs(velocity) + _sound_speed(density, pressure, params)
    face_speed = jnp.maximum(speed[:-1], speed[1:])
    flux_n = 0.5 * (density[:-1] * velocity[:-1] + density[1:] * velocity[1:]) - 0.5 * face_speed * (density[1:] - density[:-1])
    flux_m = 0.5 * (momentum[:-1] * velocity[:-1] + pressure[:-1] + momentum[1:] * velocity[1:] + pressure[1:]) - 0.5 * face_speed * (momentum[1:] - momentum[:-1])
    flux_p = 0.5 * ((5.0 / 3.0) * pressure[:-1] * velocity[:-1] + (5.0 / 3.0) * pressure[1:] * velocity[1:]) - 0.5 * face_speed * (pressure[1:] - pressure[:-1])
    target_velocity = jnp.maximum(velocity[-1], _sound_speed(density, pressure, params)[-1])
    flux_n = jnp.concatenate([jnp.zeros(1), flux_n, (density[-1] * target_velocity)[None]])
    flux_m = jnp.concatenate([pressure[:1], flux_m, (mass * density[-1] * target_velocity**2 + pressure[-1])[None]])
    flux_p = jnp.concatenate([jnp.zeros(1), flux_p, ((5.0 / 3.0) * pressure[-1] * target_velocity)[None]])
    pressure_gradient = jnp.gradient(pressure, dz)
    d_density = -(flux_n[1:] - flux_n[:-1]) / dz
    d_momentum = -(flux_m[1:] - flux_m[:-1]) / dz
    d_pressure = -(flux_p[1:] - flux_p[:-1]) / dz + (2.0 / 3.0) * velocity * pressure_gradient
    return d_density, d_momentum, d_pressure


def _implicit_diffusion(field, face_diffusivity, dt):
    n = field.shape[0]
    dz = 1.0 / n
    coefficient = dt / dz**2
    lower = jnp.zeros(n).at[1:].set(-coefficient * face_diffusivity)
    upper = jnp.zeros(n).at[:-1].set(-coefficient * face_diffusivity)
    diagonal = jnp.ones(n).at[1:-1].set(1.0 + coefficient * (face_diffusivity[:-1] + face_diffusivity[1:]))
    diagonal = diagonal.at[0].set(1.0 + coefficient * face_diffusivity[0])
    diagonal = diagonal.at[-1].set(1.0 + coefficient * face_diffusivity[-1])
    return tridiagonal_solve(lower, diagonal, upper, field, method="thomas")


def detachment_sol_step(state, params, dt):
    """Advance the self-consistent detaching SOL one operator-split step."""

    norm = params.normalization
    nz = state.ion_density.shape[0]
    dz = 1.0 / nz
    z = (jnp.arange(nz, dtype=jnp.float64) + 0.5) / nz
    density_floor = params.density_floor
    pressure_floor = params.pressure_floor
    rate_scale = norm.Nnorm * (params.parallel_length / norm.sound_speed)
    energy_scale = rate_scale / norm.Tnorm

    density = jnp.maximum(state.ion_density, density_floor)
    momentum = state.ion_momentum
    pressure = jnp.maximum(state.plasma_pressure, pressure_floor)
    neutral_density = state.neutral_density

    # Hyperbolic transport + upstream power source.
    d_density, d_momentum, d_pressure = _hyperbolic_rhs(density, momentum, pressure, params)
    power_source = params.upstream_power * jnp.exp(-(z**2) / params.power_width**2)
    density = jnp.maximum(density + dt * d_density, density_floor)
    momentum = momentum + dt * d_momentum
    pressure = jnp.maximum(pressure + dt * d_pressure + dt * (2.0 / 3.0) * power_source, pressure_floor)

    # Atomic reactions: particle exchange (implicit) + momentum friction.
    temperature = _temperature(density, pressure, params)
    physical_temperature = temperature * norm.Tnorm
    electron_density_m3 = jnp.maximum(density, density_floor) * norm.Nnorm
    electron = jnp.maximum(density, density_floor)
    ionization = rate_coefficient("d", "iz", physical_temperature, electron_density_m3) * rate_scale
    recombination = rate_coefficient("d", "rec", physical_temperature, electron_density_m3) * rate_scale
    ionization_frequency = ionization * electron
    recombination_frequency = recombination * electron
    total = density + neutral_density
    density = jnp.maximum((density + dt * ionization_frequency * total) / (1.0 + dt * (ionization_frequency + recombination_frequency)), density_floor)
    neutral_density = jnp.maximum(total - density, 0.0)
    effective_temperature = jnp.clip(2.0 * temperature * norm.Tnorm / params.ion_mass, 0.01, 1.0e4)
    charge_exchange_frequency = charge_exchange_rate_coefficient(effective_temperature) * electron * rate_scale
    momentum = momentum / (1.0 + dt * (charge_exchange_frequency * neutral_density + recombination_frequency))

    # Self-limiting radiative / ionization energy loss.
    ionization_energy = energy_loss_coefficient("d", "iz", physical_temperature, electron_density_m3)
    recombination_energy = energy_loss_coefficient("d", "rec", physical_temperature, electron_density_m3)
    energy_loss = (neutral_density * electron * ionization_energy + electron * electron * recombination_energy) * energy_scale
    loss_rate = (2.0 / 3.0) * jnp.maximum(energy_loss, 0.0) / jnp.maximum(pressure, pressure_floor)
    pressure = pressure / (1.0 + dt * loss_rate)

    # Bohm sheath heat sink at the target (semi-implicit).
    sound_speed = _sound_speed(density, pressure, params)
    sheath_rate = jnp.zeros(nz).at[-1].set(
        (2.0 / 3.0) * params.sheath_transmission * density[-1] * sound_speed[-1] * _temperature(density, pressure, params)[-1] / dz / jnp.maximum(pressure[-1], pressure_floor)
    )
    pressure = pressure / (1.0 + dt * sheath_rate)

    # Implicit Spitzer conduction.
    temperature = _temperature(density, pressure, params)
    conductivity = params.conduction_coefficient * temperature**2.5
    diffusivity = conductivity / (3.0 * jnp.maximum(density, density_floor))
    pressure = jnp.maximum(_implicit_diffusion(pressure, 0.5 * (diffusivity[:-1] + diffusivity[1:]), dt), pressure_floor)

    density = density.at[0].set(params.upstream_density)

    # Neutral parallel diffusion + target recycling.
    neutral_diffusivity = jnp.full(nz - 1, params.neutral_diffusion * 0.04 / (params.ion_mass * 0.05))
    sound_speed = _sound_speed(density, pressure, params)
    recycling_source = jnp.zeros(nz).at[-1].set(params.recycling_fraction * density[-1] * sound_speed[-1] / dz)
    neutral_density = jnp.maximum(_implicit_diffusion(neutral_density + dt * recycling_source, neutral_diffusivity, dt), 0.0)

    return DetachmentSolState(density, momentum, pressure, neutral_density)


def detachment_sol_run(state, params, *, dt, steps):
    """Advance ``steps`` operator-split steps with a jitted ``lax.scan``."""

    @jax.jit
    def _run(initial):
        def body(carry, _):
            return detachment_sol_step(carry, params, dt), None

        final, _ = jax.lax.scan(body, initial, None, length=steps)
        return final

    return _run(state)


class DetachmentDiagnostics(NamedTuple):
    target_ion_flux: jnp.ndarray
    target_temperature_ev: jnp.ndarray
    target_density: jnp.ndarray


def detachment_diagnostics(state, params) -> DetachmentDiagnostics:
    """Target ion flux (``n c_s``), target temperature [eV], and target density."""

    density = jnp.maximum(state.ion_density, params.density_floor)
    temperature = _temperature(density, jnp.maximum(state.plasma_pressure, params.pressure_floor), params)
    sound_speed = _sound_speed(density, jnp.maximum(state.plasma_pressure, params.pressure_floor), params)
    return DetachmentDiagnostics(
        target_ion_flux=density[-1] * sound_speed[-1],
        target_temperature_ev=temperature[-1] * params.normalization.Tnorm,
        target_density=density[-1],
    )
