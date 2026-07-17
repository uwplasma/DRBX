"""Coupled 1D scrape-off-layer recycling model: plasma + diffusive neutral.

A reduced SOL model that couples a hydrogenic plasma fluid to a recycled neutral
fluid through the packaged atomic reactions. Along the field coordinate ``z``
(stagnation midplane at ``z = 0``, target plate at ``z = L``) it evolves the ion
density and parallel momentum on a *prescribed* parallel temperature profile
(hot upstream, cold target -- the imposed-temperature closure common in reduced
SOL studies, which sidesteps the stiff self-consistent conduction/radiation
energy balance), together with a neutral density that is:

- **recycled** from the Bohm ion flux at the target (fraction ``R``),
- **transported** by parallel diffusion (solved implicitly with a tridiagonal
  solve, so the parabolic term is unconditionally stable), and
- **ionized / recombined** back into the plasma via the AMJUEL rate coefficients
  (an operator-split, per-cell implicit update that is stable against the stiff
  ionization source).

The result is an attached recycling SOL: neutrals born at the target penetrate
upstream, ionize where the plasma is hot, and the plasma flows back to the target
at the Bohm speed. Self-consistent detachment (an *evolved* temperature with
conduction and radiative rollover) is a further extension.

Everything is pure ``jax.numpy`` (with the solvax tridiagonal solve) and therefore
``jit``/``grad``/``vmap`` transparent. Fields are hermes-3 normalized
(density / ``Nnorm``, temperature / ``Tnorm``, velocity / the reference sound
speed); the reaction rates use physical temperature and density internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from solvax import tridiagonal_solve

from .atomic_rates import charge_exchange_rate_coefficient, rate_coefficient
from .reactions import PlasmaNormalization

__all__ = [
    "SolRecyclingParameters",
    "SolRecyclingState",
    "linear_target_temperature_profile",
    "sol_recycling_step",
    "sol_recycling_run",
    "target_ion_flux",
]


@dataclass(frozen=True)
class SolRecyclingParameters:
    """Parameters of the reduced recycling SOL model."""

    parallel_length: float = 30.0            # m, stagnation -> target
    upstream_density: float = 1.0            # fixed upstream ion density (normalized)
    recycling_fraction: float = 1.0          # target recycling fraction R
    neutral_diffusion: float = 8.0           # neutral parallel diffusion enhancement
    neutral_temperature: float = 0.04        # normalized (cold, ~2 eV) neutral temperature
    ion_mass: float = 2.0                    # deuterium
    density_floor: float = 1.0e-4
    normalization: PlasmaNormalization = PlasmaNormalization()


class SolRecyclingState(NamedTuple):
    """Evolved fields: ion density, ion momentum (A n v), neutral density."""

    ion_density: jnp.ndarray
    ion_momentum: jnp.ndarray
    neutral_density: jnp.ndarray


def linear_target_temperature_profile(nz: int, *, upstream_ev: float = 30.0, target_ev: float = 2.0,
                                      normalization: PlasmaNormalization = PlasmaNormalization()) -> jnp.ndarray:
    """Prescribed parallel temperature (normalized): quadratic hot->cold to the target."""

    z = (jnp.arange(nz, dtype=jnp.float64) + 0.5) / nz
    profile_ev = upstream_ev - (upstream_ev - target_ev) * z**2
    return profile_ev / normalization.Tnorm


def _sound_speed(density, temperature, params):
    return jnp.sqrt(jnp.maximum(2.0 * temperature / params.ion_mass, 0.0))


def _velocity(momentum, density, params):
    return momentum / (params.ion_mass * jnp.maximum(density, params.density_floor))


def _hyperbolic_rhs(density, momentum, temperature, params):
    mass = params.ion_mass
    dz = 1.0 / density.shape[0]
    velocity = _velocity(momentum, density, params)
    pressure = 2.0 * density * temperature
    speed = jnp.abs(velocity) + _sound_speed(density, temperature, params)
    face_speed = jnp.maximum(speed[:-1], speed[1:])
    flux_n = 0.5 * (density[:-1] * velocity[:-1] + density[1:] * velocity[1:]) - 0.5 * face_speed * (density[1:] - density[:-1])
    flux_m = 0.5 * (momentum[:-1] * velocity[:-1] + pressure[:-1] + momentum[1:] * velocity[1:] + pressure[1:]) - 0.5 * face_speed * (momentum[1:] - momentum[:-1])
    # noflow stagnation at z=0 (pressure force only); Bohm outflow at the target.
    target_velocity = jnp.maximum(velocity[-1], _sound_speed(density, temperature, params)[-1])
    flux_n = jnp.concatenate([jnp.zeros(1), flux_n, (density[-1] * target_velocity)[None]])
    flux_m = jnp.concatenate([pressure[:1], flux_m, (mass * density[-1] * target_velocity**2 + pressure[-1])[None]])
    d_density = -(flux_n[1:] - flux_n[:-1]) / dz
    d_momentum = -(flux_m[1:] - flux_m[:-1]) / dz
    return d_density, d_momentum


def _implicit_diffusion(field, face_diffusivity, source, dt):
    """Backward-Euler parallel diffusion with no-flux ends (tridiagonal solve)."""

    n = field.shape[0]
    dz = 1.0 / n
    coefficient = dt / dz**2
    lower = jnp.zeros(n).at[1:].set(-coefficient * face_diffusivity)
    upper = jnp.zeros(n).at[:-1].set(-coefficient * face_diffusivity)
    diagonal = jnp.ones(n).at[1:-1].set(1.0 + coefficient * (face_diffusivity[:-1] + face_diffusivity[1:]))
    diagonal = diagonal.at[0].set(1.0 + coefficient * face_diffusivity[0])
    diagonal = diagonal.at[-1].set(1.0 + coefficient * face_diffusivity[-1])
    return tridiagonal_solve(lower, diagonal, upper, field + dt * source, method="thomas")


def _reaction_update(density, momentum, neutral_density, temperature, params, dt):
    """Operator-split implicit ionization/recombination particle exchange, plus the
    charge-exchange + recombination momentum friction that drags the plasma flow
    toward the (stationary) neutrals -- the drag that keeps the target sonic."""

    norm = params.normalization
    physical_temperature = temperature * norm.Tnorm
    density_floor = params.density_floor
    electron_density_m3 = jnp.maximum(density, density_floor) * norm.Nnorm
    # Time is normalized to the parallel transit time L / c_s, so a physical
    # volumetric rate n^2 <sigma v> maps to n^2 <sigma v> * Nnorm * (L / c_s).
    rate_scale = norm.Nnorm * (params.parallel_length / norm.sound_speed)
    ionization = rate_coefficient("d", "iz", physical_temperature, electron_density_m3) * rate_scale
    recombination = rate_coefficient("d", "rec", physical_temperature, electron_density_m3) * rate_scale
    effective_temperature = jnp.clip(2.0 * temperature * norm.Tnorm / params.ion_mass, 0.01, 1.0e4)
    charge_exchange = charge_exchange_rate_coefficient(effective_temperature) * rate_scale
    electron = jnp.maximum(density, density_floor)
    ionization_frequency = ionization * electron
    recombination_frequency = recombination * electron
    total = density + neutral_density
    new_density = (density + dt * ionization_frequency * total) / (1.0 + dt * (ionization_frequency + recombination_frequency))
    # Momentum friction: charge exchange and recombination remove ion parallel
    # momentum (neutrals are ~stationary); implicit drag for stability.
    friction_frequency = charge_exchange * neutral_density + recombination_frequency
    new_momentum = momentum / (1.0 + dt * friction_frequency)
    return jnp.maximum(new_density, density_floor), new_momentum, jnp.maximum(total - new_density, 0.0)


def sol_recycling_step(state, temperature, params, dt):
    """Advance the recycling SOL one operator-split step."""

    density = jnp.maximum(state.ion_density, params.density_floor)
    momentum = state.ion_momentum
    neutral_density = state.neutral_density

    d_density, d_momentum = _hyperbolic_rhs(density, momentum, temperature, params)
    density = jnp.maximum(density + dt * d_density, params.density_floor)
    momentum = momentum + dt * d_momentum

    density, momentum, neutral_density = _reaction_update(density, momentum, neutral_density, temperature, params, dt)

    diffusivity = params.neutral_diffusion * params.neutral_temperature / (params.ion_mass * 0.05)
    face_diffusivity = jnp.full(density.shape[0] - 1, diffusivity)
    sound_speed = _sound_speed(density, temperature, params)
    dz = 1.0 / density.shape[0]
    recycling_source = jnp.zeros(density.shape[0]).at[-1].set(
        params.recycling_fraction * density[-1] * sound_speed[-1] / dz
    )
    neutral_density = jnp.maximum(_implicit_diffusion(neutral_density, face_diffusivity, recycling_source, dt), 0.0)

    # Pin the upstream ion density (Dirichlet) at the end of the step.
    density = density.at[0].set(params.upstream_density)

    return SolRecyclingState(ion_density=density, ion_momentum=momentum, neutral_density=neutral_density)


def sol_recycling_run(state, temperature, params, *, dt, steps):
    """Advance ``steps`` operator-split steps with a jitted ``lax.scan``."""

    @jax.jit
    def _run(initial):
        def body(carry, _):
            return sol_recycling_step(carry, temperature, params, dt), None

        final, _ = jax.lax.scan(body, initial, None, length=steps)
        return final

    return _run(state)


def target_ion_flux(state, temperature, params) -> jnp.ndarray:
    """Bohm ion particle flux to the target plate ``n c_s``."""

    density = jnp.maximum(state.ion_density, params.density_floor)
    return density[-1] * _sound_speed(density, temperature, params)[-1]
