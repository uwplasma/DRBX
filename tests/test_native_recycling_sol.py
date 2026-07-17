"""Gate for the coupled 1D recycling SOL model (plasma + diffusive neutral).

Checks that the operator-split coupled model is stable and reproduces the key
recycling-SOL physics: ionization localizes in the hot upstream and
recombination at the cold target; neutrals recycled at the target build up a
cushion there; and as the upstream density rises the charge-exchange friction
chokes the parallel flow (the target Mach number falls) -- the onset of
detachment. The whole solve is differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dkx.native.neutrals.atomic_rates import rate_coefficient
from dkx.native.neutrals.recycling_sol_model import (
    SolRecyclingParameters,
    SolRecyclingState,
    linear_target_temperature_profile,
    sol_recycling_run,
    target_ion_flux,
)

jax.config.update("jax_enable_x64", True)

NZ = 96


def _relax(upstream_density, *, steps=30000, recycling=0.95):
    params = SolRecyclingParameters(
        upstream_density=upstream_density, recycling_fraction=recycling, neutral_diffusion=8.0
    )
    temperature = linear_target_temperature_profile(NZ, upstream_ev=30.0, target_ev=1.5)
    state = SolRecyclingState(
        jnp.full(NZ, upstream_density), jnp.zeros(NZ), jnp.full(NZ, 0.05)
    )
    dt = 0.3 * (1.0 / NZ) / 3.0
    state = sol_recycling_run(state, temperature, params, dt=dt, steps=steps)
    return state, temperature, params


def test_recycling_sol_is_stable_and_positive() -> None:
    state, _temperature, params = _relax(2.0)
    density = np.asarray(state.ion_density)
    neutral = np.asarray(state.neutral_density)
    assert np.all(np.isfinite(density)) and np.all(np.isfinite(neutral))
    assert np.all(np.isfinite(np.asarray(state.ion_momentum)))
    assert float(density.min()) > 0.0
    assert float(neutral.min()) >= 0.0
    # Upstream density is held at the imposed value.
    assert density[0] == params.upstream_density


def test_ionization_upstream_recombination_at_target() -> None:
    state, temperature, params = _relax(4.0)
    density = np.asarray(state.ion_density)
    neutral = np.asarray(state.neutral_density)
    temperature_ev = np.asarray(temperature) * params.normalization.Tnorm
    density_m3 = np.maximum(density, params.density_floor) * params.normalization.Nnorm
    ionization = np.asarray(rate_coefficient("d", "iz", jnp.asarray(temperature_ev), jnp.asarray(density_m3)))
    recombination = np.asarray(rate_coefficient("d", "rec", jnp.asarray(temperature_ev), jnp.asarray(density_m3)))
    # The ionization coefficient is strongest in the hot upstream; the
    # recombination coefficient is strongest at the cold target.
    assert ionization[0] > 10.0 * ionization[-1]
    assert recombination[-1] > 10.0 * recombination[0]
    # Recycled neutrals build a cushion at the target (denser than upstream).
    assert neutral[-1] > neutral[0]


def test_charge_exchange_friction_chokes_flow_at_high_density() -> None:
    # Detachment onset: raising the upstream density lowers the target Mach number
    # because charge-exchange + recombination friction slow the parallel flow.
    def target_mach(upstream_density):
        state, temperature, params = _relax(upstream_density, steps=30000)
        density = np.asarray(state.ion_density)
        momentum = np.asarray(state.ion_momentum)
        sound_speed = np.sqrt(2.0 * np.asarray(temperature) / params.ion_mass)
        velocity = momentum / (params.ion_mass * np.maximum(density, params.density_floor))
        return float(velocity[-1] / sound_speed[-1])

    assert target_mach(8.0) < target_mach(1.0)


def test_target_flux_is_differentiable_in_upstream_density() -> None:
    temperature = linear_target_temperature_profile(NZ, upstream_ev=30.0, target_ev=1.5)
    dt = 0.3 * (1.0 / NZ) / 3.0

    def evolved_target_flux(upstream_density):
        params = SolRecyclingParameters(upstream_density=upstream_density, recycling_fraction=0.95)
        state = SolRecyclingState(
            jnp.full(NZ, upstream_density), jnp.zeros(NZ), jnp.full(NZ, 0.05)
        )
        state = sol_recycling_run(state, temperature, params, dt=dt, steps=3000)
        return target_ion_flux(state, temperature, params)

    gradient = float(jax.grad(evolved_target_flux)(jnp.asarray(2.0)))
    assert np.isfinite(gradient) and gradient > 0.0
