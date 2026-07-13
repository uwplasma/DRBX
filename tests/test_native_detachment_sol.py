"""Gate for the self-consistent detaching SOL model (SD1D detachment, B6).

Scans the upstream density at fixed upstream power and checks the detachment
signatures: the target temperature cools self-consistently from an attached hot
target into the recombining regime below 1 eV, and the target ion flux rises then
**rolls over** -- the SD1D detachment benchmark (Dudson et al., PPCF 61, 065008,
2019). The implicit Spitzer conduction and self-limiting radiative loss keep the
stiff energy balance stable, and the solve is differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.native.neutrals import (
    DetachmentSolParameters,
    DetachmentSolState,
    detachment_diagnostics,
    detachment_sol_run,
)

jax.config.update("jax_enable_x64", True)

NZ = 120
STEPS = 35000


def _relax(upstream_density, *, upstream_power=6.0, steps=STEPS):
    params = DetachmentSolParameters(upstream_density=upstream_density, upstream_power=upstream_power)
    state = DetachmentSolState(
        jnp.full(NZ, upstream_density),
        jnp.zeros(NZ),
        jnp.full(NZ, 2.0 * upstream_density * 0.6),
        jnp.full(NZ, 0.05),
    )
    state = detachment_sol_run(state, params, dt=0.25 * (1.0 / NZ) / 3.0, steps=steps)
    return detachment_diagnostics(state, params), state


def test_detachment_target_flux_rolls_over() -> None:
    densities = [1.5, 3.0, 6.0, 12.0, 40.0]
    flux = []
    temperature = []
    for upstream in densities:
        diagnostics, state = _relax(upstream)
        assert np.all(np.isfinite(np.asarray(state.ion_density)))
        assert np.all(np.isfinite(np.asarray(state.plasma_pressure)))
        flux.append(float(diagnostics.target_ion_flux))
        temperature.append(float(diagnostics.target_temperature_ev))

    flux = np.asarray(flux)
    temperature = np.asarray(temperature)

    # Self-consistent cooling: the target temperature falls monotonically as the
    # upstream density rises.
    assert np.all(np.diff(temperature) < 0.0)
    # Attached at low density (hot target), detached at high density (recombining,
    # below 1 eV).
    assert temperature[0] > 3.0
    assert temperature[-1] < 1.0
    # Target-flux rollover: the flux peaks at an intermediate density (detachment
    # onset) and then falls at the highest density.
    peak_index = int(np.argmax(flux))
    assert 0 < peak_index < len(densities) - 1
    assert flux[-1] < 0.85 * flux[peak_index]
    assert flux[0] < flux[peak_index]


def test_detachment_is_stable_and_differentiable() -> None:
    # A short run is finite and the target flux is differentiable in the upstream
    # power (needed for gradient-based detachment-front control).
    def evolved_target_flux(power):
        params = DetachmentSolParameters(upstream_density=6.0, upstream_power=power)
        state = DetachmentSolState(
            jnp.full(NZ, 6.0), jnp.zeros(NZ), jnp.full(NZ, 2.0 * 6.0 * 0.6), jnp.full(NZ, 0.05)
        )
        state = detachment_sol_run(state, params, dt=0.25 * (1.0 / NZ) / 3.0, steps=4000)
        return detachment_diagnostics(state, params).target_ion_flux

    gradient = float(jax.grad(evolved_target_flux)(jnp.asarray(6.0)))
    assert np.isfinite(gradient)
