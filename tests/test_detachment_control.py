"""Gate for gradient-based detachment control.

The control example rests on one claim: forward-mode autodiff gives the correct
sensitivity of the target temperature to the upstream density *through the
entire stiff operator-split SOL solve*. This gate checks that derivative
against a central finite difference in the smooth attached regime, and that it
carries the right physics (raising the upstream density cools the target).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.native.neutrals import (
    DetachmentSolParameters,
    DetachmentSolState,
    detachment_diagnostics,
    detachment_sol_run,
)

jax.config.update("jax_enable_x64", True)

NZ = 64
STEPS = 6000


def target_temperature(upstream_density):
    params = DetachmentSolParameters(upstream_density=upstream_density, upstream_power=6.0)
    state = DetachmentSolState(
        jnp.full(NZ, upstream_density),
        jnp.zeros(NZ),
        jnp.full(NZ, 2.0 * upstream_density * 0.6),
        jnp.full(NZ, 0.05),
    )
    state = detachment_sol_run(state, params, dt=0.25 * (1.0 / NZ) / 3.0, steps=STEPS)
    return detachment_diagnostics(state, params).target_temperature_ev


def test_forward_mode_sensitivity_matches_finite_difference() -> None:
    # Attached, smooth regime: the autodiff sensitivity through the whole solve
    # must match a central finite difference.
    density = jnp.asarray(2.0)
    derivative = float(jax.jacfwd(target_temperature)(density))
    step = 1e-3
    finite_difference = (float(target_temperature(density + step))
                         - float(target_temperature(density - step))) / (2 * step)
    assert np.isfinite(derivative)
    assert derivative == pytest.approx(finite_difference, rel=1e-4)
    # The physics: raising the upstream density cools the target.
    assert derivative < 0.0


def test_target_cools_monotonically_toward_detachment() -> None:
    temperatures = [float(target_temperature(jnp.asarray(n))) for n in (2.0, 3.5, 6.0)]
    assert temperatures[0] > temperatures[1] > temperatures[2]
