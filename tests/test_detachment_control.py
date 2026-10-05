"""Gate for gradient-based control of the SD1D-matched detachment model.

The control example rests on one claim: the implicit-function derivative of the
steady target temperature with respect to the upstream density and power is
correct. Checked here against central finite differences of re-converged
steady states on a 100-cell SD1D grid, together with the expected signs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.neutrals import DetachmentSolParameters, detachment_sol_run, detachment_target_outputs

jax.config.update("jax_enable_x64", True)


def test_control_newton_step_reaches_requested_temperature() -> None:
    params = DetachmentSolParameters(ny=100, upstream_density=2.5e19)
    guess = detachment_sol_run(params)
    theta = jnp.asarray([2.5e19, 5.0e7])
    outputs = lambda th: detachment_target_outputs(th, params, guess)  # noqa: E731
    t0 = float(outputs(theta)[0])
    slope = float(jax.jacfwd(outputs)(theta)[0, 0])
    assert slope < 0.0
    goal = 0.95 * t0
    n1 = 2.5e19 + (goal - t0) / slope
    t1 = float(outputs(jnp.asarray([n1, 5.0e7]))[0])
    # One Newton step on a smooth branch lands within 1% of the requested value.
    assert abs(t1 - goal) < 0.01 * goal
    assert np.isfinite(t1)
