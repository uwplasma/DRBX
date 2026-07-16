"""Gate: every differentiation method gives the same gradient through turbulence.

Reverse mode, checkpointed reverse mode, and forward mode must agree with each
other to near machine precision and with a central finite difference, on a small
Hasegawa-Wakatani rollout. This is what makes the "pick the most efficient
method" guidance safe: the choice changes cost, never the answer.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.native.hasegawa_wakatani import HasegawaWakataniParameters, hw_grid, hw_step

jax.config.update("jax_enable_x64", True)

N = 32
STEPS = 40
DT = 5.0e-3
GRID = hw_grid(N, 2.0 * np.pi * 8.0)
RNG = np.random.default_rng(0)
Z0 = jnp.fft.fft2(jnp.asarray(1.0e-2 * RNG.standard_normal((N, N))))
M0 = jnp.fft.fft2(jnp.asarray(1.0e-2 * RNG.standard_normal((N, N))))


def evolved_energy(kappa, *, checkpoint=False):
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=kappa, hyperviscosity=1.0e-3)

    def step(carry, _):
        z, m = carry
        return hw_step(z, m, GRID, params, DT), None

    body = jax.checkpoint(step) if checkpoint else step
    (zf, mf), _ = jax.lax.scan(body, (Z0, M0), None, length=STEPS)
    return jnp.real(jnp.sum(jnp.abs(zf) ** 2 + jnp.abs(mf) ** 2)) / (N**4)


def test_all_differentiation_methods_agree() -> None:
    kappa = jnp.asarray(1.0)
    reverse = float(jax.grad(evolved_energy)(kappa))
    checkpointed = float(jax.grad(partial(evolved_energy, checkpoint=True))(kappa))
    forward = float(jax.jacfwd(evolved_energy)(kappa))

    assert checkpointed == pytest.approx(reverse, rel=1e-12)
    assert forward == pytest.approx(reverse, rel=1e-10)

    step = 1e-5
    finite_difference = (float(evolved_energy(kappa + step)) - float(evolved_energy(kappa - step))) / (2 * step)
    assert reverse == pytest.approx(finite_difference, rel=1e-5)
