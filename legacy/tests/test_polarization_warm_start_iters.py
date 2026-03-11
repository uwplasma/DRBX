from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.operators.fd2d import div_n_grad, inv_div_n_grad_cg


def test_polarization_warm_start_iters_reduces_iterations() -> None:
    jax.config.update("jax_enable_x64", True)

    nx, ny = 8, 8
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    bc = BC2D.periodic()

    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    phi_true = jnp.sin(2.0 * jnp.pi * X / Lx) * jnp.sin(2.0 * jnp.pi * Y / Ly)

    n_eff = jnp.ones_like(phi_true)
    rhs = -div_n_grad(phi_true, n_eff, dx, dy, bc)

    phi0, it0 = inv_div_n_grad_cg(
        rhs,
        n_coeff=n_eff,
        dx=dx,
        dy=dy,
        bc=bc,
        maxiter=200,
        tol=1e-12,
        return_iters=True,
    )
    phi1, it1 = inv_div_n_grad_cg(
        rhs,
        n_coeff=n_eff,
        dx=dx,
        dy=dy,
        bc=bc,
        maxiter=200,
        tol=1e-12,
        x0=phi_true,
        return_iters=True,
    )

    _ = phi0
    _ = phi1
    it0 = int(jax.device_get(it0))
    it1 = int(jax.device_get(it1))
    assert it1 <= it0
    assert it1 <= 1
