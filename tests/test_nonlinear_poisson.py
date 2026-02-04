from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.spectral import inv_laplacian, laplacian


def test_inv_laplacian_inverts_laplacian_zero_mean_gauge():
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * jnp.pi, Ly=2 * jnp.pi)
    key = jax.random.key(0)
    omega = jax.random.normal(key, (grid.nx, grid.ny))
    omega = omega - jnp.mean(omega)

    phi = inv_laplacian(omega, grid.k2)
    omega_rec = laplacian(phi, grid.k2)

    # Relative error should be small, excluding numerical precision.
    err = jnp.linalg.norm((omega_rec - omega).ravel()) / jnp.linalg.norm(omega.ravel())
    assert err < 1e-10


def test_hw2d_short_run_no_nans():
    from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
    from jaxdrb.nonlinear.stepper import rk4_scan

    grid = Grid2D.make(nx=24, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    model = HW2DModel(
        params=HW2DParams(kappa=1.0, alpha=0.5, Dn=1e-3, DOmega=1e-3, bracket="spectral"),
        grid=grid,
    )
    y0 = hw2d_random_ic(jax.random.key(2), grid, amp=1e-3)

    def rhs(t, y):
        return model.rhs(t, y)

    _, y_end = rk4_scan(y0, t0=0.0, dt=0.05, nsteps=10, rhs=rhs)
    assert jnp.all(jnp.isfinite(y_end.n))
    assert jnp.all(jnp.isfinite(y_end.omega))
