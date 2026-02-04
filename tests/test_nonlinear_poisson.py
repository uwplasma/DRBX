from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.fd import inv_laplacian_cg, laplacian as laplacian_fd
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


def test_inv_laplacian_cg_dirichlet_recovers_manufactured_solution():
    from jaxdrb.bc import bc2d_from_strings

    nx = 32
    ny = 28
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Manufactured phi with zero boundary (Dirichlet).
    x = jnp.linspace(0.0, Lx, nx)[:, None]
    y = jnp.linspace(0.0, Ly, ny)[None, :]
    phi = jnp.sin(jnp.pi * x / Lx) * jnp.sin(jnp.pi * y / Ly)
    phi = phi.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)

    bc = bc2d_from_strings(bc_x="dirichlet", bc_y="dirichlet", value_x=0.0, value_y=0.0)
    omega = laplacian_fd(phi, dx, dy, bc)
    phi_rec = inv_laplacian_cg(omega, dx=dx, dy=dy, bc=bc, maxiter=400)

    err = jnp.linalg.norm((phi_rec - phi).ravel()) / jnp.linalg.norm(phi.ravel())
    assert err < 1e-6
