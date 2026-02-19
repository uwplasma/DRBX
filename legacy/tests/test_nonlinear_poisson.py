from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.fd import (
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    laplacian as laplacian_fd,
    div_n_grad,
)
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
    from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps

    grid = Grid2D.make(nx=24, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    model = HW2DModel(
        params=HW2DParams(kappa=1.0, alpha=0.5, Dn=1e-3, DOmega=1e-3, bracket="spectral"),
        grid=grid,
    )
    y0 = hw2d_random_ic(jax.random.key(2), grid, amp=1e-3)

    def rhs(t, y):
        return model.rhs(t, y)

    _, y_end = diffeqsolve_fixed_steps(rhs, y0=y0, t0=0.0, dt=0.05, nsteps=10)
    assert jnp.all(jnp.isfinite(y_end.n))
    assert jnp.all(jnp.isfinite(y_end.omega))


def test_hw2d_short_run_no_nans_cg_fd_poisson():
    from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
    from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps

    grid = Grid2D.make(nx=16, ny=16, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    model = HW2DModel(
        params=HW2DParams(
            kappa=1.0, alpha=0.5, Dn=1e-3, DOmega=1e-3, bracket="arakawa", poisson="cg_fd"
        ),
        grid=grid,
    )
    y0 = hw2d_random_ic(jax.random.key(3), grid, amp=1e-3)

    _, y_end = diffeqsolve_fixed_steps(model.rhs, y0=y0, t0=0.0, dt=0.05, nsteps=5)
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


def test_inv_laplacian_cg_periodic_recovers_manufactured_solution():
    grid = Grid2D.make(nx=32, ny=28, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    key = jax.random.key(42)

    # Manufactured discrete solution (mean-zero gauge).
    phi = jax.random.normal(key, (grid.nx, grid.ny))
    phi = phi - jnp.mean(phi)
    omega = laplacian_fd(phi, grid.dx, grid.dy, grid.bc)

    phi_rec = inv_laplacian_cg(omega, dx=grid.dx, dy=grid.dy, bc=grid.bc, maxiter=800, tol=1e-12)
    phi_rec = phi_rec - jnp.mean(phi_rec)

    err = jnp.linalg.norm((phi_rec - phi).ravel()) / jnp.linalg.norm(phi.ravel())
    assert err < 1e-6


def test_inv_laplacian_cg_neumann_recovers_manufactured_solution():
    from jaxdrb.bc import bc2d_from_strings

    nx = 32
    ny = 28
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Manufactured phi with homogeneous Neumann BCs.
    x = jnp.linspace(0.0, Lx, nx)[:, None]
    y = jnp.linspace(0.0, Ly, ny)[None, :]
    phi = jnp.cos(jnp.pi * x / Lx) * jnp.cos(jnp.pi * y / Ly)
    phi = phi - jnp.mean(phi)

    bc = bc2d_from_strings(bc_x="neumann", bc_y="neumann", grad_x=0.0, grad_y=0.0)
    omega = laplacian_fd(phi, dx, dy, bc)
    phi_rec = inv_laplacian_cg(omega, dx=dx, dy=dy, bc=bc, maxiter=800, tol=1e-12)
    phi_rec = phi_rec - jnp.mean(phi_rec)

    err = jnp.linalg.norm((phi_rec - phi).ravel()) / jnp.linalg.norm(phi.ravel())
    assert err < 1e-6


def test_inv_div_n_grad_cg_periodic_recovers_manufactured_solution():
    grid = Grid2D.make(nx=32, ny=28, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    x = jnp.linspace(0.0, grid.Lx, grid.nx, endpoint=False)[:, None]
    y = jnp.linspace(0.0, grid.Ly, grid.ny, endpoint=False)[None, :]

    phi = jnp.sin(2.0 * x) * jnp.sin(3.0 * y)
    n_coeff = 1.0 + 0.2 * jnp.sin(x) + 0.1 * jnp.cos(y)
    n_coeff = jnp.maximum(n_coeff, 0.2)

    rhs = -div_n_grad(phi, n_coeff, grid.dx, grid.dy, grid.bc)
    phi_rec = inv_div_n_grad_cg(
        rhs,
        n_coeff=n_coeff,
        dx=grid.dx,
        dy=grid.dy,
        bc=grid.bc,
        maxiter=800,
        tol=1e-12,
        preconditioner="spectral",
    )
    phi_rec = phi_rec - jnp.mean(phi_rec)
    phi0 = phi - jnp.mean(phi)

    err = jnp.linalg.norm((phi_rec - phi0).ravel()) / jnp.linalg.norm(phi0.ravel())
    assert err < 2e-4


def test_drb2d_short_run_no_nans_cg_fd_poisson():
    from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
    from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps

    grid = Grid2D.make(nx=16, ny=16, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    model = DRB2DModel(
        params=DRB2DParams(
            omega_n=1.0,
            omega_Te=0.25,
            curvature_on=True,
            curvature_coeff=0.5,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            bracket="arakawa",
            poisson="cg_fd",
            dealias_on=False,
        ),
        grid=grid,
    )

    key = jax.random.key(0)
    kn, kw = jax.random.split(key, 2)
    n0 = 1e-3 * jax.random.normal(kn, (grid.nx, grid.ny))
    omega0 = 1e-3 * jax.random.normal(kw, (grid.nx, grid.ny))
    z = jnp.zeros_like(n0)
    y0 = DRB2DState(n=n0, omega=omega0, vpar_e=z, vpar_i=z, Te=z)

    _, y_end = diffeqsolve_fixed_steps(model.rhs, y0=y0, t0=0.0, dt=0.05, nsteps=5)
    assert jnp.all(jnp.isfinite(y_end.n))
    assert jnp.all(jnp.isfinite(y_end.omega))
    assert jnp.all(jnp.isfinite(y_end.vpar_e))
    assert jnp.all(jnp.isfinite(y_end.vpar_i))
    assert jnp.all(jnp.isfinite(y_end.Te))
