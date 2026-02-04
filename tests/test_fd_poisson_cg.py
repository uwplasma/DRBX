from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import inv_laplacian_cg, laplacian


def test_inv_laplacian_cg_dirichlet_recovers_known_solution() -> None:
    nx = 33
    ny = 33
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    bc = BC2D.dirichlet(x=0.0, y=0.0)

    x = jnp.linspace(0.0, Lx, nx)
    y = jnp.linspace(0.0, Ly, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # u=0 at boundaries.
    u = jnp.sin(math.pi * X / Lx) * jnp.sin(math.pi * Y / Ly)
    rhs = laplacian(u, dx, dy, bc)
    u_rec = inv_laplacian_cg(rhs, dx=dx, dy=dy, bc=bc, maxiter=800, tol=1e-12)

    err = jnp.max(jnp.abs(u_rec - u))
    assert float(err) < 2e-3


def test_inv_laplacian_cg_neumann_recovers_solution_up_to_constant() -> None:
    nx = 33
    ny = 33
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    bc = BC2D.neumann(x=0.0, y=0.0)

    x = jnp.linspace(0.0, Lx, nx)
    y = jnp.linspace(0.0, Ly, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # ∂u/∂n = 0 at boundaries.
    u = jnp.cos(math.pi * X / Lx) * jnp.cos(math.pi * Y / Ly)
    rhs = laplacian(u, dx, dy, bc)
    u_rec = inv_laplacian_cg(rhs, dx=dx, dy=dy, bc=bc, maxiter=1200, tol=1e-12)

    # Neumann solution is only defined up to a constant. Compare after removing the mean.
    u0 = u - jnp.mean(u)
    u1 = u_rec - jnp.mean(u_rec)
    err = jnp.max(jnp.abs(u1 - u0))
    assert float(err) < 5e-3
