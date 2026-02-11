from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState


def test_drb2d_reduces_to_hw2d_ideal_limit() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=20.0, Ly=20.0, dealias=False)

    drb2d_params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        curvature_on=False,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    hw2d_params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
    )

    model_drb2d = DRB2DModel(params=drb2d_params, grid=grid)
    model_hw2d = HW2DModel(params=hw2d_params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n = 1e-3 * jax.random.normal(key, shape)
    omega = 1e-3 * jax.random.normal(jax.random.key(1), shape)

    y_drb2d = DRB2DState(
        n=n,
        omega=omega,
        vpar_e=jnp.zeros_like(n),
        vpar_i=jnp.zeros_like(n),
        Te=jnp.zeros_like(n),
    )
    y_hw2d = HW2DState(n=n, omega=omega, N=None)

    rhs_drb2d = model_drb2d.rhs(0.0, y_drb2d)
    rhs_hw2d = model_hw2d.rhs(0.0, y_hw2d)

    assert bool(jnp.allclose(rhs_drb2d.n, rhs_hw2d.n, rtol=1e-8, atol=1e-10))
    assert bool(jnp.allclose(rhs_drb2d.omega, rhs_hw2d.omega, rtol=1e-8, atol=1e-10))
