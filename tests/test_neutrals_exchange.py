from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.neutrals import NeutralParams


def test_neutral_ionization_conserves_total_particles_when_isolated():
    grid = Grid2D.make(nx=16, ny=16, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)

    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        dealias_on=True,
        neutrals=NeutralParams(enabled=True, Dn0=0.0, nu_ion=2.0, nu_rec=0.0, S0=0.0, nu_sink=0.0),
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    n = 0.5 + 0.01 * jax.random.normal(key, (grid.nx, grid.ny))
    omega = jnp.zeros((grid.nx, grid.ny))
    N = 1.0 + 0.01 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    y = HW2DState(n=n, omega=omega, N=N)

    dy = model.rhs(0.0, y)
    total_rate = jnp.mean(dy.n + dy.N)
    assert jnp.abs(total_rate) < 1e-10
