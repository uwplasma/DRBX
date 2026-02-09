from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.neutrals import NeutralParams
from jaxdrb.nonlinear.stepper import rk4_step


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


def test_neutral_ionization_plus_recombination_conserves_total_particles():
    grid = Grid2D.make(nx=16, ny=16, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        dealias_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=0.0,
            nu_ion=1.2,
            nu_rec=0.7,
            S0=0.0,
            nu_sink=0.0,
            n_background=1.0,
        ),
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(10)
    n = 0.03 * jax.random.normal(key, (grid.nx, grid.ny))
    omega = jnp.zeros((grid.nx, grid.ny))
    N = 0.9 + 0.02 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    y = HW2DState(n=n, omega=omega, N=N)

    dy = model.rhs(0.0, y)
    assert jnp.abs(jnp.mean(dy.n + dy.N)) < 1e-10


def test_neutral_source_sink_uniform_relaxes_to_analytic_equilibrium():
    """GBS-style sanity check: 0D source/sink neutral model matches analytic equilibrium."""

    grid = Grid2D.make(nx=12, ny=12, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    nu_sink = 0.6
    S0 = 0.9
    N_star = S0 / nu_sink

    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=True,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=0.0,
            nu_ion=0.0,
            nu_rec=0.0,
            S0=S0,
            nu_sink=nu_sink,
            n_background=1.0,
        ),
    )
    model = HW2DModel(params=params, grid=grid)

    y0 = HW2DState(
        n=jnp.zeros((grid.nx, grid.ny)),
        omega=jnp.zeros((grid.nx, grid.ny)),
        N=0.2 * jnp.ones((grid.nx, grid.ny)),
    )

    dt = 0.02
    nsteps = 500

    @jax.jit
    def integrate(y_init: HW2DState) -> HW2DState:
        def body(i, carry):
            t, y_ = carry
            y_next = rk4_step(y_, t, dt, model.rhs)
            return (t + dt, y_next)

        _, y_end = jax.lax.fori_loop(0, nsteps, body, (jnp.asarray(0.0), y_init))
        return y_end

    y1 = integrate(y0)
    assert y1.N is not None
    Nbar = jnp.mean(y1.N)
    t_end = dt * nsteps
    N0 = 0.2
    expected = N_star + (N0 - N_star) * math.exp(-nu_sink * t_end)
    assert jnp.abs(Nbar - expected) < 2e-8
