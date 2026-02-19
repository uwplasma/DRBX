from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
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

    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
    )
    assert y1.N is not None
    Nbar = jnp.mean(y1.N)
    t_end = dt * nsteps
    N0 = 0.2
    expected = N_star + (N0 - N_star) * math.exp(-nu_sink * t_end)
    assert jnp.abs(Nbar - expected) < 2e-8


def test_neutral_charge_exchange_drag_term_on_omega_is_applied() -> None:
    grid = Grid2D.make(nx=8, ny=8, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    nu_cx = 0.3
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
            nu_ion=0.0,
            nu_rec=0.0,
            S0=0.0,
            nu_sink=0.0,
            nu_cx_omega=nu_cx,
        ),
    )
    model = HW2DModel(params=params, grid=grid)

    n = jnp.zeros((grid.nx, grid.ny))
    omega = 0.7 * jnp.ones((grid.nx, grid.ny))
    N = 1.5 * jnp.ones((grid.nx, grid.ny))
    y = HW2DState(n=n, omega=omega, N=N)
    dy = model.rhs(0.0, y)

    expected = -nu_cx * N * omega
    assert jnp.max(jnp.abs(dy.omega - expected)) < 1e-12


def test_neutrals_passive_limit_keeps_hw2d_invariants() -> None:
    """Invariant gate with neutrals enabled but passive (no exchange/source/sink/drag)."""

    grid = Grid2D.make(nx=24, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=0.0,
        alpha=0.0,
        Dn=0.0,
        DOmega=0.0,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        neutrals=NeutralParams(
            enabled=True,
            Dn0=0.0,
            nu_ion=0.0,
            nu_rec=0.0,
            S0=0.0,
            nu_sink=0.0,
            nu_cx_omega=0.0,
        ),
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(7)
    y0 = HW2DState(
        n=1e-2 * jax.random.normal(key, (grid.nx, grid.ny)),
        omega=1e-2 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny)),
        N=1.0 + 1e-2 * jax.random.normal(jax.random.split(key, 3)[2], (grid.nx, grid.ny)),
    )

    dt = 0.01
    nsteps = 300

    d0 = model.diagnostics(y0)
    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
    )
    d1 = model.diagnostics(y1)

    relE = jnp.abs(d1["E"] - d0["E"]) / jnp.maximum(d0["E"], 1e-30)
    relZ = jnp.abs(d1["Z"] - d0["Z"]) / jnp.maximum(d0["Z"], 1e-30)
    assert relE < 1e-4
    assert relZ < 1e-4
