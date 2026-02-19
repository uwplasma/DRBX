from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.neutrals import NeutralParams


def test_drb2d_neutral_ionization_conserves_total_particles() -> None:
    grid = Grid2D.make(nx=16, ny=16, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=True,
        neutrals=NeutralParams(enabled=True, Dn0=0.0, nu_ion=2.0, nu_rec=0.0, S0=0.0, nu_sink=0.0),
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    n = 0.3 + 0.02 * jax.random.normal(key, (grid.nx, grid.ny))
    omega = jnp.zeros((grid.nx, grid.ny))
    N = 1.1 + 0.02 * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    z = jnp.zeros_like(n)
    y = DRB2DState(n=n, omega=omega, vpar_e=z, vpar_i=z, Te=z, N=N)

    dy = model.rhs(0.0, y)
    total_rate = jnp.mean(dy.n + dy.N)
    assert jnp.abs(total_rate) < 1e-10


def test_drb2d_neutral_source_sink_relaxes_to_equilibrium() -> None:
    grid = Grid2D.make(nx=12, ny=12, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    nu_sink = 0.5
    S0 = 0.8
    N_star = S0 / nu_sink

    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
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
    model = DRB2DModel(params=params, grid=grid)

    z = jnp.zeros((grid.nx, grid.ny))
    y0 = DRB2DState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, N=0.2 * jnp.ones_like(z))

    dt = 0.02
    nsteps = 400
    _, y1 = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
    )

    Nbar = jnp.mean(y1.N)
    t_end = dt * nsteps
    expected = N_star + (0.2 - N_star) * math.exp(-nu_sink * t_end)
    assert jnp.abs(Nbar - expected) < 2e-8


def test_drb2d_neutral_drag_term_on_omega_is_applied() -> None:
    grid = Grid2D.make(nx=8, ny=8, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    nu_cx = 0.4
    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
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
    model = DRB2DModel(params=params, grid=grid)

    z = jnp.zeros((grid.nx, grid.ny))
    omega = 0.6 * jnp.ones_like(z)
    N = 1.2 * jnp.ones_like(z)
    y = DRB2DState(n=z, omega=omega, vpar_e=z, vpar_i=z, Te=z, N=N)
    dy = model.rhs(0.0, y)

    expected = -nu_cx * N * omega
    assert jnp.max(jnp.abs(dy.omega - expected)) < 1e-12
