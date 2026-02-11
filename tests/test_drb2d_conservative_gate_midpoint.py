from __future__ import annotations

import jax

from jaxdrb.nonlinear.conservative import energy_drift, energy_time_series_midpoint
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def test_drb2d_conservative_energy_midpoint_gate() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=20.0, Ly=20.0, dealias=False)
    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n = 1e-3 * jax.random.normal(key, shape)
    omega = 1e-3 * jax.random.normal(jax.random.key(1), shape)
    vpar_e = 1e-3 * jax.random.normal(jax.random.key(2), shape)
    vpar_i = 1e-3 * jax.random.normal(jax.random.key(3), shape)
    Te = 1e-3 * jax.random.normal(jax.random.key(4), shape)
    y0 = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    E = energy_time_series_midpoint(
        y0=y0,
        rhs=lambda t, y: model.rhs(t, y),
        energy=model.energy,
        t0=0.0,
        dt=2e-2,
        nsteps=200,
        n_iter=6,
    )
    drift = energy_drift(E)
    assert float(drift["rel_span"]) < 5e-6
