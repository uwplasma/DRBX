from __future__ import annotations

import numpy as np

import jax

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def test_drb2d_energy_budget_closure_with_curvature_and_drives() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DParams(
        omega_n=0.8,
        omega_Te=0.3,
        kpar=0.0,
        eta=0.2,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.6,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n0 = 1e-3 * jax.random.normal(key, shape)
    omega0 = 1e-3 * jax.random.normal(jax.random.key(1), shape)
    vpar_e0 = 1e-3 * jax.random.normal(jax.random.key(2), shape)
    vpar_i0 = 1e-3 * jax.random.normal(jax.random.key(3), shape)
    Te0 = 1e-3 * jax.random.normal(jax.random.key(4), shape)
    y = DRB2DState(n=n0, omega=omega0, vpar_e=vpar_e0, vpar_i=vpar_i0, Te=Te0)

    rhs = model.rhs(0.0, y)
    edot_full = float(model.energy_rate(y, rhs))
    edot_budget = float(model.energy_budget(y)["E_dot_total"])
    assert abs(edot_full - edot_budget) / max(abs(edot_full), 1e-12) < 1e-10

    dt = 0.02
    nsteps = 80
    save_ts = np.arange(dt, dt * (nsteps + 1), dt)
    sol = model.diffeqsolve(
        y0=y,
        t0=0.0,
        t1=float(dt * nsteps),
        dt0=float(dt),
        save_ts=jax.numpy.asarray(save_ts),
        solver="dopri8",
        adaptive=False,
        rtol=1e-8,
        atol=1e-10,
        max_steps=20_000,
        progress=False,
    )
    Es = np.asarray(jax.device_get(jax.vmap(model.energy)(sol.ys)))
    Edot = np.asarray(
        jax.device_get(jax.vmap(lambda yi: model.energy_budget(yi)["E_dot_total"])(sol.ys))
    )
    dE_dt_fd = np.gradient(Es, dt)
    corr = float(np.corrcoef(dE_dt_fd, Edot)[0, 1])
    assert corr > 0.9
