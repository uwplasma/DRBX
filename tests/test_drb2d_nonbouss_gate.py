from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def test_drb2d_nonbouss_energy_budget_closure() -> None:
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DParams(
        omega_n=0.2,
        omega_Te=0.1,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.3,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-3,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        bracket="arakawa",
        poisson="cg_fd",
        dealias_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n0 = 1e-2 * jax.random.normal(key, shape)
    omega0 = 1e-2 * jax.random.normal(jax.random.key(1), shape)
    vpar_e0 = 1e-2 * jax.random.normal(jax.random.key(2), shape)
    vpar_i0 = 1e-2 * jax.random.normal(jax.random.key(3), shape)
    Te0 = 1e-2 * jax.random.normal(jax.random.key(4), shape)
    y = DRB2DState(n=n0, omega=omega0, vpar_e=vpar_e0, vpar_i=vpar_i0, Te=Te0)

    rhs = model.rhs(0.0, y)
    edot_full = float(model.energy_rate(y, rhs))
    edot_budget = float(model.energy_budget(y)["E_dot_total"])
    assert abs(edot_full - edot_budget) / max(abs(edot_full), 1e-12) < 5e-7


def test_drb2d_nonbouss_conservative_energy_drift_small() -> None:
    grid = Grid2D.make(nx=24, ny=24, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-3,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="cg_fd",
        dealias_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(7)
    shape = (grid.nx, grid.ny)
    n0 = 5e-3 * jax.random.normal(key, shape)
    omega0 = 5e-3 * jax.random.normal(jax.random.key(1), shape)
    vpar_e0 = 5e-3 * jax.random.normal(jax.random.key(2), shape)
    vpar_i0 = 5e-3 * jax.random.normal(jax.random.key(3), shape)
    Te0 = 5e-3 * jax.random.normal(jax.random.key(4), shape)
    y0 = DRB2DState(n=n0, omega=omega0, vpar_e=vpar_e0, vpar_i=vpar_i0, Te=Te0)

    dt = 0.02
    nsteps = 200
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
    )
    Es = jax.vmap(model.energy)(ys)
    rel_span = float((jnp.max(Es) - jnp.min(Es)) / jnp.maximum(jnp.abs(Es[0]), 1e-30))
    assert rel_span < 5e-4
