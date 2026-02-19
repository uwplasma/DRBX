from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.fci.drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from jaxdrb.fci.grid import FCISlabGrid


def test_fci_drb3d_sheath_budget() -> None:
    grid = FCISlabGrid.make(
        nx=32,
        ny=32,
        nz=24,
        Lx=2 * math.pi,
        Ly=2 * math.pi,
        Lz=6.0,
        Bx=0.2,
        By=0.1,
        Bz=1.0,
        open_field_line=True,
    )
    params = FCIDRB3DParams(
        kappa=0.0,
        alpha=0.0,
        kpar=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        sheath_nu=1.2,
    )
    model = FCIDRB3DModel(params=params, grid=grid)

    n0 = 5e-3 * jnp.ones((grid.nz, grid.nx, grid.ny))
    omega0 = jnp.zeros((grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DState(n=n0, omega=omega0)
    dy = model.rhs(0.0, y0)

    mass_rate = model.mass_rate(dy)
    energy_rate = model.energy_rate(y0, dy)

    mass_budget = -params.sheath_nu * jnp.mean(grid.sheath_mask * n0)
    energy_budget = -params.sheath_nu * jnp.mean(grid.sheath_mask * n0**2)

    assert float(jnp.abs(mass_rate - mass_budget)) < 1e-10
    assert float(jnp.abs(energy_rate - energy_budget)) < 1e-10
