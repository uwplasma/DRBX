from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.model import FCISlabModel, FCISlabParams, FCISlabState


def test_fci_sheath_budget_gate() -> None:
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
    params = FCISlabParams(nu_par=0.0, sheath_nu=1.3, open_field_line=True)
    model = FCISlabModel(params=params, grid=grid)

    f0 = 5e-3 * jnp.ones((grid.nz, grid.nx, grid.ny))
    y0 = FCISlabState(f=f0)
    dy = model.rhs(0.0, y0)

    mass_rate = jnp.mean(dy.f)
    energy_rate = jnp.mean(y0.f * dy.f)
    mass_budget = -model.sheath_mass_loss(y0)
    energy_budget = -model.sheath_energy_loss(y0)

    assert float(jnp.abs(mass_rate - mass_budget)) < 1e-8
    assert float(jnp.abs(energy_rate - energy_budget)) < 1e-8
