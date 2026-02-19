from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.model import FCISlabModel, FCISlabParams, FCISlabState
from jaxdrb.nonlinear.conservative import energy_drift, energy_time_series_midpoint


def test_fci_conservative_invariants_gate() -> None:
    grid = FCISlabGrid.make(
        nx=48,
        ny=48,
        nz=16,
        Lx=2 * math.pi,
        Ly=2 * math.pi,
        Lz=4.0,
        Bx=0.4,
        By=-0.2,
        Bz=1.0,
        open_field_line=False,
    )
    params = FCISlabParams(nu_par=0.0, sheath_nu=0.0, open_field_line=False)
    model = FCISlabModel(params=params, grid=grid)

    key = jax.random.key(0)
    f0 = 1e-3 * jax.random.normal(key, (grid.nz, grid.nx, grid.ny))
    y0 = FCISlabState(f=f0)

    dy = model.rhs(0.0, y0)
    mass_rate = float(jnp.mean(dy.f))
    energy_rate = float(jnp.mean(y0.f * dy.f))
    assert abs(mass_rate) < 5e-8
    assert abs(energy_rate) < 5e-7

    E = energy_time_series_midpoint(
        y0=y0,
        rhs=lambda t, y: model.rhs(t, y),
        energy=model.energy,
        t0=0.0,
        dt=0.05,
        nsteps=80,
        n_iter=6,
    )
    drift = energy_drift(E)
    assert float(drift["rel_span"]) < 2e-3
