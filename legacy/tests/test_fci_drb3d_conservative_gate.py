from __future__ import annotations

import math

import jax

from jaxdrb.fci.drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.conservative import energy_drift, energy_time_series_midpoint


def test_fci_drb3d_conservative_gate() -> None:
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
    params = FCIDRB3DParams(
        kappa=0.0,
        alpha=0.0,
        kpar=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        sheath_nu=0.0,
    )
    model = FCIDRB3DModel(params=params, grid=grid)

    key = jax.random.key(0)
    n0 = 1e-3 * jax.random.normal(key, (grid.nz, grid.nx, grid.ny))
    omega0 = 1e-3 * jax.random.normal(jax.random.key(1), (grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DState(n=n0, omega=omega0)

    dy = model.rhs(0.0, y0)
    mass_rate = float(model.mass_rate(dy))
    energy_rate = float(model.energy_rate(y0, dy))
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
