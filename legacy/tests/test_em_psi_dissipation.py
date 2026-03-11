from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.geometry.plane import Grid2D


def test_em_psi_dissipation_energy_budget_negative() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"em_on": True, "boussinesq": True, "eta": 0.5, "beta": 1.0},
            "transport": {"Dpsi": 0.2, "chi_par": 0.1},
            "numerics": {"poisson": "spectral", "term_schedule": ["diffusion"]},
        },
    )
    grid = Grid2D.make(
        nx=32,
        ny=32,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    system = DRBSystem(params=params, geom=geom)

    x = jnp.asarray(grid.x)[:, None]
    y = jnp.asarray(grid.y)[None, :]
    psi = jnp.sin(2.0 * x) * jnp.cos(3.0 * y)
    zeros = jnp.zeros_like(psi)
    state = DRBSystemState(
        n=zeros,
        omega=zeros,
        vpar_e=zeros,
        vpar_i=zeros,
        Te=zeros,
        Ti=None,
        psi=psi,
        N=None,
    )

    budget = system.energy_budget(state)
    assert "E_dot_diffusion" in budget
    assert float(budget["E_dot_diffusion"]) < 0.0

    state_zero = DRBSystemState(
        n=zeros,
        omega=zeros,
        vpar_e=zeros,
        vpar_i=zeros,
        Te=zeros,
        Ti=None,
        psi=zeros,
        N=None,
    )
    budget_zero = system.energy_budget(state_zero)
    assert float(budget["E_dot_diffusion"]) < float(budget_zero["E_dot_diffusion"])
