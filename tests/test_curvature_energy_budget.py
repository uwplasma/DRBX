from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.geometry.plane import Grid2D


def test_curvature_energy_budget_closure() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"nonlinear_on": False, "curvature_on": True, "curvature_coeff": 1.0},
            "numerics": {"term_schedule": ("curvature",), "poisson": "spectral"},
        },
    )
    grid = Grid2D.make(
        nx=8,
        ny=8,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    system = DRBSystem(params=params, geom=geom)

    rng = np.random.default_rng(0)
    n = jnp.asarray(rng.normal(size=(grid.nx, grid.ny)))
    omega = jnp.asarray(rng.normal(size=(grid.nx, grid.ny)))
    Te = jnp.asarray(rng.normal(size=(grid.nx, grid.ny)))

    y = DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=jnp.zeros_like(n),
        vpar_i=jnp.zeros_like(n),
        Te=Te,
        Ti=None,
        psi=None,
        N=None,
    )

    budget = system.energy_budget(y)
    total = float(budget["total"])
    curv = float(budget["E_dot_curvature"])
    residual = float(budget["residual"])

    assert abs(curv) > 1e-6
    assert abs(total - curv) < 1e-8
    assert abs(residual) < 1e-10
