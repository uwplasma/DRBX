from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.geometry.plane import Grid2D


def test_energy_budget_includes_diamag_pol_and_braginskii() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"diamagnetic_polarisation_on": True, "hot_ion_on": True},
            "transport": {
                "braginskii_heat_exchange_on": True,
                "braginskii_friction_on": True,
                "classical_diffusion_on": True,
            },
            "numerics": {
                "poisson": "spectral",
                "term_schedule": [
                    "classical_diffusion",
                    "braginskii_friction",
                    "braginskii_heat_exchange",
                ],
            },
            "closure": {"sol": {"sol_on": False}, "sheath": {"sheath_on": False}},
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
    Ti = jnp.asarray(rng.normal(size=(grid.nx, grid.ny)))

    y = DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=jnp.zeros_like(n),
        vpar_i=jnp.zeros_like(n),
        Te=Te,
        Ti=Ti,
        psi=None,
        N=None,
    )

    budget = system.energy_budget(y)
    assert "E_dot_diamagnetic_polarisation" in budget
    assert "E_dot_braginskii_heat_exchange" in budget
    assert "E_dot_braginskii_friction" in budget
    assert "E_dot_classical_diffusion" in budget
