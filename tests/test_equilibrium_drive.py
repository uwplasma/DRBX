from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.drive import drive_terms
from jaxdrb.core.terms.ops import ddx, ddy
from jaxdrb.core.terms.sol import sol_masks
from jaxdrb.geometry.plane import Grid2D


def test_equilibrium_drive_uses_sol_profile() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {
                "drive_from_equilibrium_on": True,
                "omega_n": 1.0,
                "omega_Te": 0.0,
            },
            "closure": {
                "sol": {
                    "sol_on": True,
                    "sol_xs": 0.5,
                    "sol_width": 0.1,
                    "sol_n_core": 2.0,
                    "sol_n_sol": 1.0,
                }
            },
            "numerics": {"poisson": "spectral"},
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

    y = jnp.asarray(grid.y)[None, :]
    phi = jnp.sin(y) * jnp.ones((grid.nx, 1))
    omega = system._omega_from_phi(phi, n=jnp.ones_like(phi))

    state = DRBSystemState(
        n=jnp.ones_like(phi),
        omega=omega,
        vpar_e=jnp.zeros_like(phi),
        vpar_i=jnp.zeros_like(phi),
        Te=jnp.ones_like(phi),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, state)
    term = drive_terms(ctx, state)

    mask_closed, _, _ = sol_masks(params, geom)
    n_eq = params.sol_n_sol + (params.sol_n_core - params.sol_n_sol) * mask_closed
    n_eq = jnp.broadcast_to(n_eq, phi.shape)
    omega_n = -ddx(params, geom, jnp.log(n_eq), ctx.bcs.n)
    dphi_dy = ddy(params, geom, ctx.phi, ctx.bcs.phi)
    expected = -(omega_n * dphi_dy)

    assert jnp.allclose(term.n, expected, rtol=1e-10, atol=1e-10)
