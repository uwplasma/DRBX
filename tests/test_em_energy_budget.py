from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.system import DRBSystem
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.fd2d import laplacian


def test_em_energy_rate_includes_psi_term() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"em_on": True, "boussinesq": True},
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

    x = jnp.asarray(grid.x)[:, None]
    y = jnp.asarray(grid.y)[None, :]
    psi = jnp.sin(2.0 * x) * jnp.cos(3.0 * y)
    dy = DRBSystemState(
        n=jnp.zeros_like(psi),
        omega=jnp.zeros_like(psi),
        vpar_e=jnp.zeros_like(psi),
        vpar_i=jnp.zeros_like(psi),
        Te=jnp.zeros_like(psi),
        Ti=None,
        psi=psi,
        N=None,
    )
    y_state = DRBSystemState(
        n=jnp.zeros_like(psi),
        omega=jnp.zeros_like(psi),
        vpar_e=jnp.zeros_like(psi),
        vpar_i=jnp.zeros_like(psi),
        Te=jnp.zeros_like(psi),
        Ti=None,
        psi=psi,
        N=None,
    )

    bc = BC2D.periodic()
    jpar = -laplacian(psi, grid.dx, grid.dy, bc)
    expected = float(params.beta) * jnp.mean(jnp.real(jnp.conj(jpar) * psi))
    got = system.energy_rate(y_state, dy)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=1e-12)
