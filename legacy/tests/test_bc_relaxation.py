from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.bc_relaxation import field_bc_relaxation
from jaxdrb.core.terms.bcs import resolve_bcs
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.fd2d import enforce_bc_relaxation


def _build_geom(
    *,
    bc_x: str,
    bc_y: str,
    value_x: float = 0.0,
    value_y: float = 0.0,
    grad_x: float = 0.0,
    grad_y: float = 0.0,
):
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "bcs": {"bc_enforce_nu": 2.0},
        },
    )
    grid = Grid2D.make(
        nx=4,
        ny=4,
        Lx=1.0,
        Ly=1.0,
        dealias=False,
        bc_x=bc_x,
        bc_y=bc_y,
        bc_value_x=value_x,
        bc_value_y=value_y,
        bc_grad_x=grad_x,
        bc_grad_y=grad_y,
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    return params, geom


def test_field_bc_relaxation_dirichlet_targets() -> None:
    params, geom = _build_geom(bc_x="dirichlet", bc_y="dirichlet", value_x=1.0, value_y=2.0)
    u = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    y = DRBSystemState(
        n=u,
        omega=jnp.zeros_like(u),
        vpar_e=jnp.zeros_like(u),
        vpar_i=jnp.zeros_like(u),
        Te=jnp.zeros_like(u),
        Ti=None,
        psi=None,
        N=None,
    )
    bcs = resolve_bcs(params, geom)
    rhs = field_bc_relaxation(params, geom, y, bcs)

    expected = enforce_bc_relaxation(
        u, dx=geom.grid.dx, dy=geom.grid.dy, bc=bcs.n, nu=float(params.bc_enforce_nu)
    )
    np.testing.assert_allclose(rhs.n, expected, rtol=1e-6, atol=1e-6)


def test_field_bc_relaxation_neumann_targets() -> None:
    params, geom = _build_geom(bc_x="neumann", bc_y="periodic", grad_x=0.5)
    u = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
    y = DRBSystemState(
        n=u,
        omega=jnp.zeros_like(u),
        vpar_e=jnp.zeros_like(u),
        vpar_i=jnp.zeros_like(u),
        Te=jnp.zeros_like(u),
        Ti=None,
        psi=None,
        N=None,
    )
    bcs = resolve_bcs(params, geom)
    rhs = field_bc_relaxation(params, geom, y, bcs)

    expected = enforce_bc_relaxation(
        u, dx=geom.grid.dx, dy=geom.grid.dy, bc=bcs.n, nu=float(params.bc_enforce_nu)
    )
    np.testing.assert_allclose(rhs.n, expected, rtol=1e-6, atol=1e-6)
