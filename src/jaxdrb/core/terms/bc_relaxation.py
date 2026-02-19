from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import enforce_bc_relaxation

from .bcs import FieldBCs
from .ops import is_2d, grid_of


def field_bc_relaxation(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState, bcs: FieldBCs
) -> DRBSystemState:
    if params.bc_enforce_nu == 0.0 or not is_2d(geom):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    grid = grid_of(geom)
    if grid is None:
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    nu = float(params.bc_enforce_nu)
    return DRBSystemState(
        n=enforce_bc_relaxation(y.n, dx=grid.dx, dy=grid.dy, bc=bcs.n, nu=nu),
        omega=enforce_bc_relaxation(y.omega, dx=grid.dx, dy=grid.dy, bc=bcs.omega, nu=nu),
        vpar_e=enforce_bc_relaxation(y.vpar_e, dx=grid.dx, dy=grid.dy, bc=bcs.vpar_e, nu=nu),
        vpar_i=enforce_bc_relaxation(y.vpar_i, dx=grid.dx, dy=grid.dy, bc=bcs.vpar_i, nu=nu),
        Te=enforce_bc_relaxation(y.Te, dx=grid.dx, dy=grid.dy, bc=bcs.Te, nu=nu),
        Ti=None if y.Ti is None else enforce_bc_relaxation(y.Ti, dx=grid.dx, dy=grid.dy, bc=bcs.Ti, nu=nu),
        psi=None if y.psi is None else enforce_bc_relaxation(y.psi, dx=grid.dx, dy=grid.dy, bc=bcs.psi, nu=nu),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
