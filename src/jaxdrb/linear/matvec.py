from __future__ import annotations

from collections.abc import Callable

import jax

from jaxdrb.models.cold_ion_drb import State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def linear_matvec(
    y0: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    t0: float = 0.0,
) -> Callable[[State], State]:
    """Return a matrix-free matvec v -> J v, where J = d(rhs)/d(y) at y0."""

    def f(y: State) -> State:
        return rhs_nonlinear(t0, y, params, geom, kx=kx, ky=ky)

    _f0, lin = jax.linearize(f, y0)
    return lin

