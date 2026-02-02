from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear
from jaxdrb.models.params import DRBParams


def linear_matvec(
    y0: State,
    params: DRBParams,
    geom,
    *,
    kx: float,
    ky: float,
    t0: float = 0.0,
    eq: Equilibrium | None = None,
) -> Callable[[State], State]:
    """Return a matrix-free matvec v -> J v, where J = d(rhs)/d(y) at y0."""

    def f(y: State) -> State:
        return rhs_nonlinear(t0, y, params, geom, kx=kx, ky=ky, eq=eq)

    _f0, lin = jax.linearize(f, y0)
    return lin


def linear_matvec_from_rhs(
    rhs: Callable[..., Any],
    y0: Any,
    params: Any,
    geom: Any,
    *,
    kx: float,
    ky: float,
    t0: float = 0.0,
    rhs_kwargs: dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """Generic matrix-free matvec v -> J v for an arbitrary RHS and PyTree state."""

    rhs_kwargs = {} if rhs_kwargs is None else dict(rhs_kwargs)

    def f(y: Any) -> Any:
        return rhs(t0, y, params, geom, kx=kx, ky=ky, **rhs_kwargs)

    _f0, lin = jax.linearize(f, y0)
    return lin
