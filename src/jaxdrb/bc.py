from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

BCType = Literal["periodic", "dirichlet", "neumann"]


@dataclass(frozen=True)
class BC1D:
    """1D boundary condition configuration for a scalar field on an interval.

    For the current linear field-line models, BCs are enforced weakly (SAT/relaxation)
    as additional RHS terms localized to the first/last grid points.

    Notes
    -----
    - `neumann` is interpreted as specifying ∂f/∂l at the boundary.
    - `dirichlet` is interpreted as specifying f at the boundary.
    - The enforcement adds terms of the form: -nu * (f - f_target) at the boundary nodes.
    """

    kind: int  # 0 periodic, 1 dirichlet, 2 neumann
    left_value: float = 0.0
    right_value: float = 0.0
    left_grad: float = 0.0
    right_grad: float = 0.0
    nu: float = 0.0  # enforcement rate (0 disables)

    @classmethod
    def periodic(cls) -> "BC1D":
        return cls(kind=0)

    @classmethod
    def dirichlet(cls, *, left: float = 0.0, right: float = 0.0, nu: float = 1.0) -> "BC1D":
        return cls(kind=1, left_value=float(left), right_value=float(right), nu=float(nu))

    @classmethod
    def neumann(cls, *, left: float = 0.0, right: float = 0.0, nu: float = 1.0) -> "BC1D":
        return cls(kind=2, left_grad=float(left), right_grad=float(right), nu=float(nu))

    def mask(self, n: int, dtype) -> jnp.ndarray:
        m = jnp.zeros((n,), dtype=dtype)
        return m.at[0].set(1).at[-1].set(1)


@dataclass(frozen=True)
class BC2D:
    """2D boundary condition configuration for a scalar field on a rectangle."""

    kind_x: int  # 0 periodic, 1 dirichlet, 2 neumann
    kind_y: int
    x_value: float = 0.0
    y_value: float = 0.0
    x_grad: float = 0.0
    y_grad: float = 0.0

    @classmethod
    def periodic(cls) -> "BC2D":
        return cls(kind_x=0, kind_y=0)

    @classmethod
    def dirichlet(cls, *, x: float = 0.0, y: float = 0.0) -> "BC2D":
        return cls(kind_x=1, kind_y=1, x_value=float(x), y_value=float(y))

    @classmethod
    def neumann(cls, *, x: float = 0.0, y: float = 0.0) -> "BC2D":
        return cls(kind_x=2, kind_y=2, x_grad=float(x), y_grad=float(y))


def bc2d_from_strings(
    *,
    bc_x: BCType,
    bc_y: BCType,
    value_x: float = 0.0,
    value_y: float = 0.0,
    grad_x: float = 0.0,
    grad_y: float = 0.0,
) -> BC2D:
    def to_kind(s: BCType) -> int:
        if s == "periodic":
            return 0
        if s == "dirichlet":
            return 1
        if s == "neumann":
            return 2
        raise ValueError(f"Unknown bc type: {s}")

    return BC2D(
        kind_x=to_kind(bc_x),
        kind_y=to_kind(bc_y),
        x_value=float(value_x),
        y_value=float(value_y),
        x_grad=float(grad_x),
        y_grad=float(grad_y),
    )
