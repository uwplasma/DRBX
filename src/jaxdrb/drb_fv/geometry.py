from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class DRBFVGeometry:
    """Geometry container for alignment-first FV operators.

    Arrays follow `(nz, nx, ny)` layout.
    """

    jacobian: jnp.ndarray
    bxcv: jnp.ndarray | None = None
    gxx: jnp.ndarray | None = None
    gxy: jnp.ndarray | None = None
    gyy: jnp.ndarray | None = None
    dpar_factor: jnp.ndarray | None = None

    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.jacobian.shape)
