from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .geometry import DRBFVGeometry
from .params import DRBFVParams
from .state import DRBFVState


@dataclass(frozen=True)
class DRBFVRHS:
    """Alignment-first RHS scaffold.

    This intentionally starts as a no-op/stub so alignment terms can be added
    incrementally with one-to-one regression gates.
    """

    params: DRBFVParams
    geom: DRBFVGeometry

    def __call__(self, t: float, y: DRBFVState) -> DRBFVState:
        _ = t
        if y.n.shape != self.params.shape():
            raise ValueError(f"State shape {y.n.shape} != params shape {self.params.shape()}")
        if y.n.shape != self.geom.shape():
            raise ValueError(f"State shape {y.n.shape} != geometry shape {self.geom.shape()}")
        return y.zeros_like()

    def te(self, y: DRBFVState) -> jnp.ndarray:
        return y.pe / jnp.maximum(y.n, self.params.n_floor)
