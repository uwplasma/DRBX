from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .geometry import ParityFVGeometry
from .params import ParityFVParams
from .state import ParityFVState


@dataclass(frozen=True)
class ParityFVRHS:
    """Parity-first RHS scaffold.

    This intentionally starts as a no-op/stub so parity terms can be added
    incrementally with one-to-one regression gates.
    """

    params: ParityFVParams
    geom: ParityFVGeometry

    def __call__(self, t: float, y: ParityFVState) -> ParityFVState:
        _ = t
        if y.n.shape != self.params.shape():
            raise ValueError(f"State shape {y.n.shape} != params shape {self.params.shape()}")
        if y.n.shape != self.geom.shape():
            raise ValueError(f"State shape {y.n.shape} != geometry shape {self.geom.shape()}")
        return y.zeros_like()

    def te(self, y: ParityFVState) -> jnp.ndarray:
        return y.pe / jnp.maximum(y.n, self.params.n_floor)
