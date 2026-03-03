from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ParityFVState:
    """State vector for parity-first FV core."""

    n: jnp.ndarray
    pe: jnp.ndarray
    vort: jnp.ndarray
    phi: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray

    def zeros_like(self) -> "ParityFVState":
        z = jnp.zeros_like
        return ParityFVState(
            n=z(self.n),
            pe=z(self.pe),
            vort=z(self.vort),
            phi=z(self.phi),
            vpar_e=z(self.vpar_e),
            vpar_i=z(self.vpar_i),
        )
