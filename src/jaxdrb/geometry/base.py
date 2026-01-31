from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

import jax.numpy as jnp


class Geometry(Protocol):
    """Field-line (flux-tube) geometry coefficients along the parallel coordinate `l`."""

    l: jnp.ndarray
    dl: float

    @abstractmethod
    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:
        """Return k_perp^2(l) for the chosen (kx, ky)."""

    @abstractmethod
    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        """Parallel derivative ∇_|| f evaluated on the l-grid."""

    @abstractmethod
    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        """Curvature operator C(f) (typically ~ i k_y ω_d(l) f)."""
