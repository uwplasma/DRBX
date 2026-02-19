from __future__ import annotations

import jax.numpy as jnp


def d2par_from_dpar(dpar, f: jnp.ndarray) -> jnp.ndarray:
    """Second parallel derivative built from a geometry-provided first derivative."""

    return dpar(dpar(f))


def laplacian_perp_fourier(kperp2: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Fourier-perpendicular Laplacian: ∇_⊥^2 f -> -k_⊥^2(l) f."""

    return -kperp2 * f
