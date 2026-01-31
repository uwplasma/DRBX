from __future__ import annotations

import jax.numpy as jnp


def d1_periodic(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Second-order central difference with periodic BCs."""

    return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2.0 * dx)
