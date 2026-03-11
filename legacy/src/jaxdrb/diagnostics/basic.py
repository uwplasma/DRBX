from __future__ import annotations

import jax.numpy as jnp


def rms(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(jnp.abs(field) ** 2))


def poloidal_average(field: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """Average over the poloidal (y) direction by default."""

    return jnp.mean(field, axis=axis)


def poloidal_profile(field: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Return a profile along the poloidal direction (average over x by default)."""

    return jnp.mean(field, axis=axis)
