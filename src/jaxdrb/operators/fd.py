from __future__ import annotations

import jax.numpy as jnp


def d1_periodic(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Second-order central difference with periodic BCs."""

    return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2.0 * dx)


def d1_open(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Second-order finite difference for an *open* (non-periodic) 1D grid.

    Uses:
      - centered stencil in the interior
      - 2nd-order one-sided stencils at the boundaries

    This operator is linear and JAX-friendly, and is useful for modeling open field lines
    (e.g. limiter/sheath entrances) where periodic wrapping is not appropriate.
    """
    f = jnp.asarray(f)
    n = int(f.size)
    if n < 3:
        raise ValueError("d1_open requires at least 3 points.")

    # Interior: centered
    df = jnp.zeros_like(f)
    df = df.at[1:-1].set((f[2:] - f[:-2]) / (2.0 * dx))

    # Boundaries: one-sided 2nd order
    df0 = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dx)
    dfN = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dx)
    df = df.at[0].set(df0)
    df = df.at[-1].set(dfN)
    return df
