from __future__ import annotations

import jax.numpy as jnp


def minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.abs(a), jnp.abs(b))


def limited_slope_centered(f: jnp.ndarray, limiter: str = "mc") -> jnp.ndarray:
    """Cell-centered slope with Hermes-compatible minmod/MC choices.

    `f` is 1D along the transport axis.
    """

    df = f[1:] - f[:-1]
    if limiter == "none":
        return jnp.zeros_like(f)

    df_b = df[:-1]
    df_f = df[1:]
    if limiter == "mc":
        slope_interior = minmod(minmod(2.0 * df_b, 2.0 * df_f), 0.5 * (df_b + df_f))
    else:
        slope_interior = minmod(df_b, df_f)

    slope = jnp.zeros_like(f)
    slope = slope.at[1:-1].set(slope_interior)
    slope = slope.at[0].set(df[0])
    slope = slope.at[-1].set(df[-1])
    return slope


def reconstruct_lr(f: jnp.ndarray, limiter: str = "mc") -> tuple[jnp.ndarray, jnp.ndarray]:
    slope = limited_slope_centered(f, limiter=limiter)
    return f - 0.5 * slope, f + 0.5 * slope
