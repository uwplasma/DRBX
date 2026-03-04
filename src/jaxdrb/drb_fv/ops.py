from __future__ import annotations

import jax.numpy as jnp


def dpar_centered(f: jnp.ndarray, dz: float) -> jnp.ndarray:
    """Centered derivative along axis-0 with one-sided boundary closure."""

    inv_dz = 1.0 / max(float(dz), 1e-30)
    out = jnp.zeros_like(f)
    if int(f.shape[0]) <= 1:
        return out
    out = out.at[1:-1].set(0.5 * (f[2:] - f[:-2]) * inv_dz)
    out = out.at[0].set((f[1] - f[0]) * inv_dz)
    out = out.at[-1].set((f[-1] - f[-2]) * inv_dz)
    return out


def ddx_centered(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Centered derivative along axis-1 with one-sided boundary closure."""

    inv_dx = 1.0 / max(float(dx), 1e-30)
    out = jnp.zeros_like(f)
    if int(f.shape[1]) <= 1:
        return out
    out = out.at[:, 1:-1, :].set(0.5 * (f[:, 2:, :] - f[:, :-2, :]) * inv_dx)
    out = out.at[:, 0, :].set((f[:, 1, :] - f[:, 0, :]) * inv_dx)
    out = out.at[:, -1, :].set((f[:, -1, :] - f[:, -2, :]) * inv_dx)
    return out
