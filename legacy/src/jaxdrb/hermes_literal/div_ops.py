"""Literal Hermes divergence operators.

Source of truth:
- `/Users/rogerio/local/hermes-3/src/div_ops.cxx`
"""

from __future__ import annotations

import jax.numpy as jnp


def _broadcast_metric(arr: jnp.ndarray | None, shape: tuple[int, ...]) -> jnp.ndarray | None:
    if arr is None:
        return None
    out = jnp.asarray(arr, dtype=jnp.float64)
    if out.ndim == 1:
        out = out[:, None, None]
    elif out.ndim == 2:
        out = out[None, :, :]
    return jnp.broadcast_to(out, shape)


def _pad_open(
    f: jnp.ndarray,
    *,
    ghost_low: jnp.ndarray | None,
    ghost_high: jnp.ndarray | None,
) -> jnp.ndarray:
    low = f[0] if ghost_low is None else jnp.asarray(ghost_low, dtype=f.dtype)
    high = f[-1] if ghost_high is None else jnp.asarray(ghost_high, dtype=f.dtype)
    return jnp.concatenate([low[None, ...], f, high[None, ...]], axis=0)


def div_par_centered(
    f: jnp.ndarray,
    *,
    dz: float,
    dy: jnp.ndarray | None = None,
    J: jnp.ndarray | None = None,
    gpar: jnp.ndarray | None = None,
    dpar_factor: jnp.ndarray | None = None,
    sign: float = 1.0,
    ghost_low: jnp.ndarray | None = None,
    ghost_high: jnp.ndarray | None = None,
    boundary_flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Literal centered open-boundary divergence for Hermes `Div_par(jpar)`."""

    f_arr = jnp.asarray(f, dtype=jnp.float64)
    fp = _pad_open(f_arr, ghost_low=ghost_low, ghost_high=ghost_high)
    face = 0.5 * (fp[:-1] + fp[1:])
    face_interior = face[1:-1]

    Jc = _broadcast_metric(J, f_arr.shape)
    gpar_c = _broadcast_metric(gpar, f_arr.shape)
    dy_c = _broadcast_metric(dy, f_arr.shape)
    div = jnp.zeros_like(f_arr)

    if dy_c is None:
        dy_c = jnp.full_like(f_arr, float(dz))

    if Jc is None:
        flux_factor_rc = 1.0 / jnp.maximum(dy_c[:-1], 1e-30)
        flux_factor_rp = 1.0 / jnp.maximum(dy_c[1:], 1e-30)
        boundary_factor_low = 1.0 / jnp.maximum(dy_c[0], 1e-30)
        boundary_factor_high = 1.0 / jnp.maximum(dy_c[-1], 1e-30)
    elif gpar_c is None:
        J_face = 0.5 * (Jc[:-1] + Jc[1:])
        flux_factor_rc = J_face / (jnp.maximum(dy_c[:-1], 1e-30) * jnp.maximum(Jc[:-1], 1e-30))
        flux_factor_rp = J_face / (jnp.maximum(dy_c[1:], 1e-30) * jnp.maximum(Jc[1:], 1e-30))
        boundary_factor_low = 1.0 / jnp.maximum(dy_c[0], 1e-30)
        boundary_factor_high = 1.0 / jnp.maximum(dy_c[-1], 1e-30)
    else:
        sqrt_g = jnp.sqrt(jnp.maximum(gpar_c, 1e-30))
        common = (Jc[:-1] + Jc[1:]) / (sqrt_g[:-1] + sqrt_g[1:])
        flux_factor_rc = common / (jnp.maximum(dy_c[:-1], 1e-30) * jnp.maximum(Jc[:-1], 1e-30))
        flux_factor_rp = common / (jnp.maximum(dy_c[1:], 1e-30) * jnp.maximum(Jc[1:], 1e-30))
        boundary_factor_low = 1.0 / (jnp.maximum(dy_c[0], 1e-30) * jnp.maximum(sqrt_g[0], 1e-30))
        boundary_factor_high = 1.0 / (jnp.maximum(dy_c[-1], 1e-30) * jnp.maximum(sqrt_g[-1], 1e-30))

    if face_interior.shape[0] > 0:
        div = div.at[:-1].add(face_interior * flux_factor_rc)
        div = div.at[1:].add(-face_interior * flux_factor_rp)
    div = div.at[0].add(-face[0] * boundary_factor_low * float(boundary_flux_scale))
    div = div.at[-1].add(face[-1] * boundary_factor_high * float(boundary_flux_scale))

    if dpar_factor is not None and gpar_c is None:
        div = div * jnp.asarray(dpar_factor, dtype=jnp.float64)
    return float(sign) * div
