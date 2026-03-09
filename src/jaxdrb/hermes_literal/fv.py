"""Literal Hermes finite-volume operators.

Source of truth:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/fv_ops.hxx`
"""

from __future__ import annotations

import jax.numpy as jnp


def _minmod_pair(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    same_sign = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return same_sign * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _reconstruct_cell_edges(f: jnp.ndarray, limiter: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    c = f[1:-1]
    m = f[:-2]
    p = f[2:]
    limiter = str(limiter).lower()
    if limiter == "none":
        slope = 0.5 * (p - m)
    elif limiter == "mc":
        slope = _minmod_pair(_minmod_pair(2.0 * (p - c), 2.0 * (c - m)), 0.5 * (p - m))
    else:
        slope = _minmod_pair(p - c, c - m)
    return c - 0.5 * slope, c + 0.5 * slope


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


def div_par_mod(
    f: jnp.ndarray,
    v: jnp.ndarray,
    wave_speed: jnp.ndarray,
    *,
    dz: float,
    dy: jnp.ndarray | None = None,
    limiter: str,
    J: jnp.ndarray | None = None,
    gpar: jnp.ndarray | None = None,
    dpar_factor: jnp.ndarray | None = None,
    sign: float = 1.0,
    fixflux: bool = True,
    ghost_low_f: jnp.ndarray | None = None,
    ghost_high_f: jnp.ndarray | None = None,
    ghost_low_v: jnp.ndarray | None = None,
    ghost_high_v: jnp.ndarray | None = None,
    boundary_flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Literal JAX translation of Hermes `FV::Div_par_mod`."""

    f_arr = jnp.asarray(f, dtype=jnp.float64)
    v_arr = jnp.asarray(v, dtype=jnp.float64)
    wave_arr = jnp.asarray(wave_speed, dtype=jnp.float64)
    fp = _pad_open(f_arr, ghost_low=ghost_low_f, ghost_high=ghost_high_f)
    vp = _pad_open(v_arr, ghost_low=ghost_low_v, ghost_high=ghost_high_v)
    left_f, right_f = _reconstruct_cell_edges(fp, limiter)
    left_v, right_v = _reconstruct_cell_edges(vp, limiter)

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
        J_r = 0.5 * (Jc[:-1] + Jc[1:])
        flux_factor_rc = J_r / (jnp.maximum(dy_c[:-1], 1e-30) * jnp.maximum(Jc[:-1], 1e-30))
        flux_factor_rp = J_r / (jnp.maximum(dy_c[1:], 1e-30) * jnp.maximum(Jc[1:], 1e-30))
        boundary_factor_low = 1.0 / jnp.maximum(dy_c[0], 1e-30)
        boundary_factor_high = 1.0 / jnp.maximum(dy_c[-1], 1e-30)
    else:
        sqrt_g = jnp.sqrt(jnp.maximum(gpar_c, 1e-30))
        common_r = (Jc[:-1] + Jc[1:]) / (sqrt_g[:-1] + sqrt_g[1:])
        flux_factor_rc = common_r / (jnp.maximum(dy_c[:-1], 1e-30) * jnp.maximum(Jc[:-1], 1e-30))
        flux_factor_rp = common_r / (jnp.maximum(dy_c[1:], 1e-30) * jnp.maximum(Jc[1:], 1e-30))
        boundary_factor_low = 1.0 / (jnp.maximum(dy_c[0], 1e-30) * jnp.maximum(sqrt_g[0], 1e-30))
        boundary_factor_high = 1.0 / (jnp.maximum(dy_c[-1], 1e-30) * jnp.maximum(sqrt_g[-1], 1e-30))

    amax = jnp.maximum(
        jnp.maximum(wave_arr[:-1], wave_arr[1:]),
        jnp.maximum(jnp.abs(v_arr[:-1]), jnp.abs(v_arr[1:])),
    )
    flux_interior = right_f[:-1] * 0.5 * (right_v[:-1] + amax) + left_f[1:] * 0.5 * (
        left_v[1:] - amax
    )
    div = div.at[:-1].add(flux_interior * flux_factor_rc)
    div = div.at[1:].add(-flux_interior * flux_factor_rp)

    vpar_low = 0.5 * (vp[1] + vp[0])
    bndryval_low = 0.5 * (fp[1] + fp[0])
    if fixflux:
        flux_low = bndryval_low * vpar_low
    else:
        amax_low = jnp.maximum(wave_arr[0], jnp.maximum(jnp.abs(vp[1]), jnp.abs(vp[0])))
        flux_low = left_f[0] * vpar_low - amax_low * (left_f[0] - bndryval_low)
    div = div.at[0].add(-flux_low * boundary_factor_low * float(boundary_flux_scale))

    vpar_high = 0.5 * (vp[-2] + vp[-1])
    bndryval_high = 0.5 * (fp[-2] + fp[-1])
    if fixflux:
        flux_high = bndryval_high * vpar_high
    else:
        amax_high = jnp.maximum(wave_arr[-1], jnp.maximum(jnp.abs(vp[-2]), jnp.abs(vp[-1])))
        flux_high = right_f[-1] * vpar_high + amax_high * (right_f[-1] - bndryval_high)
    div = div.at[-1].add(flux_high * boundary_factor_high * float(boundary_flux_scale))

    if dpar_factor is not None and gpar_c is None:
        div = div * jnp.asarray(dpar_factor, dtype=jnp.float64)
    return float(sign) * div
