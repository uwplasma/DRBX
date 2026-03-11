from __future__ import annotations

import jax.numpy as jnp

from .flux_reconstruct import reconstruct_lr


def rusanov_face_flux(
    f_left: jnp.ndarray,
    f_right: jnp.ndarray,
    v_left: jnp.ndarray,
    v_right: jnp.ndarray,
    amax: jnp.ndarray,
) -> jnp.ndarray:
    return 0.5 * (f_left * v_left + f_right * v_right) + 0.5 * amax * (f_left - f_right)


def div_parallel_fv(
    f: jnp.ndarray,
    v: jnp.ndarray,
    *,
    dz: float,
    limiter: str = "mc",
    boundary_flux_low: jnp.ndarray | None = None,
    boundary_flux_high: jnp.ndarray | None = None,
    wave_speed: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute open-field 1D FV divergence along axis-0.

    Arrays may be 1D, 2D, or 3D; transport is always along axis 0.
    """

    f_l, f_r = reconstruct_lr(f, limiter=limiter)
    v_l, v_r = reconstruct_lr(v, limiter=limiter)

    left_f = f_r[:-1]
    right_f = f_l[1:]
    left_v = v_r[:-1]
    right_v = v_l[1:]

    abs_v = jnp.abs(v)
    a_pair = abs_v if wave_speed is None else jnp.maximum(abs_v, jnp.abs(wave_speed))
    amax = jnp.maximum(a_pair[:-1], a_pair[1:])
    face_flux = rusanov_face_flux(left_f, right_f, left_v, right_v, amax)

    if boundary_flux_low is None:
        boundary_flux_low = f[0] * v[0]
    if boundary_flux_high is None:
        boundary_flux_high = f[-1] * v[-1]

    inv_dz = 1.0 / max(float(dz), 1e-30)
    out = jnp.zeros_like(f)
    out = out.at[1:-1].set((face_flux[1:] - face_flux[:-1]) * inv_dz)
    out = out.at[0].set((face_flux[0] - boundary_flux_low) * inv_dz)
    out = out.at[-1].set((boundary_flux_high - face_flux[-1]) * inv_dz)
    return out
