from __future__ import annotations

import jax
import jax.numpy as jnp

from .map import FCIBilinearMap


def parallel_derivative_centered(
    f_k: jnp.ndarray,
    *,
    f_kp1: jnp.ndarray,
    f_km1: jnp.ndarray,
    map_fwd: FCIBilinearMap,
    map_bwd: FCIBilinearMap,
) -> jnp.ndarray:
    """Centered FCI parallel derivative at plane k.

    Parameters
    ----------
    f_k:
        Field on plane k, shape (nx, ny). Included for future extensions (e.g. one-sided stencils).
    f_kp1, f_km1:
        Field on planes k+1 and k-1, shape (nx, ny).
    map_fwd, map_bwd:
        FCI maps that interpolate from planes k±1 back to the plane-k grid points.

    Returns
    -------
    d_par f:
        Approximation to ∂_|| f at plane k, shape (nx, ny).
    """

    _ = f_k
    fp = map_fwd.apply(f_kp1)
    fm = map_bwd.apply(f_km1)
    # dl can be (nx, ny) to allow spatially varying distance along B between planes.
    dl = map_fwd.dl
    return (fp - fm) / (2.0 * dl)


def parallel_derivative_centered_3d(
    f: jnp.ndarray,
    *,
    map_fwd: FCIBilinearMap,
    map_bwd: FCIBilinearMap,
    open_field_line: bool,
) -> jnp.ndarray:
    """Centered FCI parallel derivative for a full 3D stack (nz, nx, ny)."""

    nz = f.shape[0]
    idx = jnp.arange(nz)
    f_kp1 = f[(idx + 1) % nz]
    f_km1 = f[(idx - 1) % nz]

    def dpar_plane(f_k, f_kp1, f_km1):
        return parallel_derivative_centered(
            f_k, f_kp1=f_kp1, f_km1=f_km1, map_fwd=map_fwd, map_bwd=map_bwd
        )

    dpar = jax.vmap(dpar_plane, in_axes=(0, 0, 0))(f, f_kp1, f_km1)
    if open_field_line and nz >= 2:
        dpar = dpar.at[0].set(jnp.zeros_like(dpar[0]))
        dpar = dpar.at[-1].set(jnp.zeros_like(dpar[-1]))
    return dpar
