from __future__ import annotations

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
