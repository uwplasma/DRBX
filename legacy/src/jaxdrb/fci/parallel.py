from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC1D

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
    """Centered FCI parallel derivative for a full 3D stack (nz, nx, ny).

    Notes
    -----
    ``map_fwd`` and ``map_bwd`` can encode either:
      - a single (plane-independent) map with arrays shaped (nx, ny, ...), or
      - a plane-dependent stack with leading dimension (nz, nx, ny, ...).
    """

    nz = f.shape[0]
    idx = jnp.arange(nz)
    f_kp1 = f[(idx + 1) % nz]
    f_km1 = f[(idx - 1) % nz]

    fp = map_fwd.apply(f_kp1)
    fm = map_bwd.apply(f_km1)

    dl = map_fwd.dl
    if dl.ndim == 2:
        dl = jnp.broadcast_to(dl, (nz,) + dl.shape)
    dpar = (fp - fm) / (2.0 * dl)
    if open_field_line and nz >= 2:
        dpar = dpar.at[0].set(jnp.zeros_like(dpar[0]))
        dpar = dpar.at[-1].set(jnp.zeros_like(dpar[-1]))
    return dpar


def classify_target_point_kind(
    *,
    map_fwd: FCIBilinearMap,
    map_bwd: FCIBilinearMap,
) -> jnp.ndarray:
    """Classify FCI target point types in a Stegmeir-style Appendix-B sense.

    Returns an integer array with codes:
      - 0: interior B-point (no target hit in ± maps)
      - 1: C-point (forward hit only)
      - 2: C-point (backward hit only)
      - 3: X-point (both forward/backward hit)
    """

    hit_fwd = map_fwd.hit
    hit_bwd = map_bwd.hit
    if hit_fwd is None or hit_bwd is None:
        shape = map_fwd.dl.shape if map_fwd.dl.ndim == 3 else (1,) + map_fwd.dl.shape
        return jnp.zeros(shape, dtype=jnp.int32)
    hf = jnp.asarray(hit_fwd, dtype=bool)
    hb = jnp.asarray(hit_bwd, dtype=bool)
    if hf.ndim == 2:
        hf = hf[None, ...]
    if hb.ndim == 2:
        hb = hb[None, ...]
    return (
        (hf & (~hb)).astype(jnp.int32)
        + 2 * ((~hf) & hb).astype(jnp.int32)
        + 3 * (hf & hb).astype(jnp.int32)
    )


def _dpar_uneven_spacing(
    f0: jnp.ndarray,
    f_plus: jnp.ndarray,
    f_minus: jnp.ndarray,
    h_plus: jnp.ndarray,
    h_minus: jnp.ndarray,
    *,
    eps: float = 1e-14,
) -> jnp.ndarray:
    """Second-order derivative at 0 from points at -h_minus and +h_plus.

    Fits a quadratic through ( -h_minus, f_minus ), (0, f0), ( +h_plus, f_plus )
    and returns the derivative at 0.
    """

    denom = h_plus * h_minus * (h_plus + h_minus)
    denom = jnp.where(jnp.abs(denom) > eps, denom, jnp.asarray(eps, dtype=f0.dtype))
    return (h_minus**2 * (f_plus - f0) + h_plus**2 * (f0 - f_minus)) / denom


def parallel_derivative_target_aware_3d(
    f: jnp.ndarray,
    *,
    map_fwd: FCIBilinearMap,
    map_bwd: FCIBilinearMap,
    open_field_line: bool,
    bc: BC1D,
    target_scheme: str = "appendix_b",
    dirichlet_target_mode: str = "interpolate",
    neumann_target_mode: str = "gradient",
) -> jnp.ndarray:
    """Target-aware centered FCI parallel derivative on an (nz,nx,ny) stack.

    This operator uses a centered FCI stencil in the interior and switches to a
    *non-uniform* second-order stencil near targets when the map encodes
    distance-to-target information via ``map_fwd.hit/map_fwd.dl_hit`` and
    ``map_bwd.hit/map_bwd.dl_hit``.

    Boundary model
    --------------
    - Dirichlet: uses the prescribed plate values on hit points.
    - Neumann: approximates the plate values by linear extrapolation from the
      interior plane using the prescribed boundary gradients.
    - Periodic: ignores hit masks and returns the standard centered derivative.
    """

    nz = f.shape[0]
    idx = jnp.arange(nz)
    f_kp1 = f[(idx + 1) % nz]
    f_km1 = f[(idx - 1) % nz]

    fp = map_fwd.apply(f_kp1)
    fm = map_bwd.apply(f_km1)

    h_plus = map_fwd.dl
    h_minus = map_bwd.dl
    if h_plus.ndim == 2:
        h_plus = jnp.broadcast_to(h_plus, (nz,) + h_plus.shape)
    if h_minus.ndim == 2:
        h_minus = jnp.broadcast_to(h_minus, (nz,) + h_minus.shape)

    f_plus = fp
    f_minus = fm

    if open_field_line and bc.kind != 0:
        hit_fwd = map_fwd.hit
        hit_bwd = map_bwd.hit
        if hit_fwd is None or hit_bwd is None:
            raise ValueError(
                "Target-aware FCI derivative requires map_fwd.hit and map_bwd.hit when open_field_line=True."
            )
        dl_hit_fwd = map_fwd.dl_hit
        dl_hit_bwd = map_bwd.dl_hit
        if dl_hit_fwd is None or dl_hit_bwd is None:
            raise ValueError(
                "Target-aware FCI derivative requires map_fwd.dl_hit and map_bwd.dl_hit when open_field_line=True."
            )

        if target_scheme not in ("appendix_b", "compat"):
            raise ValueError(f"Unknown target_scheme: {target_scheme}")

        if bc.kind == 1:
            if dirichlet_target_mode not in ("interpolate", "extrapolate"):
                raise ValueError(f"Unknown dirichlet_target_mode: {dirichlet_target_mode}")
            if dirichlet_target_mode == "interpolate":
                f_plus_bc = jnp.asarray(float(bc.right_value), dtype=f.dtype)
                f_minus_bc = jnp.asarray(float(bc.left_value), dtype=f.dtype)
            else:
                # Linear interpolation/extrapolation from (f0, f±plane) to the intersection point.
                frac_plus = dl_hit_fwd / jnp.maximum(h_plus, 1e-14)
                frac_minus = dl_hit_bwd / jnp.maximum(h_minus, 1e-14)
                f_plus_bc = f + frac_plus * (fp - f)
                f_minus_bc = f - frac_minus * (f - fm)
        else:
            if neumann_target_mode not in ("gradient", "copy"):
                raise ValueError(f"Unknown neumann_target_mode: {neumann_target_mode}")
            if neumann_target_mode == "gradient":
                f_plus_bc = f + dl_hit_fwd * jnp.asarray(float(bc.right_grad), dtype=f.dtype)
                f_minus_bc = f - dl_hit_bwd * jnp.asarray(float(bc.left_grad), dtype=f.dtype)
            else:
                f_plus_bc = f
                f_minus_bc = f

        f_plus = jnp.where(hit_fwd, f_plus_bc, f_plus)
        f_minus = jnp.where(hit_bwd, f_minus_bc, f_minus)
        h_plus = jnp.where(hit_fwd, dl_hit_fwd, h_plus)
        h_minus = jnp.where(hit_bwd, dl_hit_bwd, h_minus)

    return _dpar_uneven_spacing(f, f_plus, f_minus, h_plus, h_minus)
