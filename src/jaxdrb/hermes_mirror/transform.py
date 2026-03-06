"""Hermes shifted-transform mirror.

Source of truth:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/parallel/shiftedmetricinterp.cxx`
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`

The current mirror implementation targets the linear interpolation path already
used in the JAX geometry adapter. It lands:

- precomputed shift weights
- a readable reference implementation
- a fused production implementation

for:

- `toFieldAligned(..., "RGN_NOX")`
- `fromFieldAligned(..., "RGN_NOBNDRY")`
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .types import ShiftedFieldAlignedWeights


def _normalize_shift_idx(shift_idx: jnp.ndarray | float, *, nx: int, npar: int) -> jnp.ndarray:
    shift = jnp.asarray(shift_idx, dtype=jnp.float64)
    if shift.ndim == 0:
        return jnp.full((npar, nx), shift, dtype=jnp.float64)
    if shift.ndim == 1:
        if shift.shape[0] == npar:
            return jnp.broadcast_to(shift[:, None], (npar, nx))
        if shift.shape[0] == nx:
            return jnp.broadcast_to(shift[None, :], (npar, nx))
        raise ValueError(
            f"1D shift_idx must have length npar={npar} or nx={nx}, got {shift.shape}."
        )
    if shift.ndim == 2 and shift.shape == (npar, nx):
        return shift
    raise ValueError(f"shift_idx must be scalar, 1D, or shape {(npar, nx)}, got {shift.shape}.")


def _build_linear_weights(
    shift_idx: jnp.ndarray, *, nbinorm: int, sign: float
) -> tuple[jnp.ndarray, ...]:
    y = jnp.arange(nbinorm, dtype=jnp.float64)
    y_src = (y[None, None, :] + (float(sign) * shift_idx)[..., None]) % float(nbinorm)
    index0 = jnp.floor(y_src).astype(jnp.int32)
    index1 = (index0 + 1) % nbinorm
    frac = y_src - index0
    return index0, index1, frac


def build_shifted_metric_weights(
    shift_idx: jnp.ndarray | float,
    *,
    nx: int,
    npar: int,
    nbinorm: int,
    open_field_line: bool,
) -> ShiftedFieldAlignedWeights:
    """Precompute linear interpolation weights for the shifted-metric transform."""

    shift = _normalize_shift_idx(shift_idx, nx=nx, npar=npar)
    f0, f1, ff = _build_linear_weights(shift, nbinorm=nbinorm, sign=1.0)
    b0, b1, bf = _build_linear_weights(shift, nbinorm=nbinorm, sign=-1.0)
    return ShiftedFieldAlignedWeights(
        shift_idx=shift,
        forward_index0=f0,
        forward_index1=f1,
        forward_frac=ff,
        backward_index0=b0,
        backward_index1=b1,
        backward_frac=bf,
        open_field_line=bool(open_field_line),
    )


def shifted_metric_weights_from_geometry(geom: Any) -> ShiftedFieldAlignedWeights:
    """Build mirror transform weights from an existing geometry adapter."""

    shape = tuple(int(v) for v in geom.shape())
    npar, nx, nbinorm = shape
    shift_idx = getattr(geom, "shift_idx", None)
    if shift_idx is None:
        raise ValueError("Geometry does not define shift_idx for a shifted transform.")
    grid = getattr(geom, "grid", None)
    open_field_line = bool(getattr(grid, "open_field_line", False))
    return build_shifted_metric_weights(
        shift_idx,
        nx=nx,
        npar=npar,
        nbinorm=nbinorm,
        open_field_line=open_field_line,
    )


def _interp_line_ref(
    line: jnp.ndarray,
    index0: jnp.ndarray,
    index1: jnp.ndarray,
    frac: jnp.ndarray,
) -> jnp.ndarray:
    return (1.0 - frac) * line[index0] + frac * line[index1]


def _apply_linear_shift_ref(
    field: jnp.ndarray,
    index0: jnp.ndarray,
    index1: jnp.ndarray,
    frac: jnp.ndarray,
) -> jnp.ndarray:
    interp_x = jax.vmap(_interp_line_ref, in_axes=(0, 0, 0, 0), out_axes=0)
    interp_par = jax.vmap(interp_x, in_axes=(0, 0, 0, 0), out_axes=0)
    return interp_par(field, index0, index1, frac)


def _apply_linear_shift_fused(
    field: jnp.ndarray,
    index0: jnp.ndarray,
    index1: jnp.ndarray,
    frac: jnp.ndarray,
) -> jnp.ndarray:
    f0 = jnp.take_along_axis(field, index0, axis=-1)
    f1 = jnp.take_along_axis(field, index1, axis=-1)
    return (1.0 - frac) * f0 + frac * f1


def _preserve_region(
    shifted: jnp.ndarray,
    original: jnp.ndarray,
    *,
    preserve_x_boundaries: bool,
    preserve_parallel_boundaries: bool,
) -> jnp.ndarray:
    out = shifted
    if preserve_x_boundaries and shifted.shape[1] > 1:
        out = out.at[:, 0, :].set(original[:, 0, :])
        out = out.at[:, -1, :].set(original[:, -1, :])
    if preserve_parallel_boundaries and shifted.shape[0] > 1:
        out = out.at[0, :, :].set(original[0, :, :])
        out = out.at[-1, :, :].set(original[-1, :, :])
    return out


def to_field_aligned_nox_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference mirror of `toFieldAligned(..., "RGN_NOX")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_linear_shift_ref(
        field_arr,
        weights.forward_index0,
        weights.forward_index1,
        weights.forward_frac,
    )
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference mirror of `fromFieldAligned(..., "RGN_NOBNDRY")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_linear_shift_ref(
        field_arr,
        weights.backward_index0,
        weights.backward_index1,
        weights.backward_frac,
    )
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=weights.open_field_line,
    )


def to_field_aligned_nox(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused mirror of `toFieldAligned(..., "RGN_NOX")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_linear_shift_fused(
        field_arr,
        weights.forward_index0,
        weights.forward_index1,
        weights.forward_frac,
    )
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused mirror of `fromFieldAligned(..., "RGN_NOBNDRY")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_linear_shift_fused(
        field_arr,
        weights.backward_index0,
        weights.backward_index1,
        weights.backward_frac,
    )
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=weights.open_field_line,
    )
