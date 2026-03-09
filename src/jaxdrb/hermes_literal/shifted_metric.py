"""Literal Hermes shifted-metric transforms.

Source of truth:
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`
- `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/parallel/shiftedmetricinterp.cxx`

This module keeps the code independent of `jaxdrb.legacy_hermes`, but mirrors
the same numerical contract as BOUT's shifted-metric transforms.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .types import ShiftedFieldAlignedWeights, ShiftedMetricFFTPhases


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
    """Literal precompute for `ShiftedMetricInterp` linear interpolation."""

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
    """Build linear shifted-metric weights from a geometry adapter."""

    npar, nx, nbinorm = (int(v) for v in geom.shape())
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


def _normalize_z_shift(z_shift: jnp.ndarray | float, *, nx: int, npar: int) -> jnp.ndarray:
    shift = jnp.asarray(z_shift, dtype=jnp.float64)
    if shift.ndim == 0:
        return jnp.full((npar, nx), shift, dtype=jnp.float64)
    if shift.ndim == 1:
        if shift.shape[0] == npar:
            return jnp.broadcast_to(shift[:, None], (npar, nx))
        if shift.shape[0] == nx:
            return jnp.broadcast_to(shift[None, :], (npar, nx))
        raise ValueError(f"1D z_shift must have length npar={npar} or nx={nx}, got {shift.shape}.")
    if shift.ndim == 2 and shift.shape == (npar, nx):
        return shift
    raise ValueError(f"z_shift must be scalar, 1D, or shape {(npar, nx)}, got {shift.shape}.")


def build_shifted_metric_fft_phases(
    z_shift: jnp.ndarray | float,
    *,
    nx: int,
    npar: int,
    nbinorm: int,
    zlength: float,
    open_field_line: bool,
) -> ShiftedMetricFFTPhases:
    """Literal precompute for FFT-based `ShiftedMetric`."""

    if float(zlength) <= 0.0:
        raise ValueError(f"zlength must be positive, got {zlength}.")
    z_shift_arr = _normalize_z_shift(z_shift, nx=nx, npar=npar)
    nmodes = (nbinorm // 2) + 1
    kwave = (jnp.arange(nmodes, dtype=jnp.float64) * (2.0 * jnp.pi)) / float(zlength)
    phase = kwave[None, None, :] * z_shift_arr[..., None]
    return ShiftedMetricFFTPhases(
        z_shift=z_shift_arr,
        zlength=float(zlength),
        to_aligned_phase=jnp.exp(1.0j * phase),
        from_aligned_phase=jnp.exp(-1.0j * phase),
        open_field_line=bool(open_field_line),
    )


def shifted_metric_fft_phases_from_geometry(geom: Any) -> ShiftedMetricFFTPhases:
    """Build FFT phase cache from a geometry adapter."""

    npar, nx, nbinorm = (int(v) for v in geom.shape())
    z_shift = getattr(geom, "z_shift", None)
    if z_shift is None:
        raise ValueError("Geometry does not define z_shift for the shifted metric transform.")
    grid = getattr(geom, "grid", None)
    open_field_line = bool(getattr(grid, "open_field_line", False))
    zlength = float(getattr(grid.perp, "dy", 1.0)) * float(nbinorm)
    return build_shifted_metric_fft_phases(
        z_shift,
        nx=nx,
        npar=npar,
        nbinorm=nbinorm,
        zlength=zlength,
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


def _fft_shift_line_ref(line: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    line_hat = jnp.fft.rfft(line)
    return jnp.fft.irfft(line_hat * phase, n=line.shape[0])


def _apply_fft_shift_ref(field: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    shift_x = jax.vmap(_fft_shift_line_ref, in_axes=(0, 0), out_axes=0)
    shift_par = jax.vmap(shift_x, in_axes=(0, 0), out_axes=0)
    return shift_par(field, phase)


def _apply_fft_shift_fused(field: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    field_hat = jnp.fft.rfft(field, axis=-1)
    return jnp.fft.irfft(field_hat * phase, n=field.shape[-1], axis=-1)


def to_field_aligned_nox_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference `toFieldAligned(..., "RGN_NOX")` for linear interpolation."""

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


def to_field_aligned_all_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference `toFieldAligned(..., "RGN_ALL")` for linear interpolation."""

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
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference `fromFieldAligned(..., "RGN_NOBNDRY")` for linear interpolation."""

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


def from_field_aligned_all_ref(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Reference `fromFieldAligned(..., "RGN_ALL")` for linear interpolation."""

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
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def to_field_aligned_nox(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused `toFieldAligned(..., "RGN_NOX")` for linear interpolation."""

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


def to_field_aligned_all(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused `toFieldAligned(..., "RGN_ALL")` for linear interpolation."""

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
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused `fromFieldAligned(..., "RGN_NOBNDRY")` for linear interpolation."""

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


def from_field_aligned_all(
    field: jnp.ndarray,
    weights: ShiftedFieldAlignedWeights,
) -> jnp.ndarray:
    """Fused `fromFieldAligned(..., "RGN_ALL")` for linear interpolation."""

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
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def to_field_aligned_nox_fft_ref(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Reference FFT `toFieldAligned(..., "RGN_NOX")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_ref(field_arr, phases.to_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=False,
    )


def to_field_aligned_all_fft_ref(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Reference FFT `toFieldAligned(..., "RGN_ALL")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_ref(field_arr, phases.to_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry_fft_ref(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Reference FFT `fromFieldAligned(..., "RGN_NOBNDRY")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_ref(field_arr, phases.from_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=phases.open_field_line,
    )


def from_field_aligned_all_fft_ref(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Reference FFT `fromFieldAligned(..., "RGN_ALL")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_ref(field_arr, phases.from_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def to_field_aligned_nox_fft(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Fused FFT `toFieldAligned(..., "RGN_NOX")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_fused(field_arr, phases.to_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=False,
    )


def to_field_aligned_all_fft(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Fused FFT `toFieldAligned(..., "RGN_ALL")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_fused(field_arr, phases.to_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )


def from_field_aligned_nobndry_fft(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Fused FFT `fromFieldAligned(..., "RGN_NOBNDRY")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_fused(field_arr, phases.from_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=True,
        preserve_parallel_boundaries=phases.open_field_line,
    )


def from_field_aligned_all_fft(
    field: jnp.ndarray,
    phases: ShiftedMetricFFTPhases,
) -> jnp.ndarray:
    """Fused FFT `fromFieldAligned(..., "RGN_ALL")`."""

    field_arr = jnp.asarray(field, dtype=jnp.float64)
    shifted = _apply_fft_shift_fused(field_arr, phases.from_aligned_phase)
    return _preserve_region(
        shifted,
        field_arr,
        preserve_x_boundaries=False,
        preserve_parallel_boundaries=False,
    )
