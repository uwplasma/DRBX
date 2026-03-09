from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Field3DLayout:
    """Guard-aware `(z, x, y)` layout for literal Hermes fields.

    Hermes/BOUT stores physical cells plus guard cells. The literal JAX path
    keeps the solver's `(z, x, y)` axis order but makes the guard extent
    explicit so boundary operations can match Hermes source order exactly.
    """

    pstart: int
    pend: int
    xstart: int
    xend: int
    guard_width: int = 2

    @property
    def interior_shape(self) -> tuple[int, int]:
        return self.pend - self.pstart + 1, self.xend - self.xstart + 1

    def validate(self, shape: tuple[int, int, int]) -> None:
        nz, nx, _ = (int(v) for v in shape)
        if self.pstart < self.guard_width or self.xstart < self.guard_width:
            raise ValueError("Interior start must leave room for the configured guard width.")
        if self.pend >= nz - self.guard_width or self.xend >= nx - self.guard_width:
            raise ValueError("Interior end must leave room for the configured guard width.")
        if self.pstart > self.pend or self.xstart > self.xend:
            raise ValueError("Interior bounds must be ordered.")


@dataclass(frozen=True)
class ShiftedFieldAlignedWeights:
    """Precomputed linear interpolation data for `ShiftedMetricInterp`.

    The literal shifted-metric helpers operate on `(npar, nx, nbinorm)` arrays:
    local parallel coordinate, radial coordinate, and binormal/shift direction.
    """

    shift_idx: jnp.ndarray
    forward_index0: jnp.ndarray
    forward_index1: jnp.ndarray
    forward_frac: jnp.ndarray
    backward_index0: jnp.ndarray
    backward_index1: jnp.ndarray
    backward_frac: jnp.ndarray
    open_field_line: bool


@dataclass(frozen=True)
class ShiftedMetricFFTPhases:
    """Precomputed FFT phase factors for BOUT `ShiftedMetric`."""

    z_shift: jnp.ndarray
    zlength: float
    to_aligned_phase: jnp.ndarray
    from_aligned_phase: jnp.ndarray
    open_field_line: bool
