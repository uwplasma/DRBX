from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GuardLayout:
    """Guard-cell layout for `(nz, nx, ny)` arrays."""

    xstart: int
    xend: int
    ystart: int
    yend: int
    x_guards: int = 2
    y_guards: int = 2

    def validate(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 3:
            raise ValueError(f"Guarded operators expect a 3D `(nz, nx, ny)` field, got {shape}.")
        _, nx, ny = shape
        if self.xstart < self.x_guards:
            raise ValueError(
                f"xstart={self.xstart} leaves fewer than {self.x_guards} lower x guards."
            )
        if self.xend >= nx - self.x_guards:
            raise ValueError(
                f"xend={self.xend} leaves fewer than {self.x_guards} upper x guards for nx={nx}."
            )
        if self.ystart < self.y_guards:
            raise ValueError(
                f"ystart={self.ystart} leaves fewer than {self.y_guards} lower y guards."
            )
        if self.yend >= ny - self.y_guards:
            raise ValueError(
                f"yend={self.yend} leaves fewer than {self.y_guards} upper y guards for ny={ny}."
            )
        if self.xend < self.xstart:
            raise ValueError(f"xend={self.xend} must be >= xstart={self.xstart}.")
        if self.yend < self.ystart:
            raise ValueError(f"yend={self.yend} must be >= ystart={self.ystart}.")


@dataclass(frozen=True)
class FieldAlignedLocalLayout:
    """Guard layout for local field-aligned `(npar, nx, nbinorm)` arrays."""

    pstart: int
    pend: int
    xstart: int
    xend: int
    p_guards: int = 2
    x_guards: int = 2
    open_field_line: bool = True

    def validate(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 3:
            raise ValueError(
                f"Field-aligned helpers expect a 3D `(npar, nx, nbinorm)` field, got {shape}."
            )
        npar, nx, _ = shape
        if self.pstart < self.p_guards:
            raise ValueError(
                f"pstart={self.pstart} leaves fewer than {self.p_guards} lower parallel guards."
            )
        if self.pend >= npar - self.p_guards:
            raise ValueError(
                f"pend={self.pend} leaves fewer than {self.p_guards} upper parallel guards for npar={npar}."
            )
        if self.xstart < self.x_guards:
            raise ValueError(
                f"xstart={self.xstart} leaves fewer than {self.x_guards} lower x guards."
            )
        if self.xend >= nx - self.x_guards:
            raise ValueError(
                f"xend={self.xend} leaves fewer than {self.x_guards} upper x guards for nx={nx}."
            )
        if self.pend < self.pstart:
            raise ValueError(f"pend={self.pend} must be >= pstart={self.pstart}.")
        if self.xend < self.xstart:
            raise ValueError(f"xend={self.xend} must be >= xstart={self.xstart}.")


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
