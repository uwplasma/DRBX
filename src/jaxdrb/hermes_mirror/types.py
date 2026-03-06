from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GuardLayout:
    """Guard-cell layout for mirror operators on `(nz, nx, ny)` arrays.

    Hermes and BOUT index fields logically as `(x, y, z)`. The active JAX
    solvers store arrays as `(nz, nx, ny)` to match the rest of the codebase
    and avoid repeated transposes. `xstart`, `xend`, `ystart`, and `yend`
    refer to the inclusive interior bounds in that JAX layout.
    """

    xstart: int
    xend: int
    ystart: int
    yend: int
    x_guards: int = 2
    y_guards: int = 2

    def validate(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 3:
            raise ValueError(f"Mirror operators expect a 3D `(nz, nx, ny)` field, got {shape}.")
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
    """Local field-aligned guard layout for `(npar, nx, nbinorm)` arrays.

    This layout is used for mirror helpers that follow Hermes/BOUT field-aligned
    operators more literally than the active solver storage contract. The first
    axis is the local parallel coordinate, the second axis is radial, and the
    last axis is the shifted/binormal coordinate.
    """

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
                f"Mirror field-aligned helpers expect a 3D `(npar, nx, nbinorm)` field, got {shape}."
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
class ShiftedFieldAlignedWeights:
    """Precomputed linear shift weights for Hermes mirror transforms.

    These weights represent the `ShiftedMetricInterp` interpolation along the
    last axis of the JAX `(npar, nx, nbinorm)` layout. The field-aligned
    coordinate lives on the first axis, the radial coordinate on the second,
    and the shifted/binormal interpolation is applied on the last axis.
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
    """Precomputed FFT phase factors for the Hermes `ShiftedMetric` transform."""

    z_shift: jnp.ndarray
    zlength: float
    to_aligned_phase: jnp.ndarray
    from_aligned_phase: jnp.ndarray
    open_field_line: bool
