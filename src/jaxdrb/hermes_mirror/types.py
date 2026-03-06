from __future__ import annotations

from dataclasses import dataclass


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
