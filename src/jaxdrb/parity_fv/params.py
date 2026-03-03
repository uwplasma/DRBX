from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParityFVParams:
    """Minimal parameter block for the parity-first FV rewrite.

    Keep this intentionally small while parity kernels are introduced.
    Additional fields are added only when a new term is implemented.
    """

    nx: int
    ny: int
    nz: int

    dx: float
    dy: float
    dz: float

    boussinesq: bool = True
    electrostatic: bool = True
    hot_ions: bool = False

    source_n0: float = 0.0
    omega_n: float = 0.0

    n_floor: float = 1e-12
    te_floor: float = 1e-12

    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.nx, self.ny)
