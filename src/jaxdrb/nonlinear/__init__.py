"""Nonlinear models and operators.

This package contains a minimal 2D periodic nonlinear testbed (HW-like) used to:
  - exercise JAX-native spectral operators (FFT Poisson solves, dealiasing),
  - provide a clear path toward full nonlinear drift-reduced Braginskii (DRB),
  - host optional additional physics (e.g. neutral interactions) as togglable modules.
"""

from .grid import Grid2D
from .hw2d import HW2DModel, HW2DParams, HW2DState
from .neutrals import NeutralParams

__all__ = [
    "Grid2D",
    "HW2DModel",
    "HW2DParams",
    "HW2DState",
    "NeutralParams",
]
