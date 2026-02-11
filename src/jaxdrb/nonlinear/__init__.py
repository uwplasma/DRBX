"""Nonlinear models and operators.

This package contains a minimal 2D periodic nonlinear testbed (HW-like) used to:
  - exercise JAX-native spectral operators (FFT Poisson solves, dealiasing),
  - provide a clear path toward full nonlinear drift-reduced Braginskii (DRB),
  - host optional additional physics (e.g. neutral interactions) as togglable modules.
"""

from .grid import Grid2D
from .hw2d import HW2DModel, HW2DParams, HW2DState
from .drb2d import DRB2DModel, DRB2DParams, DRB2DState
from .drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from .drb2d_hot_ion import DRB2DHotIonModel, DRB2DHotIonParams, DRB2DHotIonState
from .neutrals import NeutralParams

__all__ = [
    "Grid2D",
    "HW2DModel",
    "HW2DParams",
    "HW2DState",
    "DRB2DModel",
    "DRB2DParams",
    "DRB2DState",
    "DRB2DEMModel",
    "DRB2DEMParams",
    "DRB2DEMState",
    "DRB2DHotIonModel",
    "DRB2DHotIonParams",
    "DRB2DHotIonState",
    "NeutralParams",
]
