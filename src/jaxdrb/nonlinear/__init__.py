"""Nonlinear models and operators.

This package hosts both the HW2D testbed and the DRB2D nonlinear branches, including
hot-ion and EM extensions, energy-budget checks, and solver comparisons.
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
