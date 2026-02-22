"""Nonlinear models and operators.

This package hosts both the HW2D testbed and the DRB2D nonlinear branches, including
hot-ion and EM extensions, energy-budget checks, and solver comparisons.
"""

from .grid import Grid2D
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


def __getattr__(name: str):
    if name in {"HW2DModel", "HW2DParams", "HW2DState"}:
        from .hw2d import HW2DModel, HW2DParams, HW2DState

        return {"HW2DModel": HW2DModel, "HW2DParams": HW2DParams, "HW2DState": HW2DState}[name]
    if name in {"DRB2DModel", "DRB2DParams", "DRB2DState"}:
        from .drb2d import DRB2DModel, DRB2DParams, DRB2DState

        return {"DRB2DModel": DRB2DModel, "DRB2DParams": DRB2DParams, "DRB2DState": DRB2DState}[
            name
        ]
    if name in {"DRB2DEMModel", "DRB2DEMParams", "DRB2DEMState"}:
        from .drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState

        return {
            "DRB2DEMModel": DRB2DEMModel,
            "DRB2DEMParams": DRB2DEMParams,
            "DRB2DEMState": DRB2DEMState,
        }[name]
    if name in {"DRB2DHotIonModel", "DRB2DHotIonParams", "DRB2DHotIonState"}:
        from .drb2d_hot_ion import DRB2DHotIonModel, DRB2DHotIonParams, DRB2DHotIonState

        return {
            "DRB2DHotIonModel": DRB2DHotIonModel,
            "DRB2DHotIonParams": DRB2DHotIonParams,
            "DRB2DHotIonState": DRB2DHotIonState,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
