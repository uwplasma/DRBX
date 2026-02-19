"""Compatibility wrapper for the core 3D FCI DRB model."""

from __future__ import annotations

from jaxdrb.core.fci3d import (
    FCIDRB3DFullModel,
    FCIDRB3DFullParams,
    FCIDRB3DFullSplit,
    FCIDRB3DFullState,
)

__all__ = [
    "FCIDRB3DFullModel",
    "FCIDRB3DFullParams",
    "FCIDRB3DFullSplit",
    "FCIDRB3DFullState",
]
