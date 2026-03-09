"""Compatibility boundary layer for literal Hermes modules."""

from __future__ import annotations

from .boundary_standard import (
    apply_free_o2_field3d,
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    set_boundary_to as set_boundary_to_midpoint,
)

__all__ = [
    "apply_free_o2_field3d",
    "apply_neumann_boundary_average_z",
    "apply_neumann_field3d",
    "set_boundary_to_midpoint",
]
