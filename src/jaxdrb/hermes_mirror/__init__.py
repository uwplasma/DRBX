"""Hermes-aligned JAX mirror operators for strict parity work.

This package is a temporary translation layer for Milestone A parity closure.
Its public API stays narrow until the mirrored operators are validated against
Hermes dump fixtures and can be folded back into the unified core.
"""

from __future__ import annotations

from .boundary import apply_neumann_boundary_average_z, set_boundary_to_midpoint
from .primitives import Stencil1D, limit_free, mc_limiter, minmod
from .types import GuardLayout

__all__ = [
    "GuardLayout",
    "Stencil1D",
    "apply_neumann_boundary_average_z",
    "limit_free",
    "mc_limiter",
    "minmod",
    "set_boundary_to_midpoint",
]
