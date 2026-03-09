"""Fresh literal Hermes translation path.

This package is the active Milestone A parity path. Unlike the frozen hybrid
translation in `jaxdrb.legacy_hermes`, modules here are written from the
Hermes/BOUT component call graph outward rather than being adapted from the
unified `jax_drb` term registry.
"""

from .boundary_standard import (
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    set_boundary_to,
)
from .evolve_density import DensityTransformResult, density_transform_impl
from .evolve_pressure import PressureTransformResult, pressure_transform_impl
from .field import empty_guarded_field, interior_view, pad_field3d
from .sound_speed import SoundSpeedResult, compute_fastest_wave
from .state import LiteralFieldsState, LiteralSpeciesState, LiteralStage1State
from .types import Field3DLayout

__all__ = [
    "DensityTransformResult",
    "Field3DLayout",
    "LiteralFieldsState",
    "LiteralSpeciesState",
    "LiteralStage1State",
    "PressureTransformResult",
    "SoundSpeedResult",
    "apply_neumann_boundary_average_z",
    "apply_neumann_field3d",
    "compute_fastest_wave",
    "density_transform_impl",
    "empty_guarded_field",
    "interior_view",
    "pad_field3d",
    "pressure_transform_impl",
    "set_boundary_to",
]
