"""Flux-Coordinate Independent (FCI) operators (scaffolding).

FCI discretizations (Hariri et al. 2014; Stegmeir et al. 2018) represent fields on a set of
perpendicular planes, and build parallel derivatives by:

1) tracing the magnetic field line from a grid point to the next/previous plane,
2) interpolating field values at the mapped footpoints,
3) applying a finite difference along the field line.

This subpackage provides a small, JAX-native, differentiable scaffold for:

- bilinear interpolation-based field-line maps on structured 2D planes,
- matrix-free parallel derivative operators built from those maps,
- unit tests (MMS-style) for correctness and convergence.

The long-term goal is to support diverted tokamaks (X-points) and island divertors by avoiding
flux-surface coordinates in the perpendicular plane.
"""

from .integrate import line_integral_mapped, line_integral_trapezoid, map_stack_to_reference
from .drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from .drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullSplit, FCIDRB3DFullState
from .builder import (
    EssosToroidalFCIConfig,
    ZPlaneFCIConfig,
    build_fci_maps_essos_toroidal_planes,
    build_fci_maps_zplanes,
)
from .io import load_fci_maps_npz, save_fci_maps_npz
from .map import FCIBilinearMap, make_slab_fci_map, make_slab_fci_map_variable_B
from .model import FCISlabModel, FCISlabParams, FCISlabState
from .parallel import (
    classify_target_point_kind,
    parallel_derivative_centered,
    parallel_derivative_centered_3d,
    parallel_derivative_target_aware_3d,
)

__all__ = [
    "FCIBilinearMap",
    "FCIDRB3DModel",
    "FCIDRB3DParams",
    "FCIDRB3DState",
    "FCIDRB3DFullModel",
    "FCIDRB3DFullParams",
    "FCIDRB3DFullSplit",
    "FCIDRB3DFullState",
    "EssosToroidalFCIConfig",
    "FCISlabModel",
    "FCISlabParams",
    "FCISlabState",
    "ZPlaneFCIConfig",
    "build_fci_maps_essos_toroidal_planes",
    "build_fci_maps_zplanes",
    "classify_target_point_kind",
    "line_integral_mapped",
    "line_integral_trapezoid",
    "load_fci_maps_npz",
    "map_stack_to_reference",
    "make_slab_fci_map",
    "make_slab_fci_map_variable_B",
    "parallel_derivative_centered",
    "parallel_derivative_centered_3d",
    "parallel_derivative_target_aware_3d",
    "save_fci_maps_npz",
]
