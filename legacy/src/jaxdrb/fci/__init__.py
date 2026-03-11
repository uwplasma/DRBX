"""Flux-Coordinate Independent (FCI) geometry utilities."""

from .grid import FCISlabGrid
from .parallel import (
    classify_target_point_kind,
    parallel_derivative_centered,
    parallel_derivative_centered_3d,
    parallel_derivative_target_aware_3d,
)
from .map import FCIBilinearMap, make_slab_fci_map, make_slab_fci_map_variable_B
from .builder import (
    EssosToroidalFCIConfig,
    ZPlaneFCIConfig,
    build_fci_maps_essos_toroidal_planes,
    build_fci_maps_zplanes,
)
from .io import load_fci_maps_npz, save_fci_maps_npz

__all__ = [
    "FCISlabGrid",
    "FCIBilinearMap",
    "make_slab_fci_map",
    "make_slab_fci_map_variable_B",
    "EssosToroidalFCIConfig",
    "ZPlaneFCIConfig",
    "build_fci_maps_essos_toroidal_planes",
    "build_fci_maps_zplanes",
    "load_fci_maps_npz",
    "save_fci_maps_npz",
    "classify_target_point_kind",
    "parallel_derivative_centered",
    "parallel_derivative_centered_3d",
    "parallel_derivative_target_aware_3d",
]
