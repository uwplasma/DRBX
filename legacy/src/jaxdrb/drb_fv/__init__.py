"""Hermes-alignment finite-volume core (new rewrite path).

This package hosts the strict alignment-first implementation used to align
jax_drb against Hermes before broader model extensions are reintroduced.
"""

from .params import DRBFVParams
from .state import DRBFVState
from .geometry import DRBFVGeometry
from .rhs import DRBFVRHS
from .flux_parallel import div_parallel_fv
from .terms_density import density_parallel_tendency
from .terms_pressure import pressure_parallel_tendencies
from .terms_sheath import sheath_boundary_tendencies
from .terms_vorticity import vorticity_curvature_tendency, vorticity_parallel_tendency
from .system import DRBFVSystem
from .poisson_vorticity import (
    apply_invert_set_x_guard,
    apply_parallel_free_y_guard,
    copy_outer_x_guard_cells,
    finalize_phi_after_poisson,
    laplacian_xy_spectral,
    laplacian_xy_periodic,
    prepare_phi_plus_pi_for_poisson,
    solve_poisson_xy_spectral,
)

__all__ = [
    "DRBFVParams",
    "DRBFVState",
    "DRBFVGeometry",
    "DRBFVRHS",
    "div_parallel_fv",
    "density_parallel_tendency",
    "pressure_parallel_tendencies",
    "sheath_boundary_tendencies",
    "vorticity_parallel_tendency",
    "vorticity_curvature_tendency",
    "DRBFVSystem",
    "laplacian_xy_spectral",
    "laplacian_xy_periodic",
    "solve_poisson_xy_spectral",
    "apply_invert_set_x_guard",
    "apply_parallel_free_y_guard",
    "copy_outer_x_guard_cells",
    "prepare_phi_plus_pi_for_poisson",
    "finalize_phi_after_poisson",
]
