"""Hermes-parity finite-volume core (new rewrite path).

This package hosts the strict parity-first implementation used to align
jax_drb against Hermes before broader model extensions are reintroduced.
"""

from .params import ParityFVParams
from .state import ParityFVState
from .rhs import ParityFVRHS
from .flux_parallel import div_parallel_fv
from .poisson_vorticity import (
    apply_invert_set_x_guard,
    apply_parallel_free_y_guard,
    copy_outer_x_guard_cells,
    finalize_phi_after_poisson,
    prepare_phi_plus_pi_for_poisson,
)

__all__ = [
    "ParityFVParams",
    "ParityFVState",
    "ParityFVRHS",
    "div_parallel_fv",
    "apply_invert_set_x_guard",
    "apply_parallel_free_y_guard",
    "copy_outer_x_guard_cells",
    "prepare_phi_plus_pi_for_poisson",
    "finalize_phi_after_poisson",
]
