"""Term-level building blocks for the unified DRB system."""

from .bcs import FieldBCs, resolve_bcs
from .context import TermContext, build_context
from .fields import log_rhs, phys_n, phys_Te, phi_from_omega
from .advection import exb_advection_terms
from .diamagnetic import diamagnetic_current_terms, diamagnetic_terms
from .braginskii import (
    braginskii_heat_exchange_terms,
    braginskii_friction_terms,
    classical_diffusion_terms,
)
from .parallel import ParallelVars, parallel_vars, parallel_conservative_terms
from .curvature import curvature_terms
from .drive import drive_terms
from .diffusion import diffusion_terms
from .sol import (
    apply_sol_phi_bc,
    sol_masks,
    sol_sources,
    sol_sinks,
    sol_sink_open_omega,
    sol_parallel_loss,
    sol_sheath_phi_term,
    sol_sheath_omega_sink,
    sol_omega_bc_dirichlet,
    sol_vpar_bc_dirichlet,
    sol_edge_relaxation,
)
from .neutrals import neutrals_terms
from .sheath import sheath_terms
from .line_bcs import line_bc_terms
from .perp_bc import perp_bc_relaxation
from .bc_relaxation import field_bc_relaxation
from .region_bc import region_bc_relaxation
from .registry import (
    TermScheduler,
    TermSpec,
    build_scheduler,
    available_terms,
    DEFAULT_TERM_SCHEDULE,
)

__all__ = [
    "FieldBCs",
    "resolve_bcs",
    "TermContext",
    "build_context",
    "log_rhs",
    "phys_n",
    "phys_Te",
    "phi_from_omega",
    "exb_advection_terms",
    "diamagnetic_terms",
    "diamagnetic_current_terms",
    "braginskii_heat_exchange_terms",
    "braginskii_friction_terms",
    "classical_diffusion_terms",
    "ParallelVars",
    "parallel_vars",
    "parallel_conservative_terms",
    "curvature_terms",
    "drive_terms",
    "diffusion_terms",
    "apply_sol_phi_bc",
    "sol_masks",
    "sol_sources",
    "sol_sinks",
    "sol_sink_open_omega",
    "sol_parallel_loss",
    "sol_sheath_phi_term",
    "sol_sheath_omega_sink",
    "sol_omega_bc_dirichlet",
    "sol_vpar_bc_dirichlet",
    "sol_edge_relaxation",
    "neutrals_terms",
    "sheath_terms",
    "line_bc_terms",
    "perp_bc_relaxation",
    "field_bc_relaxation",
    "region_bc_relaxation",
    "TermScheduler",
    "TermSpec",
    "build_scheduler",
    "available_terms",
    "DEFAULT_TERM_SCHEDULE",
]
