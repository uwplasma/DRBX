"""Hermes-aligned JAX mirror operators for strict parity work.

This package is a temporary translation layer for Milestone A parity closure.
Its public API stays narrow until the mirrored operators are validated against
Hermes dump fixtures and can be folded back into the unified core.
"""

from __future__ import annotations

from .boundary import (
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    set_boundary_to_midpoint,
)
from .derivs import ddx_centered_guarded
from .exb import (
    div_n_bxgrad_f_b_xppm_xy_y_local,
    div_n_bxgrad_f_b_xppm_xy_y_local_from_fields,
    div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref,
    div_n_bxgrad_f_b_xppm_xy_y_local_ref,
    div_n_bxgrad_f_b_xppm_xz,
    div_n_bxgrad_f_b_xppm_xz_ref,
)
from .primitives import Stencil1D, limit_free, mc_limiter, minmod
from .species import prepare_poloidal_y_dfdx_local, prepare_poloidal_y_dfdx_local_ref
from .transform import (
    build_shifted_metric_weights,
    build_shifted_metric_fft_phases,
    from_field_aligned_all_fft,
    from_field_aligned_all_fft_ref,
    from_field_aligned_all,
    from_field_aligned_all_ref,
    from_field_aligned_nobndry_fft,
    from_field_aligned_nobndry_fft_ref,
    from_field_aligned_nobndry,
    from_field_aligned_nobndry_ref,
    shifted_metric_fft_phases_from_geometry,
    shifted_metric_weights_from_geometry,
    to_field_aligned_all_fft,
    to_field_aligned_all_fft_ref,
    to_field_aligned_all,
    to_field_aligned_all_ref,
    to_field_aligned_nox_fft,
    to_field_aligned_nox_fft_ref,
    to_field_aligned_nox,
    to_field_aligned_nox_ref,
)
from .types import (
    FieldAlignedLocalLayout,
    GuardLayout,
    ShiftedFieldAlignedWeights,
    ShiftedMetricFFTPhases,
)

__all__ = [
    "FieldAlignedLocalLayout",
    "GuardLayout",
    "Stencil1D",
    "ShiftedFieldAlignedWeights",
    "ShiftedMetricFFTPhases",
    "apply_neumann_boundary_average_z",
    "apply_neumann_field3d",
    "build_shifted_metric_fft_phases",
    "build_shifted_metric_weights",
    "ddx_centered_guarded",
    "div_n_bxgrad_f_b_xppm_xy_y_local",
    "div_n_bxgrad_f_b_xppm_xy_y_local_from_fields",
    "div_n_bxgrad_f_b_xppm_xy_y_local_from_fields_ref",
    "div_n_bxgrad_f_b_xppm_xy_y_local_ref",
    "div_n_bxgrad_f_b_xppm_xz",
    "div_n_bxgrad_f_b_xppm_xz_ref",
    "from_field_aligned_all",
    "from_field_aligned_all_fft",
    "from_field_aligned_all_fft_ref",
    "from_field_aligned_all_ref",
    "from_field_aligned_nobndry_fft",
    "from_field_aligned_nobndry_fft_ref",
    "from_field_aligned_nobndry",
    "from_field_aligned_nobndry_ref",
    "limit_free",
    "mc_limiter",
    "minmod",
    "prepare_poloidal_y_dfdx_local",
    "prepare_poloidal_y_dfdx_local_ref",
    "set_boundary_to_midpoint",
    "shifted_metric_fft_phases_from_geometry",
    "shifted_metric_weights_from_geometry",
    "to_field_aligned_all",
    "to_field_aligned_all_fft",
    "to_field_aligned_all_fft_ref",
    "to_field_aligned_all_ref",
    "to_field_aligned_nox_fft",
    "to_field_aligned_nox_fft_ref",
    "to_field_aligned_nox",
    "to_field_aligned_nox_ref",
]
