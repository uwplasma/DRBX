"""Fresh literal Hermes translation path.

This package is the active Milestone A parity path. Unlike the frozen hybrid
translation in `jaxdrb.legacy_hermes`, modules here are written from the
Hermes/BOUT component call graph outward rather than being adapted from the
unified `jax_drb` term registry.
"""

from .boundary import apply_free_o2_field3d
from .boundary_standard import (
    apply_neumann_boundary_average_z,
    apply_neumann_field3d,
    set_boundary_to,
)
from .delp2 import delp2_runtime, derive_delp2_coefficients
from .div_ops import div_par_centered
from .evolve_density import DensityTransformResult, density_transform_impl
from .evolve_pressure import PressureTransformResult, pressure_transform_impl
from .exb import div_n_bxgrad_f_b_xppm, div_n_bxgrad_f_b_xppm_xz, div_n_bxgrad_f_b_xppm_xz_ref
from .field import empty_guarded_field, interior_view, pad_field3d
from .fv import div_a_grad_perp, div_a_grad_perp_local, div_par_mod
from .shifted_metric import (
    build_shifted_metric_fft_phases,
    build_shifted_metric_weights,
    from_field_aligned_all,
    from_field_aligned_all_fft,
    from_field_aligned_all_fft_ref,
    from_field_aligned_all_ref,
    from_field_aligned_nobndry,
    from_field_aligned_nobndry_fft,
    from_field_aligned_nobndry_fft_ref,
    from_field_aligned_nobndry_ref,
    shifted_metric_fft_phases_from_geometry,
    shifted_metric_weights_from_geometry,
    to_field_aligned_all,
    to_field_aligned_all_fft,
    to_field_aligned_all_fft_ref,
    to_field_aligned_all_ref,
    to_field_aligned_nox,
    to_field_aligned_nox_fft,
    to_field_aligned_nox_fft_ref,
    to_field_aligned_nox_ref,
)
from .sound_speed import SoundSpeedResult, compute_fastest_wave
from .state import LiteralFieldsState, LiteralSpeciesState, LiteralStage1State
from .types import (
    Field3DLayout,
    FieldAlignedLocalLayout,
    GuardLayout,
    ShiftedFieldAlignedWeights,
    ShiftedMetricFFTPhases,
)
from .vorticity import full_omega_exb_advection, pi_hat

__all__ = [
    "DensityTransformResult",
    "Field3DLayout",
    "FieldAlignedLocalLayout",
    "GuardLayout",
    "LiteralFieldsState",
    "LiteralSpeciesState",
    "LiteralStage1State",
    "PressureTransformResult",
    "SoundSpeedResult",
    "ShiftedFieldAlignedWeights",
    "ShiftedMetricFFTPhases",
    "apply_free_o2_field3d",
    "apply_neumann_boundary_average_z",
    "apply_neumann_field3d",
    "build_shifted_metric_fft_phases",
    "build_shifted_metric_weights",
    "compute_fastest_wave",
    "delp2_runtime",
    "density_transform_impl",
    "derive_delp2_coefficients",
    "div_a_grad_perp",
    "div_a_grad_perp_local",
    "div_n_bxgrad_f_b_xppm",
    "div_n_bxgrad_f_b_xppm_xz",
    "div_n_bxgrad_f_b_xppm_xz_ref",
    "div_par_centered",
    "div_par_mod",
    "empty_guarded_field",
    "full_omega_exb_advection",
    "from_field_aligned_all",
    "from_field_aligned_all_fft",
    "from_field_aligned_all_fft_ref",
    "from_field_aligned_all_ref",
    "from_field_aligned_nobndry",
    "from_field_aligned_nobndry_fft",
    "from_field_aligned_nobndry_fft_ref",
    "from_field_aligned_nobndry_ref",
    "interior_view",
    "pad_field3d",
    "pi_hat",
    "pressure_transform_impl",
    "set_boundary_to",
    "shifted_metric_fft_phases_from_geometry",
    "shifted_metric_weights_from_geometry",
    "to_field_aligned_all",
    "to_field_aligned_all_fft",
    "to_field_aligned_all_fft_ref",
    "to_field_aligned_all_ref",
    "to_field_aligned_nox",
    "to_field_aligned_nox_fft",
    "to_field_aligned_nox_fft_ref",
    "to_field_aligned_nox_ref",
]
