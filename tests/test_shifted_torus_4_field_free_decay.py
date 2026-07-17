from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dkx.geometry import (
    ConservativeStencilBuilder,
    FciGeometry3D,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
)
from dkx.native import (
    Fci4FieldFreeDecayParameters,
    Fci4FieldState,
    build_perp_laplacian_face_projectors,
    build_perp_laplacian_mg_hierarchy,
    compute_4field_curvature,
    compute_4field_diffusion,
    compute_4field_free_decay_rhs,
    compute_4field_poisson_diffusion,
)
from dkx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN, BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D
from dkx.native.fci_operators import PerpLaplacianInverseSolver, perp_laplacian_conservative_op

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test_mms_shifted_torus_4_field import build_shifted_torus_4field_geometry, n0, sigma, x_max, x_min


tf = 0.6
num_steps = 200
resolution = 60
rho_star = 1.0
Te = 1.0
mi_over_me = 1836.0
eps_n = 1.0e-2
eps_phi = 1.0e-2
free_decay_initial_state_kind = "adiabatic_electron"

'''
rhs_kind = "diffusion"
rhs_kind = "poisson_diffusion"
rhs_kind = "curvature"
'''
rhs_kind = "diffusion"
phi_inversion_regularization = 0.0
phi_inversion_project_mean_zero = False


def _resolution_step_count(resolution: int, *, base_resolution: int = 20, base_steps: int = num_steps) -> int:
    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _free_decay_rhs_function(kind: str):
    if kind == "full":
        return compute_4field_free_decay_rhs
    if kind == "diffusion":
        return compute_4field_diffusion
    if kind == "poisson_diffusion":
        return compute_4field_poisson_diffusion
    if kind == "curvature":
        return compute_4field_curvature
    raise ValueError(f"unknown free-decay rhs_kind {kind!r}")


def _format_progress_bar(
    completed: int,
    total: int,
    *,
    start_time: float,
    time_value: float | None = None,
    gmres_steps_per_solve: float | None = None,
    rhs_pre_projection_compatibility_ratio: float | None = None,
    rhs_post_projection_compatibility_ratio: float | None = None,
    gmres_rel_res: float | None = None,
    width: int = 28,
) -> str:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, float(completed) / float(total)))
    filled = int(round(float(width) * fraction))
    elapsed = time_module.perf_counter() - start_time
    rate = float(completed) / elapsed if elapsed > 0.0 and completed > 0 else 0.0
    remaining = (float(total - completed) / rate) if rate > 0.0 else float("nan")
    eta_text = "--:--" if not np.isfinite(remaining) else _format_duration(remaining)
    time_text = "" if time_value is None else f" t={float(time_value):.3e}"
    gmres_text = "" if gmres_steps_per_solve is None else f" gmres/solve={float(gmres_steps_per_solve):.2f}"
    rhs_pre_text = (
        "" if rhs_pre_projection_compatibility_ratio is None else f" rhsCpre={float(rhs_pre_projection_compatibility_ratio):.2e}"
    )
    rhs_post_text = (
        "" if rhs_post_projection_compatibility_ratio is None else f" rhsCpost={float(rhs_post_projection_compatibility_ratio):.2e}"
    )
    rel_text = "" if gmres_rel_res is None else f" rel={float(gmres_rel_res):.2e}"
    return (
        f"[{'#' * filled}{'.' * (width - filled)}] "
        f"{completed:>4d}/{total:<4d} {100.0 * fraction:6.2f}% "
        f"elapsed={_format_duration(elapsed)} eta={eta_text}{time_text}{gmres_text}{rhs_pre_text}{rhs_post_text}{rel_text}"
    )


def _phi_inversion_pin_point(
    geometry: FciGeometry3D,
    regularization: float,
    *,
    project_mean_zero: bool = phi_inversion_project_mean_zero,
) -> tuple[int, int, int] | None:
    if float(regularization) > 0.0 or bool(project_mean_zero):
        return None
    return None


def _max_abs_timing(*timings: jnp.ndarray, index: int) -> float:
    return max(float(jnp.abs(timing[index])) for timing in timings)


def _max_timing(*timings: jnp.ndarray, index: int) -> float:
    return max(float(timing[index]) for timing in timings)


def _volume_weights(geometry: FciGeometry3D) -> jnp.ndarray:
    return (
        jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dx, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz, dtype=jnp.float64)
    )


def _weighted_mean(values: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _volume_weights(geometry)
    return jnp.sum(weights * values) / jnp.maximum(jnp.sum(weights), 1.0e-30)


def _weighted_l2(values: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _volume_weights(geometry)
    return jnp.sqrt(jnp.sum(weights * values * values) / jnp.maximum(jnp.sum(weights), 1.0e-30))


def _compatibility_ratio(values: jnp.ndarray, geometry: FciGeometry3D) -> tuple[float, float, float]:
    mean_value = _weighted_mean(values, geometry)
    l2_value = _weighted_l2(values, geometry)
    ratio = jnp.abs(mean_value) / jnp.maximum(l2_value, 1.0e-30)
    return float(mean_value), float(l2_value), float(ratio)


def _add_state(state: Fci4FieldState, rhs: Fci4FieldState, *, scale: float) -> Fci4FieldState:
    return state.axpy(rhs, scale=scale)


def _free_decay_logical_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
    theta = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[None, :, None]
    zeta = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)[None, None, :]
    return (
        jnp.broadcast_to(x, geometry.shape),
        jnp.broadcast_to(theta, geometry.shape),
        jnp.broadcast_to(zeta, geometry.shape),
    )


def _free_decay_phi_face_bc(geometry: FciGeometry3D) -> BoundaryFaceBC3D:
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
        value_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64),
        value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
    )


def _free_decay_neumann_face_bc(geometry: FciGeometry3D) -> BoundaryFaceBC3D:
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
        value_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64),
        value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
    )


def _free_decay_omega_from_phi(
    geometry: FciGeometry3D,
    phi: jnp.ndarray,
    *,
    conservative_stencil_builder: ConservativeStencilBuilder,
) -> jnp.ndarray:
    phi_face_bc = _free_decay_phi_face_bc(geometry)
    phi_stencil = conservative_stencil_builder(
        phi,
        geometry,
        (False, True, True),
        phi_face_bc,
    )
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    return -perp_laplacian_conservative_op(
        phi_stencil,
        geometry,
        face_projectors=face_projectors,
        face_bc=phi_face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        periodic_axes=(False, True, True),
    )


@dataclass(frozen=True)
class FreeDecayBoundaryConditions:
    phi_face_bc: BoundaryFaceBC3D
    density_face_bc: BoundaryFaceBC3D
    omega_face_bc: BoundaryFaceBC3D
    v_ion_parallel_face_bc: BoundaryFaceBC3D
    v_electron_parallel_face_bc: BoundaryFaceBC3D
    phi_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    phi_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    density_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    density_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    omega_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    omega_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    v_ion_parallel_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    v_electron_parallel_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)


def _build_free_decay_initial_state_default(geometry: FciGeometry3D, time: float) -> Fci4FieldState:
    x, theta, zeta = _free_decay_logical_coordinates(geometry)
    x_scaled = (x - float(x_min)) / (float(x_max) - float(x_min))
    x_center = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_center)
    envelope = jnp.sin(jnp.pi * x_scaled) ** 4

    density = float(n0) * (
        1.0
        + float(eps_n)
        * envelope
        * (
            jnp.cos(2.0 * theta_shift) * jnp.cos(2.0 * zeta)
            + 0.5 * jnp.sin(3.0 * theta_shift - zeta)
        )
    )
    phi = float(eps_phi) * envelope * (
        jnp.cos(2.0 * theta_shift) * jnp.sin(3.0 * zeta)
        + 0.5 * jnp.sin(3.0 * theta_shift + 2.0 * zeta)
        + 0.25 * jnp.cos(4.0 * theta_shift - zeta)
    )
    omega = _free_decay_omega_from_phi(
        geometry,
        phi,
        conservative_stencil_builder=build_conservative_stencil_from_field,
    )

    return Fci4FieldState(
        density=density,
        omega=omega,
        v_ion_parallel=jnp.zeros_like(density, dtype=jnp.float64),
        v_electron_parallel=jnp.zeros_like(density, dtype=jnp.float64),
    )


def _build_free_decay_initial_state_adiabatic_electron(geometry: FciGeometry3D, time: float) -> Fci4FieldState:
    x, theta, zeta = _free_decay_logical_coordinates(geometry)
    x_scaled = (x - float(x_min)) / (float(x_max) - float(x_min))
    x_center = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_center)
    envelope = jnp.sin(jnp.pi * x_scaled) ** 4

    phi = float(eps_phi) * envelope * (
        jnp.cos(2.0 * theta_shift) * jnp.sin(3.0 * zeta)
        + 0.5 * jnp.sin(3.0 * theta_shift + 2.0 * zeta)
        + 0.25 * jnp.cos(4.0 * theta_shift - zeta)
    )
    density = float(n0) * jnp.exp(phi)
    omega = _free_decay_omega_from_phi(
        geometry,
        phi,
        conservative_stencil_builder=build_conservative_stencil_from_field,
    )

    return Fci4FieldState(
        density=density,
        omega=omega,
        v_ion_parallel=jnp.zeros_like(density, dtype=jnp.float64),
        v_electron_parallel=jnp.zeros_like(density, dtype=jnp.float64),
    )


def _build_free_decay_initial_state(
    geometry: FciGeometry3D,
    time: float,
    *,
    kind: str = free_decay_initial_state_kind,
) -> Fci4FieldState:
    if kind == "default":
        return _build_free_decay_initial_state_default(geometry, time)
    if kind == "adiabatic_electron":
        return _build_free_decay_initial_state_adiabatic_electron(geometry, time)
    raise ValueError(f"unknown free-decay initial state kind {kind!r}")


def _free_decay_artifact_stem(kind: str) -> str:
    if kind == "full":
        return "full_4field_free_decay"
    return f"{kind}_4field_free_decay"


def _build_free_decay_boundary_conditions(geometry: FciGeometry3D, time: float) -> FreeDecayBoundaryConditions:
    del time
    phi_face_bc = _free_decay_phi_face_bc(geometry)
    homogeneous_face_bc = _free_decay_neumann_face_bc(geometry)
    return FreeDecayBoundaryConditions(
        phi_face_bc=phi_face_bc,
        density_face_bc=homogeneous_face_bc,
        omega_face_bc=homogeneous_face_bc,
        v_ion_parallel_face_bc=homogeneous_face_bc,
        v_electron_parallel_face_bc=homogeneous_face_bc,
    )


def shifted_torus_4field_free_decay_rk4(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    time: float,
    timestep: float,
    parameters: Fci4FieldFreeDecayParameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    conservative_stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    boundary_conditions: FreeDecayBoundaryConditions,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    phi_mg_hierarchy: object | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    gmres_debug: bool = False,
    phi_guess: jnp.ndarray | None = None,
    rhs_function=compute_4field_free_decay_rhs,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    phi_inverse_solver = phi_inverse_solver or PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=phi_face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=(False, True, True),
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        mg_hierarchy=phi_mg_hierarchy,
        gmres_debug=gmres_debug,
    )
    rhs_1, timings_1, phi_1 = rhs_function(
        state,
        with_diagnostics=True,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
        phi_guess=phi_guess,
        return_phi=True,
    )
    k1 = rhs_1.rhs
    phi_guess = phi_1

    stage_1 = _add_state(state, k1, scale=0.5 * timestep)
    jax.block_until_ready(stage_1.density)
    rhs_2, timings_2, phi_2 = rhs_function(
        stage_1,
        with_diagnostics=True,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
        phi_guess=phi_guess,
        return_phi=True,
    )
    k2 = rhs_2.rhs
    phi_guess = phi_2

    stage_2 = _add_state(state, k2, scale=0.5 * timestep)
    jax.block_until_ready(stage_2.density)
    rhs_3, timings_3, phi_3 = rhs_function(
        stage_2,
        with_diagnostics=True,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
        phi_guess=phi_guess,
        return_phi=True,
    )
    k3 = rhs_3.rhs
    phi_guess = phi_3

    stage_3 = _add_state(state, k3, scale=timestep)
    jax.block_until_ready(stage_3.density)
    rhs_4, timings_4, phi_4 = rhs_function(
        stage_3,
        with_diagnostics=True,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_mg_hierarchy=phi_mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
        phi_guess=phi_guess,
        return_phi=True,
    )
    k4 = rhs_4.rhs
    phi_guess = phi_4

    next_state = _add_state(
        state,
        Fci4FieldState(
            density=(k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density) / 6.0,
            omega=(k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega) / 6.0,
            v_ion_parallel=(k1.v_ion_parallel + 2.0 * k2.v_ion_parallel + 2.0 * k3.v_ion_parallel + k4.v_ion_parallel)
            / 6.0,
            v_electron_parallel=(
                k1.v_electron_parallel
                + 2.0 * k2.v_electron_parallel
                + 2.0 * k3.v_electron_parallel
                + k4.v_electron_parallel
            )
            / 6.0,
        ),
        scale=timestep,
    )
    jax.block_until_ready(next_state.density)
    return next_state, jnp.asarray(
        [
            float(timings_1[0]) + float(timings_2[0]) + float(timings_3[0]) + float(timings_4[0]),
            float(timings_1[1]) + float(timings_2[1]) + float(timings_3[1]) + float(timings_4[1]),
            float(timings_1[2]) + float(timings_2[2]) + float(timings_3[2]) + float(timings_4[2]),
            float(timings_1[3]) + float(timings_2[3]) + float(timings_3[3]) + float(timings_4[3]),
            _max_abs_timing(timings_1, timings_2, timings_3, timings_4, index=4),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=5),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=6),
            _max_abs_timing(timings_1, timings_2, timings_3, timings_4, index=7),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=8),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=9),
            _max_timing(timings_1, timings_2, timings_3, timings_4, index=10),
        ],
        dtype=jnp.float64,
    ), phi_guess


def simulate_shifted_torus_4field_free_decay(
    geometry: FciGeometry3D,
    initial_state: Fci4FieldState,
    boundary_conditions: FreeDecayBoundaryConditions,
    *,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    te_value: float = Te,
    mi_over_me_value: float = mi_over_me,
    use_multigrid_preconditioner: bool = False,
    disable_multigrid_on_failure: bool = True,
    gmres_debug: bool = False,
    show_progress: bool = False,
    parameters: Fci4FieldFreeDecayParameters | None = None,
    rhs_kind_value: str = rhs_kind,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    params = parameters or Fci4FieldFreeDecayParameters(
        rho_star=rho_star_value,
        Te=te_value,
        mi_over_me=mi_over_me_value,
        phi_inversion_regularization=phi_inversion_regularization,
        density_perp_diffusion=4.0e-2,
        omega_perp_diffusion=4.0e-2,
        v_ion_parallel_perp_diffusion=1.0e-3,
        v_electron_parallel_perp_diffusion=4.0e-2,
    )
    rhs_function = _free_decay_rhs_function(rhs_kind_value)
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)

    curvature_start = time_module.perf_counter()
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    curvature_build_time = time_module.perf_counter() - curvature_start
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()

    phi_mg_hierarchy = None
    mg_build_time = 0.0
    if use_multigrid_preconditioner:
        mg_start = time_module.perf_counter()
        phi_mg_hierarchy = build_perp_laplacian_mg_hierarchy(
            geometry,
            conservative_stencil_builder,
            face_bc=boundary_conditions.phi_face_bc,
            regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
            cut_wall_geometry=empty_cut_wall_geometry,
            cut_wall_bc=empty_cut_wall_bc,
            periodic_axes=(False, True, True),
        )
        mg_build_time = time_module.perf_counter() - mg_start
        try:
            phi_check_solver = PerpLaplacianInverseSolver(
                geometry,
                conservative_stencil_builder,
                tol=float(params.phi_inversion_tol),
                maxiter=int(params.phi_inversion_maxiter),
                restart=int(params.phi_inversion_restart),
                face_projectors=face_projectors,
                regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
                cut_wall_geometry=empty_cut_wall_geometry,
                cut_wall_bc=empty_cut_wall_bc,
                periodic_axes=(False, True, True),
                pin_point=None,
                pin_value=0.0,
                project_mean_zero=False,
                target_mean_phi=None,
                regularization_epsilon=0.0,
                mg_hierarchy=phi_mg_hierarchy,
                gmres_debug=gmres_debug,
            )
            phi_check = phi_check_solver(-initial_state.omega, face_bc=boundary_conditions.phi_face_bc)
            jax.block_until_ready(phi_check)
        except RuntimeError as error:
            if not disable_multigrid_on_failure:
                raise
            print(
                "shifted_torus_4field_free_decay multigrid preconditioner failed initial solve; "
                f"disabling for this run. Reason: {error}"
            )
            phi_mg_hierarchy = None

    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(params.phi_inversion_tol),
        maxiter=int(params.phi_inversion_maxiter),
        restart=int(params.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=(False, True, True),
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        mg_hierarchy=phi_mg_hierarchy,
        gmres_debug=gmres_debug,
    )

    state = initial_state
    time_value = 0.0
    current_phi_guess = None
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    timing_history: list[jnp.ndarray] = []
    simulation_start = time_module.perf_counter()
    progress_start = time_module.perf_counter()
    if show_progress:
        print(
            f"RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
            end="",
            flush=True,
        )

    for step_index in range(steps):
        try:
            state, step_timings, current_phi_guess = shifted_torus_4field_free_decay_rk4(
                state,
                geometry=geometry,
                time=time_value,
                timestep=dt,
                parameters=params,
                curvature_coefficients=curvature_coefficients,
                stencil_builder=stencil_builder,
                conservative_stencil_builder=conservative_stencil_builder,
                boundary_conditions=boundary_conditions,
                phi_face_projectors=face_projectors,
                phi_mg_hierarchy=phi_mg_hierarchy,
                phi_inverse_solver=phi_inverse_solver,
                gmres_debug=gmres_debug,
                phi_guess=current_phi_guess,
                rhs_function=rhs_function,
            )
        except RuntimeError as error:
            state_fields = {
                "density": state.density,
                "omega": state.omega,
                "v_ion_parallel": state.v_ion_parallel,
                "v_electron_parallel": state.v_electron_parallel,
            }
            print(
                "shifted_torus_4field_free_decay RK step failed: "
                f"step={step_index}, time={time_value:.6e}, dt={dt:.6e}"
            )
            for field_name, field_values in state_fields.items():
                values = jnp.asarray(field_values, dtype=jnp.float64)
                print(
                    f"  state {field_name}: finite={bool(jnp.all(jnp.isfinite(values)))}, "
                    f"min={float(jnp.nanmin(values)):.6e}, max={float(jnp.nanmax(values)):.6e}, "
                    f"l2={float(jnp.linalg.norm(jnp.nan_to_num(values))):.6e}"
                )
            rhs_pre = -jnp.asarray(state.omega, dtype=jnp.float64)
            rhs_post = rhs_pre - _weighted_mean(rhs_pre, geometry)
            rhs_pre_mean, rhs_pre_l2, rhs_pre_ratio = _compatibility_ratio(rhs_pre, geometry)
            rhs_post_mean, rhs_post_l2, rhs_post_ratio = _compatibility_ratio(rhs_post, geometry)
            print("  phi inversion compatibility at failed RK step start:")
            print(
                f"    rhsCpre={rhs_pre_ratio:.6e}, rhs_mean_J_pre={rhs_pre_mean:.6e}, "
                f"rhs_l2_J_pre={rhs_pre_l2:.6e}"
            )
            print(
                f"    rhsCpost={rhs_post_ratio:.6e}, rhs_mean_J_post={rhs_post_mean:.6e}, "
                f"rhs_l2_J_post={rhs_post_l2:.6e}"
            )
            raise error
        time_value += dt
        times.append(time_value)
        density_history.append(jnp.asarray(state.density, dtype=jnp.float32))
        omega_history.append(jnp.asarray(state.omega, dtype=jnp.float32))
        v_ion_history.append(jnp.asarray(state.v_ion_parallel, dtype=jnp.float32))
        v_electron_history.append(jnp.asarray(state.v_electron_parallel, dtype=jnp.float32))
        timing_history.append(step_timings)
        if show_progress:
            gmres_steps_per_solve = float(step_timings[3]) / 4.0
            progress_text = _format_progress_bar(
                step_index + 1,
                steps,
                start_time=progress_start,
                time_value=time_value,
                gmres_steps_per_solve=gmres_steps_per_solve,
                rhs_pre_projection_compatibility_ratio=float(step_timings[6]),
                rhs_post_projection_compatibility_ratio=float(step_timings[9]),
                gmres_rel_res=float(step_timings[10]),
            )
            print(
                "\r\033[K"
                f"RK4 progress: "
                f"{progress_text}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    if timing_history:
        timing_array = np.asarray(timing_history, dtype=np.float64)
        total_time = time_module.perf_counter() - simulation_start
        phi_pin_point = _phi_inversion_pin_point(geometry, params.phi_inversion_regularization)
        print(f"shifted_torus_4field_free_decay rhs_kind: {rhs_kind_value}")
        print(f"shifted_torus_4field_free_decay curvature coefficient build time: {curvature_build_time:.6e} s")
        print(
            "shifted_torus_4field_free_decay phi inversion gauge: "
            f"epsilon={float(params.phi_inversion_regularization):.6e}, "
            f"project_mean_zero={bool(phi_inversion_project_mean_zero)}, "
            f"pin_point_enabled={phi_pin_point is not None}"
        )
        if use_multigrid_preconditioner and phi_mg_hierarchy is not None:
            print(
                "shifted_torus_4field_free_decay multigrid hierarchy build time: "
                f"{mg_build_time:.6e} s, levels={len(phi_mg_hierarchy.levels) if phi_mg_hierarchy is not None else 0}"
            )
        elif use_multigrid_preconditioner:
            print(
                "shifted_torus_4field_free_decay multigrid preconditioner: "
                f"disabled after initial solve check, hierarchy_build_time={mg_build_time:.6e} s"
            )
        else:
            print("shifted_torus_4field_free_decay multigrid preconditioner: disabled")
        print(
            "shifted_torus_4field_free_decay mean timings per RK step: "
            f"phi_inverse={float(np.mean(timing_array[:, 0])):.6e} s, "
            f"local_stencil={float(np.mean(timing_array[:, 1])):.6e} s, "
            f"operator={float(np.mean(timing_array[:, 2])):.6e} s, "
            f"phi_gmres_steps_per_rk={float(np.mean(timing_array[:, 3])):.2f}, "
            f"phi_gmres_steps_per_solve={float(np.mean(timing_array[:, 3]) / 4.0):.2f}"
        )
        print(
            "shifted_torus_4field_free_decay RK4 timing: "
            f"steps={steps}, total_time={total_time:.6e} s, "
            f"avg_step_time={total_time / float(max(steps, 1)):.6e} s"
        )

    return (
        state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(omega_history, axis=0),
        jnp.stack(v_ion_history, axis=0),
        jnp.stack(v_electron_history, axis=0),
    )


def _save_shifted_torus_free_decay_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    omega_history: jnp.ndarray,
    v_ion_history: jnp.ndarray,
    v_electron_history: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    output_path: str,
    frame_stride: int = 2,
    title: str = "Shifted-torus 4-field free-decay evolution",
    z_indices: tuple[int, int, int, int] | None = None,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    if z_indices is None:
        z_indices = tuple(int(idx) for idx in np.linspace(0, int(z_values.shape[0] - 1), 4))
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    density_data = np.asarray(density_history, dtype=np.float64) - 1.0
    field_specs = (
        ("density fluctuation", density_data, "coolwarm"),
        ("omega", np.asarray(omega_history, dtype=np.float64), "coolwarm"),
        ("v_ion_parallel", np.asarray(v_ion_history, dtype=np.float64), "coolwarm"),
        ("v_electron_parallel", np.asarray(v_electron_history, dtype=np.float64), "coolwarm"),
    )
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)

    fig, axes = plt.subplots(4, 4, figsize=(14.0, 12.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    images = []
    for row, (field_name, field_data, cmap) in enumerate(field_specs):
        vmax = float(np.max(np.abs(field_data)))
        vmax = vmax if vmax > 0.0 else 1.0
        for col, z_index in enumerate(z_indices):
            ax = axes[row, col]
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, float(x_values[-1]))
            ax.set_yticklabels([])
            ax.set_title(f"{field_name}, zeta={z_values[z_index]:.3f}")
            image = ax.pcolormesh(
                theta_grid,
                radius_grid,
                field_data[0, :, :, z_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            images.append(image)
        fig.colorbar(
            images[row * len(z_indices)],
            ax=list(axes[row, :]),
            location="right",
            pad=0.02,
            shrink=0.88,
        )

    suptitle = fig.suptitle(title)

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for row, (field_name, field_data, _) in enumerate(field_specs):
            for col, z_index in enumerate(z_indices):
                images[row * len(z_indices) + col].set_array(field_data[actual_index, :, :, z_index].ravel())
                axes[row, col].set_title(f"{field_name}, zeta={z_values[z_index]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"{title}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def _save_shifted_torus_free_decay_time_traces(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    omega_history: jnp.ndarray,
    v_ion_history: jnp.ndarray,
    v_electron_history: jnp.ndarray,
    geometry: FciGeometry3D,
    boundary_conditions: FreeDecayBoundaryConditions,
    parameters: Fci4FieldFreeDecayParameters,
    *,
    output_path: str,
    title: str = "Shifted-torus 4-field free-decay time traces",
    phi_history: jnp.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    face_projectors = build_perp_laplacian_face_projectors(geometry)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        gmres_debug=False,
        check_residual=False,
    )

    if phi_history is None:
        reconstructed_phi_history: list[jnp.ndarray] = []
        total_snapshots = int(len(omega_history))
        progress_stride = max(1, total_snapshots // 10)
        for index, omega in enumerate(omega_history):
            phi = phi_inverse_solver(
                -jnp.asarray(omega, dtype=jnp.float64),
                face_bc=boundary_conditions.phi_face_bc,
            )
            reconstructed_phi_history.append(jnp.asarray(phi, dtype=jnp.float64))
            if (index + 1) % progress_stride == 0 or index + 1 == total_snapshots:
                print(
                    f"reconstructing phi for plots: {index + 1}/{total_snapshots}",
                    flush=True,
                )
        phi_history = jnp.stack(reconstructed_phi_history, axis=0)

    times_np = np.asarray(times, dtype=np.float64)
    density_np = np.asarray(density_history, dtype=np.float64)
    omega_np = np.asarray(omega_history, dtype=np.float64)
    v_ion_np = np.asarray(v_ion_history, dtype=np.float64)
    v_electron_np = np.asarray(v_electron_history, dtype=np.float64)
    phi_np = np.asarray(phi_history, dtype=np.float64)

    density_mean = np.mean(density_np, axis=(1, 2, 3))
    density_metric_integral = np.sum(np.asarray(geometry.cell_metric.J, dtype=np.float64)[None, :, :, :] * density_np, axis=(1, 2, 3))
    density_rms_fluct = np.sqrt(np.mean((density_np - density_mean[:, None, None, None]) ** 2, axis=(1, 2, 3)))
    omega_rms = np.sqrt(np.mean(omega_np**2, axis=(1, 2, 3)))
    phi_rms = np.sqrt(np.mean(phi_np**2, axis=(1, 2, 3)))
    v_ion_rms = np.sqrt(np.mean(v_ion_np**2, axis=(1, 2, 3)))
    v_electron_rms = np.sqrt(np.mean(v_electron_np**2, axis=(1, 2, 3)))
    phi_max = np.max(np.abs(phi_np), axis=(1, 2, 3))
    omega_max = np.max(np.abs(omega_np), axis=(1, 2, 3))
    v_ion_max = np.max(np.abs(v_ion_np), axis=(1, 2, 3))
    v_electron_max = np.max(np.abs(v_electron_np), axis=(1, 2, 3))
    density_min = np.min(density_np, axis=(1, 2, 3))

    series = (
        ("mean(n)", density_mean),
        ("rms(n - mean(n))", density_rms_fluct),
        ("rms(omega)", omega_rms),
        ("rms(phi)", phi_rms),
        ("rms(ve)", v_electron_rms),
        ("rms(vi)", v_ion_rms),
        ("max(abs(phi))", phi_max),
        ("max(abs(omega))", omega_max),
        ("max(abs(ve))", v_electron_max),
        ("max(abs(vi))", v_ion_max),
        ("min(n)", density_min),
        ("sum(J * n)", density_metric_integral),
    )

    fig, axes = plt.subplots(4, 3, figsize=(14.0, 12.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for index, (label, values) in enumerate(series):
        ax = axes_flat[index]
        ax.plot(times_np, values, linewidth=1.8)
        ax.set_title(label)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(series) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    geometry = build_shifted_torus_4field_geometry((resolution, resolution, resolution))
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    _free_decay_rhs_function(rhs_kind)
    artifact_stem = _free_decay_artifact_stem(rhs_kind)
    parameters = Fci4FieldFreeDecayParameters(
        rho_star=rho_star,
        Te=Te,
        mi_over_me=mi_over_me,
        phi_inversion_tol=5.0e-5,
        phi_inversion_restart=100,
        phi_inversion_regularization=phi_inversion_regularization,
        density_perp_diffusion=4.0e-2,
        omega_perp_diffusion=4.0e-2,
        v_ion_parallel_perp_diffusion=1.0e-3,
        v_electron_parallel_perp_diffusion=4.0e-2,
    )
    history_path = Path(f"{artifact_stem}_histories.npz")
    run_simulation = True
    if history_path.exists():
        with np.load(history_path, allow_pickle=False) as history:
            saved_regularization = float(history["phi_inversion_regularization"]) if "phi_inversion_regularization" in history else None
            saved_project_mean_zero = bool(history["phi_inversion_project_mean_zero"]) if "phi_inversion_project_mean_zero" in history else None
            saved_rhs_kind = str(history["rhs_kind"].item()) if "rhs_kind" in history else None
            saved_initial_state_kind = str(history["initial_state_kind"].item()) if "initial_state_kind" in history else None
            history_matches = (
                saved_regularization is not None
                and np.isclose(saved_regularization, float(parameters.phi_inversion_regularization))
                and saved_project_mean_zero is not None
                and saved_project_mean_zero == bool(phi_inversion_project_mean_zero)
                and saved_rhs_kind == rhs_kind
                and saved_initial_state_kind == free_decay_initial_state_kind
            )
            if history_matches:
                times = jnp.asarray(history["times"], dtype=jnp.float64)
                density_history = jnp.asarray(history["density"], dtype=jnp.float64)
                omega_history = jnp.asarray(history["omega"], dtype=jnp.float64)
                v_ion_history = jnp.asarray(history["v_ion_parallel"], dtype=jnp.float64)
                v_electron_history = jnp.asarray(history["v_electron_parallel"], dtype=jnp.float64)
                run_simulation = False
            else:
                print(
                    "shifted_torus_4field_free_decay history settings mismatch; "
                    f"rerunning with rhs_kind={rhs_kind!r}, initial_state_kind={free_decay_initial_state_kind!r}, "
                    f"epsilon={float(parameters.phi_inversion_regularization):.6e}, "
                    f"project_mean_zero={bool(phi_inversion_project_mean_zero)}"
                )
    if run_simulation:
        initial_state = _build_free_decay_initial_state(geometry, 0.0, kind=free_decay_initial_state_kind)
        _, times, density_history, omega_history, v_ion_history, v_electron_history = simulate_shifted_torus_4field_free_decay(
            geometry,
            initial_state,
            boundary_conditions,
            final_time=tf,
            timestep=tf / float(num_steps),
            use_multigrid_preconditioner=False,
            gmres_debug=False,
            show_progress=True,
            parameters=parameters,
            rhs_kind_value=rhs_kind,
        )
        np.savez(
            history_path,
            times=np.asarray(times, dtype=np.float64),
            density=np.asarray(density_history, dtype=np.float64),
            omega=np.asarray(omega_history, dtype=np.float64),
            v_ion_parallel=np.asarray(v_ion_history, dtype=np.float64),
            v_electron_parallel=np.asarray(v_electron_history, dtype=np.float64),
            rhs_kind=np.asarray(rhs_kind),
            initial_state_kind=np.asarray(free_decay_initial_state_kind),
            phi_inversion_regularization=np.asarray(float(parameters.phi_inversion_regularization), dtype=np.float64),
            phi_inversion_project_mean_zero=np.asarray(bool(phi_inversion_project_mean_zero), dtype=np.bool_),
        )
    _save_shifted_torus_free_decay_time_traces(
        times,
        density_history,
        omega_history,
        v_ion_history,
        v_electron_history,
        geometry,
        boundary_conditions,
        parameters,
        output_path=f"{artifact_stem}_time_traces.png",
    )
    _save_shifted_torus_free_decay_movie(
        times,
        density_history,
        omega_history,
        v_ion_history,
        v_electron_history,
        geometry,
        output_path=f"{artifact_stem}.gif",
        frame_stride=2,
    )


if __name__ == "__main__":
    main()
