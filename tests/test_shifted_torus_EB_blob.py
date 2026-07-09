from __future__ import annotations

import argparse
import time as time_module
from functools import partial
from pathlib import Path
import sys
from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    ConservativeStencilBuilder,
    FaceBFieldGeometry,
    FciGeometry3D,
    FciMaps3D,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    Spacing3D,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_fci_maps_from_b_contravariant,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryConditionBuilder,
    BoundaryFaceBC3D,
    ConservativeStencil3D,
    CoordinateFaceValueReconstructor3D,
    CoordinateNormalDerivativeConstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
    CutWallNormalDerivativeConstructor3D,
    CutWallValueReconstructor3D,
    LocalStencil1D,
    LocalStencil3D,
)
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBBoundaryConditions, FciDrbEBRhsResult, FciDrbEBState
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBRhsParameters
from jax_drb.native.fci_drb_EB_rhs import _multiply_local_stencils
from jax_drb.native.fci_drb_EB_rhs import compute_fci_drb_eb_rhs
from jax_drb.native import rk4_step
from jax_drb.native.fci_operators import (
    PerpLaplacianInverseSolver,
    _take_stencil_finite_difference,
    build_perp_laplacian_face_projectors,
    build_perp_laplacian_solver_mg_hierarchy,
    grad_parallel_op_direct,
    perp_laplacian_conservative_op,
)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_shifted_torus_4_field_blob import build_shifted_torus_4field_geometry
from test_shifted_torus_4_field_free_decay import _format_progress_bar


DEFAULT_RESOLUTION = 64
radial_b_fraction = 1.0e-2
tf = 0.1
DEFAULT_NUM_STEPS = 150
DEFAULT_INITIAL_TRANSIENT_TIME = 0.02
DEFAULT_INITIAL_VELOCITY_ALPHA = 1.0
DEFAULT_INITIAL_VELOCITY_ELL_FRACTION = 0.2
DEFAULT_PERP_DIFFUSION = 1.0e-5
PERIODIC_AXES = (False, True, True)
# Lower rho is axis-regular; y and z remain standard periodic directions.
AXIS_REGULAR_AXES = (True, False, False)
WALL_SIGN_SMOOTHING_ENABLED = True
WALL_SIGN_SMOOTHING_WIDTH_CELLS = 3.0
WALL_SIGN_SMOOTHING_FORMULA = "theta_flip_abs_tanh_v1"
INITIAL_VELOCITY_FORMULA = "sheath_bc_consistent_v2"
A_N = 0.1
rho0 = 0.5
y0 = np.pi
z0 = np.pi  # kept for symmetry; the initial density is independent of z
Lrho_cells = 8.0
Ly_cells = 8.0
SOURCE_PROFILE = "gaussian_x"
SOURCE_X0 = 0.25
SOURCE_DELTA_X = 0.1
DENSITY_SOURCE_AMPLITUDE = 1.0e-2
ELECTRON_TEMPERATURE_SOURCE_AMPLITUDE = 1.0e-2


def _bmag_from_contravariant_components(B_contra: jnp.ndarray, g_cov: jnp.ndarray) -> jnp.ndarray:
    bmag_sq = jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra)
    return jnp.sqrt(jnp.maximum(bmag_sq, 0.0))


def _normalize_periodic_axes(
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
) -> tuple[bool, bool, bool]:
    if periodic_axes is None:
        return PERIODIC_AXES
    if len(periodic_axes) != 3:
        raise ValueError(f"periodic_axes must have length 3, got {periodic_axes}")
    return tuple(False if axis is None else bool(axis) for axis in periodic_axes)


def _axis_regular_x_stencil(stencil, field: jnp.ndarray, geometry: FciGeometry3D):
    values = jnp.asarray(field, dtype=jnp.float64)
    if values.shape != geometry.shape:
        raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
    if geometry.shape[1] % 2:
        raise ValueError("axis-regular lower-rho mapping requires an even poloidal grid")

    half_turn = geometry.shape[1] // 2
    first_axis_ghost = jnp.roll(values[0], shift=-half_turn, axis=0)
    minus = jnp.asarray(stencil.minus, dtype=jnp.float64).at[0].set(first_axis_ghost)
    return stencil.replace(minus=minus)


def _outer_x_one_sided_local_stencil(
    stencil: LocalStencil1D,
    field: jnp.ndarray,
    geometry: FciGeometry3D,
) -> LocalStencil1D:
    values = jnp.asarray(field, dtype=jnp.float64)
    if values.shape != geometry.shape:
        raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
    if geometry.shape[0] < 3:
        raise ValueError("outer-x second-order one-sided derivative requires at least three x cells")

    x_centers = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
    x0 = x_centers[-1]
    x1 = x_centers[-2]
    x2 = x_centers[-3]
    w0 = (2.0 * x0 - x1 - x2) / ((x0 - x1) * (x0 - x2))
    w1 = (x0 - x2) / ((x1 - x0) * (x1 - x2))
    w2 = (x0 - x1) / ((x2 - x0) * (x2 - x1))

    minus = jnp.asarray(stencil.minus, dtype=jnp.float64).at[-1].set(values[-2])
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64).at[-1].set(values[-3])
    derivative_center_weight = jnp.asarray(stencil.derivative_center_weight, dtype=jnp.float64).at[-1].set(
        jnp.full_like(values[-1], w0)
    )
    derivative_minus_weight = jnp.asarray(stencil.derivative_minus_weight, dtype=jnp.float64).at[-1].set(
        jnp.full_like(values[-1], w1)
    )
    derivative_plus_weight = jnp.asarray(stencil.derivative_plus_weight, dtype=jnp.float64).at[-1].set(
        jnp.full_like(values[-1], w2)
    )

    return stencil.replace(
        minus=minus,
        plus=plus,
        derivative_minus_weight=derivative_minus_weight,
        derivative_center_weight=derivative_center_weight,
        derivative_plus_weight=derivative_plus_weight,
    )


def _periodic_angle_distance(angle: jnp.ndarray, center: float) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(angle - float(center)), jnp.cos(angle - float(center)))


def _periodic_angle_distance_array(angle: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(angle - center), jnp.cos(angle - center))


def _wall_sign_flip_mask(hard_sign: jnp.ndarray) -> jnp.ndarray:
    """Mark poloidal wall cells adjacent to a periodic sign flip."""

    sign_values = jnp.asarray(hard_sign, dtype=jnp.float64)
    if sign_values.ndim != 2:
        raise ValueError(f"hard_sign must have shape (ny, nz), got {sign_values.shape}")
    sign_values = jnp.where(sign_values == 0.0, 1.0, jnp.sign(sign_values))
    forward_flip = sign_values * jnp.roll(sign_values, shift=-1, axis=0) < 0.0
    return forward_flip | jnp.roll(forward_flip, shift=1, axis=0)


def _wall_sign_edge_flip_mask(hard_sign: jnp.ndarray) -> jnp.ndarray:
    sign_values = jnp.asarray(hard_sign, dtype=jnp.float64)
    if sign_values.ndim != 2:
        raise ValueError(f"hard_sign must have shape (ny, nz), got {sign_values.shape}")
    sign_values = jnp.where(sign_values == 0.0, 1.0, jnp.sign(sign_values))
    return sign_values * jnp.roll(sign_values, shift=-1, axis=0) < 0.0


def _smoothed_outer_wall_sign(
    hard_sign: jnp.ndarray,
    theta_centers: jnp.ndarray,
    width_cells: float,
) -> jnp.ndarray:
    """Smooth the outer-wall incidence sign through poloidal sign changes."""

    sign_values = jnp.asarray(hard_sign, dtype=jnp.float64)
    theta = jnp.asarray(theta_centers, dtype=jnp.float64)
    if sign_values.ndim != 2:
        raise ValueError(f"hard_sign must have shape (ny, nz), got {sign_values.shape}")
    if theta.ndim != 1 or theta.shape[0] != sign_values.shape[0]:
        raise ValueError(f"theta_centers must have shape ({sign_values.shape[0]},), got {theta.shape}")

    zero_mask = sign_values == 0.0
    sign_values = jnp.where(zero_mask, 1.0, jnp.sign(sign_values))
    edge_flip = _wall_sign_edge_flip_mask(sign_values)
    ny = int(sign_values.shape[0])
    delta = jnp.asarray(max(float(width_cells), 1.0e-12) * (2.0 * jnp.pi / float(ny)), dtype=jnp.float64)
    smooth_magnitude = jnp.ones_like(sign_values, dtype=jnp.float64)
    theta_column = theta[:, None]
    for j in range(ny):
        theta_left = theta[j]
        theta_right = theta[(j + 1) % ny]
        edge_delta = _periodic_angle_distance_array(theta_right, theta_left)
        crossing_theta = theta_left + 0.5 * edge_delta
        distance = _periodic_angle_distance_array(theta_column, crossing_theta)
        factor = jnp.abs(jnp.tanh(distance / delta))
        smooth_magnitude = jnp.where(edge_flip[j][None, :], smooth_magnitude * factor, smooth_magnitude)
    return jnp.where(zero_mask, 0.0, sign_values * smooth_magnitude)


def _outer_wall_derivative_keep_mask(
    hard_sign: jnp.ndarray,
    width_cells: float = WALL_SIGN_SMOOTHING_WIDTH_CELLS,
) -> jnp.ndarray:
    """Return False where artificial wall-sign smoothing derivatives are suppressed."""

    sign_values = jnp.asarray(hard_sign, dtype=jnp.float64)
    zero_mask = sign_values == 0.0
    smoothing_mask = _wall_sign_flip_mask(sign_values)
    radius = max(0, int(np.ceil(float(width_cells))))
    for offset in range(1, radius + 1):
        smoothing_mask = (
            smoothing_mask
            | jnp.roll(smoothing_mask, shift=offset, axis=0)
            | jnp.roll(smoothing_mask, shift=-offset, axis=0)
        )
    return ~(smoothing_mask | zero_mask)


def _x_wall_hard_sign(geometry: FciGeometry3D) -> jnp.ndarray:
    outward_sign_x = (
        jnp.zeros_like(geometry.face_bfield.x.B_contra[..., 0], dtype=jnp.float64).at[0].set(-1.0).at[-1].set(1.0)
    )
    x_bfield_normal = jnp.asarray(geometry.face_bfield.x.B_contra[..., 0], dtype=jnp.float64)
    x_wall_sign_raw = jnp.sign(outward_sign_x * x_bfield_normal)
    return jnp.where(jnp.isfinite(x_wall_sign_raw), x_wall_sign_raw, 0.0)


def _outer_x_wall_sheath_sign_data(
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None = PERIODIC_AXES,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return shared x-wall signs for sheath values and derivative relations."""

    normalized_periodic_axes = _normalize_periodic_axes(periodic_axes)
    x_hard_sign = _x_wall_hard_sign(geometry)
    outer_hard_sign = x_hard_sign[-1]
    outer_zero_sign_mask = outer_hard_sign == 0.0
    outer_sheath_sign = outer_hard_sign
    outer_derivative_keep_mask = jnp.ones_like(outer_hard_sign, dtype=bool)
    if WALL_SIGN_SMOOTHING_ENABLED and not normalized_periodic_axes[0]:
        outer_sheath_sign = _smoothed_outer_wall_sign(
            outer_hard_sign,
            geometry.grid.y.centers,
            WALL_SIGN_SMOOTHING_WIDTH_CELLS,
        )
        outer_sheath_sign = jnp.where(outer_zero_sign_mask, 0.0, outer_sheath_sign)
        outer_derivative_keep_mask = _outer_wall_derivative_keep_mask(
            outer_hard_sign,
            WALL_SIGN_SMOOTHING_WIDTH_CELLS,
        )
        outer_derivative_keep_mask = jnp.logical_and(outer_derivative_keep_mask, ~outer_zero_sign_mask)
    x_sheath_sign = x_hard_sign.at[-1].set(outer_sheath_sign)
    return x_hard_sign, x_sheath_sign, outer_zero_sign_mask, outer_derivative_keep_mask


def _build_eb_blob_initial_velocity_fields(
    geometry: FciGeometry3D,
    *,
    alpha_v: float = DEFAULT_INITIAL_VELOCITY_ALPHA,
    ell_fraction: float = DEFAULT_INITIAL_VELOCITY_ELL_FRACTION,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rho = jnp.broadcast_to(jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None], geometry.shape)
    x_faces = getattr(geometry.grid.x, "faces", None)
    if x_faces is not None:
        x_faces_np = np.asarray(x_faces, dtype=np.float64)
        x_min = float(x_faces_np[0])
        x_max = float(x_faces_np[-1])
    else:
        x_centers = np.asarray(geometry.grid.x.centers, dtype=np.float64)
        x_widths = np.asarray(geometry.grid.x.widths, dtype=np.float64)
        x_min = float(x_centers[0] - 0.5 * x_widths[0])
        x_max = float(x_centers[-1] + 0.5 * x_widths[-1])
    lx = max(float(x_max - x_min), 1.0e-30)
    ell_v = max(float(ell_fraction) * lx, 1.0e-30)
    upper_distance = float(x_max) - rho
    wall_distance = upper_distance
    wall_weight = jnp.exp(-((wall_distance / ell_v) ** 2))

    tau = jnp.asarray(1.0, dtype=jnp.float64)
    _, x_sheath_sign, _, _ = _outer_x_wall_sheath_sign_data(geometry, PERIODIC_AXES)
    te_face_x = jnp.ones_like(x_sheath_sign, dtype=jnp.float64)
    ft_face_x = jnp.sqrt(jnp.asarray(1.0 + tau, dtype=jnp.float64))
    vi_wall_x = x_sheath_sign * jnp.sqrt(te_face_x) * ft_face_x
    ve_wall_x = x_sheath_sign * jnp.sqrt(te_face_x)

    outer_wall_vi = vi_wall_x[-1]
    outer_wall_ve = ve_wall_x[-1]
    vi = jnp.asarray(float(alpha_v), dtype=jnp.float64) * wall_weight * outer_wall_vi[None, :, :]
    ve = jnp.asarray(float(alpha_v), dtype=jnp.float64) * wall_weight * outer_wall_ve[None, :, :]
    return vi, ve


def _build_eb_blob_initial_state(
    geometry: FciGeometry3D,
    *,
    velocity_initialization: str = "zero",
) -> FciDrbEBState:
    rho = jnp.broadcast_to(jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None], geometry.shape)
    y = jnp.broadcast_to(jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[None, :, None], geometry.shape)

    drho = jnp.asarray(geometry.grid.x.widths[0], dtype=jnp.float64)
    dy = jnp.asarray(geometry.grid.y.widths[0], dtype=jnp.float64)
    l_rho = float(Lrho_cells) * drho
    l_y = float(Ly_cells) * dy
    blob = jnp.exp(-((rho - float(rho0)) ** 2) / (l_rho**2)) * jnp.exp(
        -(_periodic_angle_distance(y, y0) ** 2) / (l_y**2)
    )

    density = jnp.asarray(1.0 + float(A_N) * blob, dtype=jnp.float64)
    zeros = jnp.zeros_like(density, dtype=jnp.float64)
    ones = jnp.ones_like(density, dtype=jnp.float64)
    if velocity_initialization == "zero":
        vi = zeros
        ve = zeros
    elif velocity_initialization == "sheath_taper":
        vi, ve = _build_eb_blob_initial_velocity_fields(geometry)
    else:
        raise ValueError(f"unknown velocity_initialization={velocity_initialization!r}")
    return FciDrbEBState(
        density=density,
        phi=zeros,
        Te=ones,
        Ti=ones,
        Vi=vi,
        Ve=ve,
        vorticity=zeros,
    )


def _build_eb_blob_gaussian_x_source(
    geometry: FciGeometry3D,
    *,
    amplitude: float,
    x0: float,
    delta_x: float,
) -> jnp.ndarray:
    x = jnp.broadcast_to(jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None], geometry.shape)
    delta_x_value = float(delta_x)
    if delta_x_value <= 0.0:
        raise ValueError("delta_x must be positive")
    return jnp.asarray(float(amplitude), dtype=jnp.float64) * jnp.exp(-((x - float(x0)) ** 2) / (delta_x_value**2))


def _build_eb_blob_source_terms(
    geometry: FciGeometry3D,
    *,
    amplitude: float,
    x0: float,
    delta_x: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    density_source = _build_eb_blob_gaussian_x_source(
        geometry,
        amplitude=amplitude,
        x0=x0,
        delta_x=delta_x,
    )
    electron_temperature_source = _build_eb_blob_gaussian_x_source(
        geometry,
        amplitude=amplitude,
        x0=x0,
        delta_x=delta_x,
    )
    return density_source, electron_temperature_source


def _build_eb_blob_parameters(perp_diffusion: float) -> FciDrbEBRhsParameters:
    """Build the EB blob RHS parameters for a single perpendicular diffusion sweep value."""

    perp_diffusion = float(perp_diffusion)
    return FciDrbEBRhsParameters(
        n0=1.0,
        Te0=1.0,
        Ti0=1.0,
        cs_0=1.0,
        rhos_s0=1.0,
        tau=1.0,
        mi_over_me=1836.0,
        rho_star=1.0,
        phi_inversion_iterations=500,
        phi_inversion_regularization=0,
        density_D_perp=perp_diffusion,
        density_D_parallel=0.0,
        electron_temperature_chi_parallel=0.0,
        electron_temperature_D_perp=perp_diffusion,
        ion_temperature_chi_parallel=0.0,
        ion_temperature_D_perp=perp_diffusion,
        Ve_nu=1.0e-3,
        Ve_D_perp=perp_diffusion,
        Ve_D_parallel=0.0,
        Vi_D_perp=perp_diffusion,
        Vi_D_parallel=0.0,
        vorticity_D_perp=perp_diffusion,
        vorticity_D_parallel=0.0,
    )


def _eb_blob_artifact_stem(run_name: str) -> str:
    run_name = run_name.strip()
    if not run_name:
        raise ValueError("run_name must be a non-empty string")
    return run_name


def _resolve_eb_blob_history_path(run_name: str, output_dir: Path | None = None) -> Path:
    history_name = f"{_eb_blob_artifact_stem(run_name)}_histories.npz"
    if output_dir is None:
        return Path(history_name)
    return Path(output_dir) / history_name


def _build_eb_blob_timesteps(
    *,
    final_time: float,
    num_steps: int = DEFAULT_NUM_STEPS,
    timestep: float | None = None,
    initial_transient_time: float = DEFAULT_INITIAL_TRANSIENT_TIME,
    initial_num_steps: int | None = None,
    remaining_num_steps: int | None = None,
) -> tuple[float, ...]:
    final_time_value = float(final_time)
    if final_time_value <= 0.0:
        raise ValueError(f"final_time must be positive, got {final_time_value}")

    split_requested = initial_num_steps is not None or remaining_num_steps is not None
    if split_requested:
        if timestep is not None:
            raise ValueError("timestep cannot be supplied with split initial/remaining step counts")
        if initial_num_steps is None or remaining_num_steps is None:
            raise ValueError("initial_num_steps and remaining_num_steps must be supplied together")

        initial_steps = int(initial_num_steps)
        remaining_steps = int(remaining_num_steps)
        if initial_steps <= 0:
            raise ValueError(f"initial_num_steps must be positive, got {initial_steps}")
        if remaining_steps < 0:
            raise ValueError(f"remaining_num_steps must be non-negative, got {remaining_steps}")

        split_time = min(max(float(initial_transient_time), 0.0), final_time_value)
        if split_time <= 0.0:
            if remaining_steps <= 0:
                raise ValueError("remaining_num_steps must be positive when initial_transient_time is zero")
            return tuple(float(final_time_value / float(remaining_steps)) for _ in range(remaining_steps))
        if np.isclose(split_time, final_time_value):
            if remaining_steps != 0:
                raise ValueError("remaining_num_steps must be zero when initial_transient_time reaches final_time")
            return tuple(float(final_time_value / float(initial_steps)) for _ in range(initial_steps))
        if remaining_steps <= 0:
            raise ValueError("remaining_num_steps must be positive when initial_transient_time is before final_time")

        initial_dt = split_time / float(initial_steps)
        remaining_dt = (final_time_value - split_time) / float(remaining_steps)
        return tuple(float(initial_dt) for _ in range(initial_steps)) + tuple(
            float(remaining_dt) for _ in range(remaining_steps)
        )

    if timestep is not None:
        dt = float(timestep)
        if dt <= 0.0:
            raise ValueError(f"timestep must be positive, got {dt}")
        steps = int(round(final_time_value / dt))
        if steps <= 0:
            raise ValueError(f"timestep {dt} is too large for final_time {final_time_value}")
    else:
        steps = int(num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")
    dt = final_time_value / float(steps)
    return tuple(float(dt) for _ in range(steps))


def _print_eb_blob_timestep_schedule(
    step_sizes: Sequence[float],
    *,
    initial_num_steps: int | None,
    initial_transient_time: float | None,
    final_time: float,
) -> None:
    step_sizes_np = np.asarray(step_sizes, dtype=np.float64)
    if step_sizes_np.size == 0:
        print("EB blob timestep schedule: empty", flush=True)
        return

    preview_indices = {0, 1, 2, int(step_sizes_np.size) - 3, int(step_sizes_np.size) - 2, int(step_sizes_np.size) - 1}
    if initial_num_steps is not None:
        preview_indices.update(
            {
                max(0, int(initial_num_steps) - 1),
                min(int(initial_num_steps), int(step_sizes_np.size) - 1),
            }
        )
    preview_indices = [index for index in sorted(preview_indices) if 0 <= index < int(step_sizes_np.size)]

    print("EB blob timestep schedule preview:", flush=True)
    for index in preview_indices:
        start_time = float(np.sum(step_sizes_np[:index]))
        end_time = start_time + float(step_sizes_np[index])
        print(
            f"  dt[{index}]={float(step_sizes_np[index]):.6e} "
            f"covers t={start_time:.6e} -> {end_time:.6e}",
            flush=True,
        )

    if initial_num_steps is not None and initial_transient_time is not None:
        split_index = int(initial_num_steps)
        split_index = min(max(split_index, 0), int(step_sizes_np.size))
        print(
            f"  phase boundary after step {split_index} at t={float(initial_transient_time):.6e} "
            f"of tf={float(final_time):.6e}",
            flush=True,
        )


def _history_matches_eb_blob_settings(
    metadata: dict[str, object],
    *,
    resolution: int,
    num_steps: int,
    initial_velocity_state: str,
    initial_transient_time: float | None,
    initial_num_steps: int | None,
    remaining_num_steps: int | None,
    a_n: float,
    l_rho_cells: float,
    l_y_cells: float,
    radial_b_fraction: float,
    perp_diffusion: float,
    diffusion_only: bool,
    source_profile: str,
    source_x0: float,
    source_delta_x: float,
    source_amplitude: float,
) -> bool:
    saved_resolution = int(metadata["resolution"]) if "resolution" in metadata else None
    saved_num_steps = int(metadata["num_steps"]) if "num_steps" in metadata else None
    saved_initial_velocity_state = (
        str(metadata["initial_velocity_state"]) if "initial_velocity_state" in metadata else None
    )
    saved_initial_velocity_alpha = (
        float(metadata["initial_velocity_alpha"]) if "initial_velocity_alpha" in metadata else None
    )
    saved_initial_velocity_ell_fraction = (
        float(metadata["initial_velocity_ell_fraction"])
        if "initial_velocity_ell_fraction" in metadata
        else None
    )
    saved_initial_velocity_formula = (
        str(metadata["initial_velocity_formula"]) if "initial_velocity_formula" in metadata else None
    )
    saved_wall_sign_smoothing_formula = (
        str(metadata["wall_sign_smoothing_formula"]) if "wall_sign_smoothing_formula" in metadata else None
    )
    saved_wall_sign_smoothing_enabled = (
        bool(metadata["wall_sign_smoothing_enabled"]) if "wall_sign_smoothing_enabled" in metadata else None
    )
    saved_wall_sign_smoothing_width_cells = (
        float(metadata["wall_sign_smoothing_width_cells"]) if "wall_sign_smoothing_width_cells" in metadata else None
    )
    saved_initial_transient_time = (
        float(metadata["initial_transient_time"]) if "initial_transient_time" in metadata else None
    )
    saved_initial_num_steps = int(metadata["initial_num_steps"]) if "initial_num_steps" in metadata else None
    saved_remaining_num_steps = int(metadata["remaining_num_steps"]) if "remaining_num_steps" in metadata else None
    saved_a_n = float(metadata["A_N"]) if "A_N" in metadata else None
    saved_lrho = float(metadata["Lrho_cells"]) if "Lrho_cells" in metadata else None
    saved_ly = float(metadata["Ly_cells"]) if "Ly_cells" in metadata else None
    saved_radial_b_fraction = (
        float(metadata["radial_b_fraction"]) if "radial_b_fraction" in metadata else None
    )
    saved_perp_diffusion = float(metadata["perp_diffusion"]) if "perp_diffusion" in metadata else None
    saved_diffusion_only = bool(metadata["diffusion_only"]) if "diffusion_only" in metadata else False
    saved_curvature_axis_regular = bool(metadata["curvature_axis_regular_lower_x"]) if "curvature_axis_regular_lower_x" in metadata else False
    saved_source_profile = str(metadata["source_profile"]) if "source_profile" in metadata else None
    saved_source_x0 = float(metadata["source_x0"]) if "source_x0" in metadata else None
    saved_source_delta_x = float(metadata["source_delta_x"]) if "source_delta_x" in metadata else None
    saved_source_amplitude = float(metadata["source_amplitude"]) if "source_amplitude" in metadata else None
    saved_density_source_amplitude = (
        float(metadata["density_source_amplitude"]) if "density_source_amplitude" in metadata else None
    )
    saved_electron_temperature_source_amplitude = (
        float(metadata["electron_temperature_source_amplitude"])
        if "electron_temperature_source_amplitude" in metadata
        else None
    )
    saved_perp_diffusion_fields = (
        float(metadata[key]) if key in metadata else None
        for key in (
            "density_D_perp",
            "electron_temperature_D_perp",
            "ion_temperature_D_perp",
            "Ve_D_perp",
            "Vi_D_perp",
            "vorticity_D_perp",
        )
    )
    if initial_num_steps is None and remaining_num_steps is None:
        schedule_matches = (
            saved_initial_num_steps in (None, 0)
            and saved_remaining_num_steps in (None, int(num_steps))
        )
    else:
        if initial_num_steps is None or remaining_num_steps is None:
            schedule_matches = False
        else:
            schedule_matches = (
                saved_initial_num_steps == int(initial_num_steps)
                and saved_remaining_num_steps == int(remaining_num_steps)
                and saved_initial_transient_time is not None
                and initial_transient_time is not None
                and np.isclose(saved_initial_transient_time, float(initial_transient_time))
            )
    if initial_velocity_state == "zero":
        velocity_matches = saved_initial_velocity_state in (None, "zero")
    else:
        velocity_matches = (
            saved_initial_velocity_state == str(initial_velocity_state)
            and saved_initial_velocity_alpha is not None
            and saved_initial_velocity_ell_fraction is not None
            and np.isclose(saved_initial_velocity_alpha, DEFAULT_INITIAL_VELOCITY_ALPHA)
            and np.isclose(saved_initial_velocity_ell_fraction, DEFAULT_INITIAL_VELOCITY_ELL_FRACTION)
        )
    return (
        saved_resolution == int(resolution)
        and saved_num_steps == int(num_steps)
        and schedule_matches
        and velocity_matches
        and saved_initial_velocity_formula == INITIAL_VELOCITY_FORMULA
        and saved_wall_sign_smoothing_formula == WALL_SIGN_SMOOTHING_FORMULA
        and saved_wall_sign_smoothing_enabled is not None
        and saved_wall_sign_smoothing_enabled == WALL_SIGN_SMOOTHING_ENABLED
        and saved_wall_sign_smoothing_width_cells is not None
        and np.isclose(saved_wall_sign_smoothing_width_cells, WALL_SIGN_SMOOTHING_WIDTH_CELLS)
        and saved_diffusion_only == bool(diffusion_only)
        and saved_a_n is not None
        and np.isclose(saved_a_n, float(a_n))
        and saved_lrho is not None
        and np.isclose(saved_lrho, float(l_rho_cells))
        and saved_ly is not None
        and np.isclose(saved_ly, float(l_y_cells))
        and saved_radial_b_fraction is not None
        and np.isclose(saved_radial_b_fraction, float(radial_b_fraction))
        and (
            saved_perp_diffusion is None
            or np.isclose(saved_perp_diffusion, float(perp_diffusion))
        )
        and saved_curvature_axis_regular
        and saved_source_profile == str(source_profile)
        and saved_source_x0 is not None
        and np.isclose(saved_source_x0, float(source_x0))
        and saved_source_delta_x is not None
        and np.isclose(saved_source_delta_x, float(source_delta_x))
        and (
            (
                saved_source_amplitude is not None
                and np.isclose(saved_source_amplitude, float(source_amplitude))
            )
            or (
                saved_density_source_amplitude is not None
                and saved_electron_temperature_source_amplitude is not None
                and np.isclose(saved_density_source_amplitude, float(source_amplitude))
                and np.isclose(saved_electron_temperature_source_amplitude, float(source_amplitude))
            )
        )
        and all(
            saved_value is not None and np.isclose(saved_value, float(perp_diffusion))
            for saved_value in saved_perp_diffusion_fields
        )
    )


def _matching_eb_blob_metadata() -> dict[str, object]:
    return {
        "resolution": 8,
        "num_steps": 5,
        "initial_velocity_state": "sheath_taper",
        "initial_velocity_alpha": DEFAULT_INITIAL_VELOCITY_ALPHA,
        "initial_velocity_ell_fraction": DEFAULT_INITIAL_VELOCITY_ELL_FRACTION,
        "initial_velocity_formula": INITIAL_VELOCITY_FORMULA,
        "wall_sign_smoothing_formula": WALL_SIGN_SMOOTHING_FORMULA,
        "wall_sign_smoothing_enabled": WALL_SIGN_SMOOTHING_ENABLED,
        "wall_sign_smoothing_width_cells": WALL_SIGN_SMOOTHING_WIDTH_CELLS,
        "initial_num_steps": 0,
        "remaining_num_steps": 5,
        "A_N": A_N,
        "Lrho_cells": Lrho_cells,
        "Ly_cells": Ly_cells,
        "radial_b_fraction": radial_b_fraction,
        "perp_diffusion": DEFAULT_PERP_DIFFUSION,
        "diffusion_only": False,
        "curvature_axis_regular_lower_x": True,
        "source_profile": SOURCE_PROFILE,
        "source_x0": SOURCE_X0,
        "source_delta_x": SOURCE_DELTA_X,
        "source_amplitude": DENSITY_SOURCE_AMPLITUDE,
        "density_D_perp": DEFAULT_PERP_DIFFUSION,
        "electron_temperature_D_perp": DEFAULT_PERP_DIFFUSION,
        "ion_temperature_D_perp": DEFAULT_PERP_DIFFUSION,
        "Ve_D_perp": DEFAULT_PERP_DIFFUSION,
        "Vi_D_perp": DEFAULT_PERP_DIFFUSION,
        "vorticity_D_perp": DEFAULT_PERP_DIFFUSION,
    }


def test_eb_blob_timesteps_default_uniform_grid() -> None:
    step_sizes = _build_eb_blob_timesteps(final_time=0.1, num_steps=5)

    np.testing.assert_allclose(step_sizes, np.full(5, 0.02, dtype=np.float64))
    np.testing.assert_allclose(np.sum(step_sizes), 0.1)


def test_eb_blob_timesteps_split_initial_transient() -> None:
    step_sizes = _build_eb_blob_timesteps(
        final_time=0.1,
        initial_transient_time=0.02,
        initial_num_steps=4,
        remaining_num_steps=8,
    )

    assert len(step_sizes) == 12
    np.testing.assert_allclose(step_sizes[:4], np.full(4, 0.005, dtype=np.float64))
    np.testing.assert_allclose(step_sizes[4:], np.full(8, 0.01, dtype=np.float64))
    np.testing.assert_allclose(np.sum(step_sizes), 0.1)


def test_eb_blob_sheath_taper_initial_velocity_state_is_damped_toward_the_core() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    zero_state = _build_eb_blob_initial_state(geometry)
    taper_state = _build_eb_blob_initial_state(geometry, velocity_initialization="sheath_taper")

    np.testing.assert_allclose(np.asarray(zero_state.Ve), 0.0)
    np.testing.assert_allclose(np.asarray(zero_state.Vi), 0.0)
    assert np.max(np.abs(np.asarray(taper_state.Ve))) > 0.0
    assert np.max(np.abs(np.asarray(taper_state.Vi))) > 0.0

    middle_index = int(geometry.shape[0] // 2)
    edge_ve = np.max(np.abs(np.asarray(taper_state.Ve)[-1]))
    core_ve = np.max(np.abs(np.asarray(taper_state.Ve)[middle_index]))
    edge_vi = np.max(np.abs(np.asarray(taper_state.Vi)[-1]))
    core_vi = np.max(np.abs(np.asarray(taper_state.Vi)[middle_index]))
    assert edge_ve > core_ve
    assert edge_vi > core_vi


def test_eb_blob_sheath_taper_uses_smoothed_bc_consistent_wall_velocity() -> None:
    geometry = _build_eb_blob_geometry((8, 32, 8), construct_fci_maps=False)
    taper_state = _build_eb_blob_initial_state(geometry, velocity_initialization="sheath_taper")
    _, x_sheath_sign, _, _ = _outer_x_wall_sheath_sign_data(geometry)

    x_faces = np.asarray(geometry.grid.x.faces, dtype=np.float64)
    x_max = float(x_faces[-1])
    lx = max(float(x_faces[-1] - x_faces[0]), 1.0e-30)
    ell_v = max(float(DEFAULT_INITIAL_VELOCITY_ELL_FRACTION) * lx, 1.0e-30)
    rho = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    wall_weight = np.exp(-(((x_max - rho) / ell_v) ** 2))

    expected_ve_outer = DEFAULT_INITIAL_VELOCITY_ALPHA * wall_weight[-1] * np.asarray(x_sheath_sign[-1])
    expected_vi_outer = np.sqrt(2.0) * expected_ve_outer

    np.testing.assert_allclose(np.asarray(taper_state.Ve[-1]), expected_ve_outer, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(taper_state.Vi[-1]), expected_vi_outer, rtol=1.0e-6, atol=1.0e-6)
    assert np.max(np.abs(np.asarray(taper_state.Ve[-1]))) <= DEFAULT_INITIAL_VELOCITY_ALPHA + 1.0e-6


def test_eb_blob_history_match_requires_startup_formula_metadata() -> None:
    metadata = _matching_eb_blob_metadata()
    assert _history_matches_eb_blob_settings(
        metadata,
        resolution=8,
        num_steps=5,
        initial_velocity_state="sheath_taper",
        initial_transient_time=None,
        initial_num_steps=None,
        remaining_num_steps=None,
        a_n=A_N,
        l_rho_cells=Lrho_cells,
        l_y_cells=Ly_cells,
        radial_b_fraction=radial_b_fraction,
        perp_diffusion=DEFAULT_PERP_DIFFUSION,
        diffusion_only=False,
        source_profile=SOURCE_PROFILE,
        source_x0=SOURCE_X0,
        source_delta_x=SOURCE_DELTA_X,
        source_amplitude=DENSITY_SOURCE_AMPLITUDE,
    )

    missing_formula = dict(metadata)
    missing_formula.pop("initial_velocity_formula")
    assert not _history_matches_eb_blob_settings(
        missing_formula,
        resolution=8,
        num_steps=5,
        initial_velocity_state="sheath_taper",
        initial_transient_time=None,
        initial_num_steps=None,
        remaining_num_steps=None,
        a_n=A_N,
        l_rho_cells=Lrho_cells,
        l_y_cells=Ly_cells,
        radial_b_fraction=radial_b_fraction,
        perp_diffusion=DEFAULT_PERP_DIFFUSION,
        diffusion_only=False,
        source_profile=SOURCE_PROFILE,
        source_x0=SOURCE_X0,
        source_delta_x=SOURCE_DELTA_X,
        source_amplitude=DENSITY_SOURCE_AMPLITUDE,
    )

    stale_smoothing = dict(metadata)
    stale_smoothing["wall_sign_smoothing_width_cells"] = WALL_SIGN_SMOOTHING_WIDTH_CELLS + 1.0
    assert not _history_matches_eb_blob_settings(
        stale_smoothing,
        resolution=8,
        num_steps=5,
        initial_velocity_state="sheath_taper",
        initial_transient_time=None,
        initial_num_steps=None,
        remaining_num_steps=None,
        a_n=A_N,
        l_rho_cells=Lrho_cells,
        l_y_cells=Ly_cells,
        radial_b_fraction=radial_b_fraction,
        perp_diffusion=DEFAULT_PERP_DIFFUSION,
        diffusion_only=False,
        source_profile=SOURCE_PROFILE,
        source_x0=SOURCE_X0,
        source_delta_x=SOURCE_DELTA_X,
        source_amplitude=DENSITY_SOURCE_AMPLITUDE,
    )


def _eb_blob_z_indices(geometry: FciGeometry3D, center: float, count: int = 4) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    center_index = int(np.argmin(np.abs(z_values - float(center))))
    offsets = np.arange(-(count // 2), -(count // 2) + count, dtype=np.int64)
    return tuple(int((center_index + offset) % int(z_values.shape[0])) for offset in offsets)


def _signed_norm(values: np.ndarray):
    import matplotlib.colors as colors

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("cannot build a movie from non-finite history values")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _fluctuation_norm(values: np.ndarray, *, percentile: float = 99.0):
    import matplotlib.colors as colors

    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot build a movie from non-finite history values")
    bound = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = float(np.max(np.abs(finite)))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    return colors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)


def _capped_signed_norm(values: np.ndarray, *, cap: float):
    import matplotlib.colors as colors

    cap_value = float(cap)
    if cap_value <= 0.0:
        raise ValueError(f"cap must be positive, got {cap_value}")
    return colors.TwoSlopeNorm(vcenter=0.0, vmin=-cap_value, vmax=cap_value)


def _sequential_norm(values: np.ndarray):
    import matplotlib.colors as colors

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("cannot build a movie from non-finite history values")
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _load_eb_blob_history(
    history_path: Path,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    dict[str, object],
]:
    with np.load(history_path, allow_pickle=False) as history:
        times = jnp.asarray(history["times"], dtype=jnp.float64)
        density_history = jnp.asarray(history["density"], dtype=jnp.float64)
        phi_history = jnp.asarray(history["phi"], dtype=jnp.float64)
        te_history = jnp.asarray(history["Te"], dtype=jnp.float64)
        ti_history = jnp.asarray(history["Ti"], dtype=jnp.float64)
        vi_history = jnp.asarray(history["Vi"], dtype=jnp.float64)
        ve_history = jnp.asarray(history["Ve"], dtype=jnp.float64)
        vorticity_history = jnp.asarray(history["vorticity"], dtype=jnp.float64)
        metadata: dict[str, object] = {}
        for key in history.files:
            if key in {"times", "density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"}:
                continue
            value = history[key]
            metadata[key] = value.item() if getattr(value, "shape", ()) == () else value
    return (
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        metadata,
    )


def _eb_blob_step_dump_path(output_dir: Path, step_index: int) -> Path:
    return output_dir / f"step_{step_index:05d}.npz"


def _clear_eb_blob_step_dumps(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for step_file in output_dir.glob("step_*.npz"):
        step_file.unlink()


def _diagnostic_float(diagnostics: dict[str, object], *keys: str) -> float:
    for key in keys:
        if key in diagnostics:
            return float(diagnostics[key])
    raise KeyError(f"none of the diagnostics keys {keys} were present; available keys were {tuple(sorted(diagnostics.keys()))}")


def _save_eb_blob_step_snapshot(
    output_dir: Path,
    step_index: int,
    time_value: float,
    state: FciDrbEBState,
    *,
    step_gmres_stats: jnp.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "step_index": np.asarray(step_index, dtype=np.int64),
        "time": np.asarray(time_value, dtype=np.float64),
        "density": np.asarray(state.density, dtype=np.float64),
        "phi": np.asarray(state.phi, dtype=np.float64),
        "Te": np.asarray(state.Te, dtype=np.float64),
        "Ti": np.asarray(state.Ti, dtype=np.float64),
        "Vi": np.asarray(state.Vi, dtype=np.float64),
        "Ve": np.asarray(state.Ve, dtype=np.float64),
        "vorticity": np.asarray(state.vorticity, dtype=np.float64),
    }
    if step_gmres_stats is not None:
        step_stats = np.asarray(step_gmres_stats, dtype=np.float64)
        payload.update(
            {
                "phi_time": np.asarray(step_stats[0], dtype=np.float64),
                "rhs_time": np.asarray(step_stats[1], dtype=np.float64),
                "rk4_time": np.asarray(step_stats[2], dtype=np.float64),
                "gmres_steps": np.asarray(step_stats[3], dtype=np.float64),
                "gmres_rel_res": np.asarray(step_stats[4], dtype=np.float64),
                "phi_correction_residual_l2": np.asarray(step_stats[5], dtype=np.float64),
                "phi_correction_residual_linf": np.asarray(step_stats[6], dtype=np.float64),
                "phi_physical_residual_l2": np.asarray(step_stats[7], dtype=np.float64),
                "phi_physical_residual_linf": np.asarray(step_stats[8], dtype=np.float64),
            }
        )
    np.savez_compressed(_eb_blob_step_dump_path(output_dir, step_index), **payload)


def _save_eb_blob_time_traces(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    phi_history: jnp.ndarray,
    te_history: jnp.ndarray,
    ti_history: jnp.ndarray,
    vi_history: jnp.ndarray,
    ve_history: jnp.ndarray,
    vorticity_history: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    output_path: str,
    title: str = "Shifted-torus EB blob time traces",
) -> None:
    import matplotlib.pyplot as plt

    times_np = np.asarray(times, dtype=np.float64)
    density_np = np.asarray(density_history, dtype=np.float64)
    phi_np = np.asarray(phi_history, dtype=np.float64)
    te_np = np.asarray(te_history, dtype=np.float64)
    ti_np = np.asarray(ti_history, dtype=np.float64)
    vi_np = np.asarray(vi_history, dtype=np.float64)
    ve_np = np.asarray(ve_history, dtype=np.float64)
    vorticity_np = np.asarray(vorticity_history, dtype=np.float64)
    cell_j = np.asarray(geometry.cell_metric.J, dtype=np.float64)
    j_total = np.sum(cell_j)
    j_weighted_mean = lambda values: np.sum(values * cell_j[None, :, :, :], axis=(1, 2, 3)) / j_total

    density_mean = np.mean(density_np, axis=(1, 2, 3))
    density_j_mean = j_weighted_mean(density_np)
    density_rms = np.sqrt(np.mean((density_np - density_mean[:, None, None, None]) ** 2, axis=(1, 2, 3)))
    phi_rms = np.sqrt(np.mean(phi_np**2, axis=(1, 2, 3)))
    te_mean = np.mean(te_np, axis=(1, 2, 3))
    te_min = np.min(te_np, axis=(1, 2, 3))
    te_max = np.max(te_np, axis=(1, 2, 3))
    ti_mean = np.mean(ti_np, axis=(1, 2, 3))
    ti_min = np.min(ti_np, axis=(1, 2, 3))
    ti_max = np.max(ti_np, axis=(1, 2, 3))
    vi_rms = np.sqrt(np.mean(vi_np**2, axis=(1, 2, 3)))
    ve_rms = np.sqrt(np.mean(ve_np**2, axis=(1, 2, 3)))
    vorticity_rms = np.sqrt(np.mean(vorticity_np**2, axis=(1, 2, 3)))
    phi_max = np.max(np.abs(phi_np), axis=(1, 2, 3))
    density_min = np.min(density_np, axis=(1, 2, 3))
    density_max = np.max(density_np, axis=(1, 2, 3))
    j_weighted_density_te = j_weighted_mean(density_np * te_np)
    j_weighted_density_ti = j_weighted_mean(density_np * ti_np)
    j_weighted_phi_omega = j_weighted_mean(phi_np * vorticity_np)
    j_weighted_density_vi2 = j_weighted_mean(density_np * (vi_np**2))
    j_weighted_density_ve2 = j_weighted_mean(density_np * (ve_np**2))

    series = (
        ("mean(n)", density_mean),
        ("J-weighted mean(n)", density_j_mean),
        ("rms(n-mean(n))", density_rms),
        ("rms(phi)", phi_rms),
        ("mean(Te)", te_mean),
        ("min(Te)", te_min),
        ("max(Te)", te_max),
        ("mean(Ti)", ti_mean),
        ("min(Ti)", ti_min),
        ("max(Ti)", ti_max),
        ("rms(Vi)", vi_rms),
        ("rms(Ve)", ve_rms),
        ("min(n)", density_min),
        ("max(n)", density_max),
        ("rms(omega)", vorticity_rms),
        ("J-weighted nTe", j_weighted_density_te),
        ("J-weighted nTi", j_weighted_density_ti),
        ("J-weighted phi*omega", j_weighted_phi_omega),
        ("J-weighted nVi^2", j_weighted_density_vi2),
        ("J-weighted nVe^2", j_weighted_density_ve2),
    )

    fig, axes = plt.subplots(5, 4, figsize=(18.0, 18.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for index, (label, values) in enumerate(series):
        ax = axes_flat[index]
        ax.set_title(label)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
        if values is not None:
            ax.plot(times_np, values, linewidth=1.8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_eb_blob_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    phi_history: jnp.ndarray,
    te_history: jnp.ndarray,
    ti_history: jnp.ndarray,
    vi_history: jnp.ndarray,
    ve_history: jnp.ndarray,
    vorticity_history: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    output_path: str,
    frame_stride: int = 2,
    title: str = "Shifted-torus EB blob state evolution",
    z_indices: tuple[int, int, int, int] | None = None,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    y_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    if z_indices is None:
        z_indices = tuple(int(idx) for idx in np.linspace(0, int(z_values.shape[0] - 1), 4))
    y_grid, radius_grid = np.meshgrid(y_values, x_values)

    density_fluctuation = np.asarray(density_history, dtype=np.float64) - 1.0
    phi_np = np.asarray(phi_history, dtype=np.float64)
    phi_mean = np.mean(phi_np, axis=(1, 2, 3), keepdims=True)
    phi_fluctuation = phi_np - phi_mean
    te_fluctuation = np.asarray(te_history, dtype=np.float64) - 1.0
    ti_fluctuation = np.asarray(ti_history, dtype=np.float64) - 1.0
    field_specs = (
        ("density fluctuation", density_fluctuation, "coolwarm", _fluctuation_norm(density_fluctuation)),
        ("phi fluctuation", phi_fluctuation, "coolwarm", _signed_norm(phi_fluctuation)),
        ("Te fluctuation", te_fluctuation, "coolwarm", _fluctuation_norm(te_fluctuation)),
        ("Ti fluctuation", ti_fluctuation, "coolwarm", _fluctuation_norm(ti_fluctuation)),
        ("Vi", np.asarray(vi_history, dtype=np.float64), "coolwarm", _signed_norm(np.asarray(vi_history, dtype=np.float64))),
        ("Ve", np.asarray(ve_history, dtype=np.float64), "coolwarm", _signed_norm(np.asarray(ve_history, dtype=np.float64))),
        ("vorticity", np.asarray(vorticity_history, dtype=np.float64), "coolwarm", _capped_signed_norm(np.asarray(vorticity_history, dtype=np.float64), cap=2.0)),
    )
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)
    contour_levels = [
        np.linspace(float(norm.vmin), float(norm.vmax), 21, dtype=np.float64)
        for _, _, _, norm in field_specs
    ]

    fig, axes = plt.subplots(
        nrows=len(field_specs),
        ncols=4,
        figsize=(17.0, 24.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    if len(field_specs) == 1:
        axes = np.asarray([axes])

    for row, (_, _, cmap, norm) in enumerate(field_specs):
        colorbar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar_mappable.set_array([])
        fig.colorbar(
            colorbar_mappable,
            ax=list(axes[row, :]),
            location="right",
            pad=0.02,
            shrink=0.88,
            extend="both",
        )

    suptitle = fig.suptitle(title)

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for row, (field_name, field_data, cmap, norm) in enumerate(field_specs):
            levels = contour_levels[row]
            for col, z_index in enumerate(z_indices):
                ax = axes[row, col]
                ax.clear()
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_ylim(0.0, float(x_values[-1]))
                ax.set_yticklabels([])
                ax.contourf(
                    y_grid,
                    radius_grid,
                    field_data[actual_index, :, :, z_index],
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    extend="both",
                )
                ax.set_title(f"{field_name}, z={z_values[z_index]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"{title}, t={time_value:.3f}")
        return []

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


def _axis_regular_conservative_stencil_from_field(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
    face_bc: BoundaryFaceBC3D | None,
) -> ConservativeStencil3D:
    normalized_periodic_axes = _normalize_periodic_axes(periodic_axes)
    base = build_conservative_stencil_from_field(
        field,
        geometry,
        normalized_periodic_axes,
        face_bc,
    )
    if normalized_periodic_axes[0] or not AXIS_REGULAR_AXES[0]:
        return base
    return base.replace(
        x=_axis_regular_x_stencil(base.x, field, geometry),
    )


def _axis_regular_local_stencil_from_field(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
    face_bc: BoundaryFaceBC3D | None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
) -> LocalStencil3D:
    del cut_wall_geometry, cut_wall_bc
    normalized_periodic_axes = _normalize_periodic_axes(periodic_axes)
    conservative = _axis_regular_conservative_stencil_from_field(
        field,
        geometry,
        normalized_periodic_axes,
        face_bc,
    )
    x_stencil = conservative.x
    if not normalized_periodic_axes[0]:
        x_stencil = _outer_x_one_sided_local_stencil(x_stencil, field, geometry)
    return LocalStencil3D(
        x=x_stencil,
        y=conservative.y,
        z=conservative.z,
    )


def test_coordinate_axis_regular_lower_x_boundary_constructors() -> None:
    geometry = _build_eb_blob_geometry((4, 4, 4), construct_fci_maps=False)
    field = jnp.arange(np.prod(geometry.shape), dtype=jnp.float64).reshape(geometry.shape)

    face_reconstructor = CoordinateFaceValueReconstructor3D()
    normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)

    x_faces, _, _ = face_reconstructor.extrapolate(
        field,
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    expected_lower_x = 0.5 * (field[0] + jnp.roll(field[0], shift=-(geometry.shape[1] // 2), axis=0))
    np.testing.assert_allclose(np.asarray(x_faces[0]), np.asarray(expected_lower_x))

    wall_value = face_reconstructor.extrapolate(
        field,
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    dnormal_faces, d2normal_faces = normal_derivative_constructor.normal_derivatives_from_wall_value(
        field,
        wall_value,
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    assert dnormal_faces[0].shape == geometry.shape[1:]
    assert d2normal_faces[0].shape == geometry.shape[1:]
    assert np.isfinite(np.asarray(dnormal_faces[0])).all()
    assert np.isfinite(np.asarray(d2normal_faces[0])).all()


conservative_stencil_builder = ConservativeStencilBuilder(
    _axis_regular_conservative_stencil_from_field
)
local_stencil_builder = LocalStencilBuilder(
    _axis_regular_local_stencil_from_field
)


def test_eb_local_stencil_uses_second_order_one_sided_outer_x_derivative() -> None:
    geometry = _build_eb_blob_geometry((4, 8, 4), construct_fci_maps=False)
    x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
    field = jnp.broadcast_to(x * x, geometry.shape)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))

    stencil = local_stencil_builder(field, geometry, PERIODIC_AXES, face_bc)
    dfdx = _take_stencil_finite_difference(stencil.x)

    expected_outer = jnp.full_like(dfdx[-1], 2.0 * x[-1, 0, 0])
    np.testing.assert_allclose(np.asarray(dfdx[-1]), np.asarray(expected_outer), atol=1.0e-12)


def test_eb_conservative_stencil_keeps_default_outer_x_boundary_payload() -> None:
    geometry = _build_eb_blob_geometry((4, 8, 4), construct_fci_maps=False)
    x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
    field = jnp.broadcast_to(x * x, geometry.shape)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))

    conservative = conservative_stencil_builder(field, geometry, PERIODIC_AXES, face_bc)
    local = local_stencil_builder(field, geometry, PERIODIC_AXES, face_bc)

    expected_conservative_outer_plus = 2.0 * field[-1] - field[-2]
    np.testing.assert_allclose(
        np.asarray(conservative.x.plus[-1]),
        np.asarray(expected_conservative_outer_plus),
    )
    np.testing.assert_allclose(np.asarray(local.x.plus[-1]), np.asarray(field[-3]))


def test_multiply_local_stencils_preserves_one_sided_derivative_weights() -> None:
    geometry = _build_eb_blob_geometry((4, 8, 4), construct_fci_maps=False)
    x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
    left_field = jnp.broadcast_to(1.0 + x, geometry.shape)
    right_field = jnp.broadcast_to(2.0 + x * x, geometry.shape)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))

    left = local_stencil_builder(left_field, geometry, PERIODIC_AXES, face_bc)
    right = local_stencil_builder(right_field, geometry, PERIODIC_AXES, face_bc)
    product = _multiply_local_stencils(left, right)

    np.testing.assert_allclose(
        np.asarray(product.x.derivative_minus_weight),
        np.asarray(left.x.derivative_minus_weight),
    )
    np.testing.assert_allclose(
        np.asarray(product.x.derivative_center_weight),
        np.asarray(left.x.derivative_center_weight),
    )
    np.testing.assert_allclose(
        np.asarray(product.x.derivative_plus_weight),
        np.asarray(left.x.derivative_plus_weight),
    )


def _build_eb_blob_geometry(
    shape: tuple[int, int, int],
    *,
    radial_fraction: float = radial_b_fraction,
    construct_fci_maps: bool = True,
) -> FciGeometry3D:
    rho_min = 0.5 / float(shape[0])
    rho_max = 1.0 - rho_min
    base_geometry = build_shifted_torus_4field_geometry(
        shape,
        x_min=rho_min,
        x_max=rho_max,
        construct_fci_maps=False,
    )
    grid = base_geometry.grid
    cell_theta = jnp.broadcast_to(grid.y.centers[None, :, None], shape)
    cell_radial_scale = radial_fraction * jnp.asarray(base_geometry.cell_bfield.B_contra[..., 2], dtype=jnp.float64)
    cell_Bx = cell_radial_scale * jnp.cos(cell_theta)
    cell_By = jnp.asarray(base_geometry.cell_bfield.B_contra[..., 1], dtype=jnp.float64)
    cell_Bz = jnp.asarray(base_geometry.cell_bfield.B_contra[..., 2], dtype=jnp.float64)
    cell_B_contra = jnp.stack((cell_Bx, cell_By, cell_Bz), axis=-1)
    cell_bfield = BFieldGeometry(
        B_contra=cell_B_contra,
        Bmag=_bmag_from_contravariant_components(cell_B_contra, base_geometry.cell_metric.g_cov),
    )

    face_theta_x = jnp.broadcast_to(grid.y.centers[None, :, None], base_geometry.face_metric.x.shape)
    face_radial_scale_x = radial_fraction * jnp.asarray(base_geometry.face_bfield.x.B_contra[..., 2], dtype=jnp.float64)
    face_Bx_x = face_radial_scale_x * jnp.cos(face_theta_x)
    face_By_x = jnp.asarray(base_geometry.face_bfield.x.B_contra[..., 1], dtype=jnp.float64)
    face_Bz_x = jnp.asarray(base_geometry.face_bfield.x.B_contra[..., 2], dtype=jnp.float64)
    face_B_contra_x = jnp.stack((face_Bx_x, face_By_x, face_Bz_x), axis=-1)

    face_theta_y = jnp.broadcast_to(grid.y.faces[None, :, None], base_geometry.face_metric.y.shape)
    face_radial_scale_y = radial_fraction * jnp.asarray(base_geometry.face_bfield.y.B_contra[..., 2], dtype=jnp.float64)
    face_Bx_y = face_radial_scale_y * jnp.cos(face_theta_y)
    face_By_y = jnp.asarray(base_geometry.face_bfield.y.B_contra[..., 1], dtype=jnp.float64)
    face_Bz_y = jnp.asarray(base_geometry.face_bfield.y.B_contra[..., 2], dtype=jnp.float64)
    face_B_contra_y = jnp.stack((face_Bx_y, face_By_y, face_Bz_y), axis=-1)

    face_theta_z = jnp.broadcast_to(grid.y.centers[None, :, None], base_geometry.face_metric.z.shape)
    face_radial_scale_z = radial_fraction * jnp.asarray(base_geometry.face_bfield.z.B_contra[..., 2], dtype=jnp.float64)
    face_Bx_z = face_radial_scale_z * jnp.cos(face_theta_z)
    face_By_z = jnp.asarray(base_geometry.face_bfield.z.B_contra[..., 1], dtype=jnp.float64)
    face_Bz_z = jnp.asarray(base_geometry.face_bfield.z.B_contra[..., 2], dtype=jnp.float64)
    face_B_contra_z = jnp.stack((face_Bx_z, face_By_z, face_Bz_z), axis=-1)

    face_bfield = FaceBFieldGeometry(
        x=BFieldGeometry(
            B_contra=face_B_contra_x,
            Bmag=_bmag_from_contravariant_components(face_B_contra_x, base_geometry.face_metric.x.g_cov),
        ),
        y=BFieldGeometry(
            B_contra=face_B_contra_y,
            Bmag=_bmag_from_contravariant_components(face_B_contra_y, base_geometry.face_metric.y.g_cov),
        ),
        z=BFieldGeometry(
            B_contra=face_B_contra_z,
            Bmag=_bmag_from_contravariant_components(face_B_contra_z, base_geometry.face_metric.z.g_cov),
        ),
    )

    if construct_fci_maps:
        map_fields = build_fci_maps_from_b_contravariant(
            grid,
            cell_bfield.B_contra,
            cell_bfield.Bmag,
            periodic_axes=PERIODIC_AXES,
        )
        maps = FciMaps3D(
            forward_x=map_fields["forward_x"],
            forward_y=map_fields["forward_y"],
            backward_x=map_fields["backward_x"],
            backward_y=map_fields["backward_y"],
            forward_endpoint_x=map_fields["forward_endpoint_x"],
            forward_endpoint_y=map_fields["forward_endpoint_y"],
            forward_endpoint_z=map_fields["forward_endpoint_z"],
            backward_endpoint_x=map_fields["backward_endpoint_x"],
            backward_endpoint_y=map_fields["backward_endpoint_y"],
            backward_endpoint_z=map_fields["backward_endpoint_z"],
            forward_length=map_fields["forward_length"],
            backward_length=map_fields["backward_length"],
            forward_boundary=map_fields["forward_boundary"],
            backward_boundary=map_fields["backward_boundary"],
        )
    else:
        zeros = jnp.zeros(shape, dtype=jnp.float64)
        ones = jnp.ones(shape, dtype=jnp.float64)
        maps = FciMaps3D(
            forward_x=zeros,
            forward_y=zeros,
            backward_x=zeros,
            backward_y=zeros,
            forward_endpoint_x=zeros,
            forward_endpoint_y=zeros,
            forward_endpoint_z=zeros,
            backward_endpoint_x=zeros,
            backward_endpoint_y=zeros,
            backward_endpoint_z=zeros,
            forward_length=ones,
            backward_length=ones,
            forward_boundary=zeros.astype(bool),
            backward_boundary=zeros.astype(bool),
        )

    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=base_geometry.cell_metric,
        face_metric=base_geometry.face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )


def _build_eb_boundary_conditions(
    state: FciDrbEBState | jnp.ndarray,
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
    cut_wall_geometry: CutWallGeometry3D | None,
    cut_wall_bc: CutWallBC3D | None,
    *,
    face_reconstructor: CoordinateFaceValueReconstructor3D,
    normal_derivative_constructor: CoordinateNormalDerivativeConstructor3D,
) -> FciDrbEBBoundaryConditions:
    del cut_wall_geometry
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    empty_face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    empty_cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    normalized_periodic_axes = _normalize_periodic_axes(periodic_axes)
    axis_regular_axes = tuple(
        bool(axis_regular) and not bool(periodic)
        for axis_regular, periodic in zip(AXIS_REGULAR_AXES, normalized_periodic_axes)
    )

    density = jnp.asarray(getattr(state, "density", state), dtype=jnp.float64)
    phi = jnp.asarray(getattr(state, "phi", state), dtype=jnp.float64)
    Te = jnp.asarray(getattr(state, "Te", state), dtype=jnp.float64)
    Ti = jnp.asarray(getattr(state, "Ti", state), dtype=jnp.float64)
    Vi = jnp.asarray(getattr(state, "Vi", state), dtype=jnp.float64)
    zero_face_derivative = (
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64),
        jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
        jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
    )
    Te_faces = face_reconstructor.extrapolate_neumann(
        Te,
        geometry,
        normal_derivative=zero_face_derivative,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )
    Ti_faces = face_reconstructor.extrapolate_neumann(
        Ti,
        geometry,
        normal_derivative=zero_face_derivative,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )
    phi_faces = face_reconstructor.extrapolate(
        phi,
        geometry,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )

    # Shared sheath parameters and x-wall sign logic.
    tau = jnp.asarray(1.0, dtype=jnp.float64)
    lambda_sheath = jnp.log(jnp.sqrt(jnp.asarray(1836.0, dtype=jnp.float64)) / (2.0 * jnp.pi))
    _, x_sheath_sign, outer_zero_sign_mask, outer_wall_derivative_keep_mask = _outer_x_wall_sheath_sign_data(
        geometry,
        normalized_periodic_axes,
    )

    Ft_faces = tuple(
        jnp.sqrt(jnp.maximum(1.0 + tau * ti_face / jnp.maximum(te_face, 1.0e-30), 0.0))
        for te_face, ti_face in zip(Te_faces, Ti_faces)
    )
    te_face_x = jnp.maximum(Te_faces[0], 1.0e-30)
    ft_face_x = Ft_faces[0]

    # Wall values for the coupled sheath conditions on the x faces.
    vi_wall_x = x_sheath_sign * jnp.sqrt(te_face_x) * ft_face_x
    phi_dirichlet_x = lambda_sheath * te_face_x
    phi_face_x = phi_faces[0].at[-1].set(phi_dirichlet_x[-1])
    ve_wall_x = x_sheath_sign * jnp.sqrt(te_face_x) * jnp.exp(lambda_sheath - phi_face_x / te_face_x)

    vi_face_values = face_reconstructor.extrapolate(
        Vi,
        geometry,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )
    vi_outer_wall_value = jnp.where(outer_zero_sign_mask, vi_face_values[0][-1], vi_wall_x[-1])
    vi_face_values = (
        vi_face_values[0].at[-1].set(vi_outer_wall_value),
        vi_face_values[1],
        vi_face_values[2],
    )
    vi_dnormal_faces, vi_d2normal_faces = normal_derivative_constructor.normal_derivatives_from_wall_value(
        Vi,
        vi_face_values,
        geometry,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )
    vi_dnormal_x_for_bc = vi_dnormal_faces[0].at[-1].set(
        jnp.where(outer_wall_derivative_keep_mask, vi_dnormal_faces[0][-1], 0.0)
    )
    vi_d2normal_x_for_bc = vi_d2normal_faces[0].at[-1].set(
        jnp.where(outer_wall_derivative_keep_mask, vi_d2normal_faces[0][-1], 0.0)
    )
    density_face_values = face_reconstructor.extrapolate(
        density,
        geometry,
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )

    # Normal derivatives needed by the sheath boundary relations.
    # These are outward normal derivatives on the x faces.
    density_neumann_x = jnp.where(
        outer_zero_sign_mask,
        0.0,
        -x_sheath_sign[-1]
        * (
            jnp.maximum(density_face_values[0], 1.0e-30)
            / (jnp.maximum(te_face_x, 1.0e-30) * jnp.maximum(ft_face_x, 1.0e-30))
        )
        * vi_dnormal_x_for_bc,
    )
    density_faces = face_reconstructor.extrapolate_neumann(
        density,
        geometry,
        normal_derivative=(
            density_neumann_x,
            jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
            jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
        ),
        periodic_axes=normalized_periodic_axes,
        axis_regular_axes=axis_regular_axes,
    )
    phi_neumann_x = jnp.where(
        outer_zero_sign_mask,
        0.0,
        -x_sheath_sign[-1]
        * (jnp.maximum(density_faces[0], 1.0e-30) / jnp.maximum(te_face_x, 1.0e-30))
        * vi_dnormal_x_for_bc,
    )
    vorticity_wall_x = jnp.where(
        outer_zero_sign_mask,
        0.0,
        -(
            (vi_dnormal_x_for_bc ** 2) / jnp.maximum(ft_face_x**2, 1.0e-30)
            + x_sheath_sign[-1] * jnp.sqrt(te_face_x) / jnp.maximum(ft_face_x, 1.0e-30) * vi_d2normal_x_for_bc
        ),
    )

    def _x_face_bc(kind_x: jnp.ndarray, value_x: jnp.ndarray) -> BoundaryFaceBC3D:
        return BoundaryFaceBC3D(
            kind_x=kind_x,
            kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
            kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
            value_x=value_x,
            value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
            value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
            # Lower rho is axis-regular topology; only the outer x face is a wall.
            mask_x=jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool).at[-1].set(True),
            mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
            mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
        )

    density_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(BC_NEUMANN),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(density_neumann_x[-1]),
    )
    if normalized_periodic_axes[0]:
        density_face_bc = empty_face_bc

    temperature_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(BC_NEUMANN),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64),
    )
    if normalized_periodic_axes[0]:
        temperature_face_bc = empty_face_bc
    # vorticity_face_bc = _x_face_bc(
    #     jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(BC_NEUMANN),
    #     jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64),
    # )
    vorticity_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(
            jnp.where(outer_zero_sign_mask, BC_NEUMANN, BC_DIRICHLET)
        ),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(
            jnp.where(outer_zero_sign_mask, 0.0, vorticity_wall_x[-1])
        ),
    )
    if normalized_periodic_axes[0]:
        vorticity_face_bc = empty_face_bc
    vi_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(
            jnp.where(outer_zero_sign_mask, BC_NEUMANN, BC_DIRICHLET)
        ),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(
            jnp.where(outer_zero_sign_mask, 0.0, vi_wall_x[-1])
        ),
    )
    if normalized_periodic_axes[0]:
        vi_face_bc = empty_face_bc
    ve_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(
            jnp.where(outer_zero_sign_mask, BC_NEUMANN, BC_DIRICHLET)
        ),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(
            jnp.where(outer_zero_sign_mask, 0.0, ve_wall_x[-1])
        ),
    )
    if normalized_periodic_axes[0]:
        ve_face_bc = empty_face_bc
    # potential_face_bc = _x_face_bc(
    #     jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(BC_NEUMANN),
    #     jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(phi_neumann_x[-1]),
    # )
    potential_face_bc = _x_face_bc(
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[-1].set(BC_DIRICHLET),
        jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[-1].set(phi_dirichlet_x[-1]),
    )
    if normalized_periodic_axes[0]:
        potential_face_bc = empty_face_bc
    # If we need the old Neumann potential wall condition again, restore the
    # commented-out Neumann phi block above and switch the solver back to a
    # mean-zero projection with Neumann phi.

    return FciDrbEBBoundaryConditions(
        density_face_bc=density_face_bc,
        density_cut_wall_bc=empty_cut_wall_bc,
        potential_face_bc=potential_face_bc,
        potential_cut_wall_bc=empty_cut_wall_bc,
        vorticity_face_bc=vorticity_face_bc,
        vorticity_cut_wall_bc=empty_cut_wall_bc,
        Te_face_bc=temperature_face_bc,
        Te_cut_wall_bc=empty_cut_wall_bc,
        Ti_face_bc=temperature_face_bc,
        Ti_cut_wall_bc=empty_cut_wall_bc,
        Vi_face_bc=vi_face_bc,
        Vi_cut_wall_bc=empty_cut_wall_bc,
        Ve_face_bc=ve_face_bc,
        Ve_cut_wall_bc=empty_cut_wall_bc,
    )


def test_outer_wall_sign_smoothing_reduces_flip_cells_and_preserves_far_sign() -> None:
    ny = 32
    nz = 2
    theta = jnp.arange(ny, dtype=jnp.float64) * (2.0 * jnp.pi / float(ny))
    hard_sign = jnp.broadcast_to(jnp.where(jnp.cos(theta) >= 0.0, 1.0, -1.0)[:, None], (ny, nz))

    smoothed_sign = _smoothed_outer_wall_sign(
        hard_sign,
        theta,
        WALL_SIGN_SMOOTHING_WIDTH_CELLS,
    )
    flip_mask = _wall_sign_flip_mask(hard_sign)
    far_indices = jnp.asarray([0, ny // 2], dtype=jnp.int32)

    assert np.all(np.abs(np.asarray(smoothed_sign[flip_mask])) < 0.25)
    assert np.all(np.abs(np.asarray(smoothed_sign[far_indices, :])) > 0.85)
    np.testing.assert_array_equal(
        np.sign(np.asarray(smoothed_sign[far_indices, :])),
        np.sign(np.asarray(hard_sign[far_indices, :])),
    )


def test_outer_wall_derivative_keep_mask_marks_periodic_flip_neighbors() -> None:
    hard_sign = jnp.asarray([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=jnp.float64)[:, None]
    keep_mask = _outer_wall_derivative_keep_mask(hard_sign, width_cells=0.0)
    expected_smoothing_mask = np.asarray([True, False, False, True, True, False, False, True])[:, None]

    np.testing.assert_array_equal(np.logical_not(np.asarray(keep_mask)), expected_smoothing_mask)


def test_eb_boundary_conditions_with_wall_sign_smoothing_are_finite() -> None:
    geometry = _build_eb_blob_geometry((8, 32, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    face_reconstructor = CoordinateFaceValueReconstructor3D()
    normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)

    boundary_conditions = _build_eb_boundary_conditions(
        state,
        geometry,
        PERIODIC_AXES,
        CutWallGeometry3D.empty(),
        CutWallBC3D.empty(),
        face_reconstructor=face_reconstructor,
        normal_derivative_constructor=normal_derivative_constructor,
    )

    for face_bc in (
        boundary_conditions.Vi_face_bc,
        boundary_conditions.Ve_face_bc,
        boundary_conditions.density_face_bc,
        boundary_conditions.vorticity_face_bc,
        boundary_conditions.potential_face_bc,
    ):
        assert np.isfinite(np.asarray(face_bc.value_x[-1])).all()


def _test_eb_rhs_boundary_conditions(geometry: FciGeometry3D, state: FciDrbEBState) -> FciDrbEBBoundaryConditions:
    face_reconstructor = CoordinateFaceValueReconstructor3D()
    normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
    return _build_eb_boundary_conditions(
        state,
        geometry,
        PERIODIC_AXES,
        CutWallGeometry3D.empty(),
        CutWallBC3D.empty(),
        face_reconstructor=face_reconstructor,
        normal_derivative_constructor=normal_derivative_constructor,
    )


def test_eb_boundary_conditions_use_shared_smoothed_outer_wall_sign() -> None:
    geometry = _build_eb_blob_geometry((8, 32, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    x_hard_sign, x_sheath_sign, outer_zero_sign_mask, _ = _outer_x_wall_sheath_sign_data(geometry)

    expected_ve_wall = np.where(np.asarray(outer_zero_sign_mask), 0.0, np.asarray(x_sheath_sign[-1]))
    expected_vi_wall = np.sqrt(2.0) * expected_ve_wall
    np.testing.assert_allclose(
        np.asarray(boundary_conditions.Ve_face_bc.value_x[-1]),
        expected_ve_wall,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(boundary_conditions.Vi_face_bc.value_x[-1]),
        expected_vi_wall,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    flip_mask = np.asarray(_wall_sign_flip_mask(x_hard_sign[-1]))
    assert np.any(flip_mask)
    assert np.max(np.abs(np.asarray(boundary_conditions.Ve_face_bc.value_x[-1])[flip_mask])) < 0.25


def test_eb_boundary_sheath_derivative_relations_are_masked_in_smoothing_region() -> None:
    geometry = _build_eb_blob_geometry((8, 32, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    _, _, _, outer_wall_derivative_keep_mask = _outer_x_wall_sheath_sign_data(geometry)
    blocked_mask = np.logical_not(np.asarray(outer_wall_derivative_keep_mask))

    assert np.any(blocked_mask)
    np.testing.assert_allclose(
        np.asarray(boundary_conditions.density_face_bc.value_x[-1])[blocked_mask],
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(boundary_conditions.vorticity_face_bc.value_x[-1])[blocked_mask],
        0.0,
        atol=1.0e-12,
    )


def test_eb_rhs_diffusion_only_zero_coefficients_returns_zero_rhs() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    parameters = FciDrbEBRhsParameters()
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()

    result = compute_fci_drb_eb_rhs(
        state,
        geometry=geometry,
        stencil_builder=local_stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
        density_face_bc=boundary_conditions.density_face_bc,
        potential_face_bc=boundary_conditions.potential_face_bc,
        vorticity_face_bc=boundary_conditions.vorticity_face_bc,
        electron_temperature_face_bc=boundary_conditions.Te_face_bc,
        ion_temperature_face_bc=boundary_conditions.Ti_face_bc,
        electron_velocity_parallel_face_bc=boundary_conditions.Ve_face_bc,
        ion_velocity_parallel_face_bc=boundary_conditions.Vi_face_bc,
        density_cut_wall_geometry=cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        potential_cut_wall_geometry=cut_wall_geometry,
        potential_cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
        vorticity_cut_wall_geometry=cut_wall_geometry,
        vorticity_cut_wall_bc=boundary_conditions.vorticity_cut_wall_bc,
        electron_temperature_cut_wall_geometry=cut_wall_geometry,
        electron_temperature_cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
        ion_temperature_cut_wall_geometry=cut_wall_geometry,
        ion_temperature_cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
        diffusion_only=True,
    )

    for field in (
        result.rhs.density,
        result.rhs.phi,
        result.rhs.Te,
        result.rhs.Ti,
        result.rhs.Vi,
        result.rhs.Ve,
        result.rhs.vorticity,
    ):
        np.testing.assert_allclose(np.asarray(field), 0.0, atol=0.0)


def test_eb_rhs_optional_sources_are_added_when_enabled() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    parameters = FciDrbEBRhsParameters()
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    density_source = jnp.full(geometry.shape, 0.125, dtype=jnp.float64)
    electron_temperature_source = jnp.full(geometry.shape, 0.25, dtype=jnp.float64)

    def _compute(
        *,
        density_source_value: jax.Array | None = None,
        electron_temperature_source_value: jax.Array | None = None,
    ) -> FciDrbEBRhsResult:
        return compute_fci_drb_eb_rhs(
            state,
            geometry=geometry,
            stencil_builder=local_stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters,
            curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
            density_face_bc=boundary_conditions.density_face_bc,
            potential_face_bc=boundary_conditions.potential_face_bc,
            vorticity_face_bc=boundary_conditions.vorticity_face_bc,
            electron_temperature_face_bc=boundary_conditions.Te_face_bc,
            ion_temperature_face_bc=boundary_conditions.Ti_face_bc,
            electron_velocity_parallel_face_bc=boundary_conditions.Ve_face_bc,
            ion_velocity_parallel_face_bc=boundary_conditions.Vi_face_bc,
            density_cut_wall_geometry=cut_wall_geometry,
            density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
            potential_cut_wall_geometry=cut_wall_geometry,
            potential_cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
            vorticity_cut_wall_geometry=cut_wall_geometry,
            vorticity_cut_wall_bc=boundary_conditions.vorticity_cut_wall_bc,
            electron_temperature_cut_wall_geometry=cut_wall_geometry,
            electron_temperature_cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
            ion_temperature_cut_wall_geometry=cut_wall_geometry,
            ion_temperature_cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
            electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
            electron_velocity_parallel_cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
            ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
            ion_velocity_parallel_cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
            density_source=density_source_value,
            electron_temperature_source=electron_temperature_source_value,
        )

    baseline = _compute()
    result = _compute(
        density_source_value=density_source,
        electron_temperature_source_value=electron_temperature_source,
    )

    np.testing.assert_allclose(np.asarray(result.rhs.density - baseline.rhs.density), np.asarray(density_source))
    np.testing.assert_allclose(np.asarray(result.rhs.Te - baseline.rhs.Te), np.asarray(electron_temperature_source))
    for result_field, baseline_field in (
        (result.rhs.phi, baseline.rhs.phi),
        (result.rhs.Ti, baseline.rhs.Ti),
        (result.rhs.Vi, baseline.rhs.Vi),
        (result.rhs.Ve, baseline.rhs.Ve),
        (result.rhs.vorticity, baseline.rhs.vorticity),
    ):
        np.testing.assert_allclose(np.asarray(result_field), np.asarray(baseline_field), atol=0.0)


def test_eb_rhs_density_parallel_term_uses_particle_flux() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    y = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[None, :, None]
    density = jnp.broadcast_to(1.0 + 0.1 * jnp.sin(y), geometry.shape)
    Ve = jnp.broadcast_to(0.3 + 0.2 * jnp.cos(y), geometry.shape)
    state = FciDrbEBState(
        density=density,
        phi=jnp.zeros(geometry.shape, dtype=jnp.float64),
        Te=jnp.ones(geometry.shape, dtype=jnp.float64),
        Ti=jnp.ones(geometry.shape, dtype=jnp.float64),
        Vi=jnp.zeros(geometry.shape, dtype=jnp.float64),
        Ve=Ve,
        vorticity=jnp.zeros(geometry.shape, dtype=jnp.float64),
    )
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()

    result = compute_fci_drb_eb_rhs(
        state,
        geometry=geometry,
        stencil_builder=local_stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=FciDrbEBRhsParameters(),
        curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
        density_face_bc=face_bc,
        potential_face_bc=face_bc,
        vorticity_face_bc=face_bc,
        electron_temperature_face_bc=face_bc,
        ion_temperature_face_bc=face_bc,
        electron_velocity_parallel_face_bc=face_bc,
        ion_velocity_parallel_face_bc=face_bc,
        density_cut_wall_geometry=cut_wall_geometry,
        density_cut_wall_bc=cut_wall_bc,
        potential_cut_wall_geometry=cut_wall_geometry,
        potential_cut_wall_bc=cut_wall_bc,
        vorticity_cut_wall_geometry=cut_wall_geometry,
        vorticity_cut_wall_bc=cut_wall_bc,
        electron_temperature_cut_wall_geometry=cut_wall_geometry,
        electron_temperature_cut_wall_bc=cut_wall_bc,
        ion_temperature_cut_wall_geometry=cut_wall_geometry,
        ion_temperature_cut_wall_bc=cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
    )

    particle_flux_stencil = local_stencil_builder(
        density * Ve,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )
    ve_stencil = local_stencil_builder(
        Ve,
        geometry,
        PERIODIC_AXES,
        face_bc,
        cut_wall_geometry,
        cut_wall_bc,
    )
    expected_density_rhs = -grad_parallel_op_direct(particle_flux_stencil, geometry)
    old_linearized_rhs = -grad_parallel_op_direct(ve_stencil, geometry)

    np.testing.assert_allclose(np.asarray(result.rhs.density), np.asarray(expected_density_rhs), rtol=1.0e-12, atol=1.0e-12)
    assert np.max(np.abs(np.asarray(expected_density_rhs - old_linearized_rhs))) > 1.0e-6


def test_eb_rhs_diffusion_only_uses_separate_ti_boundary_conditions() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    state = FciDrbEBState(
        density=jnp.ones(geometry.shape, dtype=jnp.float64),
        phi=jnp.zeros(geometry.shape, dtype=jnp.float64),
        Te=jnp.ones(geometry.shape, dtype=jnp.float64),
        Ti=jnp.ones(geometry.shape, dtype=jnp.float64),
        Vi=jnp.zeros(geometry.shape, dtype=jnp.float64),
        Ve=jnp.zeros(geometry.shape, dtype=jnp.float64),
        vorticity=jnp.zeros(geometry.shape, dtype=jnp.float64),
    )
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    ti_face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[-1].set(jnp.full_like(face_bc.kind_x[-1], BC_DIRICHLET)),
        value_x=face_bc.value_x.at[-1].set(jnp.full_like(face_bc.value_x[-1], 2.0)),
        mask_x=face_bc.mask_x.at[-1].set(jnp.ones_like(face_bc.mask_x[-1], dtype=bool)),
    )
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    parameters = FciDrbEBRhsParameters(ion_temperature_D_perp=1.0e-4)

    result = compute_fci_drb_eb_rhs(
        state,
        geometry=geometry,
        stencil_builder=local_stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
        density_face_bc=face_bc,
        potential_face_bc=face_bc,
        vorticity_face_bc=face_bc,
        electron_temperature_face_bc=face_bc,
        ion_temperature_face_bc=ti_face_bc,
        electron_velocity_parallel_face_bc=face_bc,
        ion_velocity_parallel_face_bc=face_bc,
        density_cut_wall_geometry=cut_wall_geometry,
        density_cut_wall_bc=cut_wall_bc,
        potential_cut_wall_geometry=cut_wall_geometry,
        potential_cut_wall_bc=cut_wall_bc,
        vorticity_cut_wall_geometry=cut_wall_geometry,
        vorticity_cut_wall_bc=cut_wall_bc,
        electron_temperature_cut_wall_geometry=cut_wall_geometry,
        electron_temperature_cut_wall_bc=cut_wall_bc,
        ion_temperature_cut_wall_geometry=cut_wall_geometry,
        ion_temperature_cut_wall_bc=cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
        diffusion_only=True,
    )
    ti_conservative_stencil = conservative_stencil_builder(
        state.Ti,
        geometry,
        PERIODIC_AXES,
        ti_face_bc,
    )
    expected_ti_rhs = parameters.ion_temperature_D_perp * perp_laplacian_conservative_op(
        ti_conservative_stencil,
        geometry,
        face_bc=ti_face_bc,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
    )

    np.testing.assert_allclose(np.asarray(result.rhs.Ti), np.asarray(expected_ti_rhs), rtol=1.0e-12, atol=1.0e-12)
    assert np.max(np.abs(np.asarray(result.rhs.Ti))) > 0.0
    np.testing.assert_allclose(np.asarray(result.rhs.Te), 0.0, atol=0.0)


def test_eb_rhs_optional_sources_are_ignored_in_diffusion_only() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    parameters = FciDrbEBRhsParameters()
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    density_source = jnp.full(geometry.shape, 0.125, dtype=jnp.float64)
    electron_temperature_source = jnp.full(geometry.shape, 0.25, dtype=jnp.float64)

    result = compute_fci_drb_eb_rhs(
        state,
        geometry=geometry,
        stencil_builder=local_stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
        density_face_bc=boundary_conditions.density_face_bc,
        potential_face_bc=boundary_conditions.potential_face_bc,
        vorticity_face_bc=boundary_conditions.vorticity_face_bc,
        electron_temperature_face_bc=boundary_conditions.Te_face_bc,
        ion_temperature_face_bc=boundary_conditions.Ti_face_bc,
        electron_velocity_parallel_face_bc=boundary_conditions.Ve_face_bc,
        ion_velocity_parallel_face_bc=boundary_conditions.Vi_face_bc,
        density_cut_wall_geometry=cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        potential_cut_wall_geometry=cut_wall_geometry,
        potential_cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
        vorticity_cut_wall_geometry=cut_wall_geometry,
        vorticity_cut_wall_bc=boundary_conditions.vorticity_cut_wall_bc,
        electron_temperature_cut_wall_geometry=cut_wall_geometry,
        electron_temperature_cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
        ion_temperature_cut_wall_geometry=cut_wall_geometry,
        ion_temperature_cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
        diffusion_only=True,
        density_source=density_source,
        electron_temperature_source=electron_temperature_source,
    )

    for field in (
        result.rhs.density,
        result.rhs.phi,
        result.rhs.Te,
        result.rhs.Ti,
        result.rhs.Vi,
        result.rhs.Ve,
        result.rhs.vorticity,
    ):
        np.testing.assert_allclose(np.asarray(field), 0.0, atol=0.0)


def test_eb_rhs_diffusion_only_density_perp_matches_direct_laplacian() -> None:
    geometry = _build_eb_blob_geometry((8, 16, 8), construct_fci_maps=False)
    state = _build_eb_blob_initial_state(geometry)
    boundary_conditions = _test_eb_rhs_boundary_conditions(geometry, state)
    parameters = FciDrbEBRhsParameters(density_D_perp=1.0e-4)
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()

    result = compute_fci_drb_eb_rhs(
        state,
        geometry=geometry,
        stencil_builder=local_stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=jnp.zeros(geometry.shape + (3,), dtype=jnp.float64),
        density_face_bc=boundary_conditions.density_face_bc,
        potential_face_bc=boundary_conditions.potential_face_bc,
        vorticity_face_bc=boundary_conditions.vorticity_face_bc,
        electron_temperature_face_bc=boundary_conditions.Te_face_bc,
        ion_temperature_face_bc=boundary_conditions.Ti_face_bc,
        electron_velocity_parallel_face_bc=boundary_conditions.Ve_face_bc,
        ion_velocity_parallel_face_bc=boundary_conditions.Vi_face_bc,
        density_cut_wall_geometry=cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        potential_cut_wall_geometry=cut_wall_geometry,
        potential_cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
        vorticity_cut_wall_geometry=cut_wall_geometry,
        vorticity_cut_wall_bc=boundary_conditions.vorticity_cut_wall_bc,
        electron_temperature_cut_wall_geometry=cut_wall_geometry,
        electron_temperature_cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
        ion_temperature_cut_wall_geometry=cut_wall_geometry,
        ion_temperature_cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
        diffusion_only=True,
    )
    density_conservative_stencil = conservative_stencil_builder(
        state.density,
        geometry,
        PERIODIC_AXES,
        boundary_conditions.density_face_bc,
    )
    expected_density_rhs = parameters.density_D_perp * perp_laplacian_conservative_op(
        density_conservative_stencil,
        geometry,
        face_bc=boundary_conditions.density_face_bc,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
    )

    np.testing.assert_allclose(np.asarray(result.rhs.density), np.asarray(expected_density_rhs))
    assert np.max(np.abs(np.asarray(result.rhs.density))) > 0.0
    for field in (result.rhs.phi, result.rhs.Te, result.rhs.Ti, result.rhs.Vi, result.rhs.Ve, result.rhs.vorticity):
        np.testing.assert_allclose(np.asarray(field), 0.0, atol=0.0)


def _add_state(state: FciDrbEBState, delta: FciDrbEBState, *, scale: float) -> FciDrbEBState:
    return state.axpy(delta, scale=scale)


def _check_nonnegative_density_temperature(state: FciDrbEBState, *, label: str) -> None:
    density_min = float(jnp.min(jnp.asarray(state.density, dtype=jnp.float64)))
    te_min = float(jnp.min(jnp.asarray(state.Te, dtype=jnp.float64)))
    ti_min = float(jnp.min(jnp.asarray(state.Ti, dtype=jnp.float64)))
    if density_min < 0.0 or te_min < 0.0 or ti_min < 0.0:
        raise ValueError(
            f"{label} produced a negative state: "
            f"density_min={density_min:.6e}, Te_min={te_min:.6e}, Ti_min={ti_min:.6e}"
        )


def _outer_x_dirichlet_phi_lift(geometry: FciGeometry3D, face_bc: BoundaryFaceBC3D) -> jnp.ndarray:
    wall_value = jnp.asarray(face_bc.value_x[-1], dtype=jnp.float64)
    return jnp.broadcast_to(wall_value[None, :, :], geometry.shape)


def _reconstruct_phi_from_state(
    state: FciDrbEBState,
    *,
    geometry: FciGeometry3D,
    parameters,
    boundary_condition_builder: BoundaryConditionBuilder[FciDrbEBBoundaryConditions],
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    phi_inverse_solver: PerpLaplacianInverseSolver,
    periodic_axes: tuple[bool, bool, bool],
    cut_wall_geometry: CutWallGeometry3D,
    cut_wall_bc: CutWallBC3D,
    phi_guess: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, object]]:
    boundary_conditions = boundary_condition_builder(
        state,
        geometry,
        periodic_axes,
        cut_wall_geometry,
        cut_wall_bc,
    )
    ti_conservative_stencil = conservative_stencil_builder(
        state.Ti,
        geometry,
        periodic_axes,
        boundary_conditions.Ti_face_bc,
    )
    ti_laplacian = perp_laplacian_conservative_op(
        ti_conservative_stencil,
        geometry,
        face_bc=boundary_conditions.Ti_face_bc,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
        periodic_axes=periodic_axes,
    )
    phi_rhs = -jnp.asarray(state.vorticity, dtype=jnp.float64) + jnp.asarray(parameters.tau, dtype=jnp.float64) * ti_laplacian
    phi, diagnostics = phi_inverse_solver(
        phi_rhs,
        phi_guess=phi_guess,
        face_bc=boundary_conditions.potential_face_bc,
        cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
        phi_lift=_outer_x_dirichlet_phi_lift(geometry, boundary_conditions.potential_face_bc),
        return_diagnostics=True,
        throw=False,
    )
    jax.block_until_ready(phi)
    if return_diagnostics:
        return phi, diagnostics
    return phi


def shifted_torus_eb_blob_rk4(
    state: FciDrbEBState,
    *,
    geometry: FciGeometry3D,
    timestep: float,
    parameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_condition_builder: BoundaryConditionBuilder[FciDrbEBBoundaryConditions],
    periodic_axes: tuple[bool, bool, bool],
    cut_wall_geometry: CutWallGeometry3D,
    cut_wall_bc: CutWallBC3D,
    phi_inverse_solver: PerpLaplacianInverseSolver,
    phi_guess: jnp.ndarray | None = None,
    density_source: jax.Array | None = None,
    electron_temperature_source: jax.Array | None = None,
    diffusion_only: bool = False,
) -> tuple[FciDrbEBState, jnp.ndarray, jnp.ndarray]:
    current_phi_guess = jnp.asarray(state.phi if phi_guess is None else phi_guess, dtype=jnp.float64)
    step_start = time_module.perf_counter()

    def _rhs_fn(
        current_state: FciDrbEBState,
        stage_time: float | jax.Array,
        carry: jnp.ndarray,
    ) -> tuple[FciDrbEBState, jnp.ndarray, dict[str, object]]:
        stage_time_value = float(stage_time)
        phi_start = time_module.perf_counter()
        phi, diagnostics = _reconstruct_phi_from_state(
            current_state,
            geometry=geometry,
            parameters=parameters,
            boundary_condition_builder=boundary_condition_builder,
            stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            phi_inverse_solver=phi_inverse_solver,
            periodic_axes=periodic_axes,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            phi_guess=carry,
            return_diagnostics=True,
        )
        phi_time = time_module.perf_counter() - phi_start
        stage_state = replace(current_state, phi=phi)
        _check_nonnegative_density_temperature(stage_state, label=f"RK4 stage at t={stage_time_value:.6e}")

        rhs_start = time_module.perf_counter()
        boundary_conditions = boundary_condition_builder(
            stage_state,
            geometry,
            periodic_axes,
            cut_wall_geometry,
            cut_wall_bc,
        )
        rhs_result = compute_fci_drb_eb_rhs(
            stage_state,
            geometry=geometry,
            stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            density_face_bc=boundary_conditions.density_face_bc,
            potential_face_bc=boundary_conditions.potential_face_bc,
            vorticity_face_bc=boundary_conditions.vorticity_face_bc,
            electron_temperature_face_bc=boundary_conditions.Te_face_bc,
            ion_temperature_face_bc=boundary_conditions.Ti_face_bc,
            electron_velocity_parallel_face_bc=boundary_conditions.Ve_face_bc,
            ion_velocity_parallel_face_bc=boundary_conditions.Vi_face_bc,
            density_cut_wall_geometry=cut_wall_geometry,
            density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
            potential_cut_wall_geometry=cut_wall_geometry,
            potential_cut_wall_bc=boundary_conditions.potential_cut_wall_bc,
            vorticity_cut_wall_geometry=cut_wall_geometry,
            vorticity_cut_wall_bc=boundary_conditions.vorticity_cut_wall_bc,
            electron_temperature_cut_wall_geometry=cut_wall_geometry,
            electron_temperature_cut_wall_bc=boundary_conditions.Te_cut_wall_bc,
            ion_temperature_cut_wall_geometry=cut_wall_geometry,
            ion_temperature_cut_wall_bc=boundary_conditions.Ti_cut_wall_bc,
            electron_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
            electron_velocity_parallel_cut_wall_bc=boundary_conditions.Ve_cut_wall_bc,
            ion_velocity_parallel_cut_wall_geometry=cut_wall_geometry,
            ion_velocity_parallel_cut_wall_bc=boundary_conditions.Vi_cut_wall_bc,
            periodic_axes=periodic_axes,
            diffusion_only=diffusion_only,
            density_source=density_source,
            electron_temperature_source=electron_temperature_source,
        )
        jax.block_until_ready(rhs_result.rhs.density)
        rhs_time = time_module.perf_counter() - rhs_start
        stage_stats = {
            "phi_time": phi_time,
            "rhs_time": rhs_time,
            "diagnostics": diagnostics,
        }
        return rhs_result.rhs, phi, stage_stats

    step_result = rk4_step(state, time=time, timestep=timestep, rhs_fn=_rhs_fn, carry=current_phi_guess)
    next_state = step_result.state
    phi4_start = time_module.perf_counter()
    phi_4, diagnostics_4 = _reconstruct_phi_from_state(
        next_state,
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_condition_builder,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=periodic_axes,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        phi_guess=step_result.carry,
        return_diagnostics=True,
    )
    final_phi_time = time_module.perf_counter() - phi4_start
    next_state = replace(next_state, phi=phi_4)
    _check_nonnegative_density_temperature(next_state, label="RK4 stage 4")
    jax.block_until_ready(next_state.density)

    stage_stats = step_result.stage_aux
    phi_time_total = sum(float(stage["phi_time"]) for stage in stage_stats)
    phi_time_total += float(final_phi_time)
    rhs_time_total = sum(float(stage["rhs_time"]) for stage in stage_stats)
    step_gmres_steps = sum(_diagnostic_float(stage["diagnostics"], "num_steps") for stage in stage_stats) + _diagnostic_float(
        diagnostics_4,
        "num_steps",
    )
    step_gmres_rel_res = max(
        *(_diagnostic_float(stage["diagnostics"], "final_residual_rel_l2") for stage in stage_stats),
        _diagnostic_float(diagnostics_4, "final_residual_rel_l2"),
    )
    step_phi_correction_residual_l2 = max(
        *(_diagnostic_float(stage["diagnostics"], "correction_residual_l2", "final_residual_l2") for stage in stage_stats),
        _diagnostic_float(diagnostics_4, "correction_residual_l2", "final_residual_l2"),
    )
    step_phi_correction_residual_linf = max(
        *(_diagnostic_float(stage["diagnostics"], "correction_residual_linf", "final_residual_linf") for stage in stage_stats),
        _diagnostic_float(diagnostics_4, "correction_residual_linf", "final_residual_linf"),
    )
    step_phi_physical_residual_l2 = max(
        *(_diagnostic_float(stage["diagnostics"], "physical_residual_l2", "final_residual_l2") for stage in stage_stats),
        _diagnostic_float(diagnostics_4, "physical_residual_l2", "final_residual_l2"),
    )
    step_phi_physical_residual_linf = max(
        *(_diagnostic_float(stage["diagnostics"], "physical_residual_linf", "final_residual_linf") for stage in stage_stats),
        _diagnostic_float(diagnostics_4, "physical_residual_linf", "final_residual_linf"),
    )
    rk4_total_time = time_module.perf_counter() - step_start
    return next_state, jnp.asarray(
        [
            phi_time_total,
            rhs_time_total,
            rk4_total_time,
            step_gmres_steps,
            step_gmres_rel_res,
            step_phi_correction_residual_l2,
            step_phi_correction_residual_linf,
            step_phi_physical_residual_l2,
            step_phi_physical_residual_linf,
        ],
        dtype=jnp.float64,
    ), phi_4


def simulate_shifted_torus_eb_blob(
    geometry: FciGeometry3D,
    initial_state: FciDrbEBState,
    boundary_condition_builder: BoundaryConditionBuilder[FciDrbEBBoundaryConditions],
    *,
    parameters,
    num_steps: int = DEFAULT_NUM_STEPS,
    timestep: float | None = None,
    timesteps: Sequence[float] | None = None,
    final_time: float = tf,
    show_progress: bool = False,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    step_output_dir: Path | None = None,
    diffusion_only: bool = False,
    density_source: jax.Array | None = None,
    electron_temperature_source: jax.Array | None = None,
) -> tuple[
    FciDrbEBState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    if phi_inverse_solver is None:
        face_projectors = build_perp_laplacian_face_projectors(
            geometry,
            axis_regular_axes=AXIS_REGULAR_AXES,
        )
        phi_inverse_solver = PerpLaplacianInverseSolver(
            geometry,
            conservative_stencil_builder,
            tol=5.0e-5,
            maxiter=100,
            restart=200,
            face_projectors=face_projectors,
            regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            periodic_axes=PERIODIC_AXES,
            axis_regular_axes=AXIS_REGULAR_AXES,
            pin_point=None,
            pin_value=0.0,
            project_mean_zero=False,
            target_mean_phi=None,
            regularization_epsilon=0.0,
            mg_hierarchy=None,
            gmres_debug=False,
        )
    if timesteps is None:
        step_sizes = _build_eb_blob_timesteps(
            final_time=float(final_time),
            num_steps=int(num_steps),
            timestep=timestep,
        )
    else:
        step_sizes = tuple(float(step_size) for step_size in timesteps)
        if not step_sizes:
            raise ValueError("timesteps must contain at least one step")
        if any(step_size <= 0.0 for step_size in step_sizes):
            raise ValueError("all timesteps must be positive")
        if not np.isclose(float(np.sum(step_sizes)), float(final_time)):
            raise ValueError(
                f"timesteps sum to {float(np.sum(step_sizes)):.16e}, expected final_time={float(final_time):.16e}"
            )
    steps = int(len(step_sizes))
    curvature_coefficients = build_curvature_coefficients(
        geometry,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )

    state = initial_state
    if show_progress:
        print("EB blob entering initial phi reconstruction", flush=True)
    state = replace(
        state,
        phi=_reconstruct_phi_from_state(
            state,
            geometry=geometry,
            parameters=parameters,
            boundary_condition_builder=boundary_condition_builder,
            stencil_builder=local_stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            phi_inverse_solver=phi_inverse_solver,
            periodic_axes=PERIODIC_AXES,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
        ),
    )
    if show_progress:
        print("EB blob initial phi reconstruction complete", flush=True)
    if step_output_dir is not None:
        _save_eb_blob_step_snapshot(step_output_dir, 0, 0.0, state)
    time_value = 0.0
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(state.density, dtype=jnp.float64)]
    phi_history: list[jnp.ndarray] = [jnp.asarray(state.phi, dtype=jnp.float64)]
    te_history: list[jnp.ndarray] = [jnp.asarray(state.Te, dtype=jnp.float64)]
    ti_history: list[jnp.ndarray] = [jnp.asarray(state.Ti, dtype=jnp.float64)]
    vi_history: list[jnp.ndarray] = [jnp.asarray(state.Vi, dtype=jnp.float64)]
    ve_history: list[jnp.ndarray] = [jnp.asarray(state.Ve, dtype=jnp.float64)]
    vorticity_history: list[jnp.ndarray] = [jnp.asarray(state.vorticity, dtype=jnp.float64)]
    timing_history: list[jnp.ndarray] = []
    current_phi_guess = jnp.asarray(state.phi, dtype=jnp.float64)
    progress_start = time_module.perf_counter()
    if show_progress:
        print(
            "EB blob progress: "
            f"{_format_progress_bar(0, steps, start_time=progress_start, time_value=time_value)}",
            end="",
            flush=True,
        )

    for step_index, dt in enumerate(step_sizes):
        if show_progress and step_index == 0:
            print("EB blob RK4 step 1 starting", flush=True)
        state, step_gmres_stats, current_phi_guess = shifted_torus_eb_blob_rk4(
            state,
            geometry=geometry,
            timestep=float(dt),
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            stencil_builder=local_stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            boundary_condition_builder=boundary_condition_builder,
            periodic_axes=PERIODIC_AXES,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            phi_inverse_solver=phi_inverse_solver,
            phi_guess=current_phi_guess,
            density_source=density_source,
            electron_temperature_source=electron_temperature_source,
            diffusion_only=diffusion_only,
        )
        if show_progress and step_index == 0:
            print("EB blob RK4 step 1 complete", flush=True)
        time_value += float(dt)
        times.append(time_value)
        density_history.append(jnp.asarray(state.density, dtype=jnp.float64))
        phi_history.append(jnp.asarray(state.phi, dtype=jnp.float64))
        te_history.append(jnp.asarray(state.Te, dtype=jnp.float64))
        ti_history.append(jnp.asarray(state.Ti, dtype=jnp.float64))
        vi_history.append(jnp.asarray(state.Vi, dtype=jnp.float64))
        ve_history.append(jnp.asarray(state.Ve, dtype=jnp.float64))
        vorticity_history.append(jnp.asarray(state.vorticity, dtype=jnp.float64))
        timing_history.append(step_gmres_stats)
        if show_progress:
            print(
                "\r\033[K"
                "EB blob progress: "
                f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start, time_value=time_value, gmres_steps_per_solve=float(step_gmres_stats[3]) / 5.0, gmres_rel_res=float(step_gmres_stats[4]))}"
                f" rk4={float(step_gmres_stats[2]):.2e}s"
                f" phi={float(step_gmres_stats[0]):.2e}s"
                f" rhs={float(step_gmres_stats[1]):.2e}s",
                end="",
                flush=True,
            )
        if step_output_dir is not None:
            _save_eb_blob_step_snapshot(
                step_output_dir,
                step_index + 1,
                time_value,
                state,
                step_gmres_stats=step_gmres_stats,
            )
    if show_progress:
        print()
    if timing_history:
        timing_array = np.asarray(timing_history, dtype=np.float64)
        print(
            "EB blob mean timings per RK step: "
            f"phi_inverse={float(np.mean(timing_array[:, 0])):.6e} s, "
            f"rhs_compute={float(np.mean(timing_array[:, 1])):.6e} s, "
            f"rk4_total={float(np.mean(timing_array[:, 2])):.6e} s, "
            f"phi_gmres_steps_per_rk={float(np.mean(timing_array[:, 3])):.2f}, "
            f"phi_gmres_steps_per_solve={float(np.mean(timing_array[:, 3]) / 5.0):.2f}, "
            f"phi_gmres_rel_res={float(np.mean(timing_array[:, 4])):.2e}"
        )
    return (
        state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(phi_history, axis=0),
        jnp.stack(te_history, axis=0),
        jnp.stack(ti_history, axis=0),
        jnp.stack(vi_history, axis=0),
        jnp.stack(ve_history, axis=0),
        jnp.stack(vorticity_history, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shifted-torus EB blob simulation and save plots.")
    parser.add_argument(
        "--run-name",
        default="eb_blob",
        help="Prefix used for saved history, plot, and movie files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Cubic grid resolution to use for all three dimensions.",
    )
    parser.add_argument(
        "--perp-diffusion",
        type=float,
        default=DEFAULT_PERP_DIFFUSION,
        help="Perpendicular diffusion coefficient applied to all diffusive EB blob fields.",
    )
    parser.add_argument(
        "--radial-b-fraction",
        type=float,
        default=radial_b_fraction,
        help="Fraction of the background B^z used to set the radial B^x perturbation amplitude.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of uniform RK4 timesteps to take over the fixed final time when split-step options are absent.",
    )
    parser.add_argument(
        "--initial-transient-time",
        type=float,
        default=DEFAULT_INITIAL_TRANSIENT_TIME,
        help="End time for the initial small-step window when using split-step options.",
    )
    parser.add_argument(
        "--initial-num-steps",
        type=int,
        default=None,
        help="Number of RK4 timesteps to use from t=0 to --initial-transient-time.",
    )
    parser.add_argument(
        "--remaining-num-steps",
        type=int,
        default=None,
        help="Number of RK4 timesteps to use from --initial-transient-time to tf.",
    )
    parser.add_argument(
        "--initial-velocity-state",
        choices=("zero", "sheath_taper"),
        default="zero",
        help="Initial Ve/Vi profile: zero everywhere or a tapered sheath-matched profile.",
    )
    parser.add_argument(
        "--source-x0",
        type=float,
        default=SOURCE_X0,
        help="Radial center of the Gaussian density and Te sources.",
    )
    parser.add_argument(
        "--source-delta-x",
        type=float,
        default=SOURCE_DELTA_X,
        help="Radial width of the Gaussian density and Te sources.",
    )
    parser.add_argument(
        "--source-amplitude",
        type=float,
        default=DENSITY_SOURCE_AMPLITUDE,
        help="Amplitude applied to both the density and Te Gaussian sources.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Directory for all EB blob outputs. Defaults to <run_name>_outputs.",
    )
    parser.add_argument(
        "--diffusion-only",
        action="store_true",
        help="Use only explicit diffusion terms in the EB RHS for sanity checks.",
    )
    args = parser.parse_args()

    resolution = int(args.resolution)
    perp_diffusion = float(args.perp_diffusion)
    radial_b_fraction_value = float(args.radial_b_fraction)
    requested_num_steps = int(args.num_steps)
    initial_transient_time_value = float(args.initial_transient_time)
    initial_num_steps = None if args.initial_num_steps is None else int(args.initial_num_steps)
    remaining_num_steps = None if args.remaining_num_steps is None else int(args.remaining_num_steps)
    initial_velocity_state = str(args.initial_velocity_state)
    split_timesteps = initial_num_steps is not None or remaining_num_steps is not None
    if split_timesteps and (initial_num_steps is None or remaining_num_steps is None):
        parser.error("--initial-num-steps and --remaining-num-steps must be supplied together")
    timestep_schedule = _build_eb_blob_timesteps(
        final_time=tf,
        num_steps=requested_num_steps,
        initial_transient_time=initial_transient_time_value,
        initial_num_steps=initial_num_steps,
        remaining_num_steps=remaining_num_steps,
    )
    num_steps = int(len(timestep_schedule))
    source_x0_value = float(args.source_x0)
    source_delta_x_value = float(args.source_delta_x)
    source_amplitude_value = float(args.source_amplitude)
    diffusion_only = bool(args.diffusion_only)
    print(f"EB blob settings: resolution={resolution}, perp_diffusion={perp_diffusion:.6e}")
    print(f"EB blob settings: radial_b_fraction={radial_b_fraction_value:.6e}")
    print(f"EB blob settings: num_steps={num_steps}")
    print(f"EB blob settings: initial_velocity_state={initial_velocity_state}")
    if split_timesteps:
        print(
            "EB blob settings: split timesteps="
            f"{initial_num_steps} over [0, {initial_transient_time_value:.6e}] and "
            f"{remaining_num_steps} over [{initial_transient_time_value:.6e}, {tf:.6e}]"
        )
    else:
        print(f"EB blob settings: uniform dt={float(timestep_schedule[0]):.6e}")
    print(f"EB blob settings: diffusion_only={diffusion_only}")
    print(
        "EB blob settings: source_profile="
        f"{SOURCE_PROFILE}, x0={source_x0_value:.6e}, delta_x={source_delta_x_value:.6e}, "
        f"source_A={source_amplitude_value:.6e}"
    )
    _print_eb_blob_timestep_schedule(
        timestep_schedule,
        initial_num_steps=initial_num_steps,
        initial_transient_time=initial_transient_time_value if split_timesteps else None,
        final_time=tf,
    )

    geometry = _build_eb_blob_geometry((resolution, resolution, resolution), radial_fraction=radial_b_fraction_value)
    density_source, electron_temperature_source = _build_eb_blob_source_terms(
        geometry,
        amplitude=source_amplitude_value,
        x0=source_x0_value,
        delta_x=source_delta_x_value,
    )
    coordinate_face_reconstructor = CoordinateFaceValueReconstructor3D()
    coordinate_normal_derivative_constructor = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
    parameters = _build_eb_blob_parameters(perp_diffusion)
    eb_bc_builder = BoundaryConditionBuilder(
        partial(
            _build_eb_boundary_conditions,
            face_reconstructor=coordinate_face_reconstructor,
            normal_derivative_constructor=coordinate_normal_derivative_constructor,
        )
    )
    face_projectors = build_perp_laplacian_face_projectors(
        geometry,
        axis_regular_axes=AXIS_REGULAR_AXES,
    )
    cut_wall_geometry = CutWallGeometry3D.empty()
    cut_wall_bc = CutWallBC3D.empty()
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    artifact_stem = _eb_blob_artifact_stem(args.run_name)
    output_dir = args.output_path if args.output_path is not None else Path(f"{artifact_stem}_outputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = _resolve_eb_blob_history_path(args.run_name, output_dir)
    step_output_dir = output_dir / "step_dumps"
    initial_state = _build_eb_blob_initial_state(geometry, velocity_initialization=initial_velocity_state)

    boundary_conditions = eb_bc_builder(
        initial_state,
        geometry,
        PERIODIC_AXES,
        cut_wall_geometry,
        cut_wall_bc,
    )

    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=5.0e-5,
        maxiter=500,
        restart=500,
        face_projectors=face_projectors,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
        axis_regular_axes=AXIS_REGULAR_AXES,
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        mg_hierarchy=None,
        gmres_debug=False,
        check_residual=True,
    )

    run_simulation = True
    if history_path.exists():
        (
            times,
            density_history,
            phi_history,
            te_history,
            ti_history,
            vi_history,
            ve_history,
            vorticity_history,
            metadata,
        ) = _load_eb_blob_history(history_path)
        if _history_matches_eb_blob_settings(
            metadata,
            resolution=resolution,
            num_steps=num_steps,
            initial_velocity_state=initial_velocity_state,
            initial_transient_time=initial_transient_time_value if split_timesteps else None,
            initial_num_steps=initial_num_steps,
            remaining_num_steps=remaining_num_steps,
            a_n=A_N,
            l_rho_cells=Lrho_cells,
            l_y_cells=Ly_cells,
            radial_b_fraction=radial_b_fraction_value,
            perp_diffusion=perp_diffusion,
            diffusion_only=diffusion_only,
            source_profile=SOURCE_PROFILE,
            source_x0=source_x0_value,
            source_delta_x=source_delta_x_value,
            source_amplitude=source_amplitude_value,
        ):
            run_simulation = False
        else:
            print(f"EB blob history settings mismatch for {history_path}; rerunning")

    if run_simulation:
        _clear_eb_blob_step_dumps(step_output_dir)
        print("EB blob initial phi solve complete; starting RK4 loop")
        state, times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history = simulate_shifted_torus_eb_blob(
            geometry,
            initial_state,
            eb_bc_builder,
            final_time=tf,
            num_steps=num_steps,
            timesteps=timestep_schedule,
            show_progress=True,
            parameters=parameters,
            phi_inverse_solver=phi_inverse_solver,
            step_output_dir=step_output_dir,
            density_source=density_source,
            electron_temperature_source=electron_temperature_source,
            diffusion_only=diffusion_only,
        )
        jax.block_until_ready(state.density)
        np.savez(
            history_path,
            times=np.asarray(times, dtype=np.float64),
            density=np.asarray(density_history, dtype=np.float64),
            phi=np.asarray(phi_history, dtype=np.float64),
            Te=np.asarray(te_history, dtype=np.float64),
            Ti=np.asarray(ti_history, dtype=np.float64),
            Vi=np.asarray(vi_history, dtype=np.float64),
            Ve=np.asarray(ve_history, dtype=np.float64),
            vorticity=np.asarray(vorticity_history, dtype=np.float64),
            resolution=np.asarray(resolution, dtype=np.int64),
            num_steps=np.asarray(num_steps, dtype=np.int64),
            initial_velocity_state=np.asarray(initial_velocity_state),
            initial_velocity_alpha=np.asarray(DEFAULT_INITIAL_VELOCITY_ALPHA, dtype=np.float64),
            initial_velocity_ell_fraction=np.asarray(DEFAULT_INITIAL_VELOCITY_ELL_FRACTION, dtype=np.float64),
            initial_velocity_formula=np.asarray(INITIAL_VELOCITY_FORMULA),
            wall_sign_smoothing_formula=np.asarray(WALL_SIGN_SMOOTHING_FORMULA),
            wall_sign_smoothing_enabled=np.asarray(WALL_SIGN_SMOOTHING_ENABLED, dtype=bool),
            wall_sign_smoothing_width_cells=np.asarray(WALL_SIGN_SMOOTHING_WIDTH_CELLS, dtype=np.float64),
            initial_transient_time=np.asarray(initial_transient_time_value if split_timesteps else 0.0, dtype=np.float64),
            initial_num_steps=np.asarray(initial_num_steps if split_timesteps else 0, dtype=np.int64),
            remaining_num_steps=np.asarray(remaining_num_steps if split_timesteps else num_steps, dtype=np.int64),
            A_N=np.asarray(A_N, dtype=np.float64),
            rho0=np.asarray(rho0, dtype=np.float64),
            y0=np.asarray(y0, dtype=np.float64),
            z0=np.asarray(z0, dtype=np.float64),
            Lrho_cells=np.asarray(Lrho_cells, dtype=np.float64),
            Ly_cells=np.asarray(Ly_cells, dtype=np.float64),
            radial_b_fraction=np.asarray(radial_b_fraction_value, dtype=np.float64),
            n0=np.asarray(parameters.n0, dtype=np.float64),
            Te0=np.asarray(parameters.Te0, dtype=np.float64),
            Ti0=np.asarray(parameters.Ti0, dtype=np.float64),
            cs_0=np.asarray(parameters.cs_0, dtype=np.float64),
            rhos_s0=np.asarray(parameters.rhos_s0, dtype=np.float64),
            tau=np.asarray(parameters.tau, dtype=np.float64),
            mi_over_me=np.asarray(parameters.mi_over_me, dtype=np.float64),
            rho_star=np.asarray(parameters.rho_star, dtype=np.float64),
            density_D_perp=np.asarray(parameters.density_D_perp, dtype=np.float64),
            density_D_parallel=np.asarray(parameters.density_D_parallel, dtype=np.float64),
            electron_temperature_chi_parallel=np.asarray(parameters.electron_temperature_chi_parallel, dtype=np.float64),
            electron_temperature_D_perp=np.asarray(parameters.electron_temperature_D_perp, dtype=np.float64),
            ion_temperature_chi_parallel=np.asarray(parameters.ion_temperature_chi_parallel, dtype=np.float64),
            ion_temperature_D_perp=np.asarray(parameters.ion_temperature_D_perp, dtype=np.float64),
            Ve_nu=np.asarray(parameters.Ve_nu, dtype=np.float64),
            Ve_D_perp=np.asarray(parameters.Ve_D_perp, dtype=np.float64),
            Ve_D_parallel=np.asarray(parameters.Ve_D_parallel, dtype=np.float64),
            Vi_D_perp=np.asarray(parameters.Vi_D_perp, dtype=np.float64),
            Vi_D_parallel=np.asarray(parameters.Vi_D_parallel, dtype=np.float64),
            vorticity_D_perp=np.asarray(parameters.vorticity_D_perp, dtype=np.float64),
            vorticity_D_parallel=np.asarray(parameters.vorticity_D_parallel, dtype=np.float64),
            perp_diffusion=np.asarray(perp_diffusion, dtype=np.float64),
            diffusion_only=np.asarray(diffusion_only, dtype=bool),
            curvature_axis_regular_lower_x=np.asarray(True, dtype=bool),
            source_profile=np.asarray(SOURCE_PROFILE),
            source_x0=np.asarray(source_x0_value, dtype=np.float64),
            source_delta_x=np.asarray(source_delta_x_value, dtype=np.float64),
            source_amplitude=np.asarray(source_amplitude_value, dtype=np.float64),
            density_source_amplitude=np.asarray(source_amplitude_value, dtype=np.float64),
            electron_temperature_source_amplitude=np.asarray(source_amplitude_value, dtype=np.float64),
        )
    else:
        state = initial_state

    time_traces_path = output_dir / f"{artifact_stem}_time_traces.png"
    movie_path = output_dir / f"{artifact_stem}.gif"
    _save_eb_blob_time_traces(
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        geometry,
        output_path=str(time_traces_path),
        title="Shifted-torus EB blob time traces",
    )
    _save_eb_blob_movie(
        times,
        density_history,
        phi_history,
        te_history,
        ti_history,
        vi_history,
        ve_history,
        vorticity_history,
        geometry,
        output_path=str(movie_path),
        frame_stride=2,
        title="Shifted-torus EB blob state evolution",
        z_indices=_eb_blob_z_indices(geometry, z0),
    )

    print(f"EB blob geometry shape: {geometry.shape}")
    print(f"EB blob resolution: {resolution}")
    print(f"EB blob perpendicular diffusion: {perp_diffusion:.6e}")
    print(f"EB blob diffusion-only RHS: {diffusion_only}")
    print(f"EB blob run name: {artifact_stem}")
    print(f"EB blob history path: {history_path}")
    print(f"EB blob step dump dir: {step_output_dir}")
    print(f"EB blob time traces path: {time_traces_path}")
    print(f"EB blob movie path: {movie_path}")
    print(f"EB blob local stencil builder: {local_stencil_builder}")
    print(f"EB blob conservative stencil builder: {conservative_stencil_builder}")
    print(f"EB blob BC builder: {eb_bc_builder}")
    print(f"EB blob parameters: {parameters}")
    print(f"EB blob density face bc x-kind shape: {boundary_conditions.density_face_bc.kind_x.shape}")
    print(f"EB blob potential face bc x-kind shape: {boundary_conditions.potential_face_bc.kind_x.shape}")
    print(f"EB blob cut wall bc size: {boundary_conditions.density_cut_wall_bc.kind.size}")
    print(f"EB blob coordinate face reconstructor: {coordinate_face_reconstructor}")
    print(f"EB blob coordinate normal derivative dnormal weights: {coordinate_normal_derivative_constructor.dnormal_weights.shape}")
    print(f"EB blob face projector count: {len(face_projectors)}")
    print(f"EB blob cut wall face count: {cut_wall_geometry.n_wall_faces}")
    print(f"EB blob cut wall bc size: {cut_wall_bc.kind.size}")
    print(f"EB blob regular face geometry x-area shape: {regular_face_geometry.x_area.shape}")


if __name__ == "__main__":
    main()
