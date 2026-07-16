"""MMS harness for the shifted-torus electrostatic Boussinesq DRB model.

The manufactured source is built from continuum analytic field derivatives,
not by subtracting the discrete EB RHS. The RK4 harness still advances the
discrete EB RHS plus this analytic source, so the observed error measures the
spatial truncation mismatch instead of cancelling to roundoff.

For the first analytic EB MMS pass all explicit diffusion coefficients are set
to zero. This keeps the source terms focused on the advective, parallel,
curvature, pressure, current, and collision pieces without requiring fourth
derivatives of the manufactured vorticity.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass
from pathlib import Path
import sys

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
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import (
    FciDrbEBBoundaryConditions,
    FciDrbEBRhsParameters,
    FciDrbEBState,
    compute_fci_drb_eb_rhs,
    perp_laplacian_conservative_op,
    rk4_step,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from jax_drb.native.fci_operators import build_perp_laplacian_face_projectors

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test_mms_shifted_torus_4_field import (  # noqa: E402
    _format_progress_bar,
    alpha_value,
    build_shifted_torus_4field_geometry,
    c_phi,
    iota,
    r0,
    sigma,
    x_max,
    x_min,
)


PERIODIC_AXES = (False, True, True)
N0 = 1.0
TE0 = 1.0
TI0 = 1.0
EPS_N = 1.0e-2
EPS_TE = 1.0e-2
EPS_TI = 1.0e-2
EPS_PHI = 1.0e-3
EPS_VE = 1.0e-3
EPS_VI = 1.0e-3
W_N = 0.7
W_TE = 0.9
W_TI = 1.1
W_PHI = 0.5
W_VE = 0.8
W_VI = 1.2
TAU = 1.0
RHO_STAR = 1.0
MI_OVER_ME = 100.0
TF = 0.02
NUM_STEPS = 20


@dataclass(frozen=True)
class ShiftedTorusEbMmsContext:
    geometry: FciGeometry3D
    parameters: FciDrbEBRhsParameters
    boundary_conditions: FciDrbEBBoundaryConditions
    curvature_coefficients: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    stencil_builder: LocalStencilBuilder
    conservative_stencil_builder: ConservativeStencilBuilder
    cut_wall_geometry: CutWallGeometry3D
    cut_wall_bc: CutWallBC3D


@dataclass(frozen=True)
class _ScalarEbGeometry:
    J: jnp.ndarray
    g_contra: jnp.ndarray
    g_cov: jnp.ndarray
    B_contra: jnp.ndarray
    Bmag: jnp.ndarray
    b_contra: jnp.ndarray
    b_cov: jnp.ndarray


@dataclass(frozen=True)
class _AnalyticMmsData:
    density: jnp.ndarray
    phi: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray
    Vi: jnp.ndarray
    Ve: jnp.ndarray
    vorticity: jnp.ndarray
    density_t: jnp.ndarray
    phi_t: jnp.ndarray
    Te_t: jnp.ndarray
    Ti_t: jnp.ndarray
    Vi_t: jnp.ndarray
    Ve_t: jnp.ndarray
    vorticity_t: jnp.ndarray
    density_grad: jnp.ndarray
    phi_grad: jnp.ndarray
    Te_grad: jnp.ndarray
    Ti_grad: jnp.ndarray
    Vi_grad: jnp.ndarray
    Ve_grad: jnp.ndarray
    vorticity_grad: jnp.ndarray


def _bmag_from_contravariant_components(B_contra: jnp.ndarray, g_cov: jnp.ndarray) -> jnp.ndarray:
    bmag_sq = jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra)
    return jnp.sqrt(jnp.maximum(bmag_sq, 0.0))


def build_shifted_torus_eb_mms_geometry(shape: tuple[int, int, int]) -> FciGeometry3D:
    """Build the shifted-torus MMS geometry with no artificial radial B."""

    base_geometry = build_shifted_torus_4field_geometry(shape, construct_fci_maps=False)
    grid = base_geometry.grid

    def _zero_radial_bfield(bfield: BFieldGeometry, metric) -> BFieldGeometry:
        B_contra = jnp.asarray(bfield.B_contra, dtype=jnp.float64)
        zero_radial = jnp.zeros_like(B_contra[..., 0], dtype=jnp.float64)
        adjusted = jnp.stack((zero_radial, B_contra[..., 1], B_contra[..., 2]), axis=-1)
        return BFieldGeometry(
            B_contra=adjusted,
            Bmag=_bmag_from_contravariant_components(adjusted, metric.g_cov),
        )

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
        cell_bfield=_zero_radial_bfield(base_geometry.cell_bfield, base_geometry.cell_metric),
        face_bfield=FaceBFieldGeometry(
            x=_zero_radial_bfield(base_geometry.face_bfield.x, base_geometry.face_metric.x),
            y=_zero_radial_bfield(base_geometry.face_bfield.y, base_geometry.face_metric.y),
            z=_zero_radial_bfield(base_geometry.face_bfield.z, base_geometry.face_metric.z),
        ),
    )


def _build_x_boundary_face_bc(
    geometry: FciGeometry3D,
    *,
    kind: int,
    value: float = 0.0,
) -> BoundaryFaceBC3D:
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=(
            jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32)
            .at[0]
            .set(int(kind))
            .at[-1]
            .set(int(kind))
        ),
        kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
        value_x=(
            jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64)
            .at[0]
            .set(float(value))
            .at[-1]
            .set(float(value))
        ),
        value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
        mask_x=(
            jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool)
            .at[0]
            .set(True)
            .at[-1]
            .set(True)
        ),
        mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
    )


def _build_mms_boundary_conditions(geometry: FciGeometry3D) -> FciDrbEBBoundaryConditions:
    empty_cut_wall_bc = CutWallBC3D.empty()
    phi_face_bc = _build_x_boundary_face_bc(geometry, kind=BC_DIRICHLET, value=0.0)
    neumann_zero_face_bc = _build_x_boundary_face_bc(geometry, kind=BC_NEUMANN, value=0.0)
    return FciDrbEBBoundaryConditions(
        density_face_bc=neumann_zero_face_bc,
        density_cut_wall_bc=empty_cut_wall_bc,
        potential_face_bc=phi_face_bc,
        potential_cut_wall_bc=empty_cut_wall_bc,
        vorticity_face_bc=neumann_zero_face_bc,
        vorticity_cut_wall_bc=empty_cut_wall_bc,
        Te_face_bc=neumann_zero_face_bc,
        Te_cut_wall_bc=empty_cut_wall_bc,
        Ti_face_bc=neumann_zero_face_bc,
        Ti_cut_wall_bc=empty_cut_wall_bc,
        Vi_face_bc=neumann_zero_face_bc,
        Vi_cut_wall_bc=empty_cut_wall_bc,
        Ve_face_bc=neumann_zero_face_bc,
        Ve_cut_wall_bc=empty_cut_wall_bc,
    )


def _build_mms_parameters() -> FciDrbEBRhsParameters:
    return FciDrbEBRhsParameters(
        n0=N0,
        Te0=TE0,
        Ti0=TI0,
        tau=TAU,
        mi_over_me=MI_OVER_ME,
        rho_star=RHO_STAR,
        density_D_perp=0.0,
        density_D_parallel=0.0,
        electron_temperature_chi_parallel=0.0,
        electron_temperature_D_perp=0.0,
        ion_temperature_chi_parallel=0.0,
        ion_temperature_D_perp=0.0,
        Ve_nu=1.0e-3,
        Ve_D_perp=0.0,
        Ve_D_parallel=0.0,
        Vi_D_perp=0.0,
        Vi_D_parallel=0.0,
        vorticity_D_perp=0.0,
        vorticity_D_parallel=0.0,
    )


def build_shifted_torus_eb_mms_context(shape: tuple[int, int, int]) -> ShiftedTorusEbMmsContext:
    geometry = build_shifted_torus_eb_mms_geometry(shape)
    return ShiftedTorusEbMmsContext(
        geometry=geometry,
        parameters=_build_mms_parameters(),
        boundary_conditions=_build_mms_boundary_conditions(geometry),
        curvature_coefficients=build_curvature_coefficients(geometry, periodic_axes=PERIODIC_AXES),
        face_projectors=build_perp_laplacian_face_projectors(geometry),
        stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
        conservative_stencil_builder=ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
    )


def _mms_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    rho = jnp.asarray(logical_grid[..., 0], dtype=jnp.float64)
    theta = jnp.asarray(logical_grid[..., 1], dtype=jnp.float64)
    zeta = jnp.asarray(logical_grid[..., 2], dtype=jnp.float64)
    rho_min = float(geometry.grid.x.faces[0])
    rho_max = float(geometry.grid.x.faces[-1])
    rho_mid = 0.5 * (rho_min + rho_max)
    theta_s = theta + float(sigma) * (rho - rho_mid)
    s = (rho - rho_min) / (rho_max - rho_min)
    H = jnp.sin(jnp.pi * s) ** 6
    return H, theta_s, theta, zeta


def _mms_radial_bounds(geometry: FciGeometry3D) -> tuple[float, float]:
    return float(geometry.grid.x.faces[0]), float(geometry.grid.x.faces[-1])


def _mms_cell_coordinates(geometry: FciGeometry3D) -> jnp.ndarray:
    return jnp.asarray(
        logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors),
        dtype=jnp.float64,
    )


def _mms_scalar_coordinates(
    coord: jnp.ndarray,
    *,
    rho_min: float,
    rho_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rho, theta, zeta = coord
    rho_mid = 0.5 * (float(rho_min) + float(rho_max))
    theta_s = theta + float(sigma) * (rho - rho_mid)
    s = (rho - float(rho_min)) / (float(rho_max) - float(rho_min))
    H = jnp.sin(jnp.pi * s) ** 6
    return H, theta_s, theta, zeta


def _mms_density_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return N0 + EPS_N * H * (
        0.70 * jnp.cos(2.0 * theta_s + 3.0 * zeta + W_N * time_value)
        + 0.30 * jnp.sin(3.0 * theta_s - 2.0 * zeta + W_N * time_value)
    )


def _mms_Te_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return TE0 + EPS_TE * H * (
        0.60 * jnp.sin(theta_s + 2.0 * zeta + W_TE * time_value)
        + 0.25 * jnp.cos(4.0 * theta_s - zeta + W_TE * time_value)
    )


def _mms_Ti_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return TI0 + EPS_TI * H * (
        0.50 * jnp.cos(3.0 * theta_s - zeta + W_TI * time_value)
        + 0.25 * jnp.sin(2.0 * theta_s + 4.0 * zeta + W_TI * time_value)
    )


def _mms_phi_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return EPS_PHI * H * (
        jnp.cos(2.0 * theta_s) * jnp.sin(3.0 * zeta + W_PHI * time_value)
        + 0.50 * jnp.sin(3.0 * theta_s + 2.0 * zeta + W_PHI * time_value)
        + 0.25 * jnp.cos(4.0 * theta_s - zeta + W_PHI * time_value)
    )


def _mms_Ve_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return EPS_VE * H * (
        jnp.sin(theta_s - 3.0 * zeta + W_VE * time_value)
        + 0.30 * jnp.cos(2.0 * theta_s + zeta + W_VE * time_value)
    )


def _mms_Vi_scalar(coord: jnp.ndarray, time: jnp.ndarray, rho_min: float, rho_max: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_scalar_coordinates(coord, rho_min=rho_min, rho_max=rho_max)
    time_value = jnp.asarray(time, dtype=jnp.float64)
    return EPS_VI * H * (
        jnp.cos(theta_s + 2.0 * zeta + W_VI * time_value)
        + 0.30 * jnp.sin(3.0 * theta_s - zeta + W_VI * time_value)
    )


def _shifted_torus_eb_geometry_scalar(coord: jnp.ndarray) -> _ScalarEbGeometry:
    rho, theta, _ = coord
    rho_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (rho - rho_mid)
    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * rho + rho * cos_shift
    Q = 1.0 + float(alpha_value) * cos_shift
    J = rho * R * Q
    J = jnp.where(jnp.abs(J) < 1.0e-14, 1.0e-14, J)

    g_contra = jnp.array(
        [
            [1.0 / (Q**2), float(alpha_value) * sin_shift / (rho * Q**2), 0.0],
            [
                float(alpha_value) * sin_shift / (rho * Q**2),
                (1.0 + 2.0 * float(alpha_value) * cos_shift + float(alpha_value) ** 2)
                / (rho**2 * Q**2),
                0.0,
            ],
            [0.0, 0.0, 1.0 / (R**2)],
        ],
        dtype=jnp.float64,
    )
    g_cov = jnp.array(
        [
            [1.0 + 2.0 * float(alpha_value) * cos_shift + float(alpha_value) ** 2, -float(alpha_value) * rho * sin_shift, 0.0],
            [-float(alpha_value) * rho * sin_shift, rho**2, 0.0],
            [0.0, 0.0, R**2],
        ],
        dtype=jnp.float64,
    )
    B_contra = jnp.array(
        [0.0, float(iota) * float(c_phi) / J, float(c_phi) / J],
        dtype=jnp.float64,
    )
    Bmag = jnp.sqrt(jnp.maximum(jnp.einsum("i,ij,j->", B_contra, g_cov, B_contra), 0.0))
    b_contra = B_contra / jnp.maximum(Bmag, 1.0e-30)
    b_cov = jnp.einsum("ij,j->i", g_cov, b_contra)
    return _ScalarEbGeometry(
        J=J,
        g_contra=g_contra,
        g_cov=g_cov,
        B_contra=B_contra,
        Bmag=Bmag,
        b_contra=b_contra,
        b_cov=b_cov,
    )


def _perp_laplacian_scalar(
    field_fn,
    coord: jnp.ndarray,
    time: jnp.ndarray,
) -> jnp.ndarray:
    def _field_at(point: jnp.ndarray) -> jnp.ndarray:
        return field_fn(point, time)

    def _flux_vector(point: jnp.ndarray) -> jnp.ndarray:
        scalar_geometry = _shifted_torus_eb_geometry_scalar(point)
        field_grad = jax.grad(_field_at)(point)
        projector = scalar_geometry.g_contra - jnp.outer(scalar_geometry.b_contra, scalar_geometry.b_contra)
        return scalar_geometry.J * jnp.einsum("ij,j->i", projector, field_grad)

    scalar_geometry = _shifted_torus_eb_geometry_scalar(coord)
    flux_jacobian = jax.jacfwd(_flux_vector)(coord)
    return jnp.trace(flux_jacobian) / jnp.maximum(scalar_geometry.J, 1.0e-30)


def _mms_vorticity_scalar(
    coord: jnp.ndarray,
    time: jnp.ndarray,
    rho_min: float,
    rho_max: float,
    tau: float,
) -> jnp.ndarray:
    phi_laplacian = _perp_laplacian_scalar(
        lambda point, time_value: _mms_phi_scalar(point, time_value, rho_min, rho_max),
        coord,
        time,
    )
    Ti_laplacian = _perp_laplacian_scalar(
        lambda point, time_value: _mms_Ti_scalar(point, time_value, rho_min, rho_max),
        coord,
        time,
    )
    return phi_laplacian + float(tau) * Ti_laplacian


def _evaluate_field_value_and_gradient(
    geometry: FciGeometry3D,
    time: float,
    field_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rho_min, rho_max = _mms_radial_bounds(geometry)
    flat_coords = _mms_cell_coordinates(geometry).reshape((-1, 3))
    time_value = jnp.asarray(time, dtype=jnp.float64)

    def _field_at(coord: jnp.ndarray) -> jnp.ndarray:
        return field_fn(coord, time_value, rho_min, rho_max)

    def _value_and_gradient(coord: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.value_and_grad(_field_at)(coord)

    values, gradients = jax.vmap(_value_and_gradient)(flat_coords)
    return values.reshape(geometry.shape), gradients.reshape(geometry.shape + (3,))


def _evaluate_vorticity_value_gradient_time(
    context: ShiftedTorusEbMmsContext,
    time: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    geometry = context.geometry
    rho_min, rho_max = _mms_radial_bounds(geometry)
    flat_coords = _mms_cell_coordinates(geometry).reshape((-1, 3))
    time_value = jnp.asarray(time, dtype=jnp.float64)
    tau = float(context.parameters.tau)

    def _omega(coord: jnp.ndarray, local_time: jnp.ndarray) -> jnp.ndarray:
        return _mms_vorticity_scalar(coord, local_time, rho_min, rho_max, tau)

    def _value_gradient_time(coord: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        value, gradients = jax.value_and_grad(_omega, argnums=(0, 1))(coord, time_value)
        coord_gradient, time_gradient = gradients
        return value, coord_gradient, time_gradient

    values, gradients, time_gradients = jax.vmap(_value_gradient_time)(flat_coords)
    return (
        values.reshape(geometry.shape),
        gradients.reshape(geometry.shape + (3,)),
        time_gradients.reshape(geometry.shape),
    )


def _analytic_mms_data(context: ShiftedTorusEbMmsContext, time: float) -> _AnalyticMmsData:
    geometry = context.geometry
    density, density_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_density_scalar)
    phi, phi_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_phi_scalar)
    Te, Te_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_Te_scalar)
    Ti, Ti_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_Ti_scalar)
    Vi, Vi_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_Vi_scalar)
    Ve, Ve_grad = _evaluate_field_value_and_gradient(geometry, time, _mms_Ve_scalar)
    vorticity, vorticity_grad, vorticity_t = _evaluate_vorticity_value_gradient_time(context, time)
    return _AnalyticMmsData(
        density=density,
        phi=phi,
        Te=Te,
        Ti=Ti,
        Vi=Vi,
        Ve=Ve,
        vorticity=vorticity,
        density_t=_mms_density_t(geometry, time),
        phi_t=_mms_phi_t(geometry, time),
        Te_t=_mms_Te_t(geometry, time),
        Ti_t=_mms_Ti_t(geometry, time),
        Vi_t=_mms_Vi_t(geometry, time),
        Ve_t=_mms_Ve_t(geometry, time),
        vorticity_t=vorticity_t,
        density_grad=density_grad,
        phi_grad=phi_grad,
        Te_grad=Te_grad,
        Ti_grad=Ti_grad,
        Vi_grad=Vi_grad,
        Ve_grad=Ve_grad,
        vorticity_grad=vorticity_grad,
    )


def _mms_density(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return N0 + EPS_N * H * (
        0.70 * jnp.cos(2.0 * theta_s + 3.0 * zeta + W_N * time)
        + 0.30 * jnp.sin(3.0 * theta_s - 2.0 * zeta + W_N * time)
    )


def _mms_density_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_N * H * W_N * (
        -0.70 * jnp.sin(2.0 * theta_s + 3.0 * zeta + W_N * time)
        + 0.30 * jnp.cos(3.0 * theta_s - 2.0 * zeta + W_N * time)
    )


def _mms_Te(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return TE0 + EPS_TE * H * (
        0.60 * jnp.sin(theta_s + 2.0 * zeta + W_TE * time)
        + 0.25 * jnp.cos(4.0 * theta_s - zeta + W_TE * time)
    )


def _mms_Te_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_TE * H * W_TE * (
        0.60 * jnp.cos(theta_s + 2.0 * zeta + W_TE * time)
        - 0.25 * jnp.sin(4.0 * theta_s - zeta + W_TE * time)
    )


def _mms_Ti(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return TI0 + EPS_TI * H * (
        0.50 * jnp.cos(3.0 * theta_s - zeta + W_TI * time)
        + 0.25 * jnp.sin(2.0 * theta_s + 4.0 * zeta + W_TI * time)
    )


def _mms_Ti_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_TI * H * W_TI * (
        -0.50 * jnp.sin(3.0 * theta_s - zeta + W_TI * time)
        + 0.25 * jnp.cos(2.0 * theta_s + 4.0 * zeta + W_TI * time)
    )


def _mms_phi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_PHI * H * (
        jnp.cos(2.0 * theta_s) * jnp.sin(3.0 * zeta + W_PHI * time)
        + 0.50 * jnp.sin(3.0 * theta_s + 2.0 * zeta + W_PHI * time)
        + 0.25 * jnp.cos(4.0 * theta_s - zeta + W_PHI * time)
    )


def _mms_phi_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_PHI * H * W_PHI * (
        jnp.cos(2.0 * theta_s) * jnp.cos(3.0 * zeta + W_PHI * time)
        + 0.50 * jnp.cos(3.0 * theta_s + 2.0 * zeta + W_PHI * time)
        - 0.25 * jnp.sin(4.0 * theta_s - zeta + W_PHI * time)
    )


def _mms_Ve(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_VE * H * (
        jnp.sin(theta_s - 3.0 * zeta + W_VE * time)
        + 0.30 * jnp.cos(2.0 * theta_s + zeta + W_VE * time)
    )


def _mms_Ve_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_VE * H * W_VE * (
        jnp.cos(theta_s - 3.0 * zeta + W_VE * time)
        - 0.30 * jnp.sin(2.0 * theta_s + zeta + W_VE * time)
    )


def _mms_Vi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_VI * H * (
        jnp.cos(theta_s + 2.0 * zeta + W_VI * time)
        + 0.30 * jnp.sin(3.0 * theta_s - zeta + W_VI * time)
    )


def _mms_Vi_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    H, theta_s, _, zeta = _mms_coordinates(geometry)
    return EPS_VI * H * W_VI * (
        -jnp.sin(theta_s + 2.0 * zeta + W_VI * time)
        + 0.30 * jnp.cos(3.0 * theta_s - zeta + W_VI * time)
    )


def _perp_laplacian(
    field: jnp.ndarray,
    context: ShiftedTorusEbMmsContext,
    *,
    face_bc: BoundaryFaceBC3D,
) -> jnp.ndarray:
    stencil = context.conservative_stencil_builder(
        field,
        context.geometry,
        PERIODIC_AXES,
        face_bc,
    )
    return perp_laplacian_conservative_op(
        stencil,
        context.geometry,
        face_projectors=context.face_projectors,
        face_bc=face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(context.geometry),
        cut_wall_geometry=context.cut_wall_geometry,
        cut_wall_bc=context.cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
    )


def _mms_vorticity(context: ShiftedTorusEbMmsContext, time: float) -> jnp.ndarray:
    vorticity, _, _ = _evaluate_vorticity_value_gradient_time(context, time)
    return vorticity


def _mms_vorticity_t(context: ShiftedTorusEbMmsContext, time: float) -> jnp.ndarray:
    _, _, vorticity_t = _evaluate_vorticity_value_gradient_time(context, time)
    return vorticity_t


def _mms_exact_state(context: ShiftedTorusEbMmsContext, time: float) -> FciDrbEBState:
    geometry = context.geometry
    return FciDrbEBState(
        density=_mms_density(geometry, time),
        phi=_mms_phi(geometry, time),
        Te=_mms_Te(geometry, time),
        Ti=_mms_Ti(geometry, time),
        Vi=_mms_Vi(geometry, time),
        Ve=_mms_Ve(geometry, time),
        vorticity=_mms_vorticity(context, time),
    )


def _mms_exact_time_derivative_state(context: ShiftedTorusEbMmsContext, time: float) -> FciDrbEBState:
    geometry = context.geometry
    return FciDrbEBState(
        density=_mms_density_t(geometry, time),
        phi=_mms_phi_t(geometry, time),
        Te=_mms_Te_t(geometry, time),
        Ti=_mms_Ti_t(geometry, time),
        Vi=_mms_Vi_t(geometry, time),
        Ve=_mms_Ve_t(geometry, time),
        vorticity=_mms_vorticity_t(context, time),
    )


def _mms_exact_state_from_data(data: _AnalyticMmsData) -> FciDrbEBState:
    return FciDrbEBState(
        density=data.density,
        phi=data.phi,
        Te=data.Te,
        Ti=data.Ti,
        Vi=data.Vi,
        Ve=data.Ve,
        vorticity=data.vorticity,
    )


def _mms_exact_time_derivative_state_from_data(data: _AnalyticMmsData) -> FciDrbEBState:
    return FciDrbEBState(
        density=data.density_t,
        phi=data.phi_t,
        Te=data.Te_t,
        Ti=data.Ti_t,
        Vi=data.Vi_t,
        Ve=data.Ve_t,
        vorticity=data.vorticity_t,
    )


def _analytic_eb_rhs_from_data(
    data: _AnalyticMmsData,
    context: ShiftedTorusEbMmsContext,
) -> FciDrbEBState:
    """Evaluate the non-diffusive continuum EB RHS on the manufactured state."""

    geometry = context.geometry
    parameters = context.parameters
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(data.density, 1.0e-30)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    tau = jnp.asarray(parameters.tau, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    Ve_nu = jnp.asarray(parameters.Ve_nu, dtype=jnp.float64)
    b_contra = jnp.asarray(geometry.cell_bfield.b_contra, dtype=jnp.float64)
    b_cov = jnp.einsum(
        "...ij,...j->...i",
        jnp.asarray(geometry.cell_metric.g_cov, dtype=jnp.float64),
        b_contra,
    )
    jacobian = jnp.maximum(jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64), 1.0e-30)

    def _poisson(left_grad: jnp.ndarray, right_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_cov * jnp.cross(left_grad, right_grad), axis=-1) / jacobian

    def _grad_parallel(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", b_contra, field_grad)

    def _curvature(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", context.curvature_coefficients, field_grad)

    Pe = data.density * data.Te
    Pi = data.density * data.Ti
    pressure = Pe + tau * Pi
    current_density = data.density * (data.Vi - data.Ve)
    Pe_grad = data.Te[..., None] * data.density_grad + data.density[..., None] * data.Te_grad
    Pi_grad = data.Ti[..., None] * data.density_grad + data.density[..., None] * data.Ti_grad
    pressure_grad = Pe_grad + tau * Pi_grad
    current_density_grad = (
        (data.Vi - data.Ve)[..., None] * data.density_grad
        + data.density[..., None] * (data.Vi_grad - data.Ve_grad)
    )
    density_flux_grad = data.Ve[..., None] * data.density_grad + data.density[..., None] * data.Ve_grad

    curvature_Pe = _curvature(Pe_grad)
    curvature_pressure = _curvature(pressure_grad)
    curvature_phi = _curvature(data.phi_grad)
    curvature_Te = _curvature(data.Te_grad)
    curvature_Ti = _curvature(data.Ti_grad)
    grad_parallel_Te = _grad_parallel(data.Te_grad)
    grad_parallel_Ti = _grad_parallel(data.Ti_grad)
    grad_parallel_Ve = _grad_parallel(data.Ve_grad)
    grad_parallel_Vi = _grad_parallel(data.Vi_grad)
    grad_parallel_phi = _grad_parallel(data.phi_grad)
    grad_parallel_Pe = _grad_parallel(Pe_grad)
    grad_parallel_pressure = _grad_parallel(pressure_grad)
    grad_parallel_current_density = _grad_parallel(current_density_grad)
    grad_parallel_density_flux = _grad_parallel(density_flux_grad)
    grad_parallel_vorticity = _grad_parallel(data.vorticity_grad)

    density_rhs = (
        -(_poisson(data.phi_grad, data.density_grad) / (rho_star * bmag))
        - grad_parallel_density_flux
        + (2.0 / bmag) * (curvature_Pe - data.density * curvature_phi)
    )
    Te_rhs = (
        -(_poisson(data.phi_grad, data.Te_grad) / (rho_star * bmag))
        - data.Ve * grad_parallel_Te
        + (4.0 * data.Te / (3.0 * bmag))
        * (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
        + (2.0 * data.Te / (3.0 * density_safe))
        * (0.71 * grad_parallel_current_density - data.density * grad_parallel_Ve)
    )
    Ti_rhs = (
        -(_poisson(data.phi_grad, data.Ti_grad) / (rho_star * bmag))
        - data.Vi * grad_parallel_Ti
        + (4.0 * data.Ti / (3.0 * bmag))
        * (curvature_Pe / density_safe - 2.5 * tau * curvature_Ti - curvature_phi)
        + (2.0 * data.Ti / (3.0 * density_safe))
        * (grad_parallel_current_density - data.density * grad_parallel_Vi)
    )
    Vi_rhs = (
        -(_poisson(data.phi_grad, data.Vi_grad) / (rho_star * bmag))
        - data.Vi * grad_parallel_Vi
        - grad_parallel_pressure / density_safe
    )
    Ve_rhs = (
        -(_poisson(data.phi_grad, data.Ve_grad) / (rho_star * bmag))
        - data.Ve * grad_parallel_Ve
        + mi_over_me
        * (
            Ve_nu * current_density
            + grad_parallel_phi
            - grad_parallel_Pe / density_safe
            - 0.71 * grad_parallel_Te
        )
    )
    vorticity_rhs = (
        -(_poisson(data.phi_grad, data.vorticity_grad) / (rho_star * bmag))
        - data.Vi * grad_parallel_vorticity
        + (bmag * bmag / density_safe) * grad_parallel_current_density
        + (2.0 * bmag / density_safe) * curvature_pressure
    )
    return FciDrbEBState(
        density=density_rhs,
        phi=data.phi,
        Te=Te_rhs,
        Ti=Ti_rhs,
        Vi=Vi_rhs,
        Ve=Ve_rhs,
        vorticity=vorticity_rhs,
    )


def _analytic_eb_rhs_from_exact_state(context: ShiftedTorusEbMmsContext, time: float) -> FciDrbEBState:
    return _analytic_eb_rhs_from_data(_analytic_mms_data(context, time), context)


def _eb_rhs_without_mms_sources(
    state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
) -> FciDrbEBState:
    bcs = context.boundary_conditions
    result = compute_fci_drb_eb_rhs(
        state,
        geometry=context.geometry,
        stencil_builder=context.stencil_builder,
        conservative_stencil_builder=context.conservative_stencil_builder,
        parameters=context.parameters,
        curvature_coefficients=context.curvature_coefficients,
        density_face_bc=bcs.density_face_bc,
        potential_face_bc=bcs.potential_face_bc,
        vorticity_face_bc=bcs.vorticity_face_bc,
        electron_temperature_face_bc=bcs.Te_face_bc,
        ion_temperature_face_bc=bcs.Ti_face_bc,
        electron_velocity_parallel_face_bc=bcs.Ve_face_bc,
        ion_velocity_parallel_face_bc=bcs.Vi_face_bc,
        density_cut_wall_geometry=context.cut_wall_geometry,
        density_cut_wall_bc=bcs.density_cut_wall_bc,
        potential_cut_wall_geometry=context.cut_wall_geometry,
        potential_cut_wall_bc=bcs.potential_cut_wall_bc,
        vorticity_cut_wall_geometry=context.cut_wall_geometry,
        vorticity_cut_wall_bc=bcs.vorticity_cut_wall_bc,
        electron_temperature_cut_wall_geometry=context.cut_wall_geometry,
        electron_temperature_cut_wall_bc=bcs.Te_cut_wall_bc,
        ion_temperature_cut_wall_geometry=context.cut_wall_geometry,
        ion_temperature_cut_wall_bc=bcs.Ti_cut_wall_bc,
        electron_velocity_parallel_cut_wall_geometry=context.cut_wall_geometry,
        electron_velocity_parallel_cut_wall_bc=bcs.Ve_cut_wall_bc,
        ion_velocity_parallel_cut_wall_geometry=context.cut_wall_geometry,
        ion_velocity_parallel_cut_wall_bc=bcs.Vi_cut_wall_bc,
        periodic_axes=PERIODIC_AXES,
    )
    return result.rhs


def _subtract_state(left: FciDrbEBState, right: FciDrbEBState) -> FciDrbEBState:
    return FciDrbEBState(
        density=left.density - right.density,
        phi=left.phi - right.phi,
        Te=left.Te - right.Te,
        Ti=left.Ti - right.Ti,
        Vi=left.Vi - right.Vi,
        Ve=left.Ve - right.Ve,
        vorticity=left.vorticity - right.vorticity,
    )


def _add_state(left: FciDrbEBState, right: FciDrbEBState, *, scale: float = 1.0) -> FciDrbEBState:
    return left.axpy(right, scale=scale)


def _mms_source_state(context: ShiftedTorusEbMmsContext, time: float) -> FciDrbEBState:
    """Return analytic source residuals S = d_t U_ex - R_cont(U_ex).

    For the six EB equations this encodes

        S_N, S_Te, S_Ti, S_Vi, S_Ve, S_omega.

    The ``phi`` entry is a test-harness carrier source

        S_phi = d_t phi_ex - phi_ex,

    because the current EB RHS stores ``state.phi`` in ``rhs.phi`` rather than
    solving an elliptic phi equation internally.
    """

    analytic_data = _analytic_mms_data(context, time)
    exact_time_derivative = _mms_exact_time_derivative_state_from_data(analytic_data)
    rhs_at_exact_state = _analytic_eb_rhs_from_data(analytic_data, context)
    return _subtract_state(exact_time_derivative, rhs_at_exact_state)


def _mms_rhs_with_sources(
    state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
    *,
    time: float,
) -> FciDrbEBState:
    return _add_state(
        _eb_rhs_without_mms_sources(state, context),
        _mms_source_state(context, time),
    )


def _rk4_step(
    state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
    *,
    time: float,
    timestep: float,
) -> FciDrbEBState:
    def _rhs_fn(current_state: FciDrbEBState, stage_time: float | jax.Array, carry: None):
        del carry
        rhs = _mms_rhs_with_sources(current_state, context, time=float(stage_time))
        return rhs, None, None

    step_result = rk4_step(state, time=time, timestep=timestep, rhs_fn=_rhs_fn, carry=None)
    return step_result.state


def simulate_mms_shifted_torus_eb(
    context: ShiftedTorusEbMmsContext,
    *,
    final_time: float = TF,
    timestep: float | None = None,
    num_steps: int = NUM_STEPS,
    show_progress: bool = False,
) -> tuple[FciDrbEBState, jnp.ndarray]:
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    state = _mms_exact_state(context, 0.0)
    time_value = 0.0
    progress_start = time_module.perf_counter()
    if show_progress:
        print(
            f"shifted_torus_EB MMS RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
            end="",
            flush=True,
        )
    for step_index in range(steps):
        state = _rk4_step(state, context, time=time_value, timestep=dt)
        jax.block_until_ready(state.density)
        time_value += dt
        if show_progress:
            print(
                f"\r\033[Kshifted_torus_EB MMS RK4 progress: "
                f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )
    if show_progress:
        print()
    return state, jnp.asarray(time_value, dtype=jnp.float64)


def _weighted_l2_error(actual: jnp.ndarray, expected: jnp.ndarray, geometry: FciGeometry3D) -> float:
    error = jnp.asarray(actual - expected, dtype=jnp.float64)
    weights = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
    return float(jnp.sqrt(jnp.sum(weights * error * error) / jnp.sum(weights)))


def _field_error_statistics(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    geometry: FciGeometry3D,
) -> tuple[float, float, float]:
    weighted_l2 = _weighted_l2_error(actual, expected, geometry)
    expected_l2 = _weighted_l2_error(expected, jnp.zeros_like(expected), geometry)
    linf = float(jnp.max(jnp.abs(jnp.asarray(actual - expected, dtype=jnp.float64))))
    return weighted_l2, linf, weighted_l2 / max(expected_l2, 1.0e-30)


def _state_error_statistics(
    actual: FciDrbEBState,
    expected: FciDrbEBState,
    geometry: FciGeometry3D,
) -> dict[str, tuple[float, float, float]]:
    return {
        "density": _field_error_statistics(actual.density, expected.density, geometry),
        "phi": _field_error_statistics(actual.phi, expected.phi, geometry),
        "Te": _field_error_statistics(actual.Te, expected.Te, geometry),
        "Ti": _field_error_statistics(actual.Ti, expected.Ti, geometry),
        "Vi": _field_error_statistics(actual.Vi, expected.Vi, geometry),
        "Ve": _field_error_statistics(actual.Ve, expected.Ve, geometry),
        "vorticity": _field_error_statistics(actual.vorticity, expected.vorticity, geometry),
    }


def _print_state_error_statistics(label: str, stats: dict[str, tuple[float, float, float]]) -> None:
    print(label)
    for field_name, (l2, linf, rel_l2) in stats.items():
        print(f"  {field_name}: weighted_l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")


def _combined_l2_error(stats: dict[str, tuple[float, float, float]]) -> float:
    return float(np.sqrt(np.mean([field_stats[0] ** 2 for field_stats in stats.values()])))


def _combined_error_statistics(
    final_state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
    time: float,
) -> tuple[float, float, float]:
    exact = _mms_exact_state(context, time)
    error = jnp.concatenate(
        [
            jnp.ravel(jnp.abs(final_state.density - exact.density)),
            jnp.ravel(jnp.abs(final_state.phi - exact.phi)),
            jnp.ravel(jnp.abs(final_state.Te - exact.Te)),
            jnp.ravel(jnp.abs(final_state.Ti - exact.Ti)),
            jnp.ravel(jnp.abs(final_state.Vi - exact.Vi)),
            jnp.ravel(jnp.abs(final_state.Ve - exact.Ve)),
            jnp.ravel(jnp.abs(final_state.vorticity - exact.vorticity)),
        ]
    )
    return float(jnp.sqrt(jnp.mean(error**2))), float(jnp.median(error)), float(jnp.max(error))


def _observed_order(error_coarse: float, error_fine: float, resolution_coarse: int, resolution_fine: int) -> float:
    if error_coarse <= 0.0 or error_fine <= 0.0:
        return float("nan")
    return float(np.log(error_coarse / error_fine) / np.log(float(resolution_fine) / float(resolution_coarse)))


def _closure_error_statistics(
    state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
) -> tuple[float, float, float]:
    bcs = context.boundary_conditions
    closed_omega = _perp_laplacian(state.phi, context, face_bc=bcs.potential_face_bc) + (
        context.parameters.tau * _perp_laplacian(state.Ti, context, face_bc=bcs.Ti_face_bc)
    )
    return _field_error_statistics(state.vorticity, closed_omega, context.geometry)


def test_mms_shifted_torus_eb_source_recovers_exact_time_derivatives() -> None:
    context = build_shifted_torus_eb_mms_context((4, 6, 4))
    time = 0.013
    exact_derivative = _mms_exact_time_derivative_state(context, time)
    computed_derivative = _add_state(
        _analytic_eb_rhs_from_exact_state(context, time),
        _mms_source_state(context, time),
    )

    for actual, expected in (
        (computed_derivative.density, exact_derivative.density),
        (computed_derivative.phi, exact_derivative.phi),
        (computed_derivative.Te, exact_derivative.Te),
        (computed_derivative.Ti, exact_derivative.Ti),
        (computed_derivative.Vi, exact_derivative.Vi),
        (computed_derivative.Ve, exact_derivative.Ve),
        (computed_derivative.vorticity, exact_derivative.vorticity),
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-12, atol=1.0e-12)


def test_mms_shifted_torus_eb_source_does_not_call_discrete_rhs() -> None:
    context = build_shifted_torus_eb_mms_context((4, 6, 4))
    original_rhs = globals()["_eb_rhs_without_mms_sources"]

    def _unexpected_discrete_rhs(*args, **kwargs):
        raise AssertionError("_mms_source_state must use the analytic RHS, not the discrete RHS")

    globals()["_eb_rhs_without_mms_sources"] = _unexpected_discrete_rhs
    try:
        source = _mms_source_state(context, 0.013)
        jax.block_until_ready(source.density)
    finally:
        globals()["_eb_rhs_without_mms_sources"] = original_rhs


def _discrete_source_residual_l2(shape: tuple[int, int, int], time: float) -> float:
    context = build_shifted_torus_eb_mms_context(shape)
    exact_state = _mms_exact_state(context, time)
    exact_derivative = _mms_exact_time_derivative_state(context, time)
    computed_derivative = _mms_rhs_with_sources(exact_state, context, time=time)
    stats = _state_error_statistics(computed_derivative, exact_derivative, context.geometry)
    return _combined_l2_error(stats)


def test_mms_shifted_torus_eb_discrete_residual_is_not_roundoff_and_refines() -> None:
    time = 0.013
    coarse_l2 = _discrete_source_residual_l2((5, 7, 5), time)
    fine_l2 = _discrete_source_residual_l2((7, 9, 7), time)

    assert np.isfinite(coarse_l2)
    assert np.isfinite(fine_l2)
    assert coarse_l2 > 1.0e-12
    assert fine_l2 < coarse_l2


def test_mms_shifted_torus_eb_short_rk4_smoke() -> None:
    context = build_shifted_torus_eb_mms_context((6, 8, 6))
    final_state, final_time = simulate_mms_shifted_torus_eb(
        context,
        final_time=0.002,
        num_steps=2,
    )
    exact_state = _mms_exact_state(context, float(final_time))
    stats = _state_error_statistics(final_state, exact_state, context.geometry)

    assert _combined_l2_error(stats) < 1.0e-2
    assert _closure_error_statistics(final_state, context)[0] < 2.0e-1


def _run_convergence() -> None:
    resolutions = (20, 40, 80)
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    previous_resolution: int | None = None
    previous_error: float | None = None
    for resolution in resolutions:
        context = build_shifted_torus_eb_mms_context((resolution, resolution, resolution))
        steps = max(4, int(round(NUM_STEPS * np.sqrt(resolution / resolutions[0]))))
        print(f"Starting shifted_torus_EB MMS run: resolution={resolution}, steps={steps}")
        start = time_module.perf_counter()
        final_state, final_time = simulate_mms_shifted_torus_eb(
            context,
            final_time=TF,
            num_steps=steps,
            show_progress=True,
        )
        elapsed = time_module.perf_counter() - start
        exact_state = _mms_exact_state(context, float(final_time))
        stats = _state_error_statistics(final_state, exact_state, context.geometry)
        combined_error, _, max_error = _combined_error_statistics(final_state, context, float(final_time))
        successful_resolutions.append(int(resolution))
        l2_errors.append(combined_error)
        max_errors.append(max_error)
        order_text = ""
        if previous_resolution is not None and previous_error is not None:
            order = _observed_order(previous_error, combined_error, previous_resolution, resolution)
            order_text = f", order={order:.3f}"
        print(
            f"resolution={resolution}: elapsed={elapsed:.6e} s, "
            f"combined_weighted_l2={combined_error:.6e}{order_text}"
        )
        _print_state_error_statistics(f"resolution={resolution} final errors", stats)
        closure_stats = _closure_error_statistics(final_state, context)
        print(
            f"  closure: weighted_l2={closure_stats[0]:.6e}, "
            f"linf={closure_stats[1]:.6e}, rel_l2={closure_stats[2]:.6e}"
        )
        previous_resolution = resolution
        previous_error = combined_error

    if successful_resolutions:
        import matplotlib.pyplot as plt

        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        print(f"shifted_torus_EB l2 convergence order: {-l2_slope:.6f}")
        print(f"shifted_torus_EB max convergence order: {-max_slope:.6f}")

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"l2, order {-l2_slope:.2f}")
        ax.loglog(plotted_resolutions, max_errors, "^-", label=f"max, order {-max_slope:.2f}")
        ax.loglog(
            plotted_resolutions,
            np.exp(l2_intercept) * plotted_resolutions.astype(np.float64) ** l2_slope,
            "--",
            color=ax.lines[0].get_color(),
        )
        ax.loglog(
            plotted_resolutions,
            np.exp(max_intercept) * plotted_resolutions.astype(np.float64) ** max_slope,
            "--",
            color=ax.lines[1].get_color(),
        )
        ax.set_xlabel("resolution")
        ax.set_ylabel("absolute error")
        ax.set_title("Shifted-torus EB MMS convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig("shifted_torus_EB_convergence.png", dpi=200)
        plt.close(fig)
    else:
        print("WARNING: no valid resolutions completed, skipping convergence plot.")


if __name__ == "__main__":
    _run_convergence()
