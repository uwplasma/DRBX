"""MMS harness for the shifted-torus electrostatic Boussinesq DRB model.

The domain-decomposed MMS run treats ``phi`` as an elliptic constraint instead
of an independently evolved field. The manufactured vorticity is built from the
same discrete perpendicular-Laplacian closure used by the local phi inversion,
while the remaining manufactured fields use analytic continuum source terms.
The live RK4 update reconstructs ``phi`` from the evolved vorticity and ion
temperature at every stage. Radial physical faces use time-dependent MMS
boundary data.

For the first analytic EB MMS pass all explicit diffusion coefficients are set
to zero. This keeps the source terms focused on the advective, parallel,
curvature, pressure, current, and collision pieces without requiring fourth
derivatives of the manufactured vorticity.
"""

from __future__ import annotations

import argparse
import time as time_module
from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    ConservativeStencilBuilder,
    FaceBFieldGeometry,
    FciGeometry3D,
    FciMaps3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    Spacing3D,
    StencilBuilderContext,
    build_curvature_coefficients,
    build_local_conservative_stencil_from_field,
    build_local_curvature_coefficients,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import (
    FciDrbEBBoundaryConditions,
    FciDrbEBRhsParameters,
    FciDrbEBState,
    SpmdGmresConfig,
    inject_owned_field_to_halo,
    perp_laplacian_conservative_op,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
)
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    PhysicalGhostCellFiller3D,
    PreparedLocalState3D,
    TopologyHaloFiller3D,
    LocalPeriodicTopologyRule3D,
)
from jax_drb.native.fci_model import FciFieldBundle, inject_owned_state_to_halo
from jax_drb.native.fci_operators import (
    _build_global_conservative_stencil_compat,
    LocalPerpLaplacianInverseSolver,
    build_conservative_stencil_from_field,
    build_local_perp_laplacian_face_projectors,
    build_perp_laplacian_face_projectors,
    local_perp_laplacian_conservative_op,
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_parallel_flux_div_op,
    local_parallel_laplacian_conservative_op,
    local_poisson_bracket_op,
)

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
from mms_domain_decomp_helpers import (  # noqa: E402
    MESH_AXIS_NAMES,
    assert_shape_divisible_by_shards,
    build_shifted_torus_local_domain,
    build_shifted_torus_local_geometry,
    expand_local_shard_pytree,
    extract_local_shard_pytree,
    local_shard_pytree_partition_spec,
    make_mesh_for_shard_counts,
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorusEbFaceBCBundle(FciFieldBundle):
    density: LocalBoundaryFaceBC3D
    phi: LocalBoundaryFaceBC3D
    Te: LocalBoundaryFaceBC3D
    Ti: LocalBoundaryFaceBC3D
    Vi: LocalBoundaryFaceBC3D
    Ve: LocalBoundaryFaceBC3D
    vorticity: LocalBoundaryFaceBC3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorusEbInvariantBundle(FciFieldBundle):
    coordinates: jnp.ndarray
    face_coordinates: jnp.ndarray
    bmag_halo: jnp.ndarray
    b_contra_halo: jnp.ndarray
    cell_metric_g_cov_halo: jnp.ndarray
    cell_metric_jacobian_halo: jnp.ndarray
    curvature_coefficients_owned: jnp.ndarray
    curvature_coefficients_halo: jnp.ndarray
    face_projector_x: jnp.ndarray
    face_projector_y: jnp.ndarray
    face_projector_z: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorusEbStageData(FciFieldBundle):
    exact_halo: FciDrbEBState
    source_halo: FciDrbEBState
    face_lower: FciDrbEBState
    face_upper: FciDrbEBState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorusEbRk4StageData(FciFieldBundle):
    stage_1: _ShiftedTorusEbStageData
    stage_2: _ShiftedTorusEbStageData
    stage_3: _ShiftedTorusEbStageData
    stage_4: _ShiftedTorusEbStageData


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _VeTermResidualBundle(FciFieldBundle):
    poisson: jnp.ndarray
    parallel_advection: jnp.ndarray
    collision: jnp.ndarray
    grad_phi: jnp.ndarray
    grad_pe: jnp.ndarray
    grad_te: jnp.ndarray
    source: jnp.ndarray
    total: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _PhiClosureDiagnosticBundle(FciFieldBundle):
    phi: jnp.ndarray
    grad_parallel_phi: jnp.ndarray
    closure_rhs: jnp.ndarray


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
        phi_inversion_iterations=500,
        phi_inversion_regularization=0.0,
        density_D_perp=0.0,
        density_D_parallel=0.0,
        electron_temperature_chi_parallel=0.0,
        electron_temperature_D_perp=0.0,
        ion_temperature_chi_parallel=0.0,
        ion_temperature_D_perp=0.0,
        Ve_nu=1.0e-3,
        Ve_D_perp=0.0,
        Ve_parallel_viscosity=0.0,
        Vi_D_perp=0.0,
        Vi_parallel_viscosity=0.0,
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
    stencil = _build_global_conservative_stencil_compat(
        context.conservative_stencil_builder,
        field,
        context.geometry,
        periodic_axes=PERIODIC_AXES,
        face_bc=face_bc,
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

    The ``phi`` entry is the source for the reconstructed-potential carrier

        S_phi = d_t phi_ex - phi_ex,

    so the discrete RHS uses the locally reconstructed phi while the continuum
    manufactured residual remains anchored to the exact phi.
    """

    analytic_data = _analytic_mms_data(context, time)
    exact_time_derivative = _mms_exact_time_derivative_state_from_data(analytic_data)
    rhs_at_exact_state = _analytic_eb_rhs_from_data(analytic_data, context)
    return _subtract_state(exact_time_derivative, rhs_at_exact_state)


def _local_coordinates(
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray:
    rho, theta, zeta = jnp.meshgrid(
        geometry.grid.x.centers_halo,
        geometry.grid.y.centers_halo,
        geometry.grid.z.centers_halo,
        indexing="ij",
    )
    return jnp.stack((rho, theta, zeta), axis=-1)


def _local_x_face_coordinates(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    layout = domain.layout
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    rho_faces = jnp.asarray(
        [geometry.grid.x.faces_halo[h], geometry.grid.x.faces_halo[h + nx]],
        dtype=jnp.float64,
    )
    rho, theta, zeta = jnp.meshgrid(
        rho_faces,
        geometry.grid.y.centers_halo[h : h + ny],
        geometry.grid.z.centers_halo[h : h + nz],
        indexing="ij",
    )
    return jnp.stack((rho, theta, zeta), axis=-1)


def _build_local_eb_invariants(
    shard_index: tuple[int, int, int],
    *,
    owned_shape: tuple[int, int, int],
    halo_width: int,
    global_shape: tuple[int, int, int],
    domain: LocalDomain3D,
) -> _ShiftedTorusEbInvariantBundle:
    local_geometry = build_shifted_torus_local_geometry(
        owned_shape,
        halo_width,
        global_shape=global_shape,
        shard_index=shard_index,
        x_min=x_min,
        x_max=x_max,
        r0=r0,
        alpha_value=alpha_value,
        iota=iota,
        c_phi=c_phi,
        sigma=sigma,
    )
    curvature_coefficients_owned = jnp.asarray(
        build_local_curvature_coefficients(local_geometry, domain),
        dtype=jnp.float64,
    )
    curvature_coefficients_halo = jnp.zeros(
        domain.layout.cell_halo_shape + (3,),
        dtype=jnp.float64,
    ).at[domain.layout.owned_slices_cell + (slice(None),)].set(curvature_coefficients_owned)
    face_projectors = build_local_perp_laplacian_face_projectors(local_geometry, domain)
    return _ShiftedTorusEbInvariantBundle(
        coordinates=_local_coordinates(local_geometry),
        face_coordinates=_local_x_face_coordinates(local_geometry, domain),
        bmag_halo=jnp.asarray(local_geometry.cell_bfield.Bmag_halo, dtype=jnp.float64),
        b_contra_halo=jnp.asarray(local_geometry.cell_bfield.b_contra, dtype=jnp.float64),
        cell_metric_g_cov_halo=jnp.asarray(local_geometry.cell_metric.g_cov, dtype=jnp.float64),
        cell_metric_jacobian_halo=jnp.asarray(local_geometry.cell_metric.J_halo, dtype=jnp.float64),
        curvature_coefficients_owned=curvature_coefficients_owned,
        curvature_coefficients_halo=curvature_coefficients_halo,
        face_projector_x=jnp.asarray(face_projectors[0], dtype=jnp.float64),
        face_projector_y=jnp.asarray(face_projectors[1], dtype=jnp.float64),
        face_projector_z=jnp.asarray(face_projectors[2], dtype=jnp.float64),
    )


def _local_analytic_data(
    coordinates: jnp.ndarray,
    time: float | jax.Array,
    *,
    rho_min: float,
    rho_max: float,
    parameters: FciDrbEBRhsParameters,
) -> _AnalyticMmsData:
    flat_coords = jnp.reshape(jnp.asarray(coordinates, dtype=jnp.float64), (-1, 3))
    time_value = jnp.asarray(time, dtype=jnp.float64)

    def _field_value_grad_time(field_fn, coord: jnp.ndarray):
        def _at(point: jnp.ndarray, local_time: jnp.ndarray) -> jnp.ndarray:
            return field_fn(point, local_time, rho_min, rho_max)

        value, grads = jax.value_and_grad(_at, argnums=(0, 1))(coord, time_value)
        return value, grads[0], grads[1]

    def _omega_value_grad_time(coord: jnp.ndarray):
        def _omega(point: jnp.ndarray, local_time: jnp.ndarray) -> jnp.ndarray:
            return _mms_vorticity_scalar(
                point,
                local_time,
                rho_min,
                rho_max,
                float(parameters.tau),
            )

        value, grads = jax.value_and_grad(_omega, argnums=(0, 1))(coord, time_value)
        return value, grads[0], grads[1]

    density, density_grad, density_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_density_scalar, coord)
    )(flat_coords)
    phi, phi_grad, phi_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_phi_scalar, coord)
    )(flat_coords)
    Te, Te_grad, Te_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_Te_scalar, coord)
    )(flat_coords)
    Ti, Ti_grad, Ti_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_Ti_scalar, coord)
    )(flat_coords)
    Vi, Vi_grad, Vi_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_Vi_scalar, coord)
    )(flat_coords)
    Ve, Ve_grad, Ve_t = jax.vmap(
        lambda coord: _field_value_grad_time(_mms_Ve_scalar, coord)
    )(flat_coords)
    vorticity, vorticity_grad, vorticity_t = jax.vmap(_omega_value_grad_time)(
        flat_coords
    )
    shape = coordinates.shape[:-1]
    return _AnalyticMmsData(
        density=density.reshape(shape),
        phi=phi.reshape(shape),
        Te=Te.reshape(shape),
        Ti=Ti.reshape(shape),
        Vi=Vi.reshape(shape),
        Ve=Ve.reshape(shape),
        vorticity=vorticity.reshape(shape),
        density_t=density_t.reshape(shape),
        phi_t=phi_t.reshape(shape),
        Te_t=Te_t.reshape(shape),
        Ti_t=Ti_t.reshape(shape),
        Vi_t=Vi_t.reshape(shape),
        Ve_t=Ve_t.reshape(shape),
        vorticity_t=vorticity_t.reshape(shape),
        density_grad=density_grad.reshape(shape + (3,)),
        phi_grad=phi_grad.reshape(shape + (3,)),
        Te_grad=Te_grad.reshape(shape + (3,)),
        Ti_grad=Ti_grad.reshape(shape + (3,)),
        Vi_grad=Vi_grad.reshape(shape + (3,)),
        Ve_grad=Ve_grad.reshape(shape + (3,)),
        vorticity_grad=vorticity_grad.reshape(shape + (3,)),
    )


def _local_analytic_eb_rhs_from_invariants(
    data: _AnalyticMmsData,
    invariants: _ShiftedTorusEbInvariantBundle,
    parameters: FciDrbEBRhsParameters,
) -> FciDrbEBState:
    bmag = jnp.maximum(invariants.bmag_halo, 1.0e-30)
    density_safe = jnp.maximum(data.density, 1.0e-30)
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    tau = jnp.asarray(parameters.tau, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    Ve_nu = jnp.asarray(parameters.Ve_nu, dtype=jnp.float64)
    b_contra = invariants.b_contra_halo
    b_cov = jnp.einsum("...ij,...j->...i", invariants.cell_metric_g_cov_halo, b_contra)
    jacobian = jnp.maximum(invariants.cell_metric_jacobian_halo, 1.0e-30)

    def _poisson(left_grad: jnp.ndarray, right_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_cov * jnp.cross(left_grad, right_grad), axis=-1) / jacobian

    def _grad_parallel(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", b_contra, field_grad)

    def _curvature(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", invariants.curvature_coefficients_halo, field_grad)

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
    del pressure, current_density

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
            Ve_nu * data.density * (data.Vi - data.Ve)
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


def _local_zero_state(shape: tuple[int, int, int]) -> FciDrbEBState:
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return FciDrbEBState(
        density=zeros,
        phi=zeros,
        Te=zeros,
        Ti=zeros,
        Vi=zeros,
        Ve=zeros,
        vorticity=zeros,
    )


def _split_local_x_face_state(state: FciDrbEBState) -> tuple[FciDrbEBState, FciDrbEBState]:
    return (
        FciDrbEBState(
            density=jnp.asarray(state.density[0], dtype=jnp.float64),
            phi=jnp.asarray(state.phi[0], dtype=jnp.float64),
            Te=jnp.asarray(state.Te[0], dtype=jnp.float64),
            Ti=jnp.asarray(state.Ti[0], dtype=jnp.float64),
            Vi=jnp.asarray(state.Vi[0], dtype=jnp.float64),
            Ve=jnp.asarray(state.Ve[0], dtype=jnp.float64),
            vorticity=jnp.asarray(state.vorticity[0], dtype=jnp.float64),
        ),
        FciDrbEBState(
            density=jnp.asarray(state.density[-1], dtype=jnp.float64),
            phi=jnp.asarray(state.phi[-1], dtype=jnp.float64),
            Te=jnp.asarray(state.Te[-1], dtype=jnp.float64),
            Ti=jnp.asarray(state.Ti[-1], dtype=jnp.float64),
            Vi=jnp.asarray(state.Vi[-1], dtype=jnp.float64),
            Ve=jnp.asarray(state.Ve[-1], dtype=jnp.float64),
            vorticity=jnp.asarray(state.vorticity[-1], dtype=jnp.float64),
        ),
    )


def _sample_local_eb_rk4_stage_data(domain: LocalDomain3D) -> _ShiftedTorusEbRk4StageData:
    zero_halo_state = _local_zero_state(domain.layout.cell_halo_shape)
    zero_face_state = _local_zero_state(domain.layout.face_control_shape(0)[1:])
    zero_stage = _ShiftedTorusEbStageData(
        exact_halo=zero_halo_state,
        source_halo=zero_halo_state,
        face_lower=zero_face_state,
        face_upper=zero_face_state,
    )
    return _ShiftedTorusEbRk4StageData(
        stage_1=zero_stage,
        stage_2=zero_stage,
        stage_3=zero_stage,
        stage_4=zero_stage,
    )


def _local_owned_state_from_halo_data(
    data: _AnalyticMmsData,
    vorticity_owned: jnp.ndarray,
    domain: LocalDomain3D,
) -> FciDrbEBState:
    owned = domain.layout.owned_slices_cell
    return FciDrbEBState(
        density=jnp.asarray(data.density[owned], dtype=jnp.float64),
        phi=jnp.asarray(data.phi[owned], dtype=jnp.float64),
        Te=jnp.asarray(data.Te[owned], dtype=jnp.float64),
        Ti=jnp.asarray(data.Ti[owned], dtype=jnp.float64),
        Vi=jnp.asarray(data.Vi[owned], dtype=jnp.float64),
        Ve=jnp.asarray(data.Ve[owned], dtype=jnp.float64),
        vorticity=jnp.asarray(vorticity_owned, dtype=jnp.float64),
    )


def _owned_state_from_halo_state(
    state_halo: FciDrbEBState,
    domain: LocalDomain3D,
) -> FciDrbEBState:
    owned = domain.layout.owned_slices_cell
    return FciDrbEBState(
        density=jnp.asarray(state_halo.density[owned], dtype=jnp.float64),
        phi=jnp.asarray(state_halo.phi[owned], dtype=jnp.float64),
        Te=jnp.asarray(state_halo.Te[owned], dtype=jnp.float64),
        Ti=jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64),
        Vi=jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64),
        Ve=jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64),
        vorticity=jnp.asarray(state_halo.vorticity[owned], dtype=jnp.float64),
    )


def _local_exact_time_derivative_owned_from_halo_data(
    data: _AnalyticMmsData,
    vorticity_t_owned: jnp.ndarray,
    domain: LocalDomain3D,
) -> FciDrbEBState:
    owned = domain.layout.owned_slices_cell
    zeros = jnp.zeros(domain.layout.owned_shape, dtype=jnp.float64)
    return FciDrbEBState(
        density=jnp.asarray(data.density_t[owned], dtype=jnp.float64),
        phi=zeros,
        Te=jnp.asarray(data.Te_t[owned], dtype=jnp.float64),
        Ti=jnp.asarray(data.Ti_t[owned], dtype=jnp.float64),
        Vi=jnp.asarray(data.Vi_t[owned], dtype=jnp.float64),
        Ve=jnp.asarray(data.Ve_t[owned], dtype=jnp.float64),
        vorticity=jnp.asarray(vorticity_t_owned, dtype=jnp.float64),
    )


def _prepare_local_field_halo(
    field_owned: jnp.ndarray,
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    face_bc: LocalBoundaryFaceBC3D,
) -> jnp.ndarray:
    field_halo = inject_owned_field_to_halo(field_owned, domain.layout)
    field_halo = topology_filler(halo_exchange(field_halo, domain), domain)
    return physical_ghost_filler(field_halo, domain, face_bc)


def _local_perp_laplacian_from_owned_field(
    field_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
) -> jnp.ndarray:
    field_halo = _prepare_local_field_halo(
        field_owned,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        face_bc=face_bc,
    )
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    stencil = build_local_conservative_stencil_from_field(field_halo, geometry, context)
    return local_perp_laplacian_conservative_op(
        stencil,
        geometry,
        domain,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=geometry.regular_face_geometry,
    )


def _local_discrete_vorticity_rhs_from_exact_state(
    state_owned: FciDrbEBState,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_bc: _ShiftedTorusEbFaceBCBundle,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients_owned: jnp.ndarray,
) -> jnp.ndarray:
    prepared = _prepare_local_eb_stage_state(
        state_owned,
        domain,
        face_bc=face_bc,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
    )
    state_halo = prepared.state_halo
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    direct_stencil_builder = build_local_direct_stencil_one_sided_physical_from_halo
    phi_stencil = direct_stencil_builder(state_halo.phi, geometry, context)
    vorticity_stencil = direct_stencil_builder(state_halo.vorticity, geometry, context)
    current_halo = state_halo.density * (state_halo.Vi - state_halo.Ve)
    pressure_halo = (
        state_halo.density * state_halo.Te
        + parameters.tau * state_halo.density * state_halo.Ti
    )
    current_stencil = direct_stencil_builder(current_halo, geometry, context)
    pressure_stencil = direct_stencil_builder(pressure_halo, geometry, context)

    owned = domain.layout.owned_slices_cell
    density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
    Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
    density_safe = jnp.maximum(density, 1.0e-30)
    bmag = jnp.maximum(
        jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
        1.0e-30,
    )
    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)

    poisson_vorticity = local_poisson_bracket_op(
        phi_stencil,
        vorticity_stencil,
        geometry,
    )
    grad_parallel_vorticity = local_grad_parallel_op_direct(
        vorticity_stencil,
        geometry,
    )
    grad_parallel_current = local_grad_parallel_op_direct(current_stencil, geometry)
    curvature_pressure = local_curvature_op(
        pressure_stencil,
        geometry,
        curvature_coefficients=curvature_coefficients_owned,
    )
    return (
        -(poisson_vorticity / (rho_star * bmag))
        - Vi * grad_parallel_vorticity
        + (bmag * bmag / density_safe) * grad_parallel_current
        + (2.0 * bmag / density_safe) * curvature_pressure
    )


def _build_local_eb_stage_data(
    invariants: _ShiftedTorusEbInvariantBundle,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    time: float | jax.Array,
    *,
    rho_min: float,
    rho_max: float,
    parameters: FciDrbEBRhsParameters,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    gmres_config: SpmdGmresConfig,
) -> _ShiftedTorusEbStageData:
    del gmres_config
    data = _local_analytic_data(
        invariants.coordinates,
        time,
        rho_min=rho_min,
        rho_max=rho_max,
        parameters=parameters,
    )
    face_data = _local_analytic_data(
        invariants.face_coordinates,
        time,
        rho_min=rho_min,
        rho_max=rho_max,
        parameters=parameters,
    )
    owned = domain.layout.owned_slices_cell
    face_lower, face_upper = _split_local_x_face_state(_local_exact_state_from_data(face_data))
    face_derivative_lower, face_derivative_upper = _split_local_x_face_state(
        _local_exact_time_derivative_from_data(face_data)
    )
    phi_face_bc = _build_local_radial_dirichlet_face_bc_from_values(
        face_lower.phi,
        face_upper.phi,
        domain,
    )
    Ti_face_bc = _build_local_radial_dirichlet_face_bc_from_values(
        face_lower.Ti,
        face_upper.Ti,
        domain,
    )
    phi_t_face_bc = _build_local_radial_dirichlet_face_bc_from_values(
        face_derivative_lower.phi,
        face_derivative_upper.phi,
        domain,
    )
    Ti_t_face_bc = _build_local_radial_dirichlet_face_bc_from_values(
        face_derivative_lower.Ti,
        face_derivative_upper.Ti,
        domain,
    )
    face_projectors = (
        invariants.face_projector_x,
        invariants.face_projector_y,
        invariants.face_projector_z,
    )
    tau = jnp.asarray(parameters.tau, dtype=jnp.float64)
    phi_laplacian_owned = _local_perp_laplacian_from_owned_field(
        jnp.asarray(data.phi[owned], dtype=jnp.float64),
        geometry,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        face_projectors=face_projectors,
        face_bc=phi_face_bc,
    )
    Ti_laplacian_owned = _local_perp_laplacian_from_owned_field(
        jnp.asarray(data.Ti[owned], dtype=jnp.float64),
        geometry,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        face_projectors=face_projectors,
        face_bc=Ti_face_bc,
    )
    vorticity_owned = phi_laplacian_owned + tau * Ti_laplacian_owned
    phi_t_laplacian_owned = _local_perp_laplacian_from_owned_field(
        jnp.asarray(data.phi_t[owned], dtype=jnp.float64),
        geometry,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        face_projectors=face_projectors,
        face_bc=phi_t_face_bc,
    )
    Ti_t_laplacian_owned = _local_perp_laplacian_from_owned_field(
        jnp.asarray(data.Ti_t[owned], dtype=jnp.float64),
        geometry,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        face_projectors=face_projectors,
        face_bc=Ti_t_face_bc,
    )
    vorticity_t_owned = phi_t_laplacian_owned + tau * Ti_t_laplacian_owned
    exact_state_owned = _local_owned_state_from_halo_data(
        data,
        vorticity_owned,
        domain,
    )
    exact_time_derivative_owned = _local_exact_time_derivative_owned_from_halo_data(
        data,
        vorticity_t_owned,
        domain,
    )
    analytic_rhs_halo = _local_analytic_eb_rhs_from_invariants(
        data,
        invariants,
        parameters,
    )
    analytic_rhs_owned = _owned_state_from_halo_state(analytic_rhs_halo, domain)
    source_owned = _subtract_state(exact_time_derivative_owned, analytic_rhs_owned)
    source_owned = source_owned.replace(phi=jnp.zeros_like(source_owned.phi))
    face_bc_bundle = _ShiftedTorusEbFaceBCBundle(
        density=_build_local_radial_dirichlet_face_bc_from_values(
            face_lower.density,
            face_upper.density,
            domain,
        ),
        phi=phi_face_bc,
        Te=_build_local_radial_dirichlet_face_bc_from_values(
            face_lower.Te,
            face_upper.Te,
            domain,
        ),
        Ti=Ti_face_bc,
        Vi=_build_local_radial_dirichlet_face_bc_from_values(
            face_lower.Vi,
            face_upper.Vi,
            domain,
        ),
        Ve=_build_local_radial_dirichlet_face_bc_from_values(
            face_lower.Ve,
            face_upper.Ve,
            domain,
        ),
        vorticity=_build_local_radial_dirichlet_face_bc_from_values(
            face_lower.vorticity,
            face_upper.vorticity,
            domain,
        ),
    )
    discrete_vorticity_rhs_owned = _local_discrete_vorticity_rhs_from_exact_state(
        exact_state_owned,
        geometry,
        domain,
        face_bc=face_bc_bundle,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
        physical_ghost_filler=physical_ghost_filler,
        parameters=parameters,
        curvature_coefficients_owned=invariants.curvature_coefficients_owned,
    )
    source_owned = source_owned.replace(
        vorticity=vorticity_t_owned - discrete_vorticity_rhs_owned,
    )
    return _ShiftedTorusEbStageData(
        exact_halo=inject_owned_state_to_halo(exact_state_owned, domain.layout),
        source_halo=inject_owned_state_to_halo(source_owned, domain.layout),
        face_lower=face_lower,
        face_upper=face_upper,
    )


def _build_local_eb_rk4_stage_data(
    invariants: _ShiftedTorusEbInvariantBundle,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    step_time: float | jax.Array,
    step_timestep: float | jax.Array,
    *,
    rho_min: float,
    rho_max: float,
    parameters: FciDrbEBRhsParameters,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    gmres_config: SpmdGmresConfig,
) -> _ShiftedTorusEbRk4StageData:
    half_step = 0.5 * step_timestep
    return _ShiftedTorusEbRk4StageData(
        stage_1=_build_local_eb_stage_data(
            invariants,
            geometry,
            domain,
            step_time,
            rho_min=rho_min,
            rho_max=rho_max,
            parameters=parameters,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            gmres_config=gmres_config,
        ),
        stage_2=_build_local_eb_stage_data(
            invariants,
            geometry,
            domain,
            step_time + half_step,
            rho_min=rho_min,
            rho_max=rho_max,
            parameters=parameters,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            gmres_config=gmres_config,
        ),
        stage_3=_build_local_eb_stage_data(
            invariants,
            geometry,
            domain,
            step_time + half_step,
            rho_min=rho_min,
            rho_max=rho_max,
            parameters=parameters,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            gmres_config=gmres_config,
        ),
        stage_4=_build_local_eb_stage_data(
            invariants,
            geometry,
            domain,
            step_time + step_timestep,
            rho_min=rho_min,
            rho_max=rho_max,
            parameters=parameters,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            gmres_config=gmres_config,
        ),
    )


def _local_exact_state_from_data(data: _AnalyticMmsData) -> FciDrbEBState:
    return FciDrbEBState(
        density=data.density,
        phi=data.phi,
        Te=data.Te,
        Ti=data.Ti,
        Vi=data.Vi,
        Ve=data.Ve,
        vorticity=data.vorticity,
    )


def _local_exact_time_derivative_from_data(data: _AnalyticMmsData) -> FciDrbEBState:
    return FciDrbEBState(
        density=data.density_t,
        phi=data.phi_t,
        Te=data.Te_t,
        Ti=data.Ti_t,
        Vi=data.Vi_t,
        Ve=data.Ve_t,
        vorticity=data.vorticity_t,
    )


def _local_analytic_eb_rhs_from_data(
    data: _AnalyticMmsData,
    geometry: LocalFciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients_halo: jnp.ndarray,
) -> FciDrbEBState:
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag_halo, dtype=jnp.float64), 1.0e-30)
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
    jacobian = jnp.maximum(jnp.asarray(geometry.cell_metric.J_halo, dtype=jnp.float64), 1.0e-30)

    def _poisson(left_grad: jnp.ndarray, right_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_cov * jnp.cross(left_grad, right_grad), axis=-1) / jacobian

    def _grad_parallel(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", b_contra, field_grad)

    def _curvature(field_grad: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...i,...i->...", curvature_coefficients_halo, field_grad)

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
    del pressure, current_density

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
            Ve_nu * data.density * (data.Vi - data.Ve)
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


def _local_mms_source_state(
    data: _AnalyticMmsData,
    geometry: LocalFciGeometry3D,
    parameters: FciDrbEBRhsParameters,
    curvature_coefficients_halo: jnp.ndarray,
) -> FciDrbEBState:
    invariants = _ShiftedTorusEbInvariantBundle(
        coordinates=jnp.zeros(data.density.shape + (3,), dtype=jnp.float64),
        face_coordinates=jnp.zeros((2,) + data.density.shape[1:] + (3,), dtype=jnp.float64),
        bmag_halo=jnp.asarray(geometry.cell_bfield.Bmag_halo, dtype=jnp.float64),
        b_contra_halo=jnp.asarray(geometry.cell_bfield.b_contra, dtype=jnp.float64),
        cell_metric_g_cov_halo=jnp.asarray(geometry.cell_metric.g_cov, dtype=jnp.float64),
        cell_metric_jacobian_halo=jnp.asarray(geometry.cell_metric.J_halo, dtype=jnp.float64),
        curvature_coefficients_owned=jnp.asarray(
            curvature_coefficients_halo[geometry.layout.owned_slices_cell + (slice(None),)],
            dtype=jnp.float64,
        ),
        curvature_coefficients_halo=jnp.asarray(curvature_coefficients_halo, dtype=jnp.float64),
        face_projector_x=jnp.zeros(geometry.layout.location_owned_shape("x_face") + (3, 3), dtype=jnp.float64),
        face_projector_y=jnp.zeros(geometry.layout.location_owned_shape("y_face") + (3, 3), dtype=jnp.float64),
        face_projector_z=jnp.zeros(geometry.layout.location_owned_shape("z_face") + (3, 3), dtype=jnp.float64),
    )
    return _local_mms_source_state_from_invariants(data, invariants, parameters)


def _local_mms_source_state_from_invariants(
    data: _AnalyticMmsData,
    invariants: _ShiftedTorusEbInvariantBundle,
    parameters: FciDrbEBRhsParameters,
) -> FciDrbEBState:
    exact_t = _local_exact_time_derivative_from_data(data)
    rhs = _local_analytic_eb_rhs_from_invariants(data, invariants, parameters)
    return _subtract_state(exact_t, rhs)


def _state_partition_spec() -> FciDrbEBState:
    spec = P(*MESH_AXIS_NAMES)
    return FciDrbEBState(
        density=spec,
        phi=spec,
        Te=spec,
        Ti=spec,
        Vi=spec,
        Ve=spec,
        vorticity=spec,
    )


def _ve_term_partition_spec() -> _VeTermResidualBundle:
    spec = P(*MESH_AXIS_NAMES)
    return _VeTermResidualBundle(
        poisson=spec,
        parallel_advection=spec,
        collision=spec,
        grad_phi=spec,
        grad_pe=spec,
        grad_te=spec,
        source=spec,
        total=spec,
    )


def _phi_closure_partition_spec() -> _PhiClosureDiagnosticBundle:
    spec = P(*MESH_AXIS_NAMES)
    return _PhiClosureDiagnosticBundle(
        phi=spec,
        grad_parallel_phi=spec,
        closure_rhs=spec,
    )


def _put_state_on_mesh(state: FciDrbEBState, mesh: Mesh) -> FciDrbEBState:
    sharding = NamedSharding(mesh, P(*MESH_AXIS_NAMES))
    return FciDrbEBState(
        density=jax.device_put(jnp.asarray(state.density, dtype=jnp.float64), sharding),
        phi=jax.device_put(jnp.asarray(state.phi, dtype=jnp.float64), sharding),
        Te=jax.device_put(jnp.asarray(state.Te, dtype=jnp.float64), sharding),
        Ti=jax.device_put(jnp.asarray(state.Ti, dtype=jnp.float64), sharding),
        Vi=jax.device_put(jnp.asarray(state.Vi, dtype=jnp.float64), sharding),
        Ve=jax.device_put(jnp.asarray(state.Ve, dtype=jnp.float64), sharding),
        vorticity=jax.device_put(jnp.asarray(state.vorticity, dtype=jnp.float64), sharding),
    )


def _gather_state_from_mesh(state: FciDrbEBState) -> FciDrbEBState:
    return FciDrbEBState(
        density=jnp.asarray(jax.device_get(state.density), dtype=jnp.float64),
        phi=jnp.asarray(jax.device_get(state.phi), dtype=jnp.float64),
        Te=jnp.asarray(jax.device_get(state.Te), dtype=jnp.float64),
        Ti=jnp.asarray(jax.device_get(state.Ti), dtype=jnp.float64),
        Vi=jnp.asarray(jax.device_get(state.Vi), dtype=jnp.float64),
        Ve=jnp.asarray(jax.device_get(state.Ve), dtype=jnp.float64),
        vorticity=jnp.asarray(jax.device_get(state.vorticity), dtype=jnp.float64),
    )


def _zero_state_like(state: FciDrbEBState) -> FciDrbEBState:
    return FciDrbEBState(
        density=jnp.zeros_like(state.density),
        phi=jnp.zeros_like(state.phi),
        Te=jnp.zeros_like(state.Te),
        Ti=jnp.zeros_like(state.Ti),
        Vi=jnp.zeros_like(state.Vi),
        Ve=jnp.zeros_like(state.Ve),
        vorticity=jnp.zeros_like(state.vorticity),
    )


def _build_ghost_filler(halo_width: int) -> PhysicalGhostCellFiller3D:
    dirichlet = GhostFillWeights1D(
        owned_weights=jnp.full((halo_width, 1), -1.0, dtype=jnp.float64),
        bc_weights=jnp.full((halo_width,), 2.0, dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _build_local_x_face_bc(
    domain: LocalDomain3D,
    *,
    kind: int,
    value: float = 0.0,
) -> LocalBoundaryFaceBC3D:
    layout = domain.layout
    kind_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.int32)
    kind_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.int32)
    kind_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.int32)
    value_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.float64)
    value_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.float64)
    value_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.float64)
    mask_x = jnp.zeros(layout.face_control_shape(0), dtype=bool)
    mask_y = jnp.zeros(layout.face_control_shape(1), dtype=bool)
    mask_z = jnp.zeros(layout.face_control_shape(2), dtype=bool)
    kind_x = kind_x.at[0].set(int(kind)).at[-1].set(int(kind))
    value_x = value_x.at[0].set(float(value)).at[-1].set(float(value))
    mask_x = mask_x.at[0].set(domain.runtime_has_physical_lower(0)).at[-1].set(
        domain.runtime_has_physical_upper(0)
    )
    return LocalBoundaryFaceBC3D(
        kind_x=kind_x,
        kind_y=kind_y,
        kind_z=kind_z,
        value_x=value_x,
        value_y=value_y,
        value_z=value_z,
        mask_x=mask_x,
        mask_y=mask_y,
        mask_z=mask_z,
        layout=layout,
    )


def _build_local_radial_dirichlet_face_bc_from_values(
    lower_x: jnp.ndarray,
    upper_x: jnp.ndarray,
    domain: LocalDomain3D,
) -> LocalBoundaryFaceBC3D:
    layout = domain.layout
    kind_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.int32)
    kind_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.int32)
    kind_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.int32)
    value_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.float64)
    value_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.float64)
    value_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.float64)
    mask_x = jnp.zeros(layout.face_control_shape(0), dtype=bool)
    mask_y = jnp.zeros(layout.face_control_shape(1), dtype=bool)
    mask_z = jnp.zeros(layout.face_control_shape(2), dtype=bool)

    kind_x = kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET)
    value_x = value_x.at[0].set(lower_x).at[-1].set(upper_x)
    mask_x = mask_x.at[0].set(domain.runtime_has_physical_lower(0)).at[-1].set(
        domain.runtime_has_physical_upper(0)
    )
    return LocalBoundaryFaceBC3D(
        kind_x=kind_x,
        kind_y=kind_y,
        kind_z=kind_z,
        value_x=value_x,
        value_y=value_y,
        value_z=value_z,
        mask_x=mask_x,
        mask_y=mask_y,
        mask_z=mask_z,
        layout=layout,
    )


def _build_local_eb_face_bcs(
    stage_data: _ShiftedTorusEbStageData,
    domain: LocalDomain3D,
) -> _ShiftedTorusEbFaceBCBundle:
    lower = stage_data.face_lower
    upper = stage_data.face_upper
    return _ShiftedTorusEbFaceBCBundle(
        density=_build_local_radial_dirichlet_face_bc_from_values(
            lower.density,
            upper.density,
            domain,
        ),
        phi=_build_local_radial_dirichlet_face_bc_from_values(lower.phi, upper.phi, domain),
        Te=_build_local_radial_dirichlet_face_bc_from_values(lower.Te, upper.Te, domain),
        Ti=_build_local_radial_dirichlet_face_bc_from_values(lower.Ti, upper.Ti, domain),
        Vi=_build_local_radial_dirichlet_face_bc_from_values(lower.Vi, upper.Vi, domain),
        Ve=_build_local_radial_dirichlet_face_bc_from_values(lower.Ve, upper.Ve, domain),
        vorticity=_build_local_radial_dirichlet_face_bc_from_values(
            lower.vorticity,
            upper.vorticity,
            domain,
        ),
    )


def _prepare_local_eb_stage_state(
    state_owned: FciDrbEBState,
    domain: LocalDomain3D,
    *,
    face_bc: _ShiftedTorusEbFaceBCBundle,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
) -> PreparedLocalState3D:
    state_halo = inject_owned_state_to_halo(state_owned, domain.layout)
    state_halo = FciDrbEBState(
        density=topology_filler(halo_exchange(state_halo.density, domain), domain),
        phi=topology_filler(halo_exchange(state_halo.phi, domain), domain),
        Te=topology_filler(halo_exchange(state_halo.Te, domain), domain),
        Ti=topology_filler(halo_exchange(state_halo.Ti, domain), domain),
        Vi=topology_filler(halo_exchange(state_halo.Vi, domain), domain),
        Ve=topology_filler(halo_exchange(state_halo.Ve, domain), domain),
        vorticity=topology_filler(halo_exchange(state_halo.vorticity, domain), domain),
    )
    state_halo = FciDrbEBState(
        density=physical_ghost_filler(state_halo.density, domain, face_bc.density),
        phi=physical_ghost_filler(state_halo.phi, domain, face_bc.phi),
        Te=physical_ghost_filler(state_halo.Te, domain, face_bc.Te),
        Ti=physical_ghost_filler(state_halo.Ti, domain, face_bc.Ti),
        Vi=physical_ghost_filler(state_halo.Vi, domain, face_bc.Vi),
        Ve=physical_ghost_filler(state_halo.Ve, domain, face_bc.Ve),
        vorticity=physical_ghost_filler(
            state_halo.vorticity,
            domain,
            face_bc.vorticity,
        ),
    )
    return PreparedLocalState3D(
        state_halo=state_halo,
        boundary_data=LocalBoundaryData3D(face_bc=face_bc),
    )


@dataclass(frozen=True)
class LocalShiftedTorusEbRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: FciDrbEBRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    gmres_config: SpmdGmresConfig

    def _prepare_phi_halo(
        self,
        phi_owned: jnp.ndarray,
        face_bc: LocalBoundaryFaceBC3D,
    ) -> jnp.ndarray:
        phi_halo = inject_owned_field_to_halo(phi_owned, self.domain.layout)
        phi_halo = self.topology_filler(
            self.halo_exchange(phi_halo, self.domain),
            self.domain,
        )
        return self.physical_ghost_filler(phi_halo, self.domain, face_bc)

    def reconstruct_phi(
        self,
        state_owned: FciDrbEBState,
        stage_data: _ShiftedTorusEbStageData,
    ) -> jnp.ndarray:
        face_bc_bundle = _build_local_eb_face_bcs(stage_data, self.domain)
        prepared = _prepare_local_eb_stage_state(
            state_owned,
            self.domain,
            face_bc=face_bc_bundle,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        state_halo = prepared.state_halo
        face_bc = prepared.boundary_data.face_bc
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        Ti_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ti,
            self.geometry,
            context,
        )
        Ti_laplacian = local_perp_laplacian_conservative_op(
            Ti_conservative_stencil,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )
        vorticity = jnp.asarray(
            state_halo.vorticity[self.domain.layout.owned_slices_cell],
            dtype=jnp.float64,
        )
        tau = jnp.asarray(self.parameters.tau, dtype=jnp.float64)
        phi_solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            regular_face_geometry=self.geometry.regular_face_geometry,
            face_bc=face_bc.phi,
            config=self.gmres_config,
        )
        return phi_solver(
            tau * Ti_laplacian - vorticity,
            guess_owned=state_owned.phi,
        )

    def evaluate_stage(
        self,
        state_owned: FciDrbEBState,
        stage_data: _ShiftedTorusEbStageData,
        carry: None,
    ) -> tuple[FciDrbEBState, None, jnp.ndarray]:
        del carry
        face_bc_bundle = _build_local_eb_face_bcs(stage_data, self.domain)
        prepared = _prepare_local_eb_stage_state(
            state_owned,
            self.domain,
            face_bc=face_bc_bundle,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        state_halo = prepared.state_halo
        face_bc = prepared.boundary_data.face_bc
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        direct_stencil_builder = build_local_direct_stencil_one_sided_physical_from_halo
        density_stencil = direct_stencil_builder(state_halo.density, self.geometry, context)
        Te_stencil = direct_stencil_builder(state_halo.Te, self.geometry, context)
        Ti_stencil = direct_stencil_builder(state_halo.Ti, self.geometry, context)
        Vi_stencil = direct_stencil_builder(state_halo.Vi, self.geometry, context)
        Ve_stencil = direct_stencil_builder(state_halo.Ve, self.geometry, context)
        vorticity_stencil = direct_stencil_builder(state_halo.vorticity, self.geometry, context)

        Pe_halo = state_halo.density * state_halo.Te
        pressure_halo = Pe_halo + self.parameters.tau * state_halo.density * state_halo.Ti
        current_halo = state_halo.density * (state_halo.Vi - state_halo.Ve)
        density_flux_halo = state_halo.density * state_halo.Ve
        Pe_stencil = direct_stencil_builder(Pe_halo, self.geometry, context)
        pressure_stencil = direct_stencil_builder(pressure_halo, self.geometry, context)
        current_stencil = direct_stencil_builder(current_halo, self.geometry, context)
        density_flux_conservative_stencil = build_local_conservative_stencil_from_field(
            density_flux_halo,
            self.geometry,
            context,
        )
        current_conservative_stencil = build_local_conservative_stencil_from_field(
            current_halo,
            self.geometry,
            context,
        )
        Ve_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ve,
            self.geometry,
            context,
        )
        Vi_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Vi,
            self.geometry,
            context,
        )

        Ti_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.Ti,
            self.geometry,
            context,
        )
        Ti_laplacian = local_perp_laplacian_conservative_op(
            Ti_conservative_stencil,
            self.geometry,
            self.domain,
            face_projectors=self.face_projectors,
            face_bc=face_bc.Ti,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )

        owned = self.domain.layout.owned_slices_cell
        density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
        Te = jnp.asarray(state_halo.Te[owned], dtype=jnp.float64)
        Ti = jnp.asarray(state_halo.Ti[owned], dtype=jnp.float64)
        Vi = jnp.asarray(state_halo.Vi[owned], dtype=jnp.float64)
        Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
        vorticity = jnp.asarray(state_halo.vorticity[owned], dtype=jnp.float64)
        current = density * (Vi - Ve)
        density_safe = jnp.maximum(density, 1.0e-30)
        bmag = jnp.maximum(jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64), 1.0e-30)
        rho_star = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        tau = jnp.asarray(self.parameters.tau, dtype=jnp.float64)
        mi_over_me = jnp.asarray(self.parameters.mi_over_me, dtype=jnp.float64)
        Ve_nu = jnp.asarray(self.parameters.Ve_nu, dtype=jnp.float64)

        phi_solver = LocalPerpLaplacianInverseSolver(
            geometry=self.geometry,
            domain=self.domain,
            stencil_builder=build_local_conservative_stencil_from_field,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
            face_projectors=self.face_projectors,
            regular_face_geometry=self.geometry.regular_face_geometry,
            face_bc=face_bc.phi,
            config=self.gmres_config,
        )
        phi_owned = phi_solver(
            tau * Ti_laplacian - vorticity,
            guess_owned=state_owned.phi,
        )
        phi_halo = self._prepare_phi_halo(phi_owned, face_bc.phi)
        phi_stencil = direct_stencil_builder(phi_halo, self.geometry, context)

        poisson_density = local_poisson_bracket_op(phi_stencil, density_stencil, self.geometry)
        poisson_Te = local_poisson_bracket_op(phi_stencil, Te_stencil, self.geometry)
        poisson_Ti = local_poisson_bracket_op(phi_stencil, Ti_stencil, self.geometry)
        poisson_Vi = local_poisson_bracket_op(phi_stencil, Vi_stencil, self.geometry)
        poisson_Ve = local_poisson_bracket_op(phi_stencil, Ve_stencil, self.geometry)
        poisson_vorticity = local_poisson_bracket_op(phi_stencil, vorticity_stencil, self.geometry)

        curvature_Pe = local_curvature_op(
            Pe_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_pressure = local_curvature_op(
            pressure_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_Te = local_curvature_op(
            Te_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_Ti = local_curvature_op(
            Ti_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        parallel_density_flux_divergence = local_parallel_flux_div_op(
            density_flux_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )
        parallel_current_flux_divergence = local_parallel_flux_div_op(
            current_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )
        parallel_Ve_flux_divergence = local_parallel_flux_div_op(
            Ve_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )
        parallel_Vi_flux_divergence = local_parallel_flux_div_op(
            Vi_conservative_stencil,
            self.geometry,
            self.domain,
            regular_face_geometry=self.geometry.regular_face_geometry,
        )
        Ve_parallel_diff = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        if float(self.parameters.Ve_parallel_viscosity) != 0.0:
            Ve_parallel_diff = jnp.asarray(
                self.parameters.Ve_parallel_viscosity,
                dtype=jnp.float64,
            ) * local_parallel_laplacian_conservative_op(
                Ve_conservative_stencil,
                self.geometry,
                self.domain,
                face_bc=face_bc.Ve,
                regular_face_geometry=self.geometry.regular_face_geometry,
            )
        Vi_parallel_diff = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        if float(self.parameters.Vi_parallel_viscosity) != 0.0:
            Vi_parallel_diff = jnp.asarray(
                self.parameters.Vi_parallel_viscosity,
                dtype=jnp.float64,
            ) * local_parallel_laplacian_conservative_op(
                Vi_conservative_stencil,
                self.geometry,
                self.domain,
                face_bc=face_bc.Vi,
                regular_face_geometry=self.geometry.regular_face_geometry,
            )
        vorticity_conservative_stencil = build_local_conservative_stencil_from_field(
            state_halo.vorticity,
            self.geometry,
            context,
        )
        vorticity_parallel_diff = jnp.zeros(self.geometry.owned_shape, dtype=jnp.float64)
        if float(self.parameters.vorticity_D_parallel) != 0.0:
            vorticity_parallel_diff = jnp.asarray(
                self.parameters.vorticity_D_parallel,
                dtype=jnp.float64,
            ) * local_parallel_laplacian_conservative_op(
                vorticity_conservative_stencil,
                self.geometry,
                self.domain,
                face_bc=face_bc.vorticity,
                regular_face_geometry=self.geometry.regular_face_geometry,
            )
        grad_parallel_Te = local_grad_parallel_op_direct(Te_stencil, self.geometry)
        grad_parallel_Ti = local_grad_parallel_op_direct(Ti_stencil, self.geometry)
        grad_parallel_Ve = local_grad_parallel_op_direct(Ve_stencil, self.geometry)
        grad_parallel_Vi = local_grad_parallel_op_direct(Vi_stencil, self.geometry)
        grad_parallel_phi = local_grad_parallel_op_direct(phi_stencil, self.geometry)
        grad_parallel_Pe = local_grad_parallel_op_direct(Pe_stencil, self.geometry)
        grad_parallel_pressure = local_grad_parallel_op_direct(pressure_stencil, self.geometry)
        grad_parallel_current = local_grad_parallel_op_direct(current_stencil, self.geometry)
        grad_parallel_vorticity = local_grad_parallel_op_direct(vorticity_stencil, self.geometry)

        density_rhs = (
            -(poisson_density / (rho_star * bmag))
            - parallel_density_flux_divergence
            + (2.0 / bmag) * (curvature_Pe - density * curvature_phi)
        )
        Te_rhs = (
            -(poisson_Te / (rho_star * bmag))
            - Ve * grad_parallel_Te
            + (4.0 * Te / (3.0 * bmag))
            * (curvature_Pe / density_safe + 2.5 * curvature_Te - curvature_phi)
            + (2.0 * Te / (3.0 * density_safe))
            * (0.71 * parallel_current_flux_divergence - density * parallel_Ve_flux_divergence)
        )
        Ti_rhs = (
            -(poisson_Ti / (rho_star * bmag))
            - Vi * grad_parallel_Ti
            + (4.0 * Ti / (3.0 * bmag))
            * (curvature_Pe / density_safe - 2.5 * tau * curvature_Ti - curvature_phi)
            + (2.0 * Ti / (3.0 * density_safe))
            * (parallel_current_flux_divergence - density * parallel_Vi_flux_divergence)
        )
        Vi_rhs = (
            -(poisson_Vi / (rho_star * bmag))
            - Vi * grad_parallel_Vi
            - grad_parallel_pressure / density_safe
            + Vi_parallel_diff
        )
        Ve_rhs = (
            -(poisson_Ve / (rho_star * bmag))
            - Ve * grad_parallel_Ve
            + mi_over_me
            * (
                Ve_nu * current
                + grad_parallel_phi
                - grad_parallel_Pe / density_safe
                - 0.71 * grad_parallel_Te
            )
            + Ve_parallel_diff
        )
        vorticity_rhs = (
            -(poisson_vorticity / (rho_star * bmag))
            - Vi * grad_parallel_vorticity
            + (bmag * bmag / density_safe) * parallel_current_flux_divergence
            + (2.0 * bmag / density_safe) * curvature_pressure
            + vorticity_parallel_diff
        )
        source = stage_data.source_halo
        rhs = FciDrbEBState(
            density=density_rhs + source.density[owned],
            phi=jnp.zeros_like(phi_owned) + source.phi[owned],
            Te=Te_rhs + source.Te[owned],
            Ti=Ti_rhs + source.Ti[owned],
            Vi=Vi_rhs + source.Vi[owned],
            Ve=Ve_rhs + source.Ve[owned],
            vorticity=vorticity_rhs + source.vorticity[owned],
        )
        return rhs, None, jnp.zeros((1,), dtype=jnp.float64)


def simulate_mms_shifted_torus_eb(
    context: ShiftedTorusEbMmsContext,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    final_time: float = TF,
    timestep: float | None = None,
    num_steps: int = NUM_STEPS,
    show_progress: bool = False,
    return_exact: bool = False,
) -> tuple[FciDrbEBState, jnp.ndarray] | tuple[FciDrbEBState, jnp.ndarray, FciDrbEBState]:
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(context.geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(context.geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(
        context.geometry.shape,
        halo_width,
        shard_counts,
    )
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    gmres_config = SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(context.parameters.phi_inversion_iterations),
        restart=min(100, int(context.parameters.phi_inversion_iterations)),
        acceptance_tol=1.0e-4,
        acceptance_atol=1.0e-4,
        regularization_epsilon=float(context.parameters.phi_inversion_regularization),
    )
    rho_min, rho_max = _mms_radial_bounds(context.geometry)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state_spec = _state_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_eb_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=context.geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(_sample_local_eb_rk4_stage_data(domain))
        )

        def invariant_kernel() -> _ShiftedTorusEbInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_eb_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=context.geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorusEbInvariantBundle,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> _ShiftedTorusEbRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            return expand_local_shard_pytree(
                _build_local_eb_rk4_stage_data(
                    local_invariants,
                    local_geometry,
                    domain,
                    step_time,
                    step_timestep,
                    rho_min=rho_min,
                    rho_max=rho_max,
                    parameters=context.parameters,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=ghost_filler,
                    gmres_config=gmres_config,
                )
            )

        def kernel(
            state_owned: FciDrbEBState,
            local_invariants: _ShiftedTorusEbInvariantBundle,
            rk_stage_data: _ShiftedTorusEbRk4StageData,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> FciDrbEBState:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            rhs = LocalShiftedTorusEbRhs(
                geometry=local_geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                parameters=context.parameters,
                curvature_coefficients_owned=local_invariants.curvature_coefficients_owned,
                face_projectors=(
                    local_invariants.face_projector_x,
                    local_invariants.face_projector_y,
                    local_invariants.face_projector_z,
                ),
                gmres_config=gmres_config,
            )
            k1, _, _ = rhs.evaluate_stage(state_owned, rk_stage_data.stage_1, None)
            stage_1 = state_owned.axpy(k1, scale=0.5 * step_timestep)
            k2, _, _ = rhs.evaluate_stage(stage_1, rk_stage_data.stage_2, None)
            stage_2 = state_owned.axpy(k2, scale=0.5 * step_timestep)
            k3, _, _ = rhs.evaluate_stage(stage_2, rk_stage_data.stage_3, None)
            stage_3 = state_owned.axpy(k3, scale=step_timestep)
            k4, _, _ = rhs.evaluate_stage(stage_3, rk_stage_data.stage_4, None)
            next_state = state_owned.axpy(
                k1.axpy(k2, scale=2.0).axpy(k3, scale=2.0).axpy(k4, scale=1.0),
                scale=step_timestep / 6.0,
            )
            next_phi = rhs.reconstruct_phi(next_state, rk_stage_data.stage_4)
            next_state = next_state.replace(phi=next_phi)
            return next_state

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P(), P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        compiled_source_kernel = jax.jit(mapped_source_kernel)

        def exact_state_kernel(
            rk_stage_data: _ShiftedTorusEbRk4StageData,
        ) -> FciDrbEBState:
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            return _owned_state_from_halo_state(
                rk_stage_data.stage_1.exact_halo,
                domain,
            )

        mapped_exact_state_kernel = shard_map(
            exact_state_kernel,
            mesh=mesh,
            in_specs=(stage_data_spec,),
            out_specs=state_spec,
            check_rep=False,
        )
        exact_state_kernel_jit = jax.jit(mapped_exact_state_kernel)
        mapped_step_kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(state_spec, invariant_spec, stage_data_spec, P(), P()),
            out_specs=state_spec,
            check_rep=False,
        )
        step_kernel = jax.jit(mapped_step_kernel)

        time_value = 0.0
        initial_stage_data = compiled_source_kernel(
            invariants,
            jnp.asarray(time_value, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
        )
        state = exact_state_kernel_jit(initial_stage_data)
        jax.block_until_ready(state.density)
        step_kernel = step_kernel.lower(
            state,
            invariants,
            initial_stage_data,
            jnp.asarray(time_value, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
        ).compile()
        progress_start = time_module.perf_counter()
        source_total = 0.0
        step_total = 0.0
        if show_progress:
            print(
                f"shifted_torus_EB MMS RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

        for step_index in range(steps):
            source_start = time_module.perf_counter()
            rk_stage_data = compiled_source_kernel(
                invariants,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(rk_stage_data.stage_1.exact_halo.density)
            source_total += time_module.perf_counter() - source_start

            step_start = time_module.perf_counter()
            state = step_kernel(
                state,
                invariants,
                rk_stage_data,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            step_total += time_module.perf_counter() - step_start
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
            mean_source = source_total / float(max(steps, 1))
            mean_step = step_total / float(max(steps, 1))
            print(
                "shifted_torus_EB mean timings per RK step: "
                f"source={mean_source:.6e} s, step={mean_step:.6e} s, "
                f"total={mean_source + mean_step:.6e} s"
            )
        final_state = _gather_state_from_mesh(state)
        if return_exact:
            final_stage_data = compiled_source_kernel(
                invariants,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            exact_state = _gather_state_from_mesh(exact_state_kernel_jit(final_stage_data))
    if return_exact:
        return final_state, jnp.asarray(time_value, dtype=jnp.float64), exact_state
    return final_state, jnp.asarray(time_value, dtype=jnp.float64)


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


def _print_state_residual_statistics(label: str, stats: dict[str, tuple[float, float, float]]) -> None:
    print(label)
    for field_name, (l2, linf, _rel_l2) in stats.items():
        print(f"  {field_name}: weighted_l2={l2:.6e}, linf={linf:.6e}")


def _combined_l2_error(stats: dict[str, tuple[float, float, float]]) -> float:
    return float(np.sqrt(np.mean([field_stats[0] ** 2 for field_stats in stats.values()])))


def _observed_order(error_coarse: float, error_fine: float, resolution_coarse: int, resolution_fine: int) -> float:
    if error_coarse <= 0.0 or error_fine <= 0.0:
        return float("nan")
    return float(np.log(error_coarse / error_fine) / np.log(float(resolution_fine) / float(resolution_coarse)))


def _domain_decomp_closure_error_statistics(
    state: FciDrbEBState,
    context: ShiftedTorusEbMmsContext,
    *,
    shard_counts: tuple[int, int, int],
    halo_width: int,
    time: float,
) -> tuple[float, float, float]:
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(context.geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(context.geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(
        context.geometry.shape,
        halo_width,
        shard_counts,
    )
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    gmres_config = SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(context.parameters.phi_inversion_iterations),
        restart=min(100, int(context.parameters.phi_inversion_iterations)),
        acceptance_tol=1.0e-4,
        acceptance_atol=1.0e-4,
        regularization_epsilon=float(context.parameters.phi_inversion_regularization),
    )
    rho_min, rho_max = _mms_radial_bounds(context.geometry)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        sharded_state = _put_state_on_mesh(state, mesh)
        state_spec = _state_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_eb_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=context.geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)

        def invariant_kernel() -> _ShiftedTorusEbInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_eb_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=context.geometry.shape,
                    domain=domain,
                )
            )

        def closure_kernel(
            state_owned: FciDrbEBState,
            local_invariants: _ShiftedTorusEbInvariantBundle,
            time_value: jax.Array,
        ) -> jnp.ndarray:
            local_invariants = extract_local_shard_pytree(local_invariants)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            stage_data = _build_local_eb_stage_data(
                local_invariants,
                local_geometry,
                domain,
                time_value,
                rho_min=rho_min,
                rho_max=rho_max,
                parameters=context.parameters,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                gmres_config=gmres_config,
            )
            face_bc_bundle = _build_local_eb_face_bcs(stage_data, domain)
            prepared = _prepare_local_eb_stage_state(
                state_owned,
                domain,
                face_bc=face_bc_bundle,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
            )
            state_halo = prepared.state_halo
            face_bc = prepared.boundary_data.face_bc
            stencil_context = StencilBuilderContext(layout=domain.layout, domain=domain)
            phi_stencil = build_local_conservative_stencil_from_field(
                state_halo.phi,
                local_geometry,
                stencil_context,
            )
            Ti_stencil = build_local_conservative_stencil_from_field(
                state_halo.Ti,
                local_geometry,
                stencil_context,
            )
            face_projectors = (
                local_invariants.face_projector_x,
                local_invariants.face_projector_y,
                local_invariants.face_projector_z,
            )
            phi_laplacian = local_perp_laplacian_conservative_op(
                phi_stencil,
                local_geometry,
                domain,
                face_projectors=face_projectors,
                face_bc=face_bc.phi,
                regular_face_geometry=local_geometry.regular_face_geometry,
            )
            Ti_laplacian = local_perp_laplacian_conservative_op(
                Ti_stencil,
                local_geometry,
                domain,
                face_projectors=face_projectors,
                face_bc=face_bc.Ti,
                regular_face_geometry=local_geometry.regular_face_geometry,
            )
            return phi_laplacian + context.parameters.tau * Ti_laplacian

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_closure_kernel = shard_map(
            closure_kernel,
            mesh=mesh,
            in_specs=(state_spec, invariant_spec, P()),
            out_specs=P(*MESH_AXIS_NAMES),
            check_rep=False,
        )
        closed_omega = jnp.asarray(
            jax.device_get(
                jax.jit(mapped_closure_kernel)(
                    sharded_state,
                    invariants,
                    jnp.asarray(time, dtype=jnp.float64),
                )
            ),
            dtype=jnp.float64,
        )
    return _field_error_statistics(state.vorticity, closed_omega, context.geometry)


def test_mms_shifted_torus_eb_exact_vorticity_uses_analytic_closure() -> None:
    context = build_shifted_torus_eb_mms_context((4, 6, 4))
    time = 0.013
    exact = _mms_exact_state(context, time)
    closed_vorticity, _, _ = _evaluate_vorticity_value_gradient_time(context, time)
    np.testing.assert_allclose(
        np.asarray(exact.vorticity),
        np.asarray(closed_vorticity),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_mms_shifted_torus_eb_vorticity_time_derivative_uses_analytic_closure() -> None:
    context = build_shifted_torus_eb_mms_context((4, 6, 4))
    time = 0.013
    exact_t = _mms_exact_time_derivative_state(context, time)
    _, _, closed_vorticity_t = _evaluate_vorticity_value_gradient_time(context, time)
    np.testing.assert_allclose(
        np.asarray(exact_t.vorticity),
        np.asarray(closed_vorticity_t),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _discrete_source_residual_l2(shape: tuple[int, int, int], time: float) -> float:
    context = build_shifted_torus_eb_mms_context(shape)
    final_state, _, exact_state = simulate_mms_shifted_torus_eb(
        context,
        final_time=time,
        timestep=time,
        num_steps=1,
        return_exact=True,
    )
    stats = _state_error_statistics(
        final_state,
        exact_state,
        context.geometry,
    )
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
    final_state, _, exact_state = simulate_mms_shifted_torus_eb(
        context,
        final_time=0.002,
        num_steps=2,
        return_exact=True,
    )
    stats = _state_error_statistics(final_state, exact_state, context.geometry)

    assert _combined_l2_error(stats) < 1.0e-2
    assert (
        _domain_decomp_closure_error_statistics(
            final_state,
            context,
            shard_counts=(1, 1, 1),
            halo_width=2,
            time=0.002,
        )[0]
        < 2.0e-1
    )


def test_mms_shifted_torus_eb_ve_term_diagnostic_source_consistency() -> None:
    context = build_shifted_torus_eb_mms_context((4, 4, 4))
    residuals = evaluate_shifted_torus_eb_ve_term_residuals(
        context,
        shard_counts=(1, 1, 1),
        halo_width=2,
        time=0.013,
        phi_mode="exact",
    )
    stats = _ve_term_error_statistics(residuals, context.geometry)

    assert np.isfinite(stats["total"][0])
    assert stats["source"][0] < 1.0e-12


def test_mms_shifted_torus_eb_discrete_phi_closure_is_self_consistent() -> None:
    context = build_shifted_torus_eb_mms_context((4, 4, 4))
    diagnostics = evaluate_shifted_torus_eb_phi_closure_diagnostic(
        context,
        shard_counts=(1, 1, 1),
        halo_width=2,
        time=0.013,
        closure_mode="discrete_omega",
    )
    stats = _phi_closure_error_statistics(diagnostics, context.geometry)

    assert stats["closure_rhs"][0] < 1.0e-12
    assert stats["phi"][0] < 1.0e-12
    assert stats["grad_parallel_phi"][0] < 1.0e-12


def evaluate_shifted_torus_eb_rhs_residual(
    context: ShiftedTorusEbMmsContext,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    time: float = 0.013,
) -> FciDrbEBState:
    """Evaluate ``R_discrete(U_exact) + S - d_t U_exact`` on the SPMD path."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(context.geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(context.geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(
        context.geometry.shape,
        halo_width,
        shard_counts,
    )
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    gmres_config = SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(context.parameters.phi_inversion_iterations),
        restart=min(100, int(context.parameters.phi_inversion_iterations)),
        acceptance_tol=1.0e-4,
        acceptance_atol=1.0e-4,
        regularization_epsilon=float(context.parameters.phi_inversion_regularization),
    )
    rho_min, rho_max = _mms_radial_bounds(context.geometry)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state_spec = _state_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_eb_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=context.geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(_sample_local_eb_rk4_stage_data(domain))
        )

        def invariant_kernel() -> _ShiftedTorusEbInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_eb_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=context.geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorusEbInvariantBundle,
            stage_time: jax.Array,
        ) -> _ShiftedTorusEbRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            return expand_local_shard_pytree(
                _build_local_eb_rk4_stage_data(
                    local_invariants,
                    local_geometry,
                    domain,
                    stage_time,
                    jnp.asarray(0.0, dtype=jnp.float64),
                    rho_min=rho_min,
                    rho_max=rho_max,
                    parameters=context.parameters,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=ghost_filler,
                    gmres_config=gmres_config,
                )
            )

        def exact_state_kernel(
            rk_stage_data: _ShiftedTorusEbRk4StageData,
        ) -> FciDrbEBState:
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            return _owned_state_from_halo_state(
                rk_stage_data.stage_1.exact_halo,
                domain,
            )

        def rhs_residual_kernel(
            state_owned: FciDrbEBState,
            local_invariants: _ShiftedTorusEbInvariantBundle,
            rk_stage_data: _ShiftedTorusEbRk4StageData,
            stage_time: jax.Array,
        ) -> FciDrbEBState:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            rhs = LocalShiftedTorusEbRhs(
                geometry=local_geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                parameters=context.parameters,
                curvature_coefficients_owned=local_invariants.curvature_coefficients_owned,
                face_projectors=(
                    local_invariants.face_projector_x,
                    local_invariants.face_projector_y,
                    local_invariants.face_projector_z,
                ),
                gmres_config=gmres_config,
            )
            actual_rhs, _, _ = rhs.evaluate_stage(
                state_owned,
                rk_stage_data.stage_1,
                None,
            )
            data = _local_analytic_data(
                local_invariants.coordinates,
                stage_time,
                rho_min=rho_min,
                rho_max=rho_max,
                parameters=context.parameters,
            )
            owned = domain.layout.owned_slices_cell
            expected_rhs = _local_exact_time_derivative_owned_from_halo_data(
                data,
                jnp.asarray(data.vorticity_t[owned], dtype=jnp.float64),
                domain,
            )
            expected_rhs = expected_rhs.replace(phi=jnp.zeros_like(expected_rhs.phi))
            return _subtract_state(actual_rhs, expected_rhs)

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        stage_data = jax.jit(mapped_source_kernel)(
            invariants,
            jnp.asarray(time, dtype=jnp.float64),
        )
        mapped_exact_state_kernel = shard_map(
            exact_state_kernel,
            mesh=mesh,
            in_specs=(stage_data_spec,),
            out_specs=state_spec,
            check_rep=False,
        )
        exact_state = jax.jit(mapped_exact_state_kernel)(stage_data)
        mapped_rhs_residual_kernel = shard_map(
            rhs_residual_kernel,
            mesh=mesh,
            in_specs=(state_spec, invariant_spec, stage_data_spec, P()),
            out_specs=state_spec,
            check_rep=False,
        )
        residual = jax.jit(mapped_rhs_residual_kernel)(
            exact_state,
            invariants,
            stage_data,
            jnp.asarray(time, dtype=jnp.float64),
        )
        jax.block_until_ready(residual.density)
        return _gather_state_from_mesh(residual)


def evaluate_shifted_torus_eb_ve_term_residuals(
    context: ShiftedTorusEbMmsContext,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    time: float = 0.013,
    phi_mode: str = "reconstructed",
) -> _VeTermResidualBundle:
    """Evaluate per-term ``Ve`` residuals on the exact MMS state."""

    if phi_mode not in {"reconstructed", "exact"}:
        raise ValueError(f"phi_mode must be 'reconstructed' or 'exact', got {phi_mode!r}")
    use_exact_phi = phi_mode == "exact"
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(context.geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(context.geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(
        context.geometry.shape,
        halo_width,
        shard_counts,
    )
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    gmres_config = SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(context.parameters.phi_inversion_iterations),
        restart=min(100, int(context.parameters.phi_inversion_iterations)),
        acceptance_tol=1.0e-4,
        acceptance_atol=1.0e-4,
        regularization_epsilon=float(context.parameters.phi_inversion_regularization),
    )
    rho_min, rho_max = _mms_radial_bounds(context.geometry)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state_spec = _state_partition_spec()
        ve_term_spec = _ve_term_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_eb_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=context.geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(_sample_local_eb_rk4_stage_data(domain))
        )

        def invariant_kernel() -> _ShiftedTorusEbInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_eb_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=context.geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorusEbInvariantBundle,
            stage_time: jax.Array,
        ) -> _ShiftedTorusEbRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            return expand_local_shard_pytree(
                _build_local_eb_rk4_stage_data(
                    local_invariants,
                    local_geometry,
                    domain,
                    stage_time,
                    jnp.asarray(0.0, dtype=jnp.float64),
                    rho_min=rho_min,
                    rho_max=rho_max,
                    parameters=context.parameters,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=ghost_filler,
                    gmres_config=gmres_config,
                )
            )

        def exact_state_kernel(
            rk_stage_data: _ShiftedTorusEbRk4StageData,
        ) -> FciDrbEBState:
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            return _owned_state_from_halo_state(
                rk_stage_data.stage_1.exact_halo,
                domain,
            )

        def ve_term_kernel(
            state_owned: FciDrbEBState,
            local_invariants: _ShiftedTorusEbInvariantBundle,
            rk_stage_data: _ShiftedTorusEbRk4StageData,
            stage_time: jax.Array,
        ) -> _VeTermResidualBundle:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            face_bc_bundle = _build_local_eb_face_bcs(rk_stage_data.stage_1, domain)
            prepared = _prepare_local_eb_stage_state(
                state_owned,
                domain,
                face_bc=face_bc_bundle,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
            )
            state_halo = prepared.state_halo
            face_bc = prepared.boundary_data.face_bc
            context_local = StencilBuilderContext(layout=domain.layout, domain=domain)
            direct_stencil_builder = build_local_direct_stencil_one_sided_physical_from_halo
            Ve_stencil = direct_stencil_builder(state_halo.Ve, local_geometry, context_local)
            Te_stencil = direct_stencil_builder(state_halo.Te, local_geometry, context_local)
            Pe_halo = state_halo.density * state_halo.Te
            current_halo = state_halo.density * (state_halo.Vi - state_halo.Ve)
            Pe_stencil = direct_stencil_builder(Pe_halo, local_geometry, context_local)

            Ti_conservative_stencil = build_local_conservative_stencil_from_field(
                state_halo.Ti,
                local_geometry,
                context_local,
            )
            Ti_laplacian = local_perp_laplacian_conservative_op(
                Ti_conservative_stencil,
                local_geometry,
                domain,
                face_projectors=(
                    local_invariants.face_projector_x,
                    local_invariants.face_projector_y,
                    local_invariants.face_projector_z,
                ),
                face_bc=face_bc.Ti,
                regular_face_geometry=local_geometry.regular_face_geometry,
            )
            owned = domain.layout.owned_slices_cell
            vorticity = jnp.asarray(state_halo.vorticity[owned], dtype=jnp.float64)
            tau = jnp.asarray(context.parameters.tau, dtype=jnp.float64)
            if use_exact_phi:
                phi_halo = state_halo.phi
            else:
                phi_solver = LocalPerpLaplacianInverseSolver(
                    geometry=local_geometry,
                    domain=domain,
                    stencil_builder=build_local_conservative_stencil_from_field,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=ghost_filler,
                    face_projectors=(
                        local_invariants.face_projector_x,
                        local_invariants.face_projector_y,
                        local_invariants.face_projector_z,
                    ),
                    regular_face_geometry=local_geometry.regular_face_geometry,
                    face_bc=face_bc.phi,
                    config=gmres_config,
                )
                phi_owned = phi_solver(
                    tau * Ti_laplacian - vorticity,
                    guess_owned=state_owned.phi,
                )
                phi_halo = inject_owned_field_to_halo(phi_owned, domain.layout)
                phi_halo = topology_filler(HaloExchange3D()(phi_halo, domain), domain)
                phi_halo = ghost_filler(phi_halo, domain, face_bc.phi)
            phi_stencil = direct_stencil_builder(phi_halo, local_geometry, context_local)
            poisson_Ve = local_poisson_bracket_op(phi_stencil, Ve_stencil, local_geometry)
            grad_parallel_Ve = local_grad_parallel_op_direct(Ve_stencil, local_geometry)
            grad_parallel_phi = local_grad_parallel_op_direct(phi_stencil, local_geometry)
            grad_parallel_Pe = local_grad_parallel_op_direct(Pe_stencil, local_geometry)
            grad_parallel_Te = local_grad_parallel_op_direct(Te_stencil, local_geometry)

            density = jnp.asarray(state_halo.density[owned], dtype=jnp.float64)
            Ve = jnp.asarray(state_halo.Ve[owned], dtype=jnp.float64)
            current = jnp.asarray(current_halo[owned], dtype=jnp.float64)
            density_safe = jnp.maximum(density, 1.0e-30)
            bmag = jnp.maximum(jnp.asarray(local_geometry.cell_bfield.Bmag_owned, dtype=jnp.float64), 1.0e-30)
            rho_star = jnp.asarray(context.parameters.rho_star, dtype=jnp.float64)
            mi_over_me = jnp.asarray(context.parameters.mi_over_me, dtype=jnp.float64)
            Ve_nu = jnp.asarray(context.parameters.Ve_nu, dtype=jnp.float64)

            discrete_poisson = -(poisson_Ve / (rho_star * bmag))
            discrete_parallel_advection = -Ve * grad_parallel_Ve
            discrete_collision = mi_over_me * Ve_nu * current
            discrete_grad_phi = mi_over_me * grad_parallel_phi
            discrete_grad_pe = -mi_over_me * grad_parallel_Pe / density_safe
            discrete_grad_te = -mi_over_me * 0.71 * grad_parallel_Te
            discrete_source = jnp.asarray(rk_stage_data.stage_1.source_halo.Ve[owned], dtype=jnp.float64)

            data = _local_analytic_data(
                local_invariants.coordinates,
                stage_time,
                rho_min=rho_min,
                rho_max=rho_max,
                parameters=context.parameters,
            )
            analytic_bmag = jnp.maximum(local_invariants.bmag_halo, 1.0e-30)
            analytic_density_safe = jnp.maximum(data.density, 1.0e-30)
            b_contra = local_invariants.b_contra_halo
            b_cov = jnp.einsum(
                "...ij,...j->...i",
                local_invariants.cell_metric_g_cov_halo,
                b_contra,
            )
            jacobian = jnp.maximum(local_invariants.cell_metric_jacobian_halo, 1.0e-30)

            def _poisson(left_grad: jnp.ndarray, right_grad: jnp.ndarray) -> jnp.ndarray:
                return jnp.sum(b_cov * jnp.cross(left_grad, right_grad), axis=-1) / jacobian

            def _grad_parallel(field_grad: jnp.ndarray) -> jnp.ndarray:
                return jnp.einsum("...i,...i->...", b_contra, field_grad)

            Pe_grad = data.Te[..., None] * data.density_grad + data.density[..., None] * data.Te_grad
            analytic_poisson_halo = -(
                _poisson(data.phi_grad, data.Ve_grad)
                / (rho_star * analytic_bmag)
            )
            analytic_parallel_advection_halo = -data.Ve * _grad_parallel(data.Ve_grad)
            analytic_collision_halo = (
                mi_over_me * Ve_nu * data.density * (data.Vi - data.Ve)
            )
            analytic_grad_phi_halo = mi_over_me * _grad_parallel(data.phi_grad)
            analytic_grad_pe_halo = (
                -mi_over_me * _grad_parallel(Pe_grad) / analytic_density_safe
            )
            analytic_grad_te_halo = -mi_over_me * 0.71 * _grad_parallel(data.Te_grad)
            analytic_non_source_halo = (
                analytic_poisson_halo
                + analytic_parallel_advection_halo
                + analytic_collision_halo
                + analytic_grad_phi_halo
                + analytic_grad_pe_halo
                + analytic_grad_te_halo
            )
            analytic_source_halo = data.Ve_t - analytic_non_source_halo
            analytic_total = jnp.asarray(data.Ve_t[owned], dtype=jnp.float64)

            actual_total = (
                discrete_poisson
                + discrete_parallel_advection
                + discrete_collision
                + discrete_grad_phi
                + discrete_grad_pe
                + discrete_grad_te
                + discrete_source
            )
            return _VeTermResidualBundle(
                poisson=discrete_poisson - analytic_poisson_halo[owned],
                parallel_advection=(
                    discrete_parallel_advection
                    - analytic_parallel_advection_halo[owned]
                ),
                collision=discrete_collision - analytic_collision_halo[owned],
                grad_phi=discrete_grad_phi - analytic_grad_phi_halo[owned],
                grad_pe=discrete_grad_pe - analytic_grad_pe_halo[owned],
                grad_te=discrete_grad_te - analytic_grad_te_halo[owned],
                source=discrete_source - analytic_source_halo[owned],
                total=actual_total - analytic_total,
            )

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        stage_data = jax.jit(mapped_source_kernel)(
            invariants,
            jnp.asarray(time, dtype=jnp.float64),
        )
        mapped_exact_state_kernel = shard_map(
            exact_state_kernel,
            mesh=mesh,
            in_specs=(stage_data_spec,),
            out_specs=state_spec,
            check_rep=False,
        )
        exact_state = jax.jit(mapped_exact_state_kernel)(stage_data)
        mapped_ve_term_kernel = shard_map(
            ve_term_kernel,
            mesh=mesh,
            in_specs=(state_spec, invariant_spec, stage_data_spec, P()),
            out_specs=ve_term_spec,
            check_rep=False,
        )
        residuals = jax.jit(mapped_ve_term_kernel)(
            exact_state,
            invariants,
            stage_data,
            jnp.asarray(time, dtype=jnp.float64),
        )
        jax.block_until_ready(residuals.total)
        return jax.tree_util.tree_map(
            lambda value: jnp.asarray(jax.device_get(value), dtype=jnp.float64),
            residuals,
        )


def _ve_term_error_statistics(
    residuals: _VeTermResidualBundle,
    geometry: FciGeometry3D,
) -> dict[str, tuple[float, float, float]]:
    zeros = jnp.zeros_like(residuals.total)
    return {
        "poisson": _field_error_statistics(residuals.poisson, zeros, geometry),
        "parallel_advection": _field_error_statistics(
            residuals.parallel_advection,
            zeros,
            geometry,
        ),
        "collision": _field_error_statistics(residuals.collision, zeros, geometry),
        "grad_phi": _field_error_statistics(residuals.grad_phi, zeros, geometry),
        "grad_pe": _field_error_statistics(residuals.grad_pe, zeros, geometry),
        "grad_te": _field_error_statistics(residuals.grad_te, zeros, geometry),
        "source": _field_error_statistics(residuals.source, zeros, geometry),
        "total": _field_error_statistics(residuals.total, zeros, geometry),
    }


def evaluate_shifted_torus_eb_phi_closure_diagnostic(
    context: ShiftedTorusEbMmsContext,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    time: float = 0.013,
    closure_mode: str = "analytic_omega",
) -> _PhiClosureDiagnosticBundle:
    """Compare reconstructed ``phi`` and ``grad_parallel(phi)`` for a closure mode."""

    if closure_mode not in {"analytic_omega", "discrete_omega"}:
        raise ValueError(
            "closure_mode must be 'analytic_omega' or 'discrete_omega', "
            f"got {closure_mode!r}"
        )
    use_discrete_omega = closure_mode == "discrete_omega"
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(context.geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(context.geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(
        context.geometry.shape,
        halo_width,
        shard_counts,
    )
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    gmres_config = SpmdGmresConfig(
        tol=1.0e-10,
        atol=1.0e-10,
        maxiter=int(context.parameters.phi_inversion_iterations),
        restart=min(100, int(context.parameters.phi_inversion_iterations)),
        acceptance_tol=1.0e-4,
        acceptance_atol=1.0e-4,
        regularization_epsilon=float(context.parameters.phi_inversion_regularization),
    )
    rho_min, rho_max = _mms_radial_bounds(context.geometry)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state_spec = _state_partition_spec()
        closure_spec = _phi_closure_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_eb_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=context.geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(_sample_local_eb_rk4_stage_data(domain))
        )

        def invariant_kernel() -> _ShiftedTorusEbInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_eb_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=context.geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorusEbInvariantBundle,
            stage_time: jax.Array,
        ) -> _ShiftedTorusEbRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            return expand_local_shard_pytree(
                _build_local_eb_rk4_stage_data(
                    local_invariants,
                    local_geometry,
                    domain,
                    stage_time,
                    jnp.asarray(0.0, dtype=jnp.float64),
                    rho_min=rho_min,
                    rho_max=rho_max,
                    parameters=context.parameters,
                    halo_exchange=HaloExchange3D(),
                    topology_filler=topology_filler,
                    physical_ghost_filler=ghost_filler,
                    gmres_config=gmres_config,
                )
            )

        def exact_state_kernel(
            rk_stage_data: _ShiftedTorusEbRk4StageData,
        ) -> FciDrbEBState:
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            return _owned_state_from_halo_state(
                rk_stage_data.stage_1.exact_halo,
                domain,
            )

        def closure_kernel(
            state_owned: FciDrbEBState,
            local_invariants: _ShiftedTorusEbInvariantBundle,
            rk_stage_data: _ShiftedTorusEbRk4StageData,
            stage_time: jax.Array,
        ) -> _PhiClosureDiagnosticBundle:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=context.geometry.shape,
                shard_index=shard_index,
                x_min=x_min,
                x_max=x_max,
                r0=r0,
                alpha_value=alpha_value,
                iota=iota,
                c_phi=c_phi,
                sigma=sigma,
            )
            face_bc_bundle = _build_local_eb_face_bcs(rk_stage_data.stage_1, domain)
            prepared = _prepare_local_eb_stage_state(
                state_owned,
                domain,
                face_bc=face_bc_bundle,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
            )
            state_halo = prepared.state_halo
            face_bc = prepared.boundary_data.face_bc
            context_local = StencilBuilderContext(layout=domain.layout, domain=domain)
            direct_stencil_builder = build_local_direct_stencil_one_sided_physical_from_halo
            face_projectors = (
                local_invariants.face_projector_x,
                local_invariants.face_projector_y,
                local_invariants.face_projector_z,
            )

            phi_stencil_exact_for_laplacian = build_local_conservative_stencil_from_field(
                state_halo.phi,
                local_geometry,
                context_local,
            )
            phi_laplacian = local_perp_laplacian_conservative_op(
                phi_stencil_exact_for_laplacian,
                local_geometry,
                domain,
                face_projectors=face_projectors,
                face_bc=face_bc.phi,
                regular_face_geometry=local_geometry.regular_face_geometry,
            )
            Ti_stencil_for_laplacian = build_local_conservative_stencil_from_field(
                state_halo.Ti,
                local_geometry,
                context_local,
            )
            Ti_laplacian = local_perp_laplacian_conservative_op(
                Ti_stencil_for_laplacian,
                local_geometry,
                domain,
                face_projectors=face_projectors,
                face_bc=face_bc.Ti,
                regular_face_geometry=local_geometry.regular_face_geometry,
            )
            owned = domain.layout.owned_slices_cell
            data = _local_analytic_data(
                local_invariants.coordinates,
                stage_time,
                rho_min=rho_min,
                rho_max=rho_max,
                parameters=context.parameters,
            )
            tau = jnp.asarray(context.parameters.tau, dtype=jnp.float64)
            rhs_analytic_omega = (
                tau * Ti_laplacian
                - jnp.asarray(data.vorticity[owned], dtype=jnp.float64)
            )
            rhs_discrete_omega = -phi_laplacian
            rhs = rhs_discrete_omega if use_discrete_omega else rhs_analytic_omega
            phi_solver = LocalPerpLaplacianInverseSolver(
                geometry=local_geometry,
                domain=domain,
                stencil_builder=build_local_conservative_stencil_from_field,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                face_projectors=face_projectors,
                regular_face_geometry=local_geometry.regular_face_geometry,
                face_bc=face_bc.phi,
                config=gmres_config,
            )
            phi_reconstructed = phi_solver(
                rhs,
                guess_owned=state_owned.phi,
            )
            phi_reconstructed_halo = inject_owned_field_to_halo(
                phi_reconstructed,
                domain.layout,
            )
            phi_reconstructed_halo = topology_filler(
                HaloExchange3D()(phi_reconstructed_halo, domain),
                domain,
            )
            phi_reconstructed_halo = ghost_filler(
                phi_reconstructed_halo,
                domain,
                face_bc.phi,
            )
            reconstructed_stencil = direct_stencil_builder(
                phi_reconstructed_halo,
                local_geometry,
                context_local,
            )
            exact_stencil = direct_stencil_builder(
                state_halo.phi,
                local_geometry,
                context_local,
            )
            grad_reconstructed = local_grad_parallel_op_direct(
                reconstructed_stencil,
                local_geometry,
            )
            grad_exact = local_grad_parallel_op_direct(
                exact_stencil,
                local_geometry,
            )
            return _PhiClosureDiagnosticBundle(
                phi=phi_reconstructed - state_owned.phi,
                grad_parallel_phi=grad_reconstructed - grad_exact,
                closure_rhs=rhs - rhs_discrete_omega,
            )

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        stage_data = jax.jit(mapped_source_kernel)(
            invariants,
            jnp.asarray(time, dtype=jnp.float64),
        )
        mapped_exact_state_kernel = shard_map(
            exact_state_kernel,
            mesh=mesh,
            in_specs=(stage_data_spec,),
            out_specs=state_spec,
            check_rep=False,
        )
        exact_state = jax.jit(mapped_exact_state_kernel)(stage_data)
        mapped_closure_kernel = shard_map(
            closure_kernel,
            mesh=mesh,
            in_specs=(state_spec, invariant_spec, stage_data_spec, P()),
            out_specs=closure_spec,
            check_rep=False,
        )
        diagnostics = jax.jit(mapped_closure_kernel)(
            exact_state,
            invariants,
            stage_data,
            jnp.asarray(time, dtype=jnp.float64),
        )
        jax.block_until_ready(diagnostics.grad_parallel_phi)
        return jax.tree_util.tree_map(
            lambda value: jnp.asarray(jax.device_get(value), dtype=jnp.float64),
            diagnostics,
        )


def _phi_closure_error_statistics(
    diagnostics: _PhiClosureDiagnosticBundle,
    geometry: FciGeometry3D,
) -> dict[str, tuple[float, float, float]]:
    zeros = jnp.zeros_like(diagnostics.phi)
    return {
        "phi": _field_error_statistics(diagnostics.phi, zeros, geometry),
        "grad_parallel_phi": _field_error_statistics(
            diagnostics.grad_parallel_phi,
            zeros,
            geometry,
        ),
        "closure_rhs": _field_error_statistics(
            diagnostics.closure_rhs,
            zeros,
            geometry,
        ),
    }


def run_shifted_torus_eb_rhs_diagnostic(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    time: float = 0.013,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    previous_resolution: int | None = None
    previous_error: float | None = None
    print("Shifted-torus EB exact-state RHS residual diagnostic")
    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        context = build_shifted_torus_eb_mms_context(shape)
        print(
            f"Starting EB RHS diagnostic: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, time={time:.6e}"
        )
        start = time_module.perf_counter()
        residual = evaluate_shifted_torus_eb_rhs_residual(
            context,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=time,
        )
        elapsed = time_module.perf_counter() - start
        stats = _state_error_statistics(
            residual,
            _zero_state_like(residual),
            context.geometry,
        )
        combined_error = _combined_l2_error(stats)
        max_error = float(max(field_stats[1] for field_stats in stats.values()))
        successful_resolutions.append(int(resolution))
        l2_errors.append(combined_error)
        max_errors.append(max_error)
        order_text = ""
        if previous_resolution is not None and previous_error is not None:
            order = _observed_order(previous_error, combined_error, previous_resolution, resolution)
            order_text = f", order={order:.3f}"
        print(
            f"resolution={resolution}: elapsed={elapsed:.6e} s, "
            f"combined_rhs_residual_l2={combined_error:.6e}{order_text}"
        )
        _print_state_residual_statistics(f"resolution={resolution} RHS residuals", stats)
        previous_resolution = resolution
        previous_error = combined_error

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
    }


def run_shifted_torus_eb_ve_term_diagnostic(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    time: float = 0.013,
    phi_mode: str = "reconstructed",
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    total_l2_errors: list[float] = []
    previous_resolution: int | None = None
    previous_error: float | None = None
    print("Shifted-torus EB Ve per-term RHS residual diagnostic")
    print(f"phi_mode = {phi_mode}")
    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        context = build_shifted_torus_eb_mms_context(shape)
        print(
            f"Starting EB Ve term diagnostic: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, time={time:.6e}"
        )
        start = time_module.perf_counter()
        residuals = evaluate_shifted_torus_eb_ve_term_residuals(
            context,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=time,
            phi_mode=phi_mode,
        )
        elapsed = time_module.perf_counter() - start
        stats = _ve_term_error_statistics(residuals, context.geometry)
        total_l2 = stats["total"][0]
        successful_resolutions.append(int(resolution))
        total_l2_errors.append(total_l2)
        order_text = ""
        if previous_resolution is not None and previous_error is not None:
            order = _observed_order(previous_error, total_l2, previous_resolution, resolution)
            order_text = f", order={order:.3f}"
        print(
            f"resolution={resolution}: elapsed={elapsed:.6e} s, "
            f"Ve_total_l2={total_l2:.6e}{order_text}"
        )
        _print_state_residual_statistics(f"resolution={resolution} Ve term residuals", stats)
        previous_resolution = resolution
        previous_error = total_l2

    return {
        "resolutions": successful_resolutions,
        "total_l2_errors": total_l2_errors,
    }


def run_shifted_torus_eb_phi_closure_diagnostic(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    time: float = 0.013,
    closure_modes: tuple[str, ...] = ("analytic_omega", "discrete_omega"),
) -> dict[str, object]:
    results: dict[str, object] = {}
    print("Shifted-torus EB phi-closure reconstruction diagnostic")
    for closure_mode in closure_modes:
        successful_resolutions: list[int] = []
        grad_l2_errors: list[float] = []
        previous_resolution: int | None = None
        previous_error: float | None = None
        print(f"closure_mode = {closure_mode}")
        for resolution in resolutions:
            shape = (int(resolution), int(resolution), int(resolution))
            assert_shape_divisible_by_shards(shape, shard_counts)
            context = build_shifted_torus_eb_mms_context(shape)
            print(
                f"Starting EB phi closure diagnostic: resolution={int(resolution)}, "
                f"shard_counts={shard_counts}, time={time:.6e}"
            )
            start = time_module.perf_counter()
            diagnostics = evaluate_shifted_torus_eb_phi_closure_diagnostic(
                context,
                shard_counts=shard_counts,
                halo_width=halo_width,
                time=time,
                closure_mode=closure_mode,
            )
            elapsed = time_module.perf_counter() - start
            stats = _phi_closure_error_statistics(diagnostics, context.geometry)
            grad_l2 = stats["grad_parallel_phi"][0]
            successful_resolutions.append(int(resolution))
            grad_l2_errors.append(grad_l2)
            order_text = ""
            if previous_resolution is not None and previous_error is not None:
                order = _observed_order(previous_error, grad_l2, previous_resolution, resolution)
                order_text = f", grad_parallel_phi_order={order:.3f}"
            print(
                f"resolution={resolution}: elapsed={elapsed:.6e} s, "
                f"grad_parallel_phi_l2={grad_l2:.6e}{order_text}"
            )
            _print_state_residual_statistics(
                f"resolution={resolution} phi closure diagnostics",
                stats,
            )
            previous_resolution = resolution
            previous_error = grad_l2
        results[closure_mode] = {
            "resolutions": successful_resolutions,
            "grad_l2_errors": grad_l2_errors,
        }
    return results


def run_shifted_torus_eb_single_step_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    step_dt: float,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    previous_resolution: int | None = None
    previous_error: float | None = None
    print("Shifted-torus EB single-step RK4 convergence diagnostic")
    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        context = build_shifted_torus_eb_mms_context(shape)
        print(
            f"Starting EB single-step diagnostic: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, dt={step_dt:.6e}"
        )
        start = time_module.perf_counter()
        final_state, final_time_value, exact_state = simulate_mms_shifted_torus_eb(
            context,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=step_dt,
            timestep=step_dt,
            num_steps=1,
            show_progress=show_progress,
            return_exact=True,
        )
        elapsed = time_module.perf_counter() - start
        stats = _state_error_statistics(final_state, exact_state, context.geometry)
        combined_error = _combined_l2_error(stats)
        max_error = float(max(field_stats[1] for field_stats in stats.values()))
        successful_resolutions.append(int(resolution))
        l2_errors.append(combined_error)
        max_errors.append(max_error)
        order_text = ""
        if previous_resolution is not None and previous_error is not None:
            order = _observed_order(previous_error, combined_error, previous_resolution, resolution)
            order_text = f", order={order:.3f}"
        print(
            f"resolution={resolution}: elapsed={elapsed:.6e} s, "
            f"final_time={float(final_time_value):.6e}, "
            f"combined_single_step_l2={combined_error:.6e}{order_text}"
        )
        _print_state_error_statistics(f"resolution={resolution} single-step errors", stats)
        previous_resolution = resolution
        previous_error = combined_error

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
    }


def run_shifted_torus_eb_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = TF,
    base_steps: int = NUM_STEPS,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    previous_resolution: int | None = None
    previous_error: float | None = None
    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        context = build_shifted_torus_eb_mms_context(shape)
        base_resolution = int(resolutions[0]) if resolutions else int(resolution)
        steps = max(1, int(round(float(base_steps) * np.sqrt(float(resolution) / float(base_resolution)))))
        print(
            f"Starting shifted_torus_EB MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}"
        )
        start = time_module.perf_counter()
        final_state, final_time_value, exact_state = simulate_mms_shifted_torus_eb(
            context,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=final_time,
            num_steps=steps,
            show_progress=show_progress,
            return_exact=True,
        )
        elapsed = time_module.perf_counter() - start
        stats = _state_error_statistics(final_state, exact_state, context.geometry)
        combined_error = _combined_l2_error(stats)
        max_error = float(max(field_stats[1] for field_stats in stats.values()))
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
        closure_stats = _domain_decomp_closure_error_statistics(
            final_state,
            context,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=float(final_time_value),
        )
        print(
            f"  closure: weighted_l2={closure_stats[0]:.6e}, "
            f"linf={closure_stats[1]:.6e}, rel_l2={closure_stats[2]:.6e}"
        )
        previous_resolution = resolution
        previous_error = combined_error

    l2_order: float | None = None
    max_order: float | None = None
    if len(successful_resolutions) >= 2:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        l2_order = float(-l2_slope)
        max_order = float(-max_slope)
        print(f"shifted_torus_EB L2 convergence order: {l2_order:.6f}")
        print(f"shifted_torus_EB Linf convergence order: {max_order:.6f}")

        if plot:
            import matplotlib.pyplot as plt

            output_path = Path(plot_path or "shifted_torus_EB_convergence.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"L2, order {l2_order:.2f}")
            ax.loglog(plotted_resolutions, max_errors, "^-", label=f"Linf, order {max_order:.2f}")
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
            ax.set_title(f"Shifted-torus EB MMS convergence ({shard_counts})")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
    elif plot:
        print("WARNING: fewer than two successful resolutions, skipping convergence plot.")

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
        "l2_order": l2_order,
        "linf_order": max_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted-torus EB MMS convergence harness")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=TF)
    parser.add_argument("--base-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--diagnose-rhs",
        action="store_true",
        help="Evaluate exact-state RHS residual convergence without advancing RK4.",
    )
    parser.add_argument(
        "--rhs-diagnostic-time",
        type=float,
        default=0.013,
        help="Time used by --diagnose-rhs (default: 0.013).",
    )
    parser.add_argument(
        "--single-step-convergence",
        action="store_true",
        help="Run one RK4 step per resolution and compare against the exact state.",
    )
    parser.add_argument(
        "--single-step-dt",
        type=float,
        default=None,
        help="Timestep for --single-step-convergence; defaults to final-time/base-steps.",
    )
    parser.add_argument(
        "--diagnose-ve",
        action="store_true",
        help="Report per-term Ve RHS residuals on the exact MMS state.",
    )
    parser.add_argument(
        "--ve-phi-mode",
        choices=("reconstructed", "exact"),
        default="reconstructed",
        help="Phi used by --diagnose-ve: live elliptic reconstruction or exact MMS phi.",
    )
    parser.add_argument(
        "--diagnose-phi-closure",
        action="store_true",
        help="Compare reconstructed phi and grad_parallel(phi) for MMS closure choices.",
    )
    parser.add_argument(
        "--phi-closure-mode",
        choices=("all", "analytic_omega", "discrete_omega"),
        default="all",
        help="Closure mode used by --diagnose-phi-closure.",
    )
    args = parser.parse_args()

    resolutions = [int(value) for value in args.resolutions]
    shard_counts = tuple(int(value) for value in args.shard_counts)
    halo_width = int(args.halo_width)
    ran_diagnostic = False

    if bool(args.diagnose_rhs):
        run_shifted_torus_eb_rhs_diagnostic(
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=float(args.rhs_diagnostic_time),
        )
        ran_diagnostic = True

    if bool(args.single_step_convergence):
        if args.single_step_dt is None:
            step_dt = float(args.final_time) / float(max(1, int(args.base_steps)))
        else:
            step_dt = float(args.single_step_dt)
        run_shifted_torus_eb_single_step_convergence(
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=halo_width,
            step_dt=step_dt,
            show_progress=bool(args.show_progress),
        )
        ran_diagnostic = True

    if bool(args.diagnose_ve):
        run_shifted_torus_eb_ve_term_diagnostic(
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=float(args.rhs_diagnostic_time),
            phi_mode=str(args.ve_phi_mode),
        )
        ran_diagnostic = True

    if bool(args.diagnose_phi_closure):
        if str(args.phi_closure_mode) == "all":
            closure_modes = ("analytic_omega", "discrete_omega")
        else:
            closure_modes = (str(args.phi_closure_mode),)
        run_shifted_torus_eb_phi_closure_diagnostic(
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=halo_width,
            time=float(args.rhs_diagnostic_time),
            closure_modes=closure_modes,
        )
        ran_diagnostic = True

    if not ran_diagnostic:
        run_shifted_torus_eb_convergence(
            resolutions=resolutions,
            shard_counts=shard_counts,
            halo_width=halo_width,
            final_time=float(args.final_time),
            base_steps=int(args.base_steps),
            plot=bool(args.plot),
            plot_path=args.plot_path,
            show_progress=bool(args.show_progress),
        )


if __name__ == "__main__":
    main()
