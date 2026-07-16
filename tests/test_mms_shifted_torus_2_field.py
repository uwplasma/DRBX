from __future__ import annotations

import argparse
import time as time_module
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FciGeometry3D,
    FciMaps3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    Grid1D,
    LocalDomain3D,
    LocalFciGeometry3D,
    LocalStencilBuilder,
    MetricGeometry,
    RegularFaceGeometry3D,
    Spacing3D,
    StencilBuilderContext,
    build_curvature_coefficients,
    build_local_curvature_coefficients,
    build_fci_maps_from_b_contravariant,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native.fci_2_field_rhs import Fci2FieldRhsParameters, Fci2FieldState
from jax_drb.native.fci_boundaries import BC_DIRICHLET, BoundaryConditionBuilder, BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D, LocalBoundaryData3D, LocalBoundaryFaceBC3D
from jax_drb.native.fci_halo import GhostFillWeights1D, HaloExchange3D, PhysicalGhostCellFiller3D, PreparedLocalState3D, TopologyHaloFiller3D, LocalPeriodicTopologyRule3D
from jax_drb.native.fci_model import FciFieldBundle, inject_owned_state_to_halo
from jax_drb.native.fci_operators import local_curvature_op, local_grad_parallel_op_direct, local_poisson_bracket_op

from mms_domain_decomp_helpers import (
    MESH_AXIS_NAMES,
    PERIODIC_AXES,
    assert_shape_divisible_by_shards,
    build_shifted_torus_local_domain,
    build_shifted_torus_local_geometry,
    expand_local_shard_pytree,
    extract_local_shard_pytree,
    local_shard_pytree_partition_spec,
    make_mesh_for_shard_counts,
)


A = 0.1
Bv = 0.1
alpha = 0.2
omega = 2.0 * jnp.pi
rho_star = 1.0
M_phi = 2
N_phi = 3
M_v = 3
N_v = 4
sigma = 0.0
r0 = 3.0
alpha_value = 0.25
iota = 1.1
c_phi = 3.0
x_min = 0.15
x_max = 1.0
tf = 0.1
num_steps = 100


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus2FieldFaceBCBundle(FciFieldBundle):
    density: LocalBoundaryFaceBC3D
    phi: LocalBoundaryFaceBC3D
    v_parallel: LocalBoundaryFaceBC3D
    density_background: LocalBoundaryFaceBC3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus2FieldInvariantBundle(FciFieldBundle):
    coord_x: jnp.ndarray
    coord_theta_shift: jnp.ndarray
    coord_theta: jnp.ndarray
    coord_zeta: jnp.ndarray
    face_x: jnp.ndarray
    face_theta_shift: jnp.ndarray
    face_theta: jnp.ndarray
    face_zeta: jnp.ndarray
    bmag_halo: jnp.ndarray
    curvature_coefficients_owned: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus2FieldStageData(FciFieldBundle):
    exact_halo: Fci2FieldState
    density_source_halo: jnp.ndarray
    v_parallel_source_halo: jnp.ndarray
    phi_face_lower: jnp.ndarray
    phi_face_upper: jnp.ndarray
    density_face_lower: jnp.ndarray
    density_face_upper: jnp.ndarray
    v_parallel_face_lower: jnp.ndarray
    v_parallel_face_upper: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus2FieldRk4StageData(FciFieldBundle):
    stage_1: _ShiftedTorus2FieldStageData
    stage_2: _ShiftedTorus2FieldStageData
    stage_3: _ShiftedTorus2FieldStageData
    stage_4: _ShiftedTorus2FieldStageData


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


def _format_progress_bar(
    completed: int,
    total: int,
    *,
    start_time: float,
    width: int = 28,
) -> str:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, float(completed) / float(total)))
    filled = int(round(float(width) * fraction))
    elapsed = time_module.perf_counter() - start_time
    rate = float(completed) / elapsed if elapsed > 0.0 and completed > 0 else 0.0
    remaining = (float(total - completed) / rate) if rate > 0.0 else float("nan")
    eta_text = "--:--" if not np.isfinite(remaining) else _format_duration(remaining)
    return (
        f"[{'#' * filled}{'.' * (width - filled)}] "
        f"{completed:>4d}/{total:<4d} {100.0 * fraction:6.2f}% "
        f"elapsed={_format_duration(elapsed)} eta={eta_text}"
    )


def build_shifted_torus_2field_geometry(
    shape: tuple[int, int, int],
    *,
    x_min: float = x_min,
    x_max: float = x_max,
    r0: float = r0,
    alpha_value: float = alpha_value,
    iota: float = iota,
    c_phi: float = c_phi,
    sigma: float = sigma,
    construct_fci_maps: bool = False,
    B_contravariant: jnp.ndarray | None = None,
) -> FciGeometry3D:
    """Build a shifted-torus FCI geometry for the two-field MMS scaffold.

    The logical coordinates are ``(x, theta, zeta)`` with periodic ``theta`` and
    ``zeta``. The helper follows the same `FciGeometry3D` construction pattern used
    in `test_fci_operators.py`, but uses the physical radial coordinate directly and
    a shifted poloidal angle ``Theta = theta + sigma * (x - x_mid)``.
    """

    nx, ny, nz = shape
    x_centers = jnp.linspace(float(x_min), float(x_max), nx, dtype=jnp.float64)
    theta_centers = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False, dtype=jnp.float64)
    zeta_centers = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D.from_centers(x_centers),
        y=Grid1D.from_centers(theta_centers),
        z=Grid1D.from_centers(zeta_centers),
    )
    target_shape = grid.shape

    def _logical_grid(x_axis: jnp.ndarray, y_axis: jnp.ndarray, z_axis: jnp.ndarray) -> jnp.ndarray:
        return logical_grid_from_axis_vectors(x_axis, y_axis, z_axis)

    def _metric(logical_grid: jnp.ndarray) -> MetricGeometry:
        x = logical_grid[..., 0]
        theta = logical_grid[..., 1]
        x_mid = 0.5 * (float(x_min) + float(x_max))
        theta_shift = theta + float(sigma) * (x - x_mid)
        cos_theta = jnp.cos(theta_shift)
        sin_theta = jnp.sin(theta_shift)
        R = float(r0) + float(alpha_value) * x + x * cos_theta
        jacobian = R * x * (1.0 + float(alpha_value) * cos_theta)
        jacobian = jnp.where(jnp.abs(jacobian) < 1.0e-14, 1.0e-14, jacobian)
        g11 = 1.0 / (1.0 + float(alpha_value) * cos_theta) ** 2
        g12 = float(alpha_value) * sin_theta / (x * (1.0 + float(alpha_value) * cos_theta) ** 2)
        g13 = jnp.zeros_like(x)
        g22 = (1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2) / (x**2 * (1.0 + float(alpha_value) * cos_theta) ** 2)
        g23 = jnp.zeros_like(x)
        g33 = 1.0 / (R**2)
        g_11 = 1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2
        g_12 = -float(alpha_value) * x * sin_theta
        g_13 = jnp.zeros_like(x)
        g_22 = x**2
        g_23 = jnp.zeros_like(x)
        g_33 = R**2
        return MetricGeometry(
            J=jacobian,
            g11=g11,
            g22=g22,
            g33=g33,
            g12=g12,
            g13=g13,
            g23=g23,
            g_11=g_11,
            g_22=g_22,
            g_33=g_33,
            g_12=g_12,
            g_13=g_13,
            g_23=g_23,
        )

    def _bfield(logical_grid: jnp.ndarray, metric: MetricGeometry) -> BFieldGeometry:
        x = logical_grid[..., 0]
        theta = logical_grid[..., 1]
        x_mid = 0.5 * (float(x_min) + float(x_max))
        theta_shift = theta + float(sigma) * (x - x_mid)
        cos_theta = jnp.cos(theta_shift)
        R = float(r0) + float(alpha_value) * x + x * cos_theta
        jacobian = metric.J
        if B_contravariant is None:
            B_contra = jnp.stack(
                (
                    jnp.zeros_like(jacobian),
                    float(iota) * float(c_phi) / jacobian,
                    float(c_phi) / jacobian,
                ),
                axis=-1,
            )
        else:
            B_contra = jnp.asarray(B_contravariant, dtype=jnp.float64)
        Bmag = jnp.sqrt((float(iota) ** 2) * x**2 + R**2) * float(c_phi) / jacobian
        return BFieldGeometry(B_contra=B_contra, Bmag=Bmag)

    cell_logical_grid = _logical_grid(grid.x.centers, grid.y.centers, grid.z.centers)
    cell_metric = _metric(cell_logical_grid)
    cell_bfield = _bfield(cell_logical_grid, cell_metric)
    face_metric = FaceMetricGeometry(
        x=_metric(_logical_grid(grid.x.faces, grid.y.centers, grid.z.centers)),
        y=_metric(_logical_grid(grid.x.centers, grid.y.faces, grid.z.centers)),
        z=_metric(_logical_grid(grid.x.centers, grid.y.centers, grid.z.faces)),
    )
    face_bfield = FaceBFieldGeometry(
        x=_bfield(_logical_grid(grid.x.faces, grid.y.centers, grid.z.centers), face_metric.x),
        y=_bfield(_logical_grid(grid.x.centers, grid.y.faces, grid.z.centers), face_metric.y),
        z=_bfield(_logical_grid(grid.x.centers, grid.y.centers, grid.z.faces), face_metric.z),
    )

    if construct_fci_maps:
        map_fields = build_fci_maps_from_b_contravariant(
            grid,
            cell_bfield.B_contra,
            cell_bfield.Bmag,
            periodic_axes=(False, True, True),
        )
    else:
        ones = jnp.ones(target_shape, dtype=jnp.float64)
        zeros = jnp.zeros(target_shape, dtype=jnp.float64)
        map_fields = {
            "forward_x": zeros,
            "forward_y": zeros,
            "backward_x": zeros,
            "backward_y": zeros,
            "forward_endpoint_x": zeros,
            "forward_endpoint_y": zeros,
            "forward_endpoint_z": zeros,
            "backward_endpoint_x": zeros,
            "backward_endpoint_y": zeros,
            "backward_endpoint_z": zeros,
            "forward_length": ones,
            "backward_length": ones,
            "forward_boundary": zeros.astype(bool),
            "backward_boundary": zeros.astype(bool),
        }

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
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], target_shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], target_shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], target_shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )


def _shifted_torus_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    x = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _shifted_torus_background_density(geometry: FciGeometry3D) -> jnp.ndarray:
    return jnp.ones(geometry.shape, dtype=jnp.float64)


def _shifted_torus_phi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    radial_envelope = jnp.sin(jnp.pi * x)
    return float(A) * radial_envelope * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * jnp.cos(float(omega) * time)


def _shifted_torus_density(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    density_background = _shifted_torus_background_density(geometry)
    return density_background * jnp.exp(_shifted_torus_phi(geometry, time))


def _shifted_torus_v_parallel(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    radial_envelope = jnp.cos(jnp.pi * x)
    return float(Bv) * radial_envelope * jnp.sin(float(M_v) * theta_shift) * jnp.cos(float(N_v) * zeta) * jnp.sin(float(omega) * time)


def _shifted_torus_exact_state(geometry: FciGeometry3D, time: float) -> Fci2FieldState:
    return Fci2FieldState(
        density=_shifted_torus_density(geometry, time),
        v_parallel=_shifted_torus_v_parallel(geometry, time),
        density_background=_shifted_torus_background_density(geometry),
    )


def _shifted_torus_dirichlet_boundary_condition_builder(field_name: str):
    def build(
        state: jnp.ndarray,
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
        cut_wall_geometry: CutWallGeometry3D | None,
        cut_wall_bc: CutWallBC3D | None,
    ) -> tuple[BoundaryFaceBC3D, CutWallBC3D]:
        del periodic_axes, cut_wall_geometry
        regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
        values = jnp.asarray(getattr(state, field_name, state), dtype=jnp.float64)
        face_bc = BoundaryFaceBC3D(
            kind_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
            kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
            kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
            value_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[0].set(values[0]).at[-1].set(values[-1]),
            value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
            value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
            mask_x=jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
            mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
            mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
        )
        return face_bc, cut_wall_bc or CutWallBC3D.empty()

    return build


def _apply_dirichlet_face_bcs_to_state(
    state: Fci2FieldState,
    density_face_bc: BoundaryFaceBC3D,
    v_parallel_face_bc: BoundaryFaceBC3D,
) -> Fci2FieldState:
    density = jnp.asarray(state.density, dtype=jnp.float64)
    v_parallel = jnp.asarray(state.v_parallel, dtype=jnp.float64)
    density = density.at[0, :, :].set(jnp.asarray(density_face_bc.value_x[0], dtype=jnp.float64))
    density = density.at[-1, :, :].set(jnp.asarray(density_face_bc.value_x[-1], dtype=jnp.float64))
    v_parallel = v_parallel.at[0, :, :].set(jnp.asarray(v_parallel_face_bc.value_x[0], dtype=jnp.float64))
    v_parallel = v_parallel.at[-1, :, :].set(jnp.asarray(v_parallel_face_bc.value_x[-1], dtype=jnp.float64))
    return Fci2FieldState(
        density=density,
        v_parallel=v_parallel,
        density_background=state.density_background,
    )


def _shifted_torus_local_coordinates(
    geometry: LocalFciGeometry3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, theta, zeta = jnp.meshgrid(
        geometry.grid.x.centers_halo,
        geometry.grid.y.centers_halo,
        geometry.grid.z.centers_halo,
        indexing="ij",
    )
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _shifted_torus_local_geometry_quantities(
    x: jnp.ndarray,
    theta_shift: jnp.ndarray,
    zeta: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    del zeta
    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * x + x * cos_shift
    Q = 1.0 + float(alpha_value) * cos_shift
    J = x * R * Q
    D2 = (float(iota) ** 2) * x**2 + R**2
    D = jnp.sqrt(D2)
    P = float(alpha_value) + cos_shift
    E = x * Q + float(alpha_value) * R
    A_term = (float(iota) ** 2) * x + R * P
    return cos_shift, sin_shift, R, Q, J, D, P, E, A_term


def _shifted_torus_local_phi_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mphi = jnp.sin(float(M_phi) * theta_shift)
    cos_mphi = jnp.cos(float(M_phi) * theta_shift)
    sin_nphi = jnp.sin(float(N_phi) * zeta)
    cos_nphi = jnp.cos(float(N_phi) * zeta)
    cos_omega_t = jnp.cos(float(omega) * time)
    sin_omega_t = jnp.sin(float(omega) * time)
    phi = float(A) * sin_u * cos_mphi * sin_nphi * cos_omega_t
    phi_u = float(A) * (
        jnp.pi * cos_u * cos_mphi
        - float(sigma) * float(M_phi) * sin_u * sin_mphi
    ) * sin_nphi * cos_omega_t
    phi_theta = -float(A) * float(M_phi) * sin_u * sin_mphi * sin_nphi * cos_omega_t
    phi_zeta = float(A) * float(N_phi) * sin_u * cos_mphi * cos_nphi * cos_omega_t
    phi_t = -float(A) * float(omega) * sin_u * cos_mphi * sin_nphi * sin_omega_t
    return phi, phi_u, phi_theta, phi_zeta, phi_t


def _shifted_torus_local_density_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, _, _, _ = coordinates_halo
    phi, phi_u, phi_theta, phi_zeta, phi_t = _shifted_torus_local_phi_derivatives(
        coordinates_halo,
        time,
    )
    density_background = jnp.ones_like(x)
    density = density_background * jnp.exp(phi)
    density_u = density * phi_u
    density_theta = density * phi_theta
    density_zeta = density * phi_zeta
    density_t = density * phi_t
    return density, density_u, density_theta, density_zeta, density_t


def _shifted_torus_local_v_parallel_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mv = jnp.sin(float(M_v) * theta_shift)
    cos_mv = jnp.cos(float(M_v) * theta_shift)
    sin_nv = jnp.sin(float(N_v) * zeta)
    cos_nv = jnp.cos(float(N_v) * zeta)
    sin_omega_t = jnp.sin(float(omega) * time)
    cos_omega_t = jnp.cos(float(omega) * time)
    v_parallel = float(Bv) * cos_u * sin_mv * cos_nv * sin_omega_t
    v_parallel_u = float(Bv) * (
        -jnp.pi * sin_u * sin_mv
        + float(sigma) * float(M_v) * cos_u * cos_mv
    ) * cos_nv * sin_omega_t
    v_parallel_theta = float(Bv) * cos_u * float(M_v) * cos_mv * cos_nv * sin_omega_t
    v_parallel_zeta = -float(Bv) * cos_u * sin_mv * float(N_v) * sin_nv * sin_omega_t
    v_parallel_t = float(Bv) * cos_u * sin_mv * cos_nv * float(omega) * cos_omega_t
    return v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_local_poisson_bracket(
    f_u: jnp.ndarray,
    f_theta: jnp.ndarray,
    f_zeta: jnp.ndarray,
    g_u: jnp.ndarray,
    g_theta: jnp.ndarray,
    g_zeta: jnp.ndarray,
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    x, theta_shift, _, zeta = coordinates_halo
    cos_shift, sin_shift, R, _, J, D, _, _, _ = _shifted_torus_local_geometry_quantities(
        x,
        theta_shift,
        zeta,
    )
    del cos_shift
    return (
        1.0
        / (J * D)
        * (
            -float(alpha_value) * float(iota) * x * sin_shift * (f_theta * g_zeta - f_zeta * g_theta)
            + float(iota) * x**2 * (f_zeta * g_u - f_u * g_zeta)
            + R**2 * (f_u * g_theta - f_theta * g_u)
        )
    )


def _shifted_torus_local_curvature(
    field_u: jnp.ndarray,
    field_theta: jnp.ndarray,
    field_zeta: jnp.ndarray,
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    x, theta_shift, _, zeta = coordinates_halo
    _cos_shift, sin_shift, R, Q, J, D, P, E, A_term = (
        _shifted_torus_local_geometry_quantities(x, theta_shift, zeta)
    )
    K_u = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * x * R * sin_shift / D
            + 2.0 * x * R**3 * sin_shift / D**3
            - x * R**2 * sin_shift * E / (D * J)
        )
    )
    K_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * P / D
            - 2.0 * R**2 * A_term / D**3
            + R**2 * Q * (R + x * P) / (D * J)
        )
    )
    K_zeta = (
        float(iota)
        / (2.0 * J)
        * (
            x * (2.0 + float(alpha_value) * _cos_shift) / D
            - 2.0 * x**2 * A_term / D**3
            + 2.0 * float(alpha_value) * x**2 * R * sin_shift**2 / D**3
            + (x**2 * Q * (R + x * P) - float(alpha_value) * x**2 * sin_shift**2 * E) / (D * J)
        )
    )
    return K_u * field_u + K_theta * field_theta + K_zeta * field_zeta


def _shifted_torus_local_grad_parallel(
    field_theta: jnp.ndarray,
    field_zeta: jnp.ndarray,
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    x, theta_shift, _, zeta = coordinates_halo
    *_unused, D, _P, _E, _A_term = _shifted_torus_local_geometry_quantities(
        x,
        theta_shift,
        zeta,
    )
    return (float(iota) * field_theta + field_zeta) / D


def _shifted_torus_local_exact_state(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> Fci2FieldState:
    density, *_ = _shifted_torus_local_density_derivatives(coordinates_halo, time)
    v_parallel, *_ = _shifted_torus_local_v_parallel_derivatives(coordinates_halo, time)
    return Fci2FieldState(
        density=density,
        v_parallel=v_parallel,
        density_background=jnp.ones_like(density),
    )


def _shifted_torus_local_density_source(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
    *,
    parameters: Fci2FieldRhsParameters,
    bmag_halo: jnp.ndarray,
) -> jnp.ndarray:
    phi, phi_u, phi_theta, phi_zeta, _phi_t = _shifted_torus_local_phi_derivatives(
        coordinates_halo,
        time,
    )
    del phi
    density, density_u, density_theta, density_zeta, density_t = (
        _shifted_torus_local_density_derivatives(coordinates_halo, time)
    )
    _v_parallel, _v_u, v_parallel_theta, v_parallel_zeta, _v_t = (
        _shifted_torus_local_v_parallel_derivatives(coordinates_halo, time)
    )
    poisson = _shifted_torus_local_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        density_u,
        density_theta,
        density_zeta,
        coordinates_halo,
    )
    curvature_density = _shifted_torus_local_curvature(
        density_u,
        density_theta,
        density_zeta,
        coordinates_halo,
    )
    curvature_phi = _shifted_torus_local_curvature(
        phi_u,
        phi_theta,
        phi_zeta,
        coordinates_halo,
    )
    grad_parallel_v = _shifted_torus_local_grad_parallel(
        v_parallel_theta,
        v_parallel_zeta,
        coordinates_halo,
    )
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(bmag_halo, dtype=jnp.float64), 1.0e-30)
    return (
        density_t
        + (1.0 / (rho_star_value * bmag)) * poisson
        - (2.0 / bmag) * curvature_density
        + (2.0 * density / bmag) * curvature_phi
        + density * grad_parallel_v
    )


def _shifted_torus_local_v_parallel_source(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
    *,
    parameters: Fci2FieldRhsParameters,
    bmag_halo: jnp.ndarray,
) -> jnp.ndarray:
    _phi, phi_u, phi_theta, phi_zeta, _phi_t = _shifted_torus_local_phi_derivatives(
        coordinates_halo,
        time,
    )
    v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t = (
        _shifted_torus_local_v_parallel_derivatives(coordinates_halo, time)
    )
    del v_parallel
    poisson = _shifted_torus_local_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        v_parallel_u,
        v_parallel_theta,
        v_parallel_zeta,
        coordinates_halo,
    )
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(bmag_halo, dtype=jnp.float64), 1.0e-30)
    return v_parallel_t + (1.0 / (rho_star_value * bmag)) * poisson


def _shifted_torus_geometry_quantities(geometry: FciGeometry3D) -> tuple[jnp.ndarray, ...]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    x = jnp.asarray(logical_grid[..., 0], dtype=jnp.float64)
    theta = jnp.asarray(logical_grid[..., 1], dtype=jnp.float64)
    zeta = jnp.asarray(logical_grid[..., 2], dtype=jnp.float64)
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * x + x * cos_shift
    Q = 1.0 + float(alpha_value) * cos_shift
    J = x * R * Q
    D2 = (float(iota) ** 2) * x**2 + R**2
    D = jnp.sqrt(D2)
    P = float(alpha_value) + cos_shift
    E = x * Q + float(alpha_value) * R
    A_term = (float(iota) ** 2) * x + R * P
    return x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, P, E, A_term


def _shifted_torus_phi_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, cos_shift, sin_shift, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mphi = jnp.sin(float(M_phi) * theta_shift)
    cos_mphi = jnp.cos(float(M_phi) * theta_shift)
    sin_nphi = jnp.sin(float(N_phi) * zeta)
    cos_nphi = jnp.cos(float(N_phi) * zeta)
    cos_omega_t = jnp.cos(float(omega) * time)
    sin_omega_t = jnp.sin(float(omega) * time)

    phi = float(A) * sin_u * cos_mphi * sin_nphi * cos_omega_t
    phi_u = float(A) * (
        jnp.pi * cos_u * cos_mphi - float(sigma) * float(M_phi) * sin_u * sin_mphi
    ) * sin_nphi * cos_omega_t
    phi_theta = -float(A) * float(M_phi) * sin_u * sin_mphi * sin_nphi * cos_omega_t
    phi_zeta = float(A) * float(N_phi) * sin_u * cos_mphi * cos_nphi * cos_omega_t
    phi_t = -float(A) * float(omega) * sin_u * cos_mphi * sin_nphi * sin_omega_t
    return phi, phi_u, phi_theta, phi_zeta, phi_t


def _shifted_torus_density_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, _, _, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    phi, phi_u, phi_theta, phi_zeta, phi_t = _shifted_torus_phi_derivatives(geometry, time)
    n0 = jnp.ones_like(x)
    n0_u = jnp.zeros_like(x)
    exp_phi = jnp.exp(phi)
    density = n0 * exp_phi
    density_u = n0_u * exp_phi + n0 * exp_phi * phi_u
    density_theta = density * phi_theta
    density_zeta = density * phi_zeta
    density_t = density * phi_t
    return density, density_u, density_theta, density_zeta, density_t


def _shifted_torus_v_parallel_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, zeta, _, _, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    sin_u = jnp.sin(jnp.pi * x)
    cos_u = jnp.cos(jnp.pi * x)
    sin_mv = jnp.sin(float(M_v) * theta_shift)
    cos_mv = jnp.cos(float(M_v) * theta_shift)
    sin_nv = jnp.sin(float(N_v) * zeta)
    cos_nv = jnp.cos(float(N_v) * zeta)
    sin_omega_t = jnp.sin(float(omega) * time)
    cos_omega_t = jnp.cos(float(omega) * time)

    v_parallel = float(Bv) * cos_u * sin_mv * cos_nv * sin_omega_t
    v_parallel_u = float(Bv) * (
        -jnp.pi * sin_u * sin_mv + float(sigma) * float(M_v) * cos_u * cos_mv
    ) * cos_nv * sin_omega_t
    v_parallel_theta = float(Bv) * cos_u * float(M_v) * cos_mv * cos_nv * sin_omega_t
    v_parallel_zeta = -float(Bv) * cos_u * sin_mv * float(N_v) * sin_nv * sin_omega_t
    v_parallel_t = float(Bv) * cos_u * sin_mv * cos_nv * float(omega) * cos_omega_t
    return v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_poisson_bracket(
    f_u: jnp.ndarray,
    f_theta: jnp.ndarray,
    f_zeta: jnp.ndarray,
    g_u: jnp.ndarray,
    g_theta: jnp.ndarray,
    g_zeta: jnp.ndarray,
    geometry: FciGeometry3D,
) -> jnp.ndarray:
    x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    return (
        1.0
        / (J * D)
        * (
            -float(alpha_value) * float(iota) * x * sin_shift * (f_theta * g_zeta - f_zeta * g_theta)
            + float(iota) * x**2 * (f_zeta * g_u - f_u * g_zeta)
            + R**2 * (f_u * g_theta - f_theta * g_u)
        )
    )


def _shifted_torus_curvature(field_u: jnp.ndarray, field_theta: jnp.ndarray, field_zeta: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    x, theta_shift, zeta, cos_shift, sin_shift, R, Q, J, D, P, E, A_term = _shifted_torus_geometry_quantities(geometry)
    K_u = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * x * R * sin_shift / D
            + 2.0 * x * R**3 * sin_shift / D**3
            - x * R**2 * sin_shift * E / (D * J)
        )
    )
    K_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * P / D
            - 2.0 * R**2 * A_term / D**3
            + R**2 * Q * (R + x * P) / (D * J)
        )
    )
    K_zeta = (
        float(iota)
        / (2.0 * J)
        * (
            x * (2.0 + float(alpha_value) * cos_shift) / D
            - 2.0 * x**2 * A_term / D**3
            + 2.0 * float(alpha_value) * x**2 * R * sin_shift**2 / D**3
            + (x**2 * Q * (R + x * P) - float(alpha_value) * x**2 * sin_shift**2 * E) / (D * J)
        )
    )
    return K_u * field_u + K_theta * field_theta + K_zeta * field_zeta


def _shifted_torus_grad_parallel(field_theta: jnp.ndarray, field_zeta: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    _, _, _, _, _, _, _, _, D, _, _, _ = _shifted_torus_geometry_quantities(geometry)
    return (float(iota) * field_theta + field_zeta) / D


def _shifted_torus_density_source(geometry: FciGeometry3D, time: float, *, parameters: Fci2FieldRhsParameters) -> jnp.ndarray:
    phi, phi_u, phi_theta, phi_zeta, phi_t = _shifted_torus_phi_derivatives(geometry, time)
    density, density_u, density_theta, density_zeta, density_t = _shifted_torus_density_derivatives(geometry, time)
    v_parallel, _, v_parallel_theta, v_parallel_zeta, _ = _shifted_torus_v_parallel_derivatives(geometry, time)
    bmag = geometry.cell_bfield.Bmag
    poisson = _shifted_torus_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        density_u,
        density_theta,
        density_zeta,
        geometry,
    )
    curvature_density = _shifted_torus_curvature(density_u, density_theta, density_zeta, geometry)
    curvature_phi = _shifted_torus_curvature(phi_u, phi_theta, phi_zeta, geometry)
    grad_parallel_v = _shifted_torus_grad_parallel(v_parallel_theta, v_parallel_zeta, geometry)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    return density_t + (1.0 / (rho_star_value * bmag)) * poisson - (2.0 / bmag) * curvature_density + (2.0 * density / bmag) * curvature_phi + density * grad_parallel_v


def _shifted_torus_v_parallel_source(geometry: FciGeometry3D, time: float, *, parameters: Fci2FieldRhsParameters) -> jnp.ndarray:
    phi, phi_u, phi_theta, phi_zeta, _ = _shifted_torus_phi_derivatives(geometry, time)
    v_parallel, v_parallel_u, v_parallel_theta, v_parallel_zeta, v_parallel_t = _shifted_torus_v_parallel_derivatives(geometry, time)
    bmag = geometry.cell_bfield.Bmag
    poisson = _shifted_torus_poisson_bracket(
        phi_u,
        phi_theta,
        phi_zeta,
        v_parallel_u,
        v_parallel_theta,
        v_parallel_zeta,
        geometry,
    )
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    return v_parallel_t + (1.0 / (rho_star_value * bmag)) * poisson


def _state_partition_spec() -> Fci2FieldState:
    spec = P(*MESH_AXIS_NAMES)
    return Fci2FieldState(
        density=spec,
        v_parallel=spec,
        density_background=spec,
    )


def _put_state_on_mesh(state: Fci2FieldState, mesh: Mesh) -> Fci2FieldState:
    sharding = NamedSharding(mesh, P(*MESH_AXIS_NAMES))
    return Fci2FieldState(
        density=jax.device_put(jnp.asarray(state.density, dtype=jnp.float64), sharding),
        v_parallel=jax.device_put(jnp.asarray(state.v_parallel, dtype=jnp.float64), sharding),
        density_background=jax.device_put(
            jnp.asarray(state.density_background, dtype=jnp.float64),
            sharding,
        ),
    )


def _gather_state_from_mesh(state: Fci2FieldState) -> Fci2FieldState:
    return Fci2FieldState(
        density=jnp.asarray(jax.device_get(state.density), dtype=jnp.float64),
        v_parallel=jnp.asarray(jax.device_get(state.v_parallel), dtype=jnp.float64),
        density_background=jnp.asarray(
            jax.device_get(state.density_background),
            dtype=jnp.float64,
        ),
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


def _apply_local_owned_dirichlet_to_field(
    field_owned: jnp.ndarray,
    exact_owned: jnp.ndarray,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    field_owned = jnp.asarray(field_owned, dtype=jnp.float64)
    exact_owned = jnp.asarray(exact_owned, dtype=jnp.float64)
    result = field_owned
    result = result.at[0, :, :].set(
        jnp.where(domain.runtime_has_physical_lower(0), exact_owned[0, :, :], result[0, :, :])
    )
    result = result.at[-1, :, :].set(
        jnp.where(domain.runtime_has_physical_upper(0), exact_owned[-1, :, :], result[-1, :, :])
    )
    return result


def _apply_local_owned_dirichlet_to_state(
    state_owned: Fci2FieldState,
    exact_owned: Fci2FieldState,
    domain: LocalDomain3D,
) -> Fci2FieldState:
    return Fci2FieldState(
        density=_apply_local_owned_dirichlet_to_field(
            state_owned.density,
            exact_owned.density,
            domain,
        ),
        v_parallel=_apply_local_owned_dirichlet_to_field(
            state_owned.v_parallel,
            exact_owned.v_parallel,
            domain,
        ),
        density_background=state_owned.density_background,
    )


def _build_local_radial_dirichlet_face_bc(
    values_halo: jnp.ndarray,
    domain: LocalDomain3D,
) -> LocalBoundaryFaceBC3D:
    layout = domain.layout
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    return _build_local_radial_dirichlet_face_bc_from_values(
        values_halo[h, h : h + ny, h : h + nz],
        values_halo[h + nx - 1, h : h + ny, h : h + nz],
        domain,
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


def _shifted_torus_local_x_face_coordinates(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    layout = domain.layout
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    x_faces = jnp.asarray(
        [geometry.grid.x.faces_halo[h], geometry.grid.x.faces_halo[h + nx]],
        dtype=jnp.float64,
    )
    x, theta, zeta = jnp.meshgrid(
        x_faces,
        geometry.grid.y.centers_halo[h : h + ny],
        geometry.grid.z.centers_halo[h : h + nz],
        indexing="ij",
    )
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _split_local_x_face_values(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.asarray(values[0], dtype=jnp.float64), jnp.asarray(values[-1], dtype=jnp.float64)


def _coordinates_from_invariants(
    invariants: _ShiftedTorus2FieldInvariantBundle,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        invariants.coord_x,
        invariants.coord_theta_shift,
        invariants.coord_theta,
        invariants.coord_zeta,
    )


def _face_coordinates_from_invariants(
    invariants: _ShiftedTorus2FieldInvariantBundle,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        invariants.face_x,
        invariants.face_theta_shift,
        invariants.face_theta,
        invariants.face_zeta,
    )


def _build_local_2field_invariants(
    shard_index: tuple[int, int, int],
    *,
    owned_shape: tuple[int, int, int],
    halo_width: int,
    global_shape: tuple[int, int, int],
    domain: LocalDomain3D,
) -> _ShiftedTorus2FieldInvariantBundle:
    local_geometry = build_shifted_torus_local_geometry(
        owned_shape,
        halo_width,
        global_shape=global_shape,
        shard_index=shard_index,
    )
    coordinates_halo = _shifted_torus_local_coordinates(local_geometry)
    face_coordinates = _shifted_torus_local_x_face_coordinates(local_geometry, domain)
    return _ShiftedTorus2FieldInvariantBundle(
        coord_x=jnp.asarray(coordinates_halo[0], dtype=jnp.float64),
        coord_theta_shift=jnp.asarray(coordinates_halo[1], dtype=jnp.float64),
        coord_theta=jnp.asarray(coordinates_halo[2], dtype=jnp.float64),
        coord_zeta=jnp.asarray(coordinates_halo[3], dtype=jnp.float64),
        face_x=jnp.asarray(face_coordinates[0], dtype=jnp.float64),
        face_theta_shift=jnp.asarray(face_coordinates[1], dtype=jnp.float64),
        face_theta=jnp.asarray(face_coordinates[2], dtype=jnp.float64),
        face_zeta=jnp.asarray(face_coordinates[3], dtype=jnp.float64),
        bmag_halo=jnp.asarray(local_geometry.cell_bfield.Bmag_halo, dtype=jnp.float64),
        curvature_coefficients_owned=jnp.asarray(
            build_local_curvature_coefficients(local_geometry, domain),
            dtype=jnp.float64,
        ),
    )


def _build_local_2field_stage_data(
    invariants: _ShiftedTorus2FieldInvariantBundle,
    time: float | jax.Array,
    *,
    parameters: Fci2FieldRhsParameters,
) -> _ShiftedTorus2FieldStageData:
    coordinates_halo = _coordinates_from_invariants(invariants)
    face_coordinates = _face_coordinates_from_invariants(invariants)
    exact_halo = _shifted_torus_local_exact_state(coordinates_halo, time)
    phi_face = _shifted_torus_local_phi_derivatives(face_coordinates, time)[0]
    density_face = _shifted_torus_local_density_derivatives(face_coordinates, time)[0]
    v_parallel_face = _shifted_torus_local_v_parallel_derivatives(face_coordinates, time)[0]
    phi_lower, phi_upper = _split_local_x_face_values(phi_face)
    density_lower, density_upper = _split_local_x_face_values(density_face)
    v_parallel_lower, v_parallel_upper = _split_local_x_face_values(v_parallel_face)
    return _ShiftedTorus2FieldStageData(
        exact_halo=exact_halo,
        density_source_halo=_shifted_torus_local_density_source(
            coordinates_halo,
            time,
            parameters=parameters,
            bmag_halo=invariants.bmag_halo,
        ),
        v_parallel_source_halo=_shifted_torus_local_v_parallel_source(
            coordinates_halo,
            time,
            parameters=parameters,
            bmag_halo=invariants.bmag_halo,
        ),
        phi_face_lower=phi_lower,
        phi_face_upper=phi_upper,
        density_face_lower=density_lower,
        density_face_upper=density_upper,
        v_parallel_face_lower=v_parallel_lower,
        v_parallel_face_upper=v_parallel_upper,
    )


def _build_local_2field_rk4_stage_data(
    invariants: _ShiftedTorus2FieldInvariantBundle,
    step_time: float | jax.Array,
    step_timestep: float | jax.Array,
    *,
    parameters: Fci2FieldRhsParameters,
) -> _ShiftedTorus2FieldRk4StageData:
    half_step = 0.5 * step_timestep
    return _ShiftedTorus2FieldRk4StageData(
        stage_1=_build_local_2field_stage_data(invariants, step_time, parameters=parameters),
        stage_2=_build_local_2field_stage_data(
            invariants,
            step_time + half_step,
            parameters=parameters,
        ),
        stage_3=_build_local_2field_stage_data(
            invariants,
            step_time + half_step,
            parameters=parameters,
        ),
        stage_4=_build_local_2field_stage_data(
            invariants,
            step_time + step_timestep,
            parameters=parameters,
        ),
    )


def _prepare_local_shifted_torus_2field_stage_state(
    state_owned: Fci2FieldState,
    stage_data: _ShiftedTorus2FieldStageData,
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
) -> PreparedLocalState3D:
    state_halo = inject_owned_state_to_halo(state_owned, domain.layout)
    state_halo = Fci2FieldState(
        density=topology_filler(halo_exchange(state_halo.density, domain), domain),
        v_parallel=topology_filler(halo_exchange(state_halo.v_parallel, domain), domain),
        density_background=topology_filler(
            halo_exchange(state_halo.density_background, domain),
            domain,
        ),
    )
    face_bc_bundle = _ShiftedTorus2FieldFaceBCBundle(
        density=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.density_face_lower,
            stage_data.density_face_upper,
            domain,
        ),
        phi=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.phi_face_lower,
            stage_data.phi_face_upper,
            domain,
        ),
        v_parallel=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.v_parallel_face_lower,
            stage_data.v_parallel_face_upper,
            domain,
        ),
        density_background=_build_local_radial_dirichlet_face_bc_from_values(jnp.ones_like(stage_data.density_face_lower), jnp.ones_like(stage_data.density_face_upper), domain),
    )
    prepared_state_halo = Fci2FieldState(
        density=physical_ghost_filler(state_halo.density, domain, face_bc_bundle.density),
        v_parallel=physical_ghost_filler(
            state_halo.v_parallel,
            domain,
            face_bc_bundle.v_parallel,
        ),
        density_background=physical_ghost_filler(
            state_halo.density_background,
            domain,
            face_bc_bundle.density_background,
        ),
    )
    return PreparedLocalState3D(
        state_halo=prepared_state_halo,
        boundary_data=LocalBoundaryData3D(face_bc=face_bc_bundle),
    )


@dataclass(frozen=True)
class LocalShiftedTorus2FieldRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    halo_exchange: HaloExchange3D
    topology_filler: TopologyHaloFiller3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci2FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray

    def evaluate_stage(
        self,
        state_owned: Fci2FieldState,
        stage_data: _ShiftedTorus2FieldStageData,
        carry: None,
    ) -> tuple[Fci2FieldState, None, jnp.ndarray]:
        del carry
        prepared = _prepare_local_shifted_torus_2field_stage_state(
            state_owned,
            stage_data,
            self.domain,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
            physical_ghost_filler=self.physical_ghost_filler,
        )
        density_halo = jnp.asarray(prepared.state_halo.density, dtype=jnp.float64)
        v_parallel_halo = jnp.asarray(
            prepared.state_halo.v_parallel,
            dtype=jnp.float64,
        )
        background_halo = jnp.asarray(
            prepared.state_halo.density_background,
            dtype=jnp.float64,
        )
        phi_halo = jnp.log(
            jnp.maximum(density_halo, 1.0e-30) / jnp.maximum(background_halo, 1.0e-30)
        )
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        density_stencil = build_local_stencil_from_field(
            density_halo,
            self.geometry,
            context,
        )
        phi_stencil = build_local_stencil_from_field(phi_halo, self.geometry, context)
        v_parallel_stencil = build_local_stencil_from_field(
            v_parallel_halo,
            self.geometry,
            context,
        )

        density_owned = density_halo[self.domain.layout.owned_slices_cell]
        bmag_owned = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        curvature_density = local_curvature_op(
            density_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        parallel_velocity_gradient = local_grad_parallel_op_direct(
            v_parallel_stencil,
            self.geometry,
        )
        poisson_v_parallel = local_poisson_bracket_op(
            phi_stencil,
            v_parallel_stencil,
            self.geometry,
        )
        density_rhs = (
            -(poisson_density / (rho_star_value * bmag_owned))
            + (2.0 / bmag_owned) * curvature_density
            - (2.0 * density_owned / bmag_owned) * curvature_phi
            - density_owned * parallel_velocity_gradient
        )
        v_parallel_rhs = -(poisson_v_parallel / (rho_star_value * bmag_owned))
        density_rhs = density_rhs + stage_data.density_source_halo[self.domain.layout.owned_slices_cell]
        v_parallel_rhs = v_parallel_rhs + stage_data.v_parallel_source_halo[self.domain.layout.owned_slices_cell]
        rhs = Fci2FieldState(
            density=jnp.asarray(density_rhs, dtype=jnp.float64),
            v_parallel=jnp.asarray(v_parallel_rhs, dtype=jnp.float64),
            density_background=jnp.zeros(self.domain.layout.owned_shape, dtype=jnp.float64),
        )
        return rhs, None, jnp.zeros((3,), dtype=jnp.float64)


def simulate_mms_2field_shifted_torus(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    show_progress: bool = False,
) -> tuple[Fci2FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve the shifted-torus MMS system through a shard-map local RHS."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    domain = build_shifted_torus_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))
    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_exact = _shifted_torus_exact_state(geometry, 0.0)
    total_runtime = 0.0
    wall_step_times: list[float] = []
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_exact.density, dtype=jnp.float32)]
    v_parallel_history: list[jnp.ndarray] = [jnp.asarray(initial_exact.v_parallel, dtype=jnp.float32)]

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = _put_state_on_mesh(initial_exact, mesh)
        state_spec = _state_partition_spec()
        host_invariant_domain = LocalDomain3D(
            shard_spec=domain.shard_spec,
            layout=domain.layout,
            mesh_axis_names=(None, None, None),
        )
        sample_invariants = expand_local_shard_pytree(
            _build_local_2field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            )
        )
        invariant_spec = local_shard_pytree_partition_spec(sample_invariants)

        def invariant_kernel() -> _ShiftedTorus2FieldInvariantBundle:
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            return expand_local_shard_pytree(
                _build_local_2field_invariants(
                    shard_index,
                    owned_shape=owned_shape,
                    halo_width=halo_width,
                    global_shape=geometry.shape,
                    domain=domain,
                )
            )

        def source_kernel(
            local_invariants: _ShiftedTorus2FieldInvariantBundle,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> _ShiftedTorus2FieldRk4StageData:
            local_invariants = extract_local_shard_pytree(local_invariants)
            return expand_local_shard_pytree(
                _build_local_2field_rk4_stage_data(
                    local_invariants,
                    step_time,
                    step_timestep,
                    parameters=parameters,
                )
            )

        def kernel(
            state_owned: Fci2FieldState,
            local_invariants: _ShiftedTorus2FieldInvariantBundle,
            rk_stage_data: _ShiftedTorus2FieldRk4StageData,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> Fci2FieldState:
            local_invariants = extract_local_shard_pytree(local_invariants)
            rk_stage_data = extract_local_shard_pytree(rk_stage_data)
            shard_index = tuple(lax.axis_index(name) for name in MESH_AXIS_NAMES)
            local_geometry = build_shifted_torus_local_geometry(
                owned_shape,
                halo_width,
                global_shape=geometry.shape,
                shard_index=shard_index,
            )
            rhs = LocalShiftedTorus2FieldRhs(
                geometry=local_geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=topology_filler,
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=local_invariants.curvature_coefficients_owned,
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
            return next_state

        mapped_invariant_kernel = shard_map(
            invariant_kernel,
            mesh=mesh,
            in_specs=(),
            out_specs=invariant_spec,
            check_rep=False,
        )
        invariants = jax.jit(mapped_invariant_kernel)()
        sample_stage_data = _build_local_2field_rk4_stage_data(
            _build_local_2field_invariants(
                (0, 0, 0),
                owned_shape=owned_shape,
                halo_width=halo_width,
                global_shape=geometry.shape,
                domain=host_invariant_domain,
            ),
            0.0,
            dt,
            parameters=parameters,
        )
        stage_data_spec = local_shard_pytree_partition_spec(
            expand_local_shard_pytree(sample_stage_data)
        )
        mapped_source_kernel = shard_map(
            source_kernel,
            mesh=mesh,
            in_specs=(invariant_spec, P(), P()),
            out_specs=stage_data_spec,
            check_rep=False,
        )
        compiled_source_kernel = jax.jit(mapped_source_kernel)
        mapped_step_kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(
                state_spec,
                invariant_spec,
                stage_data_spec,
                P(),
                P(),
            ),
            out_specs=state_spec,
            check_rep=False,
        )
        step_kernel = jax.jit(mapped_step_kernel)

        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                f"shifted_torus_2field RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

        for step_index in range(steps):
            step_start = time_module.perf_counter()
            rk_stage_data = compiled_source_kernel(
                invariants,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            state = step_kernel(
                state,
                invariants,
                rk_stage_data,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            elapsed = time_module.perf_counter() - step_start
            total_runtime += elapsed
            wall_step_times.append(elapsed)
            time_value += dt
            times.append(time_value)
            gathered_state = _gather_state_from_mesh(state)
            density_history.append(jnp.asarray(gathered_state.density, dtype=jnp.float32))
            v_parallel_history.append(
                jnp.asarray(gathered_state.v_parallel, dtype=jnp.float32)
            )
            if show_progress:
                print(
                    "\r"
                    f"shifted_torus_2field RK4 progress: "
                    f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )

        if show_progress:
            print()

        final_state = _gather_state_from_mesh(state)

    if wall_step_times:
        print(
            "shifted_torus_2field mean timings per RK step: "
            f"wall={np.mean(np.asarray(wall_step_times, dtype=np.float64)):.6e} s"
        )

    return (
        final_state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(v_parallel_history, axis=0),
    )


def _shifted_torus_z_cut_indices(geometry: FciGeometry3D, count: int) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, count)
    return tuple(int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts)


def _shifted_torus_field_slices(field: jnp.ndarray, z_indices: tuple[int, ...]) -> jnp.ndarray:
    return jnp.stack([field[:, :, z_index] for z_index in z_indices], axis=0)


def _combined_error_statistics(final_state: Fci2FieldState, geometry: FciGeometry3D, time: float) -> tuple[float, float, float]:
    exact = _shifted_torus_exact_state(geometry, time)
    density_error = jnp.abs(final_state.density - exact.density)[1:-1, :, :]
    v_parallel_error = jnp.abs(final_state.v_parallel - exact.v_parallel)[1:-1, :, :]
    error = jnp.concatenate(
        [
            jnp.ravel(density_error),
            jnp.ravel(v_parallel_error),
        ]
    )
    return float(jnp.sqrt(jnp.mean(error**2))), float(jnp.median(error)), float(jnp.max(error))


def _plot_final_slices(
    state: Fci2FieldState,
    exact_state: Fci2FieldState,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 2)

    density = np.asarray(state.density, dtype=np.float64)
    v_parallel = np.asarray(state.v_parallel, dtype=np.float64)
    exact_density = np.asarray(exact_state.density, dtype=np.float64)
    exact_v_parallel = np.asarray(exact_state.v_parallel, dtype=np.float64)

    density_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(density), z_indices), dtype=np.float64)
    exact_density_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(exact_density), z_indices), dtype=np.float64)
    v_parallel_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(v_parallel), z_indices), dtype=np.float64)
    exact_v_parallel_slices = np.asarray(_shifted_torus_field_slices(jnp.asarray(exact_v_parallel), z_indices), dtype=np.float64)

    density_vmax = float(np.max(np.abs(np.stack([density_slices, exact_density_slices], axis=0))))
    v_parallel_vmax = float(np.max(np.abs(np.stack([v_parallel_slices, exact_v_parallel_slices], axis=0))))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    density_im = None
    v_parallel_im = None
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    for cut_index, z_index in enumerate(z_indices):
        density_slice = density_slices[cut_index]
        v_parallel_slice = v_parallel_slices[cut_index]
        exact_density_slice = exact_density_slices[cut_index]
        exact_v_parallel_slice = exact_v_parallel_slices[cut_index]

        density_im = axes[0, cut_index].pcolormesh(theta_grid, radius_grid, density_slice, shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
        axes[0, cut_index].set_theta_zero_location("E")
        axes[0, cut_index].set_theta_direction(-1)
        axes[0, cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[0, cut_index].set_title(f"sim, zeta={z_values[z_index]:.3f}")
        axes[0, cut_index].set_yticklabels([])

        density_im = axes[0, 2 + cut_index].pcolormesh(theta_grid, radius_grid, exact_density_slice, shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
        axes[0, 2 + cut_index].set_theta_zero_location("E")
        axes[0, 2 + cut_index].set_theta_direction(-1)
        axes[0, 2 + cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[0, 2 + cut_index].set_title(f"exact, zeta={z_values[z_index]:.3f}")
        axes[0, 2 + cut_index].set_yticklabels([])

        v_parallel_im = axes[1, cut_index].pcolormesh(theta_grid, radius_grid, v_parallel_slice, shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
        axes[1, cut_index].set_theta_zero_location("E")
        axes[1, cut_index].set_theta_direction(-1)
        axes[1, cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[1, cut_index].set_title(f"sim, zeta={z_values[z_index]:.3f}")
        axes[1, cut_index].set_yticklabels([])

        v_parallel_im = axes[1, 2 + cut_index].pcolormesh(theta_grid, radius_grid, exact_v_parallel_slice, shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
        axes[1, 2 + cut_index].set_theta_zero_location("E")
        axes[1, 2 + cut_index].set_theta_direction(-1)
        axes[1, 2 + cut_index].set_ylim(0.0, float(x_values[-1]))
        axes[1, 2 + cut_index].set_title(f"exact, zeta={z_values[z_index]:.3f}")
        axes[1, 2 + cut_index].set_yticklabels([])

    if density_im is not None:
        fig.colorbar(density_im, ax=axes[0, :].ravel().tolist(), shrink=0.88, pad=0.02)
    if v_parallel_im is not None:
        fig.colorbar(v_parallel_im, ax=axes[1, :].ravel().tolist(), shrink=0.88, pad=0.02)

    fig.suptitle(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_shifted_torus_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    v_parallel_history: jnp.ndarray,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
    frame_stride: int = 5,
) -> None:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 4)

    density_data = np.asarray(density_history, dtype=np.float64)
    v_parallel_data = np.asarray(v_parallel_history, dtype=np.float64)
    frame_indices = np.arange(0, int(times.shape[0]), max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != int(times.shape[0]) - 1:
        frame_indices = np.append(frame_indices, int(times.shape[0]) - 1)
    density_vmax = float(np.max(np.abs(density_data)))
    v_parallel_vmax = float(np.max(np.abs(v_parallel_data)))

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    images = []
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)
    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, float(x_values[-1]))
            ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"density, zeta={z_values[z_indices[col]]:.3f}")
                image = ax.pcolormesh(theta_grid, radius_grid, density_data[0, :, :, z_indices[col]], shading="auto", cmap="viridis", vmin=-density_vmax, vmax=density_vmax)
            else:
                ax.set_title(f"v_parallel, zeta={z_values[z_indices[col]]:.3f}")
                image = ax.pcolormesh(theta_grid, radius_grid, v_parallel_data[0, :, :, z_indices[col]], shading="auto", cmap="coolwarm", vmin=-v_parallel_vmax, vmax=v_parallel_vmax)
            images.append(image)

    suptitle = fig.suptitle(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}")

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for col in range(4):
            images[col].set_array(density_data[actual_index, :, :, z_indices[col]].ravel())
            images[4 + col].set_array(v_parallel_data[actual_index, :, :, z_indices[col]].ravel())
            axes[0, col].set_title(f"density, zeta={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
            axes[1, col].set_title(f"v_parallel, zeta={z_values[z_indices[col]]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"Shifted-torus 2-field MMS fields at resolution {int(resolution)}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)



def run_shifted_torus_2field_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = tf,
    base_steps: int = num_steps,
    rho_star_value: float = rho_star,
    plot: bool = False,
    plot_path: str | None = None,
    plot_slices: bool = False,
    movie: bool = False,
    movie_stride: int = 5,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    final_resolution_state: Fci2FieldState | None = None
    final_resolution_geometry: FciGeometry3D | None = None
    final_resolution: int | None = None
    final_resolution_times: jnp.ndarray | None = None
    final_resolution_density_history: jnp.ndarray | None = None
    final_resolution_v_parallel_history: jnp.ndarray | None = None

    for resolution in resolutions:
        shape = (int(resolution), int(resolution), int(resolution))
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = build_shifted_torus_2field_geometry(shape)
        steps = _resolution_step_count(int(resolution), base_steps=base_steps)
        dt = float(final_time) / float(steps)
        print(
            f"Starting shifted_torus_2field MMS run: resolution={int(resolution)}, "
            f"shard_counts={shard_counts}, steps={steps}, dt={dt:.6e}"
        )
        start = time_module.perf_counter()
        try:
            final_state, times, density_history, v_parallel_history = simulate_mms_2field_shifted_torus(
                geometry,
                shard_counts=shard_counts,
                halo_width=halo_width,
                final_time=final_time,
                timestep=dt,
                rho_star_value=rho_star_value,
                show_progress=show_progress,
            )
            elapsed = time_module.perf_counter() - start
            mean_error, median_error, max_error = _combined_error_statistics(
                final_state,
                geometry,
                final_time,
            )
        except FloatingPointError as exc:
            elapsed = time_module.perf_counter() - start
            print(
                f"WARNING: resolution={int(resolution)} shard_counts={shard_counts} "
                f"failed after {elapsed:.6e} s: {exc}"
            )
            continue

        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        print(
            f"N={int(resolution)}: shard_counts={shard_counts}, steps={steps}, "
            f"total_runtime={elapsed:.6e} s, avg_step_runtime={elapsed / float(steps):.6e} s, "
            f"L2={mean_error:.6e}, median={median_error:.6e}, Linf={max_error:.6e}"
        )
        final_resolution_state = final_state
        final_resolution_geometry = geometry
        final_resolution = int(resolution)
        final_resolution_times = times
        final_resolution_density_history = density_history
        final_resolution_v_parallel_history = v_parallel_history

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
        print(f"shifted_torus_2field L2 convergence order: {l2_order:.6f}")
        print(f"shifted_torus_2field Linf convergence order: {max_order:.6f}")

        if plot:
            import matplotlib.pyplot as plt

            output_path = Path(plot_path or "shifted_torus_2field_convergence.png")
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
            ax.set_title(f"Shifted-torus 2-field MMS convergence ({shard_counts})")
            ax.grid(True, which="both", linestyle=":", alpha=0.45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
    elif plot:
        print("WARNING: fewer than two successful resolutions, skipping convergence plot.")

    output_base = Path(plot_path).parent if plot_path else Path(".")
    if plot_slices and final_resolution_state is not None and final_resolution_geometry is not None and final_resolution is not None:
        final_exact_state = _shifted_torus_exact_state(final_resolution_geometry, final_time)
        _plot_final_slices(
            final_resolution_state,
            final_exact_state,
            final_resolution_geometry,
            final_resolution,
            str(output_base / "shifted_torus_2field_slices.png"),
        )

    if (
        movie
        and final_resolution_times is not None
        and final_resolution_density_history is not None
        and final_resolution_v_parallel_history is not None
        and final_resolution_geometry is not None
        and final_resolution is not None
    ):
        _save_shifted_torus_movie(
            final_resolution_times,
            final_resolution_density_history,
            final_resolution_v_parallel_history,
            final_resolution_geometry,
            final_resolution,
            str(output_base / "shifted_torus_2field_slices.gif"),
            frame_stride=movie_stride,
        )

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": max_errors,
        "l2_order": l2_order,
        "linf_order": max_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shifted-torus 2-field MMS convergence harness")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[40, 60, 120])
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=tf)
    parser.add_argument("--base-steps", type=int, default=num_steps)
    parser.add_argument("--rho-star", type=float, default=rho_star)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--plot-slices", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--movie-stride", type=int, default=5)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    run_shifted_torus_2field_convergence(
        resolutions=[int(value) for value in args.resolutions],
        shard_counts=tuple(int(value) for value in args.shard_counts),
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        rho_star_value=float(args.rho_star),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        plot_slices=bool(args.plot_slices),
        movie=bool(args.movie),
        movie_stride=int(args.movie_stride),
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
