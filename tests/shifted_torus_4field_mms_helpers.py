from __future__ import annotations

import time as time_module
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.native.fci_model import FciFieldBundle

MESH_AXIS_NAMES = ("x", "y", "z")


class _HelperBundle:
    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(**dict(zip((field.name for field in fields(cls)), children)))


def Fci4FieldState(*args, **kwargs):
    from jax_drb.native.fci_4_field_rhs import Fci4FieldState as _Fci4FieldState

    return _Fci4FieldState(*args, **kwargs)

A_phi = 0.1
A_n = 0.1
A_e = 0.1
A_i = 0.08
a_phi = 0.2
a_n = 0.15
a_e = 0.1
a_i = 0.12
Omega = 2.0 * jnp.pi
M_phi = 2
N_phi = 3
M_n = 3
N_n = 2
M_e = 4
N_e = 3
M_i = 2
N_i = 4
n0 = 1.0
rho_star = 1.0
Te = 1.0
mi_over_me = 1836.0
sigma = 0.75
r0 = 3.0
alpha_value = 0.25
iota = 1.1
c_phi = 3.0
x_min = 0.2
x_max = 1.0
tf = 0.1
num_steps = 50


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus4FieldFaceBCBundle(FciFieldBundle):
    phi: LocalBoundaryFaceBC3D
    density: LocalBoundaryFaceBC3D
    omega: LocalBoundaryFaceBC3D
    v_ion_parallel: LocalBoundaryFaceBC3D
    v_electron_parallel: LocalBoundaryFaceBC3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus4FieldInvariantBundle(_HelperBundle):
    coord_x: jnp.ndarray
    coord_theta_shift: jnp.ndarray
    coord_theta: jnp.ndarray
    coord_zeta: jnp.ndarray
    face_x: jnp.ndarray
    face_theta_shift: jnp.ndarray
    face_theta: jnp.ndarray
    face_zeta: jnp.ndarray
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
class _ShiftedTorus4FieldStageData(_HelperBundle):
    stage_time: jnp.ndarray
    exact_halo: Fci4FieldState
    source_halo: Fci4FieldState
    phi_halo: jnp.ndarray
    phi_face_lower: jnp.ndarray
    phi_face_upper: jnp.ndarray
    density_face_lower: jnp.ndarray
    density_face_upper: jnp.ndarray
    omega_face_lower: jnp.ndarray
    omega_face_upper: jnp.ndarray
    v_ion_face_lower: jnp.ndarray
    v_ion_face_upper: jnp.ndarray
    v_electron_face_lower: jnp.ndarray
    v_electron_face_upper: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ShiftedTorus4FieldRk4StageData(_HelperBundle):
    stage_1: _ShiftedTorus4FieldStageData
    stage_2: _ShiftedTorus4FieldStageData
    stage_3: _ShiftedTorus4FieldStageData
    stage_4: _ShiftedTorus4FieldStageData


def _resolution_step_count(resolution: int, *, base_resolution: int = 20, base_steps: int = num_steps) -> int:
    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


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


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_shifted_torus_4field_geometry(
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
    """Build the shifted-torus FCI geometry used by the 4-field MMS test."""

    from jax_drb.geometry.fci_geometry import (
        BFieldGeometry,
        CellCenteredGrid3D,
        FaceBFieldGeometry,
        FaceMetricGeometry,
        FciGeometry3D,
        FciMaps3D,
        Grid1D,
        MetricGeometry,
        Spacing3D,
        build_fci_maps_from_b_contravariant,
        logical_grid_from_axis_vectors,
    )

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
        g22 = (1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2) / (
            x**2 * (1.0 + float(alpha_value) * cos_theta) ** 2
        )
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
    from jax_drb.geometry.fci_geometry import logical_grid_from_axis_vectors

    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    x = jnp.asarray(logical_grid[..., 0], dtype=jnp.float64)
    theta = jnp.asarray(logical_grid[..., 1], dtype=jnp.float64)
    zeta = jnp.asarray(logical_grid[..., 2], dtype=jnp.float64)
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _shifted_torus_geometry_quantities_scalar(x: float, theta: float) -> tuple[float, float, float, float, float, float, float, float]:
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * x + x * cos_shift
    Q = 1.0 + float(alpha_value) * cos_shift
    J = x * R * Q
    S = (float(iota) ** 2) * x**2 + R**2
    return theta_shift, cos_shift, sin_shift, R, Q, J, S, x_mid


def _parallel_density_flux_divergence(
    *,
    x: jnp.ndarray,
    theta_shift: jnp.ndarray,
    density: jnp.ndarray,
    v_electron_parallel: jnp.ndarray,
    density_grad: jnp.ndarray,
    v_electron_grad: jnp.ndarray,
    b_contra: jnp.ndarray,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Continuous counterpart of ``local_parallel_flux_div_op(n * v_e)``."""

    density_flux = density * v_electron_parallel
    density_flux_grad = (
        v_electron_parallel[..., None] * density_grad
        + density[..., None] * v_electron_grad
    )
    direct_parallel_gradient = jnp.einsum("...i,...i->...", b_contra, density_flux_grad)

    cos_shift = jnp.cos(theta_shift)
    sin_shift = jnp.sin(theta_shift)
    radius = float(r0) + float(alpha_value) * x + x * cos_shift
    q_value = 1.0 + float(alpha_value) * cos_shift
    metric_jacobian = x * radius * q_value
    s_value = (float(iota) ** 2) * x**2 + radius**2
    sqrt_s = jnp.sqrt(jnp.maximum(s_value, 1.0e-30))

    radius_theta = -x * sin_shift
    q_theta = -float(alpha_value) * sin_shift
    jacobian_theta = x * (radius_theta * q_value + radius * q_theta)
    sqrt_s_theta = radius * radius_theta / sqrt_s
    d_jbtheta_dtheta = float(iota) * (
        jacobian_theta * sqrt_s - metric_jacobian * sqrt_s_theta
    ) / jnp.maximum(s_value, 1.0e-30)

    div_b = d_jbtheta_dtheta / jnp.maximum(jacobian, 1.0e-30)
    return direct_parallel_gradient + density_flux * div_b


def _shifted_torus_envelopes(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, _, _, _ = _shifted_torus_coordinates(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_phi = 1.0 + float(a_phi) * jnp.cos(kx * xi)
    f_n = 1.0 + float(a_n) * jnp.sin(kx * xi)
    f_e = 1.0 + float(a_e) * jnp.cos(2.0 * kx * xi)
    f_i = 1.0 + float(a_i) * jnp.sin(2.0 * kx * xi)
    return f_phi, f_n, f_e, f_i


def _shifted_torus_phi(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    f_phi, _, _, _ = _shifted_torus_envelopes(geometry)
    ct = jnp.cos(float(Omega) * time)
    return float(A_phi) * f_phi * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * ct


def _shifted_torus_phi_scalar(x: float, theta: float, zeta: float, time: float) -> jnp.ndarray:
    theta_shift, _, _, _, _, _, _, _ = _shifted_torus_geometry_quantities_scalar(x, theta)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_phi = 1.0 + float(a_phi) * jnp.cos(kx * xi)
    ct = jnp.cos(float(Omega) * time)
    return float(A_phi) * f_phi * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * ct


def _shifted_torus_phi_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    f_phi, _, _, _ = _shifted_torus_envelopes(geometry)
    st = jnp.sin(float(Omega) * time)
    return -float(A_phi) * float(Omega) * f_phi * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * st


def _dirichlet_x_boundary_face_bc_from_values(
    lower_value: jnp.ndarray,
    upper_value: jnp.ndarray,
    geometry: FciGeometry3D,
) -> BoundaryFaceBC3D:
    from jax_drb.geometry.fci_geometry import RegularFaceGeometry3D
    from jax_drb.native.fci_boundaries import BC_DIRICHLET, BoundaryFaceBC3D

    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    lower = jnp.asarray(lower_value, dtype=jnp.float64)
    upper = jnp.asarray(upper_value, dtype=jnp.float64)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.int32).at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.int32),
        value_x=jnp.zeros_like(regular_face_geometry.x_area, dtype=jnp.float64).at[0].set(lower).at[-1].set(upper),
        value_y=jnp.zeros_like(regular_face_geometry.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(regular_face_geometry.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(regular_face_geometry.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(regular_face_geometry.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(regular_face_geometry.z_open_mask, dtype=bool),
    )


def _dirichlet_x_boundary_face_bc(field: jnp.ndarray, geometry: FciGeometry3D) -> BoundaryFaceBC3D:
    values = jnp.asarray(field, dtype=jnp.float64)
    return _dirichlet_x_boundary_face_bc_from_values(values[0], values[-1], geometry)


def _shifted_torus_density(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, f_n, _, _ = _shifted_torus_envelopes(geometry)
    st = jnp.sin(float(Omega) * time)
    return float(n0) + float(A_n) * f_n * jnp.sin(float(M_n) * theta_shift) * jnp.sin(float(N_n) * zeta) * st


def _shifted_torus_density_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, f_n, _, _ = _shifted_torus_envelopes(geometry)
    ct = jnp.cos(float(Omega) * time)
    return float(A_n) * float(Omega) * f_n * jnp.sin(float(M_n) * theta_shift) * jnp.sin(float(N_n) * zeta) * ct


def _shifted_torus_v_electron_parallel(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, f_e, _ = _shifted_torus_envelopes(geometry)
    ct = jnp.cos(float(Omega) * time)
    return float(A_e) * f_e * jnp.sin(float(M_e) * theta_shift) * jnp.cos(float(N_e) * zeta) * ct


def _shifted_torus_v_electron_parallel_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, f_e, _ = _shifted_torus_envelopes(geometry)
    st = jnp.sin(float(Omega) * time)
    return -float(A_e) * float(Omega) * f_e * jnp.sin(float(M_e) * theta_shift) * jnp.cos(float(N_e) * zeta) * st


def _shifted_torus_omega_scalar(x: float, theta: float, zeta: float, time: float) -> jnp.ndarray:
    def _flux_vector(coord: jnp.ndarray) -> jnp.ndarray:
        xx, tt, zz = coord
        theta_shift, cos_shift, sin_shift, R, Q, J, S, _ = _shifted_torus_geometry_quantities_scalar(xx, tt)
        xi = xx - float(x_min)
        kx = jnp.pi / (float(x_max) - float(x_min))
        f_phi = 1.0 + float(a_phi) * jnp.cos(kx * xi)
        f_phi_x = -float(a_phi) * kx * jnp.sin(kx * xi)
        ct = jnp.cos(float(Omega) * time)
        sin_theta_mode = jnp.sin(float(M_phi) * theta_shift)
        cos_theta_mode = jnp.cos(float(M_phi) * theta_shift)
        sin_zeta_mode = jnp.sin(float(N_phi) * zz)
        cos_zeta_mode = jnp.cos(float(N_phi) * zz)

        phi_x = float(A_phi) * sin_zeta_mode * ct * (f_phi_x * cos_theta_mode - f_phi * float(M_phi) * float(sigma) * sin_theta_mode)
        phi_theta = -float(A_phi) * float(M_phi) * f_phi * sin_theta_mode * sin_zeta_mode * ct
        phi_zeta = float(A_phi) * float(N_phi) * f_phi * cos_theta_mode * cos_zeta_mode * ct

        mcoef = 1.0 + 2.0 * float(alpha_value) * cos_shift + float(alpha_value) ** 2
        p_xx = 1.0 / (Q**2)
        p_xt = float(alpha_value) * sin_shift / (xx * Q**2)
        p_tt = mcoef / (xx**2 * Q**2) - (float(iota) ** 2) / S
        p_tz = -float(iota) / S
        p_zz = 1.0 / (R**2) - 1.0 / S
        return jnp.array(
            [
                J * (p_xx * phi_x + p_xt * phi_theta),
                J * (p_xt * phi_x + p_tt * phi_theta + p_tz * phi_zeta),
                J * (p_tz * phi_theta + p_zz * phi_zeta),
            ],
            dtype=jnp.float64,
        )

    theta_shift, cos_shift, sin_shift, R, Q, J, S, _ = _shifted_torus_geometry_quantities_scalar(x, theta)
    jacobian = jax.jacfwd(_flux_vector)(jnp.array([x, theta, zeta], dtype=jnp.float64))
    return (jacobian[0, 0] + jacobian[1, 1] + jacobian[2, 2]) / J


def _shifted_torus_omega(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    x, _, theta, zeta = _shifted_torus_coordinates(geometry)
    flat_coords = jnp.stack([x.ravel(), theta.ravel(), zeta.ravel()], axis=-1)
    omega = jax.vmap(
        lambda coord: _shifted_torus_omega_scalar(coord[0], coord[1], coord[2], time)
    )(flat_coords)
    return omega.reshape(geometry.shape)


def _shifted_torus_omega_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, _, _, _, omega_t = _shifted_torus_omega_and_derivatives(geometry, time)
    return omega_t


def _shifted_torus_omega_and_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, _, theta, zeta = _shifted_torus_coordinates(geometry)
    flat_coords = jnp.stack([x.ravel(), theta.ravel(), zeta.ravel()], axis=-1)

    def _value_and_grad(coord: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        return jax.value_and_grad(_shifted_torus_omega_scalar, argnums=(0, 1, 2, 3))(coord[0], coord[1], coord[2], time)

    values, grads = jax.vmap(_value_and_grad)(flat_coords)
    return (
        values.reshape(geometry.shape),
        grads[0].reshape(geometry.shape),
        grads[1].reshape(geometry.shape),
        grads[2].reshape(geometry.shape),
        grads[3].reshape(geometry.shape),
    )


def _shifted_torus_x_face_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from jax_drb.geometry.fci_geometry import logical_grid_from_axis_vectors

    logical_grid = logical_grid_from_axis_vectors(
        jnp.asarray([geometry.grid.x.faces[0], geometry.grid.x.faces[-1]], dtype=jnp.float64),
        geometry.grid.y.centers,
        geometry.grid.z.centers,
    )
    x = jnp.asarray(logical_grid[..., 0], dtype=jnp.float64)
    theta = jnp.asarray(logical_grid[..., 1], dtype=jnp.float64)
    zeta = jnp.asarray(logical_grid[..., 2], dtype=jnp.float64)
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    return x, theta_shift, theta, zeta


def _split_lower_upper_face_values(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    face_values = jnp.asarray(values, dtype=jnp.float64)
    return face_values[0], face_values[-1]


def _shifted_torus_phi_x_face_values(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, theta_shift, _, zeta = _shifted_torus_x_face_coordinates(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_phi = 1.0 + float(a_phi) * jnp.cos(kx * xi)
    ct = jnp.cos(float(Omega) * time)
    values = float(A_phi) * f_phi * jnp.cos(float(M_phi) * theta_shift) * jnp.sin(float(N_phi) * zeta) * ct
    return _split_lower_upper_face_values(values)


def _shifted_torus_density_x_face_values(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, theta_shift, _, zeta = _shifted_torus_x_face_coordinates(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_n = 1.0 + float(a_n) * jnp.sin(kx * xi)
    st = jnp.sin(float(Omega) * time)
    values = float(n0) + float(A_n) * f_n * jnp.sin(float(M_n) * theta_shift) * jnp.sin(float(N_n) * zeta) * st
    return _split_lower_upper_face_values(values)


def _shifted_torus_omega_x_face_values(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, _, theta, zeta = _shifted_torus_x_face_coordinates(geometry)
    flat_coords = jnp.stack([x.ravel(), theta.ravel(), zeta.ravel()], axis=-1)
    values = jax.vmap(
        lambda coord: _shifted_torus_omega_scalar(coord[0], coord[1], coord[2], time)
    )(flat_coords)
    return _split_lower_upper_face_values(values.reshape((2,) + geometry.shape[1:]))


def _shifted_torus_v_ion_parallel_x_face_values(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, theta_shift, _, zeta = _shifted_torus_x_face_coordinates(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_i = 1.0 + float(a_i) * jnp.sin(2.0 * kx * xi)
    st = jnp.sin(float(Omega) * time)
    values = float(A_i) * f_i * jnp.cos(float(M_i) * theta_shift) * jnp.sin(float(N_i) * zeta) * st
    return _split_lower_upper_face_values(values)


def _shifted_torus_v_electron_parallel_x_face_values(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, theta_shift, _, zeta = _shifted_torus_x_face_coordinates(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_e = 1.0 + float(a_e) * jnp.cos(2.0 * kx * xi)
    ct = jnp.cos(float(Omega) * time)
    values = float(A_e) * f_e * jnp.sin(float(M_e) * theta_shift) * jnp.cos(float(N_e) * zeta) * ct
    return _split_lower_upper_face_values(values)


def _shifted_torus_exact_x_face_bcs(
    geometry: FciGeometry3D,
    time: float,
) -> tuple[BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D]:
    phi_lower, phi_upper = _shifted_torus_phi_x_face_values(geometry, time)
    density_lower, density_upper = _shifted_torus_density_x_face_values(geometry, time)
    omega_lower, omega_upper = _shifted_torus_omega_x_face_values(geometry, time)
    v_ion_lower, v_ion_upper = _shifted_torus_v_ion_parallel_x_face_values(geometry, time)
    v_electron_lower, v_electron_upper = _shifted_torus_v_electron_parallel_x_face_values(geometry, time)
    return (
        _dirichlet_x_boundary_face_bc_from_values(phi_lower, phi_upper, geometry),
        _dirichlet_x_boundary_face_bc_from_values(density_lower, density_upper, geometry),
        _dirichlet_x_boundary_face_bc_from_values(omega_lower, omega_upper, geometry),
        _dirichlet_x_boundary_face_bc_from_values(v_ion_lower, v_ion_upper, geometry),
        _dirichlet_x_boundary_face_bc_from_values(v_electron_lower, v_electron_upper, geometry),
    )


def _shifted_torus_v_ion_parallel(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, _, f_i = _shifted_torus_envelopes(geometry)
    st = jnp.sin(float(Omega) * time)
    return float(A_i) * f_i * jnp.cos(float(M_i) * theta_shift) * jnp.sin(float(N_i) * zeta) * st


def _shifted_torus_v_ion_parallel_t(geometry: FciGeometry3D, time: float) -> jnp.ndarray:
    _, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, _, f_i = _shifted_torus_envelopes(geometry)
    ct = jnp.cos(float(Omega) * time)
    return float(A_i) * float(Omega) * f_i * jnp.cos(float(M_i) * theta_shift) * jnp.sin(float(N_i) * zeta) * ct


def _shifted_torus_phi_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    f_phi, _, _, _ = _shifted_torus_envelopes(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_phi = jnp.sin(float(M_phi) * theta_shift)
    cos_phi = jnp.cos(float(M_phi) * theta_shift)
    sin_zeta = jnp.sin(float(N_phi) * zeta)
    cos_zeta = jnp.cos(float(N_phi) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    field = float(A_phi) * f_phi * cos_phi * sin_zeta * cos_time
    field_x = float(A_phi) * (
        -float(a_phi) * kx * jnp.sin(kx * xi) * cos_phi
        - f_phi * float(M_phi) * float(sigma) * sin_phi
    ) * sin_zeta * cos_time
    field_theta = -float(A_phi) * float(M_phi) * f_phi * sin_phi * sin_zeta * cos_time
    field_zeta = float(A_phi) * float(N_phi) * f_phi * cos_phi * cos_zeta * cos_time
    field_t = -float(A_phi) * float(Omega) * f_phi * cos_phi * sin_zeta * sin_time
    return field, field_x, field_theta, field_zeta, field_t


def _shifted_torus_density_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, f_n, _, _ = _shifted_torus_envelopes(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_n = jnp.sin(float(M_n) * theta_shift)
    cos_n = jnp.cos(float(M_n) * theta_shift)
    sin_zeta = jnp.sin(float(N_n) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    density = float(n0) + float(A_n) * f_n * sin_n * sin_zeta * sin_time
    density_x = float(A_n) * (
        float(a_n) * kx * jnp.cos(kx * xi) * sin_n
        + f_n * float(M_n) * float(sigma) * cos_n
    ) * sin_zeta * sin_time
    density_theta = float(A_n) * float(M_n) * f_n * cos_n * sin_zeta * sin_time
    density_zeta = float(A_n) * float(N_n) * f_n * sin_n * jnp.cos(float(N_n) * zeta) * sin_time
    density_t = float(A_n) * float(Omega) * f_n * sin_n * sin_zeta * cos_time
    return density, density_x, density_theta, density_zeta, density_t


def _shifted_torus_v_ion_parallel_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, _, f_i = _shifted_torus_envelopes(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_i = jnp.sin(float(M_i) * theta_shift)
    cos_i = jnp.cos(float(M_i) * theta_shift)
    sin_zeta = jnp.sin(float(N_i) * zeta)
    cos_zeta = jnp.cos(float(N_i) * zeta)
    sin_time = jnp.sin(float(Omega) * time)
    cos_time = jnp.cos(float(Omega) * time)
    v_parallel = float(A_i) * f_i * cos_i * sin_zeta * sin_time
    v_parallel_x = float(A_i) * (
        2.0 * float(a_i) * kx * jnp.cos(2.0 * kx * xi) * cos_i
        - f_i * float(M_i) * float(sigma) * sin_i
    ) * sin_zeta * sin_time
    v_parallel_theta = -float(A_i) * float(M_i) * f_i * sin_i * sin_zeta * sin_time
    v_parallel_zeta = float(A_i) * float(N_i) * f_i * cos_i * cos_zeta * sin_time
    v_parallel_t = float(A_i) * float(Omega) * f_i * cos_i * sin_zeta * cos_time
    return v_parallel, v_parallel_x, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_v_electron_parallel_derivatives(geometry: FciGeometry3D, time: float) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = _shifted_torus_coordinates(geometry)
    _, _, f_e, _ = _shifted_torus_envelopes(geometry)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_e = jnp.sin(float(M_e) * theta_shift)
    cos_e = jnp.cos(float(M_e) * theta_shift)
    sin_zeta = jnp.sin(float(N_e) * zeta)
    cos_zeta = jnp.cos(float(N_e) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    v_parallel = float(A_e) * f_e * sin_e * cos_zeta * cos_time
    v_parallel_x = float(A_e) * (
        -2.0 * float(a_e) * kx * jnp.sin(2.0 * kx * xi) * sin_e
        + f_e * float(M_e) * float(sigma) * cos_e
    ) * cos_zeta * cos_time
    v_parallel_theta = float(A_e) * float(M_e) * f_e * cos_e * cos_zeta * cos_time
    v_parallel_zeta = -float(A_e) * float(N_e) * f_e * sin_e * sin_zeta * cos_time
    v_parallel_t = -float(A_e) * float(Omega) * f_e * sin_e * cos_zeta * sin_time
    return v_parallel, v_parallel_x, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _build_dirichlet_boundary_condition_builder(field_name: str):
    def build(
        state: jnp.ndarray,
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
        cut_wall_geometry: CutWallGeometry3D | None,
        cut_wall_bc: CutWallBC3D | None,
    ) -> tuple[BoundaryFaceBC3D, CutWallBC3D]:
        from jax_drb.geometry.fci_geometry import RegularFaceGeometry3D
        from jax_drb.native.fci_boundaries import BC_DIRICHLET, BoundaryFaceBC3D, CutWallBC3D

        del cut_wall_geometry
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
        if periodic_axes is not None and bool(periodic_axes[0]):
            face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
        return face_bc, cut_wall_bc or CutWallBC3D.empty()

    return build


def _homogeneous_boundary_face_bc(face_bc: BoundaryFaceBC3D) -> BoundaryFaceBC3D:
    """Keep regular-face BC kinds/masks, but zero values for correction solves."""

    return face_bc.replace(
        value_x=jnp.zeros_like(face_bc.value_x, dtype=jnp.float64),
        value_y=jnp.zeros_like(face_bc.value_y, dtype=jnp.float64),
        value_z=jnp.zeros_like(face_bc.value_z, dtype=jnp.float64),
    )


def _shifted_torus_exact_state(
    geometry: FciGeometry3D,
    time: float,
) -> Fci4FieldState:
    return Fci4FieldState(
        density=_shifted_torus_density(geometry, time),
        omega=_shifted_torus_omega(geometry, time),
        v_ion_parallel=_shifted_torus_v_ion_parallel(geometry, time),
        v_electron_parallel=_shifted_torus_v_electron_parallel(geometry, time),
    )


def _shifted_torus_exact_time_derivative_state(
    geometry: FciGeometry3D,
    time: float,
) -> Fci4FieldState:
    return Fci4FieldState(
        density=_shifted_torus_density_t(geometry, time),
        omega=_shifted_torus_omega_t(geometry, time),
        v_ion_parallel=_shifted_torus_v_ion_parallel_t(geometry, time),
        v_electron_parallel=_shifted_torus_v_electron_parallel_t(geometry, time),
    )


def _continuous_4field_rhs_from_exact_state(
    state: Fci4FieldState,
    geometry: FciGeometry3D,
    *,
    time: float,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
) -> Fci4FieldState:
    terms = _continuous_4field_rhs_terms_from_exact_state(
        state,
        geometry,
        time=time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    return _sum_rhs_terms(terms)


def _continuous_4field_rhs_terms_from_exact_state(
    state: Fci4FieldState,
    geometry: FciGeometry3D,
    *,
    time: float,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
) -> dict[str, dict[str, jnp.ndarray]]:
    density = jnp.asarray(state.density, dtype=jnp.float64)
    omega = jnp.asarray(state.omega, dtype=jnp.float64)
    v_ion_parallel = jnp.asarray(state.v_ion_parallel, dtype=jnp.float64)
    v_electron_parallel = jnp.asarray(state.v_electron_parallel, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)

    # Closed-form field derivatives evaluated at the MMS time. No stencil or
    # discrete operator is used in this source construction.
    _, density_x, density_theta, density_zeta, _ = _shifted_torus_density_derivatives(geometry, time)
    _, omega_x, omega_theta, omega_zeta, _ = _shifted_torus_omega_and_derivatives(geometry, time)
    _, v_ion_x, v_ion_theta, v_ion_zeta, _ = _shifted_torus_v_ion_parallel_derivatives(geometry, time)
    _, v_electron_x, v_electron_theta, v_electron_zeta, _ = _shifted_torus_v_electron_parallel_derivatives(geometry, time)
    _, phi_x, phi_theta, phi_zeta, _ = _shifted_torus_phi_derivatives(geometry, time)

    density_grad = jnp.stack((density_x, density_theta, density_zeta), axis=-1)
    omega_grad = jnp.stack((omega_x, omega_theta, omega_zeta), axis=-1)
    v_ion_grad = jnp.stack((v_ion_x, v_ion_theta, v_ion_zeta), axis=-1)
    v_electron_grad = jnp.stack((v_electron_x, v_electron_theta, v_electron_zeta), axis=-1)
    phi_grad = jnp.stack((phi_x, phi_theta, phi_zeta), axis=-1)

    b_contra = jnp.asarray(geometry.cell_bfield.b_contra, dtype=jnp.float64)
    b_unit = b_contra
    b_covariant = jnp.einsum("...ij,...j->...i", jnp.asarray(geometry.cell_metric.g_cov, dtype=jnp.float64), b_unit)
    jacobian = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)

    def _poisson(df: jnp.ndarray, dg: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_covariant * jnp.cross(df, dg), axis=-1) / jnp.maximum(jacobian, 1.0e-30)

    curvature_density = jnp.einsum("...i,...i->...", curvature_coefficients, density_grad)
    curvature_phi = jnp.einsum("...i,...i->...", curvature_coefficients, phi_grad)
    grad_parallel_density = jnp.einsum("...i,...i->...", b_contra, density_grad)
    grad_parallel_phi = jnp.einsum("...i,...i->...", b_contra, phi_grad)
    grad_parallel_v_ion = jnp.einsum("...i,...i->...", b_contra, v_ion_grad)
    grad_parallel_v_electron = jnp.einsum("...i,...i->...", b_contra, v_electron_grad)
    x_coord, theta_shift_coord, _, _ = _shifted_torus_coordinates(geometry)
    parallel_density_flux_divergence = _parallel_density_flux_divergence(
        x=x_coord,
        theta_shift=theta_shift_coord,
        density=density,
        v_electron_parallel=v_electron_parallel,
        density_grad=density_grad,
        v_electron_grad=v_electron_grad,
        b_contra=b_contra,
        jacobian=jacobian,
    )

    return {
        "density": {
            "poisson": -(_poisson(phi_grad, density_grad) / (rho_star_value * bmag)),
            "curvature_density": (2.0 * te / bmag) * curvature_density,
            "curvature_phi": -(2.0 * density / bmag) * curvature_phi,
            "parallel_density_v_electron": -parallel_density_flux_divergence,
        },
        "omega": {
            "poisson": -(_poisson(phi_grad, omega_grad) / (rho_star_value * bmag)),
            "parallel_current": (bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron),
            "curvature_density": (2.0 * bmag * te / density_safe) * curvature_density,
        },
        "v_ion_parallel": {
            "poisson": -(_poisson(phi_grad, v_ion_grad) / (rho_star_value * bmag)),
            "grad_density": -(te / density_safe) * grad_parallel_density,
        },
        "v_electron_parallel": {
            "poisson": -(_poisson(phi_grad, v_electron_grad) / (rho_star_value * bmag)),
            "grad_phi": mi_over_me * grad_parallel_phi,
            "grad_density": -mi_over_me * (te / density_safe) * grad_parallel_density,
        },
    }


def _sum_rhs_terms(terms: dict[str, dict[str, jnp.ndarray]]) -> Fci4FieldState:
    density_rhs = sum(terms["density"].values())
    omega_rhs = sum(terms["omega"].values())
    v_ion_rhs = sum(terms["v_ion_parallel"].values())
    v_electron_rhs = sum(terms["v_electron_parallel"].values())
    return Fci4FieldState(
        density=density_rhs,
        omega=omega_rhs,
        v_ion_parallel=v_ion_rhs,
        v_electron_parallel=v_electron_rhs,
    )


def _shifted_torus_mms_source_state(
    geometry: FciGeometry3D,
    time: float,
    *,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
) -> Fci4FieldState:
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_time_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    analytic_rhs = _continuous_4field_rhs_from_exact_state(
        exact_state,
        geometry,
        time=time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    return Fci4FieldState(
        density=exact_time_derivative.density - analytic_rhs.density,
        omega=exact_time_derivative.omega - analytic_rhs.omega,
        v_ion_parallel=exact_time_derivative.v_ion_parallel - analytic_rhs.v_ion_parallel,
        v_electron_parallel=exact_time_derivative.v_electron_parallel - analytic_rhs.v_electron_parallel,
    )


def _add_state(state: Fci4FieldState, rhs: Fci4FieldState, *, scale: float) -> Fci4FieldState:
    return Fci4FieldState(
        density=state.density + scale * rhs.density,
        omega=state.omega + scale * rhs.omega,
        v_ion_parallel=state.v_ion_parallel + scale * rhs.v_ion_parallel,
        v_electron_parallel=state.v_electron_parallel + scale * rhs.v_electron_parallel,
    )


def _raise_if_nonfinite_state(state: Fci4FieldState, *, label: str) -> None:
    for name, value in (
        ("density", state.density),
        ("omega", state.omega),
        ("v_ion_parallel", state.v_ion_parallel),
        ("v_electron_parallel", state.v_electron_parallel),
    ):
        if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise FloatingPointError(f"non-finite {name} encountered in {label}")


def _combined_error_statistics(
    final_state: Fci4FieldState,
    geometry: FciGeometry3D,
    time: float,
) -> tuple[float, float, float]:
    exact = _shifted_torus_exact_state(
        geometry,
        time,
    )
    error = jnp.concatenate(
        [
            jnp.ravel(jnp.abs(final_state.density - exact.density)),
            jnp.ravel(jnp.abs(final_state.omega - exact.omega)),
            jnp.ravel(jnp.abs(final_state.v_ion_parallel - exact.v_ion_parallel)),
            jnp.ravel(jnp.abs(final_state.v_electron_parallel - exact.v_electron_parallel)),
        ]
    )
    return float(jnp.sqrt(jnp.mean(error**2))), float(jnp.median(error)), float(jnp.max(error))


def _field_error_statistics(actual: jnp.ndarray, expected: jnp.ndarray) -> tuple[float, float, float]:
    error = jnp.asarray(actual - expected, dtype=jnp.float64)
    expected_array = jnp.asarray(expected, dtype=jnp.float64)
    l2 = float(jnp.sqrt(jnp.mean(jnp.square(error))))
    linf = float(jnp.max(jnp.abs(error)))
    rel_l2 = float(jnp.linalg.norm(error) / (jnp.linalg.norm(expected_array) + 1.0e-30))
    return l2, linf, rel_l2


def _state_error_statistics(actual: Fci4FieldState, expected: Fci4FieldState) -> dict[str, tuple[float, float, float]]:
    return {
        "density": _field_error_statistics(actual.density, expected.density),
        "omega": _field_error_statistics(actual.omega, expected.omega),
        "v_ion_parallel": _field_error_statistics(actual.v_ion_parallel, expected.v_ion_parallel),
        "v_electron_parallel": _field_error_statistics(actual.v_electron_parallel, expected.v_electron_parallel),
    }


def _print_state_error_statistics(label: str, stats: dict[str, tuple[float, float, float]]) -> None:
    print(label)
    for field_name, (l2, linf, rel_l2) in stats.items():
        print(f"  {field_name}: l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")


def _observed_order(error_coarse: float, error_fine: float, resolution_coarse: int, resolution_fine: int) -> float:
    if error_coarse <= 0.0 or error_fine <= 0.0:
        return float("nan")
    return float(np.log(error_coarse / error_fine) / np.log(float(resolution_fine) / float(resolution_coarse)))


def _format_order(order: float) -> str:
    return "nan" if not np.isfinite(order) else f"{order:.3f}"


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


def _shifted_torus_local_envelopes(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, _, _, _ = coordinates_halo
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    f_phi = 1.0 + float(a_phi) * jnp.cos(kx * xi)
    f_n = 1.0 + float(a_n) * jnp.sin(kx * xi)
    f_e = 1.0 + float(a_e) * jnp.cos(2.0 * kx * xi)
    f_i = 1.0 + float(a_i) * jnp.sin(2.0 * kx * xi)
    return f_phi, f_n, f_e, f_i


def _shifted_torus_local_phi_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    f_phi, _, _, _ = _shifted_torus_local_envelopes(coordinates_halo)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_phi = jnp.sin(float(M_phi) * theta_shift)
    cos_phi = jnp.cos(float(M_phi) * theta_shift)
    sin_zeta = jnp.sin(float(N_phi) * zeta)
    cos_zeta = jnp.cos(float(N_phi) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    field = float(A_phi) * f_phi * cos_phi * sin_zeta * cos_time
    field_x = float(A_phi) * (
        -float(a_phi) * kx * jnp.sin(kx * xi) * cos_phi
        - f_phi * float(M_phi) * float(sigma) * sin_phi
    ) * sin_zeta * cos_time
    field_theta = -float(A_phi) * float(M_phi) * f_phi * sin_phi * sin_zeta * cos_time
    field_zeta = float(A_phi) * float(N_phi) * f_phi * cos_phi * cos_zeta * cos_time
    field_t = -float(A_phi) * float(Omega) * f_phi * cos_phi * sin_zeta * sin_time
    return field, field_x, field_theta, field_zeta, field_t


def _shifted_torus_local_density_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    _, f_n, _, _ = _shifted_torus_local_envelopes(coordinates_halo)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_n = jnp.sin(float(M_n) * theta_shift)
    cos_n = jnp.cos(float(M_n) * theta_shift)
    sin_zeta = jnp.sin(float(N_n) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    density = float(n0) + float(A_n) * f_n * sin_n * sin_zeta * sin_time
    density_x = float(A_n) * (
        float(a_n) * kx * jnp.cos(kx * xi) * sin_n
        + f_n * float(M_n) * float(sigma) * cos_n
    ) * sin_zeta * sin_time
    density_theta = float(A_n) * float(M_n) * f_n * cos_n * sin_zeta * sin_time
    density_zeta = float(A_n) * float(N_n) * f_n * sin_n * jnp.cos(float(N_n) * zeta) * sin_time
    density_t = float(A_n) * float(Omega) * f_n * sin_n * sin_zeta * cos_time
    return density, density_x, density_theta, density_zeta, density_t


def _shifted_torus_local_v_ion_parallel_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    _, _, _, f_i = _shifted_torus_local_envelopes(coordinates_halo)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_i = jnp.sin(float(M_i) * theta_shift)
    cos_i = jnp.cos(float(M_i) * theta_shift)
    sin_zeta = jnp.sin(float(N_i) * zeta)
    cos_zeta = jnp.cos(float(N_i) * zeta)
    sin_time = jnp.sin(float(Omega) * time)
    cos_time = jnp.cos(float(Omega) * time)
    v_parallel = float(A_i) * f_i * cos_i * sin_zeta * sin_time
    v_parallel_x = float(A_i) * (
        2.0 * float(a_i) * kx * jnp.cos(2.0 * kx * xi) * cos_i
        - f_i * float(M_i) * float(sigma) * sin_i
    ) * sin_zeta * sin_time
    v_parallel_theta = -float(A_i) * float(M_i) * f_i * sin_i * sin_zeta * sin_time
    v_parallel_zeta = float(A_i) * float(N_i) * f_i * cos_i * cos_zeta * sin_time
    v_parallel_t = float(A_i) * float(Omega) * f_i * cos_i * sin_zeta * cos_time
    return v_parallel, v_parallel_x, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_local_v_electron_parallel_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, ...]:
    x, theta_shift, _, zeta = coordinates_halo
    _, _, f_e, _ = _shifted_torus_local_envelopes(coordinates_halo)
    xi = x - float(x_min)
    kx = jnp.pi / (float(x_max) - float(x_min))
    sin_e = jnp.sin(float(M_e) * theta_shift)
    cos_e = jnp.cos(float(M_e) * theta_shift)
    sin_zeta = jnp.sin(float(N_e) * zeta)
    cos_zeta = jnp.cos(float(N_e) * zeta)
    cos_time = jnp.cos(float(Omega) * time)
    sin_time = jnp.sin(float(Omega) * time)
    v_parallel = float(A_e) * f_e * sin_e * cos_zeta * cos_time
    v_parallel_x = float(A_e) * (
        -2.0 * float(a_e) * kx * jnp.sin(2.0 * kx * xi) * sin_e
        + f_e * float(M_e) * float(sigma) * cos_e
    ) * cos_zeta * cos_time
    v_parallel_theta = float(A_e) * float(M_e) * f_e * cos_e * cos_zeta * cos_time
    v_parallel_zeta = -float(A_e) * float(N_e) * f_e * sin_e * sin_zeta * cos_time
    v_parallel_t = -float(A_e) * float(Omega) * f_e * sin_e * cos_zeta * sin_time
    return v_parallel, v_parallel_x, v_parallel_theta, v_parallel_zeta, v_parallel_t


def _shifted_torus_local_omega_and_derivatives(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, _, theta, zeta = coordinates_halo
    flat_coords = jnp.stack([x.ravel(), theta.ravel(), zeta.ravel()], axis=-1)

    def _value_and_grad(coord: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        return jax.value_and_grad(_shifted_torus_omega_scalar, argnums=(0, 1, 2, 3))(
            coord[0],
            coord[1],
            coord[2],
            time,
        )

    values, grads = jax.vmap(_value_and_grad)(flat_coords)
    shape = x.shape
    return (
        values.reshape(shape),
        grads[0].reshape(shape),
        grads[1].reshape(shape),
        grads[2].reshape(shape),
        grads[3].reshape(shape),
    )


def _shifted_torus_local_exact_state(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> Fci4FieldState:
    density, *_ = _shifted_torus_local_density_derivatives(coordinates_halo, time)
    omega, *_ = _shifted_torus_local_omega_and_derivatives(coordinates_halo, time)
    v_ion, *_ = _shifted_torus_local_v_ion_parallel_derivatives(coordinates_halo, time)
    v_electron, *_ = _shifted_torus_local_v_electron_parallel_derivatives(coordinates_halo, time)
    return Fci4FieldState(
        density=density,
        omega=omega,
        v_ion_parallel=v_ion,
        v_electron_parallel=v_electron,
    )


def _shifted_torus_local_exact_time_derivative_state(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
) -> Fci4FieldState:
    *_, density_t = _shifted_torus_local_density_derivatives(coordinates_halo, time)
    *_, omega_t = _shifted_torus_local_omega_and_derivatives(coordinates_halo, time)
    *_, v_ion_t = _shifted_torus_local_v_ion_parallel_derivatives(coordinates_halo, time)
    *_, v_electron_t = _shifted_torus_local_v_electron_parallel_derivatives(coordinates_halo, time)
    return Fci4FieldState(
        density=density_t,
        omega=omega_t,
        v_ion_parallel=v_ion_t,
        v_electron_parallel=v_electron_t,
    )


def _shifted_torus_local_mms_source_state(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
    *,
    geometry: LocalFciGeometry3D,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients_halo: jnp.ndarray,
) -> Fci4FieldState:
    exact = _shifted_torus_local_exact_state(coordinates_halo, time)
    exact_t = _shifted_torus_local_exact_time_derivative_state(coordinates_halo, time)
    density = jnp.asarray(exact.density, dtype=jnp.float64)
    omega_value = jnp.asarray(exact.omega, dtype=jnp.float64)
    v_ion = jnp.asarray(exact.v_ion_parallel, dtype=jnp.float64)
    v_electron = jnp.asarray(exact.v_electron_parallel, dtype=jnp.float64)
    del omega_value, v_ion, v_electron
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag_halo, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me_value = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)

    _, density_x, density_theta, density_zeta, _ = _shifted_torus_local_density_derivatives(coordinates_halo, time)
    _, omega_x, omega_theta, omega_zeta, _ = _shifted_torus_local_omega_and_derivatives(coordinates_halo, time)
    _, v_ion_x, v_ion_theta, v_ion_zeta, _ = _shifted_torus_local_v_ion_parallel_derivatives(coordinates_halo, time)
    _, v_electron_x, v_electron_theta, v_electron_zeta, _ = _shifted_torus_local_v_electron_parallel_derivatives(coordinates_halo, time)
    _, phi_x, phi_theta, phi_zeta, _ = _shifted_torus_local_phi_derivatives(coordinates_halo, time)

    density_grad = jnp.stack((density_x, density_theta, density_zeta), axis=-1)
    omega_grad = jnp.stack((omega_x, omega_theta, omega_zeta), axis=-1)
    v_ion_grad = jnp.stack((v_ion_x, v_ion_theta, v_ion_zeta), axis=-1)
    v_electron_grad = jnp.stack((v_electron_x, v_electron_theta, v_electron_zeta), axis=-1)
    phi_grad = jnp.stack((phi_x, phi_theta, phi_zeta), axis=-1)

    b_contra = jnp.asarray(geometry.cell_bfield.b_contra, dtype=jnp.float64)
    b_covariant = jnp.einsum(
        "...ij,...j->...i",
        jnp.asarray(geometry.cell_metric.g_cov, dtype=jnp.float64),
        b_contra,
    )
    jacobian = jnp.maximum(jnp.asarray(geometry.cell_metric.J_halo, dtype=jnp.float64), 1.0e-30)

    def _poisson(df: jnp.ndarray, dg: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_covariant * jnp.cross(df, dg), axis=-1) / jacobian

    curvature_density = jnp.einsum("...i,...i->...", curvature_coefficients_halo, density_grad)
    curvature_phi = jnp.einsum("...i,...i->...", curvature_coefficients_halo, phi_grad)
    grad_parallel_density = jnp.einsum("...i,...i->...", b_contra, density_grad)
    grad_parallel_phi = jnp.einsum("...i,...i->...", b_contra, phi_grad)
    grad_parallel_v_ion = jnp.einsum("...i,...i->...", b_contra, v_ion_grad)
    grad_parallel_v_electron = jnp.einsum("...i,...i->...", b_contra, v_electron_grad)
    x_coord, theta_shift_coord, _, _ = coordinates_halo
    parallel_density_flux_divergence = _parallel_density_flux_divergence(
        x=x_coord,
        theta_shift=theta_shift_coord,
        density=exact.density,
        v_electron_parallel=exact.v_electron_parallel,
        density_grad=density_grad,
        v_electron_grad=v_electron_grad,
        b_contra=b_contra,
        jacobian=jacobian,
    )

    density_rhs = (
        -(_poisson(phi_grad, density_grad) / (rho_star_value * bmag))
        + (2.0 * te / bmag) * curvature_density
        - (2.0 * density / bmag) * curvature_phi
        - parallel_density_flux_divergence
    )
    omega_rhs = (
        -(_poisson(phi_grad, omega_grad) / (rho_star_value * bmag))
        + (bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron)
        + (2.0 * bmag * te / density_safe) * curvature_density
    )
    v_ion_rhs = (
        -(_poisson(phi_grad, v_ion_grad) / (rho_star_value * bmag))
        - (te / density_safe) * grad_parallel_density
    )
    v_electron_rhs = (
        -(_poisson(phi_grad, v_electron_grad) / (rho_star_value * bmag))
        + mi_over_me_value * grad_parallel_phi
        - mi_over_me_value * (te / density_safe) * grad_parallel_density
    )
    return Fci4FieldState(
        density=exact_t.density - density_rhs,
        omega=exact_t.omega - omega_rhs,
        v_ion_parallel=exact_t.v_ion_parallel - v_ion_rhs,
        v_electron_parallel=exact_t.v_electron_parallel - v_electron_rhs,
    )


def _state_partition_spec() -> Fci4FieldState:
    spec = P(*MESH_AXIS_NAMES)
    return Fci4FieldState(
        density=spec,
        omega=spec,
        v_ion_parallel=spec,
        v_electron_parallel=spec,
    )


def _put_state_on_mesh(state: Fci4FieldState, mesh: Mesh) -> Fci4FieldState:
    sharding = NamedSharding(mesh, P(*MESH_AXIS_NAMES))
    return Fci4FieldState(
        density=jax.device_put(jnp.asarray(state.density, dtype=jnp.float64), sharding),
        omega=jax.device_put(jnp.asarray(state.omega, dtype=jnp.float64), sharding),
        v_ion_parallel=jax.device_put(jnp.asarray(state.v_ion_parallel, dtype=jnp.float64), sharding),
        v_electron_parallel=jax.device_put(jnp.asarray(state.v_electron_parallel, dtype=jnp.float64), sharding),
    )


def _gather_state_from_mesh(state: Fci4FieldState) -> Fci4FieldState:
    return Fci4FieldState(
        density=jnp.asarray(jax.device_get(state.density), dtype=jnp.float64),
        omega=jnp.asarray(jax.device_get(state.omega), dtype=jnp.float64),
        v_ion_parallel=jnp.asarray(jax.device_get(state.v_ion_parallel), dtype=jnp.float64),
        v_electron_parallel=jnp.asarray(jax.device_get(state.v_electron_parallel), dtype=jnp.float64),
    )


def _build_ghost_filler(halo_width: int) -> PhysicalGhostCellFiller3D:
    from jax_drb.native.fci_halo import GhostFillWeights1D, PhysicalGhostCellFiller3D

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
    state_owned: Fci4FieldState,
    exact_owned: Fci4FieldState,
    domain: LocalDomain3D,
) -> Fci4FieldState:
    return Fci4FieldState(
        density=_apply_local_owned_dirichlet_to_field(state_owned.density, exact_owned.density, domain),
        omega=_apply_local_owned_dirichlet_to_field(state_owned.omega, exact_owned.omega, domain),
        v_ion_parallel=_apply_local_owned_dirichlet_to_field(
            state_owned.v_ion_parallel,
            exact_owned.v_ion_parallel,
            domain,
        ),
        v_electron_parallel=_apply_local_owned_dirichlet_to_field(
            state_owned.v_electron_parallel,
            exact_owned.v_electron_parallel,
            domain,
        ),
    )


def _build_local_radial_dirichlet_face_bc(
    values_halo: jnp.ndarray,
    domain: LocalDomain3D,
) -> LocalBoundaryFaceBC3D:
    from jax_drb.native.fci_boundaries import BC_DIRICHLET, LocalBoundaryFaceBC3D

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
    from jax_drb.native.fci_boundaries import BC_DIRICHLET, LocalBoundaryFaceBC3D

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


def _coordinates_from_4field_invariants(
    invariants: _ShiftedTorus4FieldInvariantBundle,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        invariants.coord_x,
        invariants.coord_theta_shift,
        invariants.coord_theta,
        invariants.coord_zeta,
    )


def _face_coordinates_from_4field_invariants(
    invariants: _ShiftedTorus4FieldInvariantBundle,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        invariants.face_x,
        invariants.face_theta_shift,
        invariants.face_theta,
        invariants.face_zeta,
    )


def _build_local_4field_invariants(
    shard_index: tuple[int, int, int],
    *,
    owned_shape: tuple[int, int, int],
    halo_width: int,
    global_shape: tuple[int, int, int],
    domain: LocalDomain3D,
) -> _ShiftedTorus4FieldInvariantBundle:
    from jax_drb.geometry.fci_geometry import build_local_curvature_coefficients
    from jax_drb.native.fci_operators import build_local_perp_laplacian_face_projectors
    from mms_domain_decomp_helpers import build_shifted_torus_local_geometry

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
    coordinates_halo = _shifted_torus_local_coordinates(local_geometry)
    face_coordinates = _shifted_torus_local_x_face_coordinates(local_geometry, domain)
    curvature_coefficients_owned = jnp.asarray(
        build_local_curvature_coefficients(local_geometry, domain),
        dtype=jnp.float64,
    )
    curvature_coefficients_halo = jnp.zeros(
        domain.layout.cell_halo_shape + (3,),
        dtype=jnp.float64,
    ).at[domain.layout.owned_slices_cell + (slice(None),)].set(curvature_coefficients_owned)
    face_projectors = build_local_perp_laplacian_face_projectors(local_geometry, domain)
    return _ShiftedTorus4FieldInvariantBundle(
        coord_x=jnp.asarray(coordinates_halo[0], dtype=jnp.float64),
        coord_theta_shift=jnp.asarray(coordinates_halo[1], dtype=jnp.float64),
        coord_theta=jnp.asarray(coordinates_halo[2], dtype=jnp.float64),
        coord_zeta=jnp.asarray(coordinates_halo[3], dtype=jnp.float64),
        face_x=jnp.asarray(face_coordinates[0], dtype=jnp.float64),
        face_theta_shift=jnp.asarray(face_coordinates[1], dtype=jnp.float64),
        face_theta=jnp.asarray(face_coordinates[2], dtype=jnp.float64),
        face_zeta=jnp.asarray(face_coordinates[3], dtype=jnp.float64),
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


def _shifted_torus_local_mms_source_state_from_invariants(
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    time: float | jax.Array,
    *,
    invariants: _ShiftedTorus4FieldInvariantBundle,
    parameters: Fci4FieldRhsParameters,
) -> Fci4FieldState:
    exact = _shifted_torus_local_exact_state(coordinates_halo, time)
    exact_t = _shifted_torus_local_exact_time_derivative_state(coordinates_halo, time)
    density = jnp.asarray(exact.density, dtype=jnp.float64)
    density_safe = jnp.maximum(density, 1.0e-30)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me_value = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)

    _, density_x, density_theta, density_zeta, _ = _shifted_torus_local_density_derivatives(coordinates_halo, time)
    _, omega_x, omega_theta, omega_zeta, _ = _shifted_torus_local_omega_and_derivatives(coordinates_halo, time)
    _, v_ion_x, v_ion_theta, v_ion_zeta, _ = _shifted_torus_local_v_ion_parallel_derivatives(coordinates_halo, time)
    _, v_electron_x, v_electron_theta, v_electron_zeta, _ = _shifted_torus_local_v_electron_parallel_derivatives(coordinates_halo, time)
    _, phi_x, phi_theta, phi_zeta, _ = _shifted_torus_local_phi_derivatives(coordinates_halo, time)

    density_grad = jnp.stack((density_x, density_theta, density_zeta), axis=-1)
    omega_grad = jnp.stack((omega_x, omega_theta, omega_zeta), axis=-1)
    v_ion_grad = jnp.stack((v_ion_x, v_ion_theta, v_ion_zeta), axis=-1)
    v_electron_grad = jnp.stack((v_electron_x, v_electron_theta, v_electron_zeta), axis=-1)
    phi_grad = jnp.stack((phi_x, phi_theta, phi_zeta), axis=-1)

    b_contra = invariants.b_contra_halo
    b_covariant = jnp.einsum("...ij,...j->...i", invariants.cell_metric_g_cov_halo, b_contra)
    jacobian = jnp.maximum(invariants.cell_metric_jacobian_halo, 1.0e-30)

    def _poisson(df: jnp.ndarray, dg: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(b_covariant * jnp.cross(df, dg), axis=-1) / jacobian

    curvature_density = jnp.einsum("...i,...i->...", invariants.curvature_coefficients_halo, density_grad)
    curvature_phi = jnp.einsum("...i,...i->...", invariants.curvature_coefficients_halo, phi_grad)
    grad_parallel_density = jnp.einsum("...i,...i->...", b_contra, density_grad)
    grad_parallel_phi = jnp.einsum("...i,...i->...", b_contra, phi_grad)
    grad_parallel_v_ion = jnp.einsum("...i,...i->...", b_contra, v_ion_grad)
    grad_parallel_v_electron = jnp.einsum("...i,...i->...", b_contra, v_electron_grad)
    x_coord, theta_shift_coord, _, _ = coordinates_halo
    parallel_density_flux_divergence = _parallel_density_flux_divergence(
        x=x_coord,
        theta_shift=theta_shift_coord,
        density=exact.density,
        v_electron_parallel=exact.v_electron_parallel,
        density_grad=density_grad,
        v_electron_grad=v_electron_grad,
        b_contra=b_contra,
        jacobian=jacobian,
    )
    bmag = jnp.maximum(invariants.bmag_halo, 1.0e-30)

    density_rhs = (
        -(_poisson(phi_grad, density_grad) / (rho_star_value * bmag))
        + (2.0 * te / bmag) * curvature_density
        - (2.0 * density / bmag) * curvature_phi
        - parallel_density_flux_divergence
    )
    omega_rhs = (
        -(_poisson(phi_grad, omega_grad) / (rho_star_value * bmag))
        + (bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron)
        + (2.0 * bmag * te / density_safe) * curvature_density
    )
    v_ion_rhs = (
        -(_poisson(phi_grad, v_ion_grad) / (rho_star_value * bmag))
        - (te / density_safe) * grad_parallel_density
    )
    v_electron_rhs = (
        -(_poisson(phi_grad, v_electron_grad) / (rho_star_value * bmag))
        + mi_over_me_value * grad_parallel_phi
        - mi_over_me_value * (te / density_safe) * grad_parallel_density
    )
    return Fci4FieldState(
        density=exact_t.density - density_rhs,
        omega=exact_t.omega - omega_rhs,
        v_ion_parallel=exact_t.v_ion_parallel - v_ion_rhs,
        v_electron_parallel=exact_t.v_electron_parallel - v_electron_rhs,
    )


def _build_local_4field_stage_data(
    invariants: _ShiftedTorus4FieldInvariantBundle,
    time: float | jax.Array,
    *,
    parameters: Fci4FieldRhsParameters,
) -> _ShiftedTorus4FieldStageData:
    coordinates_halo = _coordinates_from_4field_invariants(invariants)
    face_coordinates = _face_coordinates_from_4field_invariants(invariants)
    exact_halo = _shifted_torus_local_exact_state(coordinates_halo, time)
    phi_halo = _shifted_torus_local_phi_derivatives(coordinates_halo, time)[0]
    phi_face_lower, phi_face_upper = _split_local_x_face_values(
        _shifted_torus_local_phi_derivatives(face_coordinates, time)[0]
    )
    density_face_lower, density_face_upper = _split_local_x_face_values(
        _shifted_torus_local_density_derivatives(face_coordinates, time)[0]
    )
    omega_face_lower, omega_face_upper = _split_local_x_face_values(
        _shifted_torus_local_omega_and_derivatives(face_coordinates, time)[0]
    )
    v_ion_face_lower, v_ion_face_upper = _split_local_x_face_values(
        _shifted_torus_local_v_ion_parallel_derivatives(face_coordinates, time)[0]
    )
    v_electron_face_lower, v_electron_face_upper = _split_local_x_face_values(
        _shifted_torus_local_v_electron_parallel_derivatives(face_coordinates, time)[0]
    )
    return _ShiftedTorus4FieldStageData(
        stage_time=jnp.asarray(time, dtype=jnp.float64),
        exact_halo=exact_halo,
        source_halo=_shifted_torus_local_mms_source_state_from_invariants(
            coordinates_halo,
            time,
            invariants=invariants,
            parameters=parameters,
        ),
        phi_halo=phi_halo,
        phi_face_lower=phi_face_lower,
        phi_face_upper=phi_face_upper,
        density_face_lower=density_face_lower,
        density_face_upper=density_face_upper,
        omega_face_lower=omega_face_lower,
        omega_face_upper=omega_face_upper,
        v_ion_face_lower=v_ion_face_lower,
        v_ion_face_upper=v_ion_face_upper,
        v_electron_face_lower=v_electron_face_lower,
        v_electron_face_upper=v_electron_face_upper,
    )


def _build_local_4field_rk4_stage_data(
    invariants: _ShiftedTorus4FieldInvariantBundle,
    step_time: float | jax.Array,
    step_timestep: float | jax.Array,
    *,
    parameters: Fci4FieldRhsParameters,
) -> _ShiftedTorus4FieldRk4StageData:
    half_step = 0.5 * step_timestep
    return _ShiftedTorus4FieldRk4StageData(
        stage_1=_build_local_4field_stage_data(invariants, step_time, parameters=parameters),
        stage_2=_build_local_4field_stage_data(invariants, step_time + half_step, parameters=parameters),
        stage_3=_build_local_4field_stage_data(invariants, step_time + half_step, parameters=parameters),
        stage_4=_build_local_4field_stage_data(invariants, step_time + step_timestep, parameters=parameters),
    )


def _prepare_local_shifted_torus_4field_stage_state(
    state_owned: Fci4FieldState,
    stage_data: _ShiftedTorus4FieldStageData,
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
) -> PreparedLocalState3D:
    from jax_drb.native.fci_boundaries import LocalBoundaryData3D
    from jax_drb.native.fci_halo import LocalHaloClosure3D, PreparedLocalState3D
    from jax_drb.native.fci_model import inject_owned_state_to_halo

    state_halo = inject_owned_state_to_halo(state_owned, domain.layout)
    face_bc_bundle = _ShiftedTorus4FieldFaceBCBundle(
        phi=_build_local_radial_dirichlet_face_bc_from_values(stage_data.phi_face_lower, stage_data.phi_face_upper, domain),
        density=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.density_face_lower,
            stage_data.density_face_upper,
            domain,
        ),
        omega=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.omega_face_lower,
            stage_data.omega_face_upper,
            domain,
        ),
        v_ion_parallel=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.v_ion_face_lower,
            stage_data.v_ion_face_upper,
            domain,
        ),
        v_electron_parallel=_build_local_radial_dirichlet_face_bc_from_values(
            stage_data.v_electron_face_lower,
            stage_data.v_electron_face_upper,
            domain,
        ),
    )
    closure = LocalHaloClosure3D(
        physical_ghost_filler=physical_ghost_filler,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
    )
    prepared_state_halo = Fci4FieldState(
        density=closure(state_halo.density, domain, face_bc_bundle.density),
        omega=closure(state_halo.omega, domain, face_bc_bundle.omega),
        v_ion_parallel=closure(
            state_halo.v_ion_parallel,
            domain,
            face_bc_bundle.v_ion_parallel,
        ),
        v_electron_parallel=closure(
            state_halo.v_electron_parallel,
            domain,
            face_bc_bundle.v_electron_parallel,
        ),
    )
    return PreparedLocalState3D(
        state_halo=prepared_state_halo,
        boundary_data=LocalBoundaryData3D(face_bc=face_bc_bundle),
    )

__all__ = [
    'A_phi',
    'A_n',
    'A_e',
    'A_i',
    'a_phi',
    'a_n',
    'a_e',
    'a_i',
    'Omega',
    'M_phi',
    'N_phi',
    'M_n',
    'N_n',
    'M_e',
    'N_e',
    'M_i',
    'N_i',
    'n0',
    'rho_star',
    'Te',
    'mi_over_me',
    'sigma',
    'r0',
    'alpha_value',
    'iota',
    'c_phi',
    'x_min',
    'x_max',
    'tf',
    'num_steps',
    '_ShiftedTorus4FieldFaceBCBundle',
    '_ShiftedTorus4FieldInvariantBundle',
    '_ShiftedTorus4FieldStageData',
    '_ShiftedTorus4FieldRk4StageData',
    '_resolution_step_count',
    '_format_progress_bar',
    '_format_duration',
    'build_shifted_torus_4field_geometry',
    '_shifted_torus_coordinates',
    '_shifted_torus_geometry_quantities_scalar',
    '_shifted_torus_envelopes',
    '_shifted_torus_phi',
    '_shifted_torus_phi_scalar',
    '_shifted_torus_phi_t',
    '_dirichlet_x_boundary_face_bc_from_values',
    '_dirichlet_x_boundary_face_bc',
    '_shifted_torus_density',
    '_shifted_torus_density_t',
    '_shifted_torus_v_electron_parallel',
    '_shifted_torus_v_electron_parallel_t',
    '_shifted_torus_omega_scalar',
    '_shifted_torus_omega',
    '_shifted_torus_omega_t',
    '_shifted_torus_omega_and_derivatives',
    '_shifted_torus_x_face_coordinates',
    '_split_lower_upper_face_values',
    '_shifted_torus_phi_x_face_values',
    '_shifted_torus_density_x_face_values',
    '_shifted_torus_omega_x_face_values',
    '_shifted_torus_v_ion_parallel_x_face_values',
    '_shifted_torus_v_electron_parallel_x_face_values',
    '_shifted_torus_exact_x_face_bcs',
    '_shifted_torus_v_ion_parallel',
    '_shifted_torus_v_ion_parallel_t',
    '_shifted_torus_phi_derivatives',
    '_shifted_torus_density_derivatives',
    '_shifted_torus_v_ion_parallel_derivatives',
    '_shifted_torus_v_electron_parallel_derivatives',
    '_build_dirichlet_boundary_condition_builder',
    '_homogeneous_boundary_face_bc',
    '_shifted_torus_exact_state',
    '_shifted_torus_exact_time_derivative_state',
    '_continuous_4field_rhs_from_exact_state',
    '_continuous_4field_rhs_terms_from_exact_state',
    '_sum_rhs_terms',
    '_shifted_torus_mms_source_state',
    '_add_state',
    '_raise_if_nonfinite_state',
    '_combined_error_statistics',
    '_field_error_statistics',
    '_state_error_statistics',
    '_print_state_error_statistics',
    '_observed_order',
    '_format_order',
    '_shifted_torus_local_coordinates',
    '_shifted_torus_local_envelopes',
    '_shifted_torus_local_phi_derivatives',
    '_shifted_torus_local_density_derivatives',
    '_shifted_torus_local_v_ion_parallel_derivatives',
    '_shifted_torus_local_v_electron_parallel_derivatives',
    '_shifted_torus_local_omega_and_derivatives',
    '_shifted_torus_local_exact_state',
    '_shifted_torus_local_exact_time_derivative_state',
    '_shifted_torus_local_mms_source_state',
    '_state_partition_spec',
    '_put_state_on_mesh',
    '_gather_state_from_mesh',
    '_build_ghost_filler',
    '_apply_local_owned_dirichlet_to_field',
    '_apply_local_owned_dirichlet_to_state',
    '_build_local_radial_dirichlet_face_bc',
    '_build_local_radial_dirichlet_face_bc_from_values',
    '_shifted_torus_local_x_face_coordinates',
    '_split_local_x_face_values',
    '_coordinates_from_4field_invariants',
    '_face_coordinates_from_4field_invariants',
    '_build_local_4field_invariants',
    '_shifted_torus_local_mms_source_state_from_invariants',
    '_build_local_4field_stage_data',
    '_build_local_4field_rk4_stage_data',
    '_prepare_local_shifted_torus_4field_stage_state',
]
