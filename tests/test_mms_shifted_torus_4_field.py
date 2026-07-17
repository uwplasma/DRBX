from __future__ import annotations

import time as time_module

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    ConservativeStencilBuilder,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    LocalStencilBuilder,
    MetricGeometry,
    RegularFaceGeometry3D,
    Spacing3D,
    build_curvature_coefficients,
    build_conservative_stencil_from_field,
    build_fci_maps_from_b_contravariant,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import (
    Fci4FieldRhsParameters,
    Fci4FieldState,
    curvature_op,
    build_perp_laplacian_mg_hierarchy,
    build_perp_laplacian_face_projectors,
    compute_4field_rhs,
    poisson_bracket_op,
    perp_laplacian_conservative_op,
    rk4_step,
    sum_stage_outputs,
)
from jax_drb.native.fci_operators import PerpLaplacianInverseSolver, grad_parallel_op_direct
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BoundaryConditionBuilder,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
)


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
x_min = 0.15
x_max = 1.0
tf = 0.1
num_steps = 30


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

    return {
        "density": {
            "poisson": -(_poisson(phi_grad, density_grad) / (rho_star_value * bmag)),
            "curvature_density": (2.0 * te / bmag) * curvature_density,
            "curvature_phi": -(2.0 * density / bmag) * curvature_phi,
            "parallel_v_electron": -density * grad_parallel_v_electron,
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


def _print_convergence_table(
    title: str,
    results: list[tuple[int, dict[str, tuple[float, float, float]]]],
    *,
    stat_index: int = 0,
) -> None:
    print(title)
    previous_resolution: int | None = None
    previous_stats: dict[str, tuple[float, float, float]] | None = None
    for resolution, stats in results:
        print(f"  resolution={resolution}")
        for field_name, field_stats in stats.items():
            order_text = ""
            if previous_resolution is not None and previous_stats is not None and field_name in previous_stats:
                order = _observed_order(
                    previous_stats[field_name][stat_index],
                    field_stats[stat_index],
                    previous_resolution,
                    resolution,
                )
                order_text = f", order={_format_order(order)}"
            l2, linf, rel_l2 = field_stats
            print(f"    {field_name}: l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}{order_text}")
        previous_resolution = resolution
        previous_stats = stats


def _discrete_minus_laplacian_phi(
    geometry: FciGeometry3D,
    time: float,
    *,
    conservative_stencil_builder: ConservativeStencilBuilder,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    periodic_axes = (False, True, True)
    exact_phi = _shifted_torus_phi(geometry, time)
    phi_face_bc, _, _, _, _ = _shifted_torus_exact_x_face_bcs(geometry, time)
    phi_stencil = conservative_stencil_builder(
        exact_phi,
        geometry,
        periodic_axes,
        phi_face_bc,
    )
    return -perp_laplacian_conservative_op(
        phi_stencil,
        geometry,
        face_projectors=face_projectors,
        face_bc=phi_face_bc,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=periodic_axes,
    )


def _phi_vorticity_mismatch_statistics(
    geometry: FciGeometry3D,
    time: float,
) -> dict[str, tuple[float, float, float]]:
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    discrete_minus_lap_phi = _discrete_minus_laplacian_phi(
        geometry,
        time,
        conservative_stencil_builder=conservative_stencil_builder,
        face_projectors=face_projectors,
    )
    exact_omega_rhs = -_shifted_torus_exact_state(geometry, time).omega
    return {"-lap_perp(phi) vs -omega": _field_error_statistics(discrete_minus_lap_phi, exact_omega_rhs)}


def _discrete_4field_rhs_terms_with_phi(
    state: Fci4FieldState,
    phi: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
    face_bcs: tuple[BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D, BoundaryFaceBC3D],
    stencil_builder: LocalStencilBuilder,
) -> dict[str, dict[str, jnp.ndarray]]:
    periodic_axes = (False, True, True)
    phi_face_bc, density_face_bc, omega_face_bc, v_ion_face_bc, v_electron_face_bc = face_bcs
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    density = jnp.asarray(state.density, dtype=jnp.float64)
    omega = jnp.asarray(state.omega, dtype=jnp.float64)
    v_ion_parallel = jnp.asarray(state.v_ion_parallel, dtype=jnp.float64)
    v_electron_parallel = jnp.asarray(state.v_electron_parallel, dtype=jnp.float64)
    rho_star_value = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    te = jnp.asarray(parameters.Te, dtype=jnp.float64)
    mi_over_me_value = jnp.asarray(parameters.mi_over_me, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)
    density_safe = jnp.maximum(density, 1.0e-30)

    density_stencil = stencil_builder(density, geometry, periodic_axes, density_face_bc, empty_cut_wall_geometry, empty_cut_wall_bc)
    omega_stencil = stencil_builder(omega, geometry, periodic_axes, omega_face_bc, empty_cut_wall_geometry, empty_cut_wall_bc)
    phi_stencil = stencil_builder(phi, geometry, periodic_axes, phi_face_bc, empty_cut_wall_geometry, empty_cut_wall_bc)
    v_ion_stencil = stencil_builder(v_ion_parallel, geometry, periodic_axes, v_ion_face_bc, empty_cut_wall_geometry, empty_cut_wall_bc)
    v_electron_stencil = stencil_builder(
        v_electron_parallel,
        geometry,
        periodic_axes,
        v_electron_face_bc,
        empty_cut_wall_geometry,
        empty_cut_wall_bc,
    )

    poisson_density = poisson_bracket_op(phi_stencil, density_stencil, geometry)
    poisson_omega = poisson_bracket_op(phi_stencil, omega_stencil, geometry)
    poisson_v_ion = poisson_bracket_op(phi_stencil, v_ion_stencil, geometry)
    poisson_v_electron = poisson_bracket_op(phi_stencil, v_electron_stencil, geometry)
    curvature_density = curvature_op(density_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_phi = curvature_op(phi_stencil, geometry, curvature_coefficients=curvature_coefficients)
    grad_parallel_density = grad_parallel_op_direct(density_stencil, geometry)
    grad_parallel_phi = grad_parallel_op_direct(phi_stencil, geometry)
    grad_parallel_v_ion = grad_parallel_op_direct(v_ion_stencil, geometry)
    grad_parallel_v_electron = grad_parallel_op_direct(v_electron_stencil, geometry)

    return {
        "density": {
            "poisson": -(poisson_density / (rho_star_value * bmag)),
            "curvature_density": (2.0 * te / bmag) * curvature_density,
            "curvature_phi": -(2.0 * density / bmag) * curvature_phi,
            "parallel_v_electron": -density * grad_parallel_v_electron,
        },
        "omega": {
            "poisson": -(poisson_omega / (rho_star_value * bmag)),
            "parallel_current": (bmag * bmag / density_safe) * (grad_parallel_v_ion - grad_parallel_v_electron),
            "curvature_density": (2.0 * bmag * te / density_safe) * curvature_density,
        },
        "v_ion_parallel": {
            "poisson": -(poisson_v_ion / (rho_star_value * bmag)),
            "grad_density": -(te / density_safe) * grad_parallel_density,
        },
        "v_electron_parallel": {
            "poisson": -(poisson_v_electron / (rho_star_value * bmag)),
            "grad_phi": mi_over_me_value * grad_parallel_phi,
            "grad_density": -mi_over_me_value * (te / density_safe) * grad_parallel_density,
        },
    }


def _exact_phi_rhs_consistency_statistics(
    geometry: FciGeometry3D,
    time: float,
    *,
    parameters: Fci4FieldRhsParameters,
) -> dict[str, tuple[float, float, float]]:
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_terms = _discrete_4field_rhs_terms_with_phi(
        exact_state,
        _shifted_torus_phi(geometry, time),
        geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        face_bcs=_shifted_torus_exact_x_face_bcs(geometry, time),
        stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
    )
    computed_derivative = _add_state(_sum_rhs_terms(discrete_terms), source, scale=1.0)
    return _state_error_statistics(computed_derivative, exact_derivative)


def _report_exact_phi_term_breakdown(
    geometry: FciGeometry3D,
    time: float,
    *,
    parameters: Fci4FieldRhsParameters,
) -> None:
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    continuous_terms = _continuous_4field_rhs_terms_from_exact_state(
        exact_state,
        geometry,
        time=time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_terms = _discrete_4field_rhs_terms_with_phi(
        exact_state,
        _shifted_torus_phi(geometry, time),
        geometry,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        face_bcs=_shifted_torus_exact_x_face_bcs(geometry, time),
        stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
    )
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    discrete_rhs = _sum_rhs_terms(discrete_terms)
    computed_derivative = _add_state(discrete_rhs, source, scale=1.0)
    total_stats = _state_error_statistics(computed_derivative, exact_derivative)
    print(f"shifted_torus_4field exact-phi per-term RHS residual breakdown at resolution={geometry.shape[0]}")
    for field_name in ("density", "omega", "v_ion_parallel", "v_electron_parallel"):
        print(
            f"  {field_name}: total_l2={total_stats[field_name][0]:.6e}, "
            f"total_linf={total_stats[field_name][1]:.6e}, total_rel_l2={total_stats[field_name][2]:.6e}"
        )
        for term_name, discrete_value in discrete_terms[field_name].items():
            continuous_value = continuous_terms[field_name][term_name]
            l2, linf, rel_l2 = _field_error_statistics(discrete_value, continuous_value)
            print(f"    {term_name}: term_error_l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")
        source_value = getattr(source, field_name)
        expected_source = getattr(exact_derivative, field_name) - getattr(_sum_rhs_terms(continuous_terms), field_name)
        l2, linf, rel_l2 = _field_error_statistics(source_value, expected_source)
        print(f"    source_cancellation: term_error_l2={l2:.6e}, linf={linf:.6e}, rel_l2={rel_l2:.6e}")


def _report_rhs_consistency(
    geometry: FciGeometry3D,
    *,
    time: float,
    rho_star_value: float,
    use_multigrid_preconditioner: bool = False,
    gmres_debug: bool = False,
) -> None:
    """Compare discrete RHS plus MMS source against exact time derivatives."""

    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=1.0e-4,
    )
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_builder = BoundaryConditionBuilder(_build_dirichlet_boundary_condition_builder("density"))
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    periodic_axes = (False, True, True)

    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=periodic_axes)
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    exact_state = _shifted_torus_exact_state(geometry, time)
    exact_derivative = _shifted_torus_exact_time_derivative_state(geometry, time)
    source = _shifted_torus_mms_source_state(
        geometry,
        time,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
    )
    exact_phi = _shifted_torus_phi(geometry, time)

    phi_face_bc, density_face_bc, omega_face_bc, v_ion_face_bc, v_electron_face_bc = _shifted_torus_exact_x_face_bcs(
        geometry,
        time,
    )

    mg_hierarchy = None
    if use_multigrid_preconditioner:
        mg_hierarchy = build_perp_laplacian_mg_hierarchy(
            geometry,
            conservative_stencil_builder,
            face_bc=_homogeneous_boundary_face_bc(phi_face_bc),
            regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
            cut_wall_geometry=empty_cut_wall_geometry,
            cut_wall_bc=empty_cut_wall_bc,
            periodic_axes=periodic_axes,
        )
    _report_phi_inversion_consistency(
        geometry,
        time=time,
        parameters=parameters,
        conservative_stencil_builder=conservative_stencil_builder,
        boundary_builder=boundary_builder,
        face_projectors=face_projectors,
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=periodic_axes,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )

    rhs_result, _timings = compute_4field_rhs(
        exact_state,
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        phi_face_bc=phi_face_bc,
        density_face_bc=density_face_bc,
        omega_face_bc=omega_face_bc,
        v_ion_parallel_face_bc=v_ion_face_bc,
        v_electron_parallel_face_bc=v_electron_face_bc,
        phi_cut_wall_geometry=empty_cut_wall_geometry,
        phi_cut_wall_bc=empty_cut_wall_bc,
        density_cut_wall_geometry=empty_cut_wall_geometry,
        density_cut_wall_bc=empty_cut_wall_bc,
        omega_cut_wall_geometry=empty_cut_wall_geometry,
        omega_cut_wall_bc=empty_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=empty_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=empty_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=empty_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=empty_cut_wall_bc,
        phi_face_projectors=face_projectors,
        phi_mg_hierarchy=mg_hierarchy,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=gmres_debug,
    )
    computed_derivative = _add_state(rhs_result.rhs, source, scale=1.0)
    print(f"shifted_torus_4field RHS consistency at t={time:.6e}, shape={geometry.shape}")
    for field_name, actual, expected in (
        ("density", computed_derivative.density, exact_derivative.density),
        ("omega", computed_derivative.omega, exact_derivative.omega),
        ("v_ion_parallel", computed_derivative.v_ion_parallel, exact_derivative.v_ion_parallel),
        ("v_electron_parallel", computed_derivative.v_electron_parallel, exact_derivative.v_electron_parallel),
    ):
        error = jnp.asarray(actual - expected, dtype=jnp.float64)
        expected_norm = float(jnp.linalg.norm(jnp.asarray(expected, dtype=jnp.float64)))
        error_l2 = float(jnp.sqrt(jnp.mean(jnp.square(error))))
        error_linf = float(jnp.max(jnp.abs(error)))
        rel_l2 = float(jnp.linalg.norm(error) / (jnp.linalg.norm(jnp.asarray(expected, dtype=jnp.float64)) + 1.0e-30))
        print(
            f"  {field_name}: l2={error_l2:.6e}, linf={error_linf:.6e}, "
            f"rel_l2={rel_l2:.6e}, expected_l2={expected_norm:.6e}"
        )


def _report_phi_inversion_consistency(
    geometry: FciGeometry3D,
    *,
    time: float,
    parameters: Fci4FieldRhsParameters,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    mg_hierarchy: object | None = None,
    gmres_debug: bool = False,
) -> tuple[float, float, float]:
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    periodic_axes = (False, True, True)
    exact_phi = _shifted_torus_phi(geometry, time)
    exact_omega = _shifted_torus_exact_state(geometry, time).omega
    phi_face_bc, _, _, _, _ = _shifted_torus_exact_x_face_bcs(geometry, time)
    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=periodic_axes,
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        mg_hierarchy=mg_hierarchy,
        gmres_debug=gmres_debug,
    )
    phi_from_inverse, phi_diagnostics = phi_inverse_solver(
        -exact_omega,
        face_bc=phi_face_bc,
        return_diagnostics=True,
    )
    phi_error = jnp.asarray(phi_from_inverse - exact_phi, dtype=jnp.float64)
    l2_error = float(jnp.sqrt(jnp.mean(jnp.square(phi_error))))
    linf_error = float(jnp.max(jnp.abs(phi_error)))
    rel_l2_error = float(jnp.linalg.norm(phi_error) / (jnp.linalg.norm(exact_phi) + 1.0e-30))
    print(
        "shifted_torus_4field phi inversion consistency: "
        f"shape={geometry.shape}, l2={l2_error:.6e}, "
        f"linf={linf_error:.6e}, rel_l2={rel_l2_error:.6e}, "
        f"gmres_rel_res={float(phi_diagnostics['final_residual_rel_l2']):.6e}, "
        f"gmres_steps={int(phi_diagnostics['num_steps'])}"
    )
    return l2_error, linf_error, rel_l2_error


def _run_single_rhs_diagnostic_sweeps(
    resolutions: np.ndarray,
    *,
    time: float,
    rho_star_value: float,
    phi_inversion_tol: float = 1.0e-4,
) -> None:
    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=float(phi_inversion_tol),
    )
    phi_vorticity_results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    exact_phi_rhs_results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    for resolution in resolutions:
        geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
        print(f"Diagnostic operator sweep for resolution={int(resolution)}")
        phi_vorticity_results.append(
            (
                int(resolution),
                _phi_vorticity_mismatch_statistics(geometry, time),
            )
        )
        exact_phi_rhs_results.append(
            (
                int(resolution),
                _exact_phi_rhs_consistency_statistics(
                    geometry,
                    time,
                    parameters=parameters,
                ),
            )
        )
    _print_convergence_table(
        "shifted_torus_4field discrete phi-vorticity consistency convergence",
        phi_vorticity_results,
    )
    _print_convergence_table(
        "shifted_torus_4field exact-phi RHS consistency convergence",
        exact_phi_rhs_results,
    )


def _run_phi_inversion_tolerance_sweep(
    resolutions: np.ndarray,
    *,
    time: float,
    rho_star_value: float,
    tolerances: tuple[float, ...] = (1.0e-4, 1.0e-8),
) -> None:
    for tolerance in tolerances:
        print(f"shifted_torus_4field phi inversion tolerance sweep: tol={tolerance:.1e}")
        results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
        for resolution in resolutions:
            geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
            parameters = Fci4FieldRhsParameters(
                rho_star=rho_star_value,
                Te=float(Te),
                mi_over_me=float(mi_over_me),
                phi_inversion_tol=float(tolerance),
            )
            conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
            face_projectors = build_perp_laplacian_face_projectors(geometry)
            l2, linf, rel_l2 = _report_phi_inversion_consistency(
                geometry,
                time=time,
                parameters=parameters,
                conservative_stencil_builder=conservative_stencil_builder,
                boundary_builder=BoundaryConditionBuilder(_build_dirichlet_boundary_condition_builder("phi")),
                face_projectors=face_projectors,
            )
            results.append((int(resolution), {"phi_from_inverse": (l2, linf, rel_l2)}))
        _print_convergence_table(
            f"shifted_torus_4field phi inversion convergence at tol={tolerance:.1e}",
            results,
        )


def _run_timestep_convergence(
    *,
    resolution: int,
    step_counts: tuple[int, ...],
    rho_star_value: float,
) -> None:
    results: list[tuple[int, dict[str, tuple[float, float, float]]]] = []
    geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
    for steps in step_counts:
        dt = float(tf) / float(steps)
        print(f"Starting timestep convergence run for resolution={resolution}, steps={steps}, dt={dt:.6e}")
        final_state, *_ = simulate_mms_shifted_torus_4field(
            geometry,
            final_time=tf,
            timestep=dt,
            rho_star_value=rho_star_value,
            show_progress=True,
        )
        exact_state = _shifted_torus_exact_state(geometry, tf)
        stats = _state_error_statistics(final_state, exact_state)
        _print_state_error_statistics(f"timestep convergence per-field errors: steps={steps}", stats)
        results.append((int(steps), stats))
    print("shifted_torus_4field timestep convergence at fixed resolution")
    previous_steps: int | None = None
    previous_stats: dict[str, tuple[float, float, float]] | None = None
    for steps, stats in results:
        print(f"  steps={steps}")
        for field_name, field_stats in stats.items():
            order_text = ""
            if previous_steps is not None and previous_stats is not None:
                order = _observed_order(previous_stats[field_name][0], field_stats[0], previous_steps, steps)
                order_text = f", order_vs_dt={_format_order(order)}"
            print(f"    {field_name}: l2={field_stats[0]:.6e}, linf={field_stats[1]:.6e}, rel_l2={field_stats[2]:.6e}{order_text}")
        previous_steps = steps
        previous_stats = stats


def shifted_torus_4field_rk4(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    time: float,
    timestep: float,
    parameters: Fci4FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    phi_mg_hierarchy: object | None = None,
    phi_inverse_solver: PerpLaplacianInverseSolver | None = None,
    gmres_debug: bool = False,
    phi_guess: jnp.ndarray | None = None,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
    """Advance the shifted-torus four-field MMS state by one RK4 step."""

    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()
    initial_phi_guess = jnp.asarray(_shifted_torus_phi(geometry, time) if phi_guess is None else phi_guess, dtype=jnp.float64)

    def _rhs_fn(
        current_state: Fci4FieldState,
        stage_time: float | jax.Array,
        carry: jnp.ndarray,
    ) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
        stage_time_value = float(stage_time)
        phi_face_bc, density_face_bc, omega_face_bc, v_ion_face_bc, v_electron_face_bc = _shifted_torus_exact_x_face_bcs(
            geometry,
            stage_time_value,
        )
        rhs_result, timings, stage_phi = compute_4field_rhs(
            current_state,
            with_diagnostics=True,
            geometry=geometry,
            stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_stencil_builder,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            phi_guess=carry,
            phi_face_bc=phi_face_bc,
            density_face_bc=density_face_bc,
            omega_face_bc=omega_face_bc,
            v_ion_parallel_face_bc=v_ion_face_bc,
            v_electron_parallel_face_bc=v_electron_face_bc,
            phi_cut_wall_geometry=empty_cut_wall_geometry,
            phi_cut_wall_bc=empty_cut_wall_bc,
            density_cut_wall_geometry=empty_cut_wall_geometry,
            density_cut_wall_bc=empty_cut_wall_bc,
            omega_cut_wall_geometry=empty_cut_wall_geometry,
            omega_cut_wall_bc=empty_cut_wall_bc,
            v_ion_parallel_cut_wall_geometry=empty_cut_wall_geometry,
            v_ion_parallel_cut_wall_bc=empty_cut_wall_bc,
            v_electron_parallel_cut_wall_geometry=empty_cut_wall_geometry,
            v_electron_parallel_cut_wall_bc=empty_cut_wall_bc,
            phi_face_projectors=face_projectors,
            phi_mg_hierarchy=phi_mg_hierarchy,
            phi_inverse_solver=phi_inverse_solver,
            gmres_debug=gmres_debug,
            return_phi=True,
        )
        source = _shifted_torus_mms_source_state(
            geometry,
            stage_time_value,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
        )
        rhs = rhs_result.rhs.axpy(source, scale=1.0)
        return rhs, stage_phi, timings

    step_result = rk4_step(state, time=time, timestep=timestep, rhs_fn=_rhs_fn, carry=initial_phi_guess)
    next_state = step_result.state
    phi4_start = time_module.perf_counter()
    phi_4, _diagnostics_4 = _reconstruct_phi_from_state(
        next_state,
        geometry=geometry,
        parameters=parameters,
        boundary_condition_builder=boundary_builder,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        phi_inverse_solver=phi_inverse_solver,
        periodic_axes=(False, True, True),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        phi_guess=step_result.carry,
        return_diagnostics=True,
    )
    final_phi_time = time_module.perf_counter() - phi4_start
    step_timings = sum_stage_outputs(step_result.stage_aux)
    step_timings = step_timings.at[0].add(final_phi_time)
    step_timings = step_timings.at[3].add(_diagnostic_float(_diagnostics_4, "num_steps"))
    jax.block_until_ready(next_state.density)
    return next_state, step_timings, phi_4


def simulate_mms_shifted_torus_4field(
    geometry: FciGeometry3D,
    *,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    use_multigrid_preconditioner: bool = False,
    disable_multigrid_on_failure: bool = True,
    gmres_debug: bool = False,
    show_progress: bool = False,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve the shifted-torus MMS system and return the final state plus stacked history."""

    parameters = Fci4FieldRhsParameters(
        rho_star=rho_star_value,
        Te=float(Te),
        mi_over_me=float(mi_over_me),
        phi_inversion_tol=1.0e-4,
    )
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_builder = BoundaryConditionBuilder(_build_dirichlet_boundary_condition_builder("density"))
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)

    curvature_start = time_module.perf_counter()
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    curvature_build_time = time_module.perf_counter() - curvature_start
    face_projectors = build_perp_laplacian_face_projectors(geometry)
    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    empty_cut_wall_bc = CutWallBC3D.empty()

    initial_phi = _shifted_torus_phi(geometry, 0.0)
    initial_density = _shifted_torus_density(geometry, 0.0)
    initial_omega = _shifted_torus_exact_state(geometry, 0.0).omega
    initial_v_ion_parallel = _shifted_torus_v_ion_parallel(geometry, 0.0)
    initial_v_electron_parallel = _shifted_torus_v_electron_parallel(geometry, 0.0)
    initial_state = Fci4FieldState(
        density=initial_density,
        omega=initial_omega,
        v_ion_parallel=initial_v_ion_parallel,
        v_electron_parallel=initial_v_electron_parallel,
    )
    initial_phi_face_bc, _, _, _, _ = _shifted_torus_exact_x_face_bcs(geometry, 0.0)
    phi_mg_hierarchy = None
    mg_build_time = 0.0
    if use_multigrid_preconditioner:
        mg_start = time_module.perf_counter()
        phi_mg_hierarchy = build_perp_laplacian_mg_hierarchy(
            geometry,
            conservative_stencil_builder,
            face_bc=_homogeneous_boundary_face_bc(initial_phi_face_bc),
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
                tol=float(parameters.phi_inversion_tol),
                maxiter=int(parameters.phi_inversion_maxiter),
                restart=int(parameters.phi_inversion_restart),
                face_projectors=face_projectors,
                regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
                cut_wall_geometry=empty_cut_wall_geometry,
                cut_wall_bc=empty_cut_wall_bc,
                periodic_axes=(False, True, True),
                regularization_epsilon=float(parameters.phi_inversion_regularization),
                mg_hierarchy=phi_mg_hierarchy,
                gmres_debug=gmres_debug,
            )
            phi_check = phi_check_solver(-initial_omega, face_bc=initial_phi_face_bc)
            jax.block_until_ready(phi_check)
        except RuntimeError as error:
            if not disable_multigrid_on_failure:
                raise
            print(
                "shifted_torus_4field multigrid preconditioner failed initial solve; "
                f"disabling for this run. Reason: {error}"
            )
            phi_mg_hierarchy = None

    phi_inverse_solver = PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=empty_cut_wall_geometry,
        cut_wall_bc=empty_cut_wall_bc,
        periodic_axes=(False, True, True),
        regularization_epsilon=float(parameters.phi_inversion_regularization),
        mg_hierarchy=phi_mg_hierarchy,
        gmres_debug=gmres_debug,
    )

    state = initial_state
    time_value = 0.0
    current_phi_guess: jnp.ndarray | None = jnp.asarray(initial_phi, dtype=jnp.float64)
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    omega_history: list[jnp.ndarray] = [jnp.asarray(initial_state.omega, dtype=jnp.float32)]
    v_ion_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_ion_parallel, dtype=jnp.float32)]
    v_electron_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_electron_parallel, dtype=jnp.float32)]
    timing_history: list[jnp.ndarray] = []
    progress_start = time_module.perf_counter()
    if show_progress:
        print(
            f"shifted_torus_4field RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
            end="",
            flush=True,
        )

    for step_index in range(steps):
        try:
            state, step_timings, current_phi_guess = shifted_torus_4field_rk4(
                state,
                geometry=geometry,
                time=time_value,
                timestep=dt,
                parameters=parameters,
                curvature_coefficients=curvature_coefficients,
                stencil_builder=stencil_builder,
                conservative_stencil_builder=conservative_stencil_builder,
                boundary_builder=boundary_builder,
                face_projectors=face_projectors,
                phi_mg_hierarchy=phi_mg_hierarchy,
                phi_inverse_solver=phi_inverse_solver,
                gmres_debug=gmres_debug,
                phi_guess=current_phi_guess,
            )
        except RuntimeError as error:
            state_fields = {
                "density": state.density,
                "omega": state.omega,
                "v_ion_parallel": state.v_ion_parallel,
                "v_electron_parallel": state.v_electron_parallel,
            }
            print(
                "shifted_torus_4field RK step failed: "
                f"step={step_index}, time={time_value:.6e}, dt={dt:.6e}"
            )
            for field_name, field_values in state_fields.items():
                values = jnp.asarray(field_values, dtype=jnp.float64)
                print(
                    f"  state {field_name}: finite={bool(jnp.all(jnp.isfinite(values)))}, "
                    f"min={float(jnp.nanmin(values)):.6e}, max={float(jnp.nanmax(values)):.6e}, "
                    f"l2={float(jnp.linalg.norm(jnp.nan_to_num(values))):.6e}"
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
            print(
                "\r"
                f"shifted_torus_4field RK4 progress: "
                f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    if timing_history:
        timing_array = np.asarray(timing_history, dtype=np.float64)
        print(f"shifted_torus_4field curvature coefficient build time: {curvature_build_time:.6e} s")
        if use_multigrid_preconditioner and phi_mg_hierarchy is not None:
            print(
                "shifted_torus_4field multigrid hierarchy build time: "
                f"{mg_build_time:.6e} s, levels={len(phi_mg_hierarchy.levels) if phi_mg_hierarchy is not None else 0}"
            )
        elif use_multigrid_preconditioner:
            print(
                "shifted_torus_4field multigrid preconditioner: "
                f"disabled after initial solve check, hierarchy_build_time={mg_build_time:.6e} s"
            )
        else:
            print("shifted_torus_4field multigrid preconditioner: disabled")
        print(
            "shifted_torus_4field mean timings per RK step: "
            f"phi_inverse={float(np.mean(timing_array[:, 0])):.6e} s, "
            f"local_stencil={float(np.mean(timing_array[:, 1])):.6e} s, "
            f"operator={float(np.mean(timing_array[:, 2])):.6e} s, "
            f"phi_gmres_steps_per_rk={float(np.mean(timing_array[:, 3])):.2f}, "
            f"phi_gmres_steps_per_solve={float(np.mean(timing_array[:, 3]) / 5.0):.2f}"
        )

    return (
        state,
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(density_history, axis=0),
        jnp.stack(omega_history, axis=0),
        jnp.stack(v_ion_history, axis=0),
        jnp.stack(v_electron_history, axis=0),
    )


def _shifted_torus_z_cut_indices(geometry: FciGeometry3D, count: int) -> tuple[int, ...]:
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_cuts = np.linspace(0.1, 0.9, count)
    return tuple(int(np.argmin(np.abs(z_values - cut))) for cut in z_cuts)


def _shifted_torus_field_slices(field: jnp.ndarray, z_indices: tuple[int, ...]) -> jnp.ndarray:
    return jnp.stack([field[:, :, z_index] for z_index in z_indices], axis=0)


def _symmetric_color_limit(*arrays: np.ndarray) -> float:
    vmax = float(np.max(np.abs(np.stack(arrays, axis=0))))
    return vmax if vmax > 0.0 else 1.0


def _configure_shifted_torus_slice_axis(ax, x_values: np.ndarray) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, float(x_values[-1]))
    ax.set_yticklabels([])


def _plot_final_slices(
    state: Fci4FieldState,
    exact_state: Fci4FieldState,
    geometry: FciGeometry3D,
    resolution: int,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
    z_values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    z_indices = _shifted_torus_z_cut_indices(geometry, 2)
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    field_specs = (
        ("density", state.density, exact_state.density, "viridis"),
        ("omega", state.omega, exact_state.omega, "coolwarm"),
        ("v_ion_parallel", state.v_ion_parallel, exact_state.v_ion_parallel, "coolwarm"),
        ("v_electron_parallel", state.v_electron_parallel, exact_state.v_electron_parallel, "coolwarm"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(14.0, 12.5), subplot_kw={"projection": "polar"}, constrained_layout=True)

    for row, (field_name, field, exact_field, cmap) in enumerate(field_specs):
        field_slices = np.asarray(
            _shifted_torus_field_slices(jnp.asarray(field, dtype=jnp.float64), z_indices),
            dtype=np.float64,
        )
        exact_slices = np.asarray(
            _shifted_torus_field_slices(jnp.asarray(exact_field, dtype=jnp.float64), z_indices),
            dtype=np.float64,
        )
        vmax = _symmetric_color_limit(field_slices, exact_slices)
        row_image = None
        for cut_index, z_index in enumerate(z_indices):
            row_image = axes[row, cut_index].pcolormesh(
                theta_grid,
                radius_grid,
                field_slices[cut_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            _configure_shifted_torus_slice_axis(axes[row, cut_index], x_values)
            axes[row, cut_index].set_title(f"{field_name} sim, zeta={z_values[z_index]:.3f}")

            row_image = axes[row, 2 + cut_index].pcolormesh(
                theta_grid,
                radius_grid,
                exact_slices[cut_index],
                shading="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            _configure_shifted_torus_slice_axis(axes[row, 2 + cut_index], x_values)
            axes[row, 2 + cut_index].set_title(f"{field_name} exact, zeta={z_values[z_index]:.3f}")
        if row_image is not None:
            fig.colorbar(row_image, ax=axes[row, :].ravel().tolist(), shrink=0.82, pad=0.02)

    fig.suptitle(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_shifted_torus_movie(
    times: jnp.ndarray,
    density_history: jnp.ndarray,
    omega_history: jnp.ndarray,
    v_ion_history: jnp.ndarray,
    v_electron_history: jnp.ndarray,
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
    theta_grid, radius_grid = np.meshgrid(theta_values, x_values)

    field_specs = (
        ("density", np.asarray(density_history, dtype=np.float64), "viridis"),
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
            _configure_shifted_torus_slice_axis(ax, x_values)
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

    suptitle = fig.suptitle(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}")

    def update(frame_index: int):
        actual_index = int(frame_indices[frame_index])
        time_value = float(times[actual_index])
        for row, (field_name, field_data, _) in enumerate(field_specs):
            for col, z_index in enumerate(z_indices):
                images[row * len(z_indices) + col].set_array(field_data[actual_index, :, :, z_index].ravel())
                axes[row, col].set_title(f"{field_name}, zeta={z_values[z_index]:.3f}, t={time_value:.3f}")
        suptitle.set_text(f"Shifted-torus 4-field MMS fields at resolution {int(resolution)}, t={time_value:.3f}")
        return images

    animator = animation.FuncAnimation(fig, update, frames=int(frame_indices.shape[0]), interval=100, blit=False)
    writer = animation.PillowWriter(fps=10)
    animator.save(output_path, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    diagnostic_resolutions = np.asarray([ 30, 60, 120], dtype=np.int64)
    simulation_resolutions = np.asarray([30, 60,120], dtype=np.int64)
    run_debug_diagnostics = False
    run_phi_inversion_tolerance_check = False
    run_term_breakdown = False
    run_timestep_convergence = False
    run_full_simulation = True
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    final_resolution_state: Fci4FieldState | None = None
    final_resolution_geometry: FciGeometry3D | None = None
    final_resolution: int | None = None
    final_resolution_times: jnp.ndarray | None = None
    final_resolution_density_history: jnp.ndarray | None = None
    final_resolution_omega_history: jnp.ndarray | None = None
    final_resolution_v_ion_history: jnp.ndarray | None = None
    final_resolution_v_electron_history: jnp.ndarray | None = None

    if run_debug_diagnostics:
        print("Running shifted_torus_4field single-RHS diagnostic sweeps")
        _run_single_rhs_diagnostic_sweeps(
            diagnostic_resolutions,
            time=0.0,
            rho_star_value=rho_star,
        )

    if run_phi_inversion_tolerance_check:
        print("Running shifted_torus_4field phi inversion tolerance checks")
        _run_phi_inversion_tolerance_sweep(
            np.asarray([15, 30, 60], dtype=np.int64),
            time=0.0,
            rho_star_value=rho_star,
            tolerances=(1.0e-4, 1.0e-8),
        )

    if run_term_breakdown:
        breakdown_resolution = 60
        print(f"Running shifted_torus_4field per-term RHS residual breakdown for resolution={breakdown_resolution}")
        _report_exact_phi_term_breakdown(
            build_shifted_torus_4field_geometry((breakdown_resolution, breakdown_resolution, breakdown_resolution)),
            0.0,
            parameters=Fci4FieldRhsParameters(
                rho_star=rho_star,
                Te=float(Te),
                mi_over_me=float(mi_over_me),
                phi_inversion_tol=1.0e-4,
            ),
        )

    if run_timestep_convergence:
        _run_timestep_convergence(
            resolution=30,
            step_counts=(25, 50, 100, 200),
            rho_star_value=rho_star,
        )

    if not run_full_simulation:
        raise SystemExit(0)

    for resolution in simulation_resolutions:
        geometry = build_shifted_torus_4field_geometry((int(resolution), int(resolution), int(resolution)))
        steps = _resolution_step_count(int(resolution))
        dt = float(tf) / float(steps)
        print(f"Starting simulation for resolution={int(resolution)}, steps={steps}, dt={dt:.6e}")
        start = time_module.perf_counter()
        try:
            final_state, times, density_history, omega_history, v_ion_history, v_electron_history = simulate_mms_shifted_torus_4field(
                geometry,
                final_time=tf,
                timestep=dt,
                rho_star_value=rho_star,
                show_progress=True,
            )
            elapsed = time_module.perf_counter() - start
            mean_error, _, max_error = _combined_error_statistics(final_state, geometry, tf)
            per_field_stats = _state_error_statistics(final_state, _shifted_torus_exact_state(geometry, tf))
        except FloatingPointError as exc:
            elapsed = time_module.perf_counter() - start
            print(f"WARNING: res={int(resolution)} failed with non-finite values after {elapsed:.6e} s: {exc}")
            continue

        successful_resolutions.append(int(resolution))
        l2_errors.append(mean_error)
        max_errors.append(max_error)
        print(
            f"res={int(resolution)}: steps={steps}, total_time={elapsed:.6e} s, "
            f"avg_step_time={elapsed / float(steps):.6e} s, "
            f"l2_error={mean_error:.6e}, max_error={max_error:.6e}"
        )
        _print_state_error_statistics(f"res={int(resolution)} per-field final errors", per_field_stats)
        final_resolution_state = final_state
        final_resolution_geometry = geometry
        final_resolution = int(resolution)
        final_resolution_times = times
        final_resolution_density_history = density_history
        final_resolution_omega_history = omega_history
        final_resolution_v_ion_history = v_ion_history
        final_resolution_v_electron_history = v_electron_history

    if successful_resolutions:
        import matplotlib.pyplot as plt

        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        print(f"shifted_torus_4field l2 convergence order: {-l2_slope:.6f}")
        print(f"shifted_torus_4field max convergence order: {-max_slope:.6f}")

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
        ax.set_title("Shifted-torus 4-field MMS convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig("shifted_torus_4field_convergence.png", dpi=200)
        plt.close(fig)
    else:
        print("WARNING: no valid resolutions completed, skipping convergence plot.")

    if final_resolution_state is not None and final_resolution_geometry is not None and final_resolution is not None:
        final_exact_state = _shifted_torus_exact_state(final_resolution_geometry, tf)
        _plot_final_slices(
            final_resolution_state,
            final_exact_state,
            final_resolution_geometry,
            final_resolution,
            "shifted_torus_4field_slices.png",
        )

    if (
        final_resolution_times is not None
        and final_resolution_density_history is not None
        and final_resolution_omega_history is not None
        and final_resolution_v_ion_history is not None
        and final_resolution_v_electron_history is not None
        and final_resolution_geometry is not None
        and final_resolution is not None
    ):
        _save_shifted_torus_movie(
            final_resolution_times,
            final_resolution_density_history,
            final_resolution_omega_history,
            final_resolution_v_ion_history,
            final_resolution_v_electron_history,
            final_resolution_geometry,
            final_resolution,
            "shifted_torus_4field_slices.gif",
            frame_stride=5,
        )
