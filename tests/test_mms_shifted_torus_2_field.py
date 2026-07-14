from __future__ import annotations

import time as time_module

import jax
import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FciGeometry3D,
    FciMaps3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    Grid1D,
    LocalStencilBuilder,
    MetricGeometry,
    RegularFaceGeometry3D,
    Spacing3D,
    build_curvature_coefficients,
    build_fci_maps_from_b_contravariant,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import Rk4Stepper, sum_stage_outputs
from jax_drb.native.fci_2_field_rhs import Fci2FieldRhsParameters, Fci2FieldState, compute_2field_rhs
from jax_drb.native.fci_boundaries import BC_DIRICHLET, BoundaryConditionBuilder, BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D


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


def _resolution_step_count(resolution: int, *, base_resolution: int = 20, base_steps: int = num_steps) -> int:
    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


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


def shifted_torus_2field_rk4(
    state: Fci2FieldState,
    *,
    geometry: FciGeometry3D,
    time: float,
    timestep: float,
    parameters: Fci2FieldRhsParameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder,
    density_bc_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
    phi_bc_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
    v_parallel_bc_builder: BoundaryConditionBuilder[tuple[BoundaryFaceBC3D, CutWallBC3D]],
) -> tuple[Fci2FieldState, jnp.ndarray]:
    """Advance the shifted-torus two-field MMS state by one RK4 step."""

    empty_cut_wall_geometry = CutWallGeometry3D.empty()
    def _rhs_fn(
        current_state: Fci2FieldState,
        stage_time: float | jax.Array,
        carry: None,
    ) -> tuple[Fci2FieldState, None, jnp.ndarray]:
        del carry
        boundary_start = time_module.perf_counter()
        exact_state = _shifted_torus_exact_state(geometry, float(stage_time))
        density_face_bc, density_cut_wall_bc = density_bc_builder(
            exact_state.density,
            geometry,
            (False, True, True),
            empty_cut_wall_geometry,
            CutWallBC3D.empty(),
        )
        phi_face_bc, phi_cut_wall_bc = phi_bc_builder(
            _shifted_torus_phi(geometry, float(stage_time)),
            geometry,
            (False, True, True),
            empty_cut_wall_geometry,
            CutWallBC3D.empty(),
        )
        v_parallel_face_bc, v_parallel_cut_wall_bc = v_parallel_bc_builder(
            exact_state.v_parallel,
            geometry,
            (False, True, True),
            empty_cut_wall_geometry,
            CutWallBC3D.empty(),
        )
        stage_state = _apply_dirichlet_face_bcs_to_state(current_state, density_face_bc, v_parallel_face_bc)
        jax.block_until_ready(stage_state.density)
        boundary_time = time_module.perf_counter() - boundary_start
        rhs_result, timings = compute_2field_rhs(
            stage_state,
            geometry=geometry,
            stencil_builder=stencil_builder,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            periodic_axes=(False, True, True),
            density_face_bc=density_face_bc,
            phi_face_bc=phi_face_bc,
            v_parallel_face_bc=v_parallel_face_bc,
            density_cut_wall_geometry=empty_cut_wall_geometry,
            density_cut_wall_bc=density_cut_wall_bc,
            phi_cut_wall_geometry=empty_cut_wall_geometry,
            phi_cut_wall_bc=phi_cut_wall_bc,
            v_parallel_cut_wall_geometry=empty_cut_wall_geometry,
            v_parallel_cut_wall_bc=v_parallel_cut_wall_bc,
            density_source=_shifted_torus_density_source(geometry, float(stage_time), parameters=parameters),
            v_parallel_source=_shifted_torus_v_parallel_source(geometry, float(stage_time), parameters=parameters),
        )
        stage_timings = jnp.asarray([boundary_time, timings[0], timings[1]], dtype=jnp.float64)
        return rhs_result.rhs, None, stage_timings

    step_result = Rk4Stepper(_rhs_fn)(
        state,
        time=time,
        timestep=timestep,
        carry=None,
    )
    next_state = step_result.state
    final_boundary_start = time_module.perf_counter()
    exact_final = _shifted_torus_exact_state(geometry, time + timestep)
    final_density_face_bc, _ = density_bc_builder(
        exact_final.density,
        geometry,
        (False, True, True),
        empty_cut_wall_geometry,
        CutWallBC3D.empty(),
    )
    final_v_parallel_face_bc, _ = v_parallel_bc_builder(
        exact_final.v_parallel,
        geometry,
        (False, True, True),
        empty_cut_wall_geometry,
        CutWallBC3D.empty(),
    )
    next_state = _apply_dirichlet_face_bcs_to_state(next_state, final_density_face_bc, final_v_parallel_face_bc)
    jax.block_until_ready(next_state.density)
    final_boundary_time = time_module.perf_counter() - final_boundary_start
    step_timings = sum_stage_outputs(step_result.stage_aux)
    step_timings = step_timings.at[0].add(final_boundary_time)
    return next_state, step_timings


def simulate_mms_2field_shifted_torus(
    geometry: FciGeometry3D,
    *,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
) -> tuple[Fci2FieldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve the shifted-torus MMS system and return the final state plus stacked history."""

    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    density_bc_builder = BoundaryConditionBuilder(_shifted_torus_dirichlet_boundary_condition_builder("density"))
    phi_bc_builder = BoundaryConditionBuilder(_shifted_torus_dirichlet_boundary_condition_builder("phi"))
    v_parallel_bc_builder = BoundaryConditionBuilder(_shifted_torus_dirichlet_boundary_condition_builder("v_parallel"))
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    curvature_start = time_module.perf_counter()
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    curvature_build_time = time_module.perf_counter() - curvature_start
    initial_exact = _shifted_torus_exact_state(geometry, 0.0)
    initial_density_bc, _ = density_bc_builder(
        initial_exact.density,
        geometry,
        (False, True, True),
        CutWallGeometry3D.empty(),
        CutWallBC3D.empty(),
    )
    initial_v_parallel_bc, _ = v_parallel_bc_builder(
        initial_exact.v_parallel,
        geometry,
        (False, True, True),
        CutWallGeometry3D.empty(),
        CutWallBC3D.empty(),
    )
    initial_state = _apply_dirichlet_face_bcs_to_state(initial_exact, initial_density_bc, initial_v_parallel_bc)
    state = initial_state
    time_value = 0.0
    times: list[float] = [0.0]
    density_history: list[jnp.ndarray] = [jnp.asarray(initial_state.density, dtype=jnp.float32)]
    v_parallel_history: list[jnp.ndarray] = [jnp.asarray(initial_state.v_parallel, dtype=jnp.float32)]
    timing_history: list[jnp.ndarray] = []

    for _ in range(steps):
        state, step_timings = shifted_torus_2field_rk4(
            state,
            geometry=geometry,
            time=time_value,
            timestep=dt,
            parameters=parameters,
            curvature_coefficients=curvature_coefficients,
            stencil_builder=stencil_builder,
            density_bc_builder=density_bc_builder,
            phi_bc_builder=phi_bc_builder,
            v_parallel_bc_builder=v_parallel_bc_builder,
        )
        time_value += dt
        times.append(time_value)
        density_history.append(jnp.asarray(state.density, dtype=jnp.float32))
        v_parallel_history.append(jnp.asarray(state.v_parallel, dtype=jnp.float32))
        timing_history.append(step_timings)

    if timing_history:
        timing_array = np.asarray(timing_history, dtype=np.float64)
        mean_boundary_time = float(np.mean(timing_array[:, 0]))
        mean_stencil_time = float(np.mean(timing_array[:, 1]))
        mean_operator_time = float(np.mean(timing_array[:, 2]))
        print(f"shifted_torus_2field curvature coefficient build time: {curvature_build_time:.6e} s")
        print(
            "shifted_torus_2field mean timings per RK step: "
            f"boundary={mean_boundary_time:.6e} s, "
            f"stencil={mean_stencil_time:.6e} s, "
            f"operator={mean_operator_time:.6e} s"
        )

    return (
        state,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    resolutions = np.asarray([30, 60,120], dtype=np.int64)
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
        geometry = build_shifted_torus_2field_geometry((int(resolution), int(resolution), int(resolution)))
        steps = _resolution_step_count(int(resolution))
        dt = float(tf) / float(steps)
        print(f"Starting simulation for resolution={int(resolution)}, steps={steps}, dt={dt:.6e}")
        start = time_module.perf_counter()
        try:
            final_state, times, density_history, v_parallel_history = simulate_mms_2field_shifted_torus(
                geometry,
                final_time=tf,
                timestep=dt,
                rho_star_value=rho_star,
            )
            elapsed = time_module.perf_counter() - start
            mean_error, _, max_error = _combined_error_statistics(final_state, geometry, tf)
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

        final_resolution_state = final_state
        final_resolution_geometry = geometry
        final_resolution = int(resolution)
        final_resolution_times = times
        final_resolution_density_history = density_history
        final_resolution_v_parallel_history = v_parallel_history

    if successful_resolutions:
        plotted_resolutions = np.asarray(successful_resolutions, dtype=np.int64)
        log_resolutions = np.log(plotted_resolutions.astype(np.float64))
        l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
        max_log_errors = np.log(np.asarray(max_errors, dtype=np.float64))
        l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
        max_slope, max_intercept = np.polyfit(log_resolutions, max_log_errors, 1)
        print(f"shifted_torus_2field l2 convergence order: {-l2_slope:.6f}")
        print(f"shifted_torus_2field max convergence order: {-max_slope:.6f}")

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"l2, order {l2_slope:.2f}")
        ax.loglog(plotted_resolutions, max_errors, "^-", label=f"max, order {max_slope:.2f}")
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
        ax.set_title("Shifted-torus 2-field MMS convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend()
        fig.tight_layout()
        fig.savefig("shifted_torus_2field_convergence.png", dpi=200)
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
            "shifted_torus_2field_slices.png",
        )

    if (
        final_resolution_times is not None
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
            "shifted_torus_2field_slices.gif",
            frame_stride=5,
        )
