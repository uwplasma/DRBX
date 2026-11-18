from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover - optional test runner dependency
    class _PytestStub:
        class _MarkStub:
            @staticmethod
            def parametrize(*args, **kwargs):
                def decorator(function):
                    return function

                return decorator

        mark = _MarkStub()

        @staticmethod
        def importorskip(name: str):
            return None

    pytest = _PytestStub()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import jax

import jax.numpy as jnp

from drbx.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    ConservativeStencilBuilder,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    MetricGeometry,
    Spacing3D,
    build_curvature_coefficients,
    build_fci_maps_from_b_contravariant,
    logical_grid_from_axis_vectors,
)
from drbx.geometry.fci_geometry import _bmag_from_contravariant_components
from drbx.native.fci_operators import (
    _take_stencil_finite_difference,
    build_conservative_stencil_from_field,
    build_perp_laplacian_mg_hierarchy,
    build_perp_laplacian_solver_mg_hierarchy,
    curvature_op,
    grad_parallel_op_direct,
    grad_parallel_op_fci,
    grad_perp_op,
    perp_laplacian_local_op,
    perp_laplacian_conservative_op,
    parallel_laplacian_direct_op,
    parallel_laplacian_conservative_op,
    PerpLaplacianInverseSolver,
    poisson_bracket_op,
)
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryFaceBC3D,
    ConservativeStencil3D,
    CutWallBC3D,
    LocalStencil1D,
    LocalStencil3D,
)
from drbx.geometry import RegularFaceGeometry3D


A = 0.2
M = 1
N = 1
R0 = 3.0
ALPHA = 0.25
C_PHI = 3.0
IOTA = 1.1


def test_local_stencil_default_derivative_weights_match_centered_formula() -> None:
    shape = (2, 2, 2)
    center = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    minus = center - 1.5
    plus = center + 2.0
    dx_min = jnp.full(shape, 0.75, dtype=jnp.float64)
    dx_plus = jnp.full(shape, 1.25, dtype=jnp.float64)
    stencil = LocalStencil1D(
        center=center,
        minus=minus,
        plus=plus,
        dx_min=dx_min,
        dx_plus=dx_plus,
    )

    denom = dx_min * dx_plus * (dx_min + dx_plus)
    c_minus = -dx_plus * dx_plus / denom
    c_center = (dx_plus * dx_plus - dx_min * dx_min) / denom
    c_plus = dx_min * dx_min / denom
    expected = c_minus * minus + c_center * center + c_plus * plus

    np.testing.assert_allclose(
        np.asarray(_take_stencil_finite_difference(stencil)),
        np.asarray(expected),
    )


def test_local_stencil_explicit_derivative_weights_override_centered_formula() -> None:
    shape = (1, 1, 1)
    stencil = LocalStencil1D(
        center=jnp.full(shape, 4.0, dtype=jnp.float64),
        minus=jnp.full(shape, 1.0, dtype=jnp.float64),
        plus=jnp.full(shape, 0.0, dtype=jnp.float64),
        dx_min=jnp.ones(shape, dtype=jnp.float64),
        dx_plus=jnp.ones(shape, dtype=jnp.float64),
        derivative_minus_weight=jnp.full(shape, -2.0, dtype=jnp.float64),
        derivative_center_weight=jnp.full(shape, 1.5, dtype=jnp.float64),
        derivative_plus_weight=jnp.full(shape, 0.5, dtype=jnp.float64),
    )

    np.testing.assert_allclose(
        np.asarray(_take_stencil_finite_difference(stencil)),
        np.asarray(jnp.full(shape, 4.0, dtype=jnp.float64)),
    )


def build_test_fci_geometry(
    shape: tuple[int, int, int],
    *,
    r0: float = R0,
    alpha: float = ALPHA,
    C_phi: float = C_PHI,
    iota: float = IOTA,
    rho_min: float = 0.2,
    construct_fci_maps: bool = False,
) -> FciGeometry3D:
    """Build shifted circular toroidal test geometry on a cell-centered grid.

    Cell-centered logical coordinates:
      x = rho      in [rho_min, 1]
      y = theta    periodic on [0, 2pi)
      z = phi      periodic on [0, 2pi)

    Coordinate map:
      R = r0 + alpha * rho + rho * cos(theta)
      X = R * cos(phi)
      Y = R * sin(phi)
      Z = rho * sin(theta)

    Magnetic field:
      B^rho   = 0
      B^theta = iota * C_phi / J
      B^phi   = C_phi / J
    """

    nx, ny, nz = shape
    target_shape = (nx, ny, nz)

    rho_faces = jnp.linspace(float(rho_min), 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)

    rho_centers = 0.5 * (rho_faces[:-1] + rho_faces[1:])
    theta_centers = 0.5 * (theta_faces[:-1] + theta_faces[1:])
    phi_centers = 0.5 * (phi_faces[:-1] + phi_faces[1:])

    grid = CellCenteredGrid3D(
        x=Grid1D(centers=rho_centers, faces=rho_faces),
        y=Grid1D(centers=theta_centers, faces=theta_faces),
        z=Grid1D(centers=phi_centers, faces=phi_faces),
    )

    # -------------------------------------------------------------------------
    # 2. Geometry evaluator for any logical-grid location family
    # -------------------------------------------------------------------------

    def _evaluate_metric(logical_grid: jnp.ndarray) -> MetricGeometry:
        rho = logical_grid[..., 0]
        theta = logical_grid[..., 1]

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        R = float(r0) + float(alpha) * rho + rho * cos_theta
        one_plus_alpha_cos = 1.0 + float(alpha) * cos_theta

        J = R * rho * one_plus_alpha_cos

        g11 = 1.0 / one_plus_alpha_cos**2
        g12 = float(alpha) * sin_theta / (rho * one_plus_alpha_cos**2)
        g13 = jnp.zeros_like(g11)

        g22 = (
            1.0
            + 2.0 * float(alpha) * cos_theta
            + float(alpha) ** 2
        ) / (rho**2 * one_plus_alpha_cos**2)
        g23 = jnp.zeros_like(g11)
        g33 = 1.0 / R**2

        g_11 = 1.0 + 2.0 * float(alpha) * cos_theta + float(alpha) ** 2
        g_12 = -float(alpha) * rho * sin_theta
        g_13 = jnp.zeros_like(g11)

        g_22 = rho**2
        g_23 = jnp.zeros_like(g11)
        g_33 = R**2

        return MetricGeometry(
            J=J,
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

    def _evaluate_B(logical_grid: jnp.ndarray, metric: MetricGeometry) -> BFieldGeometry:
        J = metric.J
        B_contra = jnp.stack(
            (
                jnp.zeros_like(J),
                float(iota) * float(C_phi) / J,
                float(C_phi) / J,
            ),
            axis=-1,
        )
        Bmag = _bmag_from_contravariant_components(B_contra, metric.g_cov)

        return BFieldGeometry(
            B_contra=B_contra,
            Bmag=Bmag,
        )

    # -------------------------------------------------------------------------
    # 3. Cell-center metric and B
    # -------------------------------------------------------------------------

    cell_logical_grid = logical_grid_from_axis_vectors(
        grid.x.centers,
        grid.y.centers,
        grid.z.centers,
    )

    cell_metric = _evaluate_metric(cell_logical_grid)

    cell_bfield = _evaluate_B(cell_logical_grid, cell_metric)
    

    xface_grid = logical_grid_from_axis_vectors(
        grid.x.faces,
        grid.y.centers,
        grid.z.centers,
    )
    yface_grid = logical_grid_from_axis_vectors(
        grid.x.centers,
        grid.y.faces,
        grid.z.centers,
    )
    zface_grid = logical_grid_from_axis_vectors(
        grid.x.centers,
        grid.y.centers,
        grid.z.faces,
    )

    face_metric = FaceMetricGeometry(
        x=_evaluate_metric(xface_grid),
        y=_evaluate_metric(yface_grid),
        z=_evaluate_metric(zface_grid),
    )

    
    face_bfield = FaceBFieldGeometry(
        x=_evaluate_B(xface_grid, face_metric.x),
        y=_evaluate_B(yface_grid, face_metric.y),
        z=_evaluate_B(zface_grid, face_metric.z),
    )
    

    if construct_fci_maps:
        map_build_start = time.perf_counter()
        print("build_test_fci_geometry: starting FCI map construction")
        map_fields = build_fci_maps_from_b_contravariant(
            grid,
            cell_bfield.B_contra,
            cell_bfield.Bmag,
            periodic_axes=(False, True, True),
        )
        map_build_elapsed = time.perf_counter() - map_build_start
        print("build_test_fci_geometry: finished FCI map construction in", float(map_build_elapsed), "s")
    else:
        ones = jnp.ones(target_shape, dtype=jnp.float64)
        zeros = jnp.zeros(target_shape, dtype=jnp.float64)

        dphi = 2.0 * jnp.pi / float(max(nz, 1))

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
            "dz": ones * dphi,
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
    
    dx = jnp.broadcast_to(
        grid.x.widths[:, None, None],
        target_shape,
    )
    dy = jnp.broadcast_to(
        grid.y.widths[None, :, None],
        target_shape,
    )
    dz = map_fields["dz"]

    spacing = Spacing3D(
        dx=dx,
        dy=dy,
        dz=dz,
    )

    geometry = FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )

    return geometry


def _mms_parallel_field_on_points(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    return_derivatives: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MMS field, and optionally its logical derivatives, on arbitrary points."""

    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = amplitude * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    if not return_derivatives:
        return field

    dfd_rho = (
        -(jnp.pi / float(A))
        * jnp.sin(jnp.pi * rho / float(A))
        * jnp.cos(float(M) * theta)
        * jnp.sin(float(N) * phi)
    )
    dfd_theta = (
        -amplitude
        * float(M)
        * jnp.sin(float(M) * theta)
        * jnp.sin(float(N) * phi)
    )
    dfd_phi = (
        amplitude
        * float(N)
        * jnp.cos(float(M) * theta)
        * jnp.cos(float(N) * phi)
    )

    return field, dfd_rho, dfd_theta, dfd_phi


def _build_mms_grad_parallel_field_and_expected(
    geometry: FciGeometry3D,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return active-cell MMS field and exact grad_parallel."""

    rho_1d = geometry.grid.x.centers
    theta_1d = geometry.grid.y.centers
    phi_1d = geometry.grid.z.centers

    logical_grid = logical_grid_from_axis_vectors(
        rho_1d,
        theta_1d,
        phi_1d,
    )

    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]

    field, dfd_rho, dfd_theta, dfd_phi = _mms_parallel_field_on_points(
        rho,
        theta,
        phi,
        return_derivatives=True,
    )

    df = jnp.stack(
        (dfd_rho, dfd_theta, dfd_phi),
        axis=-1,
    )

    expected = jnp.einsum(
        "...i,...i->...",
        geometry.cell_bfield.b_contra,
        df,
    )

    return field, expected


def _build_mms_parallel_laplacian_field_and_expected(
    geometry: FciGeometry3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return MMS field and exact direct/conservative parallel Laplacians."""

    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]

    field, _dfd_rho, dfd_theta, dfd_phi = _mms_parallel_field_on_points(
        rho,
        theta,
        phi,
        return_derivatives=True,
    )

    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    f_thetatheta = (
        -(float(M) ** 2)
        * amplitude
        * jnp.cos(float(M) * theta)
        * jnp.sin(float(N) * phi)
    )
    f_thetaphi = (
        -float(M)
        * float(N)
        * amplitude
        * jnp.sin(float(M) * theta)
        * jnp.cos(float(N) * phi)
    )
    f_phiphi = (
        -(float(N) ** 2)
        * amplitude
        * jnp.cos(float(M) * theta)
        * jnp.sin(float(N) * phi)
    )

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + float(ALPHA) * rho + rho * cos_theta
    Q = 1.0 + float(ALPHA) * cos_theta
    J = R * rho * Q
    D = jnp.sqrt((float(IOTA) ** 2) * rho**2 + R**2)

    R_theta = -rho * sin_theta
    Q_theta = -float(ALPHA) * sin_theta
    J_theta = rho * (R_theta * Q + R * Q_theta)
    D_theta = R * R_theta / D

    numerator = float(IOTA) * dfd_theta + dfd_phi
    numerator_theta = float(IOTA) * f_thetatheta + f_thetaphi
    numerator_phi = float(IOTA) * f_thetaphi + f_phiphi

    grad_parallel_theta = numerator_theta / D - numerator * D_theta / D**2
    grad_parallel_phi = numerator_phi / D
    expected_direct = (
        float(IOTA) * grad_parallel_theta + grad_parallel_phi
    ) / D

    flux_coefficient = J / D**2
    flux_coefficient_theta = J_theta / D**2 - 2.0 * J * D_theta / D**3
    expected_conservative = (
        float(IOTA)
        * (flux_coefficient_theta * numerator + flux_coefficient * numerator_theta)
        + flux_coefficient * numerator_phi
    ) / J

    return field, expected_direct, expected_conservative


def _sample_scalar_field_on_grid(
    field_fn,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    z_coords: jnp.ndarray,
) -> jnp.ndarray:
    """Sample a scalar field function on a broadcastable coordinate grid."""

    x = jnp.asarray(x_coords, dtype=jnp.float64)
    y = jnp.asarray(y_coords, dtype=jnp.float64)
    z = jnp.asarray(z_coords, dtype=jnp.float64)

    if x.ndim == y.ndim == z.ndim == 1:
        x, y, z = jnp.broadcast_arrays(
            x[:, None, None],
            y[None, :, None],
            z[None, None, :],
        )
    else:
        x, y, z = jnp.broadcast_arrays(x, y, z)

    points = jnp.stack((x, y, z), axis=-1).reshape((-1, 3))

    def _evaluate(point):
        return field_fn(point[0], point[1], point[2])

    values = jax.vmap(_evaluate)(points)
    return jnp.asarray(values, dtype=jnp.float64).reshape(x.shape)


def _build_local_stencil_dirichlet_boundaries(
    field_fn,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> LocalStencil3D:
    """Build stencil data from an analytic field and boundary-aware coordinates."""

    periodic_axes = tuple(bool(value) for value in periodic_axes)

    x_centers = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
    y_centers = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)
    x_faces = jnp.asarray(geometry.grid.x.faces, dtype=jnp.float64)
    y_faces = jnp.asarray(geometry.grid.y.faces, dtype=jnp.float64)
    z_faces = jnp.asarray(geometry.grid.z.faces, dtype=jnp.float64)

    def _neighbor_coords(centers: jnp.ndarray, faces: jnp.ndarray, periodic: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        if periodic:
            period = faces[-1] - faces[0]
            minus = jnp.concatenate((centers[-1:] - period, centers[:-1]))
            plus = jnp.concatenate((centers[1:], centers[:1] + period))
        else:
            minus = jnp.concatenate((jnp.asarray([2.0 * faces[0] - centers[0]], dtype=jnp.float64), centers[:-1]))
            plus = jnp.concatenate((centers[1:], jnp.asarray([2.0 * faces[-1] - centers[-1]], dtype=jnp.float64)))
        return minus, plus

    x_minus_coords, x_plus_coords = _neighbor_coords(x_centers, x_faces, periodic_axes[0])
    y_minus_coords, y_plus_coords = _neighbor_coords(y_centers, y_faces, periodic_axes[1])
    z_minus_coords, z_plus_coords = _neighbor_coords(z_centers, z_faces, periodic_axes[2])

    center_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_centers)
    x_minus_values = _sample_scalar_field_on_grid(field_fn, x_minus_coords, y_centers, z_centers)
    x_plus_values = _sample_scalar_field_on_grid(field_fn, x_plus_coords, y_centers, z_centers)
    y_minus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_minus_coords, z_centers)
    y_plus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_plus_coords, z_centers)
    z_minus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_minus_coords)
    z_plus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_plus_coords)

    x_dx_min = jnp.broadcast_to(
        (x_centers - x_minus_coords)[:, None, None],
        geometry.shape,
    )
    x_dx_plus = jnp.broadcast_to(
        (x_plus_coords - x_centers)[:, None, None],
        geometry.shape,
    )
    y_dx_min = jnp.broadcast_to(
        (y_centers - y_minus_coords)[None, :, None],
        geometry.shape,
    )
    y_dx_plus = jnp.broadcast_to(
        (y_plus_coords - y_centers)[None, :, None],
        geometry.shape,
    )
    z_dx_min = jnp.broadcast_to(
        (z_centers - z_minus_coords)[None, None, :],
        geometry.shape,
    )
    z_dx_plus = jnp.broadcast_to(
        (z_plus_coords - z_centers)[None, None, :],
        geometry.shape,
    )

    return LocalStencil3D(
        x=LocalStencil1D(
            center=center_values,
            minus=x_minus_values,
            plus=x_plus_values,
            dx_min=x_dx_min,
            dx_plus=x_dx_plus,
        ),
        y=LocalStencil1D(
            center=center_values,
            minus=y_minus_values,
            plus=y_plus_values,
            dx_min=y_dx_min,
            dx_plus=y_dx_plus,
        ),
        z=LocalStencil1D(
            center=center_values,
            minus=z_minus_values,
            plus=z_plus_values,
            dx_min=z_dx_min,
            dx_plus=z_dx_plus,
        ),
    )


def _build_local_stencil_neumann_boundaries(
    field_fn,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    derivative_fn=None,
) -> LocalStencil3D:
    """Build stencil data using exact Neumann reconstruction at open boundaries."""

    periodic_axes = tuple(bool(value) for value in periodic_axes)

    x_centers = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
    y_centers = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)
    x_faces = jnp.asarray(geometry.grid.x.faces, dtype=jnp.float64)
    y_faces = jnp.asarray(geometry.grid.y.faces, dtype=jnp.float64)
    z_faces = jnp.asarray(geometry.grid.z.faces, dtype=jnp.float64)

    def _neighbor_coords(centers: jnp.ndarray, faces: jnp.ndarray, periodic: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        if periodic:
            period = faces[-1] - faces[0]
            minus = jnp.concatenate((centers[-1:] - period, centers[:-1]))
            plus = jnp.concatenate((centers[1:], centers[:1] + period))
        else:
            minus = jnp.concatenate((jnp.asarray([2.0 * faces[0] - centers[0]], dtype=jnp.float64), centers[:-1]))
            plus = jnp.concatenate((centers[1:], jnp.asarray([2.0 * faces[-1] - centers[-1]], dtype=jnp.float64)))
        return minus, plus

    def _plane_derivative(axis: int, wall_coord: jnp.ndarray, tangential_1: jnp.ndarray, tangential_2: jnp.ndarray) -> jnp.ndarray:
        tangential_1, tangential_2 = jnp.broadcast_arrays(
            jnp.asarray(tangential_1, dtype=jnp.float64),
            jnp.asarray(tangential_2, dtype=jnp.float64),
        )
        flat_1 = tangential_1.reshape(-1)
        flat_2 = tangential_2.reshape(-1)
        if derivative_fn is None:
            def _scalar_field(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
                return jnp.asarray(field_fn(x, y, z), dtype=jnp.float64)

            dfdx_fn = jax.grad(_scalar_field, argnums=0)
            dfdy_fn = jax.grad(_scalar_field, argnums=1)
            dfdz_fn = jax.grad(_scalar_field, argnums=2)
            if axis == 0:
                deriv = jax.vmap(lambda yy, zz: dfdx_fn(wall_coord, yy, zz))(flat_1, flat_2)
            elif axis == 1:
                deriv = jax.vmap(lambda xx, zz: dfdy_fn(xx, wall_coord, zz))(flat_1, flat_2)
            else:
                deriv = jax.vmap(lambda xx, yy: dfdz_fn(xx, yy, wall_coord))(flat_1, flat_2)
        else:
            deriv = jax.vmap(lambda a, b: derivative_fn(axis, wall_coord, a, b))(flat_1, flat_2)
        return jnp.asarray(deriv, dtype=jnp.float64).reshape(tangential_1.shape)

    def _neumann_ghost_value(
        *,
        center_value: jnp.ndarray,
        wall_coord: jnp.ndarray,
        interior_coord: jnp.ndarray,
        tangential_1: jnp.ndarray,
        tangential_2: jnp.ndarray,
        axis: int,
        side: str,
    ) -> jnp.ndarray:
        derivative = _plane_derivative(axis, wall_coord, tangential_1, tangential_2)
        normal_sign = -1.0 if side == "lower" else 1.0
        d = jnp.abs(wall_coord - interior_coord)
        return center_value + 2.0 * d * normal_sign * derivative

    x_minus_coords, x_plus_coords = _neighbor_coords(x_centers, x_faces, periodic_axes[0])
    y_minus_coords, y_plus_coords = _neighbor_coords(y_centers, y_faces, periodic_axes[1])
    z_minus_coords, z_plus_coords = _neighbor_coords(z_centers, z_faces, periodic_axes[2])

    center_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_centers)
    x_minus_values = _sample_scalar_field_on_grid(field_fn, x_minus_coords, y_centers, z_centers)
    x_plus_values = _sample_scalar_field_on_grid(field_fn, x_plus_coords, y_centers, z_centers)
    y_minus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_minus_coords, z_centers)
    y_plus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_plus_coords, z_centers)
    z_minus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_minus_coords)
    z_plus_values = _sample_scalar_field_on_grid(field_fn, x_centers, y_centers, z_plus_coords)

    if not periodic_axes[0]:
        lower = _neumann_ghost_value(
            center_value=center_values[0, :, :],
            wall_coord=x_faces[0],
            interior_coord=x_centers[0],
            tangential_1=y_centers[:, None],
            tangential_2=z_centers[None, :],
            axis=0,
            side="lower",
        )
        upper = _neumann_ghost_value(
            center_value=center_values[-1, :, :],
            wall_coord=x_faces[-1],
            interior_coord=x_centers[-1],
            tangential_1=y_centers[:, None],
            tangential_2=z_centers[None, :],
            axis=0,
            side="upper",
        )
        x_minus_values = x_minus_values.at[0, :, :].set(lower)
        x_plus_values = x_plus_values.at[-1, :, :].set(upper)
    if not periodic_axes[1]:
        lower = _neumann_ghost_value(
            center_value=center_values[:, 0, :],
            wall_coord=y_faces[0],
            interior_coord=y_centers[0],
            tangential_1=x_centers[:, None],
            tangential_2=z_centers[None, :],
            axis=1,
            side="lower",
        )
        upper = _neumann_ghost_value(
            center_value=center_values[:, -1, :],
            wall_coord=y_faces[-1],
            interior_coord=y_centers[-1],
            tangential_1=x_centers[:, None],
            tangential_2=z_centers[None, :],
            axis=1,
            side="upper",
        )
        y_minus_values = y_minus_values.at[:, 0, :].set(lower)
        y_plus_values = y_plus_values.at[:, -1, :].set(upper)
    if not periodic_axes[2]:
        lower = _neumann_ghost_value(
            center_value=center_values[:, :, 0],
            wall_coord=z_faces[0],
            interior_coord=z_centers[0],
            tangential_1=x_centers[:, None],
            tangential_2=y_centers[None, :],
            axis=2,
            side="lower",
        )
        upper = _neumann_ghost_value(
            center_value=center_values[:, :, -1],
            wall_coord=z_faces[-1],
            interior_coord=z_centers[-1],
            tangential_1=x_centers[:, None],
            tangential_2=y_centers[None, :],
            axis=2,
            side="upper",
        )
        z_minus_values = z_minus_values.at[:, :, 0].set(lower)
        z_plus_values = z_plus_values.at[:, :, -1].set(upper)

    x_dx_min = jnp.broadcast_to((x_centers - x_minus_coords)[:, None, None], geometry.shape)
    x_dx_plus = jnp.broadcast_to((x_plus_coords - x_centers)[:, None, None], geometry.shape)
    y_dx_min = jnp.broadcast_to((y_centers - y_minus_coords)[None, :, None], geometry.shape)
    y_dx_plus = jnp.broadcast_to((y_plus_coords - y_centers)[None, :, None], geometry.shape)
    z_dx_min = jnp.broadcast_to((z_centers - z_minus_coords)[None, None, :], geometry.shape)
    z_dx_plus = jnp.broadcast_to((z_plus_coords - z_centers)[None, None, :], geometry.shape)

    return LocalStencil3D(
        x=LocalStencil1D(
            center=center_values,
            minus=x_minus_values,
            plus=x_plus_values,
            dx_min=x_dx_min,
            dx_plus=x_dx_plus,
        ),
        y=LocalStencil1D(
            center=center_values,
            minus=y_minus_values,
            plus=y_plus_values,
            dx_min=y_dx_min,
            dx_plus=y_dx_plus,
        ),
        z=LocalStencil1D(
            center=center_values,
            minus=z_minus_values,
            plus=z_plus_values,
            dx_min=z_dx_min,
            dx_plus=z_dx_plus,
        ),
    )


def _build_field_line_stencil_for_fci_mms(
    field_fn,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> LocalStencil1D:
    """Build the 1D stencil used by the FCI parallel-gradient MMS test."""

    periodic_axes = tuple(bool(value) for value in periodic_axes)
    if not periodic_axes[2]:
        raise ValueError("This MMS helper currently expects a periodic field-line axis")

    rho = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
    theta = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
    phi = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)

    field = _sample_scalar_field_on_grid(field_fn, rho, theta, phi)

    maps = geometry.maps
    up = _sample_scalar_field_on_grid(
        field_fn,
        maps.forward_endpoint_x,
        maps.forward_endpoint_y,
        maps.forward_endpoint_z,
    )
    down = _sample_scalar_field_on_grid(
        field_fn,
        maps.backward_endpoint_x,
        maps.backward_endpoint_y,
        maps.backward_endpoint_z,
    )

    dx_min = jnp.asarray(maps.backward_length, dtype=jnp.float64)
    dx_plus = jnp.asarray(maps.forward_length, dtype=jnp.float64)

    if field.shape != dx_min.shape or field.shape != dx_plus.shape:
        raise ValueError(
            f"field-line stencil shape mismatch: field={field.shape}, "
            f"dx_min={dx_min.shape}, dx_plus={dx_plus.shape}"
        )

    return LocalStencil1D(
        center=field,
        minus=down,
        plus=up,
        dx_min=dx_min,
        dx_plus=dx_plus,
    )


def mms_test_grad_parallel_direct_dirichlet(
    geometry: FciGeometry3D,
    *,
    print_diagnostics: bool = True,
    exclude_radial_boundary_cells: bool = True,
) -> tuple[float, float, float, float]:
    """MMS test for grad_parallel_op_direct with Dirichlet radial BC."""

    _field, expected = _build_mms_grad_parallel_field_and_expected(geometry)

    stencil = _build_local_stencil_dirichlet_boundaries(
        _mms_parallel_field_on_points,
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = grad_parallel_op_direct(
        stencil,
        geometry,
    )

    if actual.shape != geometry.shape:
        raise ValueError(
            f"grad_parallel_op_direct returned shape {actual.shape}, "
            f"expected {geometry.shape}"
        )

    error = actual - expected
    abs_error = jnp.abs(error)

    rms_error = jnp.sqrt(jnp.mean(error**2))
    max_error = jnp.max(abs_error)

    if exclude_radial_boundary_cells and geometry.shape[0] > 2:
        interior_error = error[1:-1, :, :]
        interior_abs_error = abs_error[1:-1, :, :]
        interior_offset = 1
    else:
        interior_error = error
        interior_abs_error = abs_error
        interior_offset = 0

    interior_rms_error = jnp.sqrt(jnp.mean(interior_error**2))
    interior_max_error = jnp.max(interior_abs_error)

    max_error_index = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(abs_error),
            abs_error.shape,
        )
    )
    interior_max_error_index = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(interior_abs_error),
            interior_abs_error.shape,
        )
    )
    if interior_offset:
        interior_max_error_index = (
            interior_max_error_index[0] + interior_offset,
            interior_max_error_index[1],
            interior_max_error_index[2],
        )

    if print_diagnostics:
        print("grad_parallel_direct_dirichlet full error mean:", float(jnp.mean(abs_error)))
        print("grad_parallel_direct_dirichlet full error median:", float(jnp.median(abs_error)))
        print("grad_parallel_direct_dirichlet full error min:", float(jnp.min(abs_error)))
        print("grad_parallel_direct_dirichlet full error max:", float(max_error))
        print("grad_parallel_direct_dirichlet max error index:", max_error_index)
        print("grad_parallel_direct_dirichlet full rms error:", float(rms_error))

        if exclude_radial_boundary_cells:
            print("grad_parallel_direct_dirichlet interior error mean:", float(jnp.mean(interior_abs_error)))
            print("grad_parallel_direct_dirichlet interior error median:", float(jnp.median(interior_abs_error)))
            print("grad_parallel_direct_dirichlet interior error min:", float(jnp.min(interior_abs_error)))
            print("grad_parallel_direct_dirichlet interior error max:", float(interior_max_error))
            print("grad_parallel_direct_dirichlet interior max error index:", interior_max_error_index)
            print("grad_parallel_direct_dirichlet interior rms error:", float(interior_rms_error))
            print("grad_parallel_direct_dirichlet interior region: excludes radial boundary cells")
        else:
            print("grad_parallel_direct_dirichlet norm region: full active domain")

    return (
        float(rms_error),
        float(max_error),
        float(interior_rms_error),
        float(interior_max_error),
    )


def mms_test_grad_parallel_fci_dirichlet(
    geometry: FciGeometry3D,
    *,
    print_diagnostics: bool = True,
) -> tuple[float, float]:
    """MMS test for grad_parallel_op_fci using the traced field-line stencil."""

    _, expected = _build_mms_grad_parallel_field_and_expected(geometry)

    stencil = _build_field_line_stencil_for_fci_mms(
        _mms_parallel_field_on_points,
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = grad_parallel_op_fci(
        stencil,
        geometry,
    )

    if actual.shape != geometry.shape:
        raise ValueError(
            f"grad_parallel_op_fci returned shape {actual.shape}, "
            f"expected {geometry.shape}"
        )

    error = actual - expected
    abs_error = jnp.abs(error)
    rms_error = jnp.sqrt(jnp.mean(error**2))
    max_error_index = tuple(
        int(value)
        for value in jnp.unravel_index(
            jnp.argmax(abs_error),
            abs_error.shape,
        )
    )

    if print_diagnostics:
        print("grad_parallel_fci_dirichlet error mean:", float(jnp.mean(abs_error)))
        print("grad_parallel_fci_dirichlet error median:", float(jnp.median(abs_error)))
        print("grad_parallel_fci_dirichlet error min:", float(jnp.min(abs_error)))
        print("grad_parallel_fci_dirichlet error max:", float(jnp.max(abs_error)))
        print("grad_parallel_fci_dirichlet max error index:", max_error_index)
        print("grad_parallel_fci_dirichlet rms error:", float(rms_error))

    return float(rms_error), float(jnp.max(abs_error))

def mms_test_grad_perp_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 error for the manufactured perpendicular-gradient test field."""

    def field_fn(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        amplitude = 1.0 + jnp.cos(jnp.pi * x / float(A))
        return amplitude * jnp.cos(float(M) * y) * jnp.sin(float(N) * z)

    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = field_fn(rho, theta, phi)

    dfd_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    dfd_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    dfd_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + float(ALPHA) * rho + rho * cos_theta
    Q = 1.0 + float(ALPHA) * cos_theta
    D2 = (float(IOTA) ** 2) * rho**2 + R**2

    expected = jnp.stack(
        (
            (1.0 / Q**2) * dfd_rho + (float(ALPHA) * sin_theta / (rho * Q**2)) * dfd_theta,
            (float(ALPHA) * sin_theta / (rho * Q**2)) * dfd_rho
            + ((1.0 + 2.0 * float(ALPHA) * cos_theta + float(ALPHA) ** 2) / (rho**2 * Q**2) - (float(IOTA) ** 2) / D2) * dfd_theta
            - (float(IOTA) / D2) * dfd_phi,
            -(float(IOTA) / D2) * dfd_theta + (1.0 / R**2 - 1.0 / D2) * dfd_phi,
        ),
        axis=-1,
    )

    stencil = _build_local_stencil_dirichlet_boundaries(
        field_fn,
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = grad_perp_op(stencil, geometry)
    error = jnp.linalg.norm(actual - expected, axis=-1)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("grad_perp error mean:", float(jnp.mean(error)))
    print("grad_perp error median:", float(jnp.median(error)))
    print("grad_perp error min:", float(jnp.min(error)))
    print("grad_perp error max:", float(jnp.max(error)))
    print("grad_perp max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.sum((actual - expected) ** 2, axis=-1)))), float(jnp.max(error))


def mms_test_poisson_bracket_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 error for the manufactured Poisson-bracket test field."""

    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))

    f_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    g_m = float(M) + 1.0
    g_n = float(N) + 1.0
    g_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(g_m * theta) * jnp.sin(g_n * phi)
    g_theta = -amplitude * g_m * jnp.sin(g_m * theta) * jnp.sin(g_n * phi)
    g_phi = amplitude * g_n * jnp.cos(g_m * theta) * jnp.cos(g_n * phi)

    R = float(R0) + float(ALPHA) * rho + rho * jnp.cos(theta)
    Q = 1.0 + float(ALPHA) * jnp.cos(theta)
    J = rho * R * Q
    D = jnp.sqrt((float(IOTA) ** 2) * rho**2 + R**2)

    expected = (
        1.0
        / (J * D)
        * (
            -float(ALPHA) * float(IOTA) * rho * jnp.sin(theta) * (f_theta * g_phi - f_phi * g_theta)
            + float(IOTA) * rho**2 * (f_phi * g_rho - f_rho * g_phi)
            + R**2 * (f_rho * g_theta - f_theta * g_rho)
        )
    )

    f_stencil = _build_local_stencil_neumann_boundaries(
        lambda x, y, z: (1.0 + jnp.cos(jnp.pi * x / float(A))) * jnp.cos(float(M) * y) * jnp.sin(float(N) * z),
        geometry,
        periodic_axes=(False, True, True),
    )
    g_stencil = _build_local_stencil_dirichlet_boundaries(
        lambda x, y, z: (1.0 + jnp.cos(jnp.pi * x / float(A))) * jnp.cos((float(M) + 1.0) * y) * jnp.sin((float(N) + 1.0) * z),
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = poisson_bracket_op(f_stencil, g_stencil, geometry)
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("poisson_bracket error mean:", float(jnp.mean(error)))
    print("poisson_bracket error median:", float(jnp.median(error)))
    print("poisson_bracket error min:", float(jnp.min(error)))
    print("poisson_bracket error max:", float(jnp.max(error)))
    print("poisson_bracket max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected)))), float(jnp.max(error))


def mms_test_curvature_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 error for the manufactured curvature test field."""

    def field_fn(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        amplitude = 1.0 + jnp.cos(jnp.pi * x / float(A))
        return amplitude * jnp.cos(float(M) * y) * jnp.sin(float(N) * z)

    def derivative_fn(axis: int, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        amplitude = 1.0 + jnp.cos(jnp.pi * x / float(A))
        if axis == 0:
            return -(jnp.pi / float(A)) * jnp.sin(jnp.pi * x / float(A)) * jnp.cos(float(M) * y) * jnp.sin(float(N) * z)
        if axis == 1:
            return -amplitude * float(M) * jnp.sin(float(M) * y) * jnp.sin(float(N) * z)
        if axis == 2:
            return amplitude * float(N) * jnp.cos(float(M) * y) * jnp.cos(float(N) * z)
        raise ValueError(f"invalid axis {axis}")

    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    field = field_fn(rho, theta, phi)

    f_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -amplitude * float(M) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = amplitude * float(N) * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + rho * (float(ALPHA) + cos_theta)
    Q = 1.0 + float(ALPHA) * cos_theta
    J = rho * R * Q
    D2 = (float(IOTA) ** 2) * rho**2 + R**2
    D = jnp.sqrt(D2)
    P = float(ALPHA) + cos_theta
    E = rho * Q + float(ALPHA) * R
    A_term = (float(IOTA) ** 2) * rho + R * P

    K_rho = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * rho * R * sin_theta / D
            + 2.0 * rho * R**3 * sin_theta / D**3
            - rho * R**2 * sin_theta * E / (D * J)
        )
    )
    K_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * P / D
            - 2.0 * R**2 * A_term / D**3
            + R**2 * Q * (R + rho * P) / (D * J)
        )
    )
    K_phi = (
        float(IOTA)
        / (2.0 * J)
        * (
            rho * (2.0 + float(ALPHA) * cos_theta) / D
            - 2.0 * rho**2 * A_term / D**3
            + 2.0 * float(ALPHA) * rho**2 * R * sin_theta**2 / D**3
            + (rho**2 * Q * (R + rho * P) - float(ALPHA) * rho**2 * sin_theta**2 * E) / (D * J)
        )
    )

    expected = K_rho * f_rho + K_theta * f_theta + K_phi * f_phi

    stencil = _build_local_stencil_neumann_boundaries(
        field_fn,
        geometry,
        periodic_axes=(False, True, True),
        derivative_fn=derivative_fn,
    )

    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    actual = curvature_op(
        stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("curvature error mean:", float(jnp.mean(error)))
    print("curvature error median:", float(jnp.median(error)))
    print("curvature error min:", float(jnp.min(error)))
    print("curvature error max:", float(jnp.max(error)))
    print("curvature max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected)))), float(jnp.max(error))


def _perp_laplacian_mms_case(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the manufactured field, exact Laplacian, and exact radial fluxes."""

    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    phi = logical_grid[..., 2]
    Fr = 1.0 + jnp.cos(jnp.pi * rho / float(A))
    Fr_rho = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A))
    Fr_rhorho = -(jnp.pi**2 / float(A) ** 2) * jnp.cos(jnp.pi * rho / float(A))

    field = Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    f_rho = Fr_rho * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_theta = -float(M) * Fr * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_phi = float(N) * Fr * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    f_rhorho = Fr_rhorho * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_rhotheta = -float(M) * Fr_rho * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    f_rhophi = float(N) * Fr_rho * jnp.cos(float(M) * theta) * jnp.cos(float(N) * phi)
    f_thetatheta = -(float(M) ** 2) * Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
    f_thetaphi = -float(M) * float(N) * Fr * jnp.sin(float(M) * theta) * jnp.cos(float(N) * phi)
    f_phiphi = -(float(N) ** 2) * Fr * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = float(R0) + rho * (float(ALPHA) + cos_theta)
    Q = 1.0 + float(ALPHA) * cos_theta
    J = rho * R * Q
    S = (float(IOTA) ** 2) * rho**2 + R**2
    E = rho * Q + float(ALPHA) * R
    Mcoef = 1.0 + 2.0 * float(ALPHA) * cos_theta + float(ALPHA) ** 2

    P_rhorho = 1.0 / Q**2
    P_rhotheta = float(ALPHA) * sin_theta / (rho * Q**2)
    P_thetatheta = Mcoef / (rho**2 * Q**2) - (float(IOTA) ** 2) / S
    P_thetaphi = -float(IOTA) / S
    P_phiphi = 1.0 / R**2 - 1.0 / S

    C_rho = (1.0 / J) * (
        R + rho * cos_theta + (float(ALPHA) ** 2) * R * sin_theta**2 / Q**2
    )
    C_theta = (sin_theta / J) * (
        -1.0
        + float(ALPHA) * R * (float(ALPHA) ** 2 - 1.0) / (rho * Q**2)
        + rho * (float(IOTA) ** 2) * E / S
        - 2.0 * rho**2 * (float(IOTA) ** 2) * R**2 * Q / S**2
    )
    C_phi = (rho * float(IOTA) * sin_theta / J) * (
        E / S - 2.0 * rho * R**2 * Q / S**2
    )

    expected = (
        P_rhorho * f_rhorho
        + 2.0 * P_rhotheta * f_rhotheta
        + P_thetatheta * f_thetatheta
        + 2.0 * P_thetaphi * f_thetaphi
        + P_phiphi * f_phiphi
        + C_rho * f_rho
        + C_theta * f_theta
        + C_phi * f_phi
    )

    inner_flux = (
        -(rho * R / Q) * (jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
        - (float(ALPHA) * R * sin_theta / Q) * float(M) * (1.0 + jnp.cos(jnp.pi * rho / float(A))) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    )[0, :, :]
    outer_flux = (
        -(rho * R / Q) * (jnp.pi / float(A)) * jnp.sin(jnp.pi * rho / float(A)) * jnp.cos(float(M) * theta) * jnp.sin(float(N) * phi)
        - (float(ALPHA) * R * sin_theta / Q) * float(M) * (1.0 + jnp.cos(jnp.pi * rho / float(A))) * jnp.sin(float(M) * theta) * jnp.sin(float(N) * phi)
    )[-1, :, :]

    return field, expected, inner_flux, outer_flux


def mms_test_perp_laplacian_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the interior L2 error for the manufactured perpendicular-Laplacian test field."""

    _field, expected, _, _ = _perp_laplacian_mms_case(geometry)
    stencil = _build_local_stencil_dirichlet_boundaries(
        lambda x, y, z: (1.0 + jnp.cos(jnp.pi * x / float(A))) * jnp.cos(float(M) * y) * jnp.sin(float(N) * z),
        geometry,
        periodic_axes=(False, True, True),
    )
    conservative_stencil = ConservativeStencil3D(
        x=stencil.x,
        y=stencil.y,
        z=stencil.z,
    )

    y_centers = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)
    lower_neumann = (jnp.pi / float(A)) * jnp.sin(jnp.pi * geometry.grid.x.faces[0] / float(A))
    upper_neumann = -(jnp.pi / float(A)) * jnp.sin(jnp.pi * geometry.grid.x.faces[-1] / float(A))
    neumann_pattern = jnp.cos(float(M) * y_centers[:, None]) * jnp.sin(float(N) * z_centers[None, :])

    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        value_x=face_bc.value_x.at[0].set(lower_neumann * neumann_pattern).at[-1].set(upper_neumann * neumann_pattern),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    actual = perp_laplacian_conservative_op(
        conservative_stencil,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, True, True),
    )
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("perp_laplacian error mean:", float(jnp.mean(error)))
    print("perp_laplacian error median:", float(jnp.median(error)))
    print("perp_laplacian error min:", float(jnp.min(error)))
    print("perp_laplacian error max:", float(jnp.max(error)))
    print("perp_laplacian max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(error)))), float(jnp.max(error))


def mms_test_parallel_laplacian_direct_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 and max error for the chained direct parallel Laplacian MMS."""

    field, expected, _ = _build_mms_parallel_laplacian_field_and_expected(geometry)

    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=face_bc.value_x.at[0].set(field[0, :, :]).at[-1].set(field[-1, :, :]),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )

    actual = parallel_laplacian_direct_op(
        field,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, True, True),
    )

    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("parallel_laplacian_direct error mean:", float(jnp.mean(error)))
    print("parallel_laplacian_direct error median:", float(jnp.median(error)))
    print("parallel_laplacian_direct error min:", float(jnp.min(error)))
    print("parallel_laplacian_direct error max:", float(jnp.max(error)))
    print("parallel_laplacian_direct max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(error)))), float(jnp.max(error))


def mms_test_parallel_laplacian_conservative_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 and max error for the conservative parallel Laplacian MMS."""

    _field, _, expected = _build_mms_parallel_laplacian_field_and_expected(geometry)
    stencil = _build_local_stencil_dirichlet_boundaries(
        _mms_parallel_field_on_points,
        geometry,
        periodic_axes=(False, True, True),
    )
    conservative_stencil = ConservativeStencil3D(
        x=stencil.x,
        y=stencil.y,
        z=stencil.z,
    )

    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=face_bc.value_x.at[0].set(0.0).at[-1].set(0.0),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )

    actual = parallel_laplacian_conservative_op(
        conservative_stencil,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, True, True),
    )

    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("parallel_laplacian_conservative error mean:", float(jnp.mean(error)))
    print("parallel_laplacian_conservative error median:", float(jnp.median(error)))
    print("parallel_laplacian_conservative error min:", float(jnp.min(error)))
    print("parallel_laplacian_conservative error max:", float(jnp.max(error)))
    print("parallel_laplacian_conservative max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(error)))), float(jnp.max(error))


def mms_test_perp_laplacian_local_op(geometry: FciGeometry3D) -> tuple[float, float]:
    """Return the L2 and max error for the local perpendicular-Laplacian MMS."""

    _field, expected, _, _ = _perp_laplacian_mms_case(geometry)
    stencil = _build_local_stencil_neumann_boundaries(
        lambda x, y, z: (1.0 + jnp.cos(jnp.pi * x / float(A))) * jnp.cos(float(M) * y) * jnp.sin(float(N) * z),
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = perp_laplacian_local_op(
        stencil,
        geometry,
        periodic_axes=(False, True, True),
    )
    error = jnp.abs(actual - expected)
    max_error_index = tuple(int(value) for value in jnp.unravel_index(jnp.argmax(error), error.shape))
    print("perp_laplacian_local error mean:", float(jnp.mean(error)))
    print("perp_laplacian_local error median:", float(jnp.median(error)))
    print("perp_laplacian_local error min:", float(jnp.min(error)))
    print("perp_laplacian_local error max:", float(jnp.max(error)))
    print("perp_laplacian_local max error index:", max_error_index)
    return float(jnp.sqrt(jnp.mean(jnp.square(error)))), float(jnp.max(error))


def _weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    metric = geometry.cell_metric
    spacing = geometry.spacing
    weights = (
        jnp.asarray(metric.J, dtype=jnp.float64)
        * jnp.asarray(spacing.dx, dtype=jnp.float64)
        * jnp.asarray(spacing.dy, dtype=jnp.float64)
        * jnp.asarray(spacing.dz, dtype=jnp.float64)
    )
    return jnp.sum(weights * values) / jnp.sum(weights)


def _remove_weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    return jnp.asarray(field, dtype=jnp.float64) - _weighted_mean(field, geometry)


def build_identity_fci_geometry(shape: tuple[int, int, int], *, dz: float = 1.0) -> FciGeometry3D:
    nx, ny, nz = shape
    x_grid = Grid1D.from_centers(jnp.arange(nx, dtype=jnp.float64))
    y_grid = Grid1D.from_centers(jnp.arange(ny, dtype=jnp.float64))
    z_grid = Grid1D.from_centers(jnp.arange(nz, dtype=jnp.float64) * float(dz))
    grid = CellCenteredGrid3D(x=x_grid, y=y_grid, z=z_grid)
    logical_grid = logical_grid_from_axis_vectors(*grid.logical_axis_vectors)

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
        forward_length=ones * float(dz),
        backward_length=ones * float(dz),
        forward_boundary=jnp.zeros(shape, dtype=bool),
        backward_boundary=jnp.zeros(shape, dtype=bool),
    )
    spacing = Spacing3D(dx=ones, dy=ones, dz=ones * float(dz))

    def _metric(field_shape: tuple[int, int, int]) -> MetricGeometry:
        field_ones = jnp.ones(field_shape, dtype=jnp.float64)
        field_zeros = jnp.zeros(field_shape, dtype=jnp.float64)
        return MetricGeometry(
            J=field_ones,
            g11=field_ones,
            g22=field_ones,
            g33=field_ones,
            g12=field_zeros,
            g13=field_zeros,
            g23=field_zeros,
            g_11=field_ones,
            g_22=field_ones,
            g_33=field_ones,
            g_12=field_zeros,
            g_13=field_zeros,
            g_23=field_zeros,
        )

    def _bfield(field_shape: tuple[int, int, int]) -> BFieldGeometry:
        b_contra = jnp.zeros(field_shape + (3,), dtype=jnp.float64).at[..., 2].set(1.0)
        bmag = jnp.ones(field_shape, dtype=jnp.float64)
        return BFieldGeometry(B_contra=b_contra, Bmag=bmag)

    cell_metric = _metric(shape)
    face_metric = FaceMetricGeometry(
        x=_metric((nx + 1, ny, nz)),
        y=_metric((nx, ny + 1, nz)),
        z=_metric((nx, ny, nz + 1)),
    )
    cell_bfield = _bfield(shape)
    face_bfield = FaceBFieldGeometry(
        x=_bfield((nx + 1, ny, nz)),
        y=_bfield((nx, ny + 1, nz)),
        z=_bfield((nx, ny, nz + 1)),
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


def test_regularized_perp_laplacian_inverse_controls_constant_neumann_mode() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    zero_flux = jnp.zeros(geometry.shape[1:], dtype=jnp.float64)
    face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        value_x=face_bc.value_x.at[0].set(zero_flux).at[-1].set(zero_flux),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    epsilon = 2.0e-3
    rhs_value = 3.0e-2
    solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        tol=1.0e-10,
        maxiter=100,
        restart=20,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        regularization_epsilon=epsilon,
    )

    phi = solver(jnp.full(geometry.shape, rhs_value, dtype=jnp.float64), face_bc=face_bc)

    assert jnp.allclose(phi, rhs_value / epsilon, rtol=1.0e-8, atol=1.0e-8)
    try:
        PerpLaplacianInverseSolver(
            geometry,
            stencil_builder,
            regularization_epsilon=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative regularization_epsilon should raise ValueError")


def test_perp_laplacian_inverse_dirichlet_lift_constant_solution() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    wall_value = 1.75
    wall_values = jnp.full(geometry.shape[1:], wall_value, dtype=jnp.float64)
    face_bc = face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=face_bc.value_x.at[0].set(wall_values).at[-1].set(wall_values),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        tol=1.0e-10,
        maxiter=20,
        restart=10,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        project_mean_zero=False,
        check_residual=True,
    )

    rhs_physical = jnp.zeros(geometry.shape, dtype=jnp.float64)
    phi_lift = jnp.full(geometry.shape, wall_value, dtype=jnp.float64)
    phi, diagnostics = solver(
        rhs_physical,
        face_bc=face_bc,
        cut_wall_bc=CutWallBC3D.empty(),
        phi_lift=phi_lift,
        return_diagnostics=True,
    )

    assert diagnostics["lifted"] is True
    assert diagnostics["num_steps"] <= 5
    assert diagnostics["correction_residual_l2"] < 1.0e-8
    assert diagnostics["physical_residual_l2"] < 1.0e-8
    np.testing.assert_allclose(np.asarray(phi), np.asarray(phi_lift), rtol=1.0e-10, atol=1.0e-10)


def _constant_dirichlet_x_face_bc(geometry: FciGeometry3D, value: float) -> BoundaryFaceBC3D:
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    wall_values = jnp.full(geometry.shape[1:], float(value), dtype=jnp.float64)
    return face_bc.replace(
        kind_x=face_bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=face_bc.value_x.at[0].set(wall_values).at[-1].set(wall_values),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )


def _smooth_zero_dirichlet_field(geometry: FciGeometry3D) -> jnp.ndarray:
    logical_grid = logical_grid_from_axis_vectors(*geometry.logical_axis_vectors)
    rho = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    xi = (rho - rho[0, 0, 0]) / jnp.maximum(rho[-1, 0, 0] - rho[0, 0, 0], 1.0e-30)
    return jnp.sin(jnp.pi * xi) * (1.0 + 0.2 * jnp.cos(theta) * jnp.sin(zeta))


def test_perp_laplacian_inverse_rejects_mismatched_nonzero_dirichlet_mg_hierarchy() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = _constant_dirichlet_x_face_bc(geometry, 1.75)
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    bad_hierarchy = build_perp_laplacian_mg_hierarchy(
        geometry,
        stencil_builder,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        max_levels=2,
    )
    solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        mg_hierarchy=bad_hierarchy,
    )

    try:
        solver(jnp.zeros(geometry.shape, dtype=jnp.float64), face_bc=face_bc)
    except ValueError as error:
        assert "mg_hierarchy face_bc must match" in str(error)
    else:
        raise AssertionError("nonzero physical Dirichlet MG hierarchy should be rejected")


def test_perp_laplacian_inverse_accepts_solver_mg_hierarchy_for_homogeneous_dirichlet() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = _constant_dirichlet_x_face_bc(geometry, 0.0)
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    hierarchy = build_perp_laplacian_solver_mg_hierarchy(
        geometry,
        stencil_builder,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        max_levels=2,
    )
    solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        mg_hierarchy=hierarchy,
        tol=1.0e-8,
        maxiter=50,
        restart=20,
    )
    exact = _smooth_zero_dirichlet_field(geometry)
    rhs = solver._apply_A(exact, face_bc, CutWallBC3D.empty(), False)

    phi, diagnostics = solver(rhs, face_bc=face_bc, return_diagnostics=True)

    assert diagnostics["final_residual_rel_l2"] < 1.0e-6
    assert bool(jnp.all(jnp.isfinite(phi)))


def test_perp_laplacian_inverse_dirichlet_lift_with_solver_mg_hierarchy() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    wall_value = 1.75
    face_bc = _constant_dirichlet_x_face_bc(geometry, wall_value)
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    hierarchy = build_perp_laplacian_solver_mg_hierarchy(
        geometry,
        stencil_builder,
        face_bc=face_bc,
        lifted=True,
        regular_face_geometry=regular_face_geometry,
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        max_levels=2,
    )
    solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        mg_hierarchy=hierarchy,
        project_mean_zero=False,
        check_residual=True,
    )

    rhs_physical = jnp.zeros(geometry.shape, dtype=jnp.float64)
    phi_lift = jnp.full(geometry.shape, wall_value, dtype=jnp.float64)
    phi, diagnostics = solver(
        rhs_physical,
        face_bc=face_bc,
        cut_wall_bc=CutWallBC3D.empty(),
        phi_lift=phi_lift,
        return_diagnostics=True,
    )

    assert diagnostics["lifted"] is True
    assert diagnostics["correction_residual_l2"] < 1.0e-8
    np.testing.assert_allclose(np.asarray(phi), np.asarray(phi_lift), rtol=1.0e-10, atol=1.0e-10)


def test_perp_laplacian_inverse_solver_mg_preconditioner_not_worse_on_smooth_dirichlet() -> None:
    pytest.importorskip("lineax")
    geometry = build_test_fci_geometry((5, 4, 3), construct_fci_maps=False)
    regular_face_geometry = RegularFaceGeometry3D.unit(geometry)
    face_bc = _constant_dirichlet_x_face_bc(geometry, 0.0)
    stencil_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    hierarchy = build_perp_laplacian_solver_mg_hierarchy(
        geometry,
        stencil_builder,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        max_levels=2,
    )
    unpreconditioned_solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        tol=1.0e-8,
        maxiter=80,
        restart=20,
    )
    preconditioned_solver = PerpLaplacianInverseSolver(
        geometry,
        stencil_builder,
        regular_face_geometry=regular_face_geometry,
        periodic_axes=(False, True, True),
        mg_hierarchy=hierarchy,
        tol=1.0e-8,
        maxiter=80,
        restart=20,
    )
    exact = _smooth_zero_dirichlet_field(geometry)
    rhs = unpreconditioned_solver._apply_A(exact, face_bc, CutWallBC3D.empty(), False)

    _, unpreconditioned = unpreconditioned_solver(rhs, face_bc=face_bc, return_diagnostics=True)
    _, preconditioned = preconditioned_solver(rhs, face_bc=face_bc, return_diagnostics=True)

    assert preconditioned["final_residual_rel_l2"] < 1.0e-6
    assert preconditioned["num_steps"] <= unpreconditioned["num_steps"] + 2


def test_poisson_bracket_on_identity_metric_matches_cross_product_form() -> None:
    geometry = build_identity_fci_geometry((4, 5, 6))
    f_stencil = _build_local_stencil_dirichlet_boundaries(
        lambda x, y, z: x,
        geometry,
        periodic_axes=(False, True, True),
    )
    g_stencil = _build_local_stencil_dirichlet_boundaries(
        lambda x, y, z: y,
        geometry,
        periodic_axes=(False, True, True),
    )

    actual = poisson_bracket_op(f_stencil, g_stencil, geometry)

    assert jnp.allclose(actual, 1.0)


def test_curvature_op_vanishes_for_constant_field_on_identity_metric() -> None:
    geometry = build_identity_fci_geometry((4, 5, 6))
    stencil = _build_local_stencil_dirichlet_boundaries(
        lambda x, y, z: jnp.asarray(1.0, dtype=jnp.float64),
        geometry,
        periodic_axes=(False, True, True),
    )

    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    actual = curvature_op(
        stencil,
        geometry,
        curvature_coefficients=curvature_coefficients,
    )

    assert jnp.allclose(actual, 0.0)





def test_parallel_laplacian_conservative_op_returns_zero_for_constant_field() -> None:
    geometry = build_identity_fci_geometry((5, 4, 6))
    field = jnp.full(geometry.shape, 3.25, dtype=jnp.float64)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, False, False),
    )

    actual = parallel_laplacian_conservative_op(
        stencil,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, False, False),
    )

    assert jnp.allclose(actual, 0.0)


def test_parallel_laplacian_conservative_op_matches_1d_quadratic_second_derivative() -> None:
    geometry = build_identity_fci_geometry((5, 4, 6))
    z = jnp.arange(geometry.shape[2], dtype=jnp.float64)[None, None, :]
    field = jnp.broadcast_to(z * z, geometry.shape)
    face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))
    stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, False, False),
    )

    actual = parallel_laplacian_conservative_op(
        stencil,
        geometry,
        face_bc=face_bc,
        periodic_axes=(False, False, False),
    )

    assert jnp.allclose(actual, 2.0)


def test_parallel_laplacian_direct_op_returns_zero_for_constant_field() -> None:
    geometry = build_identity_fci_geometry((5, 4, 6))
    field = jnp.full(geometry.shape, 3.25, dtype=jnp.float64)

    actual = parallel_laplacian_direct_op(
        field,
        geometry,
        periodic_axes=(False, False, False),
    )

    assert jnp.allclose(actual, 0.0)


def test_parallel_laplacian_direct_op_matches_1d_quadratic_second_derivative() -> None:
    geometry = build_identity_fci_geometry((5, 4, 6))
    z = jnp.arange(geometry.shape[2], dtype=jnp.float64)[None, None, :]
    field = jnp.broadcast_to(z * z, geometry.shape)

    actual = parallel_laplacian_direct_op(
        field,
        geometry,
        periodic_axes=(False, False, False),
    )

    assert jnp.allclose(actual, 2.0)


if __name__ == "__main__":
    import math

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    operator_specs = [
        ("grad_parallel_direct", mms_test_grad_parallel_direct_dirichlet),
        ("grad_parallel_fci", mms_test_grad_parallel_fci_dirichlet),
        ('grad_perp ',mms_test_grad_perp_op),
        ('curvature ',mms_test_curvature_op),
        ('poisson bracket', mms_test_poisson_bracket_op),
        ('perp_laplacian_local', mms_test_perp_laplacian_local_op),
        ('perp_laplacian_conservative', mms_test_perp_laplacian_op),
        ('parallel_laplacian_local', mms_test_parallel_laplacian_direct_op),
        ('parallel_laplacian_conservative', mms_test_parallel_laplacian_conservative_op),        
    ]

    resolutions = [20,40,80]
    log_resolutions = np.log(np.asarray(resolutions, dtype=np.float64))
    error_results: dict[str, list[float]] = {name: [] for name, _ in operator_specs}
    max_error_results: dict[str, list[float]] = {name: [] for name, _ in operator_specs}

    for resolution in resolutions:
        geometry = build_test_fci_geometry((resolution, resolution, resolution), construct_fci_maps=True)
        for name, operator_test in operator_specs:
            result = operator_test(geometry)
            rms_error, max_error = result[:2]
            error_results[name].append(float(rms_error))
            max_error_results[name].append(float(max_error))

    ncols = 2
    nrows = math.ceil(len(operator_specs) / ncols)

    fig = None
    axes = None
    if plt is not None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)

    for index, (name, _) in enumerate(operator_specs):
        errors = np.asarray(error_results[name], dtype=np.float64)
        max_errors = np.asarray(max_error_results[name], dtype=np.float64)
        log_errors = np.log(errors)
        log_max_errors = np.log(max_errors)

        slope, intercept = np.polyfit(log_resolutions, log_errors, 1)
        convergence_order = -slope
        fit_log_errors = intercept + slope * log_resolutions
        max_slope, max_intercept = np.polyfit(log_resolutions, log_max_errors, 1)
        max_convergence_order = -max_slope
        fit_log_max_errors = max_intercept + max_slope * log_resolutions

        print(f"{name} convergence order: {convergence_order:.6f}")
        print(f"{name} max error convergence order: {max_convergence_order:.6f}")

        if axes is not None:
            ax = axes[index // ncols][index % ncols]
            ax.plot(log_resolutions, log_errors, "o-", label="rms data")
            ax.plot(log_resolutions, fit_log_errors, "--", label=f"rms fit order {convergence_order:.2f}")
            ax.plot(log_resolutions, log_max_errors, "s-", label="max data")
            ax.plot(log_resolutions, fit_log_max_errors, ":", label=f"max fit order {max_convergence_order:.2f}")
            ax.set_title(name)
            ax.set_xlabel("log(resolution)")
            ax.set_ylabel("log(error)")
            ax.grid(True, linestyle=":", linewidth=0.7)
            ax.legend()

    if axes is not None:
        for index in range(len(operator_specs), nrows * ncols):
            axes[index // ncols][index % ncols].axis("off")
        fig.tight_layout()
        fig.savefig("fci_operator_convergence.png", dpi=200)
        plt.close(fig)

    #test_invert_perp_laplacian_dirichlet_mms()
    #test_invert_perp_laplacian_dirichlet_enforces_boundary_values()
    #test_invert_perp_laplacian_neumann_mms()
    #test_invert_perp_laplacian_dirichlet_rows_are_identity_and_interior_uses_homogeneous_operator()
    #test_invert_perp_laplacian_neumann_solver_only()
    #test_invert_perp_laplacian_neumann_discrete_inverse()
    #test_invert_perp_laplacian_neumann_target_mean()
    #test_invert_perp_laplacian_neumann_boundary_source_is_rhs_only()
    #test_apply_A_is_linear_for_dirichlet_and_neumann_cases()
    #test_prepare_perp_laplacian_mg_builds_coarse_hierarchy()
    #test_restriction_and_prolongation_preserve_constants_and_smooth_modes()
    #test_mg_vcycle_reduces_residual_on_a_smooth_dirichlet_problem()
    #test_preconditioned_gmres_reduces_iteration_count()
    pass
