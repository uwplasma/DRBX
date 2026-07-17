"""Shifted-torus non-axisymmetric FCI geometry constructor.

This module provides :func:`build_shifted_torus_geometry`, a self-contained
constructor for a non-axisymmetric flux-coordinate-independent (FCI) metric on a
shifted-torus flux tube. The logical coordinates are ``(x, theta, zeta)`` with a
physical radial coordinate ``x`` and periodic poloidal/toroidal angles. The
poloidal angle is sheared by a radial shift ``Theta = theta + sigma * (x - x_mid)``
so that, for ``sigma != 0``, the metric acquires genuine off-diagonal (``g12`` /
``g_12``) cross terms and the geometry is non-axisymmetric.

The construction was verified as a manufactured-solution scaffold in
``tests/test_mms_shifted_torus_2_field.py`` (the two-field FCI MMS convergence
harness). It is promoted here so the reduced drift-reduced FCI model can be
exercised and differentiated on non-axisymmetric geometry from the package
itself.
"""

from __future__ import annotations

import jax.numpy as jnp

from .fci_geometry import (
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

__all__ = ["build_shifted_torus_geometry"]


def build_shifted_torus_geometry(
    shape: tuple[int, int, int],
    *,
    x_min: float = 0.15,
    x_max: float = 1.0,
    r0: float = 3.0,
    alpha_value: float = 0.25,
    iota: float = 1.1,
    c_phi: float = 3.0,
    sigma: float = 0.0,
    construct_fci_maps: bool = False,
    B_contravariant: jnp.ndarray | None = None,
) -> FciGeometry3D:
    """Build a shifted-torus non-axisymmetric FCI geometry.

    The logical coordinates are ``(x, theta, zeta)`` with periodic ``theta`` and
    ``zeta``. The helper follows the same `FciGeometry3D` construction pattern used
    in `test_fci_operators.py`, but uses the physical radial coordinate directly and
    a shifted poloidal angle ``Theta = theta + sigma * (x - x_mid)``. A nonzero
    ``sigma`` makes the metric non-axisymmetric (it introduces the off-diagonal
    ``g12`` / ``g_12`` terms below).

    Parameters mirror the module-level constants used by the two-field MMS
    convergence harness; the defaults reproduce that verified geometry.

    Args:
        shape: ``(nx, ny, nz)`` cell-centered grid resolution.
        x_min, x_max: physical radial coordinate bounds.
        r0: major-radius offset of the shifted torus.
        alpha_value: Shafranov-like shift amplitude.
        iota: rotational transform used to build the contravariant field.
        c_phi: toroidal field scaling constant.
        sigma: poloidal shear; ``sigma != 0`` yields a non-axisymmetric metric.
        construct_fci_maps: if ``True``, trace the parallel FCI maps from the
            contravariant field; otherwise install identity/placeholder maps.
        B_contravariant: optional override for the contravariant field
            components; when ``None`` the analytic field is used.

    Returns:
        A fully populated :class:`FciGeometry3D` of the requested ``shape``.
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
