"""Rotating-ellipse non-axisymmetric FCI geometry constructor.

This module provides :func:`build_rotating_ellipse_geometry`, a self-contained
constructor for a genuinely non-axisymmetric flux-coordinate-independent (FCI)
geometry: a torus whose elliptical cross-section *rotates* as it is followed
around toroidally. This is the classical rotating-ellipse (``l = 2``)
configuration and the canonical minimal non-axisymmetric field for exercising
FCI parallel operators, because the flux surfaces genuinely change orientation
with the toroidal angle (the metric depends on all three logical coordinates).

The logical coordinates are ``(x, theta, zeta)`` with a minor-radius label ``x``
and periodic poloidal / toroidal angles ``theta`` / ``zeta``. The physical
embedding into Cartesian space is

    p0 =  (1 + delta) * x * cos(theta)
    q0 =  (1 - delta) * x * sin(theta)
    lam = n_field_periods * zeta                 # ellipse orientation
    p   =  cos(lam) * p0 - sin(lam) * q0
    q   =  sin(lam) * p0 + cos(lam) * q0
    R   =  r0 + p
    X, Y, Z = R cos(zeta), R sin(zeta), q

so a surface ``x = const`` is an ellipse of elongation ``(1 + delta)/(1 - delta)``
whose major axis rotates ``n_field_periods`` times per toroidal turn. The metric
is obtained **by automatic differentiation** of this embedding
(``g_ij = d_i X . d_j X``) rather than by hand, which keeps the construction
exact and differentiable with respect to the shape parameters (elongation,
rotation) themselves.

The magnetic field is helical on the flux surfaces,
``B^x = 0``, ``B^theta = iota * c_phi / J``, ``B^zeta = c_phi / J``, so its field
lines wind with rotational transform ``iota`` and never leave a surface. Pass
``construct_fci_maps=True`` to trace the parallel FCI maps from this field.
"""

from __future__ import annotations

import jax
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
    _bmag_from_contravariant_components,
    build_fci_maps_from_b_contravariant,
    logical_grid_from_axis_vectors,
)

__all__ = ["build_rotating_ellipse_geometry", "rotating_ellipse_position"]


def rotating_ellipse_position(
    x: jax.Array,
    theta: jax.Array,
    zeta: jax.Array,
    *,
    r0: float = 3.0,
    elongation: float = 0.35,
    n_field_periods: int = 1,
) -> jax.Array:
    """Cartesian ``(X, Y, Z)`` position of the rotating-ellipse embedding.

    ``elongation`` is the ellipse deformation ``delta`` (the semi-axes are
    ``(1 +/- delta) * x``); ``n_field_periods`` is how many times the ellipse
    rotates per toroidal turn. Broadcasts over array-valued arguments and
    returns the Cartesian components stacked on the last axis. ``r0`` and
    ``elongation`` are kept traceable (no ``float`` cast), so this map is
    differentiable with respect to the shape parameters themselves.
    """

    lam = float(n_field_periods) * zeta
    cos_lam = jnp.cos(lam)
    sin_lam = jnp.sin(lam)
    p0 = (1.0 + elongation) * x * jnp.cos(theta)
    q0 = (1.0 - elongation) * x * jnp.sin(theta)
    p = cos_lam * p0 - sin_lam * q0
    q = sin_lam * p0 + cos_lam * q0
    major_radius = float(r0) + p
    return jnp.stack(
        (major_radius * jnp.cos(zeta), major_radius * jnp.sin(zeta), q),
        axis=-1,
    )


def build_rotating_ellipse_geometry(
    shape: tuple[int, int, int],
    *,
    r0: float = 3.0,
    x_min: float = 0.2,
    x_max: float = 1.0,
    elongation: float = 0.35,
    n_field_periods: int = 1,
    iota: float = 0.9,
    c_phi: float = 3.0,
    construct_fci_maps: bool = False,
    map_substeps: int = 8,
) -> FciGeometry3D:
    """Build a rotating-ellipse non-axisymmetric FCI geometry.

    Args:
        shape: ``(nx, ny, nz)`` cell-centered grid resolution.
        r0: major-radius offset of the torus.
        x_min, x_max: minor-radius (``x``) label bounds.
        elongation: ellipse deformation ``delta`` in ``[0, 1)``; ``0`` recovers
            a circular (still non-axisymmetric only if rotated) cross-section,
            larger values elongate it. The cross-section aspect ratio is
            ``(1 + delta) / (1 - delta)``.
        n_field_periods: ellipse rotations per toroidal turn (the field
            periodicity). Must be a positive integer for a ``zeta``-periodic
            geometry.
        iota: rotational transform of the helical field.
        c_phi: toroidal-field scaling constant.
        construct_fci_maps: if ``True``, trace the parallel FCI maps from the
            contravariant field; otherwise install identity/placeholder maps.
        map_substeps: field-line-tracer substeps per toroidal cell when
            ``construct_fci_maps`` is ``True``.

    Returns:
        A fully populated :class:`FciGeometry3D` of the requested ``shape``.
    """

    nx, ny, nz = shape
    x_faces = jnp.linspace(float(x_min), float(x_max), nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    zeta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D(centers=0.5 * (x_faces[:-1] + x_faces[1:]), faces=x_faces),
        y=Grid1D(centers=0.5 * (theta_faces[:-1] + theta_faces[1:]), faces=theta_faces),
        z=Grid1D(centers=0.5 * (zeta_faces[:-1] + zeta_faces[1:]), faces=zeta_faces),
    )
    target_shape = grid.shape

    def _position(u: jax.Array) -> jax.Array:
        return rotating_ellipse_position(
            u[0], u[1], u[2],
            r0=r0, elongation=elongation, n_field_periods=n_field_periods,
        )

    _jacobian_of_position = jax.jacfwd(_position)

    def _metric(logical_grid: jax.Array) -> MetricGeometry:
        # g_ij = d_i X . d_j X from the embedding Jacobian (autodiff). The
        # covariant metric is J_cart^T J_cart; the contravariant metric is its
        # inverse and the metric Jacobian is sqrt(det g_cov).
        points = logical_grid.reshape(-1, 3)
        cartesian_jacobian = jax.vmap(_jacobian_of_position)(points)
        g_cov = jnp.einsum("pki,pkj->pij", cartesian_jacobian, cartesian_jacobian)
        g_contra = jnp.linalg.inv(g_cov)
        det_g_cov = jnp.linalg.det(g_cov)
        location_shape = logical_grid.shape[:-1]
        g_cov = g_cov.reshape(location_shape + (3, 3))
        g_contra = g_contra.reshape(location_shape + (3, 3))
        jacobian = jnp.sqrt(jnp.abs(det_g_cov)).reshape(location_shape)
        return MetricGeometry(
            J=jacobian,
            g11=g_contra[..., 0, 0],
            g22=g_contra[..., 1, 1],
            g33=g_contra[..., 2, 2],
            g12=g_contra[..., 0, 1],
            g13=g_contra[..., 0, 2],
            g23=g_contra[..., 1, 2],
            g_11=g_cov[..., 0, 0],
            g_22=g_cov[..., 1, 1],
            g_33=g_cov[..., 2, 2],
            g_12=g_cov[..., 0, 1],
            g_13=g_cov[..., 0, 2],
            g_23=g_cov[..., 1, 2],
        )

    def _bfield(metric: MetricGeometry) -> BFieldGeometry:
        jacobian = metric.J
        B_contra = jnp.stack(
            (
                jnp.zeros_like(jacobian),
                float(iota) * float(c_phi) / jacobian,
                float(c_phi) / jacobian,
            ),
            axis=-1,
        )
        Bmag = _bmag_from_contravariant_components(B_contra, metric.g_cov)
        return BFieldGeometry(B_contra=B_contra, Bmag=Bmag)

    def _logical(x_axis: jax.Array, y_axis: jax.Array, z_axis: jax.Array) -> jax.Array:
        return logical_grid_from_axis_vectors(x_axis, y_axis, z_axis)

    cell_metric = _metric(_logical(grid.x.centers, grid.y.centers, grid.z.centers))
    cell_bfield = _bfield(cell_metric)
    face_metric = FaceMetricGeometry(
        x=_metric(_logical(grid.x.faces, grid.y.centers, grid.z.centers)),
        y=_metric(_logical(grid.x.centers, grid.y.faces, grid.z.centers)),
        z=_metric(_logical(grid.x.centers, grid.y.centers, grid.z.faces)),
    )
    face_bfield = FaceBFieldGeometry(
        x=_bfield(face_metric.x),
        y=_bfield(face_metric.y),
        z=_bfield(face_metric.z),
    )

    if construct_fci_maps:
        map_fields = build_fci_maps_from_b_contravariant(
            grid,
            cell_bfield.B_contra,
            cell_bfield.Bmag,
            periodic_axes=(False, True, True),
            substeps=int(map_substeps),
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
