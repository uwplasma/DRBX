from __future__ import annotations

import jax.numpy as jnp

from ..geometry.fci_geometry import _first_derivative_3d


def fci_zup(field, geometry):
    """Legacy helper for one-step upward transport along the toroidal index."""

    values = jnp.asarray(field, dtype=jnp.float64)
    if values.shape != geometry.shape:
        raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
    return jnp.roll(values, shift=-1, axis=2)


def grad_parallel_fci(field, geometry):
    """Legacy wrapper for the direct parallel gradient operator."""

    from .fci_operators import build_conservative_stencil_from_field, grad_parallel_op_direct

    stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        periodic_axes=(False, True, True),
        face_bc=None,
    )
    return grad_parallel_op_direct(stencil, geometry)


def logical_exb_bracket_xy(phi, field, geometry, *, periodic_x: bool = True, periodic_y: bool = True):
    """Legacy wrapper for the logical ExB bracket."""

    from .fci_operators import build_conservative_stencil_from_field, poisson_bracket_op

    phi_stencil = build_conservative_stencil_from_field(
        phi,
        geometry,
        periodic_axes=(bool(periodic_x), bool(periodic_y), True),
        face_bc=None,
    )
    field_stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        periodic_axes=(bool(periodic_x), bool(periodic_y), True),
        face_bc=None,
    )
    return poisson_bracket_op(phi_stencil, field_stencil, geometry)


def metric_weighted_scalar_laplacian_3d(
    field,
    geometry,
    coefficient=1.0,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
):
    """Legacy wrapper for the metric-weighted scalar Laplacian."""

    values = jnp.asarray(field, dtype=jnp.float64)
    coeff = jnp.asarray(coefficient, dtype=jnp.float64)
    if coeff.ndim == 0:
        coeff = jnp.broadcast_to(coeff, values.shape)

    dfdx = _first_derivative_3d(values, geometry.spacing.dx, axis=0, periodic=bool(periodic_axes[0]))
    dfdy = _first_derivative_3d(values, geometry.spacing.dy, axis=1, periodic=bool(periodic_axes[1]))
    dfdz = _first_derivative_3d(values, geometry.spacing.dz, axis=2, periodic=bool(periodic_axes[2]))
    d2fdx2 = _first_derivative_3d(dfdx, geometry.spacing.dx, axis=0, periodic=bool(periodic_axes[0]))
    d2fdy2 = _first_derivative_3d(dfdy, geometry.spacing.dy, axis=1, periodic=bool(periodic_axes[1]))
    d2fdz2 = _first_derivative_3d(dfdz, geometry.spacing.dz, axis=2, periodic=bool(periodic_axes[2]))
    return coeff * (d2fdx2 + d2fdy2 + d2fdz2)


def conservative_perp_diffusion_xy(
    field,
    coefficient,
    geometry,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
):
    """Legacy wrapper for the conservative perpendicular diffusion operator."""

    from .fci_operators import build_conservative_stencil_from_field, perp_laplacian_conservative_op

    stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=None,
    )
    diff = perp_laplacian_conservative_op(stencil, geometry, periodic_axes=periodic_axes)
    coeff = jnp.asarray(coefficient, dtype=jnp.float64)
    if coeff.ndim == 0:
        coeff = jnp.broadcast_to(coeff, diff.shape)
    return coeff * diff


def conservative_parallel_diffusion_fci(
    field,
    coefficient,
    geometry,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
):
    """Legacy wrapper for the conservative parallel diffusion operator."""

    from .fci_operators import build_conservative_stencil_from_field, parallel_laplacian_conservative_op

    stencil = build_conservative_stencil_from_field(
        field,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=None,
    )
    diff = parallel_laplacian_conservative_op(stencil, geometry, periodic_axes=periodic_axes)
    coeff = jnp.asarray(coefficient, dtype=jnp.float64)
    if coeff.ndim == 0:
        coeff = jnp.broadcast_to(coeff, diff.shape)
    return coeff * diff
