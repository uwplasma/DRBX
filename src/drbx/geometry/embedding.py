"""Metric tensors from an analytic embedding, by automatic differentiation.

Given a map from logical coordinates ``u = (x, theta, zeta)`` to Cartesian
space, the covariant metric is ``g_ij = d_i X . d_j X`` — computed here exactly
with ``jax.jacfwd`` instead of by hand. Every analytic geometry in the package
(rotating ellipse, island divertor) builds its :class:`MetricGeometry` through
this one function, so a new geometry only has to supply its embedding.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .fci_geometry import MetricGeometry

__all__ = ["metric_from_position_fn"]


def metric_from_position_fn(
    position_fn: Callable[[jax.Array], jax.Array],
    logical_grid: jax.Array,
) -> MetricGeometry:
    """Build the metric on ``logical_grid`` from ``position_fn(u) -> (X, Y, Z)``.

    The covariant metric is the Gram matrix of the embedding Jacobian, the
    contravariant metric its inverse, and the Jacobian ``sqrt(det g_cov)``.
    """

    jacobian_of_position = jax.jacfwd(position_fn)
    points = logical_grid.reshape(-1, 3)
    cartesian_jacobian = jax.vmap(jacobian_of_position)(points)
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
