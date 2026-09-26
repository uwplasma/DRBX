"""Focused runtime contracts for direct control-volume face functionals."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from drbx.fci_braginskii.geometry.fci_geometry import (
    HaloLayout3D,
)
from drbx.fci_braginskii.native.fci_boundaries import (
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalControlVolumeFieldClosure3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
)
from drbx.fci_braginskii.native import fci_operators
from drbx.fci_braginskii.native.fci_operators import (
    local_perp_laplacian_conservative_op,
)


def _geometry(
    *,
    neighbor: bool = True,
    reference_dirichlet: bool = True,
    boundary_source_shard: int = 0,
):
    layout = HaloLayout3D((1, 1, 1), 1)
    rows = LocalMomentFittedFaceRows3D(
        layout=layout,
        functional_face_id=jnp.array([8], dtype=jnp.int64),
        observation_kind=jnp.array([[
            CV_RECONSTRUCTION_EQUATION_CELL,
            CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
            CV_RECONSTRUCTION_EQUATION_DIRICHLET,
        ]], dtype=jnp.int32),
        owned_i=jnp.zeros((1, 3), dtype=jnp.int32),
        owned_j=jnp.zeros((1, 3), dtype=jnp.int32),
        owned_k=jnp.zeros((1, 3), dtype=jnp.int32),
        halo_i=jnp.array([[0, 2, 0]], dtype=jnp.int32),
        halo_j=jnp.ones((1, 3), dtype=jnp.int32),
        halo_k=jnp.ones((1, 3), dtype=jnp.int32),
        boundary_face_row=jnp.zeros((1, 3), dtype=jnp.int32),
        boundary_patch=jnp.zeros((1, 3), dtype=jnp.int32),
        boundary_quadrature=jnp.zeros((1, 3), dtype=jnp.int32),
        boundary_source_shard=jnp.full(
            (1, 3), boundary_source_shard, dtype=jnp.int32
        ),
        observation_active=jnp.array([[True, True, reference_dirichlet]]),
        projected_flux_weights=jnp.array([[1.0, 2.0, 3.0 if reference_dirichlet else 0.0]]),
        parallel_flux_weights=jnp.array([[4.0, 5.0, 6.0 if reference_dirichlet else 0.0]]),
        parallel_gradient_flux_weights=jnp.array(
            [[7.0, 8.0, 9.0 if reference_dirichlet else 0.0]]
        ),
        polynomial_order=jnp.array([3], dtype=jnp.int32),
        rank=jnp.array([20], dtype=jnp.int32),
        condition_number=jnp.array([1.0]),
        reproduction_residual=jnp.array([0.0]),
        normalized_projected_weight_norm=jnp.array([1.0]),
        normalized_parallel_weight_norm=jnp.array([1.0]),
        normalized_parallel_gradient_weight_norm=jnp.array([1.0]),
        active=jnp.array([True]),
        max_rows=1,
        max_equations=3,
    )
    faces = SimpleNamespace(
        max_rows=1,
        max_patches=1,
        active=jnp.array([True]),
        has_plus_owner=jnp.array([neighbor]),
        has_remote_owner=jnp.array([False]),
        quadrature_active=jnp.array([[[True, False, False, False]]]),
        J=jnp.array([[[2.0, 0.0, 0.0, 0.0]]]),
        area_covector_weight=jnp.array([[[[3.0, 4.0, 0.0]] * 4]]),
    )
    # The runtime builder deliberately depends only on the compiled face rows.
    geometry = object.__new__(LocalEmbeddedControlVolumeGeometry3D)
    object.__setattr__(geometry, "cells", SimpleNamespace(layout=layout))
    object.__setattr__(geometry, "irregular_faces", faces)
    object.__setattr__(geometry, "face_functionals", rows)
    return layout, geometry


def test_perpendicular_cv_path_uses_direct_flux_without_polynomial(monkeypatch):
    closure = LocalControlVolumeFieldClosure3D(
        projected_flux=jnp.array([13.0]),
        parallel_flux=jnp.array([17.0]),
        parallel_gradient_flux=jnp.array([19.0]),
        valid=jnp.array([True]), active=jnp.array([True]), max_rows=1,
    )
    geometry = object.__new__(LocalEmbeddedControlVolumeGeometry3D)
    object.__setattr__(geometry, "regular_faces", object())
    object.__setattr__(geometry, "regular_boundary_closure", None)
    stencil = SimpleNamespace(
        regular_flux=SimpleNamespace(x=jnp.array(1.0), y=jnp.array(2.0), z=jnp.array(3.0))
    )
    monkeypatch.setattr(fci_operators, "build_local_perp_laplacian_stencil", lambda *a, **k: stencil)
    monkeypatch.setattr(fci_operators, "_require_local_control_volume_field_closure", lambda c, g: c)
    monkeypatch.setattr(fci_operators, "_local_control_volume_integrated_divergence", lambda regular, irregular, *a, **k: irregular)
    result = local_perp_laplacian_conservative_op(
        object(), object(), object(), face_projectors=(jnp.array(0.0),) * 3,
        control_volume_geometry=geometry, field_closure=closure,
    )
    np.testing.assert_allclose(result, [13.0])


def test_face_row_pytree_roundtrip_preserves_ids_and_inactive_padding() -> None:
    layout, geometry = _geometry()
    rows = geometry.face_functionals
    leaves, treedef = jax.tree_util.tree_flatten(rows)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.functional_face_id.dtype == jnp.int64
    np.testing.assert_array_equal(restored.functional_face_id, [8])
    np.testing.assert_array_equal(restored.boundary_source_shard, [[0, 0, 0]])
    np.testing.assert_array_equal(restored.active, [True])

    padded = LocalMomentFittedFaceRows3D.empty(
        layout,
        max_rows=2,
        max_equations=4,
    )
    leaves, treedef = jax.tree_util.tree_flatten(padded)
    restored_padding = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(restored_padding.functional_face_id, [-1, -1])
    np.testing.assert_array_equal(
        restored_padding.boundary_source_shard,
        np.zeros((2, 4), dtype=np.int32),
    )
    assert not bool(jnp.any(restored_padding.active))
    assert not bool(jnp.any(restored_padding.observation_active))
