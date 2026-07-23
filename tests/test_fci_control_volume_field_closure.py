"""Focused runtime contracts for direct control-volume face functionals."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import HaloLayout3D
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NOFLUX,
    BC_NORMALFLUX,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFieldClosure3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
)
from drbx.native import fci_operators
from drbx.native.fci_operators import (
    build_local_control_volume_field_closure,
    local_parallel_laplacian_conservative_op,
)


def _geometry(*, neighbor: bool = True, reference_dirichlet: bool = True):
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


def _bc(kind=BC_DIRICHLET, value=7.0):
    return LocalControlVolumeBoundaryBC3D(
        kind=jnp.array([kind], dtype=jnp.int32),
        centroid_value=jnp.zeros((1,)),
        quadrature_value=jnp.array([[[value, 0.0, 0.0, 0.0]]]),
        active=jnp.array([True]),
        max_rows=1,
        max_patches=1,
    )


def _field(layout):
    field = jnp.full(layout.cell_halo_shape, jnp.nan)
    field = field.at[1, 1, 1].set(2.0)  # owned observation
    field = field.at[2, 1, 1].set(5.0)  # concrete remote-halo observation
    return field


def test_direct_face_closure_eager_jit_and_pytree() -> None:
    layout, geometry = _geometry()
    field = _field(layout)
    eager = build_local_control_volume_field_closure(field, geometry, _bc())
    compiled = jax.jit(lambda value: build_local_control_volume_field_closure(value, geometry, _bc()))(field)
    np.testing.assert_allclose(eager.projected_flux, [33.0])
    np.testing.assert_allclose(eager.parallel_flux, [75.0])
    np.testing.assert_allclose(eager.parallel_gradient_flux, [117.0])
    np.testing.assert_allclose(compiled.projected_flux, eager.projected_flux)
    assert bool(eager.valid[0])
    doubled = jax.tree.map(lambda value: value * 2 if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact) else value, eager)
    np.testing.assert_allclose(doubled.parallel_flux, [150.0])


def test_face_row_pytree_roundtrip_preserves_ids_and_inactive_padding() -> None:
    layout, geometry = _geometry()
    rows = geometry.face_functionals
    leaves, treedef = jax.tree_util.tree_flatten(rows)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.functional_face_id.dtype == jnp.int64
    np.testing.assert_array_equal(restored.functional_face_id, [8])
    np.testing.assert_array_equal(restored.active, [True])

    padded = LocalMomentFittedFaceRows3D.empty(
        layout,
        max_rows=2,
        max_equations=4,
    )
    leaves, treedef = jax.tree_util.tree_flatten(padded)
    restored_padding = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(restored_padding.functional_face_id, [-1, -1])
    assert not bool(jnp.any(restored_padding.active))
    assert not bool(jnp.any(restored_padding.observation_active))


def test_boundary_overrides_and_invalid_references() -> None:
    # The direct row was compiled with a Dirichlet trace, but a prescribed
    # target normal flux/no-flux must replace that fit exactly.
    layout, boundary_geometry = _geometry(neighbor=False)
    field = _field(layout)
    noflux = build_local_control_volume_field_closure(field, boundary_geometry, _bc(BC_NOFLUX))
    normal = build_local_control_volume_field_closure(field, boundary_geometry, _bc(BC_NORMALFLUX, 7.0))
    assert bool(noflux.valid[0])
    assert bool(normal.valid[0])
    np.testing.assert_allclose(noflux.parallel_flux, [0.0])
    np.testing.assert_allclose(normal.parallel_flux, [70.0])  # 2 * |(3,4,0)| * 7
    np.testing.assert_allclose(normal.parallel_gradient_flux, [70.0])

    layout, geometry = _geometry()
    poisoned = _field(layout).at[2, 1, 1].set(jnp.nan)
    invalid_halo = build_local_control_volume_field_closure(poisoned, geometry, _bc())
    assert not bool(invalid_halo.valid[0])
    invalid_bc = build_local_control_volume_field_closure(_field(layout), geometry, _bc(BC_NEUMANN))
    assert not bool(invalid_bc.valid[0])


def test_parallel_laplacian_uses_parallel_gradient_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel diffusion must not reuse either P-perp or advective flux."""
    closure = LocalControlVolumeFieldClosure3D(
        projected_flux=jnp.array([11.0]),
        parallel_flux=jnp.array([22.0]),
        parallel_gradient_flux=jnp.array([33.0]),
        valid=jnp.array([True]),
        active=jnp.array([True]),
        max_rows=1,
    )
    geometry = SimpleNamespace(regular_faces=object(), regular_boundary_closure=None)
    stencil = SimpleNamespace(
        regular_flux=SimpleNamespace(x=jnp.array(1.0), y=jnp.array(2.0), z=jnp.array(3.0))
    )
    monkeypatch.setattr(
        fci_operators,
        "build_local_projected_laplacian_flux_stencil",
        lambda *args, **kwargs: stencil,
    )
    monkeypatch.setattr(
        fci_operators,
        "_require_local_control_volume_field_closure",
        lambda candidate, candidate_geometry: candidate,
    )

    captured = {}

    def integrated(regular_flux, irregular_flux, *args, **kwargs):
        captured["irregular_flux"] = irregular_flux
        return irregular_flux

    monkeypatch.setattr(
        fci_operators, "_local_control_volume_integrated_divergence", integrated
    )
    result = local_parallel_laplacian_conservative_op(
        object(),
        object(),
        object(),
        face_projectors=(jnp.array(0.0),) * 3,
        control_volume_geometry=geometry,
        field_closure=closure,
    )
    np.testing.assert_allclose(result, [33.0])
    np.testing.assert_allclose(captured["irregular_flux"], [33.0])
