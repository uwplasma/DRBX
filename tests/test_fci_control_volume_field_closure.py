"""Focused runtime contracts for direct control-volume face functionals."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import HaloLayout3D, LocalDomain3D, ShardSpec3D
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NOFLUX,
    BC_NORMALFLUX,
    CV_FACE_CUT_WALL,
    CV_FACE_INTERIOR,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFieldClosure3D,
    LocalControlVolumePolynomial3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
)
from drbx.native import fci_operators
from drbx.native.fci_operators import (
    build_local_control_volume_field_closure,
    local_parallel_laplacian_conservative_op,
    replace_local_control_volume_projected_flux_with_owner_polynomials,
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


def _owner_polynomial_fixture(
    *,
    kind: int = CV_FACE_INTERIOR,
    has_plus_owner: bool = True,
    has_remote_owner: bool = False,
    global_radial_boundary: bool = False,
    minus_valid: bool = True,
    plus_valid: bool = True,
    remote_valid: bool = True,
):
    """One oriented face with constant owner polynomials.

    The x component of the face covector is two, so the selected gradients
    produce easy-to-read flux values.  The local owners are interior in the
    global radial direction unless the fixture explicitly places the shard at
    the lower global boundary.
    """
    layout = HaloLayout3D((3, 2, 2), 1)
    cells_shape = layout.owned_shape
    cells = SimpleNamespace(
        shape=cells_shape,
        centroid=jnp.zeros(cells_shape + (3,), dtype=jnp.float64),
        second_moment=jnp.zeros(cells_shape + (3, 3), dtype=jnp.float64),
        third_moment=jnp.zeros(cells_shape + (3, 3, 3), dtype=jnp.float64),
    )
    faces = SimpleNamespace(
        max_rows=1,
        max_patches=1,
        active=jnp.array([True]),
        kind=jnp.array([kind], dtype=jnp.int32),
        minus_owner_i=jnp.array([0], dtype=jnp.int32),
        minus_owner_j=jnp.array([0], dtype=jnp.int32),
        minus_owner_k=jnp.array([0], dtype=jnp.int32),
        plus_owner_i=jnp.array([1], dtype=jnp.int32),
        plus_owner_j=jnp.array([0], dtype=jnp.int32),
        plus_owner_k=jnp.array([0], dtype=jnp.int32),
        has_plus_owner=jnp.array([has_plus_owner]),
        has_remote_owner=jnp.array([has_remote_owner]),
        remote_halo_i=jnp.array([2], dtype=jnp.int32),
        remote_halo_j=jnp.array([1], dtype=jnp.int32),
        remote_halo_k=jnp.array([1], dtype=jnp.int32),
        quadrature_points=jnp.array([[[[0.0, 0.0, 0.0]] * 4]]),
        quadrature_active=jnp.array([[[True, False, False, False]]]),
        J=jnp.array([[[1.0, 0.0, 0.0, 0.0]]]),
        area_covector_weight=jnp.array([[[[2.0, 0.0, 0.0]] * 4]]),
        projector=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64), (1, 1, 4, 3, 3)
        ),
    )
    geometry = object.__new__(LocalEmbeddedControlVolumeGeometry3D)
    object.__setattr__(geometry, "cells", cells)
    object.__setattr__(geometry, "irregular_faces", faces)
    object.__setattr__(geometry, "face_functionals", None)

    gradient = jnp.zeros(cells_shape + (3,), dtype=jnp.float64)
    gradient = gradient.at[0, 0, 0, 0].set(1.0)
    gradient = gradient.at[1, 0, 0, 0].set(3.0)
    valid = jnp.ones(cells_shape, dtype=bool)
    valid = valid.at[0, 0, 0].set(minus_valid)
    valid = valid.at[1, 0, 0].set(plus_valid)
    remote_gradient = jnp.zeros((1, 1, 4, 3), dtype=jnp.float64)
    remote_gradient = remote_gradient.at[0, 0, 0, 0].set(7.0)
    polynomial = LocalControlVolumePolynomial3D(
        gradient=gradient,
        hessian=jnp.zeros(cells_shape + (3, 3), dtype=jnp.float64),
        third_derivative=jnp.zeros(cells_shape + (3, 3, 3), dtype=jnp.float64),
        valid=valid,
        polynomial_order=jnp.ones(cells_shape, dtype=jnp.int32),
        condition_number=jnp.ones(cells_shape, dtype=jnp.float64),
        owner_values=jnp.zeros(cells_shape, dtype=jnp.float64),
        remote_face_value=jnp.zeros((1, 1, 4), dtype=jnp.float64),
        remote_face_gradient=remote_gradient,
        remote_face_valid=jnp.full((1, 1, 4), remote_valid),
    )
    owned_start = ((0 if global_radial_boundary else 2), 0, 0)
    domain = LocalDomain3D(
        layout=layout,
        shard_spec=ShardSpec3D(
            global_shape=(7, 2, 2),
            owned_start=owned_start,
            owned_stop=tuple(
                start + size
                for start, size in zip(owned_start, layout.owned_shape)
            ),
            shard_index=(0, 0, 0),
            shard_counts=(1, 1, 1),
            periodic_axes=(False, False, False),
            halo_width=layout.halo_width,
        ),
    )
    closure = LocalControlVolumeFieldClosure3D(
        projected_flux=jnp.array([99.0]),
        parallel_flux=jnp.array([17.0]),
        parallel_gradient_flux=jnp.array([19.0]),
        valid=jnp.array([True]),
        active=jnp.array([True]),
        max_rows=1,
    )
    return geometry, domain, polynomial, closure


def _replace_owner_flux(*args, **kwargs):
    field_closure, polynomial, geometry, domain = args
    eager = replace_local_control_volume_projected_flux_with_owner_polynomials(
        *args, **kwargs
    )
    compiled = jax.jit(
        lambda field_closure, polynomial: (
            replace_local_control_volume_projected_flux_with_owner_polynomials(
                field_closure,
                polynomial,
                geometry,
                domain,
                **kwargs,
            )
        )
    )(field_closure, polynomial)
    return eager, compiled


def test_owner_polynomial_two_owner_flux_replaces_projected_only() -> None:
    geometry, domain, polynomial, closure = _owner_polynomial_fixture()
    eager, compiled = _replace_owner_flux(
        closure,
        polynomial,
        geometry,
        domain,
        use_two_owner_flux=True,
        use_cut_wall_owner_flux=False,
    )
    for result in (eager, compiled):
        np.testing.assert_allclose(result.projected_flux, [4.0])
        np.testing.assert_allclose(result.parallel_flux, closure.parallel_flux)
        np.testing.assert_allclose(
            result.parallel_gradient_flux, closure.parallel_gradient_flux
        )
        assert bool(result.valid[0])


def test_owner_polynomial_remote_flux_uses_remote_gradient_orientation() -> None:
    geometry, domain, polynomial, closure = _owner_polynomial_fixture(
        has_plus_owner=False,
        has_remote_owner=True,
    )
    eager, compiled = _replace_owner_flux(
        closure,
        polynomial,
        geometry,
        domain,
        use_two_owner_flux=True,
        use_cut_wall_owner_flux=False,
    )
    for result in (eager, compiled):
        # 2 * (minus_x + remote_x) / 2; the remote gradient keeps the
        # canonical row orientation and is not negated on exchange.
        np.testing.assert_allclose(result.projected_flux, [8.0])
        np.testing.assert_allclose(result.parallel_flux, closure.parallel_flux)
        np.testing.assert_allclose(
            result.parallel_gradient_flux, closure.parallel_gradient_flux
        )


def test_owner_polynomial_cut_wall_uses_minus_owner() -> None:
    geometry, domain, polynomial, closure = _owner_polynomial_fixture(
        kind=CV_FACE_CUT_WALL,
        has_plus_owner=False,
    )
    eager, compiled = _replace_owner_flux(
        closure,
        polynomial,
        geometry,
        domain,
        use_two_owner_flux=False,
        use_cut_wall_owner_flux=True,
    )
    for result in (eager, compiled):
        np.testing.assert_allclose(result.projected_flux, [2.0])
        np.testing.assert_allclose(result.parallel_flux, closure.parallel_flux)
        np.testing.assert_allclose(
            result.parallel_gradient_flux, closure.parallel_gradient_flux
        )


def test_owner_polynomial_invalid_selected_owner_invalidates_row() -> None:
    geometry, domain, polynomial, closure = _owner_polynomial_fixture(
        plus_valid=False,
    )
    eager, compiled = _replace_owner_flux(
        closure,
        polynomial,
        geometry,
        domain,
        use_two_owner_flux=True,
        use_cut_wall_owner_flux=False,
    )
    for result in (eager, compiled):
        assert not bool(result.valid[0])
        assert np.isnan(np.asarray(result.projected_flux)[0])


def test_owner_polynomial_global_radial_boundary_keeps_direct_flux() -> None:
    geometry, domain, polynomial, closure = _owner_polynomial_fixture(
        global_radial_boundary=True,
    )
    eager, compiled = _replace_owner_flux(
        closure,
        polynomial,
        geometry,
        domain,
        use_two_owner_flux=True,
        use_cut_wall_owner_flux=True,
    )
    for result in (eager, compiled):
        np.testing.assert_allclose(result.projected_flux, closure.projected_flux)
        np.testing.assert_allclose(result.parallel_flux, closure.parallel_flux)
        np.testing.assert_allclose(
            result.parallel_gradient_flux, closure.parallel_gradient_flux
        )


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
