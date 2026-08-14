"""Focused validation tests for generic angular-agglomeration metadata."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from drbx.geometry import HaloLayout3D, LocalControlVolumeCellGeometry3D
from drbx.native.fci_boundaries import (
    CV_FACE_INTERIOR,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
    LocalMomentReconstruction3D,
)
from drbx.geometry.fci_geometry import LocalRegularFaceGeometry3D


def _active_rows(logical=(2, 1, 2, 1)):
    layout = HaloLayout3D((3, 4, 2), 1)
    rows = LocalControlVolumeFaceRows3D.empty(layout, max_rows=2, max_patches=1)
    return replace(
        rows,
        kind=jnp.asarray([CV_FACE_INTERIOR, 0], dtype=jnp.int32),
        minus_owner_i=jnp.asarray([1, 0], dtype=jnp.int32),
        minus_owner_j=jnp.asarray([1, 0], dtype=jnp.int32),
        minus_owner_k=jnp.asarray([1, 0], dtype=jnp.int32),
        plus_owner_i=jnp.asarray([1, 0], dtype=jnp.int32),
        plus_owner_j=jnp.asarray([2, 0], dtype=jnp.int32),
        plus_owner_k=jnp.asarray([1, 0], dtype=jnp.int32),
        has_plus_owner=jnp.asarray([True, False]),
        patch_active=jnp.asarray([[True], [False]]),
        active=jnp.asarray([True, False]),
        logical_axis=jnp.asarray([logical[0], 99], dtype=jnp.int32),
        logical_face_i=jnp.asarray([logical[1], 99], dtype=jnp.int32),
        logical_face_j=jnp.asarray([logical[2], 99], dtype=jnp.int32),
        logical_face_k=jnp.asarray([logical[3], 99], dtype=jnp.int32),
    )


def _embedded(profile=None):
    layout = HaloLayout3D((3, 4, 2), 1)
    cells = LocalControlVolumeCellGeometry3D.identity(
        layout,
        volume=jnp.ones(layout.owned_shape),
        centroid=jnp.zeros(layout.owned_shape + (3,)),
    )
    regular = LocalRegularFaceGeometry3D(
        layout=layout,
        x_area=jnp.ones(layout.face_control_shape(0)),
        y_area=jnp.ones(layout.face_control_shape(1)),
        z_area=jnp.ones(layout.face_control_shape(2)),
        x_area_fraction=jnp.ones(layout.face_control_shape(0)),
        y_area_fraction=jnp.ones(layout.face_control_shape(1)),
        z_area_fraction=jnp.ones(layout.face_control_shape(2)),
        x_open_mask=jnp.ones(layout.face_control_shape(0), dtype=bool),
        y_open_mask=jnp.ones(layout.face_control_shape(1), dtype=bool),
        z_open_mask=jnp.ones(layout.face_control_shape(2), dtype=bool),
    )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=regular,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(layout),
        reconstruction=LocalMomentReconstruction3D.empty(layout),
        face_functionals=LocalMomentFittedFaceRows3D.empty(layout),
        angular_group_sizes=profile,
    )


def test_face_identity_defaults_and_inactive_rows_are_canonicalized():
    rows = LocalControlVolumeFaceRows3D.empty(HaloLayout3D((3, 4, 2), 1), max_rows=2)
    assert jnp.all(rows.logical_axis == -1)
    rows = _active_rows()
    assert tuple(rows.logical_axis) == (2, -1)
    assert tuple(rows.logical_face_i) == (1, -1)
    assert tuple(rows.logical_face_j) == (2, -1)
    assert tuple(rows.logical_face_k) == (1, -1)


@pytest.mark.parametrize(
    "logical",
    [(3, 0, 0, 0), (0, 4, 0, 0), (1, 0, 5, 0), (2, 0, 0, 3)],
)
def test_active_face_identity_is_checked_against_axis_specific_bounds(logical):
    with pytest.raises(ValueError, match="logical control-volume faces"):
        _active_rows(logical)


def test_face_identity_pytree_roundtrip():
    rows = _active_rows()
    restored = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(rows), jax.tree_util.tree_leaves(rows)
    )
    for name in ("logical_axis", "logical_face_i", "logical_face_j", "logical_face_k"):
        assert jnp.array_equal(getattr(restored, name), getattr(rows, name))


@pytest.mark.parametrize("profile", [(4, 2, 2), (4, 2, 1), (4, 4, 2)])
def test_angular_profile_accepts_nested_power_of_two_groups(profile):
    geometry = _embedded(profile)
    assert geometry.angular_group_sizes == profile
    assert geometry.has_angular_agglomeration


@pytest.mark.parametrize("profile", [(2, 2, 1), (4, 1, 2), (4, 3, 1), (4, 2, 2, 1)])
def test_angular_profile_rejects_invalid_profiles(profile):
    with pytest.raises(ValueError, match="angular_group_sizes"):
        _embedded(profile)


def test_generic_geometry_has_no_angular_profile_and_roundtrips():
    geometry = _embedded()
    assert geometry.angular_group_sizes is None
    assert not geometry.has_angular_agglomeration
    leaves, aux = jax.tree_util.tree_flatten(geometry)
    restored = jax.tree_util.tree_unflatten(aux, leaves)
    assert restored.angular_group_sizes is None
