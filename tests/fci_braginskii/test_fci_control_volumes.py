"""Focused contracts for global agglomeration and direct moment functionals."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from drbx.fci_braginskii.geometry.fci_geometry import (
    HaloLayout3D,
    LocalControlVolumeCellGeometry3D,
)
from drbx.fci_braginskii.native.fci_operators import expand_local_control_volume_owner_field

from drbx.fci_braginskii.native.fci_boundaries import CV_RECONSTRUCTION_EQUATION_CELL
from drbx.fci_braginskii.native.fci_control_volume_operators import (
    LocalMomentFittedFaceFunctional3D,
)


def _raw_geometry(shape: tuple[int, int, int]):
    coordinate = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.float64) + 0.5 for size in shape),
            indexing="ij",
        ),
        axis=-1,
    )
    volume = np.ones(shape, dtype=np.float64)
    second = np.zeros(shape + (3, 3), dtype=np.float64)
    second[..., 0, 0] = 1.0 / 12.0
    second[..., 1, 1] = 1.0 / 12.0
    second[..., 2, 2] = 1.0 / 12.0
    third = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    fraction = np.ones(shape, dtype=np.float64)
    return volume, coordinate, second, third, fraction


def test_control_volume_aggregate_id_defaults_to_owner_identity() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    volume, centroid, second, third, _ = _raw_geometry((2, 2, 2))
    cells = LocalControlVolumeCellGeometry3D.identity(
        layout, volume=jnp.asarray(volume), centroid=jnp.asarray(centroid),
        second_moment=jnp.asarray(second), third_moment=jnp.asarray(third),
    )
    expected = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    np.testing.assert_array_equal(np.asarray(cells.aggregate_id), expected)
    leaves, tree = jax.tree_util.tree_flatten(cells)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(np.asarray(restored.aggregate_id), expected)


def test_remote_owner_metadata_and_expansion_use_owner_halo() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    volume = jnp.ones((2, 2, 2), dtype=jnp.float64)
    centroid = jnp.zeros((2, 2, 2, 3), dtype=jnp.float64)
    base = LocalControlVolumeCellGeometry3D.identity(layout, volume=volume, centroid=centroid)
    source = (0, 0, 0)
    cells = LocalControlVolumeCellGeometry3D(
        **{**base.__dict__, "is_merged_source": base.is_merged_source.at[source].set(True),
           "is_active_owner": base.is_active_owner.at[source].set(False),
           "aggregate_volume": base.aggregate_volume.at[source].set(0.0),
           "owner_is_remote": jnp.zeros((2,2,2), dtype=bool).at[source].set(True),
           "remote_owner_halo_i": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(3),
           "remote_owner_halo_j": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(1),
           "remote_owner_halo_k": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(1)})
    values = jnp.arange(8, dtype=jnp.float64).reshape((2,2,2))
    with pytest.raises(ValueError, match="owner_values_halo"):
        expand_local_control_volume_owner_field(values, cells)
    halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64).at[3,1,1].set(37.0)
    expanded = expand_local_control_volume_owner_field(values, cells, owner_values_halo=halo)
    assert float(expanded[source]) == 37.0


def test_face_functional_reference_padding_is_valid_only_when_inactive() -> None:
    kwargs = dict(
        equation_kind=np.array([CV_RECONSTRUCTION_EQUATION_CELL]),
        sample_reference=np.array([-1]),
        value_weights=np.zeros(1),
        gradient_weights=np.zeros((3, 1)),
        polynomial_order=3,
        rank=1,
        condition_number=1.0,
        reproduction_residual=0.0,
        normalized_weight_norm=0.0,
    )
    with pytest.raises(ValueError, match="nonnegative references"):
        LocalMomentFittedFaceFunctional3D(active=np.array([True]), **kwargs)
    padded = LocalMomentFittedFaceFunctional3D(active=np.array([False]), **kwargs)
    assert int(padded.sample_reference[0]) == -1
