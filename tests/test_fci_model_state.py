from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from jax_drb.geometry import HaloLayout3D
from jax_drb.native.fci_boundaries import LocalBoundaryFaceBC3D
from jax_drb.native.fci_2_field_rhs import Fci2FieldState
from jax_drb.native.fci_4_field_rhs import Fci4FieldState
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState
from jax_drb.native.fci_model import (
    FciFieldBundle,
    assert_matching_field_names,
    extract_owned_field_from_halo,
    extract_owned_state_from_halo,
    inject_owned_field_to_halo,
    inject_owned_state_to_halo,
    update_halo_owned_slice,
    update_state_halo_owned_slices,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _MixedFieldBundle(FciFieldBundle):
    field: jax.Array
    face_bc: LocalBoundaryFaceBC3D


def _round_trip_state(state):
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in state.field_names():
        assert jnp.allclose(getattr(rebuilt, name), getattr(state, name))


def _check_state_helpers(state, *, field_names):
    assert state.field_names() == field_names
    state.assert_field_shape((2, 2, 2))
    zero = state.zeros_like()
    for name in field_names:
        assert jnp.allclose(getattr(zero, name), 0.0)
    doubled = state.axpy(state, scale=1.0)
    for name in field_names:
        assert jnp.allclose(getattr(doubled, name), 2.0 * getattr(state, name))
    _round_trip_state(state)


def test_2field_state_helpers() -> None:
    state = Fci2FieldState(
        density=jnp.ones((2, 2, 2)),
        v_parallel=2.0 * jnp.ones((2, 2, 2)),
        density_background=3.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "v_parallel", "density_background"))


def test_4field_state_helpers() -> None:
    state = Fci4FieldState(
        density=jnp.ones((2, 2, 2)),
        omega=2.0 * jnp.ones((2, 2, 2)),
        v_ion_parallel=3.0 * jnp.ones((2, 2, 2)),
        v_electron_parallel=4.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "omega", "v_ion_parallel", "v_electron_parallel"))


def test_eb_state_helpers() -> None:
    state = FciDrbEBState(
        density=jnp.ones((2, 2, 2)),
        phi=2.0 * jnp.ones((2, 2, 2)),
        Te=3.0 * jnp.ones((2, 2, 2)),
        Ti=4.0 * jnp.ones((2, 2, 2)),
        Vi=5.0 * jnp.ones((2, 2, 2)),
        Ve=6.0 * jnp.ones((2, 2, 2)),
        vorticity=7.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"))


def test_generic_field_bundle_accepts_non_array_pytree_fields() -> None:
    layout = HaloLayout3D((2, 2, 2), halo_width=1)
    bundle = _MixedFieldBundle(
        field=jnp.ones((2, 2, 2)),
        face_bc=LocalBoundaryFaceBC3D.empty(layout),
    )
    assert bundle.field_names() == ("field", "face_bc")
    assert bundle.field_items()[1][0] == "face_bc"
    replaced = bundle.replace(field=2.0 * bundle.field)
    mapped = bundle.map_fields(lambda value: value)
    assert jnp.all(replaced.field == 2.0)
    assert mapped.face_bc.layout == layout
    leaves, treedef = jax.tree_util.tree_flatten(bundle)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(restored.field, bundle.field)
    assert restored.face_bc.layout == bundle.face_bc.layout


def test_owned_halo_field_storage_helpers() -> None:
    layout = HaloLayout3D((2, 3, 4), halo_width=1)
    owned = jnp.arange(24.0).reshape(layout.owned_shape)
    halo = inject_owned_field_to_halo(owned, layout, fill_value=-7.0)
    assert halo.shape == layout.cell_halo_shape
    assert jnp.all(halo[0] == -7.0)
    assert jnp.array_equal(extract_owned_field_from_halo(halo, layout), owned)

    replacement = 2.0 * owned
    updated = update_halo_owned_slice(halo, replacement, layout)
    assert jnp.array_equal(extract_owned_field_from_halo(updated, layout), replacement)
    assert jnp.all(updated[0] == -7.0)


def test_owned_halo_state_storage_helpers() -> None:
    layout = HaloLayout3D((2, 3, 4), halo_width=1)
    state_owned = Fci2FieldState(
        density=jnp.arange(24.0).reshape(layout.owned_shape),
        v_parallel=jnp.full(layout.owned_shape, 2.0),
        density_background=jnp.full(layout.owned_shape, 3.0),
    )

    state_halo = inject_owned_state_to_halo(state_owned, layout, fill_value=-5.0)
    assert type(state_halo) is type(state_owned)
    state_halo.assert_field_shape(layout.cell_halo_shape)
    assert jnp.all(state_halo.density[0] == -5.0)
    extracted = extract_owned_state_from_halo(state_halo, layout)
    assert type(extracted) is type(state_owned)
    for name in state_owned.field_names():
        assert jnp.array_equal(getattr(extracted, name), getattr(state_owned, name))

    replacement = state_owned.replace(
        density=10.0 + state_owned.density,
        v_parallel=20.0 + state_owned.v_parallel,
        density_background=30.0 + state_owned.density_background,
    )
    updated = update_state_halo_owned_slices(state_halo, replacement, layout)
    assert type(updated) is type(state_owned)
    for name in replacement.field_names():
        assert jnp.array_equal(
            getattr(extract_owned_state_from_halo(updated, layout), name),
            getattr(replacement, name),
        )
    assert jnp.all(updated.density[0] == -5.0)


def test_state_halo_update_requires_matching_concrete_type() -> None:
    layout = HaloLayout3D((2, 2, 2), halo_width=1)
    state_2 = Fci2FieldState(
        density=jnp.ones(layout.owned_shape),
        v_parallel=jnp.ones(layout.owned_shape),
        density_background=jnp.ones(layout.owned_shape),
    )
    state_4 = Fci4FieldState(
        density=jnp.ones(layout.cell_halo_shape),
        omega=jnp.ones(layout.cell_halo_shape),
        v_ion_parallel=jnp.ones(layout.cell_halo_shape),
        v_electron_parallel=jnp.ones(layout.cell_halo_shape),
    )
    with pytest.raises(TypeError, match="matching concrete state types"):
        update_state_halo_owned_slices(state_4, state_2, layout)


def test_matching_field_names_is_explicit() -> None:
    layout = HaloLayout3D((2, 2, 2), halo_width=1)
    lhs = _MixedFieldBundle(
        field=jnp.ones(layout.owned_shape),
        face_bc=LocalBoundaryFaceBC3D.empty(layout),
    )

    @jax.tree_util.register_pytree_node_class
    @dataclass(frozen=True)
    class _OtherFieldBundle(FciFieldBundle):
        other: jax.Array

    rhs = _OtherFieldBundle(other=jnp.ones(layout.owned_shape))
    assert_matching_field_names(lhs, lhs)
    with pytest.raises(ValueError, match="matching field names"):
        assert_matching_field_names(lhs, rhs)
