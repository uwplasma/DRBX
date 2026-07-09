from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from jax_drb.geometry import (
    HaloLayout3D,
    LocalDomain3D,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
)
from jax_drb.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN
from jax_drb.native.fci_boundaries import LocalBoundaryFaceBC3D
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    PolarAxisRegularScalarRule3D,
    TopologyHaloFiller3D,
)


def _domain(
    *,
    owned_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    periodic_axes: tuple[bool, bool, bool],
    halo_width: int = 1,
) -> LocalDomain3D:
    layout = HaloLayout3D(owned_shape, halo_width)
    spec = ShardSpec3D(
        global_shape=tuple(
            size * count for size, count in zip(owned_shape, shard_counts)
        ),
        owned_start=(0, 0, 0),
        owned_stop=owned_shape,
        shard_index=(0, 0, 0),
        shard_counts=shard_counts,
        periodic_axes=periodic_axes,
        halo_width=halo_width,
    )
    return LocalDomain3D(layout=layout, shard_spec=spec)


def test_halo_exchange_static_config_roundtrips_through_pytree() -> None:
    exchange = HaloExchange3D(
        mesh_axis_names=("x", None, None),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        exchange_axes=(True, False, False),
    )

    leaves, treedef = jax.tree_util.tree_flatten(exchange)
    assert leaves == []
    restored = jax.tree_util.tree_unflatten(treedef, [])
    assert restored == exchange


def test_single_shard_exchange_preserves_input() -> None:
    domain = _domain(
        owned_shape=(2, 3, 4),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, True),
    )
    field = jnp.arange(jnp.prod(jnp.asarray(domain.layout.cell_halo_shape))).reshape(
        domain.layout.cell_halo_shape
    )
    exchange = HaloExchange3D(
        mesh_axis_names=(None, None, None),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, True),
    )

    assert jnp.array_equal(exchange(field, domain), field)


def test_shard_side_kinds_default_and_describe_global_boundaries() -> None:
    periodic = ShardSpec3D(
        global_shape=(8, 4, 4),
        owned_start=(0, 0, 0),
        owned_stop=(4, 4, 4),
        shard_index=(0, 0, 0),
        shard_counts=(2, 1, 1),
        periodic_axes=(True, False, False),
    )
    assert periodic.lower_side_kind(0) == SIDE_SIMPLE_PERIODIC
    assert periodic.upper_side_kind(0) == SIDE_SIMPLE_PERIODIC
    assert periodic.allows_regular_exchange_lower(0)
    assert periodic.allows_regular_exchange_upper(0)

    topology = ShardSpec3D(
        global_shape=(8, 4, 4),
        owned_start=(0, 0, 0),
        owned_stop=(4, 4, 4),
        shard_index=(0, 0, 0),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_PHYSICAL, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
    )
    assert topology.has_topology_lower(0)
    assert not topology.allows_regular_exchange_lower(0)
    assert topology.has_physical_upper(0) is False

    interior = ShardSpec3D(
        global_shape=(8, 4, 4),
        owned_start=(4, 0, 0),
        owned_stop=(8, 4, 4),
        shard_index=(1, 0, 0),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        side_kind_lower=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
    )
    assert interior.allows_regular_exchange_lower(0)
    assert interior.has_physical_lower(0) is False


def test_physical_ghost_filler_uses_axis_specific_dynamic_bc() -> None:
    domain = _domain(
        owned_shape=(2, 3, 4),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    field = jnp.full(layout.cell_halo_shape, -9.0)
    field = field.at[layout.owned_slices_cell].set(1.0)

    bc = LocalBoundaryFaceBC3D.empty(layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_NEUMANN),
        value_x=bc.value_x.at[0].set(4.0).at[-1].set(2.0),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )

    def weights(owned_coefficient: float, bc_coefficient: float) -> GhostFillWeights1D:
        return GhostFillWeights1D(
            owned_weights=jnp.array([[owned_coefficient]]),
            bc_weights=jnp.array([bc_coefficient]),
        )

    filler = PhysicalGhostCellFiller3D(
        dirichlet=(weights(1.0, 10.0), weights(2.0, 20.0), weights(3.0, 30.0)),
        neumann_lower=(weights(1.0, 5.0), weights(2.0, 6.0), weights(3.0, 7.0)),
        neumann_upper=(weights(1.0, 5.0), weights(2.0, 6.0), weights(3.0, 7.0)),
    )

    filled = filler(field, domain, bc)
    owned = layout.owned_slices_cell
    assert jnp.all(filled[0, owned[1], owned[2]] == 41.0)
    assert jnp.all(filled[-1, owned[1], owned[2]] == 11.0)

    # Only one-ghost-direction face slabs are supported. Halo edges/corners
    # remain untouched because current operators do not consume them.
    assert filled[0, 0, 0] == -9.0
    assert jnp.all(filled[owned[0], 0, owned[2]] == -9.0)
    assert jnp.all(filled[owned[0], owned[1], 0] == -9.0)


def test_local_periodic_topology_rule_fills_only_undecomposed_periodic_faces() -> None:
    domain = _domain(
        owned_shape=(2, 4, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, False),
    )
    layout = domain.layout
    field = jnp.full(layout.cell_halo_shape, -99.0)
    owned = jnp.arange(jnp.prod(jnp.asarray(layout.owned_shape))).reshape(layout.owned_shape)
    field = field.at[layout.owned_slices_cell].set(owned)

    filled = LocalPeriodicTopologyRule3D()(field, domain)
    h = layout.halo_width
    assert jnp.array_equal(filled[h : h + 2, 0, h : h + 2], owned[:, -1, :])
    assert jnp.array_equal(filled[h : h + 2, h + 4, h : h + 2], owned[:, 0, :])
    assert jnp.all(filled[0, :, :] == -99.0)


def test_radial_axis_regular_scalar_rule_rolls_theta_and_preserves_edges() -> None:
    base_domain = _domain(
        owned_shape=(2, 4, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, False),
    )
    spec = replace(
        base_domain.shard_spec,
        side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
    )
    domain = replace(base_domain, shard_spec=spec)
    layout = domain.layout
    field = jnp.full(layout.cell_halo_shape, -99.0)
    owned = jnp.arange(jnp.prod(jnp.asarray(layout.owned_shape))).reshape(layout.owned_shape)
    field = field.at[layout.owned_slices_cell].set(owned)

    filled = PolarAxisRegularScalarRule3D(angle_axis_name=None)(field, domain)
    expected = jnp.roll(owned[0], shift=2, axis=0)
    assert jnp.array_equal(filled[0, layout.owned_slices_cell[1], layout.owned_slices_cell[2]], expected)
    assert filled[0, 0, 0] == -99.0


def test_topology_halo_filler_applies_rules_in_order_and_roundtrips() -> None:
    domain = _domain(
        owned_shape=(2, 4, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, False),
    )
    layout = domain.layout
    field = jnp.full(layout.cell_halo_shape, -99.0)
    field = field.at[layout.owned_slices_cell].set(1.0)
    filler = TopologyHaloFiller3D(
        rules=(LocalPeriodicTopologyRule3D(fill_axes=(False, True, False)),)
    )
    leaves, treedef = jax.tree_util.tree_flatten(filler)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(restored(field, domain), filler(field, domain))


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two local devices for distributed topology exchange",
)
def test_polar_axis_regular_rule_exchanges_theta_shards() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 2, 1),
        periodic_axes=(False, True, False),
    )
    spec = replace(
        domain.shard_spec,
        side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_PHYSICAL),
    )
    domain = replace(domain, shard_spec=spec)
    layout = domain.layout
    rule = PolarAxisRegularScalarRule3D(angle_axis_name="theta")

    def make_field(theta_shard: int) -> jax.Array:
        field = jnp.full(layout.cell_halo_shape, -99.0)
        global_theta = 2 * theta_shard + jnp.arange(2)
        i = jnp.arange(2)[:, None, None]
        j = global_theta[None, :, None]
        k = jnp.arange(2)[None, None, :]
        owned = 100.0 * i + 10.0 * j + k
        return field.at[layout.owned_slices_cell].set(owned)

    fields = jax.device_put_sharded(
        [make_field(0), make_field(1)], jax.local_devices()[:2]
    )
    filled = jax.pmap(
        lambda field: rule(field, domain),
        axis_name="theta",
    )(fields)

    # A half-turn on four global angle cells maps shard 0 -> shard 1 and
    # shard 1 -> shard 0. The lower radial halo mirrors radial owned layer 0.
    assert jnp.array_equal(filled[0, 0, 1:3, 1:3], jnp.array([[20.0, 21.0], [30.0, 31.0]]))
    assert jnp.array_equal(filled[1, 0, 1:3, 1:3], jnp.array([[0.0, 1.0], [10.0, 11.0]]))


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two local devices for a collective exchange test",
)
def test_two_shard_nonperiodic_exchange_fills_only_internal_faces() -> None:
    owned_shape = (3, 2, 2)
    domain = _domain(
        owned_shape=owned_shape,
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
    )
    exchange = HaloExchange3D(
        mesh_axis_names=("x", None, None),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        exchange_axes=(True, False, False),
    )
    layout = domain.layout

    def make_field(shard: int) -> jax.Array:
        field = jnp.full(layout.cell_halo_shape, -99.0)
        owned = (100.0 + 10.0 * shard) * jnp.ones(owned_shape)
        return field.at[layout.owned_slices_cell].set(owned)

    fields = jax.device_put_sharded(
        [make_field(0), make_field(1)], jax.local_devices()[:2]
    )
    exchanged = jax.pmap(
        lambda field: exchange(field, domain),
        axis_name="x",
    )(fields)
    owned = layout.owned_slices_cell

    assert jnp.array_equal(
        exchanged[:, owned[0], owned[1], owned[2]],
        fields[:, owned[0], owned[1], owned[2]],
    )
    assert jnp.all(exchanged[0, 0, owned[1], owned[2]] == -99.0)
    assert jnp.all(exchanged[0, -1, owned[1], owned[2]] == 110.0)
    assert jnp.all(exchanged[1, 0, owned[1], owned[2]] == 100.0)
    assert jnp.all(exchanged[1, -1, owned[1], owned[2]] == -99.0)
    assert jnp.all(exchanged[:, owned[0], 0, :] == -99.0)
    assert jnp.all(exchanged[:, owned[0], owned[1], 0] == -99.0)
