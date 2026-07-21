from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

from jax_drb.geometry import (
    FCI_DEP_CUT_WALL,
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_INVALID,
    FCI_DEP_PHYSICAL_BOUNDARY,
    HaloLayout3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalCoordinateStencilRemoteDependencyTable,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciLocalDependencyTable,
    LocalFciRemoteDependencyTable,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    StencilBuilderContext,
    build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry,
)
from jax_drb.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN
from jax_drb.native.fci_boundaries import (
    LocalBoundaryConditionBuilder,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
    LocalBoundaryPreparation3D,
    LocalBoundaryRemoteDependencyTable,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalCutWallValueReconstructor3D,
)
from jax_drb.native.fci_helpers import _local_side_mask, local_physical_side_active
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalHaloClosure3D,
    LocalPeriodicTopologyRule3D,
    LocalStateAndBoundaryPreparer3D,
    PhysicalGhostCellFiller3D,
    PolarAxisRegularScalarRule3D,
    PreparedLocalState3D,
    RemoteBoundaryDependencyExchange,
    RemoteFciDependencyExchange,
    RemoteLocalStencilDependencyExchange,
    TopologyHaloFiller3D,
)
from jax_drb.native.fci_2_field_rhs import Fci2FieldState
from jax_drb.native.fci_model import FciFieldBundle


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TwoFieldFaceBCBundle(FciFieldBundle):
    density: LocalBoundaryFaceBC3D
    v_parallel: LocalBoundaryFaceBC3D
    density_background: LocalBoundaryFaceBC3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TwoFieldBoundaryDependencyBundle(FciFieldBundle):
    density: object
    v_parallel: object
    density_background: object


def _domain(
    *,
    owned_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    periodic_axes: tuple[bool, bool, bool],
    halo_width: int = 1,
    mesh_axis_names: tuple[str | None, str | None, str | None] = (None, None, None),
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
    return LocalDomain3D(
        layout=layout,
        shard_spec=spec,
        mesh_axis_names=mesh_axis_names,
    )


def _boundary_remote_table(
    *,
    dependency_kind: int = FCI_DEP_FIELD_INTERIOR,
    active: bool = True,
    source: tuple[int, int, int] = (1, 1, 1),
) -> LocalBoundaryRemoteDependencyTable:
    return LocalBoundaryRemoteDependencyTable(
        request_active=jnp.array([active], dtype=bool),
        request_dependency_kind=jnp.array([dependency_kind], dtype=jnp.int32),
        request_source_global_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((1,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((1, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.array([source[0]], dtype=jnp.int32),
        request_source_owner_local_j=jnp.array([source[1]], dtype=jnp.int32),
        request_source_owner_local_k=jnp.array([source[2]], dtype=jnp.int32),
        request_value_slot=jnp.zeros((1,), dtype=jnp.int32),
    )


def test_halo_exchange_static_config_roundtrips_through_pytree() -> None:
    exchange = HaloExchange3D(
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
        exchange_axes=(True, True, True),
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


def _linear_dirichlet_ghost_filler() -> PhysicalGhostCellFiller3D:
    dirichlet = GhostFillWeights1D(
        owned_weights=jnp.array([[-1.0]], dtype=jnp.float64),
        bc_weights=jnp.array([2.0], dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.array([[1.0]], dtype=jnp.float64),
        bc_weights=jnp.array([0.0], dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(dirichlet, dirichlet, dirichlet),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _linear_sum_face_bc(domain: LocalDomain3D) -> LocalBoundaryFaceBC3D:
    layout = domain.layout
    nx, ny, nz = layout.owned_shape
    x_centers = jnp.arange(nx, dtype=jnp.float64) + 0.5
    y_centers = jnp.arange(ny, dtype=jnp.float64) + 0.5
    z_centers = jnp.arange(nz, dtype=jnp.float64) + 0.5
    bc = LocalBoundaryFaceBC3D.empty(layout)

    value_x = bc.value_x
    value_x = value_x.at[0].set(
        y_centers[:, None] + z_centers[None, :]
    )
    value_x = value_x.at[-1].set(
        float(nx) + y_centers[:, None] + z_centers[None, :]
    )
    value_y = bc.value_y
    value_y = value_y.at[:, 0, :].set(
        x_centers[:, None] + z_centers[None, :]
    )
    value_y = value_y.at[:, -1, :].set(
        x_centers[:, None] + float(ny) + z_centers[None, :]
    )
    value_z = bc.value_z
    value_z = value_z.at[:, :, 0].set(
        x_centers[:, None] + y_centers[None, :]
    )
    value_z = value_z.at[:, :, -1].set(
        x_centers[:, None] + y_centers[None, :] + float(nz)
    )
    return replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=bc.kind_y.at[:, 0, :].set(BC_DIRICHLET).at[:, -1, :].set(
            BC_DIRICHLET
        ),
        kind_z=bc.kind_z.at[:, :, 0].set(BC_DIRICHLET).at[:, :, -1].set(
            BC_DIRICHLET
        ),
        value_x=value_x,
        value_y=value_y,
        value_z=value_z,
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
        mask_y=bc.mask_y.at[:, 0, :].set(True).at[:, -1, :].set(True),
        mask_z=bc.mask_z.at[:, :, 0].set(True).at[:, :, -1].set(True),
    )


def test_local_halo_closure_propagates_physical_faces_through_periodic_corners() -> None:
    domain = _domain(
        owned_shape=(3, 4, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, False),
    )
    layout = domain.layout
    x = jnp.arange(3, dtype=jnp.float64)[:, None, None] + 0.5
    y = jnp.arange(4, dtype=jnp.float64)[None, :, None] + 0.5
    z = jnp.arange(2, dtype=jnp.float64)[None, None, :] + 0.5
    field = jnp.full(layout.cell_halo_shape, -99.0)
    field = field.at[layout.owned_slices_cell].set(x + y + z)

    closed = LocalHaloClosure3D(
        physical_ghost_filler=_linear_dirichlet_ghost_filler(),
        topology_filler=TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),)
        ),
    )(field, domain, _linear_sum_face_bc(domain))

    h = layout.halo_width
    assert jnp.allclose(
        closed[0, 0, h : h + 2],
        closed[0, h + 3, h : h + 2],
    )
    assert jnp.allclose(
        closed[0, h + 4, h : h + 2],
        closed[0, h, h : h + 2],
    )


def test_local_halo_closure_fills_physical_edges_and_corners_by_codimension() -> None:
    owned_shape = (3, 4, 2)
    domain = _domain(
        owned_shape=owned_shape,
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    x = jnp.arange(owned_shape[0], dtype=jnp.float64)[:, None, None] + 0.5
    y = jnp.arange(owned_shape[1], dtype=jnp.float64)[None, :, None] + 0.5
    z = jnp.arange(owned_shape[2], dtype=jnp.float64)[None, None, :] + 0.5
    field = jnp.full(layout.cell_halo_shape, -99.0)
    field = field.at[layout.owned_slices_cell].set(x + y + z)

    closed = LocalHaloClosure3D(
        physical_ghost_filler=_linear_dirichlet_ghost_filler(),
    )(field, domain, _linear_sum_face_bc(domain))

    halo_coordinates = [
        jnp.arange(-0.5, extent + 0.5 + 1.0e-12, 1.0)
        for extent in owned_shape
    ]
    exact = (
        halo_coordinates[0][:, None, None]
        + halo_coordinates[1][None, :, None]
        + halo_coordinates[2][None, None, :]
    )
    assert jnp.allclose(closed, exact)


def test_local_state_and_boundary_preparer_wires_field_bundles() -> None:
    domain = _domain(
        owned_shape=(2, 3, 4),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    state_owned = Fci2FieldState(
        density=jnp.ones(layout.owned_shape),
        v_parallel=2.0 * jnp.ones(layout.owned_shape),
        density_background=3.0 * jnp.ones(layout.owned_shape),
    )
    face_bc = _TwoFieldFaceBCBundle(
        density=LocalBoundaryFaceBC3D.empty(layout),
        v_parallel=LocalBoundaryFaceBC3D.empty(layout),
        density_background=LocalBoundaryFaceBC3D.empty(layout),
    )

    def weights() -> GhostFillWeights1D:
        return GhostFillWeights1D(
            owned_weights=jnp.array([[1.0]]),
            bc_weights=jnp.array([0.0]),
        )

    weight_axes = (weights(), weights(), weights())
    ghost_filler = PhysicalGhostCellFiller3D(
        dirichlet=weight_axes,
        neumann_lower=weight_axes,
        neumann_upper=weight_axes,
    )
    builder = LocalBoundaryConditionBuilder(
        prepare_fn=lambda state, geometry, domain, cut_wall_geometry: LocalBoundaryPreparation3D(
            local_data=LocalBoundaryData3D(face_bc=face_bc)
        ),
        finalize_fn=lambda preparation, remote_values, state, geometry, domain, cut_wall_geometry: preparation.local_data,
    )
    preparer = LocalStateAndBoundaryPreparer3D(
        boundary_builder=builder,
        physical_ghost_filler=ghost_filler,
    )

    prepared = preparer(state_owned, geometry=None, domain=domain)
    assert isinstance(prepared, PreparedLocalState3D)
    assert type(prepared.state_halo) is type(state_owned)
    prepared.state_halo.assert_field_shape(layout.cell_halo_shape)
    assert prepared.boundary_data.face_bc is face_bc
    for name in state_owned.field_names():
        assert jnp.array_equal(
            getattr(prepared.state_halo, name)[layout.owned_slices_cell],
            getattr(state_owned, name),
        )


def test_remote_boundary_dependency_exchange_single_shard_field_values() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    field = jnp.arange(
        jnp.prod(jnp.asarray(layout.cell_halo_shape)),
        dtype=jnp.float64,
    ).reshape(layout.cell_halo_shape)
    state_halo = Fci2FieldState(
        density=field,
        v_parallel=2.0 * field,
        density_background=3.0 * field,
    )
    dependencies = _TwoFieldBoundaryDependencyBundle(
        density=_boundary_remote_table(source=(2, 1, 1)),
        v_parallel=LocalBoundaryRemoteDependencyTable.empty(),
        density_background=_boundary_remote_table(
            dependency_kind=FCI_DEP_INVALID,
            active=False,
        ),
    )

    values = RemoteBoundaryDependencyExchange()(
        state_halo_pre_bc=state_halo,
        dependencies=dependencies,
        domain=domain,
    )

    assert jnp.allclose(values.density, jnp.array([field[2, 1, 1]]))
    assert values.v_parallel.shape == (0,)
    assert jnp.allclose(values.density_background, jnp.array([0.0]))


def test_remote_boundary_dependency_exchange_rejects_pre_ghost_cut_wall_requests() -> None:
    domain = _domain(
        owned_shape=(1, 1, 1),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    state_halo = Fci2FieldState(
        density=jnp.zeros(layout.cell_halo_shape),
        v_parallel=jnp.zeros(layout.cell_halo_shape),
        density_background=jnp.zeros(layout.cell_halo_shape),
    )
    dependencies = _TwoFieldBoundaryDependencyBundle(
        density=_boundary_remote_table(dependency_kind=FCI_DEP_CUT_WALL),
        v_parallel=LocalBoundaryRemoteDependencyTable.empty(),
        density_background=LocalBoundaryRemoteDependencyTable.empty(),
    )

    with pytest.raises(ValueError, match="FCI_DEP_FIELD_INTERIOR"):
        RemoteBoundaryDependencyExchange()(
            state_halo_pre_bc=state_halo,
            dependencies=dependencies,
            domain=domain,
        )


def test_local_state_and_boundary_preparer_requires_exchange_for_remote_boundary_requests() -> None:
    domain = _domain(
        owned_shape=(1, 1, 1),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    state_owned = Fci2FieldState(
        density=jnp.ones(layout.owned_shape),
        v_parallel=jnp.ones(layout.owned_shape),
        density_background=jnp.ones(layout.owned_shape),
    )

    def weights() -> GhostFillWeights1D:
        return GhostFillWeights1D(
            owned_weights=jnp.array([[1.0]]),
            bc_weights=jnp.array([0.0]),
        )

    weight_axes = (weights(), weights(), weights())
    preparer = LocalStateAndBoundaryPreparer3D(
        boundary_builder=LocalBoundaryConditionBuilder(
            prepare_fn=lambda state, geometry, domain, cut_wall_geometry: LocalBoundaryPreparation3D(
                remote_dependencies=_TwoFieldBoundaryDependencyBundle(
                    density=_boundary_remote_table(),
                    v_parallel=LocalBoundaryRemoteDependencyTable.empty(),
                    density_background=LocalBoundaryRemoteDependencyTable.empty(),
                )
            ),
            finalize_fn=lambda preparation, remote_values, state, geometry, domain, cut_wall_geometry: LocalBoundaryData3D(),
        ),
        physical_ghost_filler=PhysicalGhostCellFiller3D(
            dirichlet=weight_axes,
            neumann_lower=weight_axes,
            neumann_upper=weight_axes,
        ),
    )

    with pytest.raises(ValueError, match="boundary_dependency_exchange"):
        preparer(state_owned, geometry=None, domain=domain)


def test_local_state_and_boundary_preparer_finalizes_remote_boundary_values_before_ghost_fill() -> None:
    domain = _domain(
        owned_shape=(2, 1, 1),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    state_owned = Fci2FieldState(
        density=jnp.array([[[9.0]], [[11.0]]], dtype=jnp.float64),
        v_parallel=jnp.ones(layout.owned_shape),
        density_background=jnp.ones(layout.owned_shape),
    )

    def weights(*, bc_weight: float) -> GhostFillWeights1D:
        return GhostFillWeights1D(
            owned_weights=jnp.array([[0.0]]),
            bc_weights=jnp.array([bc_weight]),
        )

    def finalize(
        preparation,
        remote_values,
        state,
        geometry,
        domain,
        cut_wall_geometry,
    ) -> LocalBoundaryData3D:
        del preparation, state, geometry, cut_wall_geometry
        density_face = LocalBoundaryFaceBC3D.empty(domain.layout)
        kind_x = density_face.kind_x.at[0, :, :].set(BC_DIRICHLET)
        value_x = density_face.value_x.at[0, :, :].set(remote_values.density[0])
        mask_x = density_face.mask_x.at[0, :, :].set(True)
        density_face = replace(
            density_face,
            kind_x=kind_x,
            value_x=value_x,
            mask_x=mask_x,
        )
        return LocalBoundaryData3D(
            face_bc=_TwoFieldFaceBCBundle(
                density=density_face,
                v_parallel=LocalBoundaryFaceBC3D.empty(domain.layout),
                density_background=LocalBoundaryFaceBC3D.empty(domain.layout),
            )
        )

    weight_axes = (
        weights(bc_weight=1.0),
        weights(bc_weight=1.0),
        weights(bc_weight=1.0),
    )
    preparer = LocalStateAndBoundaryPreparer3D(
        boundary_builder=LocalBoundaryConditionBuilder(
            prepare_fn=lambda state, geometry, domain, cut_wall_geometry: LocalBoundaryPreparation3D(
                remote_dependencies=_TwoFieldBoundaryDependencyBundle(
                    density=_boundary_remote_table(source=(1, 1, 1)),
                    v_parallel=LocalBoundaryRemoteDependencyTable.empty(),
                    density_background=LocalBoundaryRemoteDependencyTable.empty(),
                )
            ),
            finalize_fn=finalize,
        ),
        physical_ghost_filler=PhysicalGhostCellFiller3D(
            dirichlet=weight_axes,
            neumann_lower=weight_axes,
            neumann_upper=weight_axes,
        ),
        boundary_dependency_exchange=RemoteBoundaryDependencyExchange(),
    )

    prepared = preparer(state_owned, geometry=None, domain=domain)

    assert jnp.allclose(prepared.state_halo.density[0, 1, 1], 9.0)


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


def test_local_domain_mesh_axis_names_roundtrip_and_runtime_helpers() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        mesh_axis_names=("x", None, None),
    )
    leaves, treedef = jax.tree_util.tree_flatten(domain)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.mesh_axis_names == ("x", None, None)
    assert domain.runtime_shard_id(1) == 0


def test_local_physical_side_mask_respects_side_kind() -> None:
    base_domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    spec = replace(
        base_domain.shard_spec,
        side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_PHYSICAL, SIDE_PHYSICAL),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
    )
    domain = replace(base_domain, shard_spec=spec)
    assert not bool(local_physical_side_active(domain, 0, "lower"))
    assert bool(local_physical_side_active(domain, 0, "upper"))
    assert not jnp.any(_local_side_mask(domain, domain.layout, 0, "lower"))
    assert jnp.all(_local_side_mask(domain, domain.layout, 0, "upper"))


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two local devices for runtime shard helper test",
)
def test_local_domain_runtime_side_ownership_uses_spmd_index() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(2, 1, 1),
        periodic_axes=(False, False, False),
        mesh_axis_names=("x", None, None),
    )
    values = jax.pmap(
        lambda _: jnp.asarray(
            (
                domain.runtime_touches_lower(0),
                domain.runtime_touches_upper(0),
                domain.runtime_has_physical_lower(0),
                domain.runtime_has_physical_upper(0),
            )
        ),
        axis_name="x",
    )(jnp.zeros((2,)))
    assert jnp.array_equal(
        values,
        jnp.asarray(
            (
                (True, False, True, False),
                (False, True, False, True),
            )
        ),
    )


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


def _local_cut_wall_geometry(
    *,
    owner_i: jnp.ndarray,
    owner_j: jnp.ndarray,
    owner_k: jnp.ndarray,
    distance: jnp.ndarray,
    active: jnp.ndarray,
    stencil_axis: jnp.ndarray | None = None,
    stencil_side: jnp.ndarray | None = None,
    stencil_distance: jnp.ndarray | None = None,
) -> LocalCutWallGeometry3D:
    max_wall_faces = int(owner_i.size)
    zeros3 = (max_wall_faces, 3)
    zeros33 = (max_wall_faces, 3, 3)
    kwargs = {}
    if stencil_axis is not None:
        kwargs["stencil_axis"] = stencil_axis
    if stencil_side is not None:
        kwargs["stencil_side"] = stencil_side
    if stencil_distance is not None:
        kwargs["stencil_distance"] = stencil_distance
    return LocalCutWallGeometry3D(
        owner_i=owner_i,
        owner_j=owner_j,
        owner_k=owner_k,
        center=jnp.zeros(zeros3, dtype=jnp.float64),
        normal_contra=jnp.zeros(zeros3, dtype=jnp.float64),
        area_covector=jnp.zeros(zeros3, dtype=jnp.float64),
        distance=distance,
        J=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        g_contra=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), zeros33),
        g_cov=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), zeros33),
        B_contra=jnp.zeros(zeros3, dtype=jnp.float64),
        Bmag=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        sign=jnp.ones((max_wall_faces,), dtype=jnp.float64),
        active=active,
        max_wall_faces=max_wall_faces,
        **kwargs,
    )


def _local_cut_wall_value_reconstructor(
    cut_wall_geometry: LocalCutWallGeometry3D,
    *,
    neighbor_i: jnp.ndarray,
    neighbor_j: jnp.ndarray,
    neighbor_k: jnp.ndarray,
    active: jnp.ndarray,
) -> LocalCutWallValueReconstructor3D:
    return LocalCutWallValueReconstructor3D(
        cut_wall_geometry=cut_wall_geometry,
        neighbor_i=neighbor_i[:, None],
        neighbor_j=neighbor_j[:, None],
        neighbor_k=neighbor_k[:, None],
        weights=jnp.ones((neighbor_i.size, 1), dtype=jnp.float64),
        active=active,
        stencil_width=1,
        max_wall_faces=int(neighbor_i.size),
    )


def test_stencil_builder_context_accepts_local_cut_wall_objects() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    context = StencilBuilderContext(
        layout=layout,
        cut_wall_geometry=LocalCutWallGeometry3D.empty(0),
        cut_wall_bc=LocalCutWallBC3D.empty(0),
    )

    assert context.cut_wall_geometry.n_wall_faces == 0
    assert context.cut_wall_bc.n_wall_faces == 0


def test_local_cut_wall_geometry_stencil_metadata_defaults_and_roundtrips() -> None:
    geom = _local_cut_wall_geometry(
        owner_i=jnp.array([0, 1], dtype=jnp.int32),
        owner_j=jnp.array([0, 0], dtype=jnp.int32),
        owner_k=jnp.array([0, 1], dtype=jnp.int32),
        distance=jnp.array([0.5, 0.75], dtype=jnp.float64),
        active=jnp.array([True, False]),
    )

    assert jnp.array_equal(geom.stencil_axis, jnp.array([-1, -1], dtype=jnp.int32))
    assert jnp.array_equal(geom.stencil_side, jnp.zeros((2,), dtype=jnp.int32))
    assert jnp.array_equal(geom.stencil_distance, jnp.zeros((2,), dtype=jnp.float64))

    leaves, treedef = jax.tree_util.tree_flatten(geom)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(restored.stencil_axis, geom.stencil_axis)
    assert jnp.array_equal(restored.stencil_side, geom.stencil_side)
    assert jnp.array_equal(restored.stencil_distance, geom.stencil_distance)


def test_local_cut_wall_geometry_stencil_metadata_explicit_and_shape_checks() -> None:
    geom = _local_cut_wall_geometry(
        owner_i=jnp.array([0], dtype=jnp.int32),
        owner_j=jnp.array([0], dtype=jnp.int32),
        owner_k=jnp.array([0], dtype=jnp.int32),
        distance=jnp.array([0.5], dtype=jnp.float64),
        active=jnp.array([True]),
        stencil_axis=jnp.array([0], dtype=jnp.int32),
        stencil_side=jnp.array([1], dtype=jnp.int32),
        stencil_distance=jnp.array([0.25], dtype=jnp.float64),
    )

    assert int(geom.stencil_axis[0]) == 0
    assert int(geom.stencil_side[0]) == 1
    assert float(geom.stencil_distance[0]) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="stencil_axis"):
        _local_cut_wall_geometry(
            owner_i=jnp.array([0], dtype=jnp.int32),
            owner_j=jnp.array([0], dtype=jnp.int32),
            owner_k=jnp.array([0], dtype=jnp.int32),
            distance=jnp.array([0.5], dtype=jnp.float64),
            active=jnp.array([True]),
            stencil_axis=jnp.array([0, 1], dtype=jnp.int32),
        )


def test_build_local_coordinate_stencil_dependencies_from_cut_wall_geometry() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    geom = _local_cut_wall_geometry(
        owner_i=jnp.array([0, 1], dtype=jnp.int32),
        owner_j=jnp.array([1, 0], dtype=jnp.int32),
        owner_k=jnp.array([1, 0], dtype=jnp.int32),
        distance=jnp.array([0.5, 0.75], dtype=jnp.float64),
        active=jnp.array([True, True]),
        stencil_axis=jnp.array([0, -1], dtype=jnp.int32),
        stencil_side=jnp.array([1, 0], dtype=jnp.int32),
        stencil_distance=jnp.array([0.25, 0.0], dtype=jnp.float64),
    )

    dependencies = build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry(
        layout,
        geom,
    )

    assert dependencies.remote is None
    assert jnp.array_equal(
        dependencies.local.target_flat,
        jnp.array([3, 4], dtype=jnp.int32),
    )
    assert jnp.array_equal(dependencies.local.value_slot, jnp.array([0, 1]))
    assert jnp.array_equal(dependencies.local.axis, jnp.array([0, -1]))
    assert jnp.array_equal(dependencies.local.side, jnp.array([1, 0]))
    assert jnp.allclose(dependencies.local.distance, jnp.array([0.25, 0.0]))
    assert jnp.array_equal(dependencies.local.active, jnp.array([True, False]))


def test_remote_fci_dependency_exchange_populates_cut_wall_values_from_bc() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    field_halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    field_halo = field_halo.at[1, 1, 1].set(5.0)
    field_halo = field_halo.at[2, 1, 2].set(11.0)
    cut_wall_geometry = _local_cut_wall_geometry(
        owner_i=jnp.array([0, 1], dtype=jnp.int32),
        owner_j=jnp.array([0, 0], dtype=jnp.int32),
        owner_k=jnp.array([0, 1], dtype=jnp.int32),
        distance=jnp.array([0.5, 2.0], dtype=jnp.float64),
        active=jnp.array([True, True]),
    )
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.array([BC_DIRICHLET, BC_NEUMANN], dtype=jnp.int32),
        value=jnp.array([7.0, 3.0], dtype=jnp.float64),
        active=jnp.array([True, True]),
        max_wall_faces=2,
    )
    value_reconstructor = _local_cut_wall_value_reconstructor(
        cut_wall_geometry,
        neighbor_i=jnp.array([1, 2], dtype=jnp.int32),
        neighbor_j=jnp.array([1, 1], dtype=jnp.int32),
        neighbor_k=jnp.array([1, 2], dtype=jnp.int32),
        active=jnp.array([True, True]),
    )

    local = LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((1,), dtype=jnp.int32),
        source_i=jnp.zeros((1,), dtype=jnp.int32),
        source_j=jnp.zeros((1,), dtype=jnp.int32),
        source_k=jnp.zeros((1,), dtype=jnp.int32),
        weight=jnp.zeros((1,), dtype=jnp.float64),
        active=jnp.zeros((1,), dtype=bool),
    )
    remote = LocalFciRemoteDependencyTable(
        target_flat=jnp.arange(3, dtype=jnp.int32),
        weight=jnp.ones((3,), dtype=jnp.float64),
        receive_slot=jnp.arange(3, dtype=jnp.int32),
        active=jnp.ones((3,), dtype=bool),
        request_active=jnp.array([True, True, False]),
        request_dependency_kind=jnp.full((3,), FCI_DEP_CUT_WALL, dtype=jnp.int32),
        request_source_global_i=jnp.zeros((3,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((3,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((3,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((3, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((3,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.zeros((3,), dtype=jnp.int32),
        request_source_owner_local_j=jnp.zeros((3,), dtype=jnp.int32),
        request_source_owner_local_k=jnp.zeros((3,), dtype=jnp.int32),
        request_value_slot=jnp.array([0, 1, 0], dtype=jnp.int32),
    )
    direction = LocalFciDirectionMap(
        layout=layout,
        local=local,
        remote=remote,
        connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
    )

    values = RemoteFciDependencyExchange()(
        field_halo=field_halo,
        direction=direction,
        context=StencilBuilderContext(
            layout=layout,
            domain=domain,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_value_reconstructor=value_reconstructor,
        ),
        cut_wall_bc=cut_wall_bc,
    )

    assert jnp.allclose(values, jnp.array([7.0, 11.0, 0.0]))


def test_remote_fci_dependency_exchange_single_shard_values() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    n_halo_cells = (
        layout.cell_halo_shape[0]
        * layout.cell_halo_shape[1]
        * layout.cell_halo_shape[2]
    )
    field_halo = jnp.arange(
        n_halo_cells,
        dtype=jnp.float64,
    ).reshape(layout.cell_halo_shape)

    local = LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((1,), dtype=jnp.int32),
        source_i=jnp.zeros((1,), dtype=jnp.int32),
        source_j=jnp.zeros((1,), dtype=jnp.int32),
        source_k=jnp.zeros((1,), dtype=jnp.int32),
        weight=jnp.zeros((1,), dtype=jnp.float64),
        active=jnp.zeros((1,), dtype=bool),
    )
    remote = LocalFciRemoteDependencyTable(
        target_flat=jnp.arange(3, dtype=jnp.int32),
        weight=jnp.ones((3,), dtype=jnp.float64),
        receive_slot=jnp.arange(3, dtype=jnp.int32),
        active=jnp.ones((3,), dtype=bool),
        request_active=jnp.ones((3,), dtype=bool),
        request_dependency_kind=jnp.array(
            [
                FCI_DEP_FIELD_INTERIOR,
                FCI_DEP_PHYSICAL_BOUNDARY,
                FCI_DEP_CUT_WALL,
            ],
            dtype=jnp.int32,
        ),
        request_source_global_i=jnp.zeros((3,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((3,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((3,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((3, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((3,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.array([1, 0, 0], dtype=jnp.int32),
        request_source_owner_local_j=jnp.array([1, 1, 0], dtype=jnp.int32),
        request_source_owner_local_k=jnp.array([1, 1, 0], dtype=jnp.int32),
        request_value_slot=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    direction = LocalFciDirectionMap(
        layout=layout,
        local=local,
        remote=remote,
        connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
    )
    cut_wall_geometry = _local_cut_wall_geometry(
        owner_i=jnp.array([0], dtype=jnp.int32),
        owner_j=jnp.array([0], dtype=jnp.int32),
        owner_k=jnp.array([0], dtype=jnp.int32),
        distance=jnp.array([0.5], dtype=jnp.float64),
        active=jnp.array([True]),
    )
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.array([BC_DIRICHLET], dtype=jnp.int32),
        value=jnp.array([42.0], dtype=jnp.float64),
        active=jnp.array([True]),
        max_wall_faces=1,
    )
    value_reconstructor = _local_cut_wall_value_reconstructor(
        cut_wall_geometry,
        neighbor_i=jnp.array([1], dtype=jnp.int32),
        neighbor_j=jnp.array([1], dtype=jnp.int32),
        neighbor_k=jnp.array([1], dtype=jnp.int32),
        active=jnp.array([True]),
    )

    remote_values = RemoteFciDependencyExchange()(
        field_halo=field_halo,
        direction=direction,
        context=StencilBuilderContext(
            layout=layout,
            domain=domain,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_value_reconstructor=value_reconstructor,
        ),
        cut_wall_bc=cut_wall_bc,
    )

    assert jnp.allclose(
        remote_values,
        jnp.array([field_halo[1, 1, 1], field_halo[0, 1, 1], 42.0]),
    )


def test_remote_local_stencil_dependency_exchange_single_shard_cut_wall_value() -> None:
    domain = _domain(
        owned_shape=(2, 2, 2),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, False, False),
    )
    layout = domain.layout
    field_halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    field_halo = field_halo.at[1, 1, 1].set(5.0)
    cut_wall_geometry = _local_cut_wall_geometry(
        owner_i=jnp.array([0], dtype=jnp.int32),
        owner_j=jnp.array([0], dtype=jnp.int32),
        owner_k=jnp.array([0], dtype=jnp.int32),
        distance=jnp.array([0.5], dtype=jnp.float64),
        active=jnp.array([True]),
        stencil_axis=jnp.array([0], dtype=jnp.int32),
        stencil_side=jnp.array([1], dtype=jnp.int32),
        stencil_distance=jnp.array([0.25], dtype=jnp.float64),
    )
    cut_wall_bc = LocalCutWallBC3D(
        kind=jnp.array([BC_DIRICHLET], dtype=jnp.int32),
        value=jnp.array([42.0], dtype=jnp.float64),
        active=jnp.array([True]),
        max_wall_faces=1,
    )
    value_reconstructor = _local_cut_wall_value_reconstructor(
        cut_wall_geometry,
        neighbor_i=jnp.array([1], dtype=jnp.int32),
        neighbor_j=jnp.array([1], dtype=jnp.int32),
        neighbor_k=jnp.array([1], dtype=jnp.int32),
        active=jnp.array([True]),
    )
    remote = LocalCoordinateStencilRemoteDependencyTable(
        target_flat=jnp.array([0], dtype=jnp.int32),
        axis=jnp.array([0], dtype=jnp.int32),
        side=jnp.array([1], dtype=jnp.int32),
        receive_slot=jnp.array([0], dtype=jnp.int32),
        distance=jnp.array([0.25], dtype=jnp.float64),
        active=jnp.array([True]),
        request_active=jnp.array([True]),
        request_dependency_kind=jnp.array([FCI_DEP_CUT_WALL], dtype=jnp.int32),
        request_source_global_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((1,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((1, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_k=jnp.zeros((1,), dtype=jnp.int32),
        request_value_slot=jnp.zeros((1,), dtype=jnp.int32),
    )
    dependencies = LocalCoordinateStencilDependencyMap3D(
        layout=layout,
        local=LocalCoordinateStencilLocalDependencyTable.empty(),
        remote=remote,
    )

    values = RemoteLocalStencilDependencyExchange()(
        field_halo=field_halo,
        dependencies=dependencies,
        context=StencilBuilderContext(
            layout=layout,
            domain=domain,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_value_reconstructor=value_reconstructor,
        ),
        cut_wall_bc=cut_wall_bc,
    )

    assert jnp.allclose(values, jnp.array([42.0]))


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
        mesh_axis_names=("x", None, None),
    )
    exchange = HaloExchange3D(
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
