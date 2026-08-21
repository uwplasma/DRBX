"""Algebraic tests for the matrix-free source-face Galerkin transfers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.geometry import HaloLayout3D, LocalControlVolumeCellGeometry3D
from drbx.native.fci_face_galerkin import build_local_fci_face_galerkin_transfer
from drbx.native.fci_operators import build_local_outgoing_fci_face_topology
from drbx.native.fci_operators import build_local_outgoing_fci_face_topology_from_geometry
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.native.fci_halo import HaloExchange3D, LocalHaloClosure3D
from drbx.geometry import StencilBuilderContext
from axis_regular_operator_support import polar_fixture
from test_fci_projected_fine_grid_control_volume import _host, _build_physical_ghost_filler
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import local_center_to_outgoing_face_grad_parallel_fci_op


def _cells():
    """Two source aggregates, each containing two fine source edges."""

    shape = (1, 4, 1)
    layout = HaloLayout3D(shape, 1)
    owner_j = jnp.array([[[0], [0], [2], [2]]], dtype=jnp.int32)
    owner_i = jnp.zeros(shape, dtype=jnp.int32)
    owner_k = jnp.zeros(shape, dtype=jnp.int32)
    active = jnp.array([[[True], [False], [True], [False]]])
    raw_volume = jnp.array([[[1.0], [2.0], [1.5], [0.5]]])
    aggregate_volume = jnp.array([[[3.0], [0.0], [2.0], [0.0]]])
    zeros3 = jnp.zeros(shape + (3,))
    zeros33 = jnp.zeros(shape + (3, 3))
    return LocalControlVolumeCellGeometry3D(
        layout=layout,
        owner_i=owner_i, owner_j=owner_j, owner_k=owner_k,
        is_merged_source=~active, is_active_owner=active,
        is_aggregate_target=jnp.array([[[True], [False], [True], [False]]]),
        received_source_count=jnp.array([[[1], [0], [1], [0]]], dtype=jnp.int32),
        member_count=jnp.array([[[2], [0], [2], [0]]], dtype=jnp.int32),
        raw_volume=raw_volume, aggregate_volume=aggregate_volume,
        raw_centroid=zeros3, centroid=zeros3,
        raw_second_moment=zeros33, second_moment=zeros33,
    )


def _fine_gradient(values):
    """A non-symmetric, interpolating fine source-edge gradient with G 1=0."""

    matrix = jnp.array((
        (-1.0, 0.3, 0.7, 0.0),
        (0.0, -1.0, 0.2, 0.8),
        (0.4, 0.0, -1.0, 0.6),
        (0.9, 0.1, 0.0, -1.0),
    ))
    return (matrix @ values.reshape((-1,))).reshape(values.shape)


def _face_topology():
    """Face owners deliberately differ from the two cell aggregates."""

    shape = (1, 4, 1)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    # Fine source edges 0/2 share face owner 0; edges 1/3 share owner 1.
    # Cell owners instead group 0/1 and 2/3.
    face_owner_j = np.array([[[0], [1], [0], [1]]], dtype=np.int32)
    return build_local_outgoing_fci_face_topology(
        HaloLayout3D(shape, 1),
        edge_owner_i=ii, edge_owner_j=face_owner_j, edge_owner_k=kk,
        edge_measure=np.array([[[2.0], [3.0], [5.0], [7.0]]]),
        edge_destination_i=ii, edge_destination_j=jj, edge_destination_k=kk,
        edge_interpolation_provenance=np.zeros(shape + (2,)),
    )


def _transfer():
    return build_local_fci_face_galerkin_transfer(_cells(), _face_topology())


def _split_support_face_topology():
    """Split each cell aggregate into endpoint-support face owners.

    Cell owners group fine rows ``(0, 1)`` and ``(2, 3)``.  The face space
    deliberately does not: the two rows of each source aggregate retain
    separate coarse endpoint-support values.
    """

    shape = (1, 4, 1)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    return build_local_outgoing_fci_face_topology(
        HaloLayout3D(shape, 1),
        edge_owner_i=ii, edge_owner_j=jj, edge_owner_k=kk,
        edge_measure=np.array([[[2.0], [5.0], [3.0], [11.0]]]),
        edge_destination_i=ii, edge_destination_j=jj, edge_destination_k=kk,
        edge_interpolation_provenance=np.array(
            [[[[0.0, 1.0]], [[1.0, 2.0]], [[2.0, 1.5]], [[3.0, 4.0]]]]
        ),
    )


def _same_support_face_topology():
    """Keep same-support edges together despite different lengths/weights."""

    shape = (1, 4, 1)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    # Rows 0/1 (and 2/3) have the same endpoint-support owner, but their
    # fine quadrature weights and recorded leg lengths intentionally differ.
    owner_j = np.array([[[0], [0], [2], [2]]], dtype=np.int32)
    return build_local_outgoing_fci_face_topology(
        HaloLayout3D(shape, 1),
        edge_owner_i=ii, edge_owner_j=owner_j, edge_owner_k=kk,
        edge_measure=np.array([[[2.0], [9.0], [4.0], [13.0]]]),
        edge_destination_i=ii, edge_destination_j=jj, edge_destination_k=kk,
        edge_interpolation_provenance=np.array(
            [[[[0.0, 0.5]], [[0.0, 3.0]], [[2.0, 1.0]], [[2.0, 2.5]]]]
        ),
    )


def _dense_map(apply, shape):
    n = int(np.prod(shape))
    columns = [np.asarray(apply(jnp.eye(n)[i].reshape(shape))).reshape((-1,)) for i in range(n)]
    return np.stack(columns, axis=1)


def test_galerkin_pair_is_weighted_adjoint_and_aliases_are_zero():
    transfer = _transfer()
    shape = transfer.cells.shape
    Gc = _dense_map(lambda x: transfer.coarse_gradient(x, _fine_gradient), shape)
    Dc = _dense_map(lambda q: transfer.coarse_divergence(q, _fine_gradient), shape)
    mc = np.asarray(transfer.cell_mass).reshape((-1,))
    me = np.asarray(transfer.face_topology.aggregate_measure).reshape((-1,))
    active = np.asarray(transfer.active_owner).reshape((-1,))
    active_face = np.asarray(transfer.active_face_owner).reshape((-1,))
    np.testing.assert_allclose(np.diag(mc) @ Dc + Gc.T @ np.diag(me), 0.0, atol=2e-12)
    np.testing.assert_array_equal(Gc[~active_face], 0.0)
    np.testing.assert_array_equal(Dc[~active], 0.0)


def test_divergence_equals_restrict_fine_divergence_of_prolonged_face_field():
    transfer = _transfer()
    shape = transfer.cells.shape
    lhs = _dense_map(lambda q: transfer.coarse_divergence(q, _fine_gradient), shape)
    rhs = _dense_map(
        lambda q: transfer.cell_restrict(
            transfer.fine_divergence(transfer.face_prolong(q), _fine_gradient)
        ),
        shape,
    )
    np.testing.assert_allclose(lhs, rhs, atol=2e-12)


def test_constant_nullspace_and_nonpositive_gradient_divergence_energy():
    transfer = _transfer()
    shape = transfer.cells.shape
    one = jnp.ones(shape)
    np.testing.assert_allclose(
        np.asarray(transfer.coarse_gradient(one, _fine_gradient)), 0.0, atol=2e-12
    )
    u = jnp.array([[[1.2], [-4.0], [-0.7], [9.0]]])
    gu = transfer.coarse_gradient(u, _fine_gradient)
    dgu = transfer.coarse_divergence(gu, _fine_gradient)
    energy = jnp.sum(transfer.cell_mass * u * dgu)
    expected = -jnp.sum(transfer.face_topology.aggregate_measure * gu * gu)
    np.testing.assert_allclose(np.asarray(energy), np.asarray(expected), atol=2e-12)
    assert float(energy) <= 1.0e-12


def test_galerkin_pair_is_jittable():
    transfer = _transfer()

    @jax.jit
    def apply(u, q):
        return (
            transfer.coarse_gradient(u, _fine_gradient),
            transfer.coarse_divergence(q, _fine_gradient),
        )

    u = jnp.arange(4.0).reshape((1, 4, 1))
    q = -u
    gradient, divergence = apply(u, q)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.isfinite(np.asarray(divergence)))


def test_split_endpoint_support_face_space_preserves_two_values_and_adjoint_pair():
    """A source cell aggregate need not imply a single face degree of freedom."""

    transfer = build_local_fci_face_galerkin_transfer(
        _cells(), _split_support_face_topology()
    )
    weights = transfer.face_topology.edge_measure
    masses = transfer.face_topology.aggregate_measure
    owner_values = jnp.array([[[2.0], [-3.0], [5.0], [7.0]]])
    fine_values = jnp.array([[[1.5], [-4.0], [2.5], [9.0]]])
    prolonged = transfer.face_prolong(owner_values)
    restricted = transfer.face_restrict(fine_values)

    # P_e/R_e are weighted adjoints, and the two endpoint-support values for
    # the first cell aggregate survive independently (unlike P_e=P_c).
    np.testing.assert_allclose(
        np.asarray(jnp.sum(weights * prolonged * fine_values)),
        np.asarray(jnp.sum(masses * owner_values * restricted)), atol=2e-12,
    )
    np.testing.assert_allclose(np.asarray(prolonged[0, :2, 0]), (2.0, -3.0))

    shape = transfer.cells.shape
    Gc = _dense_map(lambda u: transfer.coarse_gradient(u, _fine_gradient), shape)
    Dc = _dense_map(lambda q: transfer.coarse_divergence(q, _fine_gradient), shape)
    mc = np.asarray(transfer.cell_mass).reshape((-1,))
    me = np.asarray(masses).reshape((-1,))
    np.testing.assert_allclose(np.diag(mc) @ Dc + Gc.T @ np.diag(me), 0.0, atol=2e-12)
    u = jnp.array([[[1.0], [0.0], [-2.0], [0.0]]])
    gu = transfer.coarse_gradient(u, _fine_gradient)
    energy = jnp.sum(transfer.cell_mass * u * transfer.coarse_divergence(gu, _fine_gradient))
    np.testing.assert_allclose(
        np.asarray(energy),
        np.asarray(-jnp.sum(masses * gu * gu)), atol=2e-12,
    )
    assert float(energy) <= 1.0e-12


def test_same_endpoint_support_uses_one_weighted_subface_not_cell_basis():
    """Different fine leg weights are retained by R_e, not split into DOFs."""

    transfer = build_local_fci_face_galerkin_transfer(
        _cells(), _same_support_face_topology()
    )
    topology = transfer.face_topology
    u = jnp.array([[[3.0], [99.0], [-1.0], [77.0]]])
    fine_gradient = _fine_gradient(transfer.cell_prolong(u))
    coarse_gradient = transfer.coarse_gradient(u, _fine_gradient)
    expected_first = (
        topology.edge_measure[0, 0, 0] * fine_gradient[0, 0, 0]
        + topology.edge_measure[0, 1, 0] * fine_gradient[0, 1, 0]
    ) / topology.aggregate_measure[0, 0, 0]

    # The first two fine edges form one coarse subface even though their
    # provenance lengths (.5 and 3) and weights differ.  R_e performs the
    # intended weighted G_f average and clears the alias slot.
    np.testing.assert_allclose(
        np.asarray(coarse_gradient[0, 0, 0]), np.asarray(expected_first), atol=2e-12
    )
    np.testing.assert_array_equal(np.asarray(coarse_gradient[0, 1, 0]), 0.0)
    assert not np.isclose(
        float(coarse_gradient[0, 0, 0]), float(fine_gradient[0, 0, 0])
    )
    assert not np.isclose(
        float(coarse_gradient[0, 0, 0]), float(fine_gradient[0, 1, 0])
    )


def test_real_mapped_angular_rlp_core_is_weighted_adjoint_without_legacy_divergence():
    """Execute the RHS helper on real maps and angular owner geometry."""

    geometry, domain, context, _coordinates, _exchange, topology_filler, _vector, _flux = (
        polar_fixture(shape=(3, 8, 4), halo_width=1)
    )
    cells = lower_polar_angular_agglomeration_geometry(_host((3, 8, 4)), geometry)
    topology = build_local_outgoing_fci_face_topology_from_geometry(
        cells.cells, geometry.maps,
    )
    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "geometry", geometry)
    object.__setattr__(rhs, "domain", domain)
    object.__setattr__(rhs, "halo_exchange", HaloExchange3D())
    object.__setattr__(rhs, "topology_filler", topology_filler)
    object.__setattr__(rhs, "physical_ghost_filler", _build_physical_ghost_filler(geometry.layout))
    object.__setattr__(rhs, "control_volume_geometry", cells)
    object.__setattr__(rhs, "outgoing_face_topology", topology)
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    core_context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    transfer, fine_gradient = rhs._fci_face_galerkin_core(core_context, face_bc)
    rng = np.random.default_rng(13)
    u = jnp.asarray(rng.normal(size=geometry.owned_shape)) * transfer.active_owner
    q = jnp.asarray(rng.normal(size=geometry.owned_shape)) * transfer.active_face_owner
    gu = transfer.coarse_gradient(u, fine_gradient)
    dq = transfer.coarse_divergence(q, fine_gradient)
    lhs = jnp.sum(transfer.face_topology.aggregate_measure * gu * q)
    rhs_inner = -jnp.sum(transfer.cell_mass * u * dq)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs_inner), atol=3e-11)

    # The actual (already closed) field gradient is restricted/prolonged
    # directly; it is not replaced by the homogeneous transpose core.
    field = jnp.asarray(rng.normal(size=geometry.owned_shape))
    field_halo = LocalHaloClosure3D(
        physical_ghost_filler=rhs.physical_ghost_filler,
        halo_exchange=rhs.halo_exchange, topology_filler=rhs.topology_filler,
    )(inject_owned_field_to_halo(field, geometry.layout), domain, face_bc)
    forward, backward = rhs._fci_remote_values(field_halo, core_context)
    actual_fine = local_center_to_outgoing_face_grad_parallel_fci_op(
        field_halo, geometry, context=core_context,
        forward_remote_values=forward, backward_remote_values=backward,
    )
    # Independently form P_e R_e from topology arrays (rather than calling
    # the transfer twice): actual nonhomogeneous field traces must reach the
    # face RHS as their weighted face-owner projection.
    topology = transfer.face_topology
    actual_np = np.asarray(actual_fine)
    weights = np.asarray(topology.edge_measure)
    masses = np.asarray(topology.aggregate_measure)
    owner_i = np.asarray(topology.edge_owner_i)
    owner_j = np.asarray(topology.edge_owner_j)
    owner_k = np.asarray(topology.edge_owner_k)
    active = np.asarray(topology.edge_active)
    owner_sum = np.zeros_like(actual_np)
    np.add.at(
        owner_sum, (owner_i, owner_j, owner_k),
        np.where(active, weights * actual_np, 0.0),
    )
    owner_value = np.divide(
        owner_sum, masses, out=np.zeros_like(owner_sum), where=masses > 0.0
    )
    expected_actual = np.where(
        active, owner_value[owner_i, owner_j, owner_k], 0.0
    )
    np.testing.assert_allclose(
        np.asarray(transfer.face_prolong(transfer.face_restrict(actual_fine))),
        expected_actual,
        atol=3e-11,
    )


def test_stiff_face_force_projection_is_cell_rlp_constant_preserving_and_sparse():
    """Completed momentum forces cannot retain angular subcell differences."""

    geometry, domain, _context, _coordinates, _exchange, _topology_filler, _vector, _flux = (
        polar_fixture(shape=(3, 8, 4), halo_width=1)
    )
    control_volumes = lower_polar_angular_agglomeration_geometry(
        _host((3, 8, 4)), geometry
    )
    topology = build_local_outgoing_fci_face_topology_from_geometry(
        control_volumes.cells, geometry.maps,
    )
    transfer = build_local_fci_face_galerkin_transfer(control_volumes.cells, topology)
    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "domain", domain)
    object.__setattr__(rhs, "control_volume_geometry", control_volumes)
    object.__setattr__(rhs, "parallel_velocity_layout", "fci-staggered")

    cells = control_volumes.cells
    owner = np.asarray(cells.is_active_owner)
    oi, oj, ok = (np.asarray(cells.owner_i), np.asarray(cells.owner_j), np.asarray(cells.owner_k))
    owner_index = tuple(np.argwhere(owner)[0])
    members = np.argwhere((oi == owner_index[0]) & (oj == owner_index[1]) & (ok == owner_index[2]))
    assert len(members) >= 2
    force = jnp.zeros(geometry.owned_shape)
    first, second = map(tuple, members[:2])
    force = force.at[first].set(2.0).at[second].set(10.0)

    projected = rhs._project_face_force_to_cell_rlp(force)
    expected = transfer.cell_prolong(transfer.cell_restrict(force))
    np.testing.assert_allclose(np.asarray(projected), np.asarray(expected), atol=2e-12)
    same_owner = (oi == owner_index[0]) & (oj == owner_index[1]) & (ok == owner_index[2])
    np.testing.assert_allclose(
        np.asarray(projected[same_owner]),
        np.asarray(projected[first]), atol=2e-12,
    )
    np.testing.assert_allclose(
        np.asarray(rhs._project_face_force_to_cell_rlp(jnp.ones_like(force))),
        np.ones(geometry.owned_shape), atol=2e-12,
    )
    restricted = transfer.cell_restrict(force)
    np.testing.assert_array_equal(np.asarray(restricted)[~owner], 0.0)
