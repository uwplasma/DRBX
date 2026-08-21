"""Mixed cell/face owner-space invariants for the EB state container."""

from __future__ import annotations

from pathlib import Path
import sys
import inspect
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.geometry import HaloLayout3D
from drbx.native.fci_drb_EB_rhs import FciDrbEBState, LocalFciDrbEBRhs
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.native.fci_boundaries import BC_NEUMANN, LocalBoundaryFaceBC3D
from drbx.native.fci_halo import HaloExchange3D
from drbx.native.fci_operators import (
    aggregate_local_control_volume_average,
    build_local_outgoing_fci_face_topology,
    build_local_outgoing_fci_face_topology_from_geometry,
    expand_local_control_volume_owner_field,
    local_center_to_outgoing_face_average_fci_op,
    prolong_local_outgoing_fci_face_owner_field,
    restrict_local_outgoing_fci_face_field,
)
from axis_regular_operator_support import polar_fixture
from test_fci_projected_fine_grid_control_volume import _build_physical_ghost_filler, _host


def _face_topology():
    """Two distinct source cells sharing one face owner with unequal measures."""

    shape = (1, 2, 1)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    return build_local_outgoing_fci_face_topology(
        HaloLayout3D(shape, halo_width=1),
        edge_owner_i=ii,
        edge_owner_j=np.zeros(shape, dtype=np.int32),
        edge_owner_k=kk,
        edge_measure=np.asarray([[[1.0], [3.0]]]),
        edge_destination_i=ii,
        edge_destination_j=jj,
        edge_destination_k=kk,
        edge_interpolation_provenance=np.zeros(shape + (1,)),
    )


def _mixed_rhs(topology):
    """Make the ownership-only portion of the frozen model directly testable."""

    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "control_volume_geometry", None)
    object.__setattr__(rhs, "outgoing_face_topology", topology)
    object.__setattr__(rhs, "parallel_velocity_layout", "fci-staggered")
    return rhs


def _state(values):
    return FciDrbEBState(
        density=values,
        phi=values,
        Te=values,
        Ti=values,
        Vi=values,
        Ve=2.0 * values,
        vorticity=values,
    )


def test_projected_state_keeps_center_values_but_zeros_face_aliases():
    topology = _face_topology()
    rhs = _mixed_rhs(topology)
    values = jnp.asarray([[[4.0], [9.0]]])

    projected = rhs.project_galerkin_state(_state(values))

    # No cell RLP is selected here, so center fields retain both independent
    # cell values.  Vi/Ve use the distinct outgoing-face owner map instead.
    np.testing.assert_array_equal(projected.density, values)
    np.testing.assert_array_equal(projected.Ti, values)
    np.testing.assert_array_equal(projected.Vi, jnp.asarray([[[4.0], [0.0]]]))
    np.testing.assert_array_equal(projected.Ve, jnp.asarray([[[8.0], [0.0]]]))


def test_final_mixed_restriction_uses_face_measure_not_center_storage():
    topology = _face_topology()
    rhs = _mixed_rhs(topology)
    fine = jnp.asarray([[[2.0], [10.0]]])

    restricted = rhs._restrict_fine_state(_state(fine))

    # Center fields are unchanged in this no-cell-RLP fixture.  Face R_e is
    # the measure average (1*2 + 3*10)/(1+3)=8 and clears its alias slot.
    np.testing.assert_array_equal(restricted.density, fine)
    np.testing.assert_array_equal(restricted.Te, fine)
    np.testing.assert_array_equal(restricted.Vi, jnp.asarray([[[8.0], [0.0]]]))
    np.testing.assert_array_equal(restricted.Ve, jnp.asarray([[[16.0], [0.0]]]))
    assert not np.allclose(np.asarray(restricted.Vi), np.asarray(restricted.density))


def test_staggered_perpendicular_and_source_paths_keep_one_face_transfer():
    """Guard the mixed-layout routing without reimplementing the full RHS."""

    source = inspect.getsource(LocalFciDrbEBRhs.evaluate_stage)
    # Perpendicular work is centered first, while the final Vi/Ve terms are
    # mapped back to outgoing edges.  The face finalizer is still the sole R_e.
    assert "Vi_perp_halo = self._outgoing_face_to_center_halo(" in source
    assert "Ve_perp_halo = self._outgoing_face_to_center_halo(" in source
    assert "return self._prepare_cell_rlp_halo_from_fine(centered, face_bc)" in inspect.getsource(
        LocalFciDrbEBRhs._outgoing_face_to_center_halo
    )
    parallel_source = inspect.getsource(LocalFciDrbEBRhs._fci_parallel_terms)
    assert "Vi_center_halo = self._prepare_cell_rlp_halo_from_fine(" in parallel_source
    assert "Ve_center_halo = self._prepare_cell_rlp_halo_from_fine(" in parallel_source
    assert "self._project_face_force_to_cell_rlp(" not in source
    assert "Vi_perpendicular_rhs = self._cell_force_to_outgoing_face_mass_adjoint(" in source
    assert "Ve_perpendicular_rhs = self._cell_force_to_outgoing_face_mass_adjoint(" in source
    assert "value = self._cell_force_to_outgoing_face_mass_adjoint(" in parallel_source
    assert source.count("assembled = self._restrict_fine_state(") == 1
    # Vi/Ve sources are not interpreted as stale face-owner slots: they are
    # taken from the cell-owned input and reach face storage through c2f/R_e.
    assert "source_input.Vi" in source
    assert "source_input.Ve" in source
    assert "self._restrict_fine_face_field(" in source
    # All-equation diagnostics select R_e.  Parallel-only scalar advection
    # reuses the centered staggered reconstruction, rather than multiplying
    # a raw outgoing-face velocity by a face gradient.
    assert "face_owned=self.parallel_velocity_layout == \"fci-staggered\"" in source
    parallel_only = source[source.index("if self.parallel_subsystem_only:"):]
    assert "Te_parallel_advection" in parallel_only
    assert "Ti_parallel_advection" in parallel_only
    assert "vorticity_parallel_advection" in parallel_only
    assert "-Ve_parallel_value * grad_parallel_Te" not in parallel_only
    assert "-Vi_parallel_value * grad_parallel_Ti" not in parallel_only


def test_nonzero_centered_velocity_initialization_uses_c2f_then_face_restriction():
    """Exercise the production local initialization transfer on nonzero Vi/Ve.

    This is deliberately a local-model test: it covers the same c2f -> R_e
    calls used by the launcher while avoiding a full HSX driver compilation.
    """

    geometry, domain, _context, _coordinates, exchange, topology_filler, _vector, _flux = (
        polar_fixture(shape=(3, 8, 4), halo_width=1)
    )
    control_volumes = lower_polar_angular_agglomeration_geometry(
        _host((3, 8, 4)), geometry
    )
    face_topology = build_local_outgoing_fci_face_topology_from_geometry(
        control_volumes.cells, geometry.maps
    )
    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "geometry", geometry)
    object.__setattr__(rhs, "domain", domain)
    object.__setattr__(rhs, "halo_exchange", exchange)
    object.__setattr__(rhs, "topology_filler", topology_filler)
    object.__setattr__(rhs, "physical_ghost_filler", _build_physical_ghost_filler(geometry.layout))
    object.__setattr__(rhs, "control_volume_geometry", control_volumes)
    object.__setattr__(rhs, "outgoing_face_topology", face_topology)
    object.__setattr__(rhs, "parallel_velocity_layout", "fci-staggered")

    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    face_bc = replace(
        face_bc,
        kind_x=face_bc.kind_x.at[-1].set(BC_NEUMANN),
        mask_x=face_bc.mask_x.at[-1].set(True),
    )
    i, j, k = np.indices(geometry.owned_shape, dtype=np.float64)
    vi = jnp.asarray(0.4 + 0.3 * i - 0.2 * j + 0.05 * k)
    ve = jnp.asarray(-0.7 + 0.1 * i + 0.4 * j - 0.08 * k)
    scalar = jnp.asarray(1.0 + 0.2 * i + 0.5 * j + 0.03 * k)
    context = rhs._stencil_builder_context()

    def initialize_velocity(values):
        return rhs._owner_face_field(rhs._restrict_fine_face_field(
            rhs._center_owned_to_outgoing_face(values, face_bc, context)
        ))

    initialized_vi = initialize_velocity(vi)
    initialized_ve = initialize_velocity(ve)
    # Independently spell out the FCI c2f fine result and the R_e/P_e map.
    def expected(values):
        halo = rhs._prepare_fine_storage_halo(values, face_bc)
        forward, backward = rhs._fci_remote_values(halo, context)
        fine = local_center_to_outgoing_face_average_fci_op(
            halo, geometry, context=context,
            forward_remote_values=forward, backward_remote_values=backward,
        )
        owner = restrict_local_outgoing_fci_face_field(fine, face_topology)
        return owner, prolong_local_outgoing_fci_face_owner_field(owner, face_topology)

    expected_vi, materialized_vi = expected(vi)
    expected_ve, materialized_ve = expected(ve)
    np.testing.assert_allclose(np.asarray(initialized_vi), np.asarray(expected_vi), atol=3e-12)
    np.testing.assert_allclose(np.asarray(initialized_ve), np.asarray(expected_ve), atol=3e-12)
    np.testing.assert_allclose(
        np.asarray(prolong_local_outgoing_fci_face_owner_field(initialized_vi, face_topology)),
        np.asarray(materialized_vi), atol=3e-12,
    )
    np.testing.assert_allclose(
        np.asarray(prolong_local_outgoing_fci_face_owner_field(initialized_ve, face_topology)),
        np.asarray(materialized_ve), atol=3e-12,
    )
    face_aliases = ~np.asarray(face_topology.is_active_owner)
    assert np.all(np.asarray(initialized_vi)[face_aliases] == 0.0)
    assert np.all(np.asarray(initialized_ve)[face_aliases] == 0.0)

    # Scalar initial fields retain the old cell P_cR_c route and its aliases.
    initialized_scalar = rhs._owner_field(rhs._restrict_fine_field(scalar))
    expected_scalar = aggregate_local_control_volume_average(
        scalar, control_volumes.cells, domain
    )
    np.testing.assert_allclose(np.asarray(initialized_scalar), np.asarray(expected_scalar), atol=3e-12)
    np.testing.assert_allclose(
        np.asarray(expand_local_control_volume_owner_field(initialized_scalar, control_volumes.cells)),
        np.asarray(expand_local_control_volume_owner_field(expected_scalar, control_volumes.cells)),
        atol=3e-12,
    )
    assert np.all(np.asarray(initialized_scalar)[~np.asarray(control_volumes.cells.is_active_owner)] == 0.0)
