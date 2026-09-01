"""Focused checks for the opt-in FCI cell-centred support flux pair."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.geometry import (  # noqa: E402
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_PHYSICAL_BOUNDARY,
    LocalFciRemoteDependencyTable,
)
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    LocalFciDrbEBRhs,
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_operators import (  # noqa: E402
    local_grad_parallel_op_fci_compatible_from_q,
    local_parallel_div_b_fci_from_q_op,
    local_parallel_q_flux_div_fci_op,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
from shifted_torus_4field_mms_helpers import build_shifted_torus_4field_geometry  # noqa: E402
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)


def _mapped_fixture():
    context, mesh, local, partition, fields, _cell_fields = (
        _context_and_sharded_inputs()
    )
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    sharded = build_local_fci_geometries(
        replace(context.geometry, maps=mapped_host.maps),
        (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    return (
        context,
        mesh,
        local,
        partition,
        fields,
        jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition)),
        jax.device_put(sharded.map_fields, NamedSharding(mesh, partition)),
        sharded,
    )


def test_pair_cell_mass_uses_the_rlp_restriction_measure() -> None:
    raw_volume = jnp.asarray(
        [[[1.0, 2.0], [3.0, 5.0]], [[7.0, 11.0], [13.0, 17.0]]],
        dtype=jnp.float64,
    )
    fake_rhs = SimpleNamespace(
        control_volume_geometry=SimpleNamespace(
            cells=SimpleNamespace(raw_volume=raw_volume)
        ),
        geometry=None,
    )
    np.testing.assert_array_equal(
        np.asarray(LocalFciDrbEBRhs._fci_pair_cell_mass(fake_rhs)),
        np.asarray(raw_volume),
    )


def test_support_core_and_wall_current_phi_pair_are_weighted_adjoint() -> None:
    (
        context,
        mesh,
        local,
        partition,
        fields,
        cell_fields,
        map_fields,
        sharded,
    ) = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = rhs._face_bcs(state)
        state_halo = rhs._prepare_state_halo(state, face_bc)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, geometry, rhs.domain, face_bc, tau=rhs.parameters.tau
        )
        parallel_boundary = rhs._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary
        )
        stencil_context = rhs._stencil_builder_context()
        inverse_b_halo, inverse_b_forward, inverse_b_backward = (
            rhs._fci_prepare_inverse_b(face_bc, stencil_context)
        )
        gradient, divergence, core = rhs._fci_support_core_pair(
            face_bc=face_bc,
            context=stencil_context,
        )
        phi_gradient, current_divergence, current_target = (
            rhs._fci_current_phi_boundary_pair(
                face_bc=face_bc,
                context=stencil_context,
            )
        )
        mass = rhs._fci_pair_cell_mass()
        flux = density * Ve
        current = density * (Vi - Ve)
        lhs = jnp.sum(mass * density * divergence(flux))
        adjoint_error = jnp.abs(lhs + jnp.sum(mass * gradient(density) * flux))
        constant_error = jnp.max(jnp.abs(gradient(jnp.ones_like(density))))
        conservation_error = jnp.abs(jnp.sum(mass * divergence(flux)))
        gradient_batch_inputs = jnp.stack((density, Te, phi), axis=0)
        gradient_batch_error = jnp.max(jnp.abs(
            gradient(gradient_batch_inputs)
            - jnp.stack(
                (gradient(density), gradient(Te), gradient(phi)), axis=0
            )
        ))
        divergence_batch_inputs = jnp.stack((flux, density * (Vi - Ve), Vi, Ve))
        divergence_batch_error = jnp.max(jnp.abs(
            divergence(divergence_batch_inputs)
            - jnp.stack(
                (
                    divergence(flux),
                    divergence(density * (Vi - Ve)),
                    divergence(Vi),
                    divergence(Ve),
                )
            )
        ))
        current_phi_adjoint_error = jnp.abs(
            jnp.sum(mass * phi * current_divergence(current))
            + jnp.sum(mass * phi_gradient(phi) * current)
        )
        wall = (
            (geometry.maps.forward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY)
            | (geometry.maps.backward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY)
        )
        missing_wall_rows = jnp.sum(wall & ~current_target)

        flux_halo, forward, backward = rhs._fci_prepare_flux_q(
            flux, parallel_boundary.density_flux, stencil_context
        )
        legacy = local_parallel_q_flux_div_fci_op(
            flux_halo,
            geometry,
            context=stencil_context,
            forward_remote_q_values=forward,
            backward_remote_q_values=backward,
        )
        current_halo, current_forward, current_backward = rhs._fci_prepare_flux_q(
            current, operator_boundary.current, stencil_context
        )
        legacy_current = local_parallel_q_flux_div_fci_op(
            current_halo,
            geometry,
            context=stencil_context,
            forward_remote_q_values=current_forward,
            backward_remote_q_values=current_backward,
        )
        div_b = local_parallel_div_b_fci_from_q_op(
            inverse_b_halo,
            geometry,
            context=stencil_context,
            forward_remote_q_values=inverse_b_forward,
            backward_remote_q_values=inverse_b_backward,
        )
        def legacy_gradient(name, values, trace):
            q_halo, q_forward, q_backward = rhs._fci_prepare_flux_q(
                values, trace, stencil_context
            )
            return local_grad_parallel_op_fci_compatible_from_q(
                q_halo,
                geometry,
                context=stencil_context,
                field_owned=values,
                div_b=div_b,
                forward_remote_q_values=q_forward,
                backward_remote_q_values=q_backward,
            )
        terms = rhs._fci_parallel_terms(
            state_halo=state_halo,
            face_bc=face_bc,
            operator_boundary=operator_boundary,
            parallel_boundary=parallel_boundary,
            context=stencil_context,
            return_electron_force_diagnostics=True,
        )
        expected = divergence(flux) + jnp.where(core, 0.0, legacy)
        expected_te_gradient = gradient(Te) + jnp.where(
            core, 0.0, legacy_gradient("Te", Te, parallel_boundary.Te)
        )
        expected_phi_gradient = phi_gradient(phi) + jnp.where(
            current_target, 0.0,
            legacy_gradient("phi", phi, operator_boundary.phi)
        )
        expected_vorticity_current = current_divergence(current) + jnp.where(
            current_target, 0.0, legacy_current
        )
        component_gradients = jnp.sum(
            terms["electron_force_gradient_components"], axis=1
        )
        component_sum_error = jnp.max(jnp.abs(
            component_gradients
            - jnp.stack(
                (terms["grad_Ve"], terms["grad_phi"], terms["grad_Pe"], terms["grad_Te"]),
                axis=0,
            )
        ))
        boundary = ~core
        return jnp.asarray((
            jnp.sum(core),
            jnp.sum(boundary),
            constant_error,
            adjoint_error,
            conservation_error,
            gradient_batch_error,
            divergence_batch_error,
            current_phi_adjoint_error,
            missing_wall_rows,
            jnp.max(jnp.abs(terms["density_flux_div"] - expected)),
            jnp.max(jnp.abs(terms["grad_Te"] - expected_te_gradient)),
            jnp.max(jnp.abs(terms["grad_phi"] - expected_phi_gradient)),
            jnp.max(jnp.abs(
                terms["vorticity_current_flux_div"]
                - expected_vorticity_current
            )),
            component_sum_error,
        ))

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 9,
        out_specs=P(),
        check_vma=False,
    ))
    (
        core_count,
        boundary_count,
        constant_error,
        adjoint_error,
        conservation_error,
        gradient_batch_error,
        divergence_batch_error,
        current_phi_adjoint_error,
        missing_wall_rows,
        flux_pairing_error,
        te_pairing_error,
        phi_pairing_error,
        vorticity_current_pairing_error,
        component_sum_error,
    ) = (
        np.asarray(compiled(*fields, cell_fields, map_fields))
    )
    assert core_count > 0
    assert boundary_count > 0
    assert constant_error < 2.0e-11
    assert adjoint_error < 2.0e-10
    assert conservation_error < 2.0e-10
    assert gradient_batch_error < 2.0e-12
    assert divergence_batch_error < 2.0e-12
    assert current_phi_adjoint_error < 2.0e-10
    assert missing_wall_rows == 0
    assert flux_pairing_error < 2.0e-12
    assert te_pairing_error < 2.0e-12
    assert phi_pairing_error < 2.0e-12
    assert vorticity_current_pairing_error < 2.0e-12
    assert component_sum_error < 2.0e-12


def test_remote_radial_ghost_request_is_closed_only_by_boundary_pair() -> None:
    """Only the explicit boundary closure may transpose radial-halo samples."""

    context, mesh, local, partition, fields, cells, maps, sharded = _mapped_fixture()

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed, map_values):
        geometry = assemble_local_fci_geometry(sharded, packed, map_values)
        shape = geometry.owned_shape
        local_table = geometry.maps.forward.local
        remote = LocalFciRemoteDependencyTable(
            target_flat=jnp.asarray((0,), dtype=jnp.int32),
            weight=jnp.asarray((1.0,), dtype=jnp.float64),
            receive_slot=jnp.asarray((0,), dtype=jnp.int32),
            active=jnp.asarray((True,)),
            request_active=jnp.asarray((True,)),
            request_dependency_kind=jnp.asarray((FCI_DEP_FIELD_INTERIOR,), dtype=jnp.int32),
            request_source_global_i=jnp.asarray((0,), dtype=jnp.int32),
            request_source_global_j=jnp.asarray((0,), dtype=jnp.int32),
            request_source_global_k=jnp.asarray((0,), dtype=jnp.int32),
            request_source_shard_index=jnp.zeros((1, 3), dtype=jnp.int32),
            request_source_shard_linear=jnp.asarray((0,), dtype=jnp.int32),
            # halo width is two in this real fixture, so zero is a radial ghost.
            request_source_owner_local_i=jnp.asarray((0,), dtype=jnp.int32),
            request_source_owner_local_j=jnp.asarray((2,), dtype=jnp.int32),
            request_source_owner_local_k=jnp.asarray((2,), dtype=jnp.int32),
            request_value_slot=jnp.asarray((0,), dtype=jnp.int32),
        )
        forward = replace(
            geometry.maps.forward,
            local=replace(local_table, active=jnp.zeros_like(local_table.active)),
            remote=remote,
            target_valid=jnp.ones(shape, dtype=bool),
            endpoint_kind=jnp.full(shape, FCI_DEP_FIELD_INTERIOR, dtype=jnp.int32),
        )
        backward = replace(
            geometry.maps.backward,
            local=replace(
                geometry.maps.backward.local,
                active=jnp.zeros_like(geometry.maps.backward.local.active),
            ),
            remote=None,
            target_valid=jnp.ones(shape, dtype=bool),
            endpoint_kind=jnp.full(shape, FCI_DEP_FIELD_INTERIOR, dtype=jnp.int32),
        )
        modified = replace(
            geometry,
            maps=replace(
                geometry.maps,
                forward=forward,
                backward=backward,
                mode="remote_dependencies",
            ),
        )
        rhs = replace(
            _build_rhs(context, local, modified),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
        )
        return jnp.stack(
            (
                rhs._fci_support_core_target_mask(),
                rhs._fci_pair_target_mask(include_physical_wall=True),
            ),
            axis=0,
        )

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(None, *partition), check_vma=False,
    ))
    core, boundary_pair = np.asarray(compiled(*fields, cells, maps))
    assert not core.reshape(-1)[0]
    assert np.all(core.reshape(-1)[1:])
    assert np.all(boundary_pair)


def test_explicit_legacy_pairing_matches_the_default_fci_rhs() -> None:
    """The opt-in switch has no effect unless support-core is selected."""

    context, mesh, local, partition, fields, cell_fields, map_fields, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        base = replace(_build_rhs(context, local, geometry), parallel_operator_scheme="fci")
        legacy = replace(base, parallel_flux_pairing="legacy")
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        default_rhs = base.evaluate_stage(state, phi_owned=phi)
        legacy_rhs = legacy.evaluate_stage(state, phi_owned=phi)
        return jnp.asarray(tuple(
            jnp.max(jnp.abs(getattr(default_rhs, name) - getattr(legacy_rhs, name)))
            for name in default_rhs.field_names()
        ))

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 9,
        out_specs=P(),
        check_vma=False,
    ))
    assert np.all(np.asarray(compiled(*fields, cell_fields, map_fields)) == 0.0)
