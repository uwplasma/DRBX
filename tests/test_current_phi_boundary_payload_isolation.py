"""Regressions for the derived-current physical support closure."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
import simulate_hsx_blob as sim

from drbx.geometry import FCI_DEP_FIELD_INTERIOR, FCI_DEP_PHYSICAL_BOUNDARY
from drbx.native import FciDrbEBState
from drbx.native.fci_boundaries import BC_DIRICHLET, BC_NEUMANN
from drbx.native.fci_halo import MetricAwarePhysicalGhostCellFiller3D
from drbx.native.fci_sharding import assemble_local_fci_geometry

from fci_drb_eb_test_helpers import _build_rhs
from test_fci_drb_eb_parallel_flux_pairing import _mapped_fixture


_ROOT = Path(__file__).resolve().parents[1]
_AFTERFIX = (
    _ROOT
    / "work/boundary_load_audit/root_cause_hsx_control/root_cause_hsx"
    / "actual_current_affine_afterfix"
)


def test_derived_current_ignores_density_wall_payload_on_mapped_ghost_legs():
    """Density wall data cannot become an affine source in the current pair."""

    context, mesh, local, partition, fields, cells, maps, sharded = (
        _mapped_fixture()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed, map_values):
        geometry = assemble_local_fci_geometry(sharded, packed, map_values)
        # The shifted-torus fixture supplies real radial-ghost map support but
        # no active physical FCI endpoint.  Mark one otherwise valid active row
        # as a forward physical endpoint so the endpoint-current affine lift is
        # exercised without removing the ordinary mapped rows under test.
        endpoint_candidate = (
            geometry.active_cell_mask_owned
            & geometry.maps.forward.target_valid
            & geometry.maps.backward.target_valid
            & (geometry.maps.backward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
        )
        endpoint_flat = jnp.argmax(endpoint_candidate.reshape(-1))
        endpoint_mask = (
            jnp.arange(int(np.prod(geometry.owned_shape))) == endpoint_flat
        ).reshape(geometry.owned_shape)
        geometry = replace(
            geometry,
            maps=replace(
                geometry.maps,
                forward=replace(
                    geometry.maps.forward,
                    endpoint_kind=jnp.where(
                        endpoint_mask,
                        FCI_DEP_PHYSICAL_BOUNDARY,
                        geometry.maps.forward.endpoint_kind,
                    ),
                ),
            ),
        )
        production_model = sim.build_local_eb_model(
            geometry,
            local.domain,
            context.parameters,
            gmres_target_tolerance=1.0e-10,
            gmres_acceptance_tolerance=1.0e-8,
            gmres_max_iterations=80,
            gmres_restart=20,
            neumann_ghost_scheme="physical",
            parallel_velocity_wall_bc="neumann",
            physical_wall_model="legacy-velocity-trace",
            parallel_operator_scheme="fci",
            polarization_operator_form="conservative",
        )
        assert isinstance(
            production_model.physical_ghost_filler,
            MetricAwarePhysicalGhostCellFiller3D,
        )
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            physical_ghost_filler=production_model.physical_ghost_filler,
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        face_bc = rhs._face_bcs(state)
        stencil_context = rhs._stencil_builder_context()

        def density_template(kind, scale):
            template = face_bc.density
            return replace(
                template,
                kind_x=jnp.where(template.mask_x, kind, template.kind_x),
                kind_y=jnp.where(template.mask_y, kind, template.kind_y),
                kind_z=jnp.where(template.mask_z, kind, template.kind_z),
                value_x=jnp.where(template.mask_x, scale, template.value_x),
                value_y=jnp.where(template.mask_y, -2.0 * scale, template.value_y),
                value_z=jnp.where(template.mask_z, 3.0 * scale, template.value_z),
            )

        # The first template represents the MPE case: nonzero prescribed
        # physical-normal density derivative.  A deliberately different
        # Dirichlet template ensures both primitive values and kinds are
        # excluded from the derived-current closure.
        neumann_bundle = replace(
            face_bc, density=density_template(BC_NEUMANN, 0.375)
        )
        different_bundle = replace(
            face_bc, density=density_template(BC_DIRICHLET, -1.25)
        )
        zero_neumann_bundle = replace(
            face_bc, density=density_template(BC_NEUMANN, 0.0)
        )

        # Establish that this is a physically responsive Neumann fixture
        # before checking that the derived-current closure removes the
        # primitive density payload.  The former neutral test filler had zero
        # boundary weights and made this prerequisite vacuous.
        zero_halo = jnp.zeros(
            geometry.layout.cell_halo_shape, dtype=jnp.float64
        )
        density_ghost_response = (
            rhs.physical_ghost_filler(
                zero_halo, rhs.domain, neumann_bundle.density
            )
            - rhs.physical_ghost_filler(
                zero_halo, rhs.domain, zero_neumann_bundle.density
            )
        )
        density_ghost_response_norm = jnp.max(jnp.abs(density_ghost_response))

        gradient, divergence, target = rhs._fci_current_phi_boundary_pair(
            face_bc=neumann_bundle,
            context=stencil_context,
        )
        _, different_divergence, different_target = (
            rhs._fci_current_phi_boundary_pair(
                face_bc=different_bundle,
                context=stencil_context,
            )
        )

        zeros = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
        current = density * (Vi - Ve)
        mass = rhs._fci_pair_cell_mass()
        d0_zero = divergence(zeros)
        template_error = jnp.max(
            jnp.abs(divergence(current) - different_divergence(current))
        )
        adjoint_error = jnp.abs(
            jnp.sum(mass * phi * divergence(current))
            + jnp.sum(mass * gradient(phi) * current)
        )

        # This endpoint is deliberately injected into the stencil contract;
        # it is not presented as a material-wall simulation.
        endpoint_values = (
            jnp.full(geometry.owned_shape, 0.123, dtype=jnp.float64),
            jnp.full(geometry.owned_shape, -0.234, dtype=jnp.float64),
        )
        _, lifted_divergence, _ = rhs._fci_current_phi_boundary_pair(
            face_bc=neumann_bundle,
            context=stencil_context,
            wall_endpoint_current_values=endpoint_values,
            build_adjoint=False,
        )
        _, different_lifted_divergence, _ = rhs._fci_current_phi_boundary_pair(
            face_bc=different_bundle,
            context=stencil_context,
            wall_endpoint_current_values=endpoint_values,
            build_adjoint=False,
        )
        lift = lifted_divergence(zeros)
        lift_template_error = jnp.max(
            jnp.abs(lift - different_lifted_divergence(zeros))
        )

        h = geometry.layout.halo_width
        radial_hi = h + geometry.owned_shape[0]
        owned_size = int(np.prod(geometry.owned_shape))

        def local_radial_ghost_rows(direction):
            table = direction.local
            radial = (table.source_i < h) | (table.source_i >= radial_hi)
            targets = jnp.clip(table.target_flat, 0, owned_size - 1)
            return (
                jnp.zeros((owned_size,), dtype=jnp.int32)
                .at[targets]
                .max((table.active & radial).astype(jnp.int32))
                .reshape(geometry.owned_shape)
                .astype(bool)
            )

        ghost_supported = local_radial_ghost_rows(
            geometry.maps.forward
        ) | local_radial_ghost_rows(geometry.maps.backward)
        ordinary_mapped = (
            (geometry.maps.forward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
            & (geometry.maps.backward.endpoint_kind == FCI_DEP_FIELD_INTERIOR)
        )
        admitted_ghost_supported = ghost_supported & target & ordinary_mapped
        return jnp.asarray(
            (
                jnp.sum(admitted_ghost_supported),
                jnp.max(jnp.where(target, jnp.abs(d0_zero), 0.0)),
                template_error,
                adjoint_error,
                jnp.max(jnp.where(target, jnp.abs(lift), 0.0)),
                lift_template_error,
                jnp.sum(target != different_target),
                jnp.sum(endpoint_candidate),
                density_ghost_response_norm,
            ),
            dtype=jnp.float64,
        )

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    (
        ghost_supported_count,
        d0_zero_error,
        template_error,
        adjoint_error,
        endpoint_lift_norm,
        endpoint_lift_template_error,
        target_difference_count,
        endpoint_candidate_count,
        density_ghost_response_norm,
    ) = np.asarray(compiled(*fields, cells, maps))

    assert ghost_supported_count > 0
    assert d0_zero_error < 2.0e-12
    assert template_error < 2.0e-12
    assert adjoint_error < 2.0e-10
    assert endpoint_lift_norm > 1.0e-6
    assert endpoint_lift_template_error < 2.0e-12
    assert target_difference_count == 0
    assert endpoint_candidate_count > 0
    assert density_ghost_response_norm > 1.0e-6


def test_actual_hsx_afterfix_artifact_recounts_removed_affine_component():
    """The fixed-state HSX replay removes exactly the pre-fix affine leak."""

    report_path = _AFTERFIX / "actual_current_affine_afterfix.json"
    arrays_path = _AFTERFIX / "actual_current_affine_afterfix_arrays.npz"
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["source_change_allowed"]
    assert report["fixed_state"]["matches_pre_fix_fixture"]
    assert report["execution_limits"] == {
        "cached_geometry_model_only": True,
        "dense_reconstruction_calls": 0,
        "full_rhs_calls": 0,
        "production_run": False,
        "timestep_advances": 0,
    }
    assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == report[
        "arrays_sha256"
    ]

    with np.load(arrays_path, allow_pickle=False) as saved:
        before = saved["D0_zero_before"]
        after = saved["D0_zero_after"]
        zero_payload = saved["D0_zero_after_with_zero_density_payload"]
        removed = saved["removed_affine_component"]
        target = saved["target"]
        dependency = saved["radial_dependency"]

    assert np.count_nonzero(target) == 192
    assert np.count_nonzero(dependency) == 48
    assert np.count_nonzero(before[target]) == 8
    np.testing.assert_allclose(
        np.max(np.abs(before[target])),
        0.009991029357685087,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(after[target], 0.0)
    np.testing.assert_array_equal(after[dependency], 0.0)
    np.testing.assert_array_equal(zero_payload, after)
    np.testing.assert_array_equal(removed, before - after)
