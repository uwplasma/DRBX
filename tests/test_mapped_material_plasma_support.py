"""Regression coverage for mapped material plasma-side endpoint support."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry import (  # noqa: E402
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_INVALID,
    FCI_DEP_PHYSICAL_BOUNDARY,
    LocalFciRemoteDependencyTable,
    build_local_fci_stencil_from_field,
)
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    build_local_fci_drb_eb_operator_boundary_bundle,
)
from drbx.native.fci_parallel_production_flux import (  # noqa: E402
    parallel_characteristic_wall_data,
)
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from fci_drb_eb_test_helpers import _build_rhs  # noqa: E402
from test_fci_drb_eb_parallel_production_wiring import _mapped_fixture  # noqa: E402


def test_mapped_material_support_uses_plasma_side_under_jit():
    """A field-interior interpolation cannot consume a radial ghost payload."""

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
    if sharded.map_fields is None:
        raise AssertionError("mapped fixture did not provide FCI map fields")
    map_fields = jax.device_put(
        sharded.map_fields,
        NamedSharding(mesh, partition),
    )
    sharding = NamedSharding(mesh, partition)
    fields = tuple(jax.device_put(value, sharding) for value in fields)
    cell_fields = jax.device_put(cell_fields, sharding)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        shape = geometry.owned_shape
        n_owned = int(np.prod(shape))
        # The source owner index i=0 is the first radial ghost cell for this
        # two-cell halo.  The request remains FIELD_INTERIOR so this exercises
        # the same mapped interpolation path as a real remote target.
        remote = LocalFciRemoteDependencyTable(
            target_flat=jnp.asarray((0, 1), dtype=jnp.int32),
            weight=jnp.asarray((1.0, 1.0), dtype=jnp.float64),
            receive_slot=jnp.asarray((0, 1), dtype=jnp.int32),
            active=jnp.asarray((True, True)),
            request_active=jnp.asarray((True, True)),
            request_dependency_kind=jnp.asarray(
                (FCI_DEP_FIELD_INTERIOR, FCI_DEP_FIELD_INTERIOR), dtype=jnp.int32
            ),
            request_source_global_i=jnp.asarray((0, 0), dtype=jnp.int32),
            request_source_global_j=jnp.asarray((0, 0), dtype=jnp.int32),
            request_source_global_k=jnp.asarray((0, 1), dtype=jnp.int32),
            request_source_shard_index=jnp.zeros((2, 3), dtype=jnp.int32),
            request_source_shard_linear=jnp.asarray((0, 0), dtype=jnp.int32),
            request_source_owner_local_i=jnp.asarray((0, 2), dtype=jnp.int32),
            request_source_owner_local_j=jnp.asarray((2, 2), dtype=jnp.int32),
            request_source_owner_local_k=jnp.asarray((2, 3), dtype=jnp.int32),
            request_value_slot=jnp.asarray((0, 1), dtype=jnp.int32),
        )
        # Every target with no dependency row is explicitly invalid.  Leaving
        # those rows labeled FIELD_INTERIOR would create zero-valued fake
        # interior endpoints and make the derivative check test the malformed
        # fixture rather than the production support route.
        endpoint_kind = jnp.full((n_owned,), FCI_DEP_INVALID, dtype=jnp.int32)
        endpoint_kind = endpoint_kind.at[1].set(FCI_DEP_PHYSICAL_BOUNDARY)
        endpoint_kind = endpoint_kind.at[0].set(FCI_DEP_FIELD_INTERIOR)
        target_valid = endpoint_kind != FCI_DEP_INVALID
        forward = replace(
            geometry.maps.forward,
            local=replace(
                geometry.maps.forward.local,
                active=jnp.zeros_like(geometry.maps.forward.local.active),
            ),
            remote=remote,
            target_valid=target_valid.reshape(shape),
            endpoint_kind=endpoint_kind.reshape(shape),
        )
        geometry = replace(
            geometry,
            maps=replace(geometry.maps, forward=forward, mode="remote_dependencies"),
        )
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_boundary_pairing="characteristic-sat",
            parallel_short_leg_treatment="local-backward-euler",
            parallel_short_leg_selection="all-physical-walls",
        )
        state = FciDrbEBState(
            density, phi, Te, Ti, Vi, jnp.ones_like(Ve), vorticity
        )
        # A deliberately enormous coordinate-face velocity payload makes any
        # accidental ghost interpolation immediately visible.
        face = rhs._face_bcs(state)
        face = replace(
            face,
            Ve=replace(face.Ve, value_x=jnp.full_like(face.Ve.value_x, 1.0e9)),
        )
        state_halo = rhs._prepare_state_halo(state, face)
        operator_boundary = build_local_fci_drb_eb_operator_boundary_bundle(
            state_halo, geometry, rhs.domain, face, tau=rhs.parameters.tau
        )
        parallel_boundary = rhs._parallel_operator_boundary(
            state_halo=state_halo, operator_boundary=operator_boundary
        )
        context_local = rhs._stencil_builder_context()
        data = rhs._fci_parallel_characteristic_wall_data(
            state_halo=state_halo,
            face_bc=face,
            parallel_boundary=parallel_boundary,
            context=context_local,
            short_leg_selection_dt=1.0,
            evaluate_wall_data=True,
        )
        data_no_wall = rhs._fci_parallel_characteristic_wall_data(
            state_halo=state_halo,
            face_bc=face,
            parallel_boundary=parallel_boundary,
            context=context_local,
            short_leg_selection_dt=1.0,
            evaluate_wall_data=False,
        )

        primitive_names = ("density", "Te", "Ti", "Vi", "Ve")
        ordinary_stencils = []
        plasma_stencils = []
        owned = rhs.domain.layout.owned_slices_cell
        for name in primitive_names:
            field_halo, forward_remote, backward_remote = rhs._fci_prepare_q(
                getattr(state_halo, name)[owned],
                getattr(parallel_boundary, name),
                context_local,
            )
            ordinary_stencils.append(
                build_local_fci_stencil_from_field(
                    field_halo,
                    geometry,
                    context_local,
                    forward_remote_values=forward_remote,
                    backward_remote_values=backward_remote,
                )
            )
            plasma_stencils.append(
                rhs._fci_plasma_side_stencil(
                    getattr(state_halo, name)[owned], getattr(face, name), context_local
                )
            )
        ordinary_plus = jnp.stack(tuple(stencil.plus for stencil in ordinary_stencils), axis=-1)
        ordinary_minus = jnp.stack(tuple(stencil.minus for stencil in ordinary_stencils), axis=-1)
        ordinary_center = jnp.stack(tuple(stencil.center for stencil in ordinary_stencils), axis=-1)
        ordinary_wall = parallel_characteristic_wall_data(
            ordinary_center,
            ordinary_minus,
            ordinary_plus,
            ordinary_stencils[0].dx_min,
            ordinary_stencils[0].dx_plus,
            rhs.parameters.tau,
            rhs.parameters.mi_over_me,
            selection_dt=1.0,
            parallel_short_leg_selection="all-physical-walls",
            backward_wall=data["backward_wall"],
            forward_wall=data["forward_wall"],
            backward_wall_state=ordinary_minus,
            forward_wall_state=ordinary_plus,
            parallel_characteristic_wall_law=rhs.parameters.parallel_characteristic_wall_law,
        )
        interior = geometry.maps.forward.endpoint_kind == FCI_DEP_FIELD_INTERIOR
        physical = geometry.maps.forward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY
        mapped_ve = data["primitive_stencils"][4].plus
        ordinary_ve = ordinary_stencils[4].plus
        plasma_ve = plasma_stencils[4].plus
        physical_endpoint_delta = jnp.max(
            jnp.where(physical, jnp.abs(mapped_ve - ordinary_ve), 0.0)
        )
        derivative_delta = jnp.max(
            jnp.abs(
                data["primitive_stencils"][4].derivative_center_weight
                - ordinary_stencils[4].derivative_center_weight
            )
        )
        wall_mask_delta = jnp.max(
            jnp.abs(
                data["wall_data"]["selected_forward_wall"].astype(jnp.int32)
                - ordinary_wall["selected_forward_wall"].astype(jnp.int32)
            )
        )
        def plasma_plus(value_owned):
            return rhs._fci_plasma_side_stencil(
                value_owned, face.Ve, context_local
            ).plus

        _, plasma_jvp = jax.jvp(
            plasma_plus,
            (getattr(state_halo, "Ve")[owned],),
            (jnp.ones(shape),),
        )
        interior_jvp_error = jnp.max(
            jnp.where(interior, jnp.abs(plasma_jvp - 1.0), 0.0)
        )
        selected_wall_count = jnp.count_nonzero(
            data["wall_data"]["selected_forward_wall"]
        )
        return jnp.asarray(
            (
                jnp.count_nonzero(interior),
                jnp.count_nonzero(physical),
                jnp.max(jnp.where(interior, jnp.abs(ordinary_ve - plasma_ve), 0.0)),
                jnp.max(jnp.where(interior, jnp.abs(mapped_ve - plasma_ve), 0.0)),
                physical_endpoint_delta,
                derivative_delta,
                wall_mask_delta,
                jnp.max(jnp.abs(data["center"] - data_no_wall["center"])),
                jnp.asarray(data_no_wall["wall_data"] is None, dtype=jnp.float64),
                interior_jvp_error,
                selected_wall_count,
            )
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
        interior_count,
        physical_count,
        ghost_delta,
        mapped_error,
        physical_endpoint_delta,
        derivative_delta,
        wall_mask_delta,
        center_delta,
        no_wall,
        interior_jvp_error,
        selected_wall_count,
    ) = np.asarray(compiled(*fields, cell_fields, map_fields))
    assert interior_count > 0
    assert physical_count > 0
    assert ghost_delta > 1.0e8
    assert mapped_error < 1.0e-10
    assert physical_endpoint_delta < 1.0e-10
    assert derivative_delta == 0.0
    assert wall_mask_delta == 0.0
    assert center_delta == 0.0
    assert no_wall == 1.0
    assert interior_jvp_error < 1.0e-10
    assert selected_wall_count > 0
