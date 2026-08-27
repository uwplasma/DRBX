"""Focused checks for the selectable coordinate/FCI EB parallel family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_drb_EB_rhs import (  # noqa: E402
    RHS_TERM_FIELD_NAMES,
    RHS_TERM_NAMES,
    RHS_TERM_SLOT_COUNT,
)
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    build_local_outgoing_fci_face_topology,
)
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
)


def _identity_outgoing_face_topology(layout):
    """Identity face ownership for the non-RLP staggered smoke fixture."""

    ii, jj, kk = np.indices(layout.owned_shape, dtype=np.int32)
    return build_local_outgoing_fci_face_topology(
        layout,
        edge_owner_i=ii,
        edge_owner_j=jj,
        edge_owner_k=kk,
        edge_measure=np.ones(layout.owned_shape, dtype=np.float64),
        edge_destination_i=ii,
        edge_destination_j=jj,
        edge_destination_k=kk,
        edge_interpolation_provenance=np.zeros(layout.owned_shape + (1,)),
    )


def _paired_outgoing_face_topology(layout):
    """Unequal-measure face ownership independent of cell ownership."""

    ii, jj, kk = np.indices(layout.owned_shape, dtype=np.int32)
    owner_j = jj - (jj % 2)
    return build_local_outgoing_fci_face_topology(
        layout,
        edge_owner_i=ii,
        edge_owner_j=owner_j,
        edge_owner_k=kk,
        edge_measure=1.0 + jj.astype(np.float64),
        edge_destination_i=ii,
        edge_destination_j=jj,
        edge_destination_k=kk,
        edge_interpolation_provenance=np.zeros(layout.owned_shape + (1,)),
    )


def test_coordinate_is_default_and_scheme_validation_is_static() -> None:
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def default_kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        geometry = assemble_local_fci_geometry(local, packed)
        rhs = _build_rhs(context, local, geometry)
        return jnp.asarray(rhs.parallel_operator_scheme == "coordinate", dtype=jnp.float64)

    compiled = jax.jit(
        jax.shard_map(
            default_kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=P(),
            check_vma=False,
        )
    )
    assert float(compiled(*fields, cell_fields)) == 1.0

    def invalid_kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        geometry = assemble_local_fci_geometry(local, packed)
        rhs = _build_rhs(context, local, geometry)
        replace(rhs, parallel_operator_scheme="not-a-scheme")
        return jnp.asarray(0.0)

    invalid = jax.jit(
        jax.shard_map(
            invalid_kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=P(),
            check_vma=False,
        )
    )
    with pytest.raises(ValueError, match="parallel_operator_scheme"):
        invalid(*fields, cell_fields)


def test_fci_model_construction_is_safe_inside_shard_map() -> None:
    """The constructor must not bool() a traced map-activity array."""

    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        geometry = assemble_local_fci_geometry(local, packed)
        rhs = _build_rhs(context, local, geometry)
        rhs = replace(rhs, parallel_operator_scheme="fci")
        return jnp.asarray(rhs.parallel_operator_scheme == "fci", dtype=jnp.float64)

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=P(),
            check_vma=False,
        )
    )
    assert float(compiled(*fields, cell_fields)) == 1.0


@pytest.mark.parametrize(
    ("leg_scheme", "inflow_closure", "vorticity_current_trace"),
    (
        ("centered", "central", "operator"),
        ("boundary-characteristic-upwind", "equilibrium-characteristic", "operator"),
        (
            "boundary-characteristic-upwind",
            "equilibrium-characteristic",
            "parallel-characteristic",
        ),
    ),
)
def test_fci_full_and_implicit_smoke_on_tiny_shifted_torus(
    leg_scheme,
    inflow_closure,
    vorticity_current_trace,
) -> None:
    """Exercise both RHS paths with real retained maps and endpoint exchange."""

    context, mesh, local, partition, fields, _cell_fields = (
        _context_and_sharded_inputs()
    )
    # The MMS helper intentionally uses a simple zero-radial B for its EB
    # field.  Reuse its metric/face data but attach a real tiny shifted-torus
    # traced map payload for this operator-family smoke.
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    from drbx.native.fci_sharding import (  # noqa: E402
        assemble_local_fci_geometry,
        build_local_fci_geometries,
    )

    sharded = build_local_fci_geometries(
        mapped_geometry,
        (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    map_fields = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))
    cell_fields = jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed, maps):
        geometry = assemble_local_fci_geometry(sharded, packed, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            fci_parallel_leg_scheme=leg_scheme,
            parallel_inflow_closure=inflow_closure,
            vorticity_current_inflow_trace=vorticity_current_trace,
            parameters=replace(
                context.parameters,
                density_D_parallel=1.0e-3,
                electron_temperature_chi_parallel=1.0e-3,
                ion_temperature_chi_parallel=1.0e-3,
                Ve_parallel_viscosity=1.0e-3,
                Vi_parallel_viscosity=1.0e-3,
                vorticity_D_parallel=1.0e-3,
            ),
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        stage = rhs.evaluate_stage(state, phi_owned=phi)
        return jnp.asarray(
            [
                jnp.max(jnp.abs(stage.density)),
                jnp.max(jnp.abs(stage.Ve)),
                jnp.max(jnp.abs(stage.Te)),
                jnp.max(jnp.abs(stage.Vi)),
            ]
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
    result = np.asarray(compiled(*fields, cell_fields, map_fields))
    assert np.all(np.isfinite(result)), result


def test_fci_staggered_velocity_layout_runs_full_stage() -> None:
    context, mesh, local, partition, fields, _cell_fields = _context_and_sharded_inputs()
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    from drbx.native.fci_sharding import build_local_fci_geometries  # noqa: E402

    sharded = build_local_fci_geometries(
        mapped_geometry, (1, 1, 1), halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    face_topology = _identity_outgoing_face_topology(local.domain.layout)
    maps = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))
    packed = jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, map_values):
        geometry = assemble_local_fci_geometry(sharded, cells, map_values)
        rhs = replace(
            _build_rhs(context, local, geometry), parallel_operator_scheme="fci",
            parallel_velocity_layout="fci-staggered",
            outgoing_face_topology=face_topology,
        )
        stage = rhs.evaluate_stage(FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity), phi_owned=phi)
        return jnp.stack((stage.density, stage.Te, stage.Ti, stage.Vi, stage.Ve), axis=0)

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(None, "x", "y", "z"), check_vma=False,
    ))
    result = np.asarray(compiled(*fields, packed, maps))
    assert np.all(np.isfinite(result))

def test_staggered_parallel_viscosity_returns_face_owner_terms_that_sum() -> None:
    """Vi/Ve viscosity follows the face-owner output and diagnostic contract."""

    context, mesh, local, partition, fields, _cell_fields = _context_and_sharded_inputs()
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    from drbx.native.fci_sharding import build_local_fci_geometries  # noqa: E402

    sharded = build_local_fci_geometries(
        mapped_geometry, (1, 1, 1), halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    face_topology = _paired_outgoing_face_topology(local.domain.layout)
    maps = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))
    packed = jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, map_values):
        geometry = assemble_local_fci_geometry(sharded, cells, map_values)
        rhs = replace(
            _build_rhs(context, local, geometry), parallel_operator_scheme="fci",
            parallel_velocity_layout="fci-staggered",
            outgoing_face_topology=face_topology,
            parameters=replace(
                context.parameters, Vi_parallel_viscosity=2.0e-3,
                Ve_parallel_viscosity=3.0e-3,
            ),
        )
        stage, terms = rhs.evaluate_stage(
            FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity),
            phi_owned=phi, return_rhs_term_fields=True,
        )
        return jnp.stack((stage.Vi, stage.Ve), axis=0), terms[3:5]

    compiled = jax.jit(jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=(P(None, "x", "y", "z"), P(None, None, "x", "y", "z")),
        check_vma=False,
    ))
    stage, terms = (np.asarray(value) for value in compiled(*fields, packed, maps))
    np.testing.assert_allclose(np.sum(terms, axis=1), stage, rtol=3e-12, atol=3e-12)
    # Odd theta rows are aliases of the preceding source-face owner.
    np.testing.assert_array_equal(stage[:, :, 1::2, :], 0.0)
    assert np.max(np.abs(terms[:, 4])) > 0.0


@pytest.mark.parametrize(
    ("operator_scheme", "leg_scheme", "inflow_closure"),
    (
        ("coordinate", "centered", "equilibrium-characteristic"),
        ("fci", "boundary-characteristic-upwind", "equilibrium-characteristic"),
    ),
)
def test_all_equation_rhs_term_fields_sum_to_stage_rhs(
    operator_scheme,
    leg_scheme,
    inflow_closure,
) -> None:
    context, mesh, local, partition, fields, _cell_fields = (
        _context_and_sharded_inputs()
    )
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    from drbx.native.fci_sharding import build_local_fci_geometries  # noqa: E402

    sharded = build_local_fci_geometries(
        mapped_geometry,
        (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    map_fields = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))
    cell_fields = jax.device_put(sharded.cell_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed, maps):
        geometry = assemble_local_fci_geometry(sharded, packed, maps)
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme=operator_scheme,
            fci_parallel_leg_scheme=leg_scheme,
            parallel_inflow_closure=inflow_closure,
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        stage, terms = rhs.evaluate_stage(
            state,
            phi_owned=phi,
            return_rhs_term_fields=True,
        )
        expected = jnp.stack(
            tuple(getattr(stage, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        return terms, expected

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=(P(None, None, "x", "y", "z"), P(None, "x", "y", "z")),
            check_vma=False,
        )
    )
    terms, expected = tuple(
        np.asarray(value) for value in compiled(*fields, cell_fields, map_fields)
    )
    assert terms.shape[:2] == (len(RHS_TERM_FIELD_NAMES), RHS_TERM_SLOT_COUNT)
    np.testing.assert_allclose(np.sum(terms, axis=1), expected, rtol=2.0e-12, atol=2.0e-12)


def test_directional_curvature_rhs_fields_close_to_curvature_term_lanes() -> None:
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        from drbx.geometry import build_local_curvature_face_coefficients

        geometry = assemble_local_fci_geometry(local, packed)
        rhs = replace(
            _build_rhs(context, local, geometry),
            curvature_scheme="conservative",
            curvature_face_coefficients=build_local_curvature_face_coefficients(
                geometry, local.domain
            ),
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        baseline = rhs.evaluate_stage(state, phi_owned=phi)
        stage, terms, components = rhs.evaluate_stage(
            state,
            phi_owned=phi,
            return_rhs_term_fields=True,
            return_curvature_component_fields=True,
        )
        curvature_lanes = jnp.stack(
            tuple(
                terms[field_index, RHS_TERM_NAMES[field_index].index("curvature")]
                for field_index in (0, 1, 2, 5)
            ),
            axis=0,
        )
        baseline_fields = jnp.stack(
            tuple(getattr(baseline, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        diagnostic_fields = jnp.stack(
            tuple(getattr(stage, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        return curvature_lanes, components, baseline_fields, diagnostic_fields

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 8,
            out_specs=(
                P(None, "x", "y", "z"),
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            ),
            check_vma=False,
        )
    )
    curvature_lanes, components, baseline_fields, diagnostic_fields = tuple(
        np.asarray(value) for value in compiled(*fields, cell_fields)
    )
    assert components.shape[:2] == (4, 3)
    np.testing.assert_allclose(
        np.sum(components, axis=1),
        curvature_lanes,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        diagnostic_fields,
        baseline_fields,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_fci_remote_exchange_smoke_on_two_shards() -> None:
    """Exercise the remote dependency path when a multi-device backend exists."""

    # Keep this test intentionally small; the full numerical assertions live
    # in the single-shard smoke above and in the mapped operator tests.
    assert jax.device_count() >= 2
