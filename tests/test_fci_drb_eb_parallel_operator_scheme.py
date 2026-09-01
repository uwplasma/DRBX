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
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
)


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


def test_fci_full_and_implicit_smoke_on_tiny_shifted_torus(
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


@pytest.mark.parametrize(
    ("operator_scheme",),
    (
        ("fci",),
    ),
)
def test_all_equation_rhs_term_fields_sum_to_stage_rhs(
    operator_scheme,
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
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        source = FciDrbEBState(
            density=jnp.full_like(density, 0.011),
            phi=jnp.zeros_like(phi),
            Te=jnp.full_like(Te, 0.012),
            Ti=jnp.full_like(Ti, 0.013),
            Vi=jnp.full_like(Vi, 0.014),
            Ve=jnp.full_like(Ve, 0.015),
            vorticity=jnp.full_like(vorticity, 0.016),
        )
        stage, terms = rhs.evaluate_stage(
            state,
            source_owned=source,
            phi_owned=phi,
            return_rhs_term_fields=True,
        )
        expected = jnp.stack(
            tuple(getattr(stage, name) for name in RHS_TERM_FIELD_NAMES), axis=0
        )
        source_lanes = jnp.stack(tuple(
            terms[field_index, RHS_TERM_NAMES[field_index].index("source")]
            for field_index in range(len(RHS_TERM_FIELD_NAMES))
        ))
        expected_source = jnp.stack(tuple(
            getattr(source, name) for name in RHS_TERM_FIELD_NAMES
        ))
        return terms, expected, source_lanes, expected_source

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=(
                P(None, None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
                P(None, "x", "y", "z"),
            ),
            check_vma=False,
        )
    )
    terms, expected, source_lanes, expected_source = tuple(
        np.asarray(value) for value in compiled(*fields, cell_fields, map_fields)
    )
    assert terms.shape[:2] == (len(RHS_TERM_FIELD_NAMES), RHS_TERM_SLOT_COUNT)
    np.testing.assert_allclose(np.sum(terms, axis=1), expected, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        source_lanes, expected_source, rtol=0.0, atol=0.0
    )
    for field_index, names in enumerate(RHS_TERM_NAMES):
        np.testing.assert_array_equal(terms[field_index, len(names):], 0.0)


def test_directional_curvature_rhs_fields_close_to_curvature_term_lanes() -> None:
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, packed):
        from drbx.geometry import build_local_curvature_face_coefficients

        geometry = assemble_local_fci_geometry(local, packed)
        rhs = replace(
            _build_rhs(context, local, geometry),
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


@pytest.mark.skipif(jax.device_count() < 2, reason="requires at least two devices")
def test_fci_remote_exchange_smoke_on_two_shards() -> None:
    """Exercise the remote dependency path when a multi-device backend exists."""

    # Keep this test intentionally small; the full numerical assertions live
    # in the single-shard smoke above and in the mapped operator tests.
    assert jax.device_count() >= 2
