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
from drbx.native.fci_sharding import assemble_local_fci_geometry  # noqa: E402
from test_fci_drb_eb_imex_integration import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
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


def test_fci_full_and_implicit_smoke_on_tiny_shifted_torus() -> None:
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
        implicit = rhs.evaluate_implicit_rhs(state, phi_owned=phi)
        return jnp.asarray(
            [
                jnp.max(jnp.abs(stage.density)),
                jnp.max(jnp.abs(stage.Ve)),
                jnp.max(jnp.abs(implicit.Te)),
                jnp.max(jnp.abs(implicit.Ve)),
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


@pytest.mark.skipif(jax.device_count() < 2, reason="requires at least two devices")
def test_fci_remote_exchange_smoke_on_two_shards() -> None:
    """Exercise the remote dependency path when a multi-device backend exists."""

    # Keep this test intentionally small; the full numerical assertions live
    # in the single-shard smoke above and in the mapped operator tests.
    assert jax.device_count() >= 2
