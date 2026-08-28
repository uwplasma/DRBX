"""Tests for the production-curvature component evolution ablation."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry import build_local_curvature_face_coefficients  # noqa: E402
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
from fci_drb_eb_test_helpers import (  # noqa: E402
    _build_rhs,
    _context_and_sharded_inputs,
)
from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
)
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


def test_component_selector_is_environment_backed_and_validated():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/drbx/native/fci_drb_EB_rhs.py"
    ).read_text()
    assert 'DRBX_CURVATURE_EVOLUTION_COMPONENT", "full"' in source
    assert '"full", "centered-only", "dissipation-only"' in source
    assert "request_split_diagnostics" in source
    assert "directional_centered_transfer" in source
    assert "directional_characteristic_dissipation" in source


@pytest.mark.slow
def test_mapped_component_evolution_closes_to_full_production_curvature():
    context, mesh, local, partition, fields, cell_fields = (
        _context_and_sharded_inputs()
    )
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    sharded = build_local_fci_geometries(
        mapped_geometry,
        (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    map_fields = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        common = dict(
            curvature_scheme="conservative",
            curvature_face_coefficients=build_local_curvature_face_coefficients(
                geometry, local.domain
            ),
            curvature_split_scheme="production-path",
            curvature_component_diagnostic_scheme="directional",
        )
        full = replace(_build_rhs(context, local, geometry), **common)
        centered = replace(
            full, curvature_evolution_component="centered-only"
        )
        dissipation = replace(
            full, curvature_evolution_component="dissipation-only"
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        outputs = []
        for model in (full, centered, dissipation):
            _stage, components = model.evaluate_stage(
                state, phi_owned=phi, return_curvature_component_fields=True
            )
            outputs.append(jnp.sum(components, axis=1))
        full_value, centered_value, dissipation_value = outputs
        closure_error = jnp.max(
            jnp.abs(centered_value + dissipation_value - full_value)
        )
        return jnp.asarray((closure_error, jnp.max(jnp.abs(full_value))))

    compiled = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition,) * 9,
            out_specs=P(),
            check_vma=False,
        )
    )
    closure_error, full_norm = np.asarray(
        compiled(*fields, cell_fields, map_fields)
    )
    assert full_norm > 0.0
    assert closure_error <= 2.0e-10 * max(1.0, full_norm)

