"""Focused contracts for the production all-wall material treatment."""

from pathlib import Path
import inspect
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.native.fci_parallel_production_flux import (  # noqa: E402
    parallel_short_wall_backward_euler,
    parallel_target_row_material_residual,
)
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    build_local_fci_geometries,
)
from fci_drb_eb_test_helpers import _build_rhs, _context_and_sharded_inputs  # noqa: E402
from shifted_torus_4field_mms_helpers import build_shifted_torus_4field_geometry  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


RHS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "drbx"
    / "native"
    / "fci_drb_EB_rhs.py"
).read_text()


def _state():
    return jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2], dtype=jnp.float64)


def test_default_target_and_explicit_nonselected_target_are_identical():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    default, _ = parallel_target_row_material_residual(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0, div_b=0.0
    )
    explicit, _ = parallel_target_row_material_residual(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0, div_b=0.0,
        omit_backward_wall=False, omit_forward_wall=False,
    )
    np.testing.assert_array_equal(default, explicit)


def test_backward_euler_validates_coupled_shapes():
    center = _state()
    with pytest.raises(ValueError, match="coupled_residual"):
        parallel_short_wall_backward_euler(
            center, center, center, 0.1, 0.1, 4.0, 10.0,
            selection_dt=0.1, backward_wall=True,
            coupled_residual=jnp.zeros((4,)),
        )
    with pytest.raises(ValueError, match="coupled_jacobian"):
        parallel_short_wall_backward_euler(
            center, center, center, 0.1, 0.1, 4.0, 10.0,
            selection_dt=0.1, backward_wall=True,
            coupled_jacobian=jnp.zeros((4, 4)),
        )


def test_short_wall_backward_euler_propagates_nonfinite_local_solve():
    center = jnp.broadcast_to(_state(), (2, 5))
    updated, increment, info = parallel_short_wall_backward_euler(
        center,
        center,
        center,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
        4.0,
        10.0,
        selection_dt=jnp.asarray((0.02, 0.02)),
        solve_dt=jnp.asarray((jnp.nan, 1.0e-3)),
        backward_wall=jnp.asarray((True, False)),
        forward_wall=jnp.asarray((False, False)),
    )
    assert bool(jnp.any(~jnp.isfinite(increment[0])))
    assert bool(jnp.any(~jnp.isfinite(updated[0])))
    assert bool(info["implicit_solve_fallback"][0])
    assert not bool(info["implicit_finite"][0])
    assert bool(jnp.all(jnp.isfinite(increment[1])))
    np.testing.assert_array_equal(increment[1], jnp.zeros(5))
    assert not bool(info["implicit_solve_fallback"][1])
    assert bool(info["implicit_finite"][1])


def test_model_api_has_owner_increment_and_preserves_diagnostic_fields():
    source = inspect.getsource(__import__(
        "drbx.native.fci_drb_EB_rhs", fromlist=["LocalFciDrbEBRhs"]
    ).LocalFciDrbEBRhs.apply_short_leg_implicit_material_step)
    assert "self._restrict_fine_field" in source
    assert "coupled_force" in source
    assert 'parallel_terms["grad_Ti"]' in source
    assert 'parallel_terms["grad_phi"]' not in source
    assert "increment_owner" in source
    assert "selected_complete_residual_owner" in source
    assert "self.control_volume_geometry.cells.is_active_owner" in source
    update = source[source.index("updated_state =") : source.index("if not return_increment")]
    assert "vorticity=" not in update
    assert "phi=" not in update


def test_handoff_moves_ti_balance_but_keeps_current_phi_pair_together():
    evaluate = inspect.getsource(__import__(
        "drbx.native.fci_drb_EB_rhs", fromlist=["LocalFciDrbEBRhs"]
    ).LocalFciDrbEBRhs.evaluate_stage)
    assert "Ve_phi_force_term = mi_over_me * grad_parallel_phi" in evaluate
    assert "Ve_Ti_force_complete_term" in evaluate
    assert "jnp.where(selected_short_wall, 0.0, Ve_Ti_force_complete_term)" in evaluate
    assert "Ve_electrostatic_term = Ve_phi_force_term + Ve_Ti_force_term" in evaluate
    assert "vorticity_current_flux_divergence" in evaluate
    assert "jnp.where(selected_short_wall, 0.0, Ve_phi_force_term)" not in evaluate


@pytest.mark.slow
def test_mapped_model_step_changes_only_selected_material_owners():
    """Exercise the complete model method on a small real FCI map fixture."""
    context, mesh, local, partition, fields, cell_fields = _context_and_sharded_inputs()
    mapped_host = build_shifted_torus_4field_geometry(
        context.geometry.shape, construct_fci_maps=True
    )
    mapped_geometry = replace(context.geometry, maps=mapped_host.maps)
    sharded = build_local_fci_geometries(
        mapped_geometry, (1, 1, 1),
        halo_width=local.domain.layout.halo_width,
        periodic_axes=(False, True, True),
    )
    assert sharded.maps_valid
    map_fields = jax.device_put(sharded.map_fields, NamedSharding(mesh, partition))

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, cells, maps):
        geometry = assemble_local_fci_geometry(sharded, cells, maps)
        primitive_context = replace(
            context,
            parameters=replace(
                context.parameters,
                parallel_characteristic_wall_law="primitive-least-residual",
            ),
        )
        rhs = replace(
            _build_rhs(primitive_context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
            parallel_material_scheme="production-path",
            parallel_short_leg_treatment="local-backward-euler",
            parallel_short_leg_selection="all-physical-walls",
        )
        state = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        updated = rhs.apply_short_leg_implicit_material_step(
            state, solve_dt=1.0e-3, selection_dt=1.0,
            phi_owned=phi,
        )
        return jnp.asarray((
            jnp.max(jnp.abs(updated.density - density)),
            jnp.max(jnp.abs(updated.Te - Te)),
            jnp.max(jnp.abs(updated.Ti - Ti)),
            jnp.max(jnp.abs(updated.Vi - Vi)),
            jnp.max(jnp.abs(updated.Ve - Ve)),
            jnp.max(jnp.abs(updated.phi - phi)),
            jnp.max(jnp.abs(updated.vorticity - vorticity)),
            jnp.all(jnp.isfinite(updated.density)),
        ))

    eager = jax.shard_map(
        kernel, mesh=mesh, in_specs=(partition,) * 9,
        out_specs=P(), check_vma=False,
    )
    with jax.disable_jit():
        result = np.asarray(eager(*fields, cell_fields, map_fields))
    assert np.all(np.isfinite(result[:5]))
    assert result[7]
    assert result[5] == 0.0
    assert result[6] == 0.0
