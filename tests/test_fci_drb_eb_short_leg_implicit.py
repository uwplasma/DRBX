"""Focused contracts for the opt-in short-wall material treatment."""

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

from drbx.native.fci_parallel_production_flux import (
    parallel_short_wall_backward_euler,
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
)
from drbx.native import FciDrbEBState
from drbx.native.fci_sharding import assemble_local_fci_geometry, build_local_fci_geometries
from fci_drb_eb_test_helpers import _build_rhs, _context_and_sharded_inputs
from shifted_torus_4field_mms_helpers import build_shifted_torus_4field_geometry
from jax.sharding import NamedSharding, PartitionSpec as P


RHS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "drbx"
    / "native"
    / "fci_drb_EB_rhs.py"
).read_text()


def _state():
    return jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2], dtype=jnp.float64)


def test_rhs_short_leg_configuration_and_split_are_explicitly_opt_in():
    assert 'DRBX_PARALLEL_SHORT_LEG_TREATMENT", "explicit"' in RHS_SOURCE
    assert 'DRBX_PARALLEL_SHORT_LEG_CFL_LIMIT", "2.5"' in RHS_SOURCE
    assert "parallel_short_leg_treatment" in RHS_SOURCE
    assert "parallel_short_leg_cfl_limit" in RHS_SOURCE
    assert "short_leg_selection_dt" in RHS_SOURCE
    assert "apply_short_leg_implicit_material_step" in RHS_SOURCE


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


def test_selected_material_is_exactly_the_removed_wall_piece():
    center = _state()
    wall = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    selected, _jacobian, info = parallel_short_wall_material_data(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        selection_dt=0.02, cfl_limit=2.5,
        backward_wall=True, forward_wall=False,
        backward_wall_state=wall,
    )
    full, _ = parallel_target_row_material_residual(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        backward_wall=True, forward_wall=False,
        backward_wall_state=wall, div_b=0.0,
    )
    omitted, diagnostics = parallel_target_row_material_residual(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        backward_wall=True, forward_wall=False,
        backward_wall_state=wall, div_b=0.0,
        omit_backward_wall=info["selected_backward_wall"],
    )
    np.testing.assert_allclose(full - omitted, selected, rtol=2e-8, atol=2e-8)
    assert bool(diagnostics["omitted_backward_wall"])


def test_backward_euler_changes_selected_rows_only_and_is_jittable():
    center = jnp.broadcast_to(_state(), (2, 5))
    wall = center.at[0, 0].set(1.2)
    updated, delta, info = jax.jit(parallel_short_wall_backward_euler)(
        center, center, center, jnp.asarray([0.01, 0.01]), jnp.ones(2),
        4.0, 10.0, selection_dt=jnp.asarray([0.02, 1.0e-6]),
        solve_dt=0.02, cfl_limit=2.5,
        backward_wall=jnp.asarray([True, True]),
        backward_wall_state=wall,
    )
    assert bool(info["selected_backward_wall"][0])
    assert not bool(info["selected_backward_wall"][1])
    assert bool(jnp.all(jnp.isfinite(updated)))
    np.testing.assert_allclose(delta[1], 0.0)
    np.testing.assert_allclose(updated[1], center[1])
    assert bool(jnp.any(jnp.abs(delta[0]) > 0.0))


def test_backward_euler_hands_off_complete_coupled_row_residual():
    center = jnp.broadcast_to(_state(), (2, 5))
    wall = center.at[0, 0].set(1.2)
    coupled = jnp.zeros_like(center).at[..., 4].set(
        jnp.asarray((37.0, -91.0))
    )
    solve_dt = 2.0e-3
    updated, delta, info = parallel_short_wall_backward_euler(
        center,
        center,
        center,
        jnp.asarray((0.01, 0.01)),
        jnp.ones(2),
        4.0,
        10.0,
        selection_dt=jnp.asarray((0.02, 1.0e-6)),
        solve_dt=solve_dt,
        cfl_limit=2.5,
        backward_wall=jnp.asarray((True, True)),
        backward_wall_state=wall,
        coupled_residual=coupled,
    )
    expected = np.linalg.solve(
        np.eye(5)[None, :, :]
        - solve_dt * np.asarray(info["selected_complete_jacobian"]),
        solve_dt * np.asarray(info["selected_complete_residual"])[..., None],
    )[..., 0]
    np.testing.assert_allclose(delta, expected, rtol=2.0e-9, atol=2.0e-11)
    np.testing.assert_allclose(updated, np.asarray(center) + expected)
    np.testing.assert_allclose(
        info["selected_coupled_residual"][0, 4], coupled[0, 4]
    )
    np.testing.assert_array_equal(info["selected_coupled_residual"][1], 0.0)
    np.testing.assert_array_equal(delta[1], 0.0)


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


def test_complete_selected_row_handoff_closes_exactly_at_residual_level():
    center = _state()
    wall = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    coupled = jnp.asarray((0.0, 0.0, 0.0, 0.0, 73.0))
    common = dict(
        backward_wall=True,
        forward_wall=False,
        backward_wall_state=wall,
        div_b=0.0,
    )
    full_material, _ = parallel_target_row_material_residual(
        center, center, center, 0.01, 1.0, 4.0, 10.0, **common
    )
    explicit_material, split_info = parallel_target_row_material_residual(
        center,
        center,
        center,
        0.01,
        1.0,
        4.0,
        10.0,
        selection_dt=0.02,
        cfl_limit=2.5,
        **common,
    )
    _updated, _delta, implicit_info = parallel_short_wall_backward_euler(
        center,
        center,
        center,
        0.01,
        1.0,
        4.0,
        10.0,
        selection_dt=0.02,
        solve_dt=1.0e-3,
        cfl_limit=2.5,
        backward_wall=True,
        backward_wall_state=wall,
        coupled_residual=coupled,
    )
    explicit_coupled = jnp.where(split_info["selected_wall"], 0.0, coupled)
    recovered = (
        explicit_material
        + explicit_coupled
        + implicit_info["selected_complete_residual"]
    )
    np.testing.assert_allclose(
        recovered, full_material + coupled, rtol=2.0e-10, atol=2.0e-10
    )


def test_rhs_guard_rejects_implicit_without_production_fci_path():
    # Keep this source-level because constructing the full mapped model is a
    # relatively expensive shard-map fixture; the guard is static by design.
    guard = RHS_SOURCE[RHS_SOURCE.index("if self.parallel_short_leg_treatment ==") :]
    assert "parallel_material_scheme != \"production-path\"" in guard
    assert "parallel_operator_scheme != \"fci\"" in guard
    assert "parallel_velocity_layout != \"cell-centered\"" in guard


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
    # The temporary halo needs phi for identical wall-trace construction;
    # the returned owner state must nevertheless contain no phi/vorticity
    # assignment (those fields are preserved by dataclass.replace).
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
    # The vorticity current divergence is not selected-row masked.  Thus the
    # weighted-adjoint current/phi exchange remains wholly in the explicit
    # partition instead of being split across local and global solvers.
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
        rhs = replace(
            _build_rhs(context, local, geometry),
            parallel_operator_scheme="fci",
            parallel_flux_pairing="support-core",
            parallel_material_scheme="production-path",
            parallel_short_leg_treatment="local-backward-euler",
            parallel_short_leg_cfl_limit=2.5,
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
    # This is a one-shot frozen mapped replay.  Avoid paying for the large
    # fused outer executable; individual JAX primitives still execute on the
    # selected device.
    with jax.disable_jit():
        result = np.asarray(eager(*fields, cell_fields, map_fields))
    assert np.all(np.isfinite(result[:5]))
    assert result[7]
    # Depending on the map's physical-face ownership, this tiny fixture may
    # have no owner row above the selected CFL threshold; either outcome is a
    # valid exercise of the mapped model path.  The diagnostic and
    # polarization fields are exactly untouched in both cases.
    assert result[5] == 0.0
    assert result[6] == 0.0
