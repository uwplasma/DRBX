"""Native canonical-driver contracts for production flux selectors."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from drbx.fci_braginskii.geometry.fci_geometry import (
    HaloLayout3D,
    LocalCurvatureFaceCoefficients3D,
)


DRIVER = Path(__file__).resolve().parents[2] / "simulate_hsx_blob.py"


def _driver_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_hsx_blob_production_selectors", DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_driver_is_tracked_at_repository_root():
    assert DRIVER.is_file()
    assert (DRIVER.parent / "pyproject.toml").is_file()
    assert (DRIVER.parent / "src" / "drbx").is_dir()


def test_parser_exposes_no_removed_production_selector_flags():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    for removed in (
        "parallel_flux_pairing",
        "parallel_boundary_pairing",
        "parallel_characteristic_wall_law",
        "parallel_short_leg_treatment",
        "parallel_short_leg_selection",
        "parallel_short_leg_cfl_limit",
        "poisson_bracket_scheme",
    ):
        assert not hasattr(args, removed)
        assert all(
            action.dest != removed for action in parser._actions
        )


def test_wall_law_metadata_is_the_fixed_physical_boundary_state_contract():
    driver = _driver_module()
    physical = driver._parallel_characteristic_wall_metadata()
    assert physical["parallel_material_wall_flux_closure"] == (
        "live-characteristic-physical-boundary-state"
    )
    assert physical["parallel_material_wall_flux_closure_source"] == (
        "fixed production configuration"
    )
    assert physical["parallel_characteristic_wall_equilibrium_reference"] is None
    assert physical["parallel_characteristic_wall_provenance"] == (
        "physical-face-trace-live-characteristic-split"
    )
    assert physical["parallel_characteristic_wall_energy_normalizer"] is None


def test_short_leg_split_is_native_to_compiled_imex_source():
    source = DRIVER.read_text(encoding="utf-8")
    assert "short_leg_selection_dt=dt" in source
    assert "model.apply_short_leg_implicit_material_step(" in source
    assert "solve_dt=gamma_dt" in source
    assert "selection_dt=dt" in source
    assert "full_imex_advance" in source
    assert "IMEX_SSP222_GAMMA" in source


def test_time_advance_exposes_true_eager_mode_and_staged_compiled_default():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    assert args.advance_execution == "staged-compiled"
    action = next(
        action for action in parser._actions if action.dest == "advance_execution"
    )
    assert tuple(action.choices) == ("compiled", "staged-compiled", "eager")
    source = DRIVER.read_text(encoding="utf-8")
    assert "with jax.disable_jit(advance_execution == \"eager\")" in source
    assert "compiled_advance = sharded_advance" in source
    assert "staged_implicit_sharded = jax.shard_map(" in source
    assert "staged_explicit_sharded = jax.shard_map(" in source
    assert "staged_phi_sharded = jax.shard_map(" in source
    assert "staged_finalize_sharded = jax.shard_map(" in source
    assert "def compile_staged_kernel(label: str, sharded_kernel" in source
    assert '"implicit+phi"' in source
    assert '"explicit-rhs"' in source
    assert '"standalone-phi"' in source
    assert '"stage-diagnostics"' in source
    # The staged device-side operations must retain the SSP222 stage algebra
    # of full_imex_advance.
    assert "stage_2_base_before_phi = current.axpy(" in source
    assert "weighted_rate = explicit_1.axpy(explicit_2, scale=1.0).axpy(" in source
    assert "next_state = current.axpy(weighted_rate, scale=dt_dynamic)" in source


def test_eager_advance_keeps_cell_centered_setup_kernels_compiled():
    source = DRIVER.read_text(encoding="utf-8")
    assert "curvature_face_setup = jax.jit(" in source
    assert "reconstruct_phi = jax.jit(" in source
    assert "jax.device_put(\n            np.asarray(value" in source


def test_restart_phi_reuse_flag_is_not_shadowed_by_setup_kernel():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    run_full_eb = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_eb"
    )
    argument_names = {
        argument.arg
        for argument in (*run_full_eb.args.args, *run_full_eb.args.kwonlyargs)
    }
    nested_function_names = {
        node.name
        for node in run_full_eb.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "reconstruct_initial_phi" in argument_names
    assert "reconstruct_initial_phi" not in nested_function_names
    assert "reconstruct_initial_phi_kernel" in nested_function_names


def test_invariant_curvature_faces_round_trip_through_cell_channels():
    driver = _driver_module()
    layout = HaloLayout3D((3, 4, 5), halo_width=2)
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=layout,
        x=jnp.arange(4 * 4 * 5, dtype=jnp.float64).reshape(4, 4, 5),
        y=(1000.0 + jnp.arange(3 * 5 * 5, dtype=jnp.float64)).reshape(3, 5, 5),
        z=(2000.0 + jnp.arange(3 * 4 * 6, dtype=jnp.float64)).reshape(3, 4, 6),
    )
    packed = driver._pack_curvature_face_coefficients(coefficients)
    assert packed.shape == (3, 4, 5, 6)
    recovered = driver._unpack_curvature_face_coefficients(packed, layout)
    for actual, expected in zip(recovered.axes, coefficients.axes, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_run_full_eb_reconstructs_phi_after_short_leg_implicit_step():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    run_full_eb = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_eb"
    )
    implicit = [
        node.lineno for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_short_leg_implicit_material_step"
    ]
    reconstruct = [
        node.lineno for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reconstruct_stage_phi"
    ]
    assert implicit and reconstruct
    assert max(implicit) < max(reconstruct)


def test_run_metadata_attributes_selectors_to_fixed_production_configuration():
    source = DRIVER.read_text(encoding="utf-8")
    for key in (
        "parallel_flux_pairing",
        "parallel_characteristic_wall_law",
        "parallel_boundary_pairing",
        "parallel_short_leg_treatment",
        "parallel_short_leg_selection",
        "parallel_material_scheme",
        "poisson_bracket_scheme",
    ):
        assert f'"{key}_source": "fixed production configuration"' in source
    assert '"parallel_characteristic_wall_law": "physical-boundary-state"' in source
    assert '"parallel_short_leg_selection": "all-physical-walls"' in source


def test_startup_announces_parallel_characteristic_wall_law():
    source = DRIVER.read_text(encoding="utf-8")
    assert "[simulation] parallel characteristic wall law:" in source
    assert "physical-boundary-state (fixed production configuration)" in source


def test_canonical_driver_uses_cell_centered_velocity_basis():
    source = DRIVER.read_text(encoding="utf-8")
    assert '"field_locations": {"Vi": "cell-center", "Ve": "cell-center"}' in source


def test_initial_owner_sparse_check_uses_current_two_argument_api():
    source = DRIVER.read_text(encoding="utf-8")
    assert "_assert_owner_sparse(initial_state, owner_host_geometry)" in source
    assert "_assert_owner_sparse(initial_state, owner_host_geometry, None)" not in source
