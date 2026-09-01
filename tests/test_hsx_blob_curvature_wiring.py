"""Regression tests for the static HSX curvature experiment wiring."""

import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
RHS_PATH = REPOSITORY / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_tracked_advance_uses_output_specs_and_scalar_diagnostic_halo():
    rhs_tree = _tree(RHS_PATH)
    diagnostic = _function(
        rhs_tree,
        "ion_temperature_curvature_chain_rule_diagnostics",
    )
    diagnostic_source = ast.get_source_segment(RHS_PATH.read_text(), diagnostic)
    assert diagnostic_source is not None
    assert "self._prepare_scalar_halo" in diagnostic_source
    assert "prepare_local_fci_drb_eb_state" not in diagnostic_source

    driver_tree = _tree(DRIVER_PATH)
    run_full_eb = _function(driver_tree, "run_full_eb")
    out_spec_assignment = next(
        node
        for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Assign)
        and any(
                isinstance(target, ast.Name) and target.id == "advance_out_specs"
            for target in node.targets
        )
    )
    assert isinstance(out_spec_assignment.value, ast.IfExp)
    assert isinstance(out_spec_assignment.value.test, ast.Name)
    assert out_spec_assignment.value.test.id == (
        "track_curvature_chain_rule_defect"
    )
    shard_map_calls = [
        node
        for node in ast.walk(run_full_eb)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shard_map"
    ]
    advance_call = next(
        call
        for call in shard_map_calls
        if any(
            keyword.arg == "out_specs"
            and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "advance_out_specs"
            for keyword in call.keywords
        )
    )
    assert advance_call is not None


def test_hsx_gmres_uses_tight_target_and_preserves_looser_acceptance():
    source = DRIVER_PATH.read_text()
    assert "GMRES_TARGET_TOLERANCE = 1.0e-8" in source
    build = _function(_tree(DRIVER_PATH), "build_local_eb_model")
    build_source = ast.get_source_segment(source, build)
    assert build_source is not None
    assert "tol=float(gmres_target_tolerance)" in build_source
    assert "atol=float(gmres_target_tolerance)" in build_source
    assert "acceptance_tol=float(gmres_acceptance_tolerance)" in build_source
    assert "acceptance_atol=float(gmres_acceptance_tolerance)" in build_source


def test_hsx_gmres_cli_separates_target_and_acceptance_with_legacy_alias():
    source = DRIVER_PATH.read_text()
    assert '"--gmres-target-tolerance"' in source
    assert '"--gmres-acceptance-tolerance"' in source
    assert '"--gmres-tolerance"' in source
    assert 'dest="gmres_acceptance_tolerance"' in source
    assert 'default=GMRES_TARGET_TOLERANCE' in source
    assert 'default=5.0e-5' in source
    assert 'gmres_target_tolerance=float(args.gmres_target_tolerance)' in source
    assert (
        'gmres_acceptance_tolerance=float(args.gmres_acceptance_tolerance)'
        in source
    )


def test_hsx_rk4_returns_replicated_solvax_diagnostics():
    source = DRIVER_PATH.read_text()
    run_source = ast.get_source_segment(
        source,
        _function(_tree(DRIVER_PATH), "run_full_eb"),
    )
    assert run_source is not None
    assert "_format_phi_solver_diagnostics(info)" in run_source
    assert "info.num_steps" in source
    assert "info.final_residual_rel_l2" in source
    assert "info.failed" in source
    assert "gmres_info_2" in run_source
    assert "gmres_info_3" in run_source
    assert "gmres_info_4" in run_source
    assert "gmres_info_next" in run_source
    assert "gmres_stage_diagnostics" in run_source
    assert "gmres_iterations =" in run_source
    assert "replicated_spec" in run_source
    assert "gmres_iterations=gmres_iterations_host" in run_source
    assert "gmres_relative_residual=gmres_relative_residual_host" in run_source
    assert "rejected phi inversion" in run_source
    assert "gmres-iters(avg4)=" in source
    assert "gmres-relres(max4)=" in source


def test_hsx_phi_reconstruction_exposes_solvax_gmres_info():
    rhs_source = RHS_PATH.read_text()
    rhs_tree = _tree(RHS_PATH)
    reconstruct = _function(rhs_tree, "reconstruct_phi")
    reconstruct_source = ast.get_source_segment(rhs_source, reconstruct)
    assert reconstruct_source is not None
    assert "return_diagnostics" in reconstruct_source
    assert "return_diagnostics=return_diagnostics" in reconstruct_source
    assert "SolvaxGmresInfo" in rhs_source


def test_full_eb_keeps_centered_conservative_compatibility_mode():
    rhs_source = RHS_PATH.read_text()
    helper = _function(_tree(RHS_PATH), "_conservative_curvature")
    helper_source = ast.get_source_segment(rhs_source, helper)
    assert helper_source is not None
    assert "local_curvature_conservative_op(" in helper_source
    assert 'inflow_closure="equilibrium"' not in helper_source
