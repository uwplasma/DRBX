"""Regression tests for the static HSX curvature experiment wiring."""

import ast
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
RHS_PATH = REPOSITORY / "src" / "drbx" / "fci_braginskii" / "native" / "fci_drb_EB_rhs.py"
DRIVER_PATH = REPOSITORY / "src" / "drbx" / "fci_braginskii" / "run.py"
DRIVER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "drbx" / "fci_braginskii" / "run.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_advance_wires_out_specs_to_shard_map():
    driver_tree = _tree(DRIVER_SOURCE)
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
    assert out_spec_assignment is not None
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
    source = DRIVER_SOURCE.read_text()
    assert "GMRES_TARGET_TOLERANCE = 1.0e-8" in source
    build = _function(_tree(DRIVER_SOURCE), "build_local_eb_model")
    build_source = ast.get_source_segment(source, build)
    assert build_source is not None
    assert "tol=float(gmres_target_tolerance)" in build_source
    assert "atol=float(gmres_target_tolerance)" in build_source
    assert "acceptance_tol=float(gmres_acceptance_tolerance)" in build_source
    assert "acceptance_atol=float(gmres_acceptance_tolerance)" in build_source




def test_hsx_imex_returns_replicated_solvax_diagnostics():
    source = DRIVER_SOURCE.read_text()
    run_source = ast.get_source_segment(
        source,
        _function(_tree(DRIVER_SOURCE), "run_full_eb"),
    )
    assert run_source is not None
    assert "_format_phi_solver_diagnostics(info)" in run_source
    assert "info.num_steps" in source
    assert "info.final_residual_rel_l2" in source
    assert "info.failed" in source
    assert "gmres_info_2" in source
    assert "gmres_info_next" in source
    assert "gmres_stage_diagnostics" in run_source
    assert "gmres_iterations =" in run_source
    assert "replicated_spec" in run_source
    assert "gmres_iterations=gmres_iterations_host" in run_source
    assert "gmres_relative_residual=gmres_relative_residual_host" in run_source
    assert "rejected phi inversion" in source
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


def test_step_failure_reports_the_first_rejected_property():
    import numpy as np

    from drbx.fci_braginskii import run

    names = ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    healthy = np.zeros((7, 3))
    healthy[[0, 2, 3], 0], healthy[[0, 2, 3], 1] = 0.9, 1.1  # min, max of n, Te, Ti
    stages = np.zeros((4, 7, 4))
    stages[:, [0, 2, 3], 0] = 0.9
    gmres = np.zeros((4, 6))

    def failure(diagnostics=healthy, rk=stages, solves=gmres):
        return run._step_failure(3, names, diagnostics, rk, solves, "state")

    assert failure() is None
    bad = stages.copy()
    bad[1, 2, 0] = -1.0e-3
    assert failure(rk=bad)[1] == "invalid-imex-ssp222-stage"
    rejected = gmres.copy()
    rejected[2, 2] = 1.0
    assert failure(solves=rejected)[1] == "unaccepted-phi-inversion"
    nonfinite = healthy.copy()
    nonfinite[3, 1] = np.inf
    assert failure(diagnostics=nonfinite)[1] == "nonfinite-eb-state"
    negative = healthy.copy()
    negative[0, 0] = 0.0
    assert failure(diagnostics=negative)[1] == "nonpositive-eb-state"


def test_wall_step_vector_pairs_applied_and_wall_face_increments():
    import numpy as np

    from drbx.fci_braginskii import run

    ledger = np.arange(20.0)
    step, applied, face = run._wall_step_vector(ledger, 0.5)
    assert step.shape == (10,)
    np.testing.assert_allclose(applied, 0.25 * (ledger[0:4] + ledger[8:12]))
    np.testing.assert_allclose(face, 0.25 * (ledger[4:8] + ledger[12:16]))
    np.testing.assert_allclose(step[0:8:2], applied)  # (applied, face) interleaved per conserved quantity
    np.testing.assert_allclose(step[1:8:2], face)
    np.testing.assert_allclose(step[8:], 0.25 * np.array([ledger[16] + ledger[17], ledger[18] + ledger[19]]))
