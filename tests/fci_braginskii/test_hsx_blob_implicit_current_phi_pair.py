"""Regression tests for the implicit current/phi coupled potential solve.

Covers the driver's default-on CLI wiring, the two ``run_full_eb`` call
sites that must dispatch between the coupled solve and the ordinary
standalone potential solve, the rho* consistency in
``LocalFciDrbEBRhs``, and the ``extra_operator`` plumbing in
``LocalPerpLaplacianInverseSolver``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"
DRIVER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "drbx" / "fci_braginskii" / "run.py"
RHS_PATH = REPOSITORY / "src" / "drbx" / "fci_braginskii" / "native" / "fci_drb_EB_rhs.py"
OPERATORS_PATH = (
    REPOSITORY / "src" / "drbx" / "fci_braginskii" / "native" / "fci_operators.py"
)


def _driver_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_hsx_blob_implicit_current_phi_pair", DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_parser_defaults_enable_implicit_pair_at_the_validated_rho_star():
    driver = _driver_module()
    parser = driver._build_parser()
    args = parser.parse_args(())
    assert args.rho_star == 5.0e-4
    assert args.implicit_current_phi_pair is True

    disabled = parser.parse_args(("--no-implicit-current-phi-pair",))
    assert disabled.implicit_current_phi_pair is False

    enabled_explicit = parser.parse_args(("--implicit-current-phi-pair",))
    assert enabled_explicit.implicit_current_phi_pair is True


def test_library_entry_points_default_the_flag_off():
    tree = _tree(DRIVER_SOURCE)
    for name in ("build_local_eb_model", "run_full_eb"):
        function = _function(tree, name)
        pair_argument = next(
            argument
            for argument in function.args.kwonlyargs
            if argument.arg == "implicit_current_phi_pair"
        )
        index = function.args.kwonlyargs.index(pair_argument)
        default = function.args.kw_defaults[index]
        assert isinstance(default, ast.Constant) and default.value is False


def test_run_full_eb_both_implicit_stage_paths_dispatch_on_the_flag():
    tree = _tree(DRIVER_SOURCE)
    run_full_eb = _function(tree, "run_full_eb")
    source = DRIVER_SOURCE.read_text()

    stage_functions = [
        node
        for node in ast.walk(run_full_eb)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("implicit_stage", "staged_implicit_kernel")
    ]
    assert {node.name for node in stage_functions} == {
        "implicit_stage",
        "staged_implicit_kernel",
    }

    for stage_function in stage_functions:
        segment = ast.get_source_segment(source, stage_function)
        assert segment is not None
        assert "if model.implicit_current_phi_pair:" in segment
        assert "model.solve_implicit_current_phi_pair(" in segment
        assert "reconstruct_stage_phi(updated, model)" in segment
        # The guard must precede the fallback in source order.
        assert segment.index("if model.implicit_current_phi_pair:") < segment.index(
            "reconstruct_stage_phi(updated, model)"
        )


def test_run_metadata_records_implicit_current_phi_pair():
    source = DRIVER_SOURCE.read_text()
    assert '"implicit_current_phi_pair": bool(args.implicit_current_phi_pair)' in source
    assert "implicit_current_phi_pair=bool(args.implicit_current_phi_pair)" in source


def test_evaluate_stage_selects_affine_flux_and_zeros_phi_force_under_the_flag():
    tree = _tree(RHS_PATH)
    evaluate_stage = _function(tree, "evaluate_stage")
    source = RHS_PATH.read_text()
    segment = ast.get_source_segment(source, evaluate_stage)
    assert segment is not None
    assert '"vorticity_current_affine_flux_div"\n            if self.implicit_current_phi_pair' in segment
    assert "else \"vorticity_current_flux_div\"" in segment
    assert "jnp.zeros_like(grad_parallel_phi)\n            if self.implicit_current_phi_pair" in segment
    assert "else mi_over_me * grad_parallel_phi" in segment


def test_exb_brackets_and_curvature_scale_by_rho_star_not_divide():
    tree = _tree(RHS_PATH)
    evaluate_stage = _function(tree, "evaluate_stage")
    source = RHS_PATH.read_text()
    segment = ast.get_source_segment(source, evaluate_stage)
    assert segment is not None
    for poisson_term in (
        "poisson_density",
        "poisson_Te",
        "poisson_Ti",
        "poisson_Vi",
        "poisson_Ve",
        "poisson_vorticity",
    ):
        assert f"{poisson_term} * rho_star" in segment
        assert f"{poisson_term} / rho_star" not in segment
    for curvature_term in (
        "curvature_density_contribution",
        "curvature_Te_contribution",
        "curvature_Ti_contribution",
        "curvature_vorticity_contribution",
    ):
        assert f"{curvature_term} = rho_star * {curvature_term}" in segment


def test_polarization_relation_divides_vorticity_by_rho_star_squared():
    tree = _tree(RHS_PATH)
    source = RHS_PATH.read_text()

    reconstruct = _function(tree, "_reconstruct_phi_from_prepared")
    reconstruct_segment = ast.get_source_segment(source, reconstruct)
    assert reconstruct_segment is not None
    assert "self.parameters.rho_star, dtype=jnp.float64) ** 2" in reconstruct_segment

    pair = _function(tree, "solve_implicit_current_phi_pair")
    pair_segment = ast.get_source_segment(source, pair)
    assert pair_segment is not None
    assert "omega_pred / rho_star ** 2" in pair_segment
    assert "(h ** 2 * mu) / rho_star ** 2" in pair_segment

    polarization = _function(tree, "polarization_balance_terms")
    polarization_segment = ast.get_source_segment(source, polarization)
    assert polarization_segment is not None
    assert "rho_star" in polarization_segment
    assert "** 2" in polarization_segment


def test_solve_implicit_current_phi_pair_exists_and_uses_extra_operator():
    tree = _tree(RHS_PATH)
    source = RHS_PATH.read_text()
    function = _function(tree, "solve_implicit_current_phi_pair")
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    assert "extra_operator=extra_operator" in segment
    assert "def extra_operator(field_owned: jnp.ndarray)" in segment


def test_local_perp_laplacian_inverse_solver_carries_extra_operator():
    tree = _tree(OPERATORS_PATH)
    source = OPERATORS_PATH.read_text()
    solver = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LocalPerpLaplacianInverseSolver"
    )
    class_segment = ast.get_source_segment(source, solver)
    assert class_segment is not None
    assert "extra_operator: Callable[[jnp.ndarray], jnp.ndarray] | None = None" in (
        class_segment
    )

    apply_a = next(
        node
        for node in ast.walk(solver)
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_A"
    )
    apply_a_segment = ast.get_source_segment(source, apply_a)
    assert apply_a_segment is not None
    assert "if self.extra_operator is not None:" in apply_a_segment
    assert "result = result + self.extra_operator(values)" in apply_a_segment

    tree_flatten = next(
        node
        for node in ast.walk(solver)
        if isinstance(node, ast.FunctionDef) and node.name == "tree_flatten"
    )
    tree_flatten_segment = ast.get_source_segment(source, tree_flatten)
    assert tree_flatten_segment is not None
    assert "self.extra_operator," in tree_flatten_segment

    tree_unflatten = next(
        node
        for node in ast.walk(solver)
        if isinstance(node, ast.FunctionDef) and node.name == "tree_unflatten"
    )
    tree_unflatten_segment = ast.get_source_segment(source, tree_unflatten)
    assert tree_unflatten_segment is not None
    assert "extra_operator," in tree_unflatten_segment
    assert "extra_operator=extra_operator," in tree_unflatten_segment


def test_no_production_wording_labels_the_implicit_pair_as_experimental():
    for path in (RHS_PATH, OPERATORS_PATH, DRIVER_SOURCE):
        source = path.read_text()
        lowered = source.lower()
        for needle in ("experiment:", "experimental:", "# experiment ("):
            assert needle not in lowered, f"{needle!r} still present in {path.name}"
