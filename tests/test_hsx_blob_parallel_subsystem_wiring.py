"""Source-level contract tests for the parallel-subsystem diagnostic mode.

These tests deliberately avoid constructing HSX geometry.  The mode is a
static experiment switch, so the important regression surface is the wiring
from the CLI to the jitted local RHS and the contents of the enabled branch.
"""

import ast
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RHS_PATH = WORKSPACE / "DRBX" / "src" / "drbx" / "native" / "fci_drb_EB_rhs.py"
DRIVER_PATH = WORKSPACE / "simulate_hsx_blob.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _source(path: Path, node: ast.AST) -> str:
    result = ast.get_source_segment(path.read_text(), node)
    assert result is not None
    return result


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _argument_names(function: ast.FunctionDef) -> set[str]:
    return {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }


def _keyword_call(
    tree: ast.AST,
    function_name: str,
    keyword: str,
) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
        and any(argument.arg == keyword for argument in node.keywords)
    ]


def _parallel_only_branch() -> ast.If:
    tree = _tree(RHS_PATH)
    evaluate_stage = _function(tree, "evaluate_stage")
    branches = [
        node
        for node in ast.walk(evaluate_stage)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "self"
        and node.test.attr == "parallel_subsystem_only"
    ]
    assert len(branches) == 1, "evaluate_stage must have one diagnostic branch"
    return branches[0]


def test_rhs_has_static_disabled_parallel_subsystem_switch():
    tree = _tree(RHS_PATH)
    rhs_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LocalFciDrbEBRhs"
    )
    field = next(
        node
        for node in rhs_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "parallel_subsystem_only"
    )
    assert isinstance(field.value, ast.Constant)
    assert field.value.value is False


def test_parallel_subsystem_flag_is_wired_through_driver_layers():
    driver = _tree(DRIVER_PATH)
    build = _function(driver, "build_local_eb_model")
    run = _function(driver, "run_full_eb")
    main = _function(driver, "main")

    for function in (build, run):
        assert "parallel_subsystem_only" in _argument_names(function)

    build_source = _source(DRIVER_PATH, build)
    run_source = _source(DRIVER_PATH, run)
    main_source = _source(DRIVER_PATH, main)
    assert "parallel_subsystem_only=parallel_subsystem_only" in build_source
    assert "parallel_subsystem_only=parallel_subsystem_only" in run_source
    assert "parallel_subsystem_only=bool(args.parallel_subsystem_only)" in main_source

    rhs_calls = _keyword_call(driver, "LocalFciDrbEBRhs", "parallel_subsystem_only")
    assert rhs_calls, "build_local_eb_model must pass the switch to the RHS"


def test_cli_exposes_parallel_subsystem_only_as_opt_in_flag():
    source = DRIVER_PATH.read_text()
    tree = _tree(DRIVER_PATH)
    parser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--parallel-subsystem-only"
    ]
    assert len(parser_calls) == 1
    assert "action=\"store_true\"" in _source(DRIVER_PATH, parser_calls[0])
    assert "parallel_subsystem_only" in source


def test_parallel_subsystem_mode_is_restricted_to_rk4():
    run = _function(_tree(DRIVER_PATH), "run_full_eb")
    source = _source(DRIVER_PATH, run)
    assert "parallel_subsystem_only" in source
    assert "time_integrator != \"rk4\"" in source
    assert "parallel_subsystem_only" in source[source.index("time_integrator != \"rk4\"") :]


def test_enabled_branch_reuses_parallel_operators_and_wall_traces():
    branch = _parallel_only_branch()
    source = _source(RHS_PATH, branch)

    # These are the production conservative parallel quantities.  The branch
    # must not reconstruct a second, diagnostic-only set of operators.
    for name in (
        "parallel_density_flux_divergence",
        "parallel_current_flux_divergence",
        "parallel_Ve_flux_divergence",
        "parallel_Vi_flux_divergence",
        "grad_parallel_Te",
        "grad_parallel_Ti",
        "grad_parallel_Ve",
        "grad_parallel_Vi",
        "grad_parallel_phi",
        "grad_parallel_Pe",
        "grad_parallel_pressure",
        "grad_parallel_current",
    ):
        assert name in source

    assert "boundary_trace=operator_boundary." in source
    assert "local_parallel_flux_div_op" in source
    assert "local_grad_parallel_op_conservative" in source


def test_parallel_subsystem_branch_keeps_seven_field_rhs_and_excludes_nonparallel_terms():
    branch = _parallel_only_branch()
    source = _source(RHS_PATH, branch)

    # Seven returned fields: algebraic phi plus the six evolved equations.
    for name in (
        "density_rhs",
        "Te_rhs",
        "Ti_rhs",
        "Vi_rhs",
        "Ve_rhs",
        "vorticity_rhs",
        "phi=jnp.zeros_like(phi_owned)",
    ):
        assert name in source

    # The diagnostic isolates the parallel subsystem.  These contributions
    # must remain outside its enabled branch.
    for forbidden in (
        "poisson_",
        "curvature_",
        "_field_perp_diffusion",
        "source_owned",
        "density_diff",
        "Te_diff",
        "Ti_diff",
        "Vi_diff",
        "Ve_diff",
        "vorticity_diff",
    ):
        assert forbidden not in source


def _wall_mask_context() -> tuple[ast.FunctionDef, list[ast.Assign]]:
    """Return the snapshot inspection function and its mask assignments."""

    tree = _tree(DRIVER_PATH)
    run = _function(tree, "run_full_eb")
    inspect_state = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.FunctionDef) and node.name == "inspect_state"
    )
    assignments = [
        node
        for node in ast.walk(inspect_state)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "wall_masks"
            for target in node.targets
        )
    ]
    assert len(assignments) >= 2
    return inspect_state, assignments


def test_snapshot_wall_mask_is_gated_by_runtime_physical_boundaries_on_all_axes():
    inspect_state, _ = _wall_mask_context()
    source = _source(DRIVER_PATH, inspect_state)

    # A periodic axis must contribute no wall cells.  In particular, toroidal
    # theta/eta seams and the lower toroidal axis are topology, not vessel
    # walls, so every axis-side mask must be conditioned by the runtime domain
    # ownership rather than by global index alone.
    loops = [
        node
        for node in ast.walk(inspect_state)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "axis"
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Constant)
        and node.iter.args[0].value == 3
    ]
    assert loops, "wall masks must cover all three logical axes"
    runtime_calls = [
        node
        for node in ast.walk(inspect_state)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in ("runtime_has_physical_lower", "runtime_has_physical_upper")
    ]
    assert {node.func.attr for node in runtime_calls} == {
        "runtime_has_physical_lower",
        "runtime_has_physical_upper",
    }
    assert all(
        len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "axis"
        for node in runtime_calls
    )
    assert "global_coordinates[axis]" in source
    assert "coordinates < wall_term_count" in source
    assert "coordinates >= global_shape[axis] - wall_term_count" in source


def test_snapshot_wall_mask_does_not_use_unconditional_side_comparisons():
    inspect_state, _ = _wall_mask_context()
    source = _source(DRIVER_PATH, inspect_state)

    # The old failure mode was a raw coordinate comparison such as
    # ``global_coordinates[1] < wall_term_count`` OR'ed directly into the
    # mask.  Require each lower/upper side to be represented by a conditional
    # runtime mask; the exact implementation may use a loop or explicit axes.
    comparisons = [
        node
        for node in ast.walk(inspect_state)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "coordinates"
    ]
    assert len(comparisons) >= 2
    assert "wall_term_count" in source
    assert "runtime_has_physical_lower" in source
    assert "runtime_has_physical_upper" in source
    assert "& lower_physical" in source
    assert "& upper_physical" in source
