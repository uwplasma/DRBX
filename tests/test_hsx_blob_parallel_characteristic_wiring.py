"""Driver/CLI wiring tests for the parallel characteristic wall closure.

These tests intentionally inspect the driver source only.  Constructing an
HSX metric is unnecessary for verifying that the option reaches the local
RHS and the recorded run metadata.
"""

import ast
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
DRIVER_PATH = WORKSPACE / "simulate_hsx_blob.py"


def _tree() -> ast.Module:
    return ast.parse(DRIVER_PATH.read_text())


def _source(node: ast.AST) -> str:
    result = ast.get_source_segment(DRIVER_PATH.read_text(), node)
    assert result is not None
    return result


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keyword_calls(function: ast.AST, callee: str, keyword: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
        and any(argument.arg == keyword for argument in node.keywords)
    ]


def test_parallel_inflow_closure_is_threaded_to_local_rhs_for_coordinate_path():
    build = _function("build_local_eb_model")
    run = _function("run_full_eb")

    build_source = _source(build)
    run_source = _source(run)

    assert "parallel_inflow_closure" in {
        argument.arg
        for argument in (
            *build.args.args,
            *build.args.kwonlyargs,
        )
    }
    assert "parallel_inflow_closure" in {
        argument.arg
        for argument in (
            *run.args.args,
            *run.args.kwonlyargs,
        )
    }
    assert "parallel_inflow_closure=parallel_inflow_closure" in run_source
    assert "parallel_inflow_closure=parallel_inflow_closure" in build_source
    assert _keyword_calls(build, "LocalFciDrbEBRhs", "parallel_inflow_closure")


def test_parallel_inflow_closure_is_independent_of_velocity_wall_bc():
    build_source = _source(_function("build_local_eb_model"))
    run_source = _source(_function("run_full_eb"))

    assert "parallel_velocity_wall_bc=parallel_velocity_wall_bc" in build_source
    assert "parallel_inflow_closure=parallel_inflow_closure" in build_source
    assert "parallel_velocity_wall_bc=parallel_velocity_wall_bc" in run_source
    assert "parallel_inflow_closure=parallel_inflow_closure" in run_source
    assert '"local-characteristic"' in build_source
    assert '"equilibrium-characteristic"' in build_source


def test_cli_declares_local_five_field_characteristic_closure_with_central_default():
    parser = _function("_build_parser")
    calls = [
        node
        for node in ast.walk(parser)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--parallel-inflow-closure"
    ]
    assert len(calls) == 1

    call = calls[0]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["default"], ast.Constant)
    assert keywords["default"].value == "central"
    assert isinstance(keywords["choices"], ast.Tuple)
    assert [element.value for element in keywords["choices"].elts] == [
        "central",
        "local-characteristic",
        "equilibrium-characteristic",
    ]

    help_text = _source(call)
    assert "five-field material characteristics" in help_text
    assert "excludes phi and vorticity" in help_text
    assert "--parallel-velocity-wall-bc" in help_text
    assert "incoming" in help_text
    assert "(1,1,1,0,0)" in help_text


def test_selected_closure_is_printed_and_recorded_in_run_metadata():
    main_source = _source(_function("main"))
    assert "args.parallel_inflow_closure" in main_source
    assert "[simulation] parallel inflow closure" in main_source
    assert '"parallel_inflow_closure": str(args.parallel_inflow_closure)' in (
        main_source
    )
    assert (
        "parallel_inflow_closure=str(args.parallel_inflow_closure)"
        in main_source
    )
