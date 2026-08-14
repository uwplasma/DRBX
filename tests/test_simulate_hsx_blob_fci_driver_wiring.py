"""Focused driver-contract tests for the selectable HSX FCI path.

These tests intentionally stop before expensive metric fitting or time
integration.  They verify the driver owns the selection, map validation, and
explicit map operand plumbing; the mapped operator implementation is tested
in the DRBX library tests.
"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
DRIVER_PATH = WORKSPACE / "simulate_hsx_blob.py"


def _tree() -> ast.Module:
    return ast.parse(DRIVER_PATH.read_text())


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _driver_module():
    spec = importlib.util.spec_from_file_location("hsx_driver_fci_wiring", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_toroidal_rlp_mocks(monkeypatch, hsx):
    topology = SimpleNamespace(
        is_active_owner=np.ones((4, 8, 12), dtype=bool),
        is_merge_source=np.zeros((4, 8, 12), dtype=bool),
    )
    host = SimpleNamespace(
        angular_group_size=np.asarray((8, 4, 2, 1)),
        topology=topology,
    )
    native = SimpleNamespace(
        irregular_faces=SimpleNamespace(max_rows=0, max_patches=4)
    )
    monkeypatch.setattr(
        hsx,
        "build_metric_aware_polar_angular_agglomeration_geometry",
        lambda *_a, **_k: (host, 0.5),
    )
    monkeypatch.setattr(
        hsx, "assemble_single_device_local_fci_geometry", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        hsx, "lower_polar_angular_agglomeration_geometry", lambda *_a, **_k: native
    )
    return host, native


def test_parser_exposes_coordinate_default_and_fci_trace_controls():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args([])
    assert args.parallel_operator_scheme == "coordinate"
    assert args.fci_trace_substeps == 4

    scheme_action = next(
        action
        for action in parser._actions
        if "--parallel-operator-scheme" in action.option_strings
    )
    assert scheme_action.choices == ("coordinate", "fci")
    trace_action = next(
        action
        for action in parser._actions
        if "--fci-trace-substeps" in action.option_strings
    )
    assert trace_action.type("7") == 7


def test_fci_requires_toroidal_topology_before_geometry(monkeypatch):
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(["--parallel-operator-scheme", "fci"])
    assert error.value.code == 2


def test_every_geometry_assembling_kernel_has_a_map_operand_and_spec():
    tree = _tree()
    run = _function(tree, "run_full_eb")
    kernel_names = {
        "precompute_wall_projectors",
        "reconstruct_initial_phi",
        "full_rk4_advance",
        "full_ark2_imex_advance",
        "full_imex_explicit_rhs",
        "full_imex_bdf2_advance",
        "inspect_state",
    }
    kernels = {
        node.name: node
        for node in ast.walk(run)
        if isinstance(node, ast.FunctionDef) and node.name in kernel_names
    }
    assert kernels.keys() == kernel_names
    for name, kernel in kernels.items():
        arguments = {
            argument.arg
            for argument in (
                *kernel.args.posonlyargs,
                *kernel.args.args,
                *kernel.args.kwonlyargs,
            )
        }
        assert "map_fields_owned" in arguments, name
        source = ast.get_source_segment(DRIVER_PATH.read_text(), kernel)
        assert source is not None

    shard_maps = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shard_map"
    ]
    # Each runtime geometry kernel carries cell geometry and map geometry as
    # separate leading-axis-sharded operands.  The map-only coordinate path
    # still receives the same zero placeholder and disables it at assembly.
    for call in shard_maps:
        in_specs = next(keyword.value for keyword in call.keywords if keyword.arg == "in_specs")
        if isinstance(in_specs, ast.Tuple):
            geometry_specs = [
                value
                for value in in_specs.elts
                if isinstance(value, ast.Name) and value.id == "geometry_spec"
            ]
            if geometry_specs:
                assert len(geometry_specs) >= 2

    assemble_calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assemble_local_fci_geometry"
    ]
    assert len(assemble_calls) == 2
    assert all(len(call.args) >= 3 for call in assemble_calls)
    assert all(
        any(
            isinstance(node, ast.Name) and node.id == "map_fields_owned"
            for node in ast.walk(call)
        )
        for call in assemble_calls
    )


def test_geometry_only_fci_requests_map_generation_and_records_substeps(
    monkeypatch, tmp_path
):
    hsx = _driver_module()
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")
    global_calls = []
    lowering_calls = []
    metric_evaluator = object()
    _install_toroidal_rlp_mocks(monkeypatch, hsx)

    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    monkeypatch.setattr(
        hsx,
        "build_hsx_fci_geometry",
        lambda **kwargs: global_calls.append(kwargs)
        or (object(), np.zeros((4, 8, 12, 3)), 2, None, metric_evaluator),
    )
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *args, **kwargs: lowering_calls.append((args, kwargs))
        or SimpleNamespace(
            global_shape=(4, 8, 12),
            shard_counts=(1, 1, 1),
            cell_fields=object(),
            maps_valid=True,
            map_fields=np.zeros((4, 8, 12, 8)),
            domain=SimpleNamespace(
                layout=SimpleNamespace(
                    owned_shape=(4, 8, 12), cell_halo_shape=(6, 10, 14)
                ),
                periodic_axes=(False, True, True),
                axis_regular_axes=(True, False, False),
            ),
        ),
    )

    hsx.main(
        [
            "--topology",
            "toroidal",
            "--parallel-operator-scheme",
            "fci",
            "--fci-trace-substeps",
            "7",
            "--geometry-only",
            "--makegrid",
            str(makegrid),
            "--vessel",
            str(vessel),
            "--resolution",
            "4",
            "8",
            "12",
            "--metric-mesh-shape",
            "5",
            "8",
            "6",
        ]
    )
    assert global_calls[0]["construct_fci_maps"] is True
    assert global_calls[0]["fci_trace_substeps"] == 7
    assert lowering_calls


def test_fci_main_passes_scheme_and_metadata_to_run(monkeypatch, tmp_path):
    hsx = _driver_module()
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    _install_toroidal_rlp_mocks(monkeypatch, hsx)
    monkeypatch.setattr(
        hsx,
        "build_hsx_fci_geometry",
        lambda **kwargs: (object(), np.zeros((4, 8, 12, 3)), 2, None, object()),
    )
    sharded = SimpleNamespace(
        global_shape=(4, 8, 12),
        shard_counts=(1, 1, 1),
        maps_valid=True,
        map_fields=np.zeros((4, 8, 12, 8)),
        cell_fields=object(),
        domain=SimpleNamespace(
            layout=SimpleNamespace(
                owned_shape=(4, 8, 12), cell_halo_shape=(6, 10, 14)
            ),
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
        ),
    )
    monkeypatch.setattr(hsx, "build_local_fci_geometries", lambda *_a, **_k: sharded)
    monkeypatch.setattr(hsx, "build_initial_state", lambda *_a, **_k: object())
    monkeypatch.setattr(hsx, "_aggregate_initial_owner_state", lambda state, _host: state)
    monkeypatch.setattr(hsx, "_assert_owner_sparse", lambda *_a, **_k: None)
    calls = []
    monkeypatch.setattr(hsx, "run_full_eb", lambda *args, **kwargs: calls.append(kwargs))

    hsx.main(
        [
            "--topology",
            "toroidal",
            "--parallel-operator-scheme",
            "fci",
            "--fci-trace-substeps",
            "6",
            "--makegrid",
            str(makegrid),
            "--vessel",
            str(vessel),
            "--resolution",
            "4",
            "8",
            "12",
            "--metric-mesh-shape",
            "5",
            "8",
            "6",
            "--num-steps",
            "1",
            "--final-time",
            "1e-6",
        ]
    )
    assert calls
    assert calls[0]["parallel_operator_scheme"] == "fci"
    assert calls[0]["run_metadata"]["parallel_operator_scheme"] == "fci"
    assert calls[0]["run_metadata"]["fci_trace_substeps"] == 6
