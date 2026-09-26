"""Focused driver-contract tests for the selectable HSX FCI path.

These tests intentionally stop before expensive metric fitting or time
integration.  They verify the driver owns the selection, map validation, and
explicit map operand plumbing; the mapped operator implementation is tested
in the DRBX library tests.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "simulate_hsx_blob.py"


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
    descriptor = object()
    fields = np.zeros((4, 8, 12, hsx.RLP_PACKED_FIELD_COUNT))
    monkeypatch.setattr(
        hsx,
        "build_sharded_polar_angular_agglomeration_payload",
        lambda *_a, **_k: (descriptor, fields),
    )
    return descriptor


def _sharded_geometry():
    return SimpleNamespace(
        global_shape=(4, 8, 12),
        shard_counts=(1, 1, 1),
        maps_valid=True,
        map_fields=np.zeros((4, 8, 12, 8)),
        cell_fields=object(),
        domain=SimpleNamespace(
            layout=SimpleNamespace(owned_shape=(4, 8, 12), cell_halo_shape=(6, 10, 14)),
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
        ),
    )


def _main_error(hsx, capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as error:
        hsx.main(list(argv))
    assert error.value.code == 2
    return capsys.readouterr().err


def test_parser_exposes_production_controls_without_removed_selectors():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args([])
    assert args.gmres_residual_correction_steps == 1
    assert args.checkpoint_every == 0
    assert not any(
        "--production-characteristic-solver" in action.option_strings
        for action in parser._actions
    )
    assert not any(
        "--flux-framework" in action.option_strings for action in parser._actions
    )
    assert not any(
        "--parallel-operator-scheme" in action.option_strings
        for action in parser._actions
    )
    assert not any(
        "--time-integrator" in action.option_strings for action in parser._actions
    )
    assert not any("--fci-trace-substeps" in action.option_strings for action in parser._actions)
    assert not any(
        "--curvature-rlp-face-scheme" in action.option_strings
        for action in parser._actions
    )


def test_periodic_checkpoint_interval_is_validated_before_lowering(
    fake_geometry_artifact, capsys
):
    hsx = _driver_module()
    path, _ = fake_geometry_artifact(hsx)
    message = _main_error(hsx, capsys, "--geometry", str(path), "--checkpoint-every", "-1")
    assert "--checkpoint-every must be nonnegative" in message


def test_periodic_checkpoint_is_step_indexed_and_plumbed_to_full_run():
    source = DRIVER_PATH.read_text()
    run = _function(_tree(), "run_full_eb")
    argument_names = [argument.arg for argument in run.args.args]
    argument_names.extend(argument.arg for argument in run.args.kwonlyargs)
    assert "checkpoint_every" in argument_names
    assert "checkpoint_step{int(step):06d}" in source
    assert "checkpoint_every=int(args.checkpoint_every)" in source
    assert "periodic_checkpoint=True" in source


def test_periodic_checkpoint_payload_is_atomic_and_restartable(tmp_path):
    hsx = _driver_module()
    shape = (2, 3, 4)
    fields = {
        name: np.full(shape, index + 0.25, dtype=np.float64)
        for index, name in enumerate(hsx.FciDrbEBState.__dataclass_fields__)
    }
    checkpoint = tmp_path / "run.checkpoint_step000025.npz"
    hsx._atomic_save_npz(
        checkpoint,
        **fields,
        time=np.asarray(0.0125, dtype=np.float64),
        step=np.asarray(25, dtype=np.int64),
    )
    assert checkpoint.is_file()
    assert not tuple(tmp_path.glob(f".{checkpoint.name}.*.npz"))

    state, restart_time = hsx._load_restart_state(
        checkpoint,
        resolution=shape,
        frame=-1,
    )
    assert restart_time == pytest.approx(0.0125)
    for name, expected in fields.items():
        np.testing.assert_array_equal(getattr(state, name), expected)


def test_materialized_cell_checkpoint_restores_exact_owner_values():
    hsx = _driver_module()
    shape = (2, 2, 1)
    owner_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    owner_index[1, 0, 0] = (0, 0, 0)
    owner_mask = np.ones(shape, dtype=bool)
    owner_mask[1, 0, 0] = False
    host = SimpleNamespace(
        topology=SimpleNamespace(
            owner_index=owner_index,
            is_active_owner=owner_mask,
        )
    )
    fields = {}
    for index, name in enumerate(hsx.FciDrbEBState.__dataclass_fields__):
        value = index + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        value[1, 0, 0] = value[0, 0, 0]
        fields[name] = value
    state = hsx.FciDrbEBState(**fields)

    restored, reused = hsx._restore_materialized_cell_owner_state(state, host)

    assert reused
    for name, value in fields.items():
        np.testing.assert_array_equal(
            getattr(restored, name), np.where(owner_mask, value, 0.0)
        )


def test_noncanonical_cell_checkpoint_uses_aggregation_fallback():
    hsx = _driver_module()
    shape = (2, 2, 1)
    owner_index = np.stack(np.indices(shape, dtype=np.int32), axis=-1)
    owner_index[1, 0, 0] = (0, 0, 0)
    owner_mask = np.ones(shape, dtype=bool)
    owner_mask[1, 0, 0] = False
    host = SimpleNamespace(
        topology=SimpleNamespace(
            owner_index=owner_index,
            is_active_owner=owner_mask,
        )
    )
    fields = {
        name: np.zeros(shape, dtype=np.float64)
        for name in hsx.FciDrbEBState.__dataclass_fields__
    }
    fields["density"][1, 0, 0] = 1.0
    state = hsx.FciDrbEBState(**fields)

    candidate, reused = hsx._restore_materialized_cell_owner_state(state, host)

    assert not reused
    assert candidate is state


def test_non_toroidal_geometry_artifact_is_rejected(fake_geometry_artifact, capsys):
    hsx = _driver_module()
    path, _ = fake_geometry_artifact(hsx, topology="square")
    message = _main_error(hsx, capsys, "--geometry", str(path))
    assert "could not load --geometry artifact" in message
    assert "unknown topology" in message


def test_toroidal_artifact_without_rlp_topology_is_rejected(fake_geometry_artifact, capsys):
    hsx = _driver_module()
    path, artifact = fake_geometry_artifact(hsx)
    artifact.polar_angular_geometry = None
    assert "RLP topology" in _main_error(hsx, capsys, "--geometry", str(path))


def test_every_geometry_assembling_kernel_has_a_map_operand_and_spec():
    tree = _tree()
    run = _function(tree, "run_full_eb")
    kernel_names = {
        "reconstruct_initial_phi_kernel",
        "full_imex_advance",
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
        assert "control_volume_fields_owned" in arguments, name
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
    assert len(assemble_calls) == 1
    assert all(len(call.args) >= 3 for call in assemble_calls)
    assert all(
        any(
            isinstance(node, ast.Name) and node.id == "map_fields_owned"
            for node in ast.walk(call)
        )
        for call in assemble_calls
    )


def test_geometry_only_lowers_the_artifact_and_stops(monkeypatch, fake_geometry_artifact):
    hsx = _driver_module()
    path, artifact = fake_geometry_artifact(hsx)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    _install_toroidal_rlp_mocks(monkeypatch, hsx)
    lowering_calls = []
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *args, **kwargs: lowering_calls.append(args) or _sharded_geometry(),
    )
    monkeypatch.setattr(
        hsx, "run_full_eb", lambda *_a, **_k: pytest.fail("--geometry-only must not run")
    )

    hsx.main(["--geometry", str(path), "--geometry-only"])

    assert [call[0] for call in lowering_calls] == [artifact.geometry]


def test_fci_main_passes_production_scheme_and_metadata_to_run(
    monkeypatch, fake_geometry_artifact
):
    hsx = _driver_module()
    path, _ = fake_geometry_artifact(hsx, metadata={"metric_radial_degree": 17})
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    _install_toroidal_rlp_mocks(monkeypatch, hsx)
    monkeypatch.setattr(hsx, "build_local_fci_geometries", lambda *_a, **_k: _sharded_geometry())
    monkeypatch.setattr(hsx, "build_initial_state", lambda *_a, **_k: object())
    monkeypatch.setattr(hsx, "_aggregate_initial_owner_state", lambda state, _host: state)
    monkeypatch.setattr(hsx, "_assert_owner_sparse", lambda *_a, **_k: None)
    calls = []
    monkeypatch.setattr(hsx, "run_full_eb", lambda *args, **kwargs: calls.append(kwargs))

    hsx.main(
        [
            "--geometry",
            str(path),
            "--num-steps",
            "1",
            "--final-time",
            "1e-6",
            "--checkpoint-every",
            "17",
        ]
    )
    assert calls
    metadata = calls[0]["run_metadata"]
    assert calls[0]["parallel_operator_scheme"] == "fci"
    assert metadata["parallel_operator_scheme"] == "fci"
    assert metadata["geometry"] == str(path.resolve())
    assert metadata["geometry_manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert metadata["fci_trace_substeps"] == 7
    assert metadata["metric_radial_degree"] == 17
    assert metadata["angular_owner_profile"] == "geometry-artifact"
    assert calls[0]["checkpoint_every"] == 17
    assert calls[0]["control_volume_descriptor"] is not None
    assert calls[0]["control_volume_fields_host"].shape == (
        4,
        8,
        12,
        hsx.RLP_PACKED_FIELD_COUNT,
    )


def test_production_split_guard_requires_compatible_runtime():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args(
        [
            "--parallel-flux-pairing", "support-core",
            "--poisson-bracket-scheme", "compatible-flux",
        ]
    )
    hsx._validate_flux_framework(args)

    args = parser.parse_args(["--parallel-flux-pairing", "legacy"])
    with pytest.raises(ValueError, match="support-core"):
        hsx._validate_flux_framework(args)


def test_production_split_metadata_contract_is_recorded():
    source = DRIVER_PATH.read_text()
    assert '"flux_framework": "production-split"' in source
    assert '"flux_framework_source": "fixed production configuration"' in source
    assert '"curvature_operator": "production-characteristic-owner-face"' in source
    assert '"curvature_operator_source": "fixed production method"' in source
    assert '"parallel_material_scheme": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME")' in source
    assert '"production_characteristic_solver": "canonical-face-state"' in source
    assert '"production_characteristic_solver_source": "fixed production method"' in source
    assert "DRBX_PRODUCTION_CHARACTERISTIC_SOLVER" not in source
