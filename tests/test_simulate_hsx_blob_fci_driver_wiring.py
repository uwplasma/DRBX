"""Focused contracts for the explicit-artifact HSX simulation consumer.

These tests stop before time integration.  They verify that physical geometry
production is absent and that the named artifact is only lowered and wired to
runtime operators.
"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
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




def test_parser_requires_explicit_geometry_and_removes_producer_controls():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args([])
    assert args.geometry is None
    assert args.blob_initialization == "logical"
    assert args.parallel_operator_scheme == "coordinate"
    assert args.gmres_residual_correction_steps == 1
    assert args.checkpoint_every == 0
    assert not args.rhs_replay_electron_force_wall_audit
    assert args.flux_framework == "legacy"
    framework_action = next(
        action for action in parser._actions if "--flux-framework" in action.option_strings
    )
    assert framework_action.choices == ("legacy", "production-split")
    assert not any(
        "--production-characteristic-solver" in action.option_strings
        for action in parser._actions
    )

    scheme_action = next(
        action
        for action in parser._actions
        if "--parallel-operator-scheme" in action.option_strings
    )
    assert scheme_action.choices == ("coordinate", "fci")
    electron_force_action = next(
        action
        for action in parser._actions
        if "--rhs-replay-electron-force-wall-audit" in action.option_strings
    )
    assert electron_force_action.default is False
    assert not any(
        "--curvature-rlp-face-scheme" in action.option_strings
        for action in parser._actions
    )
    retired = (
        "--makegrid", "--vessel", "--resolution", "--metric-cache-dir",
        "--metric-mesh-shape", "--metric-spline-degree",
        "--fci-trace-substeps", "--fieldline-substeps-per-plane",
        "--blob-reference-eta", "--blob-parallel-half-length",
        "--axis-core-radius", "--agglomeration-volume-ratio",
    )
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings.isdisjoint(retired)


def test_driver_has_no_physical_geometry_producer_imports_or_helpers():
    source = DRIVER_PATH.read_text()
    assert "hsx_fci_builder" not in source
    for retired_api in (
        "bfield_evaluator_from_makegrid",
        "scalar_potential_evaluator_from_bfield",
        "build_metric_evaluator",
        "build_fci_maps_from_callbacks",
        "build_hsx_metric_evaluator",
        "build_hsx_fci_geometry",
        "_trace_logical_labels_to_eta_plane",
        "build_field_aligned_filament_profile",
        "_filament_cache_path",
    ):
        assert retired_api not in source


def test_missing_geometry_fails_before_any_runtime_setup():
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main([])
    assert error.value.code == 2


def test_explicit_geometry_is_loaded_and_only_lowered(monkeypatch, tmp_path):
    hsx = _driver_module()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("simulation consumer attempted geometry production")

    from drbx.geometry import fci_geometry as fci_geometry_module
    from drbx.geometry import fci_owner_boundary_overlap as overlap_module
    from drbx.geometry import hsx_fci_builder as producer_module
    from drbx.geometry import hsx_simulation_geometry as orchestration_module

    for name in (
        "build_hsx_fci_geometry",
        "build_hsx_metric_evaluator",
        "_find_compatible_metric_cache",
        "bfield_evaluator_from_makegrid",
    ):
        monkeypatch.setattr(producer_module, name, forbidden)
    monkeypatch.setattr(
        orchestration_module, "build_hsx_simulation_geometry", forbidden
    )
    monkeypatch.setattr(
        overlap_module, "build_owner_boundary_overlap_geometry", forbidden
    )
    monkeypatch.setattr(
        fci_geometry_module,
        "trace_fci_points_to_plane_from_callbacks",
        forbidden,
    )
    artifact = SimpleNamespace(
        global_geometry=SimpleNamespace(shape=(4, 8, 12)),
        cell_positions=np.zeros((4, 8, 12, 3), dtype=np.float64),
        nfp=1,
        topology=SimpleNamespace(name="square"),
        curvature_edge_one_form=None,
        owner_geometry=None,
        owner_overlap=None,
        metadata={},
    )
    loaded = []
    monkeypatch.setattr(
        hsx, "load_fci_simulation_geometry", lambda path: loaded.append(path) or artifact
    )
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda counts: object())
    lowered = []
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *args, **kwargs: lowered.append((args, kwargs)) or SimpleNamespace(
            global_shape=(4, 8, 12),
            shard_counts=(1, 1, 1),
            maps_valid=True,
            map_fields=None,
            domain=SimpleNamespace(
                layout=SimpleNamespace(
                    owned_shape=(4, 8, 12), cell_halo_shape=(6, 10, 14)
                ),
                periodic_axes=(False, False, True),
                axis_regular_axes=(False, False, False),
            ),
        ),
    )
    hsx.main(["--geometry", str(tmp_path), "--geometry-only"])
    assert loaded == [tmp_path]
    assert lowered and lowered[0][0][0] is artifact.global_geometry


def test_periodic_checkpoint_interval_is_validated_before_geometry():
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(["--checkpoint-every", "-1"])
    assert error.value.code == 2


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


def test_fci_requires_toroidal_topology_before_geometry(monkeypatch):
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(["--parallel-operator-scheme", "fci"])
    assert error.value.code == 2


def test_every_geometry_assembling_kernel_has_a_map_operand_and_spec():
    tree = _tree()
    run = _function(tree, "run_full_eb")
    kernel_names = {
        "reconstruct_initial_phi_kernel",
        "full_rk4_advance",
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
    # Each runtime geometry kernel carries artifact cell geometry and map
    # geometry as separate leading-axis-sharded operands.
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




def test_production_split_guard_requires_compatible_runtime():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args(
        [
            "--flux-framework", "production-split",
            "--parallel-operator-scheme", "fci",
            "--parallel-flux-pairing", "support-core",
            "--poisson-bracket-scheme", "compatible-flux",
        ]
    )
    hsx._validate_flux_framework(args)

    args = parser.parse_args(
        [
            "--flux-framework", "production-split",
            "--parallel-operator-scheme", "fci",
            "--parallel-flux-pairing", "legacy",
        ]
    )
    with pytest.raises(ValueError, match="support-core"):
        hsx._validate_flux_framework(args)


def test_production_split_metadata_contract_is_recorded():
    source = DRIVER_PATH.read_text()
    assert '"flux_framework": str(args.flux_framework)' in source
    assert '"flux_framework_source": "simulate_hsx_blob.py:--flux-framework"' in source
    assert '"curvature_operator": "production-characteristic-owner-face"' in source
    assert '"curvature_operator_source": "fixed production method"' in source
    assert '"parallel_material_scheme": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME")' in source
    assert '"production_characteristic_solver": (' in source
    assert '"canonical-face-state"' in source
    assert '"fixed production method"' in source
    assert "DRBX_PRODUCTION_CHARACTERISTIC_SOLVER" not in source


def test_initial_phi_reconstruction_fails_fast_when_gmres_does_not_accept(
    monkeypatch,
    tmp_path,
):
    hsx = _driver_module()
    real_make_shard_mesh = hsx.make_shard_mesh

    def _replace(value, **changes):
        payload = dict(vars(value))
        payload.update(changes)
        return SimpleNamespace(**payload)

    monkeypatch.setattr(hsx, "replace", _replace)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda counts: real_make_shard_mesh(counts))
    layout = SimpleNamespace(halo_width=2)
    domain = SimpleNamespace(
        layout=layout,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        mesh_axis_names=("x", "y", "z"),
    )
    global_geometry = SimpleNamespace(shape=(1, 1, 1))
    sharded_geometry = SimpleNamespace(
        global_shape=(1, 1, 1),
        shard_counts=(1, 1, 1),
        maps_valid=True,
        map_fields=np.zeros((1, 1, 1, len(hsx.FCI_MAP_FIELDS)), dtype=np.float64),
        cell_fields=np.zeros((1, 1, 1, 1), dtype=np.float64),
        domain=domain,
    )
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *args, **kwargs: SimpleNamespace(
            domain=SimpleNamespace(mesh_axis_names=("x", "y", "z"))
        ),
    )
    monkeypatch.setattr(
        hsx,
        "assemble_single_device_local_fci_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(layout=layout),
    )
    monkeypatch.setattr(
        hsx,
        "build_local_curvature_face_coefficients",
        lambda *_args, **_kwargs: SimpleNamespace(
            axes=(
                np.ones((2, 1, 1), dtype=np.float64),
                np.ones((1, 2, 1), dtype=np.float64),
                np.ones((1, 1, 2), dtype=np.float64),
            )
        ),
    )
    monkeypatch.setattr(
        hsx,
        "LocalCurvatureFaceCoefficients3D",
        lambda **kwargs: SimpleNamespace(
            axes=(kwargs["x"], kwargs["y"], kwargs["z"]),
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        hsx,
        "assemble_local_fci_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(layout=layout),
    )

    info = SimpleNamespace(
        num_steps=1000,
        final_residual_rel_l2=1.0e-2,
        failed=True,
        converged=False,
    )

    class _FakeModel:
        def reconstruct_phi(self, _state, *, return_diagnostics=False):
            assert return_diagnostics
            return np.zeros((1, 1, 1), dtype=np.float64), info

    monkeypatch.setattr(hsx, "build_local_eb_model", lambda *_a, **_k: _FakeModel())

    state = hsx.FciDrbEBState(
        density=np.ones((1, 1, 1), dtype=np.float64),
        phi=np.zeros((1, 1, 1), dtype=np.float64),
        Te=np.ones((1, 1, 1), dtype=np.float64),
        Ti=np.ones((1, 1, 1), dtype=np.float64),
        Vi=np.zeros((1, 1, 1), dtype=np.float64),
        Ve=np.zeros((1, 1, 1), dtype=np.float64),
        vorticity=np.zeros((1, 1, 1), dtype=np.float64),
    )
    parameters = hsx.FciDrbEBRhsParameters(
        parallel_characteristic_wall_law="physical-boundary-state"
    )

    with pytest.raises(FloatingPointError, match="initial phi reconstruction"):
        hsx.run_full_eb(
            state,
            simulation_geometry=SimpleNamespace(
                global_geometry=global_geometry,
                cell_positions=np.zeros((1, 1, 1, 3), dtype=np.float64),
                nfp=1,
                curvature_edge_one_form=None,
                owner_geometry=None,
                metadata={},
            ),
            sharded_geometry=sharded_geometry,
            mesh=real_make_shard_mesh((1, 1, 1)),
            parameters=parameters,
            gmres_target_tolerance=1.0e-8,
            gmres_acceptance_tolerance=5.0e-5,
            gmres_max_iterations=10,
            gmres_restart=10,
            gmres_preconditioner="none",
            time_integrator="rk4",
            advance_execution="compiled",
            num_steps=1,
            timestep=1.0e-4,
            start_time=0.0,
            output_path=tmp_path / "out.npz",
            save_every=1,
            phase_timing=False,
            reconstruct_initial_phi=True,
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            physical_wall_model="no-flow",
        )


def test_rk_stage_diagnostics_have_explicit_finite_bit_gate():
    hsx = _driver_module()
    diagnostics = np.ones((5, 7, 5), dtype=np.float64)
    diagnostics[2, 3, 4] = 0.0
    assert not hsx._rk_stage_diagnostics_have_finite_bit(diagnostics)

    diagnostics[2, 3, 4] = 1.0
    assert hsx._rk_stage_diagnostics_have_finite_bit(diagnostics)
