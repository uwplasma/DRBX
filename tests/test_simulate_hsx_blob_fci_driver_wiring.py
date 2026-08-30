"""Focused driver-contract tests for the selectable HSX FCI path.

These tests intentionally stop before expensive metric fitting or time
integration.  They verify the driver owns the selection, map validation, and
explicit map operand plumbing; the mapped operator implementation is tested
in the DRBX library tests.
"""

import ast
from dataclasses import dataclass
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


def _install_toroidal_rlp_mocks(monkeypatch, hsx):
    topology = SimpleNamespace(
        is_active_owner=np.ones((4, 8, 12), dtype=bool),
        is_merge_source=np.zeros((4, 8, 12), dtype=bool),
    )
    host = SimpleNamespace(
        angular_group_size=np.asarray((8, 4, 2, 1)),
        topology=topology,
    )
    descriptor = object()
    fields = np.zeros((4, 8, 12, hsx.RLP_PACKED_FIELD_COUNT))
    monkeypatch.setattr(
        hsx,
        "build_metric_aware_polar_angular_agglomeration_geometry",
        lambda *_a, **_k: (host, 0.5),
    )
    monkeypatch.setattr(
        hsx,
        "build_sharded_polar_angular_agglomeration_payload",
        lambda *_a, **_k: (descriptor, fields),
    )
    return host, descriptor


def test_parser_exposes_coordinate_default_and_fci_trace_controls():
    hsx = _driver_module()
    parser = hsx._build_parser()
    args = parser.parse_args([])
    assert args.parallel_operator_scheme == "coordinate"
    assert args.fci_parallel_leg_scheme == "centered"
    assert args.fci_trace_substeps == 4
    assert args.vorticity_current_inflow_trace == "operator"
    assert args.gmres_residual_correction_steps == 1
    assert not args.rhs_replay_electron_force_wall_audit
    assert args.curvature_transition_audit_face is None
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
    leg_action = next(
        action
        for action in parser._actions
        if "--fci-parallel-leg-scheme" in action.option_strings
    )
    assert leg_action.choices == ("centered", "boundary-characteristic-upwind")
    trace_action = next(
        action
        for action in parser._actions
        if "--fci-trace-substeps" in action.option_strings
    )
    assert trace_action.type("7") == 7
    vorticity_trace_action = next(
        action
        for action in parser._actions
        if "--vorticity-current-inflow-trace" in action.option_strings
    )
    assert vorticity_trace_action.choices == (
        "operator",
        "parallel-characteristic",
    )
    electron_force_action = next(
        action
        for action in parser._actions
        if "--rhs-replay-electron-force-wall-audit" in action.option_strings
    )
    assert electron_force_action.default is False
    curvature_action = next(
        action
        for action in parser._actions
        if "--curvature-rlp-face-scheme" in action.option_strings
    )
    assert "fine-glue-characteristic" in curvature_action.choices
    assert "fine-glue-characteristic-bulk" in curvature_action.choices


def test_characteristic_fine_glue_requires_all_coupled_equations_before_geometry():
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(
            [
                "--curvature-rlp-face-scheme",
                "fine-glue-characteristic",
                "--curvature-equations",
                "density",
                "Te",
            ]
        )
    assert error.value.code == 2


def test_bulk_characteristic_rejects_single_transition_audit_before_geometry():
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(
            [
                "--curvature-rlp-face-scheme",
                "fine-glue-characteristic-bulk",
                "--curvature-transition-audit-face",
                "1",
            ]
        )
    assert error.value.code == 2


def test_fci_requires_toroidal_topology_before_geometry(monkeypatch):
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(["--parallel-operator-scheme", "fci"])
    assert error.value.code == 2


def test_transition_audit_face_requires_the_audit_mode_before_geometry():
    hsx = _driver_module()
    with pytest.raises(SystemExit) as error:
        hsx.main(["--curvature-transition-audit-face", "1"])
    assert error.value.code == 2


def test_transition_audit_baseline_clears_the_fine_glue_face_selector():
    hsx = _driver_module()

    @dataclass(frozen=True)
    class Candidate:
        curvature_rlp_face_scheme: str
        curvature_rlp_fine_glue_transition_face: int | None
        marker: str

    baseline = hsx._curvature_transition_audit_baseline(
        Candidate("fine-glue-sat", 1, "frozen-frame-75")
    )
    assert baseline.curvature_rlp_face_scheme == "projected-fine"
    assert baseline.curvature_rlp_fine_glue_transition_face is None
    assert baseline.marker == "frozen-frame-75"


def test_every_geometry_assembling_kernel_has_a_map_operand_and_spec():
    tree = _tree()
    run = _function(tree, "run_full_eb")
    kernel_names = {
        "precompute_wall_projectors",
        "reconstruct_initial_phi",
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
        if name != "precompute_wall_projectors":
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
            "--fci-parallel-leg-scheme",
            "boundary-characteristic-upwind",
            "--parallel-inflow-closure",
            "equilibrium-characteristic",
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


def test_metric_context_reuses_continuous_evaluator_across_resolutions(monkeypatch):
    """The explicit context path samples grids without refitting HSX metrics."""

    hsx = _driver_module()
    logical_u = np.linspace(0.0, 1.0, 5)
    logical_theta = 2.0 * np.pi * np.arange(8) / 8.0
    logical_eta = np.pi * np.arange(6) / 6.0

    class FakeMetricEvaluator:
        topology = "toroidal"
        nfp = 2
        period = np.pi
        u = logical_u
        v = logical_theta
        eta = logical_eta

        def evaluate(self, points, **_kwargs):
            shape = np.asarray(points).shape[:-1]
            tensor = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
            tensor[..., 1, 1] = 1.2
            tensor[..., 2, 2] = 1.4
            return SimpleNamespace(
                signed_J=np.ones(shape),
                g_contra=tensor,
                g_cov=tensor,
                position=np.zeros(shape + (3,)),
            )

        def evaluate_magnetic_field(self, points, _bfield, **_kwargs):
            shape = np.asarray(points).shape[:-1]
            return SimpleNamespace(
                B_contravariant=np.broadcast_to(
                    np.asarray((1.0, 0.0, 0.1)), shape + (3,)
                ).copy(),
                magnitude=np.ones(shape),
            )

    class FakeBField:
        nfp = 2

    # The production class check remains meaningful while keeping this test
    # independent of the expensive real HSX metric construction.
    monkeypatch.setattr(hsx, "MetricEvaluator", FakeMetricEvaluator)
    context = hsx.HSXMetricContext(FakeMetricEvaluator(), FakeBField(), 2)
    refit_calls = []
    monkeypatch.setattr(
        hsx,
        "build_hsx_metric_evaluator",
        lambda **_kwargs: refit_calls.append(True),
    )

    common = dict(
        makegrid_path=Path("/tmp/unused-makegrid"),
        vessel_path=Path("/tmp/unused-vessel"),
        fit_sample_shape=(4, 4, 4),
        radial_degree=2,
        vertical_degree=2,
        toroidal_modes=2,
        metric_spline_degree=3,
        mmpde_iterations=0,
        axis_core_radius=0.1,
        reference_magnetic_field=1.0,
        topology="toroidal",
        metric_mesh_shape=(5, 8, 6),
        metric_cache_dir=None,
        metric_context=context,
    )
    for radial_cells in (4, 6, 8):
        geometry, *_ = hsx.build_hsx_fci_geometry(
            resolution=(radial_cells, 8, 12), **common
        )
        assert geometry.shape == (radial_cells, 8, 12)
    assert not refit_calls


@pytest.mark.parametrize(
    ("face_scheme", "audit_face"),
    (("fine-glue-characteristic", 1), ("fine-glue-characteristic-bulk", None)),
)
def test_fci_main_passes_scheme_and_metadata_to_run(
    monkeypatch, tmp_path, face_scheme, audit_face
):
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
            "--fci-parallel-leg-scheme",
            "boundary-characteristic-upwind",
            "--parallel-inflow-closure",
            "equilibrium-characteristic",
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
            "--curvature-scheme",
            "conservative",
            "--curvature-rlp-face-scheme",
            face_scheme,
        ]
        + (
            [
                "--curvature-transition-audit-output",
                str(tmp_path / "audit.npz"),
                "--curvature-transition-audit-face",
                str(audit_face),
            ]
            if audit_face is not None
            else []
        )
    )
    assert calls
    assert calls[0]["parallel_operator_scheme"] == "fci"
    assert calls[0]["fci_parallel_leg_scheme"] == "boundary-characteristic-upwind"
    assert calls[0]["run_metadata"]["parallel_operator_scheme"] == "fci"
    assert (
        calls[0]["run_metadata"]["fci_parallel_leg_scheme"]
        == "boundary-characteristic-upwind"
    )
    assert calls[0]["run_metadata"]["fci_trace_substeps"] == 6
    assert calls[0]["curvature_rlp_fine_glue_transition_face"] == audit_face
    assert calls[0]["curvature_rlp_face_scheme"] == face_scheme
    assert calls[0]["run_metadata"]["curvature_transition_audit_face"] == audit_face
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
            "--flux-framework", "production-split",
            "--parallel-operator-scheme", "fci",
            "--parallel-flux-pairing", "support-core",
            "--curvature-scheme", "conservative",
            "--curvature-rlp-face-scheme", "projected-fine",
            "--poisson-bracket-scheme", "compatible-flux",
        ]
    )
    hsx._validate_flux_framework(args)

    args = parser.parse_args(
        [
            "--flux-framework", "production-split",
            "--parallel-operator-scheme", "fci",
            "--parallel-flux-pairing", "legacy",
            "--curvature-rlp-face-scheme", "projected-fine",
        ]
    )
    with pytest.raises(ValueError, match="support-core"):
        hsx._validate_flux_framework(args)


def test_production_split_metadata_contract_is_recorded():
    source = DRIVER_PATH.read_text()
    assert '"flux_framework": str(args.flux_framework)' in source
    assert '"flux_framework_source": "simulate_hsx_blob.py:--flux-framework"' in source
    assert '"curvature_split_scheme": os.environ.get("DRBX_CURVATURE_SPLIT_SCHEME")' in source
    assert '"parallel_material_scheme": os.environ.get("DRBX_PARALLEL_MATERIAL_SCHEME")' in source
    assert '"production_characteristic_solver": (' in source
    assert '"canonical-face-state"' in source
    assert '"fixed production method"' in source
    assert "DRBX_PRODUCTION_CHARACTERISTIC_SOLVER" not in source
