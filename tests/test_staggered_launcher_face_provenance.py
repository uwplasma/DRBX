"""Static contracts for the isolated staggered launcher transformation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py"
    spec = importlib.util.spec_from_file_location("staggered_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_transformation_carries_actual_face_topology_to_output_and_diagnostics():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )
    compile(transformed, str(launcher.SHARED_DRIVER), "exec")

    assert "def staggered_face_preflight(" in transformed
    assert "build_local_outgoing_fci_face_topology_from_geometry(" in transformed
    assert '"face_owner_map_sha256"' in transformed
    assert '"face_measure_sha256"' in transformed
    assert '"face_provenance_sha256"' in transformed
    assert "OUTGOING_FCI_FACE_OWNERSHIP_POLICY" in transformed
    assert '"cell_velocity_projection": "PcRc-after-face-to-center"' in transformed
    assert '"face_native_parallel_forces": "direct-Gc-and-compatible-Dc"' in transformed
    assert '"center_force_to_face_transfer": "Pe-L-Rc-mass-adjoint-f2c"' in transformed
    assert '"initial_velocity_projection": "center-to-outgoing-face-Re"' in transformed
    assert "edge_destination_support" in transformed
    assert "outgoing_face_topology_host" in transformed
    assert "prolong_local_outgoing_fci_face_owner_field(current_state.Vi, face)" in transformed
    assert 'f"face_topology_{name}"' in transformed
    assert "values[owner_i, owner_j, owner_k]" in transformed
    assert "_materialize_face_owner_array" in transformed
    assert "face_mask if name in (\"Vi\", \"Ve\") else cell_mask" in transformed
    assert "face_owner_count_basis" not in transformed
    assert '"vorticity_current_flux_divergence",' in transformed

    diagnostic_calls = [
        node for node in ast.walk(ast.parse(transformed))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "diagnostic_state"
    ]
    assert diagnostic_calls
    assert all(
        len(node.args) == 3
        and isinstance(node.args[2], ast.Attribute)
        and node.args[2].attr == "outgoing_face_topology"
        for node in diagnostic_calls
    )

def test_staggered_initializer_preserves_centered_velocities_until_local_c2f_restriction():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )

    # Scalars retain the shared host P_cR_c initialization.  Vi/Ve deliberately
    # bypass it so their actual local FCI map and remote halo rows can perform
    # c2f then R_e; this is essential for any future nonzero velocity seed.
    assert 'and name in ("Vi", "Ve")' in transformed
    assert "result[name] = raw" in transformed
    assert "project_initial_staggered_velocities" in transformed
    assert "model._center_owned_to_outgoing_face(values, bc, context)" in transformed
    assert "model._restrict_fine_face_field(" in transformed
    assert "model._owner_face_field(" in transformed
    assert "_assert_owner_sparse(\n                state, owner_host_geometry, outgoing_face_topology_host" in transformed


def test_rhs_term_history_diagnostic_is_read_only_and_uses_mixed_materialization():
    launcher = _launcher_module()
    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text(encoding="utf-8")
    )

    assert "DRBX_RHS_TERM_HISTORY" in transformed
    assert "return_rhs_term_fields=True" in transformed
    assert "phi_owned=local_state.phi" in transformed
    assert "prolong_local_outgoing_fci_face_owner_field(value, model.outgoing_face_topology)" in transformed
    assert "return materialized.at[3].set(vi_face_terms).at[4].set(ve_face_terms)" in transformed
    assert "_vi_near_band_report(vi_terms, saved_vi, near_start)" in transformed
    assert "return state" in transformed


def test_rhs_term_history_arguments_are_launcher_only_and_guarded():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--rhs-term-history", type=Path)' in source
    assert 'parser.add_argument("--rhs-term-frames", default="100,180,225")' in source
    assert 'parser.add_argument("--rhs-term-output", type=Path)' in source
    assert "--rhs-term-history requires --parallel-velocity-layout fci-staggered" in source
    assert "--rhs-term-history requires --rhs-term-output" in source


def test_parallel_flux_pairing_is_launcher_controlled_and_recorded_in_metadata():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument(\n        "--parallel-flux-pairing",' in source
    assert 'choices=("legacy", "support-core")' in source
    assert 'default="legacy"' in source
    assert "support-core requires --parallel-velocity-layout cell-centered" in source
    assert "support-core requires --parallel-operator-scheme fci" in source
    assert 'os.environ["DRBX_PARALLEL_FLUX_PAIRING"] = args.parallel_flux_pairing' in source
    assert 'os.environ["DRBX_SOURCE_ROOT"] = str(WORKTREE / "src")' in source
    assert '"parallel_flux_pairing": os.environ.get("DRBX_PARALLEL_FLUX_PAIRING", "legacy")' in source
    assert 'parser.add_argument(\n        "--parallel-boundary-pairing",' in source
    assert 'choices=("legacy", "current-phi")' in source
    assert '"parallel_boundary_pairing": os.environ.get("DRBX_PARALLEL_BOUNDARY_PAIRING", "legacy")' in source


def test_production_boundary_pairing_is_defaulted_exported_and_replay_ablatable(
    monkeypatch, capsys
):
    launcher = _launcher_module()
    base = [
        str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
        "--flux-framework", "production-split",
        "--parallel-flux-pairing", "support-core",
        "--parallel-operator-scheme", "fci",
        "--topology", "toroidal",
    ]
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            'assert os.environ["DRBX_PARALLEL_BOUNDARY_PAIRING"] == "current-phi"\n'
        ),
    )
    monkeypatch.setattr(sys, "argv", base)
    launcher.main()
    assert "parallel_boundary_pairing=current-phi" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", base + ["--parallel-boundary-pairing", "legacy"])
    with pytest.raises(SystemExit, match="trajectories require.*current-phi"):
        launcher.main()

    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            'assert os.environ["DRBX_PARALLEL_BOUNDARY_PAIRING"] == "legacy"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "--parallel-boundary-pairing", "legacy",
            "--rhs-replay-history", "/tmp/frozen-history.npz",
        ],
    )
    launcher.main()


def test_flux_framework_parser_and_production_environment_provenance(monkeypatch, capsys):
    launcher = _launcher_module()
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert '"--flux-framework"' in source
    assert 'choices=("legacy", "production-split")' in source
    assert 'os.environ["DRBX_FLUX_FRAMEWORK"] = args.flux_framework' in source
    assert 'os.environ["DRBX_CURVATURE_SPLIT_SCHEME"] = "production-path"' in source
    assert 'os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] = "production-path"' in source
    assert '"curvature_wall_flux_closure": os.environ.get(' in source
    assert '"parallel_material_wall_flux_closure": os.environ.get(' in source

    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            'assert os.environ["DRBX_FLUX_FRAMEWORK"] == "production-split"\n'
            'assert os.environ["DRBX_CURVATURE_SPLIT_SCHEME"] == "production-path"\n'
            'assert os.environ["DRBX_PARALLEL_MATERIAL_SCHEME"] == "production-path"\n'
            'assert os.environ["DRBX_CURVATURE_WALL_FLUX_CLOSURE"] == "equilibrium-exterior-osher"\n'
            'assert os.environ["DRBX_PARALLEL_MATERIAL_WALL_FLUX_CLOSURE"] == "characteristic-projected-operator-trace-osher"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--flux-framework", "production-split",
            "--parallel-flux-pairing", "support-core",
            "--parallel-operator-scheme", "fci",
            "--topology", "toroidal",
        ],
    )
    launcher.main()
    output = capsys.readouterr().out
    assert "flux_framework=production-split" in output
    assert "curvature_wall_flux_closure=equilibrium-exterior-osher" in output
    assert (
        "parallel_material_wall_flux_closure="
        "characteristic-projected-operator-trace-osher"
    ) in output


def test_flux_framework_production_guard_rejects_legacy_pairing(monkeypatch):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--flux-framework", "production-split",
            "--parallel-operator-scheme", "fci",
            "--topology", "toroidal",
        ],
    )
    with pytest.raises(SystemExit, match="support-core"):
        launcher.main()


def test_production_current_trace_ablation_is_replay_only(monkeypatch):
    launcher = _launcher_module()
    base = [
        str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
        "--flux-framework", "production-split",
        "--parallel-flux-pairing", "support-core",
        "--parallel-operator-scheme", "fci",
        "--topology", "toroidal",
        "--vorticity-current-inflow-trace", "parallel-characteristic",
    ]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit, match="boundary-only current correction"):
        launcher.main()

    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text()
    )
    assert (
        'str(args.vorticity_current_inflow_trace) != "operator"\n'
        "        and args.rhs_replay_history is None"
    ) in transformed
    monkeypatch.setattr(launcher, "_transform_shared_driver_source", lambda _: "pass\n")
    monkeypatch.setattr(
        sys,
        "argv",
        base + ["--rhs-replay-history", "/tmp/frozen-history.npz"],
    )
    launcher.main()


def test_curvature_characteristic_axes_is_launcher_controlled_and_recorded_in_metadata():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument(\n        "--curvature-characteristic-axes",' in source
    assert 'choices=("legacy", "radial", "radial-poloidal")' in source
    assert 'default="legacy"' in source
    assert 'os.environ["DRBX_CURVATURE_CHARACTERISTIC_AXES"] = args.curvature_characteristic_axes' in source
    assert '"curvature_characteristic_axes": os.environ.get("DRBX_CURVATURE_CHARACTERISTIC_AXES", "legacy")' in source
    assert '"curvature_characteristic_axes_source": "run_staggered_hsx_blob.py:--curvature-characteristic-axes"' in source
    assert '"--poloidal-characteristic-penalty"' in source
    assert 'os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY"] = str(poloidal_penalty)' in source
    assert '"poloidal_characteristic_penalty_source"' in source


def test_curvature_radial_provenance_replay_is_exported_and_self_describing(
    monkeypatch, capsys
):
    launcher = _launcher_module()
    source = (launcher.WORKTREE / "run_staggered_hsx_blob.py").read_text()
    assert '"--curvature-component-diagnostic-scheme"' in source
    assert '("directional", "centered-dissipation", "radial-provenance")' in source
    assert 'os.environ["DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME"]' in source

    transformed = launcher._transform_shared_driver_source(
        launcher.SHARED_DRIVER.read_text()
    )
    assert "curvature_component_diagnostic_names()" in transformed
    assert '"curvature_component_diagnostic_scheme"' in transformed

    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            'assert os.environ["DRBX_CURVATURE_COMPONENT_DIAGNOSTIC_SCHEME"] '
            '== "radial-provenance"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-component-diagnostic-scheme",
            "radial-provenance",
            "--rhs-replay-history",
            "/tmp/frozen-history.npz",
        ],
    )
    launcher.main()
    assert "curvature_component_diagnostic_scheme=radial-provenance" in capsys.readouterr().out


def test_curvature_radial_provenance_requires_frozen_replay(monkeypatch):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-component-diagnostic-scheme",
            "radial-provenance",
        ],
    )
    with pytest.raises(SystemExit, match="require --rhs-replay-history"):
        launcher.main()


def test_radial_characteristic_scheme_defaults_to_legacy_and_records_cli_provenance():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument(\n        "--curvature-radial-characteristic-scheme",' in source
    assert 'choices=("legacy", "third-order-upwind")' in source
    assert 'default="legacy"' in source
    assert 'os.environ["DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME"] = (' in source
    assert '"curvature_radial_characteristic_scheme": os.environ.get("DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME", "legacy")' in source
    assert '"curvature_radial_characteristic_scheme_source": "run_staggered_hsx_blob.py:--curvature-radial-characteristic-scheme"' in source


def test_poloidal_characteristic_scheme_defaults_to_legacy_and_records_cli_provenance():
    source = (Path(__file__).resolve().parents[1] / "run_staggered_hsx_blob.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument(\n        "--curvature-poloidal-characteristic-scheme",' in source
    assert 'choices=("legacy", "third-order-upwind")' in source
    assert 'default="legacy"' in source
    assert 'os.environ["DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME"] = (' in source
    assert '"curvature_poloidal_characteristic_scheme": os.environ.get("DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME", "legacy")' in source
    assert '"curvature_poloidal_characteristic_scheme_source": "run_staggered_hsx_blob.py:--curvature-poloidal-characteristic-scheme"' in source


def test_third_order_radial_scheme_is_exported_and_announced(monkeypatch, capsys):
    launcher = _launcher_module()
    monkeypatch.setenv("DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME", "test-sentinel")
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            'assert os.environ["DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME"] == "third-order-upwind"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-radial-characteristic-scheme",
            "third-order-upwind",
            "--curvature-scheme",
            "conservative",
            "--topology",
            "toroidal",
            "--curvature-rlp-face-scheme",
            "projected-fine",
        ],
    )

    launcher.main()

    assert "curvature_radial_characteristic_scheme=third-order-upwind" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "expected_scheme"),
    (
        ((), "legacy"),
        (
            (
                "--curvature-poloidal-characteristic-scheme",
                "third-order-upwind",
                "--curvature-radial-characteristic-scheme",
                "third-order-upwind",
                "--curvature-scheme",
                "conservative",
                "--topology",
                "toroidal",
                "--curvature-rlp-face-scheme",
                "projected-fine",
            ),
            "third-order-upwind",
        ),
    ),
)
def test_poloidal_characteristic_scheme_is_exported_and_announced(
    monkeypatch, capsys, arguments, expected_scheme
):
    launcher = _launcher_module()
    monkeypatch.setenv("DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME", "test-sentinel")
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            f'assert os.environ["DRBX_CURVATURE_POLOIDAL_CHARACTERISTIC_SCHEME"] == "{expected_scheme}"\n'
            f'assert os.environ["DRBX_CURVATURE_RADIAL_CHARACTERISTIC_SCHEME"] == "{expected_scheme}"\n'
            'assert os.environ["DRBX_CURVATURE_CHARACTERISTIC_AXES"] == "legacy"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launcher.WORKTREE / "run_staggered_hsx_blob.py"), *arguments],
    )

    launcher.main()

    announcement = capsys.readouterr().out
    assert f"curvature_poloidal_characteristic_scheme={expected_scheme}" in announcement
    assert f"curvature_radial_characteristic_scheme={expected_scheme}" in announcement
    assert "curvature_characteristic_axes=legacy" in announcement


def test_third_order_poloidal_scheme_requires_matching_radial_scheme(monkeypatch):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-poloidal-characteristic-scheme",
            "third-order-upwind",
        ],
    )

    with pytest.raises(
        SystemExit,
        match=(
            "third-order-upwind poloidal curvature requires "
            "--curvature-radial-characteristic-scheme third-order-upwind"
        ),
    ):
        launcher.main()


@pytest.mark.parametrize(
    "extra, message",
    (
        (
            ("--topology", "toroidal", "--curvature-rlp-face-scheme", "projected-fine"),
            "third-order-upwind radial curvature requires --curvature-scheme conservative",
        ),
        (
            ("--curvature-scheme", "conservative", "--curvature-rlp-face-scheme", "projected-fine"),
            "third-order-upwind radial curvature requires --topology toroidal",
        ),
        (
            (
                "--curvature-scheme", "conservative", "--topology", "toroidal",
                "--curvature-rlp-face-scheme", "projected-fine",
                "--curvature-characteristic-axes", "radial",
            ),
            "third-order-upwind radial curvature requires --curvature-characteristic-axes legacy",
        ),
        (
            (
                "--curvature-scheme", "conservative", "--topology", "toroidal",
                "--curvature-rlp-face-scheme", "fine-glue-characteristic-bulk",
            ),
            "third-order-upwind radial curvature requires --curvature-rlp-face-scheme projected-fine",
        ),
        (
            ("--curvature-scheme", "conservative", "--topology", "toroidal"),
            "third-order-upwind radial curvature requires --curvature-rlp-face-scheme projected-fine",
        ),
    ),
)
def test_third_order_radial_scheme_rejects_incompatible_runtime_options(
    monkeypatch, extra, message
):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-radial-characteristic-scheme",
            "third-order-upwind",
            *extra,
        ],
    )

    with pytest.raises(SystemExit, match=message):
        launcher.main()


@pytest.mark.parametrize(
    ("arguments", "expected_pairing"),
    (
        ((), "legacy"),
        (
            (
                "--parallel-flux-pairing",
                "support-core",
                "--parallel-operator-scheme",
                "fci",
            ),
            "support-core",
        ),
    ),
)
def test_parallel_flux_pairing_is_exported_and_announced(
    monkeypatch, capsys, arguments, expected_pairing
):
    launcher = _launcher_module()
    monkeypatch.setenv("DRBX_PARALLEL_FLUX_PAIRING", "test-sentinel")
    monkeypatch.setenv("DRBX_PARALLEL_VELOCITY_LAYOUT", "test-sentinel")
    monkeypatch.setenv("DRBX_SOURCE_ROOT", "test-sentinel")
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            f'assert os.environ["DRBX_PARALLEL_FLUX_PAIRING"] == "{expected_pairing}"\n'
            f'assert os.environ["DRBX_SOURCE_ROOT"] == {str(launcher.WORKTREE / "src")!r}\n'
        ),
    )
    monkeypatch.setattr(sys, "argv", [str(launcher.WORKTREE / "run_staggered_hsx_blob.py"), *arguments])

    launcher.main()

    assert f"parallel_flux_pairing={expected_pairing}" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ("legacy", "radial", "radial-poloidal"))
def test_curvature_characteristic_axes_is_exported_and_announced(
    monkeypatch, capsys, mode
):
    launcher = _launcher_module()
    monkeypatch.setenv("DRBX_CURVATURE_CHARACTERISTIC_AXES", "test-sentinel")
    monkeypatch.setenv("DRBX_PARALLEL_VELOCITY_LAYOUT", "test-sentinel")
    monkeypatch.setenv("DRBX_PARALLEL_FLUX_PAIRING", "test-sentinel")
    monkeypatch.setenv("DRBX_SOURCE_ROOT", "test-sentinel")
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            f'assert os.environ["DRBX_CURVATURE_CHARACTERISTIC_AXES"] == "{mode}"\n'
        ),
    )
    args = ("--curvature-characteristic-axes", mode)
    if mode == "radial-poloidal":
        args += (
            "--curvature-rlp-face-scheme",
            "fine-glue-characteristic-bulk",
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launcher.WORKTREE / "run_staggered_hsx_blob.py"), *args],
    )

    launcher.main()

    assert f"curvature_characteristic_axes={mode}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra, message",
    (
        (
            ("--curvature-scheme", "direct"),
            "radial-poloidal curvature characteristics require --curvature-scheme conservative",
        ),
        (
            (),
            "radial-poloidal curvature characteristics require --curvature-rlp-face-scheme",
        ),
    ),
)
def test_curvature_characteristic_axes_rejects_incompatible_runtime_options(
    monkeypatch, extra, message
):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--curvature-characteristic-axes",
            "radial-poloidal",
            *extra,
        ],
    )

    with pytest.raises(SystemExit, match=message):
        launcher.main()


@pytest.mark.parametrize(
    ("arguments", "expected", "source"),
    (
        (
            ("--curvature-rlp-fine-glue-penalty", "0.65"),
            "0.65",
            "inherited-from-curvature-rlp-fine-glue-penalty",
        ),
        (
            (
                "--curvature-rlp-fine-glue-penalty",
                "0.65",
                "--poloidal-characteristic-penalty",
                "0.2",
            ),
            "0.2",
            "run_staggered_hsx_blob.py:--poloidal-characteristic-penalty",
        ),
    ),
)
def test_poloidal_characteristic_penalty_is_exported_with_inheritance(
    monkeypatch, capsys, arguments, expected, source
):
    launcher = _launcher_module()
    monkeypatch.delenv("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY", raising=False)
    monkeypatch.delenv("DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE", raising=False)
    monkeypatch.setattr(
        launcher,
        "_transform_shared_driver_source",
        lambda _: (
            "import os\n"
            f'assert os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY"] == "{expected}"\n'
            f'assert os.environ["DRBX_POLOIDAL_CHARACTERISTIC_PENALTY_SOURCE"] == "{source}"\n'
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launcher.WORKTREE / "run_staggered_hsx_blob.py"), *arguments],
    )

    launcher.main()

    announcement = capsys.readouterr().out
    assert f"poloidal_characteristic_penalty={expected}" in announcement
    assert source in announcement


def test_poloidal_characteristic_penalty_rejects_negative_value(monkeypatch):
    launcher = _launcher_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.WORKTREE / "run_staggered_hsx_blob.py"),
            "--poloidal-characteristic-penalty",
            "-0.1",
        ],
    )

    with pytest.raises(SystemExit, match="poloidal-characteristic-penalty must be finite and nonnegative"):
        launcher.main()


@pytest.mark.parametrize(
    "arguments, message",
    (
        (
            (
                "--parallel-flux-pairing",
                "support-core",
                "--parallel-velocity-layout",
                "fci-staggered",
                "--parallel-operator-scheme",
                "fci",
            ),
            "support-core requires --parallel-velocity-layout cell-centered",
        ),
        (
            ("--parallel-flux-pairing", "support-core"),
            "support-core requires --parallel-operator-scheme fci",
        ),
    ),
)
def test_parallel_flux_pairing_rejects_incompatible_runtime_options(monkeypatch, arguments, message):
    launcher = _launcher_module()
    monkeypatch.setattr(sys, "argv", [str(launcher.WORKTREE / "run_staggered_hsx_blob.py"), *arguments])

    with pytest.raises(SystemExit, match=message):
        launcher.main()


def test_vi_near_band_report_has_consistent_energy_and_complex_inner_products():
    launcher = _launcher_module()
    # theta=4; retain modes 1 and 2.  The first term equals the state, while
    # the second is its negative, so their sum cancels exactly in the band.
    state = np.asarray([[[1.0], [-1.0], [1.0], [-1.0]]])
    terms = np.stack((state, -state), axis=0)
    report = launcher._vi_near_band_report(terms, state, near_start=1)
    state_spectrum = np.fft.rfft(state, axis=1)[:, 1:, :]
    expected_energy = float(np.sum(np.abs(state_spectrum) ** 2))
    assert report["rfft_normalization"] == "numpy-unnormalized"
    np.testing.assert_allclose(report["term_near_band_energy"], (expected_energy, expected_energy))
    assert report["term_near_band_inner_product_with_saved_Vi"][0] == {
        "real": expected_energy, "imag": 0.0
    }
    assert report["term_near_band_inner_product_with_saved_Vi"][1] == {
        "real": -expected_energy, "imag": 0.0
    }
    assert report["sum_term_near_band_energy"] == 0.0
    assert report["sum_term_near_band_inner_product_with_saved_Vi"] == {"real": 0.0, "imag": 0.0}
    with pytest.raises(ValueError, match="match state"):
        launcher._vi_near_band_report(np.zeros((2, 1, 4, 1)), np.zeros((1, 3, 1)), 1)
