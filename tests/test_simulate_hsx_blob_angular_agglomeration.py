"""Driver-level production angular-RLP tests."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simulate_hsx_blob as driver
from drbx.geometry import fci_geometry
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry


def _synthetic_geometry(*, nt=8):
    nu, ne = 4, 4
    x_faces = np.linspace(0.0, 1.0, nu + 1)
    y_faces = np.linspace(0.0, 2.0 * np.pi, nt + 1)
    z_faces = np.linspace(0.0, 2.0 * np.pi, ne + 1)
    grid = SimpleNamespace(
        x=SimpleNamespace(faces=x_faces, centers=0.5 * (x_faces[:-1] + x_faces[1:])),
        y=SimpleNamespace(faces=y_faces, centers=0.5 * (y_faces[:-1] + y_faces[1:])),
        z=SimpleNamespace(faces=z_faces, centers=0.5 * (z_faces[:-1] + z_faces[1:])),
    )
    return SimpleNamespace(grid=grid)


def test_removed_axis_experiment_cli_options_are_absent():
    parser = driver._build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "axis_treatment" not in destinations
    assert "pole_owner_profile" not in destinations
    assert "pole_collapsed_radial_rings" not in destinations
    assert "phi_solver_space" not in destinations
    assert "axis_core_state_space" not in destinations
    assert "axis-core-line-u" not in next(
        action.choices for action in parser._actions
        if action.dest == "gmres_preconditioner"
    )


def test_radius_dependent_profile_accepts_composite_ntheta():
    geometry = _synthetic_geometry(nt=6)

    class Evaluator:
        def evaluate(self, points):
            metric = np.broadcast_to(
                np.eye(3), np.asarray(points).shape[:-1] + (3, 3)
            ).copy()
            return SimpleNamespace(g_cov=metric)

    profile, ratio = fci_geometry.metric_aware_angular_group_profile(
        geometry, Evaluator()
    )
    assert profile[0] == 6
    assert np.all(6 % profile == 0)
    assert np.all(profile[:-1] % profile[1:] == 0)
    assert ratio >= 1.0


def test_parser_exposes_metric_checked_diagnostic_angular_profile():
    parser = driver._build_parser()
    args = parser.parse_args(
        [
            "--topology",
            "toroidal",
            "--resolution",
            "4",
            "8",
            "4",
            "--angular-group-profile",
            "8,4,2,1",
        ]
    )
    assert args.angular_group_profile == "8,4,2,1"


def test_main_rejects_angular_profile_outside_toroidal_topology():
    with np.testing.assert_raises(SystemExit):
        driver.main(["--angular-group-profile", "8,4,2,1,1,1,1,1"])


def test_parser_exposes_manufactured_curvature_audit_output(tmp_path):
    output = tmp_path / "manufactured_curvature.npz"
    args = driver._build_parser().parse_args(
        [
            "--curvature-scheme",
            "conservative",
            "--curvature-manufactured-output",
            str(output),
        ]
    )
    assert args.curvature_manufactured_output == output


def test_main_rejects_manufactured_audit_without_conservative_curvature(tmp_path):
    with np.testing.assert_raises(SystemExit):
        driver.main(
            [
                "--curvature-scheme",
                "disabled",
                "--curvature-manufactured-output",
                str(tmp_path / "manufactured_curvature.npz"),
            ]
        )


def test_angular_cache_roundtrip_contains_only_rlp_payload(tmp_path):
    geometry = _synthetic_geometry()
    host = build_polar_angular_agglomeration_geometry(
        geometry.grid.x.faces,
        geometry.grid.y.faces,
        geometry.grid.z.faces,
        lambda points: np.ones(np.asarray(points).shape[:-1]),
        quadrature_order=2,
        angular_group_size=(8, 4, 2, 1),
    )
    path = tmp_path / "angular.npz"
    fci_geometry.write_angular_agglomeration_host_geometry_cache(path, host)
    loaded = fci_geometry.load_angular_agglomeration_host_geometry_cache(
        path, np.array([8, 4, 2, 1])
    )
    assert loaded is not None
    assert np.array_equal(loaded.angular_group_size, host.angular_group_size)
    assert np.array_equal(loaded.topology.owner_index, host.topology.owner_index)
    with np.load(path, allow_pickle=False) as cached:
        assert int(cached["format_version"]) == 3
        assert "host_face_observation_count" not in cached.files
        assert "host_face_design_matrix_condition" not in cached.files


def test_driver_has_one_canonical_toroidal_lowering():
    source = open(driver.__file__, encoding="utf-8").read()
    assert "build_sharded_polar_angular_agglomeration_payload" in source
    assert "assemble_local_polar_angular_agglomeration_geometry" in source
    main_source = source[source.index("def main("):]
    assert "lower_polar_angular_agglomeration_geometry" not in main_source
    assert "lower_pole_control_volume_geometry(" not in main_source
    assert "control_volume_operator_mode" not in source
