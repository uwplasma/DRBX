"""Analytic topology tests for the evaluate_mesh reporting layer."""

from dataclasses import asdict

import numpy as np

import evaluate_mesh
from drbx.geometry import MetricEvaluator, ToroidalQualityReport, evaluate_toroidal_quality


class Eta:
    def evaluate_cartesian(self, points, wrapped=False):
        points = np.asarray(points)
        return np.arctan2(points[..., 1], points[..., 0])


class BField:
    def evaluate_cartesian(self, points):
        points = np.asarray(points)
        return np.broadcast_to(np.array([0.0, 1.0, 0.0]), points.shape)


class Wall:
    def constant_eta_boundary_curve(self, eta_evaluator, eta, npoints=256):
        theta = np.linspace(0.0, 2.0 * np.pi, int(npoints), endpoint=True)
        phi = float(eta)
        return np.column_stack((
            (1.0 + 0.2 * np.cos(theta)) * np.cos(phi),
            (1.0 + 0.2 * np.cos(theta)) * np.sin(phi),
            -0.2 * np.sin(theta),
        ))


class WallFactory:
    wall = Wall()

    @classmethod
    def from_file(cls, path):
        return cls.wall


def analytic_metric(nu=5, ntheta=8, neta=8):
    u = np.linspace(0.0, 1.0, nu)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    eta = 2.0 * np.pi * np.arange(neta) / neta
    uu, tt, ee = np.meshgrid(u, theta, eta, indexing="ij")
    R = 1.0 + 0.2 * uu * np.cos(tt)
    Z = -0.2 * uu * np.sin(tt)
    return MetricEvaluator(
        u, theta, eta,
        np.stack((R * np.cos(ee), R * np.sin(ee), Z), axis=-1),
        period=2.0 * np.pi, nfp=1, topology="toroidal",
        radial_degree=4, poloidal_modes=2, toroidal_modes=0,
    )


def test_parser_toroidal_defaults_and_wiring():
    args = evaluate_mesh._build_parser().parse_args([])
    assert args.topology == "square"
    assert args.metric_mesh_shape == (32, 32, 32)
    assert (args.metric_radial_degree, args.metric_poloidal_modes, args.metric_toroidal_modes) == (17, 15, 16)
    assert args.eta_projection_iterations == 0


def test_main_wires_toroidal_builder_options(monkeypatch, tmp_path, capsys):
    captured = {}
    metric = analytic_metric()
    report = ToroidalQualityReport(
        *(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 0, 0.0, 0.0, 0.0, 0.0, 8)
    )

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return metric, Eta(), BField(), 1

    monkeypatch.setattr(evaluate_mesh, "build_hsx_metric_evaluator", fake_builder)
    monkeypatch.setattr(evaluate_mesh, "_toroidal_quality_report", lambda *args: report)
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.touch()
    vessel.touch()
    monkeypatch.setattr(
        "sys.argv", ["evaluate_mesh.py", "--makegrid", str(makegrid), "--vessel", str(vessel),
                      "--topology", "toroidal", "--metric-mesh-shape", "5", "8", "8",
                      "--metric-radial-degree", "4", "--metric-poloidal-modes", "2",
                      "--metric-toroidal-modes", "3", "--eta-projection-iterations", "2",
                      "--no-advanced-diagnostics"],
    )
    evaluate_mesh.main()
    assert captured["topology"] == "toroidal"
    assert captured["metric_mesh_shape"] == (5, 8, 8)
    assert captured["metric_radial_degree"] == 4
    assert captured["metric_poloidal_modes"] == 2
    assert captured["metric_toroidal_modes"] == 3
    assert captured["eta_projection_iterations"] == 2
    assert "toroidal" in capsys.readouterr().out.lower()


def test_toroidal_base_quality_has_wall_and_positive_j():
    metric = analytic_metric()
    original = evaluate_mesh.WallEvaluator
    evaluate_mesh.WallEvaluator = WallFactory
    try:
        report = evaluate_mesh._toroidal_quality_report(metric, Eta(), "analytic", 1)
    finally:
        evaluate_mesh.WallEvaluator = original
    values = asdict(report)
    assert values["wall_error_max"] < 1.0e-10
    assert values["wall_error_rms"] < 1.0e-10
    assert values["seam_residual_max"] < 1.0e-10
    assert values["seam_residual_rms"] < 1.0e-10
    assert values["min_J_reg"] > 0.0
    assert values["nonpositive_J_reg_count"] == 0


def test_advanced_toroidal_theta_quadrature_regions_and_regularized_axis():
    metric = analytic_metric()
    diagnostics = evaluate_mesh._build_advanced_diagnostics(
        metric, BField(), resolution=(4, 8, 8), nfp=1, gauss_order=2,
        parallel_speed=1.0, parallel_diffusivity=0.1,
        perpendicular_diffusivity=0.2, cfl_number=1.0,
        diffusive_cfl_number=0.5,
    )
    assert diagnostics["topology"] == "toroidal"
    assert diagnostics["sampling"]["one_period_cell_counts"] == (4, 8, 8)
    assert set(("u", "theta", "eta")) <= set(diagnostics["directional_widths"])
    assert "corner_d<0.05" not in diagnostics["fixed_logical_regions"]
    interior = diagnostics["fixed_logical_regions"]["interior_0.10<=u<=0.90"]
    assert interior["sample_count"] > 0
    assert interior["sample_count"] < diagnostics["sampling"]["quadrature_points"]
    assert diagnostics["regularized"]["axis"]["sample_count"] == 8 * 8
    assert diagnostics["regularized"]["axis"]["nonpositive_or_nonfinite_count"] == 0


def test_periodic_seam_location_indexing():
    metric = analytic_metric(ntheta=8)
    q = np.array([[0.5, 2.0 * np.pi - 1.0e-8, 1.0]])
    location = evaluate_mesh._diagnostic_location(
        0, np.array([1.0]), q, metric.position(q), np.array([[0, 7, 0]]), metric
    )
    assert location["evaluator_spline_cell"][1] == 7
    assert location["periodic_coordinate"] == "theta"


def test_square_quality_path_remains_distinct():
    u = np.linspace(0.0, 1.0, 4)
    v = np.linspace(0.0, 1.0, 4)
    eta = 2.0 * np.pi * np.arange(8) / 8
    uu, vv, ee = np.meshgrid(u, v, eta, indexing="ij")
    square = MetricEvaluator(
        u, v, eta,
        np.stack(((1.0 + uu) * np.cos(ee), (1.0 + uu) * np.sin(ee), vv), axis=-1),
        period=2.0 * np.pi, nfp=1, topology="square",
    )
    diagnostics = evaluate_mesh._build_advanced_diagnostics(
        square, BField(), resolution=(3, 3, 8), nfp=1, gauss_order=2,
        parallel_speed=1.0, parallel_diffusivity=0.0,
        perpendicular_diffusivity=0.0, cfl_number=1.0,
        diffusive_cfl_number=0.5,
    )
    assert set(("u", "v", "eta")) <= set(diagnostics["directional_widths"])
    assert "corner_d<0.05" in diagnostics["fixed_logical_regions"]
