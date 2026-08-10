"""Focused synthetic coverage for toroidal HSX geometry integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate_hsx_blob as hsx  # noqa: E402


TOPOLOGY_KWARGS = {
    "topology": "toroidal",
    "metric_mesh_shape": (5, 8, 6),
    "metric_radial_degree": 4,
    "metric_poloidal_modes": 3,
    "metric_toroidal_modes": 2,
    "eta_projection_iterations": 1,
}


def _geometry_kwargs(tmp_path: Path, **overrides):
    kwargs = {
        "makegrid_path": tmp_path / "synthetic.nc",
        "vessel_path": tmp_path / "synthetic.vessel",
        "resolution": (4, 8, 12),
        "fit_sample_shape": (3, 4, 3),
        "radial_degree": 2,
        "vertical_degree": 2,
        "toroidal_modes": 2,
        "metric_spline_degree": 1,
        "mmpde_iterations": 0,
        "axis_core_radius": 0.03,
        "reference_magnetic_field": 1.0,
        "metric_cache_dir": None,
        "rebuild_metric_cache": False,
    }
    kwargs.update(TOPOLOGY_KWARGS)
    kwargs.update(overrides)
    return kwargs


class SyntheticMetricEvaluator:
    """Small axis-regular evaluator with explicit parity-bearing channels."""

    def __init__(self, *, nfp: int = 2):
        self.topology = "toroidal"
        self.coordinate_names = ("u", "theta", "eta")
        self.periodic_axes = (False, True, True)
        self.axis_regular_axes = (True, False, False)
        self.u = np.linspace(0.0, 1.0, 5)
        self.v = 2.0 * np.pi * np.arange(8) / 8.0
        self.theta = self.v
        self.eta = np.pi * np.arange(6) / 6.0
        self.period = np.pi
        self.nfp = nfp
        self.radial_degree = 4
        self.poloidal_modes = 3
        self.toroidal_modes = 2
        self.ordinary_metric_calls = []
        self.ordinary_magnetic_calls = []

    def _values(self, points):
        points = np.asarray(points, dtype=float)
        u, theta, eta = np.moveaxis(points, -1, 0)
        radius = 3.0 + 0.5 * u * np.cos(theta)
        position = np.stack(
            (radius * np.cos(eta), radius * np.sin(eta), 0.5 * u * np.sin(theta)),
            axis=-1,
        )
        signed_j = u.copy()
        identity = np.broadcast_to(np.eye(3), points.shape[:-1] + (3, 3)).copy()
        # T g T-compatible mixed terms, with nontrivial theta dependence.
        identity[..., 0, 1] = identity[..., 1, 0] = 0.1 * u * np.cos(theta)
        identity[..., 0, 2] = identity[..., 2, 0] = 0.1 * u * np.sin(theta)
        # The theta-eta component is even under the half-turn.
        identity[..., 1, 2] = identity[..., 2, 1] = 0.05 + 0.01 * u * np.cos(theta)
        return position, signed_j, identity

    def evaluate(self, logical_points):
        points = np.asarray(logical_points, dtype=float)
        self.ordinary_metric_calls.append(points.copy())
        if np.any(np.isclose(points[..., 0], 0.0)):
            raise AssertionError("ordinary metric evaluation received u=0")
        position, signed_j, tensor = self._values(points)
        return SimpleNamespace(
            position=position,
            signed_J=signed_j,
            g_contra=tensor,
            g_cov=tensor,
        )

    def evaluate_magnetic_field(self, logical_points, _bfield):
        points = np.asarray(logical_points, dtype=float)
        self.ordinary_magnetic_calls.append(points.copy())
        if np.any(np.isclose(points[..., 0], 0.0)):
            raise AssertionError("ordinary magnetic evaluation received u=0")
        u, theta, _eta = np.moveaxis(points, -1, 0)
        b_contra = np.stack(
            (0.2 * u * np.sin(theta), 1.0 + 0.1 * np.cos(theta), 0.7 + 0.0 * u),
            axis=-1,
        )
        return SimpleNamespace(
            B_contravariant=b_contra,
            magnitude=np.ones(points.shape[:-1], dtype=float),
        )

    def to_cache_payload(self, *, prefix=""):
        return {
            f"{prefix}topology": np.asarray(self.topology),
            f"{prefix}coordinate_names": np.asarray(self.coordinate_names),
            f"{prefix}periodic_axes": np.asarray(self.periodic_axes),
            f"{prefix}axis_regular_axes": np.asarray(self.axis_regular_axes),
            f"{prefix}nfp": np.asarray(self.nfp, dtype=np.int64),
            f"{prefix}period": np.asarray(self.period),
            f"{prefix}radial_degree": np.asarray(self.radial_degree),
            f"{prefix}poloidal_modes": np.asarray(self.poloidal_modes),
            f"{prefix}toroidal_modes": np.asarray(self.toroidal_modes),
        }


def _fake_metric_builder(spy, evaluator=None):
    evaluator = evaluator or SyntheticMetricEvaluator()

    def build(**kwargs):
        spy.append(kwargs)
        return evaluator, object(), SimpleNamespace(nfp=evaluator.nfp), evaluator.nfp

    return build


def _install_synthetic_metric(monkeypatch, spy, evaluator=None):
    evaluator = evaluator or SyntheticMetricEvaluator()
    monkeypatch.setattr(hsx, "build_hsx_metric_evaluator", _fake_metric_builder(spy, evaluator))
    return evaluator


def test_square_parser_defaults_and_legacy_semantics_remain_intact():
    args = hsx._build_parser().parse_args(())

    assert args.topology == "square"
    assert args.resolution == (8, 8, 8)
    assert args.fit_sample_shape == (8, 9, 8)
    assert args.toroidal_modes == 2
    assert args.metric_spline_degree == 1
    assert args.mmpde_iterations == 0
    assert args.neumann_ghost_scheme == "physical"
    assert args.geometry_only is False
    assert args.axis_core_gradient_degree == 3
    assert args.axis_core_gradient_observation_rings == 6
    assert args.axis_core_gradient_target_rings == 3
    assert args.poisson_bracket_scheme == "direct"


def test_poisson_bracket_scheme_parser_override_and_invalid_value():
    args = hsx._build_parser().parse_args(
        ("--poisson-bracket-scheme", "compatible-flux")
    )
    assert args.poisson_bracket_scheme == "compatible-flux"

    with pytest.raises(SystemExit):
        hsx._build_parser().parse_args(
            ("--poisson-bracket-scheme", "not-a-scheme")
        )


def test_axis_core_line_u_preconditioner_cli_choice():
    args = hsx._build_parser().parse_args(
        ("--gmres-preconditioner", "axis-core-line-u")
    )
    assert args.gmres_preconditioner == "axis-core-line-u"


def test_phi_solver_space_cli_defaults_and_override():
    args = hsx._build_parser().parse_args(())
    assert args.phi_solver_space == "full-grid"

    args = hsx._build_parser().parse_args(
        ("--phi-solver-space", "axis-core-reduced")
    )
    assert args.phi_solver_space == "axis-core-reduced"


def test_phi_solver_diagnostics_preserve_first_four_and_append_reduced_residuals(
):
    info = SimpleNamespace(
        num_steps=7,
        final_residual_rel_l2=1.0e-4,
        failed=True,
        converged=False,
        rhs=SimpleNamespace(total_l2=2.0, incompatible_l2=0.5),
        final_residual=SimpleNamespace(total_l2=0.2, incompatible_l2=0.1),
    )

    reduced = np.asarray(
        hsx._format_phi_solver_diagnostics(info, "axis-core-reduced")
    )
    full_grid = np.asarray(hsx._format_phi_solver_diagnostics(info, "full-grid"))

    assert reduced.shape == (7,)
    np.testing.assert_array_equal(reduced[:4], [7.0, 1.0e-4, 1.0, 0.0])
    np.testing.assert_allclose(reduced[4:], [0.25, 0.05, 0.1])
    assert full_grid.shape == (7,)
    np.testing.assert_array_equal(full_grid[:4], reduced[:4])
    np.testing.assert_array_equal(full_grid[4:], np.zeros(3))


def test_reduced_phi_compatibility_reporting_uses_maximum_and_is_opt_in(capsys):
    stages = np.asarray(
        [
            [10.0, 1.0e-5, 0.0, 1.0, 0.2, 0.03, 0.4],
            [11.0, 2.0e-5, 0.0, 1.0, 0.5, 0.02, 0.6],
            [12.0, 3.0e-5, 0.0, 1.0, 0.3, 0.04, 0.5],
            [13.0, 4.0e-5, 0.0, 1.0, 0.4, 0.01, 0.3],
        ]
    )

    assert hsx._print_reduced_phi_compatibility_diagnostics(
        "axis-core-reduced", stages, step=9
    )
    output = capsys.readouterr().out
    assert "[diagnostics] reduced-phi compatibility:" in output
    assert "step=9" in output
    assert "rhs-incompatible-rel=5.000e-01" in output
    assert "final-incompatible-rel=4.000e-02" in output
    assert "final-full-rel=6.000e-01" in output

    assert not hsx._print_reduced_phi_compatibility_diagnostics(
        "full-grid", stages, step=9
    )
    assert capsys.readouterr().out == ""

    # IMEX host conversion remains on its established four-entry shape and
    # therefore does not accidentally enter the RK compatibility reporter.
    assert not hsx._print_reduced_phi_compatibility_diagnostics(
        "axis-core-reduced", np.zeros((1, 4)), step=9
    )


def test_build_local_eb_model_reduced_space_is_built_once_and_attached(monkeypatch):
    captured = {}

    @dataclass(frozen=True)
    class FakeRhs:
        phi_solver_space: str = "full-grid"
        axis_core_reduced_space: object | None = None

        def __init__(self, **kwargs):
            captured.update(kwargs)
            object.__setattr__(
                self,
                "phi_solver_space",
                kwargs.get("phi_solver_space", "full-grid"),
            )
            object.__setattr__(
                self,
                "axis_core_reduced_space",
                kwargs.get("axis_core_reduced_space"),
            )

        def build_axis_core_reduced_phi_space(self):
            captured["build_calls"] = captured.get("build_calls", 0) + 1
            return "reduced-space"

    monkeypatch.setattr(hsx, "LocalFciDrbEBRhs", FakeRhs)
    monkeypatch.setattr(hsx, "HaloExchange3D", lambda: object())
    monkeypatch.setattr(hsx, "TopologyHaloFiller3D", lambda **_kwargs: object())
    monkeypatch.setattr(
        hsx,
        "make_default_topology_halo_filler_3d",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hsx, "PhysicalGhostCellFiller3D", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        hsx,
        "MetricAwarePhysicalGhostCellFiller3D",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hsx,
        "build_local_perp_laplacian_face_projectors",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )

    geometry = SimpleNamespace(
        layout=SimpleNamespace(halo_width=1),
        owned_shape=(2, 2, 2),
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            y=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            z=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
        ),
    )
    domain = SimpleNamespace(
        mesh_axis_names=("x", "y", "z"),
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        layout=SimpleNamespace(owned_shape=(2, 2, 2)),
    )

    model = hsx.build_local_eb_model(
        geometry,
        domain,
        hsx.FciDrbEBRhsParameters(),
        gmres_target_tolerance=1.0e-8,
        gmres_acceptance_tolerance=1.0e-6,
        gmres_max_iterations=4,
        gmres_preconditioner="line-u",
        phi_solver_space="axis-core-reduced",
        curvature_scheme="disabled",
    )

    assert model.phi_solver_space == "axis-core-reduced"
    assert model.axis_core_reduced_space == "reduced-space"
    assert captured["build_calls"] == 1


def test_build_local_eb_model_reduced_space_validates_topology_and_preconditioner():
    geometry = SimpleNamespace(
        layout=SimpleNamespace(halo_width=1),
        owned_shape=(2, 2, 2),
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            y=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            z=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
        ),
    )

    square_domain = SimpleNamespace(
        mesh_axis_names=("x", "y", "z"),
        periodic_axes=(False, False, False),
        axis_regular_axes=(False, False, False),
        layout=SimpleNamespace(owned_shape=(2, 2, 2)),
    )
    with pytest.raises(ValueError, match="toroidal/axis-regular"):
        hsx.build_local_eb_model(
            geometry,
            square_domain,
            hsx.FciDrbEBRhsParameters(),
            gmres_target_tolerance=1.0e-8,
            gmres_acceptance_tolerance=1.0e-6,
            gmres_max_iterations=4,
            phi_solver_space="axis-core-reduced",
        )

    toroidal_domain = SimpleNamespace(
        mesh_axis_names=("x", "y", "z"),
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        layout=SimpleNamespace(owned_shape=(2, 2, 2)),
    )
    with pytest.raises(ValueError, match="none.*line-u"):
        hsx.build_local_eb_model(
            geometry,
            toroidal_domain,
            hsx.FciDrbEBRhsParameters(),
            gmres_target_tolerance=1.0e-8,
            gmres_acceptance_tolerance=1.0e-6,
            gmres_max_iterations=4,
            gmres_preconditioner="axis-core-line-u",
            phi_solver_space="axis-core-reduced",
        )


def test_axis_core_gradient_cli_overrides_are_parsed_for_both_topologies():
    args = hsx._build_parser().parse_args(
        (
            "--topology", "square",
            "--axis-core-gradient-degree", "5",
            "--axis-core-gradient-observation-rings", "8",
            "--axis-core-gradient-target-rings", "4",
        )
    )
    assert args.axis_core_gradient_degree == 5
    assert args.axis_core_gradient_observation_rings == 8
    assert args.axis_core_gradient_target_rings == 4

    args = hsx._build_parser().parse_args(
        (
            "--topology", "toroidal",
            "--axis-core-gradient-degree", "0",
            "--axis-core-gradient-observation-rings", "2",
            "--axis-core-gradient-target-rings", "1",
        )
    )
    assert args.axis_core_gradient_degree == 0
    assert args.axis_core_gradient_observation_rings == 2
    assert args.axis_core_gradient_target_rings == 1


def test_build_local_eb_model_forwards_axis_core_gradient_controls(monkeypatch):
    captured = {}

    class FakeRhs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(hsx, "LocalFciDrbEBRhs", FakeRhs)
    monkeypatch.setattr(hsx, "HaloExchange3D", lambda: object())
    monkeypatch.setattr(hsx, "TopologyHaloFiller3D", lambda **_kwargs: object())
    monkeypatch.setattr(
        hsx,
        "make_default_topology_halo_filler_3d",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hsx, "PhysicalGhostCellFiller3D", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        hsx,
        "MetricAwarePhysicalGhostCellFiller3D",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hsx,
        "build_local_perp_laplacian_face_projectors",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )

    geometry = SimpleNamespace(
        layout=SimpleNamespace(halo_width=1),
        owned_shape=(2, 2, 2),
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            y=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            z=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
        ),
    )
    domain = SimpleNamespace(
        mesh_axis_names=("x", "y", "z"),
        periodic_axes=(False, False, False),
        axis_regular_axes=(False, False, False),
        layout=SimpleNamespace(owned_shape=(2, 2, 2)),
    )

    hsx.build_local_eb_model(
        geometry,
        domain,
        hsx.FciDrbEBRhsParameters(),
        gmres_target_tolerance=1.0e-8,
        gmres_acceptance_tolerance=1.0e-6,
        gmres_max_iterations=4,
        curvature_scheme="disabled",
        poisson_bracket_scheme="compatible-flux",
        axis_core_gradient_polynomial_degree=5,
        axis_core_gradient_observation_ring_count=8,
        axis_core_gradient_target_ring_count=4,
    )

    assert captured["axis_core_gradient_polynomial_degree"] == 5
    assert captured["axis_core_gradient_observation_ring_count"] == 8
    assert captured["axis_core_gradient_target_ring_count"] == 4
    assert captured["poisson_bracket_scheme"] == "compatible-flux"


def test_build_local_eb_model_rejects_invalid_poisson_bracket_scheme(monkeypatch):
    monkeypatch.setattr(hsx, "LocalFciDrbEBRhs", lambda **kwargs: kwargs)
    geometry = SimpleNamespace(
        layout=SimpleNamespace(halo_width=1),
        owned_shape=(2, 2, 2),
        grid=SimpleNamespace(
            x=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            y=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
            z=SimpleNamespace(centers_halo=np.arange(4, dtype=float)),
        ),
    )
    domain = SimpleNamespace(
        mesh_axis_names=("x", "y", "z"),
        periodic_axes=(False, False, False),
        axis_regular_axes=(False, False, False),
        layout=SimpleNamespace(owned_shape=(2, 2, 2)),
    )

    with pytest.raises(ValueError, match="poisson_bracket_scheme"):
        hsx.build_local_eb_model(
            geometry,
            domain,
            hsx.FciDrbEBRhsParameters(),
            gmres_target_tolerance=1.0e-8,
            gmres_acceptance_tolerance=1.0e-6,
            gmres_max_iterations=4,
            curvature_scheme="disabled",
            poisson_bracket_scheme="not-a-scheme",
        )


def test_axis_core_gradient_wiring_and_metadata_are_present():
    source = (ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    for name in (
        "axis_core_gradient_polynomial_degree",
        "axis_core_gradient_observation_ring_count",
        "axis_core_gradient_target_ring_count",
    ):
        assert name in source
    assert '"axis_core_gradient_degree"' in source
    assert '"axis_core_gradient_observation_rings"' in source
    assert '"axis_core_gradient_target_rings"' in source


def test_toroidal_cli_options_are_forwarded_to_global_geometry(monkeypatch, tmp_path):
    calls = []
    lowering_calls = []

    def fake_global_geometry(**kwargs):
        calls.append(kwargs)
        return object(), np.zeros((1, 1, 1, 3)), 2, None

    monkeypatch.setattr(hsx, "build_hsx_fci_geometry", fake_global_geometry)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda _counts: object())
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *a, **k: lowering_calls.append((a, k))
        or SimpleNamespace(
            global_shape=(4, 8, 12),
            domain=SimpleNamespace(
                layout=SimpleNamespace(
                    owned_shape=(4, 8, 12),
                    cell_halo_shape=(6, 10, 14),
                ),
                periodic_axes=(False, True, True),
                axis_regular_axes=(True, False, False),
            ),
        ),
    )
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")

    hsx.main(
        [
            "--topology", "toroidal",
            "--geometry-only",
            "--makegrid", str(makegrid),
            "--vessel", str(vessel),
            "--resolution", "4", "8", "12",
            "--metric-mesh-shape", "5", "8", "6",
            "--metric-radial-degree", "4",
            "--metric-poloidal-modes", "3",
            "--metric-toroidal-modes", "2",
            "--eta-projection-iterations", "1",
        ]
    )

    assert calls
    assert lowering_calls
    assert lowering_calls[0][1]["periodic_axes"] == (False, True, True)
    assert lowering_calls[0][1]["axis_regular_axes"] == (True, False, False)
    for key, value in TOPOLOGY_KWARGS.items():
        assert calls[0][key] == value


def test_synthetic_toroidal_global_geometry_has_expected_axes_and_shapes(monkeypatch, tmp_path):
    spy = []
    evaluator = _install_synthetic_metric(monkeypatch, spy)
    geometry, positions, nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )

    assert nfp == 2
    assert geometry.grid.x.faces[0] == 0.0
    assert geometry.grid.x.faces[-1] == 1.0
    assert geometry.grid.y.faces[0] == 0.0
    assert np.isclose(float(geometry.grid.y.faces[-1]), 2.0 * np.pi)
    assert np.isclose(float(geometry.grid.z.faces[-1] - geometry.grid.z.faces[0]), 2.0 * np.pi)
    assert geometry.shape == (4, 8, 12)
    assert positions.shape == (4, 8, 12, 3)
    assert spy[0]["topology"] == "toroidal"
    assert evaluator.ordinary_metric_calls
    assert all(np.all(call[..., 0] > 0.0) for call in evaluator.ordinary_metric_calls)


def test_toroidal_geometry_lowering_preserves_axis_regular_parity(monkeypatch, tmp_path):
    evaluator = _install_synthetic_metric(monkeypatch, [])
    geometry, _positions, _nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )
    sharded = hsx.build_local_fci_geometries(
        geometry,
        (1, 1, 1),
        halo_width=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    assert sharded.domain.periodic_axes == (False, True, True)
    assert sharded.domain.axis_regular_axes == (True, False, False)
    assert not sharded.domain.has_physical_lower(0)
    assert sharded.domain.has_physical_upper(0)

    mesh = hsx.make_shard_mesh((1, 1, 1))
    geometry_spec = P("x", "y", "z", None)
    fields = jax.device_put(
        sharded.cell_fields,
        NamedSharding(mesh, geometry_spec),
    )

    def inspect_axis(fields_owned):
        local = hsx.assemble_local_fci_geometry(sharded, fields_owned)
        curvature_faces = hsx.build_local_curvature_face_coefficients(
            local, sharded.domain
        )
        curvature_divergence = (
            (curvature_faces.x[1:] - curvature_faces.x[:-1])
            / local.spacing.dx_owned
            + (curvature_faces.y[:, 1:] - curvature_faces.y[:, :-1])
            / local.spacing.dy_owned
            + (curvature_faces.z[:, :, 1:] - curvature_faces.z[:, :, :-1])
            / local.spacing.dz_owned
        )
        exact_face_errors = []
        for local_metric, global_metric in zip(
            local.face_metric.axes, geometry.face_metric.axes
        ):
            for name in (
                "J", "g11", "g22", "g33", "g12", "g13", "g23",
                "g_11", "g_22", "g_33", "g_12", "g_13", "g_23",
            ):
                exact_face_errors.append(
                    jnp.max(
                        jnp.abs(
                            getattr(local_metric, f"{name}_owned")
                            - getattr(global_metric, name)
                        )
                    )
                )
        for local_bfield, global_bfield in zip(
            local.face_bfield.axes, geometry.face_bfield.axes
        ):
            exact_face_errors.extend(
                (
                    jnp.max(
                        jnp.abs(
                            local_bfield.Bmag_owned - global_bfield.Bmag
                        )
                    ),
                    jnp.max(
                        jnp.abs(
                            local_bfield.B_contra_owned
                            - global_bfield.B_contra
                        )
                    ),
                )
            )
        return jnp.stack(
            (
                jnp.max(jnp.abs(local.face_metric.x.J_owned[0])),
                jnp.max(jnp.abs(local.face_bfield.x.B_contra_owned[0, ..., 0])),
                jnp.max(jnp.stack(exact_face_errors)),
                jnp.max(jnp.abs(curvature_faces.x[0])),
                jnp.max(jnp.abs(curvature_divergence)),
            ),
        )

    diagnostics = jax.jit(
        jax.shard_map(
            inspect_axis,
            mesh=mesh,
            in_specs=geometry_spec,
            out_specs=P(),
            check_vma=False,
        )
    )(fields)
    assert np.all(np.isfinite(np.asarray(diagnostics)))
    assert float(diagnostics[0]) < 1.0e-12
    # B^u itself need not vanish pointwise at a polar axis; the signed-J
    # vector density (and therefore radial magnetic flux) collapses there.
    assert float(diagnostics[0] * diagnostics[1]) < 1.0e-12
    assert float(diagnostics[2]) < 1.0e-12
    assert float(diagnostics[3]) < 1.0e-12
    assert float(diagnostics[4]) < 1.0e-11
    assert evaluator.ordinary_metric_calls


def test_distributed_toroidal_curvature_complex_is_axis_regular(monkeypatch, tmp_path):
    if jax.local_device_count() < 4:
        pytest.skip("requires four local devices for radial/theta sharding")

    _install_synthetic_metric(monkeypatch, [])
    geometry, _positions, _nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )
    shard_counts = (2, 2, 1)
    sharded = hsx.build_local_fci_geometries(
        geometry,
        shard_counts,
        halo_width=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    mesh = hsx.make_shard_mesh(shard_counts)
    geometry_spec = P("x", "y", "z", None)
    fields = jax.device_put(
        sharded.cell_fields,
        NamedSharding(mesh, geometry_spec),
    )

    def inspect(fields_owned):
        local = hsx.assemble_local_fci_geometry(sharded, fields_owned)
        coefficients = hsx.build_local_curvature_face_coefficients(
            local, sharded.domain
        )
        divergence = (
            (coefficients.x[1:] - coefficients.x[:-1])
            / local.spacing.dx_owned
            + (coefficients.y[:, 1:] - coefficients.y[:, :-1])
            / local.spacing.dy_owned
            + (coefficients.z[:, :, 1:] - coefficients.z[:, :, :-1])
            / local.spacing.dz_owned
        )
        is_axis_owner = sharded.domain.runtime_has_axis_regular_lower(0)
        diagnostics = jnp.stack(
            (
                jnp.where(
                    is_axis_owner,
                    jnp.max(jnp.abs(coefficients.x[0])),
                    0.0,
                ),
                jnp.max(jnp.abs(divergence)),
            )
        )
        for axis_name, count in zip(("x", "y", "z"), shard_counts):
            if count > 1:
                diagnostics = jax.lax.pmax(diagnostics, axis_name=axis_name)
        return diagnostics

    diagnostics = jax.jit(
        jax.shard_map(
            inspect,
            mesh=mesh,
            in_specs=geometry_spec,
            out_specs=P(),
            check_vma=False,
        )
    )(fields)
    assert float(diagnostics[0]) < 1.0e-12
    assert float(diagnostics[1]) < 1.0e-11


def test_toroidal_field_aligned_initialization_is_finite(monkeypatch, tmp_path):
    _install_synthetic_metric(monkeypatch, [])
    geometry, _positions, _nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )
    state = hsx.build_initial_state(
        geometry,
        initialization="field-aligned",
        density_amplitude=0.1,
        temperature_amplitude=0.0,
        blob_center=(0.5, 0.0),
        blob_width=0.2,
        blob_reference_eta=0.0,
        blob_parallel_half_length=np.pi,
        fieldline_substeps_per_plane=1,
        filament_cache_dir=None,
        rebuild_filament_cache=False,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    for _name, values in state.field_items():
        assert np.all(np.isfinite(np.asarray(values)))
    assert float(jnp.max(state.density)) > 1.0


@pytest.mark.slow
def test_synthetic_toroidal_full_eb_rk4_step_is_finite(monkeypatch, tmp_path):
    _install_synthetic_metric(monkeypatch, [])
    geometry, positions, nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )
    sharded = hsx.build_local_fci_geometries(
        geometry,
        (1, 1, 1),
        halo_width=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    ones = jnp.ones(geometry.shape, dtype=jnp.float64)
    zeros = jnp.zeros(geometry.shape, dtype=jnp.float64)
    initial = hsx.FciDrbEBState(
        density=ones,
        phi=zeros,
        Te=ones,
        Ti=ones,
        Vi=zeros,
        Ve=zeros,
        vorticity=zeros,
    )
    output = tmp_path / "toroidal_one_step.npz"
    final = hsx.run_full_eb(
        initial,
        global_geometry=geometry,
        cell_positions=positions,
        nfp=nfp,
        sharded_geometry=sharded,
        mesh=hsx.make_shard_mesh((1, 1, 1)),
        parameters=hsx.FciDrbEBRhsParameters(
            phi_inversion_iterations=6,
            phi_inversion_regularization=1.0e-9,
        ),
        metric_cache_path=None,
        gmres_target_tolerance=1.0e-8,
        gmres_acceptance_tolerance=1.0e-6,
        gmres_max_iterations=6,
        gmres_preconditioner="none",
        time_integrator="rk4",
        newton_config=hsx.SolvaxNewtonConfig(max_steps=2),
        newton_preconditioner="none",
        num_steps=1,
        timestep=1.0e-6,
        start_time=0.0,
        output_path=output,
        save_every=1,
        phase_timing=False,
        diagnostic_every=0,
        reconstruct_initial_phi=False,
        curvature_scheme="disabled",
        neumann_ghost_scheme="physical",
        run_metadata=hsx._topology_metadata(hsx.topology_descriptor("toroidal")),
    )
    for _name, values in final.field_items():
        assert np.all(np.isfinite(np.asarray(values)))
    assert output.is_file()


def test_odd_global_ntheta_is_rejected(monkeypatch, tmp_path):
    spy = []
    _install_synthetic_metric(monkeypatch, spy)
    with pytest.raises((ValueError, SystemExit), match="(?i)theta|even"):
        hsx.build_hsx_fci_geometry(
            **_geometry_kwargs(tmp_path, resolution=(4, 7, 12))
        )
    assert not spy


def test_full_torus_neta_divisibility_is_retained(monkeypatch, tmp_path):
    spy = []
    evaluator = SyntheticMetricEvaluator(nfp=4)
    _install_synthetic_metric(monkeypatch, spy, evaluator)
    with pytest.raises(ValueError, match="divisible"):
        hsx.build_hsx_fci_geometry(
            **_geometry_kwargs(tmp_path, resolution=(4, 8, 10))
        )


def test_axis_face_is_finite_collapsed_and_uses_half_turn_parity(monkeypatch, tmp_path):
    spy = []
    evaluator = _install_synthetic_metric(monkeypatch, spy)
    geometry, _positions, _nfp, _cache = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(tmp_path)
    )
    axis_metric = geometry.face_metric.x
    axis_bfield = geometry.face_bfield.x

    assert np.all(np.asarray(axis_metric.J)[0] == 0.0)
    for name in ("J", "g11", "g22", "g33", "g12", "g13", "g23",
                 "g_11", "g_22", "g_33", "g_12", "g_13", "g_23"):
        assert np.all(np.isfinite(np.asarray(getattr(axis_metric, name))))
    assert np.all(np.isfinite(np.asarray(axis_bfield.B_contra)))
    assert np.all(np.isfinite(np.asarray(axis_bfield.Bmag)))

    ntheta = np.asarray(axis_metric.g22).shape[1]
    assert ntheta % 2 == 0
    half = ntheta // 2
    # The collapsed face is represented by approach-angle data.  Validate the
    # complete half-turn transformation instead of requiring odd-u channels
    # to vanish pointwise.
    tensor_signs = {
        "g11": 1.0, "g22": 1.0, "g33": 1.0,
        "g12": -1.0, "g13": -1.0, "g23": 1.0,
        "g_11": 1.0, "g_22": 1.0, "g_33": 1.0,
        "g_12": -1.0, "g_13": -1.0, "g_23": 1.0,
    }
    for name, sign in tensor_signs.items():
        values = np.asarray(getattr(axis_metric, name))[0]
        np.testing.assert_allclose(
            np.roll(values, -half, axis=0), sign * values, atol=1.0e-12
        )
    b_values = np.asarray(axis_bfield.B_contra)[0]
    for component, sign in enumerate((-1.0, 1.0, 1.0)):
        np.testing.assert_allclose(
            np.roll(b_values[..., component], -half, axis=0),
            sign * b_values[..., component],
            atol=1.0e-12,
        )
    bmag = np.asarray(axis_bfield.Bmag)[0]
    np.testing.assert_allclose(np.roll(bmag, -half, axis=0), bmag, atol=1.0e-12)
    assert evaluator.ordinary_magnetic_calls
    assert all(np.all(call[..., 0] > 0.0) for call in evaluator.ordinary_magnetic_calls)


def test_toroidal_cache_round_trip_preserves_topology_and_modes(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    source_mgrid = tmp_path / "mgrid.nc"
    source_vessel = tmp_path / "vessel.txt"
    source_mgrid.write_bytes(b"mgrid")
    source_vessel.write_bytes(b"vessel")
    spy = []
    evaluator = _install_synthetic_metric(monkeypatch, spy)

    hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(
            tmp_path,
            makegrid_path=source_mgrid,
            vessel_path=source_vessel,
            metric_cache_dir=cache_dir,
        )
    )
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    with np.load(cache_files[0], allow_pickle=False) as cached:
        spec = json.loads(str(cached["cache_spec"].item()))
        assert spec["topology"] == "toroidal"
        assert spec["metric_mesh_shape"] == [5, 8, 6]
        assert spec["metric_radial_degree"] == 4
        assert spec["metric_poloidal_modes"] == 3
        assert spec["metric_toroidal_modes"] == 2
        assert spec["eta_projection_iterations"] == 1

    class CachedMetricEvaluator(SyntheticMetricEvaluator):
        @classmethod
        def from_cache_payload(cls, payload, *, prefix=""):
            assert str(payload[f"{prefix}topology"].item()) == "toroidal"
            result = cls(nfp=int(payload[f"{prefix}nfp"].item()))
            result.radial_degree = int(payload[f"{prefix}radial_degree"].item())
            poloidal = int(payload[f"{prefix}poloidal_modes"].item())
            toroidal = int(payload[f"{prefix}toroidal_modes"].item())
            result.poloidal_modes = tuple(range(-poloidal, poloidal + 1))
            result.toroidal_modes = tuple(range(-toroidal, toroidal + 1))
            return result

    monkeypatch.setattr(hsx, "MetricEvaluator", CachedMetricEvaluator)
    monkeypatch.setattr(hsx, "build_hsx_metric_evaluator", lambda **_: pytest.fail("cache miss"))
    geometry, _positions, _nfp, cache_path = hsx.build_hsx_fci_geometry(
        **_geometry_kwargs(
            tmp_path,
            makegrid_path=source_mgrid,
            vessel_path=source_vessel,
            metric_cache_dir=cache_dir,
        )
    )
    assert cache_path == cache_files[0]
    assert geometry.shape == (4, 8, 12)


def test_corrupted_toroidal_cache_metadata_is_rejected_or_rebuilt(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    source_mgrid = tmp_path / "mgrid.nc"
    source_vessel = tmp_path / "vessel.txt"
    source_mgrid.write_bytes(b"mgrid")
    source_vessel.write_bytes(b"vessel")
    spy = []
    evaluator = _install_synthetic_metric(monkeypatch, spy)
    build_kwargs = _geometry_kwargs(
        tmp_path,
        makegrid_path=source_mgrid,
        vessel_path=source_vessel,
        metric_cache_dir=cache_dir,
    )
    hsx.build_hsx_fci_geometry(**build_kwargs)
    cache_file = next(cache_dir.glob("*.npz"))

    with np.load(cache_file, allow_pickle=False) as cached:
        payload = {name: np.array(cached[name], copy=True) for name in cached.files}
    corrupted_spec = json.loads(str(payload["cache_spec"].item()))
    corrupted_spec["topology"] = "square"
    corrupted_spec["coordinate_names"] = ["u", "v", "eta"]
    payload["cache_spec"] = np.asarray(
        json.dumps(corrupted_spec, sort_keys=True, separators=(",", ":"))
    )
    np.savez(cache_file, **payload)

    rebuild_calls = []

    def rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        return evaluator, object(), SimpleNamespace(nfp=evaluator.nfp), evaluator.nfp

    monkeypatch.setattr(hsx, "build_hsx_metric_evaluator", rebuild)
    try:
        hsx.build_hsx_fci_geometry(**build_kwargs)
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        # Rejecting the inconsistent cache is an acceptable fail-closed path.
        assert "cache" in str(error).lower() or "topolog" in str(error).lower() or "coordinate" in str(error).lower()
    else:
        # The other acceptable path is to ignore the cache and rebuild using
        # the requested toroidal specification.
        assert rebuild_calls, "corrupted cache was accepted without rebuilding"


def test_toroidal_geometry_only_lowers_axis_regular_geometry(monkeypatch, tmp_path):
    calls = []
    lowering_calls = []
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")

    def fake_global(**kwargs):
        calls.append(kwargs)
        return object(), np.zeros((4, 8, 12, 3)), 2, None

    monkeypatch.setattr(hsx, "build_hsx_fci_geometry", fake_global)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *a, **k: lowering_calls.append((a, k))
        or SimpleNamespace(
            global_shape=(4, 8, 12),
            domain=SimpleNamespace(
                layout=SimpleNamespace(
                    owned_shape=(4, 8, 12),
                    cell_halo_shape=(6, 10, 14),
                ),
                periodic_axes=(False, True, True),
                axis_regular_axes=(True, False, False),
            ),
        ),
    )

    hsx.main([
        "--topology", "toroidal", "--geometry-only",
        "--makegrid", str(makegrid), "--vessel", str(vessel),
        "--resolution", "4", "8", "12",
        "--metric-mesh-shape", "5", "8", "6",
    ])
    assert calls and calls[0]["topology"] == "toroidal"
    assert lowering_calls[0][1]["axis_regular_axes"] == (True, False, False)


def test_toroidal_without_geometry_only_reaches_simulation(monkeypatch, tmp_path):
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")
    geometry = object()
    monkeypatch.setattr(
        hsx,
        "build_hsx_fci_geometry",
        lambda **_: (geometry, np.zeros((4, 8, 12, 3)), 2, None),
    )
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    sharded = SimpleNamespace(
        global_shape=(4, 8, 12),
        shard_counts=(1, 1, 1),
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
    run_calls = []
    monkeypatch.setattr(hsx, "run_full_eb", lambda *a, **k: run_calls.append((a, k)))

    hsx.main([
        "--topology", "toroidal",
        "--makegrid", str(makegrid), "--vessel", str(vessel),
        "--resolution", "4", "8", "12",
        "--metric-mesh-shape", "5", "8", "6",
        "--poisson-bracket-scheme", "compatible-flux",
        "--phi-solver-space", "axis-core-reduced",
        "--num-steps", "1", "--final-time", "1e-6",
    ])
    assert run_calls
    assert run_calls[0][1]["poisson_bracket_scheme"] == "compatible-flux"
    assert run_calls[0][1]["phi_solver_space"] == "axis-core-reduced"
    assert (
        run_calls[0][1]["run_metadata"]["poisson_bracket_scheme"]
        == "compatible-flux"
    )
    assert (
        run_calls[0][1]["run_metadata"]["phi_solver_space"]
        == "axis-core-reduced"
    )


def test_square_geometry_only_keeps_default_topology_semantics(monkeypatch, tmp_path):
    calls = []
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")

    def fake_global(**kwargs):
        calls.append(kwargs)
        return object(), np.zeros((2, 2, 2, 3)), 2, None

    monkeypatch.setattr(hsx, "build_hsx_fci_geometry", fake_global)
    monkeypatch.setattr(hsx, "make_shard_mesh", lambda *_: object())
    monkeypatch.setattr(
        hsx,
        "build_local_fci_geometries",
        lambda *a, **k: SimpleNamespace(
            global_shape=(4, 4, 8),
            domain=SimpleNamespace(
                layout=SimpleNamespace(
                    owned_shape=(4, 4, 8), cell_halo_shape=(8, 8, 12)
                ),
                periodic_axes=(False, False, True),
                axis_regular_axes=(False, False, False),
            ),
        ),
    )
    hsx.main([
        "--geometry-only", "--makegrid", str(makegrid), "--vessel", str(vessel),
        "--resolution", "4", "4", "8",
    ])

    assert calls
    assert calls[0].get("topology", "square") == "square"


def test_square_main_validates_shard_mesh_before_global_geometry(monkeypatch, tmp_path, capsys):
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"synthetic")
    vessel.write_bytes(b"synthetic")

    def fail_shard_mesh(*_args):
        raise ValueError("sentinel shard-mesh validation failure")

    monkeypatch.setattr(hsx, "make_shard_mesh", fail_shard_mesh)
    monkeypatch.setattr(
        hsx,
        "build_hsx_fci_geometry",
        lambda **_kwargs: pytest.fail("global geometry was reached before shard validation"),
    )

    with pytest.raises(SystemExit) as error:
        hsx.main([
            "--makegrid", str(makegrid), "--vessel", str(vessel),
            "--resolution", "4", "4", "8",
        ])
    assert error.value.code == 2
    assert "sentinel shard-mesh validation failure" in capsys.readouterr().err
