"""Focused contracts for the canonical HSX/RLP Stage-7 MMS harness."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "simulate_hsx_mms.py"
REFERENCE = ROOT / "hsx_mms_continuum_reference.py"
FROZEN_DIAGNOSTIC_CASE = ROOT / "tests" / "frozen_eb_diagnostic_case.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _SyntheticMetric:
    """Tiny smooth metric with a nonzero ``div(b)`` for algebraic checks."""

    period = 2.0 * np.pi

    def evaluate(self, points, *, reject_nonpositive_J=False):
        q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        eye = np.broadcast_to(np.eye(3), (q.shape[0], 3, 3)).copy()
        return SimpleNamespace(
            signed_J=np.ones(q.shape[0]),
            covariant_metric=eye,
            contravariant_metric=eye,
        )

    def evaluate_magnetic_field(
        self, points, bfield, *, reject_nonpositive_J=False
    ):
        q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        # The field remains tangent to eta, while its magnitude varies along
        # eta.  Thus the reference's B*b.grad(1/B) is nonzero and exercises
        # the production geometric source terms.
        magnitude = 1.0 + 0.15 * np.sin(q[:, 2])
        return SimpleNamespace(
            B_contravariant=np.stack(
                (np.zeros_like(magnitude), np.zeros_like(magnitude), magnitude),
                axis=-1,
            ),
            magnitude=magnitude,
        )


def _synthetic_parallel_data(reference, points):
    """Return deterministic positive state values and logical gradients."""

    q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    count = q.shape[0]
    values = {
        "density": 1.15 + 0.03 * np.cos(q[:, 1]),
        "Te": 0.92 + 0.04 * np.sin(q[:, 0] + q[:, 2]),
        "Ti": 1.08 + 0.02 * np.cos(q[:, 0] - q[:, 2]),
        "Vi": 0.17 + 0.01 * np.sin(q[:, 1]),
        "Ve": -0.11 + 0.015 * np.cos(q[:, 0]),
        "phi": 0.06 * np.sin(q[:, 1] - 0.3 * q[:, 2]),
        "vorticity": 0.02 * np.cos(q[:, 0]),
    }
    gradients = {
        name: np.column_stack(
            (
                0.11 + 0.01 * (index + 1) * q[:, 0],
                -0.07 + 0.013 * (index + 2) * q[:, 1],
                0.05 - 0.009 * (index + 3) * q[:, 2],
            )
        )
        for index, name in enumerate(values)
    }
    derivatives = {name: np.zeros(count) for name in values}
    hessians = {
        name: np.zeros((count, 3, 3), dtype=np.float64)
        for name in values
    }
    return reference.PointData(values, derivatives, gradients, hessians)


def test_independent_reference_identity_self_test_is_finite():
    reference = _load(REFERENCE, "hsx_mms_continuum_reference_test")
    result = reference.analytic_identity_metric_self_test()
    assert np.isfinite(result["max_source"])
    assert result["max_vorticity"] == 0.0
    assert result["max_term_closure"] <= 1.0e-13


def test_continuum_parallel_material_terms_match_production_matrix_and_forces():
    """Check the independent material algebra against the live five-field symbol.

    This deliberately uses a synthetic metric and a hand-built point data
    object.  It therefore checks the continuum-reference formulas without
    constructing HSX geometry, FCI maps, or any production RHS.  The first
    five terms are ``-A parallel_grad(U)`` plus the three ``div(b)`` source
    entries; the electron collision/electrostatic lanes remain explicit
    addenda to the matrix material term.
    """

    reference = _load(
        REFERENCE, "hsx_mms_continuum_parallel_matrix_test"
    )
    from drbx.native.fci_parallel_production_flux import (
        parallel_production_principal_matrix,
    )

    points = np.asarray(
        (
            (0.23, 0.4, 0.6),
            (0.41, 1.7, 2.2),
            (0.67, 2.8, 4.9),
            (0.79, 5.1, 5.7),
        ),
        dtype=np.float64,
    )
    tau = 0.73
    mu = 17.0
    ve_nu = 0.041
    rho_star = 1.0
    ref = reference.ContinuumMmsReference(
        _SyntheticMetric(), object(), 1.0,
        tau=tau,
        mi_over_me=mu,
        rho_star=rho_star,
        Ve_nu=ve_nu,
    )
    data = _synthetic_parallel_data(reference, points)
    terms = ref._continuum_terms_from_data(points, data)
    metric = ref._metric(points)
    parallel = lambda value: np.einsum("ni,ni->n", metric["b"], value)

    state_names = ("density", "Te", "Ti", "Vi", "Ve")
    state = np.stack([data.values[name] for name in state_names], axis=-1)
    parallel_gradient = np.stack(
        [parallel(data.gradients[name]) for name in state_names], axis=-1
    )
    matrix = np.asarray(
        parallel_production_principal_matrix(
            state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4],
            tau, mu,
        )
    )
    matrix_material = -np.einsum("nij,nj->ni", matrix, parallel_gradient)

    n, Te, Ti, Vi, Ve = [state[:, index] for index in range(5)]
    current = n * (Vi - Ve)
    div_b = np.asarray(ref._div_b(points))
    geometric = np.zeros_like(matrix_material)
    geometric[:, 0] = -n * Ve * div_b
    geometric[:, 1] = (
        2.0 * Te / (3.0 * n) * (0.71 * current - n * Ve) * div_b
    )
    geometric[:, 2] = (
        2.0 * Ti / (3.0 * n) * (current - n * Vi) * div_b
    )

    explicit_ve = mu * ve_nu * current + mu * (
        parallel(data.gradients["phi"])
        + tau * parallel(data.gradients["Ti"])
    )
    observed = np.stack(
        (
            terms["density"]["parallel_density_flux_divergence"],
            terms["Te"]["parallel_advection"],
            terms["Ti"]["parallel_advection"],
            terms["Vi"]["parallel_self_advection"],
            terms["Ve"]["parallel_self_advection"]
            + terms["Ve"]["collision"]
            + terms["Ve"]["electrostatic"],
        ),
        axis=-1,
    )
    expected = matrix_material + geometric
    expected[:, 4] += explicit_ve
    np.testing.assert_allclose(observed, expected, rtol=2.0e-12, atol=2.0e-12)


def test_continuum_reference_term_sums_close_on_nontrivial_synthetic_metric():
    """Every independent continuum RHS lane must sum to its returned RHS."""

    reference = _load(
        REFERENCE, "hsx_mms_continuum_term_closure_test"
    )
    points = np.asarray(
        (
            (0.29, 0.8, 0.2),
            (0.52, 2.0, 2.9),
            (0.71, 3.6, 5.4),
        ),
        dtype=np.float64,
    )
    ref = reference.ContinuumMmsReference(
        _SyntheticMetric(), object(), 1.0, tau=0.63, mi_over_me=11.0,
        Ve_nu=0.023, perp_diffusion=1.0e-5,
    )
    rhs = ref.continuum_rhs(points, 0.17)
    terms = ref.continuum_terms(points, 0.17)
    for field in reference.EVOLVED_FIELDS:
        summed = np.sum(np.stack(tuple(terms[field].values())), axis=0)
        np.testing.assert_allclose(summed, rhs[field], rtol=2.0e-13, atol=2.0e-13)
        assert np.all(np.isfinite(summed))


def test_identity_perpendicular_diffusion_uses_cached_tensor_and_analytic_hessian():
    """For identity geometry, the active perpendicular operator is ``f_uu+f_tt``."""

    reference = _load(
        REFERENCE, "simulate_hsx_mms_perpendicular_diffusion_test"
    )
    points = np.asarray(
        (
            (0.24, 0.35, 0.7),
            (0.46, 1.9, 2.8),
            (0.73, 4.4, 5.3),
        ),
        dtype=np.float64,
    )
    coefficient = 1.0e-5
    ref = reference.ContinuumMmsReference(
        _SyntheticMetric(), object(), 1.0, perp_diffusion=coefficient
    )
    data = ref.evaluate(points, 0.19)
    prepared = ref.prepare(points)
    terms = ref.continuum_terms(points, 0.19, prepared=prepared)
    for field in reference.EVOLVED_FIELDS:
        expected_operator = (
            data.hessians[field][:, 0, 0] + data.hessians[field][:, 1, 1]
        )
        cached_operator = ref._perpendicular_operator(
            points, data.gradients[field], data.hessians[field], prepared=prepared
        )
        uncached_operator = ref._perpendicular_operator(
            points, data.gradients[field], data.hessians[field]
        )
        np.testing.assert_allclose(
            cached_operator, expected_operator, rtol=2.0e-11, atol=2.0e-11
        )
        np.testing.assert_allclose(
            uncached_operator, expected_operator, rtol=2.0e-11, atol=2.0e-11
        )
        np.testing.assert_allclose(
            terms[field]["perpendicular_diffusion"],
            coefficient * expected_operator,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
    np.testing.assert_allclose(
        prepared.perpendicular_flux_tensor,
        np.broadcast_to(np.diag((1.0, 1.0, 0.0)), (points.shape[0], 3, 3)),
        rtol=0.0,
        atol=0.0,
    )


def test_structured_fourth_order_derivatives_cover_nonuniform_periodic_stencils():
    """The projector derivative helper handles radial ends and wrapped axes."""

    driver = _load(DRIVER, "simulate_hsx_mms_structured_derivative_test")
    radial = np.asarray((0.04, 0.13, 0.25, 0.39, 0.51, 0.66, 0.79, 0.91, 0.97))
    theta = (np.arange(12, dtype=np.float64) + 0.43) * 2.0 * np.pi / 12.0
    theta += 0.08 * (2.0 * np.pi / 12.0) * np.sin(2.0 * np.pi * theta / (2.0 * np.pi))
    eta = (np.arange(12, dtype=np.float64) + 0.31) * 4.0 * np.pi / 12.0
    eta += 0.06 * (4.0 * np.pi / 12.0) * np.sin(2.0 * np.pi * eta / (4.0 * np.pi))
    u, t, e = np.meshgrid(radial, theta, eta, indexing="ij")
    values = u**4 + np.sin(t) + 0.5 * np.cos(e)
    gradient, hessian = driver._fourth_order_structured_derivatives(
        values, (radial, theta, eta), periods=(None, 2.0 * np.pi, 4.0 * np.pi)
    )
    np.testing.assert_allclose(gradient[..., 0], 4.0 * u**3, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(gradient[..., 1], np.cos(t), rtol=0.0, atol=3.5e-3)
    np.testing.assert_allclose(gradient[..., 2], -0.5 * np.sin(e), rtol=0.0, atol=2.5e-2)
    np.testing.assert_allclose(hessian[..., 0, 0], 12.0 * u**2, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(hessian[..., 1, 1], -np.sin(t), rtol=0.0, atol=6.0e-3)
    np.testing.assert_allclose(hessian[..., 2, 2], -0.5 * np.cos(e), rtol=0.0, atol=4.0e-2)
    np.testing.assert_allclose(hessian[..., 0, 1], 0.0, rtol=0.0, atol=2.0e-10)


def test_structured_periodic_derivative_converges_fourth_order_on_nonuniform_grid():
    driver = _load(DRIVER, "simulate_hsx_mms_structured_order_test")

    def error(count):
        period = 2.0 * np.pi
        index = np.arange(count, dtype=np.float64)
        coordinates = (index + 0.37) * period / count
        coordinates += 0.05 * period / count * np.sin(2.0 * np.pi * index / count)
        values = np.sin(coordinates)
        gradient, _ = driver._fourth_order_structured_derivatives(
            values[:, None, None] * np.ones((1, 6, 6)),
            (coordinates, np.arange(6.0), np.arange(6.0)),
            periods=(period, None, None),
        )
        return float(np.max(np.abs(gradient[..., 0] - np.cos(coordinates)[:, None, None])))

    coarse = error(12)
    fine = error(24)
    assert coarse / fine > 8.0


def test_generalized_potential_polarization_and_vorticity_lanes_are_nonzero():
    reference = _load(
        REFERENCE, "simulate_hsx_mms_generalized_potential_test"
    )
    ref = reference.ContinuumMmsReference(
        _SyntheticMetric(), object(), 1.0, tau=0.8, perp_diffusion=1.0e-5,
        enable_generalized_potential=True,
    )
    points = np.asarray(
        ((0.19, 0.31, 0.42), (0.37, 1.48, 2.31), (0.58, 2.79, 4.74),
         (0.76, 4.51, 5.82)),
        dtype=np.float64,
    )
    prepared = ref.prepare(points)
    data = ref.evaluate(points, 0.23, prepared=prepared)
    psi, psi_du, psi_dtheta, psi_deta, _, psi_hessian = ref._psi_raw(points)
    expected_phi = -ref.tau * (data.values["Ti"] - 1.0) + psi
    np.testing.assert_allclose(data.values["phi"], expected_phi, rtol=0.0, atol=1.0e-13)
    expected_omega = psi_hessian[:, 0, 0] + psi_hessian[:, 1, 1]
    np.testing.assert_allclose(prepared.mms_omega, expected_omega, rtol=0.0, atol=1.0e-12)
    assert np.max(np.abs(data.values["vorticity"])) > 1.0e-6
    terms = ref.continuum_terms(points, 0.23, prepared=prepared)
    for lane in ("poisson_bracket", "parallel_advection", "perpendicular_diffusion"):
        assert np.max(np.abs(terms["vorticity"][lane])) > 1.0e-14


def test_production_driver_enables_generalized_potential_and_lane_assertions():
    source = DRIVER.read_text(encoding="utf-8")
    assert "enable_generalized_potential=True" in source
    assert "generalized-potential MMS produced identically zero exact omega" in source
    assert "vorticity lane" in source


def test_mms_physical_parameter_contract_enables_only_perpendicular_diffusion():
    driver = _load(DRIVER, "simulate_hsx_mms_physical_parameters_test")
    assert driver.PHYSICAL_PARAMETERS["Ve_nu"] == 0.0
    assert all(
        driver.PHYSICAL_PARAMETERS[name] == 0.0
        for name in (
            "density_D_parallel",
            "electron_temperature_chi_parallel",
            "ion_temperature_chi_parallel",
            "Vi_parallel_viscosity",
            "Ve_parallel_viscosity",
            "vorticity_D_parallel",
        )
    )
    assert all(
        driver.PHYSICAL_PARAMETERS[name] == 1.0e-5
        for name in (
            "density_D_perp",
            "electron_temperature_D_perp",
            "ion_temperature_D_perp",
            "Vi_D_perp",
            "Ve_D_perp",
            "vorticity_D_perp",
        )
    )


def test_mms_uses_complete_current_production_contract():
    driver = _load(DRIVER, "simulate_hsx_mms_production_contract_test")
    config = driver._production_configuration((1, 1, 4), 4)

    assert driver.PRODUCTION_GMRES == {
        "target_tolerance": 1.0e-8,
        "acceptance_tolerance": 5.0e-5,
        "max_iterations": 500,
        "restart": 100,
        "preconditioner": "line-u",
        "residual_correction_steps": 1,
    }
    assert config["parallel_velocity_layout"] == "cell-centered"
    assert config["characteristic_sat_affine_current_lift"] == "enabled"
    assert config["parallel_current_phi_pair"] == "enabled"
    assert config["parallel_inflow_closure"] == "central"
    assert config["fci_parallel_leg_scheme"] == "centered"
    assert config["curvature_scheme"] == "conservative"
    assert config["curvature_operator"] == (
        "production-characteristic-owner-face"
    )
    assert config["curvature_rlp_face_scheme"] == "projected-fine"
    assert config["curvature_wall_flux_closure"] == (
        "bc-characteristic-operator-trace-canonical-face-state"
    )
    assert config["curvature_equations"] == [
        "density", "Te", "Ti", "vorticity"
    ]
    assert config["gmres_max_iterations"] == 500
    assert config["gmres_residual_correction_steps"] == 1
    assert config["evolved_initial_phi"] == "analytic-manufactured"


def test_evolved_mms_starts_from_analytic_manufactured_phi():
    source = DRIVER.read_text(encoding="utf-8")
    evolved = source[source.index("else:\n                advanced = blob.run_full_eb(") :]
    assert "reconstruct_initial_phi=False" in evolved
    assert "**_production_configuration(" in evolved


def test_canonical_wiring_path_uses_current_production_selectors(capsys):
    driver = _load(DRIVER, "simulate_hsx_mms_test")
    driver.main(("--self-test", "--wiring-only"))
    output = capsys.readouterr().out
    for selector in (
        "production-split",
        "fci",
        "support-core",
        "characteristic-sat",
        "energy-absorbing",
        "all-physical-walls",
        "local-backward-euler",
        "imex-ssp222",
        "material-scalar-third-order-upwind",
    ):
        assert selector in output


def test_wiring_only_does_not_require_cluster_device_count(capsys):
    driver = _load(DRIVER, "simulate_hsx_mms_wiring_device_count_test")
    # This repository's local test process normally exposes one device; the
    # explicit remote mesh is still valid for a no-geometry wiring check.
    driver.main(
        (
            "--self-test",
            "--wiring-only",
            "--shard-counts",
            "1",
            "1",
            "4",
        )
    )
    output = capsys.readouterr().out
    assert "[mms-wiring]" in output


def test_harness_requires_real_resolution_local_maps_and_canonical_imex():
    source = DRIVER.read_text(encoding="utf-8")
    target_build = source[source.index("for n in resolutions") :]
    assert "construct_fci_maps=True" in target_build
    assert "fci_trace_substeps=4" in target_build
    assert "blob.run_full_eb(" in target_build
    assert 'time_integrator="imex-ssp222"' in target_build
    assert "source_evaluator=stage_source" in target_build
    assert 'history_dtype="float64"' in target_build
    assert "source_aware_rk4_step" not in source
    assert "reference_derivative_method" in source
    assert "reference_derivative_order" in source


def test_canonical_evolved_schedule_is_20_steps():
    driver = _load(DRIVER, "simulate_hsx_mms_step_schedule_test")
    steps, timestep = driver._step_schedule(0.0, 2.0e-5, 1.0e-6)
    assert steps == 20
    assert timestep == pytest.approx(1.0e-6, rel=0.0, abs=1.0e-20)

    # A genuinely nonintegral interval retains the conservative ceiling.
    steps, timestep = driver._step_schedule(0.0, 1.0, 0.3)
    assert steps == 4
    assert timestep == 0.25
    assert driver._step_schedule(1.0e-6, 1.0e-6, 1.0e-6) == (0, 0.0)


def test_reuse_history_requires_analytic_initial_phi_and_current_contract():
    driver = _load(DRIVER, "simulate_hsx_mms_reuse_contract_test")
    shape = (2, 2, 2)
    state = driver.blob.FciDrbEBState(**{
        field: np.full(shape, index + 0.25, dtype=np.float64)
        for index, field in enumerate(driver.FIELDS)
    })
    configuration = {"evolved_initial_phi": "analytic-manufactured"}
    history = {
        "run_metadata_json": np.asarray(json.dumps(configuration)),
        "times": np.asarray((0.0, 1.0e-6)),
        **{
            field: np.stack((getattr(state, field), getattr(state, field)))
            for field in driver.FIELDS
        },
    }
    kwargs = dict(
        expected_configuration=configuration,
        expected_initial_state=state,
        start_time=0.0,
        timestep=1.0e-6,
        num_steps=1,
        save_every=1,
    )
    driver._validate_reusable_history(history, "history.npz", **kwargs)

    stale_metadata = dict(history)
    stale_metadata["run_metadata_json"] = np.asarray(json.dumps({
        "evolved_initial_phi": "reconstruct-from-vorticity"
    }))
    with pytest.raises(ValueError, match="production configuration mismatch"):
        driver._validate_reusable_history(
            stale_metadata, "history.npz", **kwargs
        )

    reconstructed = dict(history)
    reconstructed["phi"] = np.asarray(history["phi"]).copy()
    reconstructed["phi"][0, 0, 0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="initial phi is not the analytical"):
        driver._validate_reusable_history(
            reconstructed, "history.npz", **kwargs
        )


def test_host_frozen_stage_uses_requested_dt_for_short_leg_selection():
    source = DRIVER.read_text(encoding="utf-8")
    host_branch = source[
        source.index("else:\n        short_leg_step"):
        source.index("spatial, ledger = frozen_stage", source.index("else:\n        short_leg_step"))
    ]
    assert "short_leg_selection_dt=float(args.dt)" in host_branch


def test_production_history_exposes_owner_measure_for_temporal_self_convergence():
    production_source = (ROOT / "simulate_hsx_blob.py").read_text(encoding="utf-8")
    history_writer = production_source[production_source.index("np.savez_compressed(\n        output_path") :]
    assert '"owner_active"' in history_writer
    assert '"owner_aggregate_volume"' in history_writer
    assert "owner_host_geometry.topology.is_active_owner" in history_writer
    assert "owner_host_geometry.aggregate_chart_volume" in history_writer

    mms_source = DRIVER.read_text(encoding="utf-8")
    assert "continuum_total_error_by_field=integration_by_field" in mms_source
    assert "includes-spatial-reconstruction-and-time-integration-error" in mms_source


def test_compact_rows_preserve_execution_metadata_after_private_runtime_drop():
    source = DRIVER.read_text(encoding="utf-8")
    assert 'result["frozen_execution"] = result["_runtime"].frozen_execution' in source
    aggregate = source[source.index("np.savez(args.output") :]
    assert 'rows[0]["frozen_execution"]' in aggregate
    assert 'rows[0]["_runtime"]' not in aggregate


def test_harness_exposes_eta_sharded_frozen_and_evolved_runtime_contract():
    source = DRIVER.read_text(encoding="utf-8")
    assert '"--shard-counts"' in source
    assert "_validate_shard_configuration" in source
    assert "build_local_fci_geometries(\n        geometry, shard_counts" in source
    assert "mesh=make_shard_mesh(shard_counts)" in source
    assert 'frozen_execution = "eta-sharded"' in source
    assert "FrozenEbDiagnosticRequest(" in source
    assert 'evolved_execution="eta-sharded"' in source
    assert '"--metric-cache-dir"' in source
    assert '"--rebuild-metric-cache"' in source
    assert "metric_cache_dir=args.metric_cache_dir" in source
    assert "rebuild_metric_cache=bool(args.rebuild_metric_cache)" in source


def test_eta_shard_configuration_rejects_invalid_or_unavailable_layouts():
    driver = _load(DRIVER, "simulate_hsx_mms_sharding_contract_test")
    assert driver._validate_shard_configuration(
        (1, 1, 4), (32, 48, 64), available_devices=4
    ) == ((1, 1, 4), 4)
    with pytest.raises(ValueError, match="eta-only"):
        driver._validate_shard_configuration(
            (1, 2, 2), (32,), available_devices=4
        )
    with pytest.raises(ValueError, match="requires 4 JAX devices"):
        driver._validate_shard_configuration(
            (1, 1, 4), (32,), available_devices=1
        )
    with pytest.raises(ValueError, match="not divisible"):
        driver._validate_shard_configuration(
            (1, 1, 4), (30,), available_devices=4
        )


def test_frozen_diagnostic_hook_assembles_global_two_device_outputs():
    """All structured frozen outputs retain their global shard-map shapes."""

    environment = dict(os.environ)
    flags = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        f"{flags} --xla_force_host_platform_device_count=2"
    ).strip()
    environment["JAX_ENABLE_X64"] = "true"
    environment["DRBX_PARALLEL_SHORT_LEG_TREATMENT"] = "local-backward-euler"
    completed = subprocess.run(
        [sys.executable, str(FROZEN_DIAGNOSTIC_CASE)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-5000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["device_count"] == 2
    assert payload["source_pairing"] < 1.0e-14
    assert payload["reconstructed_shift"] < 1.0e-14
    assert payload["exact_term_shape"] == [6, 10, 2, 3, 4]
    assert payload["sourced_term_shape"] == [6, 10, 2, 3, 4]
    assert payload["reconstructed_term_shape"] == [6, 10, 2, 3, 4]
    assert payload["implicit_shape"] == [2, 3, 4, 5]
    assert payload["selected_all"]
    assert payload["reconstructed_selected_all"]
    assert payload["phi_diagnostics"][2:4] == [0.0, 1.0]


def test_runtime_avoids_duplicate_host_model_for_sharded_frozen_diagnostic(
    monkeypatch,
):
    """The sharded frozen path must bind only the production shard payload."""

    driver = _load(DRIVER, "simulate_hsx_mms_runtime_split_test")
    import drbx.native as native
    import drbx.native.fci_angular_agglomeration as angular

    monkeypatch.setattr(
        driver.blob.jax, "devices", lambda: [object(), object()]
    )
    # The production helper uses dataclasses.replace on LocalDomain3D.  Keep
    # this unit test focused on payload binding rather than reproducing that
    # native domain dataclass.
    monkeypatch.setattr(driver, "replace", lambda obj, **changes: obj)
    calls = []
    descriptors = []

    class Local:
        def __init__(self, counts):
            self.domain = SimpleNamespace(mesh_axis_names=("x", "y", "z"))
            self.shard_counts = counts

    def fake_build_local(geometry, counts, **kwargs):
        counts = tuple(counts)
        calls.append(counts)
        return Local(counts)

    def fake_payload(host, domain, **kwargs):
        descriptor = SimpleNamespace(compact_face_count=0, token=len(descriptors))
        descriptors.append(descriptor)
        return descriptor, f"control-{descriptor.token}"

    def fake_model(*args, **kwargs):
        return SimpleNamespace(
            parallel_operator_scheme="fci",
            parallel_material_scheme="production-path",
            parallel_flux_pairing="support-core",
            parallel_boundary_pairing="characteristic-sat",
            parallel_short_leg_treatment="local-backward-euler",
            parallel_short_leg_selection="all-physical-walls",
            poisson_bracket_scheme="material-scalar-third-order-upwind",
            parameters=SimpleNamespace(
                parallel_characteristic_wall_law="energy-absorbing"
            ),
            control_volume_geometry=object(),
            neumann_normal_scheme="physical",
            physical_wall_model_name="legacy-velocity-trace",
            gmres_config=SimpleNamespace(
                tol=1.0e-8,
                acceptance_tol=5.0e-5,
                maxiter=500,
                restart=100,
                preconditioner="line-u",
                residual_correction_steps=1,
            ),
        )

    monkeypatch.setattr(native, "build_local_fci_geometries", fake_build_local)
    monkeypatch.setattr(
        native, "assemble_single_device_local_fci_geometry",
        lambda local: SimpleNamespace(host_counts=local.shard_counts),
    )
    monkeypatch.setattr(
        native, "make_shard_mesh", lambda counts: ("mesh", tuple(counts))
    )
    monkeypatch.setattr(
        angular, "build_sharded_polar_angular_agglomeration_payload",
        fake_payload,
    )
    monkeypatch.setattr(
        angular,
        "assemble_local_polar_angular_agglomeration_geometry",
        lambda descriptor, fields, local: ("cv", descriptor.token, fields),
    )
    monkeypatch.setattr(
        angular,
        "empty_angular_agglomeration_boundary_bc",
        lambda **kwargs: ("bc", kwargs["max_rows"]),
    )
    monkeypatch.setattr(
        driver.blob,
        "FciDrbEBRhsParameters",
        lambda **kwargs: SimpleNamespace(
            parallel_characteristic_wall_law=kwargs[
                "parallel_characteristic_wall_law"
            ]
        ),
    )
    monkeypatch.setattr(driver.blob, "build_local_eb_model", fake_model)

    args = SimpleNamespace(shard_counts=(1, 1, 2))
    geometry = SimpleNamespace(shape=(8, 8, 8))
    runtime = driver._runtime(geometry, object(), args)

    assert calls == [(1, 1, 2)]
    assert runtime.sharded_geometry.shard_counts == (1, 1, 2)
    assert runtime.control_volume_descriptor is descriptors[0]
    assert runtime.host_control_volume_descriptor is None
    assert runtime.control_volume_fields == "control-0"
    assert runtime.host_control_volume_fields is None
    assert runtime.control_volume_boundary_bc == ("bc", 0)
    assert runtime.host_control_volume_boundary_bc is None
    assert runtime.model is None
    assert runtime.local_geometry is None
    assert runtime.mesh == ("mesh", (1, 1, 2))
    assert runtime.frozen_execution == "eta-sharded"
    assert runtime.evolved_execution == "eta-sharded"


def test_runtime_preserves_single_device_host_frozen_path(monkeypatch):
    """One-device development smokes retain the established local model."""

    driver = _load(DRIVER, "simulate_hsx_mms_runtime_single_test")
    import drbx.native as native
    import drbx.native.fci_angular_agglomeration as angular

    monkeypatch.setattr(driver.blob.jax, "devices", lambda: [object()])
    monkeypatch.setattr(driver, "replace", lambda obj, **changes: obj)

    local = SimpleNamespace(
        domain=SimpleNamespace(mesh_axis_names=("x", "y", "z")),
        shard_counts=(1, 1, 1),
    )
    descriptor = SimpleNamespace(compact_face_count=0)
    model = SimpleNamespace(
        parallel_operator_scheme="fci",
        parallel_material_scheme="production-path",
        parallel_flux_pairing="support-core",
        parallel_boundary_pairing="characteristic-sat",
        parallel_short_leg_treatment="local-backward-euler",
        parallel_short_leg_selection="all-physical-walls",
        poisson_bracket_scheme="material-scalar-third-order-upwind",
        parameters=SimpleNamespace(
            parallel_characteristic_wall_law="energy-absorbing"
        ),
        control_volume_geometry=object(),
        neumann_normal_scheme="physical",
        physical_wall_model_name="legacy-velocity-trace",
        gmres_config=SimpleNamespace(
            tol=1.0e-8,
            acceptance_tol=5.0e-5,
            maxiter=500,
            restart=100,
            preconditioner="line-u",
            residual_correction_steps=1,
        ),
    )
    monkeypatch.setattr(
        native, "build_local_fci_geometries", lambda *args, **kwargs: local
    )
    monkeypatch.setattr(
        native, "assemble_single_device_local_fci_geometry",
        lambda value: SimpleNamespace(local=value),
    )
    monkeypatch.setattr(native, "make_shard_mesh", lambda counts: "mesh")
    monkeypatch.setattr(
        angular,
        "build_sharded_polar_angular_agglomeration_payload",
        lambda *args, **kwargs: (descriptor, "fields"),
    )
    monkeypatch.setattr(
        angular,
        "assemble_local_polar_angular_agglomeration_geometry",
        lambda *args: "cv",
    )
    monkeypatch.setattr(
        angular,
        "empty_angular_agglomeration_boundary_bc",
        lambda **kwargs: "bc",
    )
    monkeypatch.setattr(
        driver.blob,
        "FciDrbEBRhsParameters",
        lambda **kwargs: SimpleNamespace(
            parallel_characteristic_wall_law="energy-absorbing"
        ),
    )
    monkeypatch.setattr(driver.blob, "build_local_eb_model", lambda *a, **k: model)

    runtime = driver._runtime(
        SimpleNamespace(shape=(8, 8, 8)),
        object(),
        SimpleNamespace(shard_counts=(1, 1, 1)),
    )
    assert runtime.model is model
    assert runtime.local_geometry is not None
    assert runtime.host_control_volume_descriptor is descriptor
    assert runtime.frozen_execution == "host-single-device"


def test_region_partition_is_disjoint_and_complete_for_active_owners():
    driver = _load(DRIVER, "simulate_hsx_mms_regions_test")
    shape = (4, 6, 4)
    active = np.ones(shape, dtype=bool)
    owner_index = np.stack(np.indices(shape), axis=-1)
    host = type("Host", (), {
        "topology": type("Topology", (), {
            "is_active_owner": active,
            "owner_index": owner_index,
            "aggregate_id": np.arange(np.prod(shape)).reshape(shape),
        })(),
        "angular_group_size": np.asarray((6, 3, 1, 1)),
    })()
    forward = np.zeros(shape, dtype=bool)
    backward = np.zeros(shape, dtype=bool)
    forward[3, :2, :] = True
    backward[3, 1:3, :] = True
    geometry = type("Geometry", (), {
        "maps": type("Maps", (), {
            "forward_boundary": forward,
            "backward_boundary": backward,
        })(),
    })()
    masks = driver._region_masks(geometry, host, forward | backward)
    stacked = np.stack([masks[name] for name in driver.REGIONS])
    assert np.all(np.sum(stacked, axis=0) == active.astype(np.int64))


def test_region_partition_promotes_alias_wall_hits_to_rlp_owner():
    driver = _load(DRIVER, "simulate_hsx_mms_alias_regions_test")
    shape = (1, 4, 1)
    active = np.zeros(shape, dtype=bool)
    active[0, 0, 0] = True
    active[0, 2, 0] = True
    owner_index = np.zeros(shape + (3,), dtype=np.int64)
    owner_index[0, :, 0, 0] = 0
    owner_index[0, :2, 0, 1] = 0
    owner_index[0, 2:, 0, 1] = 2
    aggregate_id = np.asarray((0, 0, 2, 2), dtype=np.int64).reshape(shape)
    host = type("Host", (), {
        "topology": type("Topology", (), {
            "is_active_owner": active,
            "owner_index": owner_index,
            "aggregate_id": aggregate_id,
        })(),
        "angular_group_size": np.asarray((2,)),
    })()
    forward = np.zeros(shape, dtype=bool)
    backward = np.zeros(shape, dtype=bool)
    forward[0, 1, 0] = True
    geometry = type("Geometry", (), {
        "maps": type("Maps", (), {
            "forward_boundary": forward,
            "backward_boundary": backward,
        })(),
    })()
    masks = driver._region_masks(geometry, host, forward)
    assert masks["short_leg_topology_transition"][0, 0, 0]
    assert not masks["ordinary_bulk"][0, 0, 0]


@pytest.mark.parametrize(
    ("amplitudes", "expected"),
    (
        ((1.0, 1.0, 1.0, 1.0, 1.0), "bounded-or-decaying-closure-layer"),
        ((1.0, 1.4, 2.0, 2.8, 4.0), "positive-growth"),
    ),
)
def test_short_leg_growth_classifier_requires_sustained_growth(
    tmp_path, amplitudes, expected
):
    driver = _load(
        DRIVER, f"simulate_hsx_mms_growth_{expected.replace('-', '_')}"
    )
    shape = (4, 6, 4)
    times = np.arange(len(amplitudes), dtype=np.float64)
    angular = np.sin(2.0 * np.pi * 2.0 * np.arange(shape[1]) / shape[1])
    mode = np.broadcast_to(angular[None, :, None], shape)
    payload = {"times": times}
    for field in driver.EVOLVED:
        payload[field] = np.stack([
            float(amplitude) * mode for amplitude in amplitudes
        ])
    path = tmp_path / "history.npz"
    np.savez(path, **payload)
    owner_index = np.stack(np.indices(shape), axis=-1)
    host = type("Host", (), {
        "topology": type("Topology", (), {"owner_index": owner_index})(),
    })()
    masks = {
        "short_leg_topology_transition": np.ones(shape, dtype=bool),
        "physical_wall": np.zeros(shape, dtype=bool),
    }
    zero = np.zeros(shape)
    exact = driver.blob.FciDrbEBState(
        density=zero,
        phi=zero,
        Te=zero,
        Ti=zero,
        Vi=zero,
        Ve=zero,
        vorticity=zero,
    )
    result = driver._short_leg_mode_history(
        path, lambda _time: (exact, None, None, None), host, masks
    )
    assert result["classification"] == [expected] * len(driver.EVOLVED)


def test_short_leg_exact_initial_frame_has_zero_finite_high_mode_fraction(
    tmp_path,
):
    driver = _load(DRIVER, "simulate_hsx_mms_zero_initial_spectrum")
    shape = (3, 6, 2)
    angular = np.sin(2.0 * np.pi * 2.0 * np.arange(shape[1]) / shape[1])
    mode = np.broadcast_to(angular[None, :, None], shape)
    path = tmp_path / "history.npz"
    np.savez(
        path,
        times=np.asarray((0.0, 1.0)),
        **{
            field: np.stack((np.zeros(shape), mode))
            for field in driver.EVOLVED
        },
    )
    owner_index = np.stack(np.indices(shape), axis=-1)
    host = type("Host", (), {
        "topology": type("Topology", (), {"owner_index": owner_index})(),
    })()
    masks = {
        "short_leg_topology_transition": np.ones(shape, dtype=bool),
        "physical_wall": np.zeros(shape, dtype=bool),
    }
    zero = np.zeros(shape)
    exact = driver.blob.FciDrbEBState(
        density=zero,
        phi=zero,
        Te=zero,
        Ti=zero,
        Vi=zero,
        Ve=zero,
        vorticity=zero,
    )
    result = driver._short_leg_mode_history(
        path, lambda _time: (exact, None, None, None), host, masks
    )
    assert np.all(result["high_mode_fraction"][0] == 0.0)
    assert np.all(np.isfinite(result["high_mode_fraction"]))
