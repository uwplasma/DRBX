"""Focused tests for the login-node Stage-7 MMS artifact analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_hsx_mms.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_hsx_mms_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONFIGURATION = dict(_load().EXPECTED_CONFIGURATION)
FIELDS = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")
REGIONS = (
    "ordinary_bulk",
    "rlp_rings",
    "rlp_transition_rings",
    "physical_wall",
    "short_leg_topology_transition",
    "double_hit",
)
TERMS = ("parallel", "curvature", "bracket", "diffusion")
PHYSICAL_PARAMETERS = {
    "tau": 1.0,
    "mi_over_me": 1836.0,
    "rho_star": 1.0,
    "density_D_perp": 1.0e-5,
    "density_D_parallel": 0.0,
    "electron_temperature_chi_parallel": 0.0,
    "electron_temperature_D_perp": 1.0e-5,
    "ion_temperature_chi_parallel": 0.0,
    "ion_temperature_D_perp": 1.0e-5,
    "Ve_nu": 0.0,
    "Ve_D_perp": 1.0e-5,
    "Ve_parallel_viscosity": 0.0,
    "Vi_D_perp": 1.0e-5,
    "Vi_parallel_viscosity": 0.0,
    "vorticity_D_perp": 1.0e-5,
    "vorticity_D_parallel": 0.0,
}


def test_expected_configuration_uses_artifact_geometry_contract() -> None:
    configuration = _load().EXPECTED_CONFIGURATION
    assert configuration["fci_trace_substeps"] == 64
    assert {
        "fit_sample_shape",
        "toroidal_modes",
        "metric_radial_degree",
        "metric_poloidal_modes",
        "metric_toroidal_modes",
        "eta_projection_iterations",
        "axis_core_radius",
        "makegrid_currents",
    }.isdisjoint(configuration)


def _write_aggregate(
    path: Path,
    resolutions: tuple[int, ...],
    *,
    dt: float | None = None,
    integration_scale: float | None = None,
    classes: list[list[str]] | None = None,
    diagnostic_path: Path | None = None,
    bad_source: bool = False,
    bad_configuration: bool = False,
    exact_multiplier: float = 1.0,
    direct_metadata: tuple[float, float, float] | None = None,
    command_dt: float | None = None,
    phi_converged: bool = True,
    phi_failed: bool = False,
    physical_parameters: dict[str, float] | None = None,
    omit_physical: bool = False,
    exact_power: float = -2.0,
    integration_spatial_power: float = 0.0,
    num_steps: int | None = None,
    shard_counts: tuple[int, int, int] = (1, 1, 4),
    device_count: int = 4,
    frozen_execution: str = "eta-sharded",
    evolved_execution: str = "eta-sharded",
    frozen_stage_graph_contract: str = "shared-mms-audited-stage-graph-v1",
    counterfactuals_enabled: bool = True,
    include_execution_contract: bool = True,
    include_reference_contract: bool = True,
    omit_generalized_potential: bool = False,
    history_paths: list[str | None] | None = None,
) -> Path:
    nrows = len(resolutions)
    ns = np.asarray(resolutions, dtype=float)
    errors = exact_multiplier * ns[:, None] ** exact_power * np.ones((nrows, len(FIELDS)))
    representation = ns[:, None] ** -3 * np.ones((nrows, len(FIELDS)))
    source = np.full((nrows, len(FIELDS)), 1.0e-14 if not bad_source else 1.0e-3)
    regional = ns[:, None, None] ** -2 * np.ones((nrows, len(REGIONS), len(FIELDS)))
    regional_terms = ns[:, None, None, None] ** -2 * np.ones((nrows, len(REGIONS), len(FIELDS), len(TERMS)))
    term_errors = ns[:, None, None] ** -2 * np.ones((nrows, len(FIELDS), len(TERMS)))
    material_controls = ns[:, None, None] ** -2 * np.ones((nrows, 4, 5))
    material_force_controls = ns[:, None] ** -2 * np.ones((nrows, 4))
    poisson_controls = ns[:, None, None] ** -2 * np.ones(
        (nrows, 4, len(FIELDS))
    )
    generalized_controls = ns[:, None] ** -2 * np.ones((nrows, 3))
    generalized_differences = ns[:, None] ** -2 * np.ones((nrows, 3))
    counts = np.full((nrows, len(REGIONS)), 10, dtype=np.int64)
    if integration_scale is None:
        integration = np.full(nrows, np.nan)
        integration_by_field = np.full((nrows, len(FIELDS)), np.nan)
    else:
        integration_by_field = (
            integration_scale
            * float(dt) ** 2
            * ns[:, None] ** integration_spatial_power
            * np.ones((nrows, len(FIELDS)))
        )
        integration = np.mean(integration_by_field, axis=1)
    if classes is None:
        classes = [[] for _ in resolutions]
    if diagnostic_path is None:
        diagnostic_paths = [None for _ in resolutions]
    else:
        diagnostic_paths = [str(diagnostic_path) for _ in resolutions]
    if history_paths is None:
        history_paths = [None for _ in resolutions]
    config = dict(CONFIGURATION)
    if bad_configuration:
        config["time_integrator"] = "rk4"
    command = ["python", "simulate_hsx_mms.py"]
    if dt is not None:
        command += ["--dt", str(command_dt if command_dt is not None else dt), "--time", "0", "--final-time", "0.001"]
    payload = dict(
        resolutions=np.asarray(resolutions, dtype=np.int64),
        metric_reference_resolution=np.asarray((64, 64, 64), dtype=np.int64),
        reference_magnetic_field=np.asarray(1.25),
        nfp=np.asarray(4, dtype=np.int32),
        fci_trace_substeps=np.asarray(64, dtype=np.int32),
        shard_counts=np.asarray(shard_counts, dtype=np.int32),
        device_count=np.asarray(device_count, dtype=np.int32),
        frozen_execution=np.asarray(frozen_execution),
        evolved_execution=np.asarray(evolved_execution),
        frozen_stage_graph_contract=np.asarray(frozen_stage_graph_contract),
        counterfactuals_enabled=np.asarray(counterfactuals_enabled, dtype=bool),
        exact_phi_residual=errors,
        forced_residual=errors * 1.1,
        source_increment=source,
        source_total_subtraction_residual=source,
        source_pairing_method=np.asarray("direct-source-term-ledger"),
        representation_error=representation,
        phi_reconstruction_difference=errors[:, 0],
        reconstructed_phi_residual=errors[:, 0] * 1.2,
        phi_reconstruction_rhs_difference=errors[:, 0] * 1.3,
        phi_converged=np.asarray(phi_converged),
        phi_failed=np.asarray(phi_failed),
        integration_error=integration,
        integration_error_by_field=integration_by_field,
        partitioned_exact_phi_residual=regional,
        partitioned_forced_residual=regional * 1.1,
        partitioned_rhs_term_norms=regional_terms,
        rhs_term_error_norms=term_errors,
        partitioned_rhs_term_error_norms=regional_terms,
        material_counterfactual_error_norms=material_controls,
        partitioned_material_counterfactual_error_norms=np.broadcast_to(
            material_controls[:, None], (nrows, len(REGIONS), 4, 5)
        ),
        material_force_control_norms=material_force_controls,
        partitioned_material_force_control_norms=np.broadcast_to(
            material_force_controls[:, None], (nrows, len(REGIONS), 4)
        ),
        poisson_operand_error_norms=poisson_controls,
        partitioned_poisson_operand_error_norms=np.broadcast_to(
            poisson_controls[:, None],
            (nrows, len(REGIONS), 4, len(FIELDS)),
        ),
        generalized_potential_control_error_norms=generalized_controls,
        partitioned_generalized_potential_control_error_norms=np.broadcast_to(
            generalized_controls[:, None],
            (nrows, len(REGIONS), 3),
        ),
        generalized_potential_difference_norms=generalized_differences,
        partitioned_generalized_potential_difference_norms=np.broadcast_to(
            generalized_differences[:, None],
            (nrows, len(REGIONS), 3),
        ),
        region_cell_counts=counts,
        field_names_json=np.asarray(json.dumps(FIELDS)),
        region_names_json=np.asarray(json.dumps(REGIONS)),
        rhs_term_names_json=np.asarray(json.dumps(TERMS)),
        material_counterfactual_names_json=np.asarray(json.dumps(
            ("live", "centered", "difference", "principal")
        )),
        material_force_control_names_json=np.asarray(json.dumps(
            ("material_ti", "canonical_ti", "sum", "complete")
        )),
        poisson_operand_control_names_json=np.asarray(json.dumps(
            ("raw_raw", "H_raw", "raw_H", "H_H")
        )),
        generalized_potential_control_names_json=np.asarray(json.dumps((
            "compatible_composite_gradient",
            "split_primitive_gradient",
            "explicit_electrostatic_gradient",
        ))),
        generalized_potential_error_control_names_json=np.asarray(json.dumps((
            "compatible_composite_gradient",
            "split_primitive_gradient",
            "explicit_electrostatic_gradient",
        ))),
        generalized_potential_difference_names_json=np.asarray(json.dumps((
            "compatible_minus_split",
            "split_minus_primitive_sum",
            "ledger_minus_explicit_control",
        ))),
        production_configuration_json=np.asarray(json.dumps(config)),
        command_json=np.asarray(json.dumps(command)),
        short_leg_diagnostics_paths_json=np.asarray(json.dumps(diagnostic_paths)),
        short_leg_classification_json=np.asarray(json.dumps(classes)),
        history_paths_json=np.asarray(json.dumps(history_paths)),
        **({
            "actual_timestep": np.asarray(direct_metadata[0]),
            "start_time": np.asarray(direct_metadata[1]),
            "final_time": np.asarray(direct_metadata[2]),
        } if direct_metadata is not None else {}),
        **({"num_steps": np.asarray(num_steps, dtype=np.int64)} if num_steps is not None else {}),
    )
    if not omit_physical:
        payload["physical_parameters_json"] = np.asarray(
            json.dumps(PHYSICAL_PARAMETERS if physical_parameters is None else physical_parameters)
        )
    if not include_execution_contract:
        for key in (
            "shard_counts", "device_count", "frozen_execution", "evolved_execution"
        ):
            payload.pop(key)
    if include_reference_contract:
        if not omit_generalized_potential:
            payload["generalized_potential_enabled"] = np.asarray(True, dtype=bool)
        payload["reference_projection_method"] = np.asarray(
            "jacobian-weighted-cell-midpoint"
        )
        payload["reference_projection_order"] = np.asarray(2, dtype=np.int32)
        payload["reference_derivative_method"] = np.asarray(
            "structured-nonuniform-five-point-finite-difference"
        )
        payload["reference_derivative_order"] = np.asarray(4, dtype=np.int32)
        payload["reference_periodic_axes_json"] = np.asarray(json.dumps({
            "theta": "geometry-domain",
            "eta": "geometry-domain",
        }))
    np.savez(path, **payload)
    return path


def _write_diagnostic(path: Path, *, positive: bool = False) -> Path:
    amplitudes = np.asarray((1.0, 1.4, 2.0, 2.8, 4.0) if positive else (1.0,) * 5)
    np.savez(
        path,
        times=np.arange(amplitudes.size, dtype=float),
        high_mode_fraction=amplitudes[:, None],
        high_mode_rms=amplitudes[:, None],
        maximum_poloidal_jump=amplitudes[:, None],
        late_log_growth_rate=np.asarray((0.3,)),
        late_growth_factor=np.asarray((4.0 if positive else 1.0,)),
        late_growth_r_squared=np.asarray((0.99,)),
        classification_json=np.asarray(json.dumps(
            ["positive-growth" if positive else "bounded-or-decaying-closure-layer"]
            * len(FIELDS)
        )),
    )
    return path


def _write_temporal_history(
    path: Path,
    *,
    timestep: float,
    temporal_power: float = 2.0,
    include_owner_measure: bool = True,
) -> Path:
    shape = (2, 2, 2)
    active = np.ones(shape, dtype=bool)
    active[1, 1, 1] = False
    volume = np.arange(1, 9, dtype=np.float64).reshape(shape)
    pattern = np.arange(1, 9, dtype=np.float64).reshape(shape) / 8.0
    payload: dict[str, np.ndarray] = {
        "times": np.asarray((0.0, 2.0e-5), dtype=np.float64),
        "history_dtype": np.asarray("float64"),
    }
    if include_owner_measure:
        payload["owner_active"] = active
        payload["owner_aggregate_volume"] = volume
    for index, field in enumerate(FIELDS, start=1):
        common_spatial_state = index + 0.05 * pattern
        final = (
            common_spatial_state
            + index * 1.0e-3 * (timestep / 1.0e-6) ** temporal_power * pattern
        )
        payload[field] = np.stack((common_spatial_state, final), axis=0)
    np.savez(path, **payload)
    return path


def test_analyzer_merges_spatial_temporal_and_short_leg_artifacts(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "frozen.npz", (32, 48, 64))
    diagnostic = _write_diagnostic(tmp_path / "short_leg.npz")
    evolved_coarse = _write_aggregate(
        tmp_path / "evolved_N64_dt1e-3.npz", (64,), dt=1.0e-3,
        integration_scale=2.0, classes=[["bounded-or-decaying-closure-layer"] * 6],
        diagnostic_path=diagnostic,
        direct_metadata=(1.0e-3, 0.0, 2.0e-3),
        command_dt=9.0e-4,
    )
    evolved_fine = _write_aggregate(
        tmp_path / "evolved_N64_dt5e-4.npz", (64,), dt=5.0e-4,
        integration_scale=2.0, classes=[["bounded-or-decaying-closure-layer"] * 6],
        diagnostic_path=diagnostic,
        direct_metadata=(5.0e-4, 0.0, 2.0e-3),
        command_dt=4.0e-4,
    )
    report = analyzer.analyze((frozen, evolved_coarse, evolved_fine))
    assert report["ok"]
    assert report["merged_resolutions"] == [32, 48, 64]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["production_configuration_and_fixed_metric"]["status"] == "pass"
    assert checks["finite_populated_region_norms"]["status"] == "pass"
    assert checks["independent_source_pairing_roundoff"]["status"] == "pass"
    assert checks["frozen_stage_graph_consistency"]["status"] == "pass"
    assert checks["phi_reconstruction_evidence"]["status"] == "pass"
    counterfactuals = checks["mms_material_and_poisson_counterfactuals"]
    assert counterfactuals["status"] == "pass"
    assert np.allclose(
        counterfactuals["diagnostics"]["material"]["global"]["orders"],
        2.0,
    )
    assert np.allclose(
        counterfactuals["diagnostics"]["generalized_potential"]["global"]
        ["orders"],
        2.0,
    )
    assert np.allclose(
        counterfactuals["diagnostics"]["generalized_potential_identities"]
        ["global"]["orders"],
        2.0,
    )
    total = checks["continuum_total_error_by_timestep"]
    assert total["status"] == "diagnostic"
    assert "no temporal order is inferred" in total["interpretation"]
    assert checks["complete_stage7_acceptance"]["status"] == "unavailable"
    spatial = report["spatial"]["quantities"]
    assert np.allclose(spatial["rhs_term_error_norms"]["orders"], 2.0)
    owner = checks["owner_vs_fine_rlp_representation_error"]
    assert owner["status"] == "pass"
    assert owner["representation_to_owner_ratio"][0][0] == 1.0 / 32.0
    short_leg = checks["short_leg_growth_classification"]
    assert short_leg["status"] == "pass"
    assert short_leg["records"][-1]["sample_count"] == 5
    assert report["artifacts"][1]["final_time"] == 2.0e-3
    assert report["artifacts"][1]["dt"] == 1.0e-3


def test_spatial_selection_prefers_multi_resolution_campaign(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "frozen.npz", (32, 48, 64))
    temporal = _write_aggregate(
        tmp_path / "evolved_N64_dt1e-3.npz",
        (64,),
        dt=1.0e-3,
        integration_scale=2.0,
        exact_multiplier=1.0e6,
    )
    report = analyzer.analyze((frozen, temporal))
    quantity = report["spatial"]["quantities"]["exact_phi_residual"]
    assert quantity["resolutions"] == [32, 48, 64]
    # The N64 row must come from frozen.npz, not the single-resolution
    # temporal artifact with deliberately incompatible spatial error.
    assert np.isclose(quantity["values"][-1][0], 64.0 ** -2)


def test_single_resolution_counterfactual_controls_remain_reported(tmp_path: Path):
    analyzer = _load()
    audit = _write_aggregate(tmp_path / "audit_N64.npz", (64,))
    report = analyzer.analyze((audit,))
    check = {item["name"]: item for item in report["checks"]}[
        "mms_material_and_poisson_counterfactuals"
    ]
    assert check["status"] == "warning"
    assert "at least two resolutions" in check["reason"]
    material = check["diagnostics"]["material"]["global"]
    assert material["resolutions"] == [64]
    assert len(material["values"]) == 1
    assert material["orders"] == []


def test_phi_flags_are_hard_failures(tmp_path: Path):
    analyzer = _load()
    bad = _write_aggregate(
        tmp_path / "bad_phi.npz",
        (32, 48, 64),
        phi_converged=False,
        phi_failed=True,
    )
    report = analyzer.analyze((bad,))
    check = {item["name"]: item for item in report["checks"]}["phi_reconstruction_evidence"]
    assert check["status"] == "fail"
    assert not report["ok"]


def test_analyzer_reports_hard_gate_failures(tmp_path: Path):
    analyzer = _load()
    bad = _write_aggregate(
        tmp_path / "bad.npz", (32, 48, 64), bad_source=True, bad_configuration=True
    )
    report = analyzer.analyze((bad,))
    assert not report["ok"]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["production_configuration_and_fixed_metric"]["status"] == "fail"
    assert checks["independent_source_pairing_roundoff"]["status"] == "fail"
    assert report["failure_count"] >= 2


def test_legacy_total_rhs_source_subtraction_is_not_a_pairing_failure(tmp_path: Path):
    analyzer = _load()
    artifact = _write_aggregate(tmp_path / "legacy_source.npz", (32, 48, 64))
    with np.load(artifact, allow_pickle=False) as stored:
        payload = {
            key: np.asarray(stored[key])
            for key in stored.files
            if key not in {
                "source_total_subtraction_residual",
                "source_pairing_method",
            }
        }
    payload["source_increment"] = np.full((3, len(FIELDS)), 2.1e-6)
    np.savez(artifact, **payload)

    report = analyzer.analyze((artifact,))
    check = {item["name"]: item for item in report["checks"]}[
        "independent_source_pairing_roundoff"
    ]
    assert check["status"] == "warning"
    assert check["maximum"] is None
    assert np.isclose(check["total_subtraction_maximum"], 2.1e-6)
    assert check["legacy_diagnostics"]
    # A legacy cancellation diagnostic is advisory: it cannot prove pairing,
    # but it must not turn an otherwise valid artifact into a hard failure.
    assert report["ok"]


def test_physical_parameters_are_required_and_identical(tmp_path: Path):
    analyzer = _load()
    mismatch = dict(PHYSICAL_PARAMETERS)
    mismatch["Ve_nu"] = 1.0e-3
    first = _write_aggregate(tmp_path / "first.npz", (32, 48, 64))
    second = _write_aggregate(
        tmp_path / "second.npz", (32, 48, 64), physical_parameters=mismatch
    )
    report = analyzer.analyze((first, second))
    physical = {item["name"]: item for item in report["checks"]}["physical_parameters"]
    assert physical["status"] == "fail"
    assert any("differs exactly" in message for message in physical["failures"])

    missing = _write_aggregate(
        tmp_path / "new_missing_parameters.npz", (32, 48, 64), omit_physical=True
    )
    report = analyzer.analyze((missing,))
    physical = {item["name"]: item for item in report["checks"]}["physical_parameters"]
    assert physical["status"] == "fail"
    assert report["failure_count"] >= 1


def test_known_legacy_8_artifact_missing_parameters_is_warning(tmp_path: Path):
    analyzer = _load()
    legacy = _write_aggregate(
        tmp_path / "hsx_mms_frozen8_final.npz", (8,), omit_physical=True
    )
    report = analyzer.analyze((legacy,))
    physical = {item["name"]: item for item in report["checks"]}["physical_parameters"]
    assert physical["status"] == "warning"
    assert report["ok"]


def test_analyzer_allows_frozen_unavailable_temporal_ledger(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "frozen.npz", (32, 48, 64))
    report = analyzer.analyze((frozen,))
    assert report["ok"]
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["complete_stage7_acceptance"]["status"] == "unavailable"
    assert checks["short_leg_growth_classification"]["status"] == "unavailable"
    assert report["incomplete_count"] == 2


def test_strict_cli_fails_for_incomplete_optional_checks(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "frozen.npz", (32, 48, 64))
    assert analyzer.main([str(frozen)]) == 0
    assert analyzer.main([str(frozen), "--strict"]) == 2


def test_require_spatial_accepts_frozen_only_without_optional_campaigns(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "frozen.npz", (32, 48, 64))
    report = analyzer.analyze((frozen,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}["spatial_convergence_gate"]
    assert report["ok"]
    assert gate["status"] == "pass"
    assert gate["scope"] == "frozen-only"
    assert gate["full_stage7_spatial_gate"] is False
    assert gate["evolved_field_error"]["status"] == "not-run"
    assert report["spatial"]["status"] == "pass"
    assert analyzer.main([str(frozen), "--require-spatial"]) == 0


def test_lean_frozen_graph_is_descriptive_only_and_cannot_qualify(tmp_path: Path):
    analyzer = _load()
    lean = _write_aggregate(
        tmp_path / "lean.npz",
        (32, 48, 64),
        frozen_stage_graph_contract="shared-lean-stage-graph-v1",
        counterfactuals_enabled=False,
    )
    descriptive = analyzer.analyze((lean,))
    descriptive_check = {
        check["name"]: check for check in descriptive["checks"]
    }["frozen_stage_graph_consistency"]
    assert descriptive["ok"]
    assert descriptive_check["status"] == "warning"

    qualifying = analyzer.analyze((lean,), require_spatial=True)
    qualifying_check = {
        check["name"]: check for check in qualifying["checks"]
    }["frozen_stage_graph_consistency"]
    assert not qualifying["ok"]
    assert qualifying_check["status"] == "fail"
    assert any(
        "frozen_stage_graph_contract" in failure
        for failure in qualifying_check["failures"]
    )


def test_require_spatial_rejects_increasing_exact_phi_errors(tmp_path: Path):
    analyzer = _load()
    increasing = _write_aggregate(
        tmp_path / "increasing.npz", (32, 48, 64), exact_power=1.0
    )
    report = analyzer.analyze((increasing,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}["spatial_convergence_gate"]
    assert not report["ok"]
    assert gate["status"] == "fail"
    density = gate["exact_phi_owner_residual"]["fields"][0]
    assert density["strictly_decreasing"] is False
    assert density["finest_pair_order"] < 0.0
    assert report["spatial"]["quantities"]["exact_phi_residual"]["status"] == "fail"
    assert analyzer.main([str(increasing), "--require-spatial"]) == 2


def test_two_resolution_increase_is_not_reported_as_spatial_pass(tmp_path: Path):
    analyzer = _load()
    increasing = _write_aggregate(
        tmp_path / "two_increasing.npz", (32, 64), exact_power=1.0
    )
    report = analyzer.analyze((increasing,))
    exact = report["spatial"]["quantities"]["exact_phi_residual"]
    assert exact["status"] == "fail"
    assert exact["strictly_decreasing_component_count"] == 0
    # General analysis remains non-gating unless --require-spatial is asked
    # for, preserving its use as a descriptive partial-campaign report.
    assert report["ok"]


def test_require_spatial_rejects_subthreshold_evolved_finest_pair_order(tmp_path: Path):
    analyzer = _load()
    low_order = _write_aggregate(
        tmp_path / "low_order.npz",
        (32, 48, 64),
        dt=1.0e-3,
        integration_scale=1.0,
        integration_spatial_power=-1.7,
    )
    report = analyzer.analyze((low_order,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}["spatial_convergence_gate"]
    assert gate["status"] == "fail"
    assert gate["evolved_field_error"]["fields"][0]["finest_pair_order"] < 1.8


def test_require_spatial_distinguishes_decrease_from_second_order(tmp_path: Path):
    analyzer = _load()
    low_order = _write_aggregate(
        tmp_path / "low_order_frozen.npz", (32, 48, 64), exact_power=-1.0
    )
    report = analyzer.analyze((low_order,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "spatial_convergence_gate"
    ]
    exact = gate["exact_phi_owner_residual"]
    assert exact["strict_decrease_status"] == "pass"
    assert exact["observed_order_status"] == "fail"
    assert exact["full_second_order_acceptance"] is False
    assert gate["status"] == "fail"


def test_require_spatial_gates_evolved_fields_when_present(tmp_path: Path):
    analyzer = _load()
    evolved = _write_aggregate(
        tmp_path / "evolved.npz",
        (32, 48, 64),
        dt=1.0e-3,
        integration_scale=1.0,
        integration_spatial_power=1.0,
    )
    report = analyzer.analyze((evolved,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}["spatial_convergence_gate"]
    assert not report["ok"]
    assert gate["scope"] == "frozen-and-evolved"
    assert gate["evolved_field_error"]["status"] == "fail"
    assert gate["full_stage7_spatial_gate"] is False

    converged = _write_aggregate(
        tmp_path / "evolved_converged.npz",
        (32, 48, 64),
        dt=1.0e-3,
        integration_scale=1.0,
        integration_spatial_power=-2.0,
    )
    report = analyzer.analyze((converged,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}["spatial_convergence_gate"]
    assert report["ok"]
    assert gate["full_stage7_spatial_gate"] is True


def test_per_term_zero_and_unavailable_components_are_diagnostic(tmp_path: Path):
    analyzer = _load()
    artifact = _write_aggregate(tmp_path / "terms.npz", (32, 48, 64))
    with np.load(artifact, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    term_errors = payload["rhs_term_error_norms"].copy()
    term_errors[:, 0, 0] = 0.0
    term_errors[:, 1, 1] = np.nan
    payload["rhs_term_error_norms"] = term_errors
    np.savez(artifact, **payload)

    report = analyzer.analyze((artifact,))
    terms = report["spatial"]["quantities"]["rhs_term_error_norms"]
    assert terms["status"] == "diagnostic"
    assert terms["exact_zero_or_disabled_component_count"] == 1
    assert terms["unavailable_component_count"] == 1
    assert terms["assessed_positive_component_count"] == len(FIELDS) * len(TERMS) - 2


def test_descriptive_spatial_report_prefers_evolved_rows_over_frozen_placeholders(tmp_path: Path):
    analyzer = _load()
    frozen = _write_aggregate(tmp_path / "aaa_frozen.npz", (32, 48, 64))
    evolved = _write_aggregate(
        tmp_path / "zzz_evolved.npz",
        (32, 48, 64),
        dt=1.0e-3,
        integration_scale=1.0,
        integration_spatial_power=-2.0,
    )
    report = analyzer.analyze((frozen, evolved))
    assert report["ok"]
    assert report["spatial"]["quantities"]["integration_error_by_field"]["status"] == "pass"

    gated = analyzer.analyze((frozen, evolved), require_spatial=True)
    gate = {check["name"]: check for check in gated["checks"]}["spatial_convergence_gate"]
    assert not gated["ok"]
    assert gate["remote_contract"]["status"] == "fail"
    assert any("exactly one aggregate" in failure for failure in gate["failures"])


def test_require_spatial_rejects_non_remote_or_incomplete_contracts(tmp_path: Path):
    analyzer = _load()
    cases = (
        (
            "legacy",
            dict(include_execution_contract=False, include_reference_contract=False),
            "shard_counts must be",
        ),
        (
            "host",
            dict(frozen_execution="host-single-device"),
            "frozen_execution must be 'eta-sharded'",
        ),
        (
            "single_shard",
            dict(shard_counts=(1, 1, 1), device_count=1),
            "shard_counts must be",
        ),
        (
            "missing_generalized",
            dict(omit_generalized_potential=True),
            "generalized_potential_enabled must be present",
        ),
    )
    for name, options, expected_failure in cases:
        artifact = _write_aggregate(
            tmp_path / f"{name}.npz", (32, 48, 64), **options
        )
        descriptive = analyzer.analyze((artifact,))
        assert descriptive["ok"], name

        gated = analyzer.analyze((artifact,), require_spatial=True)
        gate = {check["name"]: check for check in gated["checks"]}[
            "spatial_convergence_gate"
        ]
        assert not gated["ok"], name
        assert gate["remote_contract"]["status"] == "fail"
        assert any(expected_failure in failure for failure in gate["failures"]), name


def test_require_spatial_rejects_unordered_resolutions_and_bad_periodic_metadata(
    tmp_path: Path,
):
    analyzer = _load()
    artifact = _write_aggregate(tmp_path / "bad_contract.npz", (32, 64, 48))
    with np.load(artifact, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["reference_periodic_axes_json"] = np.asarray(json.dumps({
        "theta": "geometry-domain",
        "eta": "field-period",
    }))
    payload["reference_derivative_method"] = np.asarray("legacy-centered-difference")
    payload["reference_derivative_order"] = np.asarray(2, dtype=np.int32)
    payload["reference_projection_method"] = np.asarray("tensor-gauss-2x2x2")
    payload["reference_projection_order"] = np.asarray(4, dtype=np.int32)
    np.savez(artifact, **payload)

    report = analyzer.analyze((artifact,), require_spatial=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "spatial_convergence_gate"
    ]
    assert gate["status"] == "fail"
    assert any("ordered one-dimensional sequence" in item for item in gate["failures"])
    assert any("both theta and eta" in item for item in gate["failures"])
    assert any("reference_derivative_method" in item for item in gate["failures"])
    assert any("reference_derivative_order" in item for item in gate["failures"])
    assert any("reference_projection_method" in item for item in gate["failures"])
    assert any("reference_projection_order" in item for item in gate["failures"])


def _write_valid_evolved_prerequisites(
    tmp_path: Path, *, positive: bool = False
) -> tuple[Path, Path, Path]:
    frozen = _write_aggregate(
        tmp_path / "frozen.npz",
        (32, 48, 64),
        direct_metadata=(0.0, 1.0e-6, 1.0e-6),
        num_steps=0,
    )
    diagnostic = _write_diagnostic(
        tmp_path / "baseline.short_leg_modes.npz", positive=positive
    )
    baseline = _write_aggregate(
        tmp_path / "evolved.npz",
        (32, 48, 64),
        dt=1.0e-6,
        integration_scale=1.0,
        integration_spatial_power=-2.0,
        diagnostic_path=diagnostic,
        classes=[
            ["bounded-or-decaying-closure-layer"] * len(FIELDS)
            for _ in (32, 48, 64)
        ],
        direct_metadata=(1.0e-6, 0.0, 2.0e-5),
        num_steps=20,
    )
    return frozen, baseline, diagnostic


def test_require_evolved_accepts_complete_baseline_and_positive_growth_warning(
    tmp_path: Path,
):
    analyzer = _load()
    frozen, baseline, _ = _write_valid_evolved_prerequisites(tmp_path)
    report = analyzer.analyze((frozen, baseline), require_evolved=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "evolved_prerequisite_gate"
    ]
    assert report["ok"]
    assert gate["status"] == "pass"
    assert gate["frozen_spatial_gate"]["scope"] == "frozen-only"
    assert len(gate["diagnostic_records"]) == 3
    assert analyzer.main(
        [str(frozen), str(baseline), "--require-evolved"]
    ) == 0

    positive_dir = tmp_path / "positive"
    positive_dir.mkdir()
    frozen, baseline, _ = _write_valid_evolved_prerequisites(
        positive_dir, positive=True
    )
    report = analyzer.analyze((frozen, baseline), require_evolved=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "evolved_prerequisite_gate"
    ]
    assert report["ok"]
    assert gate["status"] == "warning"
    assert gate["positive_growth_count"] == 3 * len(FIELDS)


def test_require_evolved_rejects_wrong_metadata_nonfinite_error_and_missing_diagnostic(
    tmp_path: Path,
):
    analyzer = _load()
    frozen, baseline, diagnostic = _write_valid_evolved_prerequisites(tmp_path)
    with np.load(baseline, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["num_steps"] = np.asarray(19, dtype=np.int64)
    payload["integration_error_by_field"] = payload[
        "integration_error_by_field"
    ].copy()
    payload["integration_error_by_field"][1, 2] = np.nan
    np.savez(baseline, **payload)
    diagnostic.unlink()

    report = analyzer.analyze((frozen, baseline), require_evolved=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "evolved_prerequisite_gate"
    ]
    assert not report["ok"]
    assert gate["status"] == "fail"
    assert any("num_steps" in failure for failure in gate["failures"])
    assert any("must be finite" in failure for failure in gate["failures"])
    assert any("unresolved short-leg" in failure for failure in gate["failures"])


def test_require_evolved_rejects_noncanonical_frozen_and_unknown_classification(
    tmp_path: Path,
):
    analyzer = _load()
    frozen, baseline, diagnostic = _write_valid_evolved_prerequisites(tmp_path)
    with np.load(frozen, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["resolutions"] = np.asarray((32, 64, 48), dtype=np.int64)
    np.savez(frozen, **payload)
    with np.load(diagnostic, allow_pickle=False) as source:
        diag_payload = {key: np.asarray(source[key]) for key in source.files}
    diag_payload["classification_json"] = np.asarray(
        json.dumps(["not-a-classification"] * len(FIELDS))
    )
    np.savez(diagnostic, **diag_payload)

    report = analyzer.analyze((frozen, baseline), require_evolved=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "evolved_prerequisite_gate"
    ]
    assert gate["status"] == "fail"
    assert any("frozen resolutions" in failure for failure in gate["failures"])
    assert any("unknown short-leg" in failure for failure in gate["failures"])


def _write_complete_stage7_artifacts(
    tmp_path: Path,
    *,
    temporal_power: float = 2.0,
) -> tuple[Path, Path, Path, Path]:
    frozen = _write_aggregate(
        tmp_path / "frozen.npz",
        (32, 48, 64),
        direct_metadata=(0.0, 1.0e-6, 1.0e-6),
        num_steps=0,
    )
    diagnostic = _write_diagnostic(tmp_path / "short_leg.npz")
    histories = [
        _write_temporal_history(
            tmp_path / f"history_{index}.npz",
            timestep=dt,
            temporal_power=temporal_power,
        )
        for index, dt in enumerate((1.0e-6, 5.0e-7, 2.5e-7))
    ]
    baseline = _write_aggregate(
        tmp_path / "evolved.npz",
        (32, 48, 64),
        dt=1.0e-6,
        integration_scale=1.0,
        integration_spatial_power=-2.0,
        diagnostic_path=diagnostic,
        direct_metadata=(1.0e-6, 0.0, 2.0e-5),
        num_steps=20,
        history_paths=[None, None, str(histories[0])],
    )
    fine = _write_aggregate(
        tmp_path / "temporal_5e-7.npz",
        (64,),
        dt=5.0e-7,
        integration_scale=1.0,
        diagnostic_path=diagnostic,
        direct_metadata=(5.0e-7, 0.0, 2.0e-5),
        num_steps=40,
        history_paths=[str(histories[1])],
    )
    finest = _write_aggregate(
        tmp_path / "temporal_2p5e-7.npz",
        (64,),
        dt=2.5e-7,
        integration_scale=1.0,
        diagnostic_path=diagnostic,
        direct_metadata=(2.5e-7, 0.0, 2.0e-5),
        num_steps=80,
        history_paths=[str(histories[2])],
    )
    return frozen, baseline, fine, finest


def test_require_complete_uses_history_self_convergence_not_total_continuum_error(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert report["ok"]
    assert gate["status"] == "pass"
    assert gate["total_continuum_error_is_not_temporal_error"] is True
    assert np.allclose(
        [item["order"] for item in gate["temporal_self_convergence_by_field"]],
        2.0,
        atol=1.0e-10,
    )
    total = {check["name"]: check for check in report["checks"]}[
        "continuum_total_error_by_timestep"
    ]
    assert total["status"] == "diagnostic"
    assert analyzer.main([*(str(path) for path in artifacts), "--require-complete"]) == 0


def test_require_complete_does_not_require_normalized_high_mode_fraction(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    diagnostic_path = tmp_path / "short_leg.npz"
    with np.load(diagnostic_path, allow_pickle=False) as source:
        payload = {
            key: np.asarray(source[key])
            for key in source.files
            if key != "high_mode_fraction"
        }
    np.savez(diagnostic_path, **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert report["ok"]
    assert gate["status"] == "pass"


def test_require_complete_hard_fails_subsecond_order_and_missing_temporal_diagnostic(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path, temporal_power=1.0)
    with np.load(artifacts[2], allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["short_leg_diagnostics_paths_json"] = np.asarray(json.dumps([None]))
    np.savez(artifacts[2], **payload)
    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert gate["status"] == "fail"
    assert any("below 1.8" in failure for failure in gate["failures"])
    assert any("unresolved short-leg" in failure for failure in gate["failures"])


def test_require_complete_rejects_cross_history_owner_measure_shape_mismatch(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    history_paths = json.loads(
        np.load(artifacts[2], allow_pickle=False)["history_paths_json"].item()
    )
    history_path = Path(history_paths[0])
    with np.load(history_path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["owner_active"] = np.ones((2, 2, 3), dtype=bool)
    payload["owner_aggregate_volume"] = np.ones((2, 2, 3), dtype=np.float64)
    for field in FIELDS:
        payload[field] = np.ones((2, 2, 2, 3), dtype=np.float64)
    np.savez(history_path, **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert any("owner measure shape differs" in failure for failure in gate["failures"])


def test_require_complete_reports_field_metadata_mismatch_without_keyerror(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    with np.load(artifacts[2], allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["field_names_json"] = np.asarray(json.dumps(list(FIELDS[:-1])))
    np.savez(artifacts[2], **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert any("field-name tuples" in failure for failure in gate["failures"])


def test_require_complete_rejects_nonpositive_active_owner_volume(tmp_path: Path):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    history_paths = json.loads(
        np.load(artifacts[2], allow_pickle=False)["history_paths_json"].item()
    )
    history_path = Path(history_paths[0])
    with np.load(history_path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["owner_aggregate_volume"] = np.array(
        payload["owner_aggregate_volume"], copy=True
    )
    payload["owner_aggregate_volume"][0, 0, 0] = 0.0
    np.savez(history_path, **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert any("aggregate-volume measure is invalid" in failure for failure in gate["failures"])


def test_require_complete_rejects_nonproduction_temporal_execution_contract(
    tmp_path: Path,
):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    with np.load(artifacts[2], allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    # Fractional values must not be truncated to the canonical integer tuple.
    payload["shard_counts"] = np.asarray((1.9, 1.0, 4.0), dtype=np.float64)
    payload["device_count"] = np.asarray(1, dtype=np.int32)
    payload["evolved_execution"] = np.asarray("host-single-device")
    np.savez(artifacts[2], **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert any("shard_counts" in failure for failure in gate["failures"])
    assert any("device_count" in failure for failure in gate["failures"])
    assert any("evolved_execution" in failure for failure in gate["failures"])


def test_require_complete_hard_gates_evolved_spatial_convergence(tmp_path: Path):
    analyzer = _load()
    artifacts = _write_complete_stage7_artifacts(tmp_path)
    with np.load(artifacts[1], allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload["integration_error_by_field"] = np.asarray((32.0, 48.0, 64.0))[:, None] \
        * np.ones((3, len(FIELDS)))
    payload["integration_error"] = np.mean(
        payload["integration_error_by_field"], axis=1
    )
    np.savez(artifacts[1], **payload)

    report = analyzer.analyze(artifacts, require_complete=True)
    gate = {check["name"]: check for check in report["checks"]}[
        "complete_stage7_acceptance"
    ]
    assert not report["ok"]
    assert any("evolved 32/48/64 spatial" in failure for failure in gate["failures"])
    assert gate["evolved_spatial_convergence_gate"]["status"] == "fail"
