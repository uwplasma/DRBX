from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def _load_compare_script():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_recycling_transient_modes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_recycling_transient_modes", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_script = _load_compare_script()


def test_parser_accepts_and_documents_fixed_full_field_jvp_mode() -> None:
    args = compare_script._parse_args(
        [
            "--reference-root",
            "/tmp/reference",
            "--mode",
            "bdf_fixed_full_field_jvp",
            "--mode",
            "bdf_active_array_jvp",
            "--mode",
            "adaptive_bdf_jax_linearized",
            "--mode",
            "fixed_bdf2_jax_linearized",
            "--mode",
            "fixed_bdf2_active_array_jax_linearized",
            "--require-bdf-pairwise-max",
            "1e-5",
            "--require-fixed-jvp-diagnostics",
            "--require-fixed-bdf2-diagnostics",
            "--require-fixed-bdf2-linear-preconditioner",
            "local-block",
            "--require-fixed-bdf2-linear-solver-backend",
            "bicgstab",
            "--require-fixed-bdf2-linear-operator-jitted",
            "--require-fixed-bdf2-line-search-mode",
            "full",
            "--require-fixed-bdf2-max-linear-iterations",
            "3600",
            "--require-fixed-bdf2-max-linear-update-residual",
            "1e-8",
            "--require-fixed-bdf2-max-linear-update-relative-residual",
            "1e-4",
            "--require-fixed-bdf2-max-preconditioner-builds",
            "2",
            "--require-fixed-bdf2-max-preconditioner-applies",
            "40",
            "--require-adaptive-bdf-no-fallback",
            "--require-adaptive-bdf-no-unconverged-substeps",
            "--require-adaptive-bdf-linear-preconditioner",
            "parallel-line",
            "--require-adaptive-bdf-max-error-ratio",
            "0.95",
            "--require-adaptive-bdf-max-accepted-error-ratio",
            "0.75",
            "--require-adaptive-bdf-max-linear-update-residual",
            "1e-8",
            "--require-adaptive-bdf-max-linear-update-relative-residual",
            "1e-4",
            "--mode-timeout-seconds",
            "2.5",
            "--override",
            "solver:rtol=1e-9",
            "--timestep",
            "0.05",
            "--max-nonlinear-iterations",
            "3",
            "--steps",
            "2",
            "--output-json",
            "/tmp/report.json",
            "--diagnostics-only",
        ]
    )

    assert args.reference_root == Path("/tmp/reference")
    assert args.modes == [
        "bdf_fixed_full_field_jvp",
        "bdf_active_array_jvp",
        "adaptive_bdf_jax_linearized",
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized",
    ]
    assert args.require_bdf_pairwise_max == 1.0e-5
    assert args.require_fixed_jvp_diagnostics is True
    assert args.require_fixed_bdf2_diagnostics is True
    assert args.require_fixed_bdf2_linear_preconditioner == "local-block"
    assert args.require_fixed_bdf2_linear_solver_backend == "bicgstab"
    assert args.require_fixed_bdf2_linear_operator_jitted is True
    assert args.require_fixed_bdf2_line_search_mode == "full"
    assert args.require_fixed_bdf2_max_linear_iterations == 3600
    assert args.require_fixed_bdf2_max_linear_update_residual == 1.0e-8
    assert args.require_fixed_bdf2_max_linear_update_relative_residual == 1.0e-4
    assert args.require_fixed_bdf2_max_preconditioner_builds == 2
    assert args.require_fixed_bdf2_max_preconditioner_applies == 40
    assert args.require_adaptive_bdf_no_fallback is True
    assert args.require_adaptive_bdf_no_unconverged_substeps is True
    assert args.require_adaptive_bdf_linear_preconditioner == "parallel-line"
    assert args.require_adaptive_bdf_max_error_ratio == 0.95
    assert args.require_adaptive_bdf_max_accepted_error_ratio == 0.75
    assert args.require_adaptive_bdf_max_linear_update_residual == 1.0e-8
    assert args.require_adaptive_bdf_max_linear_update_relative_residual == 1.0e-4
    assert args.mode_timeout_seconds == 2.5
    assert args.overrides == ["solver:rtol=1e-9"]
    assert args.timestep == 0.05
    assert args.max_nonlinear_iterations == 3
    assert args.steps == 2
    assert args.output_json == Path("/tmp/report.json")
    assert args.diagnostics_only is True
    help_text = compare_script._build_parser().format_help()
    normalized_help = " ".join(help_text.split()).replace("full- field", "full-field")
    assert "bdf_fixed_full_field_jvp" in help_text
    assert "bdf_active_array_jvp" in help_text
    assert "adaptive_bdf_jax_linearized" in help_text
    assert "fixed_bdf2_jax_linearized" in help_text
    assert "fixed_bdf2_active_array_jax_linearized" in help_text
    assert "--require-fixed-bdf2-diagnostics" in help_text
    assert "--require-fixed-bdf2-linear-preconditioner" in help_text
    assert "--require-fixed-bdf2-linear-solver-backend" in help_text
    assert "--require-fixed-bdf2-linear-operator-jitted" in help_text
    assert "--require-fixed-bdf2-line-search-mode" in help_text
    assert "--require-fixed-bdf2-max-linear-iterations" in help_text
    assert "--require-fixed-bdf2-max-linear-update-residual" in help_text
    assert "--require-fixed-bdf2-max-linear-update-relative-residual" in help_text
    assert "--require-fixed-bdf2-max-preconditioner-builds" in help_text
    assert "--require-fixed-bdf2-max-preconditioner-applies" in help_text
    assert "--require-adaptive-bdf-linear-preconditioner" in help_text
    assert "--require-adaptive-bdf-max-linear-update-residual" in help_text
    assert "--require-adaptive-bdf-max-linear-update-relative-residual" in help_text
    assert "fixed-layout JVP BDF paths" in normalized_help


def test_resolve_output_timestep_uses_configured_value_by_default() -> None:
    args = SimpleNamespace(timestep=None)
    run_config = SimpleNamespace(time=SimpleNamespace(timestep=2.5))

    assert compare_script._resolve_output_timestep(args, run_config) == 2.5


def test_resolve_output_timestep_accepts_positive_override() -> None:
    args = SimpleNamespace(timestep=0.05)
    run_config = SimpleNamespace(time=SimpleNamespace(timestep=5000.0))

    assert compare_script._resolve_output_timestep(args, run_config) == 0.05


def test_resolve_output_timestep_rejects_nonpositive_override() -> None:
    args = SimpleNamespace(timestep=0.0)
    run_config = SimpleNamespace(time=SimpleNamespace(timestep=5000.0))

    try:
        compare_script._resolve_output_timestep(args, run_config)
    except ValueError as exc:
        assert "--timestep must be a positive finite value" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_resolve_max_nonlinear_iterations_accepts_positive_value() -> None:
    assert (
        compare_script._resolve_max_nonlinear_iterations(
            SimpleNamespace(max_nonlinear_iterations=4)
        )
        == 4
    )


def test_resolve_max_nonlinear_iterations_rejects_nonpositive_value() -> None:
    try:
        compare_script._resolve_max_nonlinear_iterations(
            SimpleNamespace(max_nonlinear_iterations=0)
        )
    except ValueError as exc:
        assert "--max-nonlinear-iterations must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_resolve_steps_accepts_positive_value() -> None:
    assert compare_script._resolve_steps(SimpleNamespace(steps=2)) == 2


def test_resolve_steps_rejects_nonpositive_value() -> None:
    try:
        compare_script._resolve_steps(SimpleNamespace(steps=0))
    except ValueError as exc:
        assert "--steps must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_default_modes_include_fixed_full_field_jvp_after_bdf() -> None:
    one_step_modes = compare_script._default_modes("recycling_1d_one_step")
    dthe_modes = compare_script._default_modes("recycling_dthe_one_step")

    assert one_step_modes == (
        "continuation",
        "bdf",
        "bdf_fixed_full_field_jvp",
        "bdf_active_array_jvp",
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized",
        "adaptive_be",
        "adaptive_bdf",
    )
    assert dthe_modes == (
        "bdf",
        "bdf_fixed_full_field_jvp",
        "bdf_active_array_jvp",
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized",
        "adaptive_be",
        "adaptive_bdf",
    )


def test_bdf_pairwise_delta_report_formats_worst_field_first() -> None:
    mode_variables = {
        "bdf": {
            "Nd+": np.asarray([1.0, 2.0]),
            "Pd+": np.asarray([10.0, 15.0]),
        },
        "bdf_fixed_full_field_jvp": {
            "Nd+": np.asarray([1.25, 2.5]),
            "Pd+": np.asarray([13.0, 15.0]),
        },
        "bdf_active_array_jvp": {
            "Nd+": np.asarray([1.0, 2.125]),
            "Pd+": np.asarray([10.5, 14.75]),
        },
    }

    lines = compare_script._format_bdf_pairwise_delta_report(
        mode_variables,
        fields=("Nd+", "Pd+"),
    )

    assert lines == [
        "pairwise_delta=bdf_vs_bdf_fixed_full_field_jvp",
        "  Pd+: max_abs_delta=3.00000000e+00",
        "  Nd+: max_abs_delta=5.00000000e-01",
        "  worst=Pd+ delta=3.00000000e+00",
        "pairwise_delta=bdf_vs_bdf_active_array_jvp",
        "  Pd+: max_abs_delta=5.00000000e-01",
        "  Nd+: max_abs_delta=1.25000000e-01",
        "  worst=Pd+ delta=5.00000000e-01",
    ]


def test_mode_diagnostics_report_formats_sorted_values() -> None:
    lines = compare_script._format_mode_diagnostics_report(
        "bdf_fixed_full_field_jvp",
        {
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jacobian_callback_seconds": 0.125,
            "bdf_jacobian_base_rhs_evaluation_count": 0,
        },
    )

    assert lines == [
        "diagnostics mode=bdf_fixed_full_field_jvp",
        "  bdf_jacobian_base_rhs_evaluation_count=0",
        "  bdf_jacobian_callback_seconds=1.25000000e-01",
        "  bdf_rhs_backend=fixed_full_field_array",
    ]


def test_bdf_pairwise_delta_report_crops_both_outputs_to_active_mesh() -> None:
    mesh = SimpleNamespace(xstart=1, xend=2, ystart=0, yend=1)
    bdf = np.zeros((1, 4, 3, 1))
    fixed_jvp = np.zeros((1, 4, 3, 1))
    fixed_jvp[:, 0, 0, :] = 99.0
    fixed_jvp[:, 2, 1, :] = 0.75

    lines = compare_script._format_bdf_pairwise_delta_report(
        {
            "bdf": {"Nd+": bdf},
            "bdf_fixed_full_field_jvp": {"Nd+": fixed_jvp},
            "bdf_active_array_jvp": {"Nd+": fixed_jvp * 0.5},
        },
        fields=("Nd+",),
        mesh=mesh,
    )

    assert lines == [
        "pairwise_delta=bdf_vs_bdf_fixed_full_field_jvp",
        "  Nd+: max_abs_delta=7.50000000e-01",
        "  worst=Nd+ delta=7.50000000e-01",
        "pairwise_delta=bdf_vs_bdf_active_array_jvp",
        "  Nd+: max_abs_delta=3.75000000e-01",
        "  worst=Nd+ delta=3.75000000e-01",
    ]


def test_bdf_pairwise_delta_report_is_omitted_without_both_modes() -> None:
    lines = compare_script._format_bdf_pairwise_delta_report(
        {"bdf": {"Nd+": np.asarray([1.0])}},
        fields=("Nd+",),
    )

    assert lines == []


def test_bdf_pairwise_worst_delta_returns_active_mesh_worst_field() -> None:
    mesh = SimpleNamespace(xstart=1, xend=2, ystart=0, yend=1)
    bdf = np.zeros((1, 4, 3, 1))
    fixed_jvp = np.zeros((1, 4, 3, 1))
    fixed_jvp[:, 0, 0, :] = 99.0
    fixed_jvp[:, 1, 1, :] = 0.25
    fixed_jvp[:, 2, 1, :] = 0.75

    field, delta = compare_script._bdf_pairwise_worst_delta(
        {
            "bdf": {"Nd+": bdf},
            "bdf_fixed_full_field_jvp": {"Nd+": fixed_jvp},
            "bdf_active_array_jvp": {"Nd+": fixed_jvp * 2.0},
        },
        fields=("Nd+",),
        mesh=mesh,
    )

    assert field == "Nd+"
    assert delta == 1.5


def test_json_report_writer_preserves_diagnostics_and_sanitizes_paths(
    tmp_path: Path,
) -> None:
    report = compare_script._build_json_report(
        case_name="recycling_1d_one_step",
        configured_timestep=1.0,
        timestep=0.5,
        max_nonlinear_iterations=3,
        steps=2,
        fields=("Pe",),
        modes=("adaptive_bdf_jax_linearized",),
        diagnostics_only=True,
        mode_elapsed_seconds={"adaptive_bdf_jax_linearized": 1.25},
        mode_diagnostics={
            "adaptive_bdf_jax_linearized": {
                "adaptive_bdf_accepted_steps": np.int64(2),
                "linear_solver_status": Path("/tmp/private/status"),
            }
        },
        bdf_pairwise_worst=("Pe", np.float64(0.0)),
        adaptive_bdf_gate_errors={"adaptive_bdf_jax_linearized": []},
    )
    path = tmp_path / "nested" / "report.json"
    compare_script._write_json_report(path, report)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case"] == "recycling_1d_one_step"
    assert payload["steps"] == 2
    assert (
        payload["mode_diagnostics"]["adaptive_bdf_jax_linearized"][
            "adaptive_bdf_accepted_steps"
        ]
        == 2
    )
    assert (
        payload["mode_diagnostics"]["adaptive_bdf_jax_linearized"][
            "linear_solver_status"
        ]
        == "status"
    )
    assert payload["bdf_pairwise_worst"] == {"field": "Pe", "delta": 0.0}


def test_fixed_full_field_jvp_diagnostics_gate_accepts_expected_route() -> None:
    errors = compare_script._validate_fixed_full_field_jvp_diagnostics(
        {
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jacobian_mode": "jvp",
            "bdf_jacobian_base_rhs_evaluation_count": 0,
            "bdf_jvp_rhs_evaluation_count": 3,
            "bdf_jvp_jacobian_prebuilt_direction_batch_uses": 1,
        }
    )

    assert errors == []


def test_active_array_jvp_diagnostics_gate_accepts_expected_route() -> None:
    errors = compare_script._validate_bdf_jvp_diagnostics(
        "bdf_active_array_jvp",
        {
            "bdf_rhs_backend": "active_array",
            "bdf_jacobian_mode": "jvp",
            "bdf_jacobian_base_rhs_evaluation_count": 0,
            "bdf_jvp_rhs_evaluation_count": 3,
            "bdf_jvp_jacobian_prebuilt_direction_batch_uses": 1,
        },
    )

    assert errors == []


def test_fixed_full_field_jvp_diagnostics_gate_reports_wrong_route() -> None:
    errors = compare_script._validate_fixed_full_field_jvp_diagnostics(
        {
            "bdf_rhs_backend": "host_bridge",
            "bdf_jacobian_mode": "fd",
            "bdf_jacobian_base_rhs_evaluation_count": 2,
            "bdf_jvp_rhs_evaluation_count": 0,
            "bdf_jvp_jacobian_prebuilt_direction_batch_uses": 0,
        }
    )

    assert errors == [
        "bdf_fixed_full_field_jvp did not report bdf_rhs_backend=fixed_full_field_array",
        "bdf_fixed_full_field_jvp did not report bdf_jacobian_mode=jvp",
        "bdf_fixed_full_field_jvp reported finite-difference base RHS Jacobian evaluations",
        "bdf_fixed_full_field_jvp did not report any JVP RHS evaluations",
        "bdf_fixed_full_field_jvp did not report prebuilt JVP direction-batch reuse",
    ]


def test_active_array_jvp_diagnostics_gate_reports_wrong_route() -> None:
    errors = compare_script._validate_bdf_jvp_diagnostics(
        "bdf_active_array_jvp",
        {
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jacobian_mode": "fd",
            "bdf_jacobian_base_rhs_evaluation_count": 2,
            "bdf_jvp_rhs_evaluation_count": 0,
            "bdf_jvp_jacobian_prebuilt_direction_batch_uses": 0,
        },
    )

    assert errors == [
        "bdf_active_array_jvp did not report bdf_rhs_backend=active_array",
        "bdf_active_array_jvp did not report bdf_jacobian_mode=jvp",
        "bdf_active_array_jvp reported finite-difference base RHS Jacobian evaluations",
        "bdf_active_array_jvp did not report any JVP RHS evaluations",
        "bdf_active_array_jvp did not report prebuilt JVP direction-batch reuse",
    ]


def test_fixed_bdf2_modes_to_validate_selects_only_fixed_bdf2_variants() -> None:
    modes = compare_script._fixed_bdf2_modes_to_validate(
        (
            "bdf",
            "fixed_bdf2_jax_linearized",
            "adaptive_bdf",
            "fixed_bdf2_jax_linearized_lineax",
            "fixed_bdf2_active_array_jax_linearized",
        )
    )

    assert modes == (
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_jax_linearized_lineax",
        "fixed_bdf2_active_array_jax_linearized",
    )


def test_fixed_bdf2_diagnostics_gate_accepts_jax_linearized_route() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_jax_linearized",
            "fixed_bdf2_step_solver_mode": "jax_linearized",
            "fixed_bdf2_fixed_full_field_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
        },
    )

    assert errors == []


def test_fixed_bdf2_diagnostics_gate_accepts_lineax_route() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_jax_linearized_lineax",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_jax_linearized_lineax",
            "fixed_bdf2_step_solver_mode": "jax_linearized_lineax",
            "fixed_bdf2_fixed_full_field_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_lineax_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
        },
    )

    assert errors == []


def test_fixed_bdf2_diagnostics_gate_accepts_active_array_route() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_linear_preconditioner": "local_block_diag",
            "fixed_bdf2_total_linear_preconditioner_build_count": 2,
            "fixed_bdf2_total_linear_preconditioner_build_seconds": 0.125,
            "fixed_bdf2_linear_solver_backend": "jax_gmres",
            "fixed_bdf2_linear_operator_jitted_steps": 2,
            "fixed_bdf2_line_search_mode": "full_step",
            "fixed_bdf2_total_linear_iterations": 3200,
            "fixed_bdf2_total_linear_operator_call_count": 96,
        },
        required_linear_preconditioner="local-block-diag",
        required_linear_solver_backend="gmres",
        require_linear_operator_jitted=True,
        required_line_search_mode="full",
        max_linear_iterations=3600,
        max_linear_operator_calls=128,
        max_preconditioner_builds=2,
    )

    assert errors == []


def test_fixed_bdf2_diagnostics_gate_reports_wrong_linear_backend() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_linear_solver_backend": "jax_gmres",
        },
        required_linear_solver_backend="bicgstab",
    )

    assert errors == [
        (
            "fixed_bdf2_active_array_jax_linearized did not report "
            "fixed_bdf2_linear_solver_backend=jax_bicgstab; reported jax_gmres"
        )
    ]


def test_fixed_bdf2_diagnostics_gate_reports_missing_jitted_linear_operator() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_linear_operator_jitted_steps": 1,
        },
        require_linear_operator_jitted=True,
    )

    assert errors == [
        (
            "fixed_bdf2_active_array_jax_linearized did not report "
            "JIT-wrapped linear operators on every JAX-linearized step: 1/2"
        )
    ]


def test_fixed_bdf2_diagnostics_gate_reports_wrong_line_search_mode() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_line_search_mode": "backtracking",
        },
        required_line_search_mode="full_step",
    )

    assert errors == [
        (
            "fixed_bdf2_active_array_jax_linearized did not report "
            "fixed_bdf2_line_search_mode=full_step; reported backtracking"
        )
    ]


def test_fixed_bdf2_diagnostics_gate_accepts_active_array_lineax_route() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized_lineax",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized_lineax",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized_lineax",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_lineax_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
        },
    )

    assert errors == []


def test_fixed_bdf2_diagnostics_gate_rejects_unhealthy_solver_status() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_unconverged_solver_steps": 1,
            "fixed_bdf2_unknown_convergence_solver_steps": 1,
            "fixed_bdf2_linear_solver_failed_steps": 1,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-2,
        },
        max_residual_inf_norm=1.0e-5,
    )

    assert errors == [
        "fixed_bdf2_active_array_jax_linearized reported 1 unconverged fixed BDF2 implicit steps",
        "fixed_bdf2_active_array_jax_linearized reported 1 unknown-convergence fixed BDF2 implicit steps",
        "fixed_bdf2_active_array_jax_linearized reported 1 failed fixed BDF2 linear solves",
        "fixed_bdf2_active_array_jax_linearized fixed_bdf2_max_residual_inf_norm=1.00000000e-02 exceeds 1.00000000e-05",
    ]


def test_fixed_bdf2_diagnostics_gate_reports_fallback_route() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_jax_linearized_lineax",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_jax_linearized",
            "fixed_bdf2_step_solver_mode": "sparse",
            "fixed_bdf2_fixed_full_field_rhs_steps": 0,
            "fixed_bdf2_jax_linearized_action_steps": 0,
            "fixed_bdf2_lineax_action_steps": 0,
            "fixed_bdf2_startup_steps": 0,
            "fixed_bdf2_bdf2_steps": 0,
            "fixed_bdf2_evolve_feedback_integrals": False,
            "fixed_bdf2_max_residual_inf_norm": "nan",
        },
    )

    assert errors == [
        "fixed_bdf2_jax_linearized_lineax did not report fixed_bdf2_solver_mode=fixed_bdf2_jax_linearized_lineax",
        "fixed_bdf2_jax_linearized_lineax did not report fixed_bdf2_step_solver_mode=jax_linearized_lineax",
        "fixed_bdf2_jax_linearized_lineax did not report any fixed_full_field_array fixed-layout RHS steps",
        "fixed_bdf2_jax_linearized_lineax did not report any JAX-linearized solver steps",
        "fixed_bdf2_jax_linearized_lineax did not report any Lineax solver steps",
        "fixed_bdf2_jax_linearized_lineax did not evolve packed feedback integrals",
        "fixed_bdf2_jax_linearized_lineax did not report any accepted fixed BDF2 intervals",
        "fixed_bdf2_jax_linearized_lineax did not report any actual fixed BDF2 corrector steps",
        "fixed_bdf2_jax_linearized_lineax did not report a finite fixed BDF2 residual norm",
    ]


def test_fixed_bdf2_diagnostics_gate_reports_missing_preconditioner() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_linear_preconditioner": None,
            "fixed_bdf2_total_linear_preconditioner_build_count": 0,
            "fixed_bdf2_total_linear_preconditioner_build_seconds": float("nan"),
        },
        required_linear_preconditioner="parallel-line",
    )

    assert errors == [
        (
            "fixed_bdf2_active_array_jax_linearized did not report "
            "fixed_bdf2_linear_preconditioner=parallel_line"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized did not report any "
            "parallel_line preconditioner builds"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized did not report finite "
            "nonnegative fixed_bdf2_total_linear_preconditioner_build_seconds"
        ),
    ]


def test_fixed_bdf2_diagnostics_gate_accepts_sheath_line_alias() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_linear_preconditioner": "sheath_line",
            "fixed_bdf2_total_linear_preconditioner_build_count": 1,
            "fixed_bdf2_total_linear_preconditioner_build_seconds": 0.0,
        },
        required_linear_preconditioner="target-sheath",
    )

    assert errors == []


def test_fixed_bdf2_diagnostics_gate_reports_performance_budget_failures() -> None:
    errors = compare_script._validate_fixed_bdf2_diagnostics(
        "fixed_bdf2_active_array_jax_linearized",
        {
            "fixed_bdf2_solver_mode": "fixed_bdf2_active_array_jax_linearized",
            "fixed_bdf2_step_solver_mode": "active_array_jax_linearized",
            "fixed_bdf2_active_array_rhs_steps": 2,
            "fixed_bdf2_jax_linearized_action_steps": 2,
            "fixed_bdf2_startup_steps": 1,
            "fixed_bdf2_bdf2_steps": 1,
            "fixed_bdf2_evolve_feedback_integrals": True,
            "fixed_bdf2_max_residual_inf_norm": 1.0e-11,
            "fixed_bdf2_total_linear_iterations": 3600,
            "fixed_bdf2_total_linear_operator_call_count": 512,
            "fixed_bdf2_max_linear_update_residual_inf_norm": 2.0e-8,
            "fixed_bdf2_max_linear_update_relative_residual": 2.0e-4,
            "fixed_bdf2_total_linear_preconditioner_build_count": 9,
            "fixed_bdf2_total_linear_preconditioner_apply_count": 35,
        },
        max_linear_iterations=3200,
        max_linear_operator_calls=128,
        max_linear_update_residual_inf_norm=1.0e-8,
        max_linear_update_relative_residual=1.0e-4,
        max_preconditioner_builds=2,
        max_preconditioner_applies=30,
    )

    assert errors == [
        (
            "fixed_bdf2_active_array_jax_linearized reported 3600 fixed BDF2 "
            "linear iterations, exceeding 3200"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized reported 512 fixed BDF2 "
            "linear operator calls, exceeding 128"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized reported 2.00000000e-08 "
            "fixed BDF2 linear-update residual inf-norm, exceeding 1.00000000e-08"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized reported 2.00000000e-04 "
            "fixed BDF2 linear-update relative residual, exceeding 1.00000000e-04"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized reported 9 fixed BDF2 "
            "preconditioner builds, exceeding 2"
        ),
        (
            "fixed_bdf2_active_array_jax_linearized reported 35 fixed BDF2 "
            "preconditioner applies, exceeding 30"
        ),
    ]


def test_adaptive_bdf_modes_to_validate_selects_only_adaptive_bdf_variants() -> None:
    modes = compare_script._adaptive_bdf_modes_to_validate(
        (
            "bdf",
            "adaptive_be",
            "adaptive_bdf",
            "adaptive_bdf_jax_linearized",
            "adaptive_bdf_active_array_jax_linearized",
            "bdf_fixed_full_field_jvp",
            "bdf_active_array_jvp",
        )
    )

    assert modes == (
        "adaptive_bdf",
        "adaptive_bdf_jax_linearized",
        "adaptive_bdf_active_array_jax_linearized",
    )


def test_adaptive_bdf_diagnostics_gate_accepts_stable_jax_linearized_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_jax_linearized",
        {
            "adaptive_bdf_step_solver_mode": "jax_linearized",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_fixed_full_field_rhs_solver_steps": 3,
            "adaptive_bdf_jax_linearized_action_solver_steps": 3,
            "adaptive_bdf_max_error_ratio": 0.75,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
            "adaptive_bdf_max_linear_update_residual_inf_norm": 2.0e-9,
            "adaptive_bdf_max_linear_update_relative_residual": 5.0e-5,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
        max_linear_update_residual_inf_norm=1.0e-8,
        max_linear_update_relative_residual=1.0e-4,
    )

    assert errors == []


def test_adaptive_bdf_diagnostics_gate_accepts_sparse_jvp_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_sparse_jvp",
        {
            "adaptive_bdf_step_solver_mode": "sparse_jvp",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_fixed_full_field_rhs_solver_steps": 3,
            "adaptive_bdf_sparse_jvp_jacobian_solver_steps": 3,
            "adaptive_bdf_max_error_ratio": 0.75,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
    )

    assert errors == []


def test_adaptive_bdf_diagnostics_gate_accepts_active_array_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_active_array_jax_linearized",
        {
            "adaptive_bdf_step_solver_mode": "active_array_jax_linearized",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_active_array_rhs_solver_steps": 3,
            "adaptive_bdf_jax_linearized_action_solver_steps": 3,
            "adaptive_bdf_max_error_ratio": 0.75,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
            "adaptive_bdf_linear_preconditioner": "parallel_line",
            "adaptive_bdf_linear_preconditioner_build_count": 3,
            "adaptive_bdf_linear_preconditioner_build_seconds": 0.25,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
        required_linear_preconditioner="parallel-line",
    )

    assert errors == []


def test_adaptive_bdf_diagnostics_gate_accepts_active_array_lineax_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_active_array_jax_linearized_lineax",
        {
            "adaptive_bdf_step_solver_mode": "active_array_jax_linearized_lineax",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_active_array_rhs_solver_steps": 3,
            "adaptive_bdf_jax_linearized_action_solver_steps": 3,
            "adaptive_bdf_lineax_action_solver_steps": 3,
            "adaptive_bdf_max_error_ratio": 0.75,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
    )

    assert errors == []


def test_adaptive_bdf_diagnostics_gate_reports_failed_linear_solves() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_jax_linearized_lineax",
        {
            "adaptive_bdf_step_solver_mode": "jax_linearized_lineax",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_linear_solver_failed_steps": 2,
            "adaptive_bdf_fixed_full_field_rhs_solver_steps": 3,
            "adaptive_bdf_jax_linearized_action_solver_steps": 3,
            "adaptive_bdf_lineax_action_solver_steps": 3,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=None,
        max_accepted_error_ratio=0.95,
    )

    assert errors == [
        "adaptive_bdf_jax_linearized_lineax reported 2 failed adaptive BDF linear solves"
    ]


def test_adaptive_bdf_diagnostics_gate_reports_unstable_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_jax_linearized",
        {
            "adaptive_bdf_step_solver_mode": "sparse",
            "adaptive_bdf_interval_count": 0,
            "adaptive_bdf_accepted_steps": 0,
            "adaptive_bdf_minimum_dt_fallbacks": 2,
            "adaptive_bdf_unconverged_solver_steps": 3,
            "adaptive_bdf_max_error_ratio": 1.25,
            "adaptive_bdf_max_accepted_error_ratio": 1.1,
            "adaptive_bdf_max_linear_update_residual_inf_norm": 2.0e-8,
            "adaptive_bdf_max_linear_update_relative_residual": 2.0e-4,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
        max_linear_update_residual_inf_norm=1.0e-8,
        max_linear_update_relative_residual=1.0e-4,
    )

    assert errors == [
        "adaptive_bdf_jax_linearized did not report adaptive_bdf_step_solver_mode=jax_linearized",
        "adaptive_bdf_jax_linearized did not report any adaptive BDF output intervals",
        "adaptive_bdf_jax_linearized did not report any accepted adaptive BDF substeps",
        "adaptive_bdf_jax_linearized did not report any fixed_full_field_array adaptive BDF solver steps",
        "adaptive_bdf_jax_linearized did not report any JAX-linearized adaptive BDF solver steps",
        "adaptive_bdf_jax_linearized reported 2 minimum-dt fallback accepts",
        "adaptive_bdf_jax_linearized reported 3 unconverged adaptive BDF implicit substeps",
        "adaptive_bdf_jax_linearized adaptive_bdf_max_error_ratio=1.25000000e+00 exceeds 9.50000000e-01",
        "adaptive_bdf_jax_linearized adaptive_bdf_max_accepted_error_ratio=1.10000000e+00 exceeds 9.50000000e-01",
        (
            "adaptive_bdf_jax_linearized reported 2.00000000e-08 adaptive BDF "
            "linear-update residual inf-norm, exceeding 1.00000000e-08"
        ),
        (
            "adaptive_bdf_jax_linearized reported 2.00000000e-04 adaptive BDF "
            "linear-update relative residual, exceeding 1.00000000e-04"
        ),
    ]


def test_mode_timeout_helper_returns_callback_value() -> None:
    assert compare_script._run_with_mode_timeout(1.0, lambda: "ok") == "ok"


def test_mode_timeout_helper_rejects_nonpositive_timeout() -> None:
    try:
        compare_script._run_with_mode_timeout(0.0, lambda: None)
    except ValueError as exc:
        assert "--mode-timeout-seconds must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_mode_timeout_helper_raises_timeout_when_supported() -> None:
    if not hasattr(compare_script.signal, "SIGALRM") or not hasattr(
        compare_script.signal, "setitimer"
    ):
        return
    started = time.perf_counter()
    try:
        compare_script._run_with_mode_timeout(0.01, lambda: time.sleep(1.0))
    except compare_script._ModeTimeoutError as exc:
        assert "exceeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected _ModeTimeoutError")
    assert time.perf_counter() - started < 0.5
