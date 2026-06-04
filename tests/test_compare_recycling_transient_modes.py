from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def _load_compare_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_recycling_transient_modes.py"
    spec = importlib.util.spec_from_file_location("compare_recycling_transient_modes", script_path)
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
            "adaptive_bdf_jax_linearized",
            "--require-bdf-pairwise-max",
            "1e-5",
            "--require-fixed-jvp-diagnostics",
            "--require-adaptive-bdf-no-fallback",
            "--require-adaptive-bdf-no-unconverged-substeps",
            "--require-adaptive-bdf-max-error-ratio",
            "0.95",
            "--require-adaptive-bdf-max-accepted-error-ratio",
            "0.75",
            "--mode-timeout-seconds",
            "2.5",
            "--override",
            "solver:rtol=1e-9",
            "--timestep",
            "0.05",
            "--max-nonlinear-iterations",
            "3",
            "--diagnostics-only",
        ]
    )

    assert args.reference_root == Path("/tmp/reference")
    assert args.modes == ["bdf_fixed_full_field_jvp", "adaptive_bdf_jax_linearized"]
    assert args.require_bdf_pairwise_max == 1.0e-5
    assert args.require_fixed_jvp_diagnostics is True
    assert args.require_adaptive_bdf_no_fallback is True
    assert args.require_adaptive_bdf_no_unconverged_substeps is True
    assert args.require_adaptive_bdf_max_error_ratio == 0.95
    assert args.require_adaptive_bdf_max_accepted_error_ratio == 0.75
    assert args.mode_timeout_seconds == 2.5
    assert args.overrides == ["solver:rtol=1e-9"]
    assert args.timestep == 0.05
    assert args.max_nonlinear_iterations == 3
    assert args.diagnostics_only is True
    help_text = compare_script._build_parser().format_help()
    normalized_help = " ".join(help_text.split()).replace("full- field", "full-field")
    assert "bdf_fixed_full_field_jvp" in help_text
    assert "adaptive_bdf_jax_linearized" in help_text
    assert "fixed full-field JVP BDF path" in normalized_help


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
    assert compare_script._resolve_max_nonlinear_iterations(SimpleNamespace(max_nonlinear_iterations=4)) == 4


def test_resolve_max_nonlinear_iterations_rejects_nonpositive_value() -> None:
    try:
        compare_script._resolve_max_nonlinear_iterations(SimpleNamespace(max_nonlinear_iterations=0))
    except ValueError as exc:
        assert "--max-nonlinear-iterations must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_default_modes_include_fixed_full_field_jvp_after_bdf() -> None:
    one_step_modes = compare_script._default_modes("recycling_1d_one_step")
    dthe_modes = compare_script._default_modes("recycling_dthe_one_step")

    assert one_step_modes == (
        "continuation",
        "bdf",
        "bdf_fixed_full_field_jvp",
        "adaptive_be",
        "adaptive_bdf",
    )
    assert dthe_modes == ("bdf", "bdf_fixed_full_field_jvp", "adaptive_be", "adaptive_bdf")


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
        },
        fields=("Nd+",),
        mesh=mesh,
    )

    assert lines == [
        "pairwise_delta=bdf_vs_bdf_fixed_full_field_jvp",
        "  Nd+: max_abs_delta=7.50000000e-01",
        "  worst=Nd+ delta=7.50000000e-01",
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
        },
        fields=("Nd+",),
        mesh=mesh,
    )

    assert field == "Nd+"
    assert delta == 0.75


def test_fixed_full_field_jvp_diagnostics_gate_accepts_expected_route() -> None:
    errors = compare_script._validate_fixed_full_field_jvp_diagnostics(
        {
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jacobian_mode": "jvp",
            "bdf_jacobian_base_rhs_evaluation_count": 0,
            "bdf_jvp_rhs_evaluation_count": 3,
        }
    )

    assert errors == []


def test_fixed_full_field_jvp_diagnostics_gate_reports_wrong_route() -> None:
    errors = compare_script._validate_fixed_full_field_jvp_diagnostics(
        {
            "bdf_rhs_backend": "host_bridge",
            "bdf_jacobian_mode": "fd",
            "bdf_jacobian_base_rhs_evaluation_count": 2,
            "bdf_jvp_rhs_evaluation_count": 0,
        }
    )

    assert errors == [
        "bdf_fixed_full_field_jvp did not report bdf_rhs_backend=fixed_full_field_array",
        "bdf_fixed_full_field_jvp did not report bdf_jacobian_mode=jvp",
        "bdf_fixed_full_field_jvp reported finite-difference base RHS Jacobian evaluations",
        "bdf_fixed_full_field_jvp did not report any JVP RHS evaluations",
    ]


def test_adaptive_bdf_modes_to_validate_selects_only_adaptive_bdf_variants() -> None:
    modes = compare_script._adaptive_bdf_modes_to_validate(
        (
            "bdf",
            "adaptive_be",
            "adaptive_bdf",
            "adaptive_bdf_jax_linearized",
            "bdf_fixed_full_field_jvp",
        )
    )

    assert modes == ("adaptive_bdf", "adaptive_bdf_jax_linearized")


def test_adaptive_bdf_diagnostics_gate_accepts_stable_jax_linearized_route() -> None:
    errors = compare_script._validate_adaptive_bdf_diagnostics(
        "adaptive_bdf_jax_linearized",
        {
            "adaptive_bdf_step_solver_mode": "jax_linearized",
            "adaptive_bdf_interval_count": 1,
            "adaptive_bdf_accepted_steps": 3,
            "adaptive_bdf_minimum_dt_fallbacks": 0,
            "adaptive_bdf_unconverged_solver_steps": 0,
            "adaptive_bdf_max_error_ratio": 0.75,
            "adaptive_bdf_max_accepted_error_ratio": 0.75,
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
    )

    assert errors == []


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
        },
        require_no_fallback=True,
        require_no_unconverged_substeps=True,
        max_error_ratio=0.95,
        max_accepted_error_ratio=0.95,
    )

    assert errors == [
        "adaptive_bdf_jax_linearized did not report adaptive_bdf_step_solver_mode=jax_linearized",
        "adaptive_bdf_jax_linearized did not report any adaptive BDF output intervals",
        "adaptive_bdf_jax_linearized did not report any accepted adaptive BDF substeps",
        "adaptive_bdf_jax_linearized reported 2 minimum-dt fallback accepts",
        "adaptive_bdf_jax_linearized reported 3 unconverged adaptive BDF implicit substeps",
        "adaptive_bdf_jax_linearized adaptive_bdf_max_error_ratio=1.25000000e+00 exceeds 9.50000000e-01",
        "adaptive_bdf_jax_linearized adaptive_bdf_max_accepted_error_ratio=1.10000000e+00 exceeds 9.50000000e-01",
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
    if not hasattr(compare_script.signal, "SIGALRM") or not hasattr(compare_script.signal, "setitimer"):
        return
    started = time.perf_counter()
    try:
        compare_script._run_with_mode_timeout(0.01, lambda: time.sleep(1.0))
    except compare_script._ModeTimeoutError as exc:
        assert "exceeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected _ModeTimeoutError")
    assert time.perf_counter() - started < 0.5
