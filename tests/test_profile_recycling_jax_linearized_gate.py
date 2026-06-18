from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "profile_recycling_jax_linearized_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "profile_recycling_jax_linearized_gate", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_accepts_preconditioner_and_budget_gates() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--input-path",
            "/tmp/BOUT.inp",
            "--case",
            "dthe",
            "--linear-preconditioner",
            "local-block-diag",
            "--linear-preconditioner-refresh",
            "100",
            "--require-linear-preconditioner",
            "local_block_diag",
            "--linear-restart",
            "20",
            "--linear-maxiter",
            "20",
            "--linear-tolerance-factor",
            "10",
            "--line-search-initial-step-scale",
            "0.25",
            "--require-max-linear-iterations",
            "3200",
            "--require-max-residual-inf-norm",
            "7.4",
            "--require-max-residual-evaluations",
            "2",
            "--require-max-line-search-trials",
            "1",
            "--require-min-linear-operator-calls",
            "1",
            "--require-max-linear-operator-calls",
            "500",
            "--require-min-linear-iterations",
            "1",
            "--require-min-nonlinear-iterations",
            "1",
            "--require-max-preconditioner-builds",
            "2",
        ]
    )

    assert args.input_path == Path("/tmp/BOUT.inp")
    assert args.case == "dthe"
    assert args.linear_preconditioner == "local-block-diag"
    assert args.linear_preconditioner_refresh == 100
    assert args.linear_restart == 20
    assert args.linear_maxiter == 20
    assert args.linear_tolerance_factor == 10.0
    assert args.line_search_initial_step_scale == 0.25
    assert args.require_linear_preconditioner == "local_block_diag"
    assert args.require_max_linear_iterations == 3200
    assert args.require_max_residual_inf_norm == 7.4
    assert args.require_max_residual_evaluations == 2
    assert args.require_max_line_search_trials == 1
    assert args.require_min_linear_operator_calls == 1
    assert args.require_max_linear_operator_calls == 500
    assert args.require_min_linear_iterations == 1
    assert args.require_min_nonlinear_iterations == 1
    assert args.require_max_preconditioner_builds == 2


def test_help_documents_preconditioner_and_budget_gates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module._parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "--linear-preconditioner" in help_text
    assert "--linear-preconditioner-refresh" in help_text
    assert "--linear-restart" in help_text
    assert "--linear-maxiter" in help_text
    assert "--linear-tolerance-factor" in help_text
    assert "--line-search-initial-step-scale" in help_text
    assert "--require-linear-preconditioner" in help_text
    assert "--require-max-linear-iterations" in help_text
    assert "--require-max-residual-inf-norm" in help_text
    assert "--require-max-residual-evaluations" in help_text
    assert "--require-max-line-search-trials" in help_text
    assert "--require-min-linear-operator-calls" in help_text
    assert "--require-max-linear-operator-calls" in help_text
    assert "--require-min-linear-iterations" in help_text
    assert "--require-min-nonlinear-iterations" in help_text
    assert "--require-max-preconditioner-builds" in help_text


def test_effective_overrides_append_linear_preconditioner_controls() -> None:
    module = _load_module()
    args = SimpleNamespace(
        override=["mesh:ny=64"],
        jit_residual=True,
        skip_initial_residual_check=True,
        gmres_solve_method="incremental",
        linear_preconditioner="local_block_diag",
        linear_preconditioner_refresh=100,
        linear_restart=20,
        linear_maxiter=10,
        linear_tolerance_factor=2.5,
        line_search_initial_step_scale=0.25,
    )

    assert module._effective_overrides(args) == [
        "mesh:ny=64",
        "runtime:recycling_jax_linear_jit_residual=true",
        "runtime:recycling_jax_linear_check_initial_residual=false",
        "runtime:recycling_jax_linear_gmres_solve_method=incremental",
        "runtime:recycling_jax_linear_restart=20",
        "runtime:recycling_jax_linear_maxiter=10",
        "runtime:recycling_jax_linear_tolerance_factor=2.5",
        "runtime:recycling_jax_linear_line_search_initial_step_scale=0.25",
        "runtime:recycling_jax_linear_preconditioner=local_block_diag",
        "runtime:recycling_jax_linear_preconditioner_refresh=100",
    ]


def test_validate_args_rejects_invalid_preconditioner_controls() -> None:
    module = _load_module()

    for kwargs, expected in (
        ({"linear_preconditioner": " "}, "--linear-preconditioner must be nonempty"),
        (
            {"linear_preconditioner_refresh": 0},
            "--linear-preconditioner-refresh must be positive",
        ),
        (
            {"require_linear_preconditioner": ""},
            "--require-linear-preconditioner must be nonempty",
        ),
        (
            {"require_max_linear_iterations": -1},
            "--require-max-linear-iterations must be nonnegative",
        ),
        ({"linear_restart": 0}, "--linear-restart must be positive"),
        ({"linear_maxiter": 0}, "--linear-maxiter must be positive"),
        (
            {"linear_tolerance_factor": 0.0},
            "--linear-tolerance-factor must be finite and positive",
        ),
        (
            {"line_search_initial_step_scale": 1.5},
            "--line-search-initial-step-scale must be finite and in (0, 1]",
        ),
        (
            {"require_max_residual_inf_norm": float("nan")},
            "--require-max-residual-inf-norm must be finite and nonnegative",
        ),
        (
            {"require_max_residual_evaluations": -1},
            "--require-max-residual-evaluations must be nonnegative",
        ),
        (
            {"require_max_line_search_trials": -1},
            "--require-max-line-search-trials must be nonnegative",
        ),
        (
            {"require_min_linear_operator_calls": -1},
            "--require-min-linear-operator-calls must be nonnegative",
        ),
        (
            {"require_max_linear_operator_calls": -1},
            "--require-max-linear-operator-calls must be nonnegative",
        ),
        (
            {"require_min_linear_iterations": -1},
            "--require-min-linear-iterations must be nonnegative",
        ),
        (
            {"require_min_nonlinear_iterations": -1},
            "--require-min-nonlinear-iterations must be nonnegative",
        ),
        (
            {"require_max_preconditioner_builds": -1},
            "--require-max-preconditioner-builds must be nonnegative",
        ),
    ):
        values = {
            "linear_preconditioner": None,
            "linear_preconditioner_refresh": None,
            "require_linear_preconditioner": None,
            "linear_restart": None,
            "linear_maxiter": None,
            "linear_tolerance_factor": None,
            "line_search_initial_step_scale": None,
            "require_max_linear_iterations": None,
            "require_max_residual_inf_norm": None,
            "require_max_residual_evaluations": None,
            "require_max_line_search_trials": None,
            "require_min_linear_operator_calls": None,
            "require_max_linear_operator_calls": None,
            "require_min_linear_iterations": None,
            "require_min_nonlinear_iterations": None,
            "require_max_preconditioner_builds": None,
        }
        values.update(kwargs)
        args = SimpleNamespace(**values)
        with pytest.raises(SystemExit) as excinfo:
            module._validate_args(args)
        assert expected in str(excinfo.value)


def test_profile_gate_errors_accept_dynamic_preconditioner_with_budgets() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="local-block-diag",
        require_max_linear_iterations=3200,
        require_max_residual_inf_norm=7.4,
        require_max_residual_evaluations=2,
        require_max_line_search_trials=1,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=500,
        require_min_linear_iterations=1,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=2,
    )
    profile_report = {
        "linear_iterations": 3200,
        "nonlinear_iterations": 1,
        "residual_inf_norm": 7.315,
        "diagnostics": {
            "linear_preconditioner": "local_block_diag",
            "linear_preconditioner_build_count": 2,
            "linear_preconditioner_build_seconds": 0.125,
            "residual_evaluation_count": 2,
            "line_search_trial_count": 1,
            "linear_operator_call_count": 128,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_static_preconditioner_without_builds() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="state-scale",
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=None,
    )
    profile_report = {
        "linear_iterations": 1,
        "diagnostics": {"linear_preconditioner": "state_scale"},
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_report_mismatch_and_budget_failures() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="parallel-line",
        require_max_linear_iterations=10,
        require_max_residual_inf_norm=7.4,
        require_max_residual_evaluations=2,
        require_max_line_search_trials=1,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=2,
        require_min_linear_iterations=1,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=2,
    )
    profile_report = {
        "linear_iterations": 24,
        "nonlinear_iterations": 1,
        "residual_inf_norm": 8.1,
        "diagnostics": {
            "linear_preconditioner": "local_block_diag",
            "linear_preconditioner_build_count": 3,
            "linear_preconditioner_build_seconds": float("nan"),
            "residual_evaluation_count": 4,
            "line_search_trial_count": 3,
            "linear_operator_call_count": 3,
        },
    }

    errors = module._profile_gate_errors(profile_report, args)

    assert "profile did not report linear_preconditioner=parallel_line" in errors
    assert (
        "profile did not report finite nonnegative "
        "linear_preconditioner_build_seconds"
    ) in errors
    assert "profile reported 24 linear iterations, exceeding 10" in errors
    assert any(
        "profile reported 8.10000000e+00 residual inf-norm" in error
        for error in errors
    )
    assert "profile reported 4 residual evaluations, exceeding 2" in errors
    assert "profile reported 3 line-search trials, exceeding 1" in errors
    assert "profile reported 3 linear-operator calls, exceeding 2" in errors
    assert "profile reported 3 preconditioner builds, exceeding 2" in errors


def test_profile_gate_errors_reject_noop_profiles() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=1,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=None,
    )
    profile_report = {
        "linear_iterations": 0,
        "nonlinear_iterations": 0,
        "diagnostics": {},
    }

    assert module._profile_gate_errors(profile_report, args) == [
        "profile did not report linear_operator_call_count",
        "profile reported 0 linear iterations, below 1",
        "profile reported 0 nonlinear iterations, below 1",
    ]
