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
            "--jit-linear-operator",
            "--linear-operator-counting",
            "direct",
            "--active-array-rhs",
            "--rhs-backend",
            "active_array",
            "--linear-preconditioner",
            "local-block-diag",
            "--linear-preconditioner-refresh",
            "100",
            "--require-linear-preconditioner",
            "local_block_diag",
            "--require-initial-residual-mode",
            "linearize",
            "--require-linear-operator-jitted",
            "--require-linear-operator-finite",
            "--require-rhs-backend",
            "active_array",
            "--linear-restart",
            "20",
            "--linear-maxiter",
            "20",
            "--linear-tolerance-factor",
            "10",
            "--line-search-initial-step-scale",
            "0.25",
            "--line-search-min-step-scale",
            "0.001",
            "--require-max-linear-iterations",
            "3200",
            "--require-max-residual-inf-norm",
            "7.4",
            "--require-max-linear-update-residual",
            "1e-8",
            "--require-max-linear-update-relative-residual",
            "1e-4",
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
            "--require-min-linear-solve-count",
            "1",
            "--require-min-nonlinear-iterations",
            "1",
            "--require-max-preconditioner-builds",
            "2",
            "--require-max-preconditioner-applies",
            "40",
        ]
    )

    assert args.input_path == Path("/tmp/BOUT.inp")
    assert args.case == "dthe"
    assert args.jit_linear_operator is True
    assert args.linear_operator_counting == "direct"
    assert args.active_array_rhs is True
    assert args.rhs_backend == "active_array"
    assert args.linear_preconditioner == "local-block-diag"
    assert args.linear_preconditioner_refresh == 100
    assert args.linear_restart == 20
    assert args.linear_maxiter == 20
    assert args.linear_tolerance_factor == 10.0
    assert args.line_search_initial_step_scale == 0.25
    assert args.line_search_min_step_scale == 0.001
    assert args.require_linear_preconditioner == "local_block_diag"
    assert args.require_initial_residual_mode == "linearize"
    assert args.require_linear_operator_jitted is True
    assert args.require_linear_operator_finite is True
    assert args.require_rhs_backend == "active_array"
    assert args.require_max_linear_iterations == 3200
    assert args.require_max_residual_inf_norm == 7.4
    assert args.require_max_linear_update_residual == 1.0e-8
    assert args.require_max_linear_update_relative_residual == 1.0e-4
    assert args.require_max_residual_evaluations == 2
    assert args.require_max_line_search_trials == 1
    assert args.require_min_linear_operator_calls == 1
    assert args.require_max_linear_operator_calls == 500
    assert args.require_min_linear_iterations == 1
    assert args.require_min_linear_solve_count == 1
    assert args.require_min_nonlinear_iterations == 1
    assert args.require_max_preconditioner_builds == 2
    assert args.require_max_preconditioner_applies == 40

    promoted_args = module._parse_args(
        [
            "--input-path",
            "/tmp/BOUT.inp",
            "--rhs-backend",
            "promoted_active_sources",
            "--require-rhs-backend",
            "promoted_active_sources",
        ]
    )
    assert promoted_args.rhs_backend == "promoted_active_sources"
    assert promoted_args.require_rhs_backend == "promoted_active_sources"


def test_help_documents_preconditioner_and_budget_gates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module._parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "--linear-preconditioner" in help_text
    assert "--linear-preconditioner-refresh" in help_text
    assert "--jit-linear-operator" in help_text
    assert "--linear-operator-counting" in help_text
    assert "--linear-restart" in help_text
    assert "--linear-maxiter" in help_text
    assert "--linear-tolerance-factor" in help_text
    assert "--line-search-initial-step-scale" in help_text
    assert "--active-array-rhs" in help_text
    assert "--rhs-backend" in help_text
    assert "--require-linear-preconditioner" in help_text
    assert "--require-initial-residual-mode" in help_text
    assert "--require-linear-operator-jitted" in help_text
    assert "--require-linear-operator-finite" in help_text
    assert "--require-rhs-backend" in help_text
    assert "--require-max-linear-iterations" in help_text
    assert "--require-max-residual-inf-norm" in help_text
    assert "--require-max-linear-update-residual" in help_text
    assert "--require-max-linear-update-relative-residual" in help_text
    assert "--require-max-residual-evaluations" in help_text
    assert "--require-max-line-search-trials" in help_text
    assert "--require-min-linear-operator-calls" in help_text
    assert "--require-max-linear-operator-calls" in help_text
    assert "--require-min-linear-iterations" in help_text
    assert "--require-min-linear-solve-count" in help_text
    assert "--require-min-nonlinear-iterations" in help_text
    assert "--require-max-preconditioner-builds" in help_text
    assert "--require-max-preconditioner-applies" in help_text


def test_effective_overrides_append_linear_preconditioner_controls() -> None:
    module = _load_module()
    args = SimpleNamespace(
        override=["mesh:ny=64"],
        jit_residual=True,
        jit_linear_operator=True,
        linear_operator_counting="direct",
        skip_initial_residual_check=True,
        initial_residual_mode="linearize",
        gmres_solve_method="incremental",
        linear_preconditioner="local_block_diag",
        linear_preconditioner_refresh=100,
        linear_restart=20,
        linear_maxiter=10,
        linear_tolerance_factor=2.5,
        line_search_initial_step_scale=0.25,
        line_search_min_step_scale=0.001,
    )

    assert module._effective_overrides(args) == [
        "mesh:ny=64",
        "runtime:recycling_jax_linear_jit_residual=true",
        "runtime:recycling_jax_linear_jit_linear_operator=true",
        "runtime:recycling_jax_linear_operator_counting=direct",
        "runtime:recycling_jax_linear_check_initial_residual=false",
        "runtime:recycling_jax_linear_initial_residual_mode=linearize",
        "runtime:recycling_jax_linear_gmres_solve_method=incremental",
        "runtime:recycling_jax_linear_restart=20",
        "runtime:recycling_jax_linear_maxiter=10",
        "runtime:recycling_jax_linear_tolerance_factor=2.5",
        "runtime:recycling_jax_linear_line_search_initial_step_scale=0.25",
        "runtime:recycling_jax_linear_line_search_min_step_scale=0.001",
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
            {
                "active_array_rhs": True,
                "rhs_backend": "promoted_active_sources",
            },
            "--active-array-rhs is a compatibility alias for --rhs-backend=active_array",
        ),
        (
            {
                "rhs_backend": "promoted_active_sources",
                "linear_solver_backend": "lineax_gmres",
            },
            "--rhs-backend=promoted_active_sources currently supports --linear-solver-backend=jax_gmres only",
        ),
        (
            {"require_initial_residual_mode": "bad"},
            "--require-initial-residual-mode must be 'residual' or 'linearize'",
        ),
        (
            {"initial_residual_mode": "bad"},
            "--initial-residual-mode must be 'residual' or 'linearize'",
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
            {"line_search_min_step_scale": 1.5},
            "--line-search-min-step-scale must be finite and in (0, 1]",
        ),
        (
            {"require_max_residual_inf_norm": float("nan")},
            "--require-max-residual-inf-norm must be finite and nonnegative",
        ),
        (
            {"require_max_linear_update_residual": float("nan")},
            "--require-max-linear-update-residual must be finite and nonnegative",
        ),
        (
            {"require_max_linear_update_relative_residual": float("nan")},
            "--require-max-linear-update-relative-residual must be finite and nonnegative",
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
            {
                "linear_operator_counting": "direct",
                "require_max_linear_operator_calls": 5,
            },
            "--linear-operator-counting=direct disables Python-visible operator call counts",
        ),
        (
            {"require_min_linear_iterations": -1},
            "--require-min-linear-iterations must be nonnegative",
        ),
        (
            {"require_min_linear_solve_count": -1},
            "--require-min-linear-solve-count must be nonnegative",
        ),
        (
            {"require_min_nonlinear_iterations": -1},
            "--require-min-nonlinear-iterations must be nonnegative",
        ),
        (
            {"require_max_preconditioner_builds": -1},
            "--require-max-preconditioner-builds must be nonnegative",
        ),
        (
            {"require_max_preconditioner_applies": -1},
            "--require-max-preconditioner-applies must be nonnegative",
        ),
    ):
        values = {
            "linear_preconditioner": None,
            "linear_preconditioner_refresh": None,
            "require_linear_preconditioner": None,
            "rhs_backend": None,
            "active_array_rhs": False,
            "linear_solver_backend": "jax_gmres",
            "require_initial_residual_mode": None,
            "initial_residual_mode": None,
            "linear_operator_counting": None,
            "linear_restart": None,
            "linear_maxiter": None,
            "linear_tolerance_factor": None,
            "line_search_initial_step_scale": None,
            "line_search_min_step_scale": None,
            "require_max_linear_iterations": None,
            "require_max_residual_inf_norm": None,
            "require_max_linear_update_residual": None,
            "require_max_linear_update_relative_residual": None,
            "require_max_residual_evaluations": None,
            "require_max_line_search_trials": None,
            "require_min_linear_operator_calls": None,
            "require_max_linear_operator_calls": None,
            "require_min_linear_iterations": None,
            "require_min_linear_solve_count": None,
            "require_min_nonlinear_iterations": None,
            "require_max_preconditioner_builds": None,
            "require_max_preconditioner_applies": None,
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
        require_initial_residual_mode="linearized",
        require_linear_operator_jitted=True,
        require_linear_operator_finite=True,
        require_rhs_backend="active_array",
        require_max_linear_iterations=3200,
        require_max_residual_inf_norm=7.4,
        require_max_linear_update_residual=1.0e-8,
        require_max_linear_update_relative_residual=1.0e-4,
        require_max_residual_evaluations=2,
        require_max_line_search_trials=1,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=500,
        require_min_linear_iterations=1,
        require_min_linear_solve_count=1,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=2,
        require_max_preconditioner_applies=40,
    )
    profile_report = {
        "linear_iterations": 3200,
        "nonlinear_iterations": 1,
        "residual_inf_norm": 7.315,
        "diagnostics": {
            "linear_preconditioner": "local_block_diag",
            "initial_residual_mode": "linearize",
            "linear_operator_jitted": True,
            "linear_operator_finite": True,
            "rhs_backend": "active_array",
            "linear_preconditioner_build_count": 2,
            "linear_preconditioner_build_seconds": 0.125,
            "linear_preconditioner_apply_count": 35,
            "residual_evaluation_count": 2,
            "line_search_trial_count": 1,
            "linear_operator_call_count": 128,
            "linear_solve_count": 1,
            "linear_update_residual_inf_norm": 2.0e-9,
            "linear_update_relative_residual": 5.0e-5,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_field_diag_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="field-diag",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "field_diag",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.01,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_field_block_feedback_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="field-block-feedback-diag",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "field_block_feedback_diag",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.01,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_static_preconditioner_without_builds() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="state-scale",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=None,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 1,
        "diagnostics": {"linear_preconditioner": "state_scale"},
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_neutral_line_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="neutral-line",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "neutral_line",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.02,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_momentum_line_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="momentum-line",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "momentum_line",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.02,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_sheath_line_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="target-sheath",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "sheath_line",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.02,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_accept_target_schur_as_dynamic_preconditioner() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="sheath-schur",
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_rhs_backend=None,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=None,
        require_min_nonlinear_iterations=None,
        require_max_preconditioner_builds=1,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 4,
        "diagnostics": {
            "linear_preconditioner": "target_schur",
            "linear_preconditioner_build_count": 1,
            "linear_preconditioner_build_seconds": 0.03,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []


def test_profile_gate_errors_report_mismatch_and_budget_failures() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner="parallel-line",
        require_initial_residual_mode="linearize",
        require_linear_operator_jitted=True,
        require_linear_operator_finite=True,
        require_rhs_backend="active_array",
        require_max_linear_iterations=10,
        require_max_residual_inf_norm=7.4,
        require_max_linear_update_residual=1.0e-8,
        require_max_linear_update_relative_residual=1.0e-4,
        require_max_residual_evaluations=2,
        require_max_line_search_trials=1,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=2,
        require_min_linear_iterations=1,
        require_min_linear_solve_count=2,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=2,
        require_max_preconditioner_applies=2,
    )
    profile_report = {
        "linear_iterations": 24,
        "nonlinear_iterations": 1,
        "residual_inf_norm": 8.1,
        "diagnostics": {
            "linear_preconditioner": "local_block_diag",
            "initial_residual_mode": "residual",
            "linear_operator_jitted": False,
            "linear_operator_finite": False,
            "rhs_backend": "fixed_full_field_array",
            "linear_preconditioner_build_count": 3,
            "linear_preconditioner_build_seconds": float("nan"),
            "linear_preconditioner_apply_count": 5,
            "residual_evaluation_count": 4,
            "line_search_trial_count": 3,
            "linear_operator_call_count": 3,
            "linear_solve_count": 1,
            "linear_update_residual_inf_norm": 2.0e-8,
            "linear_update_relative_residual": 2.0e-4,
        },
    }

    errors = module._profile_gate_errors(profile_report, args)

    assert "profile did not report linear_preconditioner=parallel_line" in errors
    assert (
        "profile reported diagnostics.initial_residual_mode=residual, "
        "expected linearize"
    ) in errors
    assert "profile did not report diagnostics.linear_operator_jitted=true" in errors
    assert "profile did not report diagnostics.linear_operator_finite=true" in errors
    assert (
        "profile reported diagnostics.rhs_backend='fixed_full_field_array', "
        "expected 'active_array'"
    ) in errors
    assert (
        "profile did not report finite nonnegative "
        "linear_preconditioner_build_seconds"
    ) in errors
    assert "profile reported 24 linear iterations, exceeding 10" in errors
    assert any(
        "profile reported 8.10000000e+00 residual inf-norm" in error
        for error in errors
    )
    assert any(
        "profile reported 2.00000000e-08 linear-update residual inf-norm" in error
        for error in errors
    )
    assert any(
        "profile reported 2.00000000e-04 linear-update relative residual" in error
        for error in errors
    )
    assert "profile reported 4 residual evaluations, exceeding 2" in errors
    assert "profile reported 3 line-search trials, exceeding 1" in errors
    assert "profile reported 3 linear-operator calls, exceeding 2" in errors
    assert "profile reported 1 linear solve attempts, below 2" in errors
    assert "profile reported 3 preconditioner builds, exceeding 2" in errors
    assert "profile reported 5 preconditioner applies, exceeding 2" in errors


def test_profile_gate_errors_reject_noop_profiles() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner=None,
        require_initial_residual_mode=None,
        require_linear_operator_jitted=False,
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=1,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=1,
        require_min_linear_solve_count=1,
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
        "profile did not report linear_solve_count",
        "profile reported 0 nonlinear iterations, below 1",
    ]


def test_profile_gate_errors_accept_direct_counting_solve_attempt_gate() -> None:
    module = _load_module()
    args = SimpleNamespace(
        require_linear_preconditioner=None,
        require_initial_residual_mode="linearize",
        require_linear_operator_jitted=True,
        require_rhs_backend="active_array",
        require_max_linear_iterations=None,
        require_max_residual_inf_norm=None,
        require_max_residual_evaluations=None,
        require_max_line_search_trials=None,
        require_min_linear_operator_calls=None,
        require_max_linear_operator_calls=None,
        require_min_linear_iterations=None,
        require_min_linear_solve_count=1,
        require_min_nonlinear_iterations=1,
        require_max_preconditioner_builds=None,
        require_max_preconditioner_applies=None,
    )
    profile_report = {
        "linear_iterations": 0,
        "nonlinear_iterations": 1,
        "diagnostics": {
            "initial_residual_mode": "linearize",
            "linear_operator_jitted": True,
            "rhs_backend": "active_array",
            "linear_operator_counting": "direct",
            "linear_solve_count": 1,
        },
    }

    assert module._profile_gate_errors(profile_report, args) == []
