from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_profile_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_curated_case.py"
    spec = importlib.util.spec_from_file_location("profile_curated_case", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile_script = _load_profile_script()


def test_json_ready_diagnostics_preserves_native_run_counters() -> None:
    result = SimpleNamespace(
        diagnostics={
            "recycling_transient_solver_mode": "bdf_fixed_full_field_jvp",
            "bdf_jacobian_mode": "jvp",
            "bdf_jacobian_base_rhs_evaluation_count": np.int64(0),
            "bdf_jacobian_callback_seconds": np.float64(0.25),
            "bdf_jvp_batch_size": None,
        }
    )

    diagnostics = profile_script._json_ready_diagnostics(result)

    assert diagnostics == {
        "recycling_transient_solver_mode": "bdf_fixed_full_field_jvp",
        "bdf_jacobian_mode": "jvp",
        "bdf_jacobian_base_rhs_evaluation_count": 0,
        "bdf_jacobian_callback_seconds": 0.25,
        "bdf_jvp_batch_size": None,
    }


def test_native_diagnostic_gate_accepts_exact_and_numeric_minimum() -> None:
    errors = profile_script._native_diagnostic_gate_errors(
        {
            "recycling_transient_solver_mode": "bdf_fixed_full_field_jvp",
            "bdf_jacobian_mode": "jvp",
            "bdf_rhs_backend": "fixed_full_field_array",
            "bdf_jvp_jacobian_gather_on_device": True,
            "bdf_jvp_jacobian_batch_count": 3,
        },
        exact_requirements=(
            "recycling_transient_solver_mode=bdf_fixed_full_field_jvp",
            "bdf_jacobian_mode=jvp",
            "bdf_rhs_backend=fixed_full_field_array",
            "bdf_jvp_jacobian_gather_on_device=True",
        ),
        minimum_requirements=("bdf_jvp_jacobian_batch_count=1",),
    )

    assert errors == []


def test_native_diagnostic_gate_reports_missing_mismatch_and_low_count() -> None:
    errors = profile_script._native_diagnostic_gate_errors(
        {
            "recycling_transient_solver_mode": "bdf",
            "bdf_jvp_jacobian_batch_count": 0,
        },
        exact_requirements=(
            "recycling_transient_solver_mode=bdf_fixed_full_field_jvp",
            "bdf_jacobian_mode=jvp",
        ),
        minimum_requirements=("bdf_jvp_jacobian_batch_count=1",),
    )

    assert (
        "native diagnostics reported recycling_transient_solver_mode='bdf', "
        "expected 'bdf_fixed_full_field_jvp'"
    ) in errors
    assert "native diagnostics did not report 'bdf_jacobian_mode'" in errors
    assert (
        "native diagnostics reported bdf_jvp_jacobian_batch_count=0, "
        "expected at least 1"
    ) in errors


def test_native_diagnostic_gate_rejects_invalid_requirement() -> None:
    with np.testing.assert_raises_regex(
        ValueError, "--require-native-diagnostic requires KEY=VALUE"
    ):
        profile_script._native_diagnostic_gate_errors(
            {"bdf_jacobian_mode": "jvp"},
            exact_requirements=("bdf_jacobian_mode",),
        )


def test_sanitize_profile_text_removes_local_absolute_paths() -> None:
    reference_root = Path.home() / "local" / "hermes-3"
    text = "\n".join(
        (
            f"{Path.cwd() / 'src' / 'jax_drb' / 'native' / 'runner.py'}:153(run_curated_case)",
            f"{reference_root / 'examples' / 'tokamak-2D'}",
            f"{Path.home() / 'base_env' / 'lib' / 'python3.13' / 'site-packages'}",
        )
    )

    sanitized = profile_script._sanitize_profile_text(text, reference_root=reference_root)

    assert "/Users/" not in sanitized
    assert "<repo-root>/src/jax_drb/native/runner.py" in sanitized
    assert "<reference-root>/examples/tokamak-2D" in sanitized
    assert "<home>/base_env/lib/python3.13/site-packages" in sanitized
