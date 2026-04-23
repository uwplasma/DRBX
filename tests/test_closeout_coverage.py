from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_REPO = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    path = _REPO / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_closeout_coverage_script_uses_expected_threshold_and_targets() -> None:
    module = _load_script_module("scripts/run_closeout_coverage.py", "closeout_coverage_script")

    assert module.MIN_TOTAL_COVERAGE == 95.0
    assert "tests/test_validation_temperature_feedback_campaign.py" in module.CLOSEOUT_TESTS
    assert "tests/test_validation_autodiff_diffusion_uncertainty.py" in module.CLOSEOUT_TESTS
    assert "tests/test_packaging_metadata.py" in module.CLOSEOUT_TESTS
    assert "src/jax_drb/validation/tokamak_native_selected_field.py" in module.COVERAGE_TARGETS
    assert "src/jax_drb/validation/detachment_controller_campaign.py" in module.COVERAGE_TARGETS


def test_closeout_coverage_builds_pytest_and_report_commands() -> None:
    module = _load_script_module("scripts/run_closeout_coverage.py", "closeout_coverage_commands")

    pytest_command = module._build_pytest_command(python_executable="python")
    report_command = module._build_report_command(python_executable="python")

    assert pytest_command[:6] == ["python", "-m", "coverage", "run", "-m", "pytest"]
    assert "tests/test_release_surface.py" in pytest_command
    assert report_command[:4] == ["python", "-m", "coverage", "report"]
    assert "src/jax_drb/validation/controller_feedback_campaign.py" in report_command


def test_closeout_coverage_parses_total_line() -> None:
    module = _load_script_module("scripts/run_closeout_coverage.py", "closeout_coverage_parse")

    parsed = module._parse_total_coverage(
        "Name Stmts Miss Cover\n"
        "foo.py 10 0 100%\n"
        "TOTAL 20 1 95%\n"
    )

    assert parsed == 95.0
    with pytest.raises(ValueError, match="Could not parse TOTAL coverage"):
        module._parse_total_coverage("no total here")


def test_promoted_solver_coverage_script_tracks_solver_targets() -> None:
    module = _load_script_module("scripts/run_promoted_solver_coverage.py", "promoted_solver_coverage_script")

    assert module.MIN_TOTAL_COVERAGE == 95.0
    assert "-m" in module.PROMOTED_SOLVER_TESTS
    assert "not slow" in module.PROMOTED_SOLVER_TESTS
    assert "tests/test_native_recycling_1d.py" in module.PROMOTED_SOLVER_TESTS
    assert "tests/test_native_runner.py" in module.PROMOTED_SOLVER_TESTS
    assert "src/jax_drb/native/recycling_1d.py" in module.PROMOTED_SOLVER_TARGETS
    assert "src/jax_drb/native/runner.py" in module.PROMOTED_SOLVER_TARGETS
    assert "src/jax_drb/cli.py" in module.PROMOTED_SOLVER_TARGETS


def test_promoted_solver_coverage_builds_pytest_and_report_commands() -> None:
    module = _load_script_module("scripts/run_promoted_solver_coverage.py", "promoted_solver_coverage_commands")

    pytest_command = module._build_pytest_command(python_executable="python")
    report_command = module._build_report_command(python_executable="python")

    assert pytest_command[:6] == ["python", "-m", "coverage", "run", "-m", "pytest"]
    assert "--maxfail=1" in pytest_command
    assert "tests/test_native_integrated_2d_recycling.py" in pytest_command
    assert report_command[:4] == ["python", "-m", "coverage", "report"]
    assert "src/jax_drb/native/recycling_targets.py" in report_command
