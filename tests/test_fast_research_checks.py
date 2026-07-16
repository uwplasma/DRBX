from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
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


def test_default_fast_research_slices_include_mms_and_native() -> None:
    module = _load_script_module("scripts/run_fast_research_checks.py", "fast_research_checks_slices")

    slices = module.default_slices()

    assert {slice_.name for slice_ in slices} == {
        "runtime_surface",
        "precision_surface",
        "mms_operator",
        "native_operator",
    }


def test_all_research_slices_include_optional_convergence_campaign() -> None:
    module = _load_script_module("scripts/run_fast_research_checks.py", "fast_research_checks_all_slices")

    slices = module.all_slices()

    assert {slice_.name for slice_ in slices} == {
        "runtime_surface",
        "precision_surface",
        "mms_operator",
        "native_operator",
        "convergence_campaign",
    }


def test_build_pytest_command_adds_coverage_flags_only_when_requested() -> None:
    module = _load_script_module("scripts/run_fast_research_checks.py", "fast_research_checks_command")
    slice_ = module.PytestSlice(
        name="demo",
        description="demo slice",
        pytest_args=("tests/test_demo.py",),
    )

    without_cov = module.build_pytest_command(
        slice_,
        python_executable="python",
        with_coverage=False,
        coverage_append=False,
        extra_pytest_args=("-k", "demo"),
    )
    with_cov = module.build_pytest_command(
        slice_,
        python_executable="python",
        with_coverage=True,
        coverage_append=True,
    )

    assert without_cov == ("python", "-m", "pytest", "-q", "--maxfail=1", "tests/test_demo.py", "-k", "demo")
    assert "--cov=src/jax_drb" in with_cov
    assert "--cov-append" in with_cov


def test_run_checked_command_reports_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_script_module("scripts/run_fast_research_checks.py", "fast_research_checks_timeout")

    def fake_run(*args, **kwargs):
        assert str(tmp_path / "src") in kwargs["env"]["PYTHONPATH"]
        assert kwargs["env"]["JAX_DRB_PRECISION"] == "float64"
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=5)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_checked_command(("python", "-m", "pytest"), cwd=tmp_path, timeout_seconds=5)

    assert result.timed_out is True
    assert result.returncode == 124


def test_resolve_slices_rejects_unknown_name() -> None:
    module = _load_script_module("scripts/run_fast_research_checks.py", "fast_research_checks_resolve")

    with pytest.raises(ValueError, match="unknown slice"):
        module.resolve_slices(("not_a_slice",))
