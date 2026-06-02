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


def test_research_campaign_defaults_to_scheduled_fast_slice() -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_default")

    commands = module.build_campaign_commands(
        campaign_names=(),
        python_executable="python",
        repo_root=_REPO,
        reference_root=None,
        output_root=_REPO / "docs" / "data",
        fast_timeout_seconds=123,
    )

    assert [command.name for command in commands] == ["scheduled-fast-research"]
    assert "run_fast_research_checks.py" in commands[0].command[1]
    assert commands[0].command[-1] == "123"


def test_research_campaign_live_reference_requires_reference_root() -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_reference")

    with pytest.raises(ValueError, match="requires --reference-root"):
        module.build_campaign_commands(
            campaign_names=("live-reference",),
            python_executable="python",
            repo_root=_REPO,
            reference_root=None,
            output_root=_REPO / "docs" / "data",
            fast_timeout_seconds=300,
        )


def test_research_campaign_heavy_profile_uses_reference_and_rss() -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_heavy")

    commands = module.build_campaign_commands(
        campaign_names=("heavy-recycling-profile", "dthe-jax-linearized-gate"),
        python_executable="python",
        repo_root=_REPO,
        reference_root=Path("/reference"),
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    heavy, gate = commands
    assert heavy.name == "heavy-recycling-profile"
    assert "recycling_dthe_one_step" in heavy.command
    assert "--reference-root" in heavy.command
    assert "/reference" in heavy.command
    assert "--rss-profile" in heavy.command
    assert gate.name == "dthe-jax-linearized-gate"
    assert "--case" in gate.command
    assert "dthe" in gate.command
    assert "--skip-cprofile" in gate.command


def test_research_campaign_gpu_bundle_adds_repeatable_trace_commands() -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_gpu")

    commands = module.build_campaign_commands(
        campaign_names=("all-gpu",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=Path("/reference"),
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    linearized, batched = commands
    assert linearized.name == "gpu-dthe-jax-linearized-gate"
    assert "--timed-runs" in linearized.command
    assert "--jax-trace" in linearized.command
    assert "--device-memory-profile" in linearized.command
    assert "--compilation-cache-dir" in linearized.command
    assert "mesh:ny=400" in linearized.command
    assert batched.name == "gpu-dthe-batched-jvp-gate"
    assert "--batch-sizes" in batched.command
    assert "2,4,8,16,32,64,128" in batched.command
    assert "--skip-objective-grad-check" in batched.command
    assert "--jax-trace" in batched.command
    assert "--device-memory-profile" in batched.command
    assert "--compilation-cache-dir" in batched.command


def test_research_campaign_command_runner_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_timeout")
    command = module.CampaignCommand(name="demo", description="demo", command=("python", "-c", "pass"))

    def fake_run(*args, **kwargs):
        assert str(tmp_path / "src") in kwargs["env"]["PYTHONPATH"]
        assert kwargs["env"]["JAX_ENABLE_X64"] == "true"
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_campaign_command(command, cwd=tmp_path, timeout_seconds=1)

    assert result.timed_out is True
    assert result.returncode == 124
