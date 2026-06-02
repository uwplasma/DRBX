from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "research-campaigns.yml"
_DTHE_REFERENCE_INPUT = Path("tests") / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp"


def _load_script_module(relative_path: str, module_name: str):
    path = _REPO / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_dthe_reference_root(tmp_path: Path) -> Path:
    reference_root = tmp_path / "reference"
    input_path = reference_root / _DTHE_REFERENCE_INPUT
    input_path.parent.mkdir(parents=True)
    input_path.write_text("# minimal test deck\n", encoding="utf-8")
    return reference_root


def _workflow_campaign_options() -> tuple[str, ...]:
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    in_campaign = False
    in_options = False
    options: list[str] = []
    for line in lines:
        stripped = line.strip()
        if line.startswith("      campaign:"):
            in_campaign = True
            continue
        if in_campaign and stripped == "options:":
            in_options = True
            continue
        if not in_options:
            continue
        if stripped.startswith("- "):
            options.append(stripped[2:])
            continue
        if options and stripped:
            break
    assert options, "research-campaigns workflow must expose campaign dispatch choices"
    return tuple(options)


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


def test_research_campaign_workflow_choices_match_supported_campaigns() -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_workflow")

    supported_campaigns = set(
        module._campaign_command_map(
            python_executable="python",
            repo_root=_REPO,
            reference_root=Path("/reference"),
            output_root=_REPO / "docs" / "data",
            fast_timeout_seconds=300,
        )
    ) | {"all-ci", "all-gpu", "all-local"}
    workflow_options = _workflow_campaign_options()

    assert len(workflow_options) == len(set(workflow_options))
    assert set(workflow_options) == supported_campaigns
    assert workflow_options[0] == "scheduled-fast-research"


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


def test_research_campaign_heavy_profile_uses_reference_and_rss(tmp_path: Path) -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_heavy")
    reference_root = _make_dthe_reference_root(tmp_path)

    commands = module.build_campaign_commands(
        campaign_names=("heavy-recycling-profile", "dthe-jax-linearized-gate"),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    heavy, gate = commands
    assert heavy.name == "heavy-recycling-profile"
    assert "recycling_dthe_one_step" in heavy.command
    assert "--reference-root" in heavy.command
    assert str(reference_root) in heavy.command
    assert "--rss-profile" in heavy.command
    assert gate.name == "dthe-jax-linearized-gate"
    assert "--case" in gate.command
    assert "dthe" in gate.command
    assert "--skip-cprofile" in gate.command


def test_research_campaign_gpu_bundle_adds_repeatable_trace_commands(tmp_path: Path) -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_gpu")
    reference_root = _make_dthe_reference_root(tmp_path)

    commands = module.build_campaign_commands(
        campaign_names=("all-gpu",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    linearized, batched = commands
    assert linearized.name == "gpu-dthe-jax-linearized-gate"
    assert linearized.required_reference_inputs == ("dthe",)
    assert linearized.requires_gpu is True
    assert "--timed-runs" in linearized.command
    assert "--jax-trace" in linearized.command
    assert "--device-memory-profile" in linearized.command
    assert "--compilation-cache-dir" in linearized.command
    assert "mesh:ny=400" in linearized.command
    assert batched.name == "gpu-dthe-batched-jvp-gate"
    assert batched.required_reference_inputs == ("dthe",)
    assert batched.requires_gpu is True
    assert "--batch-sizes" in batched.command
    assert "2,4,8,16,32,64,128" in batched.command
    assert "--skip-objective-grad-check" in batched.command
    assert "--jax-trace" in batched.command
    assert "--device-memory-profile" in batched.command
    assert "--compilation-cache-dir" in batched.command


def test_research_campaign_gpu_bundle_requires_expected_dthe_reference_deck(tmp_path: Path) -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_gpu_reference")

    with pytest.raises(ValueError, match="tests/integrated/1D-recycling-dthe/data/BOUT.inp"):
        module.build_campaign_commands(
            campaign_names=("all-gpu",),
            python_executable="python",
            repo_root=_REPO,
            reference_root=tmp_path,
            output_root=Path("/output"),
            fast_timeout_seconds=300,
        )


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


def test_research_campaign_gpu_command_runner_sets_cuda_prerequisite_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("scripts/run_research_campaign_bundle.py", "research_campaign_gpu_env")
    command = module.CampaignCommand(
        name="gpu-demo",
        description="demo",
        command=("python", "-c", "pass"),
        requires_gpu=True,
    )
    captured_env: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    result = module.run_campaign_command(command, cwd=tmp_path, timeout_seconds=1)

    assert result.returncode == 0
    assert captured_env["JAX_PLATFORMS"] == "cuda"
    assert captured_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_batched_jvp_profiler_reports_missing_reference_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_missing_input")
    monkeypatch.delenv("JAX_DRB_REFERENCE_ROOT", raising=False)
    args = SimpleNamespace(reference_root=tmp_path, input_path=None, case="dthe")

    with pytest.raises(SystemExit) as excinfo:
        module._resolve_input(args)

    assert "tests/integrated/1D-recycling-dthe/data/BOUT.inp" in str(excinfo.value)
    assert "--input-path /path/to/BOUT.inp" in str(excinfo.value)


def test_batched_jvp_profiler_accepts_explicit_staged_input(tmp_path: Path) -> None:
    module = _load_script_module("scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_staged_input")
    input_path = tmp_path / "1D-recycling-dthe" / "data" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("# staged deck\n", encoding="utf-8")
    args = SimpleNamespace(reference_root=None, input_path=input_path, case="dthe")

    assert module._resolve_input(args) == input_path.resolve()
