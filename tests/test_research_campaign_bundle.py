from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "research-campaigns.yml"
_DTHE_REFERENCE_INPUT = (
    Path("tests") / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp"
)


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
    hydrogen_input = (
        reference_root / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp"
    )
    hydrogen_input.parent.mkdir(parents=True)
    hydrogen_input.write_text("# minimal hydrogen test deck\n", encoding="utf-8")
    return reference_root


def _assert_fixed_bdf2_direct_counting_command(command, *, requires_gpu: bool) -> None:
    assert command.required_reference_inputs == ("hydrogen",)
    assert command.requires_gpu is requires_gpu
    assert command.timeout_seconds == (720 if requires_gpu else 300)
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_1d_one_step" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "46"
    assert command.command[command.command.index("--timestep") + 1] == "10"
    assert command.command[command.command.index("--steps") + 1] == "2"
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command


def _assert_fixed_bdf2_linear_update_residual_command(command) -> None:
    assert command.required_reference_inputs == ("hydrogen",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 360
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_1d_one_step" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-linear-update-residual") + 1
    ] == "2e-8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-linear-update-relative-residual")
        + 1
    ] == "2e-5"
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "runtime:recycling_jax_linear_diagnose_update_residual=true" in (
        command.command
    )
    assert "--output-json" in command.command


def _assert_dthe_fixed_bdf2_active_array_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 300
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-10"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "2"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "4"
    assert command.command[command.command.index("--timestep") + 1] == "1e-4"
    assert command.command[command.command.index("--steps") + 1] == "2"
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command


def _assert_dthe_fixed_bdf2_active_array_long_window_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 420
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-10"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "16"
    assert command.command[command.command.index("--timestep") + 1] == "1e-4"
    assert command.command[command.command.index("--steps") + 1] == "8"
    assert command.command[command.command.index("--mode-timeout-seconds") + 1] == (
        "300"
    )
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command
    assert any(
        "recycling_dthe_fixed_bdf2_active_array_long_window_cpu" in part
        for part in command.command
    )


def _assert_dthe_fixed_bdf2_active_array_physical_parity_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 420
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert command.command.count("--mode") == 2
    assert "bdf" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-10"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "16"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-max") + 1
    ] == "2.5e-7"
    assert command.command[command.command.index("--timestep") + 1] == "1e-4"
    assert command.command[command.command.index("--steps") + 1] == "8"
    assert command.command[command.command.index("--mode-timeout-seconds") + 1] == (
        "300"
    )
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command
    assert any(
        "recycling_dthe_fixed_bdf2_active_array_physical_parity_cpu" in part
        for part in command.command
    )


def _assert_dthe_fixed_bdf2_active_array_parity_ramp_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 540
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert command.command.count("--mode") == 2
    assert "bdf" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-9"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "16"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-max") + 1
    ] == "2.5e-5"
    assert command.command[command.command.index("--timestep") + 1] == "1e-3"
    assert command.command[command.command.index("--steps") + 1] == "8"
    assert command.command[command.command.index("--mode-timeout-seconds") + 1] == (
        "360"
    )
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command
    assert any(
        "recycling_dthe_fixed_bdf2_active_array_parity_ramp_cpu" in part
        for part in command.command
    )


def _assert_dthe_fixed_bdf2_active_array_scalar_observable_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 360
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert command.command.count("--mode") == 2
    assert "bdf" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-7"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "2"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "4"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-l2-rel-max") + 1
    ] == "5e-5"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-inventory-rel-max") + 1
    ] == "1e-5"
    fields = [
        command.command[index + 1]
        for index, part in enumerate(command.command)
        if part == "--field"
    ]
    assert fields == ["Nd+", "Pd+", "Nd", "Pd", "Pe"]
    assert command.command[command.command.index("--timestep") + 1] == "1e-2"
    assert command.command[command.command.index("--steps") + 1] == "2"
    assert command.command[command.command.index("--mode-timeout-seconds") + 1] == (
        "240"
    )
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command
    assert any(
        "recycling_dthe_fixed_bdf2_active_array_scalar_observable_cpu" in part
        for part in command.command
    )


def _assert_dthe_fixed_bdf2_active_array_substepped_full_field_command(
    command,
) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 900
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "recycling_dthe_one_step" in command.command
    assert command.command.count("--mode") == 2
    assert "bdf" in command.command
    assert "fixed_bdf2_active_array_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual") + 1
    ] == "1e-8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "8"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "16"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-max") + 1
    ] == "1.25e-4"
    assert command.command[command.command.index("--timestep") + 1] == "1e-2"
    assert command.command[command.command.index("--steps") + 1] == "2"
    assert command.command[command.command.index("--mode-timeout-seconds") + 1] == (
        "720"
    )
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_operator_counting=direct" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "runtime:recycling_fixed_bdf2_max_internal_timestep=2.5e-3" in (
        command.command
    )
    assert "--output-json" in command.command
    assert any(
        "recycling_dthe_fixed_bdf2_active_array_substepped_full_field_cpu" in part
        for part in command.command
    )


def _assert_batched_jvp_command(
    command,
    *,
    requires_gpu: bool,
    rhs_backend: str,
    batch_sizes: str,
) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is requires_gpu
    assert "profile_recycling_batched_jvp_gate.py" in command.command[1]
    assert command.command[command.command.index("--case") + 1] == "dthe"
    if rhs_backend == "fixed_full_field_array":
        assert "--rhs-backend" not in command.command
    else:
        assert command.command[command.command.index("--rhs-backend") + 1] == (
            rhs_backend
        )
    assert command.command[command.command.index("--batch-sizes") + 1] == batch_sizes
    if not requires_gpu:
        assert "--disable-pmap" in command.command


def _assert_linearized_update_command(
    command,
    *,
    preconditioner: str = "none",
    skip_residual_diagnostic: bool = False,
) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == (420 if preconditioner == "jvp_diag" else 300)
    assert "profile_recycling_batched_jvp_gate.py" in command.command[1]
    assert command.command[command.command.index("--case") + 1] == "dthe"
    assert command.command[command.command.index("--rhs-backend") + 1] == (
        "active_array"
    )
    assert command.command[command.command.index("--override") + 1] == "mesh:ny=16"
    assert command.command[command.command.index("--batch-sizes") + 1] == "1"
    assert command.command[command.command.index("--timed-runs") + 1] == "1"
    assert "--disable-pmap" in command.command
    assert "--skip-objective-grad-check" in command.command
    assert "--check-linearized-update" in command.command
    assert "--linearized-update-jit-operator" in command.command
    assert command.command[
        command.command.index("--linearized-update-tolerance") + 1
    ] == "1e-8"
    assert command.command[
        command.command.index("--linearized-update-restart") + 1
    ] == "8"
    assert command.command[
        command.command.index("--linearized-update-maxiter") + 1
    ] == "8"
    assert command.command[
        command.command.index("--linearized-update-preconditioner") + 1
    ] == preconditioner
    if preconditioner == "jvp_diag":
        assert command.command[
            command.command.index("--linearized-update-preconditioner-max-unknowns")
            + 1
        ] == "512"
    if skip_residual_diagnostic:
        assert "--skip-linearized-update-residual-diagnostic" in command.command
    else:
        assert "--skip-linearized-update-residual-diagnostic" not in command.command


def _assert_current_dthe_jax_linearized_command(
    command,
    *,
    requires_gpu: bool,
) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is requires_gpu
    assert "profile_recycling_jax_linearized_gate.py" in command.command[1]
    assert command.command[command.command.index("--case") + 1] == "dthe"
    assert command.command[command.command.index("--timestep") + 1] == "1.0"
    assert command.command[command.command.index("--linear-restart") + 1] == "20"
    assert command.command[command.command.index("--linear-maxiter") + 1] == "20"
    assert command.command[
        command.command.index("--line-search-initial-step-scale") + 1
    ] == "0.25"
    assert "--skip-initial-residual-check" not in command.command
    assert "--jit-linear-operator" in command.command
    assert "--require-linear-operator-jitted" in command.command
    assert command.command[command.command.index("--initial-residual-mode") + 1] == (
        "linearize"
    )
    assert command.command[command.command.index("--require-initial-residual-mode") + 1] == (
        "linearize"
    )
    assert command.command[
        command.command.index("--require-min-nonlinear-iterations") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-min-linear-iterations") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-max-linear-iterations") + 1
    ] == "400"
    assert command.command[
        command.command.index("--require-max-residual-inf-norm") + 1
    ] == "7.4"
    assert command.command[
        command.command.index("--require-max-residual-evaluations") + 1
    ] == "2"
    assert command.command[
        command.command.index("--require-max-line-search-trials") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-min-linear-operator-calls") + 1
    ] == "1"
    assert "--rss-profile" in command.command
    assert "--skip-cprofile" in command.command
    if requires_gpu:
        assert command.timeout_seconds == 900
        assert "--jax-trace" in command.command
        assert "--device-memory-profile" in command.command
        assert "--compilation-cache-dir" in command.command
        assert any(
            "recycling_dthe_jax_linearized_gate_gpu_current" in part
            for part in command.command
        )
    else:
        assert "--jax-trace" not in command.command
        assert "--device-memory-profile" not in command.command


def _assert_dthe_promoted_active_sources_profile_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 300
    assert "profile_recycling_jax_linearized_gate.py" in command.command[1]
    assert command.command[command.command.index("--case") + 1] == "dthe"
    assert command.command[command.command.index("--rhs-backend") + 1] == (
        "promoted_active_sources"
    )
    assert command.command[command.command.index("--require-rhs-backend") + 1] == (
        "promoted_active_sources"
    )
    assert "mesh:ny=100" in command.command
    assert command.command[command.command.index("--timestep") + 1] == "1e-4"
    assert command.command[command.command.index("--residual-tolerance") + 1] == (
        "1e-6"
    )
    assert command.command[command.command.index("--max-nonlinear-iterations") + 1] == (
        "1"
    )
    assert command.command[command.command.index("--initial-residual-mode") + 1] == (
        "linearize"
    )
    assert command.command[
        command.command.index("--require-initial-residual-mode") + 1
    ] == "linearize"
    assert command.command[
        command.command.index("--require-min-nonlinear-iterations") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-min-linear-solve-count") + 1
    ] == "1"
    assert command.command[
        command.command.index("--require-min-linear-operator-calls") + 1
    ] == "1"
    assert command.command[command.command.index("--require-max-residual-inf-norm") + 1] == (
        "1e-6"
    )
    assert command.command[command.command.index("--cprofile-top") + 1] == "35"
    assert "--rss-profile" in command.command
    assert "--skip-cprofile" not in command.command


def _assert_dthe_fixed_bdf2_promoted_active_sources_command(command) -> None:
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 300
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert command.command[command.command.index("--case") + 1] == (
        "recycling_dthe_one_step"
    )
    assert "bdf" in command.command
    assert "fixed_bdf2_promoted_active_sources_jax_linearized" in command.command
    assert "--diagnostics-only" in command.command
    assert "--require-fixed-bdf2-diagnostics" in command.command
    assert command.command[command.command.index("--require-fixed-bdf2-max-residual") + 1] == (
        "1e-4"
    )
    assert command.command[
        command.command.index("--require-fixed-bdf2-linear-solver-backend") + 1
    ] == "jax_gmres"
    assert "--require-fixed-bdf2-linear-operator-jitted" in command.command
    assert command.command[
        command.command.index("--require-fixed-bdf2-min-linear-solve-count") + 1
    ] == "2"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-residual-evaluations") + 1
    ] == "4"
    assert command.command[
        command.command.index("--require-fixed-bdf2-max-linear-operator-calls") + 1
    ] == "12"
    assert command.command[
        command.command.index("--require-fixed-bdf2-pairwise-max") + 1
    ] == "1e-6"
    assert command.command[command.command.index("--timestep") + 1] == "1e-4"
    assert command.command[command.command.index("--steps") + 1] == "2"
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in command.command
    assert "runtime:recycling_jax_linear_initial_residual_mode=linearize" in (
        command.command
    )
    assert "--output-json" in command.command


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
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_default"
    )

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


def test_research_campaign_all_local_includes_fixed_bdf2_direct_counting() -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_all_local"
    )

    assert "fixed-bdf2-direct-counting-gate" in module.expand_campaign_names(
        ("all-local",)
    )
    assert "fixed-bdf2-linear-update-residual-gate" in module.expand_campaign_names(
        ("all-local",)
    )
    assert "dthe-active-array-batched-jvp-gate" in module.expand_campaign_names(
        ("all-local",)
    )
    assert "dthe-active-array-linearized-update-gate" in module.expand_campaign_names(
        ("all-local",)
    )
    assert "dthe-active-array-linearized-update-throughput-probe" in (
        module.expand_campaign_names(("all-local",))
    )
    assert "dthe-promoted-active-sources-profile-gate" in (
        module.expand_campaign_names(("all-local",))
    )
    assert "dthe-fixed-bdf2-promoted-active-sources-gate" in (
        module.expand_campaign_names(("all-local",))
    )
    assert "dthe-fixed-bdf2-active-array-gate" in module.expand_campaign_names(
        ("all-local",)
    )


def test_research_campaign_workflow_choices_match_supported_campaigns() -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_workflow"
    )

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
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_reference"
    )

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
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_heavy"
    )
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
    _assert_current_dthe_jax_linearized_command(gate, requires_gpu=False)


def test_research_campaign_promoted_active_sources_profile_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_promoted_active_sources_profile",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-promoted-active-sources-profile-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-promoted-active-sources-profile-gate"
    _assert_dthe_promoted_active_sources_profile_command(command)


def test_research_campaign_promoted_active_sources_fixed_bdf2_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_promoted_active_sources_fixed_bdf2",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-promoted-active-sources-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-promoted-active-sources-gate"
    _assert_dthe_fixed_bdf2_promoted_active_sources_command(command)


def test_research_campaign_adaptive_bdf_gate_writes_json_report(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_adaptive_bdf"
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("adaptive-bdf-jax-lineax-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "adaptive-bdf-jax-lineax-gate"
    assert command.required_reference_inputs == ("hydrogen",)
    assert "compare_recycling_transient_modes.py" in command.command[1]
    assert "--mode" in command.command
    assert "adaptive_bdf_jax_linearized" in command.command
    assert "adaptive_bdf_jax_linearized_lineax" in command.command
    assert "--require-adaptive-bdf-no-fallback" in command.command
    assert "--require-adaptive-bdf-no-unconverged-substeps" in command.command
    assert "--output-json" in command.command
    assert any(
        "recycling_1d_adaptive_bdf_jax_lineax_gate" in part for part in command.command
    )


def test_research_campaign_fixed_bdf2_direct_counting_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_fixed_bdf2_direct",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("fixed-bdf2-direct-counting-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "fixed-bdf2-direct-counting-gate"
    _assert_fixed_bdf2_direct_counting_command(command, requires_gpu=False)


def test_research_campaign_fixed_bdf2_linear_update_residual_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_fixed_bdf2_linear_update_residual",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("fixed-bdf2-linear-update-residual-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "fixed-bdf2-linear-update-residual-gate"
    _assert_fixed_bdf2_linear_update_residual_command(command)


def test_research_campaign_dthe_fixed_bdf2_active_array_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_active_array",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-gate"
    _assert_dthe_fixed_bdf2_active_array_command(command)


def test_research_campaign_dthe_fixed_bdf2_active_array_long_window_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_active_array_long_window",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-long-window-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-long-window-gate"
    _assert_dthe_fixed_bdf2_active_array_long_window_command(command)


def test_research_campaign_dthe_fixed_bdf2_active_array_physical_parity_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_active_array_physical_parity",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-physical-parity-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-physical-parity-gate"
    _assert_dthe_fixed_bdf2_active_array_physical_parity_command(command)


def test_research_campaign_dthe_fixed_bdf2_active_array_parity_ramp_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_active_array_parity_ramp",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-parity-ramp-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-parity-ramp-gate"
    _assert_dthe_fixed_bdf2_active_array_parity_ramp_command(command)


def test_research_campaign_dthe_fixed_bdf2_scalar_observable_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_scalar_observable",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-scalar-observable-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-scalar-observable-gate"
    _assert_dthe_fixed_bdf2_active_array_scalar_observable_command(command)


def test_research_campaign_dthe_fixed_bdf2_substepped_full_field_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_dthe_fixed_bdf2_substepped_full_field",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-fixed-bdf2-active-array-substepped-full-field-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-fixed-bdf2-active-array-substepped-full-field-gate"
    _assert_dthe_fixed_bdf2_active_array_substepped_full_field_command(command)


def test_research_campaign_gpu_bundle_adds_repeatable_trace_commands(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_gpu"
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    commands = module.build_campaign_commands(
        campaign_names=("all-gpu",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    (
        current_linearized,
        linearized,
        fixed_bdf2,
        active_batched,
        active_output,
        full_output,
        batched,
    ) = commands
    assert current_linearized.name == "gpu-dthe-current-jax-linearized-gate"
    _assert_current_dthe_jax_linearized_command(
        current_linearized,
        requires_gpu=True,
    )
    assert linearized.name == "gpu-dthe-jax-linearized-gate"
    assert linearized.required_reference_inputs == ("dthe",)
    assert linearized.requires_gpu is True
    assert "--timed-runs" in linearized.command
    assert "--jax-trace" in linearized.command
    assert "--device-memory-profile" in linearized.command
    assert "--compilation-cache-dir" in linearized.command
    assert "mesh:ny=400" in linearized.command
    assert "--active-array-rhs" in linearized.command
    assert "--jit-linear-operator" in linearized.command
    assert "--require-linear-operator-jitted" in linearized.command
    assert linearized.command[linearized.command.index("--require-rhs-backend") + 1] == (
        "active_array"
    )
    assert fixed_bdf2.name == "gpu-fixed-bdf2-direct-counting-gate"
    _assert_fixed_bdf2_direct_counting_command(fixed_bdf2, requires_gpu=True)
    assert active_batched.name == "gpu-dthe-active-array-batched-jvp-gate"
    assert active_batched.timeout_seconds == 900
    _assert_batched_jvp_command(
        active_batched,
        requires_gpu=True,
        rhs_backend="active_array",
        batch_sizes="2,4,8,16,32,64,128",
    )
    assert "--disable-pmap" in active_batched.command
    assert active_batched.command[
        active_batched.command.index("--residual-partition-size") + 1
    ] == "16"
    assert active_batched.command[
        active_batched.command.index("--jvp-partition-size") + 1
    ] == "16"
    assert "--skip-objective-grad-check" in active_batched.command
    assert "--jax-trace" in active_batched.command
    assert "--device-memory-profile" in active_batched.command
    assert "--compilation-cache-dir" in active_batched.command
    assert active_output.name == "gpu-dthe-active-array-output-jvp-profile"
    assert active_output.required_reference_inputs == ("dthe",)
    assert active_output.requires_gpu is True
    assert "recycling_dthe_one_step" in active_output.command
    assert (
        "runtime:recycling_transient_solver_mode=bdf_active_array_jvp"
        in active_output.command
    )
    assert "recycling_transient_solver_mode=bdf_active_array_jvp" in active_output.command
    assert "bdf_jacobian_mode=jvp" in active_output.command
    assert "bdf_rhs_backend=active_array" in active_output.command
    assert "bdf_jvp_jacobian_gather_on_device=True" in active_output.command
    assert "bdf_jvp_jacobian_batch_count=1" in active_output.command
    assert "--jax-trace" in active_output.command
    assert "--device-memory-profile" in active_output.command
    assert "--compilation-cache-dir" in active_output.command
    assert full_output.name == "gpu-dthe-full-output-jvp-profile"
    assert full_output.required_reference_inputs == ("dthe",)
    assert full_output.requires_gpu is True
    assert "recycling_dthe_one_step" in full_output.command
    assert (
        "runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp"
        in full_output.command
    )
    assert "--require-native-diagnostic" in full_output.command
    assert (
        "recycling_transient_solver_mode=bdf_fixed_full_field_jvp"
        in full_output.command
    )
    assert "bdf_jacobian_mode=jvp" in full_output.command
    assert "bdf_rhs_backend=fixed_full_field_array" in full_output.command
    assert "bdf_jvp_jacobian_gather_on_device=True" in full_output.command
    assert "--require-min-native-diagnostic" in full_output.command
    assert "bdf_jvp_jacobian_batch_count=1" in full_output.command
    assert "--jax-trace" in full_output.command
    assert "--device-memory-profile" in full_output.command
    assert "--compilation-cache-dir" in full_output.command
    assert batched.name == "gpu-dthe-batched-jvp-gate"
    _assert_batched_jvp_command(
        batched,
        requires_gpu=True,
        rhs_backend="fixed_full_field_array",
        batch_sizes="2,4,8,16,32,64,128",
    )
    assert "--skip-objective-grad-check" in batched.command
    assert "--jax-trace" in batched.command
    assert "--device-memory-profile" in batched.command
    assert "--compilation-cache-dir" in batched.command


def test_research_campaign_active_array_batched_jvp_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_active_array_batched_jvp",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-active-array-batched-jvp-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-active-array-batched-jvp-gate"
    _assert_batched_jvp_command(
        command,
        requires_gpu=False,
        rhs_backend="active_array",
        batch_sizes="1,4,16,64",
    )
    assert "mesh:ny=100" in command.command


def test_research_campaign_active_array_linearized_update_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_active_array_linearized_update",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-active-array-linearized-update-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-active-array-linearized-update-gate"
    _assert_linearized_update_command(command)


def test_research_campaign_active_array_linearized_update_throughput_probe_skips_diagnostic(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_active_array_linearized_update_throughput",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-active-array-linearized-update-throughput-probe",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-active-array-linearized-update-throughput-probe"
    assert "linearized_update_throughput" in " ".join(command.command)
    _assert_linearized_update_command(command, skip_residual_diagnostic=True)


def test_research_campaign_active_array_linearized_update_jvp_diag_gate_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_active_array_linearized_update_jvp_diag",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-active-array-linearized-update-jvp-diag-gate",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-active-array-linearized-update-jvp-diag-gate"
    _assert_linearized_update_command(command, preconditioner="jvp_diag")


def test_research_campaign_active_array_output_profile_is_gated(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py",
        "research_campaign_active_array_output",
    )
    reference_root = _make_dthe_reference_root(tmp_path)

    (command,) = module.build_campaign_commands(
        campaign_names=("dthe-active-array-output-jvp-profile",),
        python_executable="python",
        repo_root=_REPO,
        reference_root=reference_root,
        output_root=Path("/output"),
        fast_timeout_seconds=300,
    )

    assert command.name == "dthe-active-array-output-jvp-profile"
    assert command.required_reference_inputs == ("dthe",)
    assert command.requires_gpu is False
    assert command.timeout_seconds == 300
    assert "recycling_dthe_one_step" in command.command
    assert "runtime:recycling_transient_solver_mode=bdf_active_array_jvp" in command.command
    assert "recycling_transient_solver_mode=bdf_active_array_jvp" in command.command
    assert "bdf_jacobian_mode=jvp" in command.command
    assert "bdf_rhs_backend=active_array" in command.command
    assert "bdf_jvp_jacobian_gather_on_device=True" in command.command
    assert "bdf_jvp_jacobian_batch_count=1" in command.command
    assert "--rss-profile" in command.command
    assert "--skip-cprofile" in command.command


def test_research_campaign_gpu_bundle_requires_expected_dthe_reference_deck(
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_gpu_reference"
    )

    with pytest.raises(
        ValueError, match="tests/integrated/1D-recycling-dthe/data/BOUT.inp"
    ):
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
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_timeout"
    )
    command = module.CampaignCommand(
        name="demo", description="demo", command=("python", "-c", "pass")
    )

    def fake_run(*args, **kwargs):
        assert str(tmp_path / "src") in kwargs["env"]["PYTHONPATH"]
        assert kwargs["env"]["JAX_ENABLE_X64"] == "true"
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_campaign_command(command, cwd=tmp_path, timeout_seconds=1)

    assert result.timed_out is True
    assert result.returncode == 124
    assert result.timeout_seconds == 1


def test_research_campaign_command_runner_uses_per_campaign_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_command_timeout"
    )
    command = module.CampaignCommand(
        name="bounded-demo",
        description="demo",
        command=("python", "-c", "pass"),
        timeout_seconds=7,
    )

    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_campaign_command(command, cwd=tmp_path, timeout_seconds=99)

    assert result.timed_out is True
    assert result.returncode == 124
    assert result.timeout_seconds == 7


def test_research_campaign_gpu_command_runner_sets_cuda_prerequisite_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "scripts/run_research_campaign_bundle.py", "research_campaign_gpu_env"
    )
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
    module = _load_script_module(
        "scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_missing_input"
    )
    monkeypatch.delenv("JAX_DRB_REFERENCE_ROOT", raising=False)
    args = SimpleNamespace(
        reference_root=tmp_path / "empty-reference-root", input_path=None, case="dthe"
    )

    with pytest.raises(SystemExit) as excinfo:
        module._resolve_input(args)

    assert "tests/integrated/1D-recycling-dthe/data/BOUT.inp" in str(excinfo.value)
    assert "--input-path /path/to/BOUT.inp" in str(excinfo.value)


def test_batched_jvp_profiler_defaults_to_fixture_reference_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module(
        "scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_fixture_input"
    )
    monkeypatch.delenv("JAX_DRB_REFERENCE_ROOT", raising=False)
    args = SimpleNamespace(reference_root=None, input_path=None, case="dthe")

    assert (
        module._resolve_reference_root(args) == module.FIXTURE_REFERENCE_ROOT.resolve()
    )
    assert (
        module._resolve_input(args)
        == (
            module.FIXTURE_REFERENCE_ROOT
            / "tests"
            / "integrated"
            / "1D-recycling-dthe"
            / "data"
            / "BOUT.inp"
        ).resolve()
    )


def test_batched_jvp_profiler_prefers_env_reference_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_script_module(
        "scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_env_input"
    )
    root = tmp_path / "reference-root"
    input_path = root / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("# env deck\n", encoding="utf-8")
    monkeypatch.setenv("JAX_DRB_REFERENCE_ROOT", str(root))
    args = SimpleNamespace(reference_root=None, input_path=None, case="hydrogen")

    assert module._resolve_reference_root(args) == root.resolve()
    assert module._resolve_input(args) == input_path.resolve()


def test_batched_jvp_profiler_accepts_explicit_staged_input(tmp_path: Path) -> None:
    module = _load_script_module(
        "scripts/profile_recycling_batched_jvp_gate.py", "batched_jvp_staged_input"
    )
    input_path = tmp_path / "1D-recycling-dthe" / "data" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("# staged deck\n", encoding="utf-8")
    args = SimpleNamespace(reference_root=None, input_path=input_path, case="dthe")

    assert module._resolve_input(args) == input_path.resolve()


def test_jax_linearized_profiler_reports_lineax_solver_mode() -> None:
    module = _load_script_module(
        "scripts/profile_recycling_jax_linearized_gate.py",
        "jax_linearized_mode_mapping",
    )

    assert module._solver_mode_for_backend("jax_gmres") == "jax_linearized"
    assert module._solver_mode_for_backend("lineax_gmres") == "jax_linearized_lineax"
    assert (
        module._solver_mode_for_backend("jax_gmres", active_array_rhs=True)
        == "active_array_jax_linearized"
    )
    assert (
        module._solver_mode_for_backend("lineax_gmres", active_array_rhs=True)
        == "active_array_jax_linearized_lineax"
    )
    assert (
        module._solver_mode_for_backend(
            "jax_gmres", rhs_backend="promoted_active_sources"
        )
        == "promoted_active_sources_jax_linearized"
    )
    with pytest.raises(ValueError, match="promoted_active_sources"):
        module._solver_mode_for_backend(
            "lineax_gmres", rhs_backend="promoted_active_sources"
        )


def test_jax_linearized_profiler_jit_residual_appends_runtime_override() -> None:
    module = _load_script_module(
        "scripts/profile_recycling_jax_linearized_gate.py",
        "jax_linearized_jit_residual_override",
    )

    args = SimpleNamespace(
        override=["mesh:ny=64"],
        jit_residual=True,
        skip_initial_residual_check=True,
        gmres_solve_method="incremental",
    )

    assert module._effective_overrides(args) == [
        "mesh:ny=64",
        "runtime:recycling_jax_linear_jit_residual=true",
        "runtime:recycling_jax_linear_check_initial_residual=false",
        "runtime:recycling_jax_linear_gmres_solve_method=incremental",
    ]
