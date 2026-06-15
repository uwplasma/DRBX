from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_recycling_jvp_promotion_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_recycling_jvp_promotion_gate", script_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recycling_jvp_promotion_gate_builds_single_ion_bdf_jvp_command() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
    )

    assert command[:2] == [
        "python",
        str(module.REPO_ROOT / "scripts" / "compare_recycling_transient_modes.py"),
    ]
    assert command[command.index("--case") + 1] == "recycling_1d_one_step"
    assert command[command.index("--reference-root") + 1] == "/tmp/reference-root"
    assert command.count("--mode") == 2
    assert "bdf" in command
    assert "bdf_fixed_full_field_jvp" in command
    assert "bdf_active_array_jvp" not in command
    assert "--diagnostics-only" in command
    assert "--require-fixed-jvp-diagnostics" in command
    assert "--require-fixed-bdf2-diagnostics" not in command
    assert command[command.index("--require-bdf-pairwise-max") + 1] == "1.00000000e-05"
    assert command[command.index("--mode-timeout-seconds") + 1] == "300"
    assert command[command.index("--steps") + 1] == "2"
    assert command.count("--field") == 3
    assert "Pe" in command
    assert "Nd+" in command
    assert "Pd+" in command


def test_recycling_jvp_promotion_gate_builds_bounded_fixed_bdf2_command() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        gate_phase="fixed_bdf2",
        fixed_bdf2_timestep=10.0,
    )

    modes = [
        command[index + 1]
        for index, item in enumerate(command)
        if item == "--mode"
    ]
    assert modes == [
        "fixed_bdf2_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized",
    ]
    assert "--require-fixed-jvp-diagnostics" not in command
    assert "--require-fixed-bdf2-diagnostics" in command
    assert "--require-bdf-pairwise-max" not in command
    assert command[command.index("--timestep") + 1] == "10"
    assert command.count("--field") == 3


def test_recycling_jvp_promotion_gate_has_bounded_dthe_fixed_bdf2_phase() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_dthe_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        gate_phase="fixed_bdf2",
        fixed_bdf2_timestep=gate_case.fixed_bdf2_timestep,
    )

    assert gate_case.fixed_bdf2_timestep == 1.0
    assert gate_case.fixed_bdf2_max_internal_timestep == 0.5
    assert command[command.index("--timestep") + 1] == "1"
    assert command[command.index("--override") + 1] == (
        "runtime:recycling_fixed_bdf2_max_internal_timestep=0.5"
    )
    assert "fixed_bdf2_jax_linearized" in command
    assert "fixed_bdf2_active_array_jax_linearized" in command
    assert command.count("--field") == 4


def test_recycling_jvp_promotion_gate_can_opt_into_active_array_jvp() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        include_active_array_jvp=True,
    )

    assert command.count("--mode") == 3
    assert "bdf_active_array_jvp" in command


def test_recycling_jvp_promotion_gate_can_override_mode_timeout() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        mode_timeout_seconds=45.0,
    )

    assert command[command.index("--mode-timeout-seconds") + 1] == "45"


def test_recycling_jvp_promotion_gate_builds_command_with_json_report() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_dthe_one_step"]
    output_json = Path("/tmp/recycling_dthe_one_step.json")

    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        output_json=output_json,
    )

    assert command[command.index("--output-json") + 1] == str(output_json)
    assert command[command.index("--require-bdf-pairwise-max") + 1] == "2.00000000e-05"
    assert command[command.index("--mode-timeout-seconds") + 1] == "600"
    assert command[command.index("--steps") + 1] == "2"


def test_recycling_jvp_promotion_gate_defaults_to_all_cases() -> None:
    module = _load_module()

    assert tuple(gate.case for gate in module._selected_cases(())) == (
        "recycling_1d_one_step",
        "recycling_dthe_one_step",
    )
    assert tuple(
        gate.case for gate in module._selected_cases(("recycling_dthe_one_step",))
    ) == ("recycling_dthe_one_step",)


def test_recycling_jvp_promotion_gate_writes_dry_run_summary(tmp_path: Path) -> None:
    module = _load_module()
    reference_root = module.FIXTURE_REFERENCE_ROOT
    output_dir = tmp_path / "promotion_gate"

    exit_code = module.main(
        [
            "--reference-root",
            str(reference_root),
            "--case",
            "recycling_1d_one_step",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--mode-timeout-seconds",
            "12.5",
        ]
    )

    assert exit_code == 0
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["all_cases_passed"] is True
    assert summary["reference_root"] == str(reference_root.resolve())
    case_reports = summary["case_reports"]
    assert len(case_reports) == 2
    assert [report["phase"] for report in case_reports] == ["bdf_jvp", "fixed_bdf2"]
    assert all(report["case"] == "recycling_1d_one_step" for report in case_reports)
    assert all(report["returncode"] == 0 for report in case_reports)
    assert case_reports[0]["output_json"].endswith(
        "recycling_1d_one_step.bdf_jvp.json"
    )
    assert case_reports[1]["output_json"].endswith(
        "recycling_1d_one_step.fixed_bdf2.json"
    )
    assert case_reports[1]["fixed_bdf2_timestep"] == 10.0
    assert all(report["mode_timeout_seconds"] == 12.5 for report in case_reports)
    assert all("--output-json" in report["command"] for report in case_reports)
