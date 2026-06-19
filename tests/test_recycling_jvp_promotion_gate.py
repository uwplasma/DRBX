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
    assert "--require-fixed-bdf2-linear-preconditioner" not in command
    assert "--require-bdf-pairwise-max" not in command
    assert command[command.index("--timestep") + 1] == "10"
    assert command.count("--field") == 3


def test_recycling_jvp_promotion_gate_builds_preconditioned_fixed_bdf2_command() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]
    command = module._build_case_command(
        gate_case,
        reference_root=Path("/tmp/reference-root"),
        python_executable="python",
        gate_phase="fixed_bdf2",
        fixed_bdf2_timestep=10.0,
        fixed_bdf2_linear_preconditioner="local_block_diag",
        fixed_bdf2_linear_preconditioner_refresh=100,
        fixed_bdf2_linear_restart=8,
        fixed_bdf2_linear_maxiter=3,
        fixed_bdf2_linear_tolerance_factor=25.0,
        fixed_bdf2_jit_linear_operator=True,
        fixed_bdf2_max_linear_iterations=3200,
        fixed_bdf2_max_linear_operator_calls=128,
        fixed_bdf2_max_preconditioner_builds=2,
        fixed_bdf2_max_preconditioner_applies=40,
    )

    overrides = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--override"
    ]
    assert (
        "runtime:recycling_jax_linear_preconditioner=local_block_diag"
        in overrides
    )
    assert (
        "runtime:recycling_jax_linear_preconditioner_refresh=100"
        in overrides
    )
    assert "runtime:recycling_jax_linear_restart=8" in overrides
    assert "runtime:recycling_jax_linear_maxiter=3" in overrides
    assert "runtime:recycling_jax_linear_tolerance_factor=25" in overrides
    assert "runtime:recycling_jax_linear_jit_linear_operator=true" in overrides
    assert "--require-fixed-bdf2-linear-operator-jitted" in command
    assert command[
        command.index("--require-fixed-bdf2-linear-preconditioner") + 1
    ] == "local_block_diag"
    assert command[
        command.index("--require-fixed-bdf2-max-linear-iterations") + 1
    ] == "3200"
    assert command[
        command.index("--require-fixed-bdf2-max-linear-operator-calls") + 1
    ] == "128"
    assert command[
        command.index("--require-fixed-bdf2-max-preconditioner-builds") + 1
    ] == "2"
    assert command[
        command.index("--require-fixed-bdf2-max-preconditioner-applies") + 1
    ] == "40"


def test_recycling_jvp_promotion_gate_rejects_invalid_preconditioner_refresh() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]

    try:
        module._build_case_command(
            gate_case,
            reference_root=Path("/tmp/reference-root"),
            python_executable="python",
            gate_phase="fixed_bdf2",
            fixed_bdf2_timestep=10.0,
            fixed_bdf2_linear_preconditioner="local_block_diag",
            fixed_bdf2_linear_preconditioner_refresh=0,
        )
    except ValueError as exc:
        assert "fixed_bdf2_linear_preconditioner_refresh must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_recycling_jvp_promotion_gate_rejects_invalid_performance_budgets() -> None:
    module = _load_module()
    gate_case = module.GATE_CASES["recycling_1d_one_step"]

    for kwargs, expected in (
        (
            {"fixed_bdf2_max_linear_iterations": -1},
            "fixed_bdf2_max_linear_iterations must be nonnegative",
        ),
        (
            {"fixed_bdf2_max_linear_operator_calls": -1},
            "fixed_bdf2_max_linear_operator_calls must be nonnegative",
        ),
        (
            {"fixed_bdf2_max_preconditioner_builds": -1},
            "fixed_bdf2_max_preconditioner_builds must be nonnegative",
        ),
        (
            {"fixed_bdf2_max_preconditioner_applies": -1},
            "fixed_bdf2_max_preconditioner_applies must be nonnegative",
        ),
        (
            {"fixed_bdf2_linear_restart": 0},
            "fixed_bdf2_linear_restart must be positive",
        ),
        (
            {"fixed_bdf2_linear_maxiter": 0},
            "fixed_bdf2_linear_maxiter must be positive",
        ),
        (
            {"fixed_bdf2_linear_tolerance_factor": 0.0},
            "fixed_bdf2_linear_tolerance_factor must be positive",
        ),
    ):
        try:
            module._build_case_command(
                gate_case,
                reference_root=Path("/tmp/reference-root"),
                python_executable="python",
                gate_phase="fixed_bdf2",
                fixed_bdf2_timestep=10.0,
                **kwargs,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")


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


def test_recycling_jvp_promotion_gate_can_run_fixed_bdf2_only_dry_run(
    tmp_path: Path,
) -> None:
    module = _load_module()
    reference_root = module.FIXTURE_REFERENCE_ROOT
    output_dir = tmp_path / "fixed_bdf2_only"

    exit_code = module.main(
        [
            "--reference-root",
            str(reference_root),
            "--case",
            "recycling_1d_one_step",
            "--dry-run",
            "--fixed-bdf2-only",
            "--fixed-bdf2-linear-preconditioner",
            "neutral_line",
            "--fixed-bdf2-linear-preconditioner-refresh",
            "100",
            "--fixed-bdf2-linear-restart",
            "8",
            "--fixed-bdf2-linear-maxiter",
            "3",
            "--fixed-bdf2-linear-tolerance-factor",
            "25",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    case_reports = summary["case_reports"]
    assert len(case_reports) == 1
    report = case_reports[0]
    assert report["phase"] == "fixed_bdf2"
    assert report["fixed_bdf2_only"] is True
    assert report["fixed_bdf2_linear_preconditioner"] == "neutral_line"
    assert report["fixed_bdf2_linear_preconditioner_refresh"] == 100
    assert report["fixed_bdf2_linear_restart"] == 8
    assert report["fixed_bdf2_linear_maxiter"] == 3
    assert report["fixed_bdf2_linear_tolerance_factor"] == 25.0
    assert report["output_json"].endswith("recycling_1d_one_step.fixed_bdf2.json")
    assert "--require-fixed-jvp-diagnostics" not in report["command"]
    assert "--require-fixed-bdf2-diagnostics" in report["command"]
    assert "--require-fixed-bdf2-linear-preconditioner" in report["command"]
    overrides = [
        report["command"][index + 1]
        for index, item in enumerate(report["command"][:-1])
        if item == "--override"
    ]
    assert "runtime:recycling_jax_linear_restart=8" in overrides
    assert "runtime:recycling_jax_linear_maxiter=3" in overrides
    assert "runtime:recycling_jax_linear_tolerance_factor=25" in overrides


def test_recycling_jvp_promotion_gate_rejects_active_jvp_with_fixed_bdf2_only(
    tmp_path: Path,
) -> None:
    module = _load_module()

    try:
        module.main(
            [
                "--reference-root",
                str(module.FIXTURE_REFERENCE_ROOT),
                "--dry-run",
                "--fixed-bdf2-only",
                "--include-active-array-jvp",
                "--output-dir",
                str(tmp_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit from incompatible options")
