from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.cli as cli_module
import jax_drb.native as native_module
import jax_drb.parity.arrays as arrays_module
import jax_drb.parity.diff as diff_module
import jax_drb.parity.reference as reference_module
import jax_drb.validation as validation_module
from jax_drb.cli import (
    _analyze_alfven_wave_command,
    _analyze_drift_wave_command,
    _analyze_neutral_mixed_command,
    _compare_arrays_command,
    _compare_alfven_wave_command,
    _compare_blob2d_command,
    _compare_drift_wave_command,
    _compare_neutral_mixed_command,
    _compare_neutral_mixed_accepted_traces_command,
    _compare_recycling_command,
    _compare_summary_command,
    _default_command,
    _diagnose_neutral_mixed_substeps_command,
    _inspect_command,
    _parse_substep_csv,
    _normalize_cli_argv,
    _reference_cases_command,
    _run_case_command,
    _run_reference_case_command,
    _trace_neutral_mixed_accepted_steps_command,
    _trace_neutral_mixed_reference_accepted_steps_command,
    _validate_reference_baselines_command,
    main,
)
from jax_drb.parity.arrays import (
    build_portable_array_payload,
    write_portable_array_payload,
)


def _summary_payload() -> dict[str, object]:
    return {
        "case_name": "toy",
        "parity_mode": "one_rhs",
        "compare_variables": ["Ne"],
        "component_labels": ["e:evolve_density"],
        "dimensions": {"t": 1, "x": 3},
        "time_points": [0.0],
        "dataset_scalars": {"Nnorm": 1.0},
        "variable_summaries": {
            "Ne": {
                "name": "Ne",
                "dimensions": ["t", "x"],
                "shape": [1, 3],
                "minimum": 1.0,
                "maximum": 3.0,
                "mean": 2.0,
                "max_abs_delta_last_first": None,
            }
        },
    }


def test_default_command_and_argv_normalization_errors_are_explicit() -> None:
    assert _normalize_cli_argv([]) == []
    assert _normalize_cli_argv(["inspect", "case.inp"]) == ["inspect", "case.inp"]
    assert _normalize_cli_argv(["--help"]) == ["--help"]
    assert _normalize_cli_argv(["case.inp", "--dry-run"]) == [
        "run",
        "case.inp",
        "--dry-run",
    ]

    assert (
        _default_command(argparse.Namespace(subcommand="demo", command=lambda args: 42))
        == 42
    )
    with pytest.raises(SystemExit):
        _default_command(argparse.Namespace(subcommand=None))
    with pytest.raises(SystemExit):
        main([])

    assert _normalize_cli_argv(
        ["diagnose-neutral-mixed-substeps", "--substeps", "1,2"]
    ) == [
        "diagnose-neutral-mixed-substeps",
        "--substeps",
        "1,2",
    ]
    assert _normalize_cli_argv(
        ["trace-neutral-mixed-accepted-steps", "--json-out", "trace.json"]
    ) == [
        "trace-neutral-mixed-accepted-steps",
        "--json-out",
        "trace.json",
    ]
    assert _normalize_cli_argv(
        [
            "trace-neutral-mixed-reference-accepted-steps",
            "--reference-root",
            "reference",
            "--workdir",
            "work",
        ]
    ) == [
        "trace-neutral-mixed-reference-accepted-steps",
        "--reference-root",
        "reference",
        "--workdir",
        "work",
    ]
    assert _normalize_cli_argv(
        ["compare-neutral-mixed-accepted-traces", "native.json", "reference.jsonl"]
    ) == [
        "compare-neutral-mixed-accepted-traces",
        "native.json",
        "reference.jsonl",
    ]


def test_reference_cases_command_reports_missing_and_resolved_cases(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert _reference_cases_command(argparse.Namespace(reference_root=None)) == 1
    assert "set --reference-root" in capsys.readouterr().out

    resolved = (
        SimpleNamespace(
            exists=False,
            case=SimpleNamespace(
                name="missing_case",
                parity_mode="one_rhs",
                capability_tier="native_exact",
            ),
            input_path=tmp_path / "missing" / "BOUT.inp",
            run_config=None,
        ),
        SimpleNamespace(
            exists=True,
            case=SimpleNamespace(
                name="ready_case",
                parity_mode="one_step",
                capability_tier="native_operational",
            ),
            input_path=tmp_path / "ready" / "BOUT.inp",
            run_config=SimpleNamespace(
                time=SimpleNamespace(nout=2, timestep=5.0),
                components=(SimpleNamespace(label="e:evolve_density"),),
            ),
        ),
    )
    monkeypatch.setattr(
        cli_module, "resolve_reference_cases", lambda reference_root: resolved
    )

    assert _reference_cases_command(argparse.Namespace(reference_root=tmp_path)) == 0
    output = capsys.readouterr().out
    assert "missing_case: missing [native_exact]" in output
    assert "ready_case: one_step [native_operational]" in output
    assert "components=e:evolve_density" in output


def test_inspect_command_reports_resolved_normalization(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "normalized.inp"
    input_path.write_text(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 2
        ny = 2
        nz = 1
        dx = 1
        dy = 1
        dz = 1
        J = 1

        [model]
        components = e
        Nnorm = 1e19
        Tnorm = 100
        Bnorm = 2

        [e]
        type = evolve_density
        charge = -1
        AA = 1

        [Ne]
        function = 1
        """,
        encoding="utf-8",
    )

    assert _inspect_command(argparse.Namespace(input_file=input_path)) == 0
    output = capsys.readouterr().out
    assert "normalization:" in output
    assert "Nnorm=1e+19" in output


def test_compare_summary_and_array_commands_report_ok_and_mismatch(
    tmp_path: Path, capsys
) -> None:
    expected_json = tmp_path / "expected.json"
    actual_json = tmp_path / "actual.json"
    mismatch_json = tmp_path / "mismatch.json"
    payload = _summary_payload()
    expected_json.write_text(json.dumps(payload), encoding="utf-8")
    actual_json.write_text(json.dumps(payload), encoding="utf-8")
    mismatch_json.write_text(
        json.dumps({**payload, "case_name": "changed"}), encoding="utf-8"
    )

    args = argparse.Namespace(
        expected_json=expected_json,
        actual_json=actual_json,
        scalar_rtol=1e-10,
        scalar_atol=1e-12,
    )
    assert _compare_summary_command(args) == 0
    assert "comparison: ok" in capsys.readouterr().out
    assert (
        _compare_summary_command(
            argparse.Namespace(**{**vars(args), "actual_json": mismatch_json})
        )
        == 1
    )
    assert "case_name" in capsys.readouterr().out

    expected_npz = tmp_path / "expected.npz"
    actual_npz = tmp_path / "actual.npz"
    mismatch_npz = tmp_path / "mismatch.npz"
    array_payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 1, "x": 2},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0]])},
    )
    write_portable_array_payload(array_payload, expected_npz)
    write_portable_array_payload(array_payload, actual_npz)
    write_portable_array_payload(
        {**array_payload, "case_name": "changed"}, mismatch_npz
    )
    array_args = argparse.Namespace(
        expected_npz=expected_npz,
        actual_npz=actual_npz,
        scalar_rtol=1e-10,
        scalar_atol=1e-12,
        array_rtol=1e-10,
        array_atol=1e-12,
    )

    assert _compare_arrays_command(array_args) == 0
    assert "comparison: ok" in capsys.readouterr().out
    assert (
        _compare_arrays_command(
            argparse.Namespace(**{**vars(array_args), "actual_npz": mismatch_npz})
        )
        == 1
    )
    assert "case_name" in capsys.readouterr().out


def test_run_reference_case_command_writes_summary_and_arrays(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    case = SimpleNamespace(
        name="toy",
        trim_x_guards=True,
        trim_y_guards=True,
        input_path=lambda root: tmp_path / "BOUT.inp",
    )
    summary = SimpleNamespace(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        workdir=str(tmp_path / "work"),
        overrides=("nout=0",),
        time_points=(0.0,),
        compare_variables=("Ne",),
        variable_summaries={
            "Ne": SimpleNamespace(
                shape=(1, 2),
                minimum=1.0,
                maximum=2.0,
                mean=1.5,
                max_abs_delta_last_first=None,
            )
        },
        artifacts={"BOUT.dmp.0.nc": str(tmp_path / "BOUT.dmp.0.nc")},
        component_labels=("e:evolve_density",),
        nout=0,
        timestep=1.0,
    )
    monkeypatch.setattr(reference_module, "find_reference_case", lambda case_name: case)
    monkeypatch.setattr(
        reference_module,
        "run_reference_case",
        lambda *args, **kwargs: SimpleNamespace(summary=summary),
    )
    monkeypatch.setattr(
        reference_module, "write_case_baseline_json", lambda summary, path: Path(path)
    )
    monkeypatch.setattr(cli_module, "load_bout_input", lambda path: object())
    monkeypatch.setattr(
        cli_module.RunConfiguration,
        "from_config",
        lambda config: SimpleNamespace(mesh=SimpleNamespace(mxg=1, myg=2)),
    )
    monkeypatch.setattr(
        arrays_module,
        "build_dataset_array_payload",
        lambda *args, **kwargs: {"case_name": "toy"},
    )
    monkeypatch.setattr(
        arrays_module, "write_portable_array_payload", lambda payload, path: Path(path)
    )

    assert (
        _run_reference_case_command(
            argparse.Namespace(
                reference_root=None,
                reference_binary=None,
                case_name="toy",
                workdir=None,
                override=[],
                json_out=None,
                arrays_out=None,
            )
        )
        == 1
    )
    assert "set --reference-root" in capsys.readouterr().out

    exit_code = _run_reference_case_command(
        argparse.Namespace(
            reference_root=tmp_path,
            reference_binary=tmp_path / "binary",
            case_name="toy",
            workdir=tmp_path / "work",
            override=["nout=0"],
            json_out=tmp_path / "summary.json",
            arrays_out=tmp_path / "arrays.npz",
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "case: toy" in output
    assert "json_out:" in output
    assert "arrays_out:" in output


def test_run_case_and_validate_reference_commands_cover_success_and_errors(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    assert _run_case_command(argparse.Namespace(reference_root=None)) == 1
    assert "set --reference-root" in capsys.readouterr().out

    payload = {
        **_summary_payload(),
        "producer": "jax-drb",
        "capability_tier": "native_exact",
    }
    result = SimpleNamespace(
        payload=payload, variables={"Ne": np.array([[1.0, 2.0, 3.0]])}
    )
    monkeypatch.setattr(
        native_module, "run_curated_case", lambda *args, **kwargs: result
    )
    run_case_args = argparse.Namespace(
        reference_root=tmp_path,
        case_name="toy",
        override=["runtime:neutral_mixed_internal_substeps=4"],
        json_out=tmp_path / "native_summary.json",
        arrays_out=tmp_path / "native_arrays.npz",
    )
    assert _run_case_command(run_case_args) == 0
    output = capsys.readouterr().out
    assert "case: toy" in output
    assert "json_out:" in output
    assert "arrays_out:" in output

    assert (
        _validate_reference_baselines_command(
            argparse.Namespace(
                reference_root=None,
                reference_binary=None,
                case=[],
                baseline_dir=tmp_path,
            )
        )
        == 1
    )
    assert "set --reference-root" in capsys.readouterr().out
    monkeypatch.setattr(
        reference_module,
        "validate_reference_baselines",
        lambda **kwargs: (
            SimpleNamespace(case_name="ok_case", ok=True, issues=()),
            SimpleNamespace(case_name="bad_case", ok=False, issues=("Nnorm mismatch",)),
        ),
    )
    assert (
        _validate_reference_baselines_command(
            argparse.Namespace(
                reference_root=tmp_path,
                reference_binary=None,
                case=["bad_case"],
                baseline_dir=tmp_path,
            )
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "ok_case: ok" in output
    assert "bad_case: mismatch" in output
    assert "Nnorm mismatch" in output


def test_run_case_forwards_extra_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}
    payload = {
        **_summary_payload(),
        "producer": "jax-drb",
        "capability_tier": "native_exact",
    }

    def fake_run_curated_case(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            payload=payload, variables={"Ne": np.array([[1.0, 2.0, 3.0]])}
        )

    monkeypatch.setattr(native_module, "run_curated_case", fake_run_curated_case)

    assert (
        _run_case_command(
            argparse.Namespace(
                reference_root=tmp_path,
                case_name="toy",
                override=[
                    "runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp"
                ],
                json_out=None,
                arrays_out=None,
            )
        )
        == 0
    )

    assert captured["kwargs"]["extra_overrides"] == (
        "runtime:recycling_transient_solver_mode=bdf_fixed_full_field_jvp",
    )
    assert "case: toy" in capsys.readouterr().out


def test_diagnose_neutral_mixed_substeps_command_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    report = {
        "diagnostic": "neutral_mixed_substep_hybrid_state",
        "case_name": "neutral_mixed_one_step",
        "field": "NVh",
        "sweep_points": (
            {
                "internal_substeps": 1,
                "status": "ok",
                "final_field_error_register": {"fields": {"NVh": {"max_abs": 1.0e-3}}},
            },
            {
                "internal_substeps": 8,
                "status": "failed",
                "error_type": "NoConvergence",
                "error_message": "stalled",
            },
        ),
        "best": {
            "internal_substeps": 1,
            "metric": "NVh_final_max_abs",
            "value": 1.0e-3,
        },
    }
    monkeypatch.setattr(
        validation_module,
        "build_neutral_mixed_substep_hybrid_report",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(
        validation_module,
        "write_neutral_mixed_substep_hybrid_json",
        lambda report, path: Path(path),
    )

    assert _parse_substep_csv("1, 2,4") == (1, 2, 4)
    assert (
        _diagnose_neutral_mixed_substeps_command(
            argparse.Namespace(
                reference_root=tmp_path,
                case_name="neutral_mixed_one_step",
                input_path=None,
                reference_arrays_npz=None,
                substeps="1,8",
                json_out=tmp_path / "substeps.json",
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "substeps=1: ok" in output
    assert "substeps=8: failed" in output
    assert "json_out:" in output

    assert (
        _diagnose_neutral_mixed_substeps_command(
            argparse.Namespace(
                reference_root=tmp_path,
                case_name="neutral_mixed_one_step",
                input_path=None,
                reference_arrays_npz=None,
                substeps="0",
                json_out=None,
            )
        )
        == 1
    )
    assert "positive integers" in capsys.readouterr().out


def test_trace_neutral_mixed_accepted_steps_command_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    report = {
        "diagnostic": "neutral_mixed_native_accepted_step_trace",
        "case_name": "neutral_mixed_one_step",
        "trace_point_count": 9,
        "sample_y_indices": [0, 1, 2, 3, 10, 11, 12, 13],
    }
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(
        validation_module,
        "build_neutral_mixed_native_accepted_step_trace_report",
        fake_build,
    )
    monkeypatch.setattr(
        validation_module,
        "write_neutral_mixed_native_accepted_step_trace_json",
        lambda report, path: Path(path),
    )

    assert (
        _trace_neutral_mixed_accepted_steps_command(
            argparse.Namespace(
                reference_root=tmp_path,
                case_name="neutral_mixed_one_step",
                input_path=None,
                internal_substeps=8,
                steps=1,
                reference_trace_json=tmp_path / "reference_trace.jsonl",
                reference_stage="post_accepted",
                time_tolerance=1.0e-9,
                solver_mode="sparse",
                residual_tolerance=1.0e-11,
                step_tolerance=1.0e-12,
                max_nonlinear_iterations=12,
                linear_restart=10,
                linear_maxiter=100,
                linear_rtol=1.0e-10,
                json_out=tmp_path / "native_trace.json",
            )
        )
        == 0
    )
    assert captured["internal_substeps"] == 8
    assert captured["steps"] == 1
    assert captured["reference_trace_json"] == tmp_path / "reference_trace.jsonl"
    assert captured["reference_stage"] == "post_accepted"
    assert captured["time_tolerance"] == pytest.approx(1.0e-9)
    assert captured["solver_mode"] == "sparse"
    assert captured["residual_tolerance"] == pytest.approx(1.0e-11)
    assert captured["step_tolerance"] == pytest.approx(1.0e-12)
    assert captured["max_nonlinear_iterations"] == 12
    assert captured["linear_restart"] == 10
    assert captured["linear_maxiter"] == 100
    assert captured["linear_rtol"] == pytest.approx(1.0e-10)
    output = capsys.readouterr().out
    assert "neutral_mixed_native_accepted_step_trace" in output
    assert "trace_point_count: 9" in output
    assert "json_out:" in output

    assert (
        _trace_neutral_mixed_accepted_steps_command(
            argparse.Namespace(
                reference_root=tmp_path,
                case_name="neutral_mixed_one_step",
                input_path=None,
                internal_substeps=0,
                steps=1,
                json_out=tmp_path / "native_trace.json",
            )
        )
        == 1
    )
    assert "--internal-substeps must be positive" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"steps": 0}, "--steps must be positive"),
        ({"time_tolerance": 0.0}, "--time-tolerance must be positive"),
        ({"residual_tolerance": 0.0}, "--residual-tolerance must be positive"),
        ({"step_tolerance": 0.0}, "--step-tolerance must be positive"),
        (
            {"max_nonlinear_iterations": 0},
            "--max-nonlinear-iterations must be positive",
        ),
        ({"linear_restart": 0}, "--linear-restart must be positive"),
        ({"linear_maxiter": 0}, "--linear-maxiter must be positive"),
        ({"linear_rtol": 0.0}, "--linear-rtol must be positive"),
    ],
)
def test_trace_neutral_mixed_accepted_steps_rejects_invalid_solver_controls(
    tmp_path: Path,
    capsys,
    override: dict[str, object],
    message: str,
) -> None:
    args = {
        "reference_root": tmp_path,
        "case_name": "neutral_mixed_one_step",
        "input_path": None,
        "internal_substeps": 8,
        "steps": 1,
        "reference_trace_json": None,
        "reference_stage": "post_accepted",
        "time_tolerance": 1.0e-9,
        "solver_mode": "matrix_free",
        "residual_tolerance": 1.0e-8,
        "step_tolerance": 1.0e-10,
        "max_nonlinear_iterations": 8,
        "linear_restart": 20,
        "linear_maxiter": 200,
        "linear_rtol": 1.0e-8,
        "json_out": tmp_path / "native_trace.json",
    }
    args.update(override)

    assert _trace_neutral_mixed_accepted_steps_command(argparse.Namespace(**args)) == 1
    assert message in capsys.readouterr().out


def test_compare_neutral_mixed_accepted_traces_command_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    report = {
        "diagnostic": "neutral_mixed_accepted_step_trace_parity",
        "matched_trace_point_count": 2,
        "ranked_fields": [
            {
                "field": "NVh",
                "max_target_adjacent_delta": 4.0,
                "max_guard_delta": 1.0,
            }
        ],
    }
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(
        validation_module,
        "build_neutral_mixed_accepted_step_trace_parity_report",
        fake_build,
    )
    monkeypatch.setattr(
        validation_module,
        "write_neutral_mixed_accepted_step_trace_parity_json",
        lambda report, path: Path(path),
    )

    assert (
        _compare_neutral_mixed_accepted_traces_command(
            argparse.Namespace(
                native_trace_json=tmp_path / "native.json",
                reference_trace_json=tmp_path / "reference.jsonl",
                input_path=tmp_path / "BOUT.inp",
                reference_stage="post_accepted",
                time_tolerance=1.0e-8,
                reference_cvode_max_order=2,
                json_out=tmp_path / "trace_parity.json",
            )
        )
        == 0
    )
    assert captured["reference_stage"] == "post_accepted"
    assert captured["time_tolerance"] == 1.0e-8
    assert captured["reference_cvode_max_order"] == 2
    assert captured["input_path"] == tmp_path / "BOUT.inp"
    output = capsys.readouterr().out
    assert "neutral_mixed_accepted_step_trace_parity" in output
    assert "matched_trace_point_count: 2" in output
    assert "worst_field: NVh" in output

    assert (
        _compare_neutral_mixed_accepted_traces_command(
            argparse.Namespace(
                native_trace_json=tmp_path / "native.json",
                reference_trace_json=tmp_path / "reference.jsonl",
                input_path=None,
                reference_stage="post_accepted",
                time_tolerance=0.0,
                reference_cvode_max_order=None,
                json_out=tmp_path / "trace_parity.json",
            )
        )
        == 1
    )
    assert "--time-tolerance must be positive" in capsys.readouterr().out


def test_trace_neutral_mixed_reference_accepted_steps_command_writes_jsonl(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "reference_trace.jsonl"

    monkeypatch.setattr(
        validation_module,
        "run_neutral_mixed_hermes_accepted_step_trace",
        fake_run,
    )

    assert (
        _trace_neutral_mixed_reference_accepted_steps_command(
            argparse.Namespace(
                reference_root=tmp_path / "reference",
                workdir=tmp_path / "work",
                hermes_binary=tmp_path / "hermes-3",
                trace_out=tmp_path / "reference_trace.jsonl",
                species="h",
                cvode_max_order=2,
                timeout_seconds=30.0,
            )
        )
        == 0
    )
    assert captured["species"] == "h"
    assert captured["trace_jsonl_path"] == tmp_path / "reference_trace.jsonl"
    assert captured["cvode_max_order"] == 2
    output = capsys.readouterr().out
    assert "neutral_mixed_reference_accepted_step_trace" in output
    assert "cvode_max_order: 2" in output
    assert "trace_jsonl:" in output

    assert (
        _trace_neutral_mixed_reference_accepted_steps_command(
            argparse.Namespace(
                reference_root=tmp_path / "reference",
                workdir=tmp_path / "work",
                hermes_binary=None,
                trace_out=None,
                species="h",
                cvode_max_order=None,
                timeout_seconds=0.0,
            )
        )
        == 1
    )
    assert "--timeout-seconds must be positive" in capsys.readouterr().out


def test_trace_neutral_mixed_reference_accepted_steps_rejects_bad_cvode_order(
    tmp_path: Path, capsys
) -> None:
    assert (
        _trace_neutral_mixed_reference_accepted_steps_command(
            argparse.Namespace(
                reference_root=tmp_path / "reference",
                workdir=tmp_path / "work",
                hermes_binary=None,
                trace_out=None,
                species="h",
                cvode_max_order=0,
                timeout_seconds=30.0,
            )
        )
        == 1
    )
    assert "--cvode-max-order must be positive" in capsys.readouterr().out


def test_compare_recycling_command_uses_formatted_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        diff_module,
        "compare_recycling_artifacts",
        lambda *args, **kwargs: SimpleNamespace(ok=False),
    )
    monkeypatch.setattr(
        diff_module, "format_recycling_diff_report", lambda result: "formatted diff"
    )

    exit_code = _compare_recycling_command(
        argparse.Namespace(
            expected_artifact=Path("expected.json"),
            actual_artifact=Path("actual.json"),
            artifact_kind="auto",
            scalar_rtol=1e-10,
            scalar_atol=1e-12,
            array_rtol=1e-10,
            array_atol=1e-12,
        )
    )

    assert exit_code == 1
    assert capsys.readouterr().out == "formatted diff\n"


def test_wave_and_blob_cli_analysis_commands_write_optional_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    drift = SimpleNamespace(
        density_variable="Ni",
        trace_x_index=0,
        trace_y_index=1,
        fit_points=4,
        benchmark=SimpleNamespace(
            wstar=1.0,
            sigmapar=2.0,
            sigmapar_over_wstar=2.0,
            analytic_gamma_over_wstar=0.1,
            analytic_omega_over_wstar=0.2,
        ),
        measured_gamma_over_wstar=0.11,
        measured_omega_over_wstar=0.21,
    )
    drift_parity = SimpleNamespace(
        expected=drift,
        actual=SimpleNamespace(
            measured_gamma_over_wstar=0.12, measured_omega_over_wstar=0.22
        ),
        variable_errors={"Ni": SimpleNamespace(max_abs_error=1.0e-3, rms_error=2.0e-4)},
    )
    blob_parity = SimpleNamespace(
        expected=SimpleNamespace(density_variable="Ne", background_density=1.0),
        peak_max_abs_error=1.0e-3,
        peak_rms_error=2.0e-4,
        center_of_mass_x_max_abs_error=3.0e-3,
        center_of_mass_z_max_abs_error=4.0e-3,
    )
    monkeypatch.setattr(
        validation_module,
        "compare_drift_wave_npz",
        lambda *args, **kwargs: drift_parity,
    )
    monkeypatch.setattr(
        validation_module,
        "write_drift_wave_parity_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_drift_wave_parity_plot",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "compare_blob2d_artifacts",
        lambda *args, **kwargs: blob_parity,
    )
    monkeypatch.setattr(
        validation_module, "write_blob2d_parity_json", lambda result, path: Path(path)
    )
    monkeypatch.setattr(
        validation_module, "save_blob2d_parity_plot", lambda result, path: Path(path)
    )

    assert (
        _compare_drift_wave_command(
            argparse.Namespace(
                input_file=tmp_path / "BOUT.inp",
                expected_npz=tmp_path / "expected.npz",
                actual_npz=tmp_path / "actual.npz",
                density_variable="Ni",
                x_index=0,
                y_index=1,
                fit_points=4,
                json_out=tmp_path / "drift.json",
                plot_out=tmp_path / "drift.png",
            )
        )
        == 0
    )
    assert (
        _compare_blob2d_command(
            argparse.Namespace(
                expected_artifact=tmp_path / "expected.json",
                actual_artifact=tmp_path / "actual.json",
                density_variable="Ne",
                background_density=1.0,
                json_out=tmp_path / "blob.json",
                plot_out=tmp_path / "blob.png",
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "expected_gamma_over_wstar" in output
    assert "center_of_mass_z_max_abs_error" in output


def test_drift_and_alfven_cli_analysis_commands_write_optional_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    drift = SimpleNamespace(
        density_variable="Ni",
        trace_x_index=0,
        trace_y_index=1,
        fit_points=4,
        benchmark=SimpleNamespace(
            wstar=1.0,
            sigmapar=2.0,
            sigmapar_over_wstar=2.0,
            analytic_gamma_over_wstar=0.1,
            analytic_omega_over_wstar=0.2,
        ),
        measured_gamma_over_wstar=0.11,
        measured_omega_over_wstar=0.21,
    )
    alfven = SimpleNamespace(
        field_variable="phi",
        x_index=2,
        benchmark=SimpleNamespace(
            kpar=1.0,
            kperp=2.0,
            analytic_phase_speed=3.0,
            analytic_omega=4.0,
        ),
        measured_phase_speed=3.1,
        measured_omega=4.1,
        relative_phase_speed_error=0.033,
    )
    alfven_parity = SimpleNamespace(
        expected=alfven,
        actual=SimpleNamespace(measured_phase_speed=3.2, measured_omega=4.2),
        phase_speed_error=0.1,
        omega_error=0.2,
        mean_square_max_abs_error=1.0e-3,
        mean_square_rms_error=2.0e-4,
    )
    monkeypatch.setattr(
        validation_module, "analyze_drift_wave_npz", lambda *args, **kwargs: drift
    )
    monkeypatch.setattr(
        validation_module,
        "write_drift_wave_analysis_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_drift_wave_diagnostic_plot",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module, "analyze_alfven_wave_npz", lambda *args, **kwargs: alfven
    )
    monkeypatch.setattr(
        validation_module,
        "write_alfven_wave_analysis_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_alfven_wave_diagnostic_plot",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "compare_alfven_wave_npz",
        lambda *args, **kwargs: alfven_parity,
    )
    monkeypatch.setattr(
        validation_module,
        "write_alfven_wave_parity_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_alfven_wave_parity_plot",
        lambda result, path: Path(path),
    )

    assert (
        _analyze_drift_wave_command(
            argparse.Namespace(
                input_file=tmp_path / "BOUT.inp",
                arrays_npz=tmp_path / "arrays.npz",
                density_variable="Ni",
                x_index=0,
                y_index=1,
                fit_points=4,
                json_out=tmp_path / "drift.json",
                plot_out=tmp_path / "drift.png",
            )
        )
        == 0
    )
    assert (
        _analyze_alfven_wave_command(
            argparse.Namespace(
                input_file=tmp_path / "BOUT.inp",
                arrays_npz=tmp_path / "arrays.npz",
                field_variable="phi",
                x_index=2,
                json_out=tmp_path / "alfven.json",
                plot_out=tmp_path / "alfven.png",
            )
        )
        == 0
    )
    assert (
        _compare_alfven_wave_command(
            argparse.Namespace(
                input_file=tmp_path / "BOUT.inp",
                expected_npz=tmp_path / "expected.npz",
                actual_npz=tmp_path / "actual.npz",
                field_variable="phi",
                x_index=2,
                json_out=tmp_path / "alfven_parity.json",
                plot_out=tmp_path / "alfven_parity.png",
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "measured_gamma_over_wstar" in output
    assert "relative_phase_speed_error" in output
    assert "mean_square_rms_error" in output


def test_neutral_mixed_cli_analysis_and_compare_commands_write_optional_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    analysis = SimpleNamespace(
        density_variable="Nh",
        pressure_variable="Ph",
        momentum_variable="NVh",
        center_index_x=1,
        center_index_y=2,
        center_index_z=0,
        center_density_history=(1.0, 1.1),
        center_pressure_history=(2.0, 2.1),
        center_momentum_history=(3.0, 3.1),
        center_temperature_history=(4.0, 4.1),
        total_density_history=(5.0, 5.1),
        total_pressure_history=(6.0, 6.1),
        momentum_rms_history=(7.0, 7.1),
    )
    parity = SimpleNamespace(
        expected=analysis,
        series_errors={
            "density": SimpleNamespace(max_abs_error=1.0e-3, rms_error=2.0e-4),
            "pressure": SimpleNamespace(max_abs_error=3.0e-3, rms_error=4.0e-4),
        },
    )
    monkeypatch.setattr(
        validation_module, "analyze_neutral_mixed_npz", lambda *args, **kwargs: analysis
    )
    monkeypatch.setattr(
        validation_module,
        "write_neutral_mixed_analysis_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_neutral_mixed_diagnostic_plot",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "compare_neutral_mixed_artifacts",
        lambda *args, **kwargs: parity,
    )
    monkeypatch.setattr(
        validation_module,
        "write_neutral_mixed_parity_json",
        lambda result, path: Path(path),
    )
    monkeypatch.setattr(
        validation_module,
        "save_neutral_mixed_parity_plot",
        lambda result, path: Path(path),
    )

    assert (
        _analyze_neutral_mixed_command(
            argparse.Namespace(
                arrays_npz=tmp_path / "arrays.npz",
                density_variable="Nh",
                pressure_variable="Ph",
                momentum_variable="NVh",
                x_index=1,
                y_index=2,
                z_index=0,
                json_out=tmp_path / "neutral.json",
                plot_out=tmp_path / "neutral.png",
            )
        )
        == 0
    )
    assert (
        _compare_neutral_mixed_command(
            argparse.Namespace(
                expected_artifact=tmp_path / "expected.json",
                actual_artifact=tmp_path / "actual.json",
                density_variable="Nh",
                pressure_variable="Ph",
                momentum_variable="NVh",
                x_index=1,
                y_index=2,
                z_index=0,
                json_out=tmp_path / "neutral_parity.json",
                plot_out=tmp_path / "neutral_parity.png",
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "center_temperature_final" in output
    assert "pressure: max_abs_error" in output
