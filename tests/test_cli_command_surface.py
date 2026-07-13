from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax_drb.cli as cli_module
import jax_drb.native as native_module
import jax_drb.parity.diff as diff_module
from jax_drb.cli import (
    _compare_arrays_command,
    _compare_recycling_command,
    _compare_summary_command,
    _default_command,
    _inspect_command,
    _normalize_cli_argv,
    _reference_cases_command,
    _run_case_command,
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

    assert _normalize_cli_argv(["run-case", "toy"]) == ["run-case", "toy"]
    assert _normalize_cli_argv(["compare-summary", "a.json", "b.json"]) == [
        "compare-summary",
        "a.json",
        "b.json",
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


def test_run_case_command_covers_success_and_errors(
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
