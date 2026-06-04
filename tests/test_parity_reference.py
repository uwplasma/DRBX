from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import jax_drb.parity.reference as reference_module
from jax_drb.parity.reference import (
    _assert_artifacts_exist,
    _prepare_workdir,
    _reference_command,
    _run_reference_binary,
    _stage_case_artifacts,
    _stage_mesh_file,
    _summarize_run,
    _summarize_variable,
    build_case_baseline_payload,
    discover_reference_binary,
    find_reference_case,
    _summarize_dataset,
    merge_overrides,
    make_default_overrides,
    resolve_reference_case,
    run_reference_case,
    validate_reference_baselines,
    write_case_baseline_json,
    write_run_summary_json,
)
from jax_drb.reference.cases import ReferenceCase, load_reference_cases
from jax_drb.parity.reference import ReferenceExecutionResult, ReferenceRunSummary, VariableSummary


def _write_reference_manifest(path: Path, *, case_name: str = "toy_rhs") -> Path:
    path.write_text(
        f"""
        [[case]]
        name = "{case_name}"
        stage = "unit"
        reference_path = "cases/toy/BOUT.inp"
        parity_mode = "one_rhs"
        rationale = "Unit-test reference case."
        compare_variables = ["Ne"]
        extra_overrides = ["mesh:file={{reference_root}}/mesh.nc"]
        """,
        encoding="utf-8",
    )
    return path


def _toy_case(**overrides: object) -> ReferenceCase:
    values: dict[str, object] = {
        "name": "toy_rhs",
        "stage": "unit",
        "reference_path": "cases/toy/BOUT.inp",
        "parity_mode": "one_rhs",
        "rationale": "Unit-test reference case.",
        "compare_variables": ("Ne",),
    }
    values.update(overrides)
    return ReferenceCase(**values)


def _toy_summary(**overrides: object) -> ReferenceRunSummary:
    values: dict[str, object] = {
        "case_name": "toy_rhs",
        "parity_mode": "one_rhs",
        "capability_tier": "native_exact",
        "reference_binary": "/tmp/reference-binary",
        "overrides": ("nout=0",),
        "workdir": "/tmp/run",
        "artifacts": {"BOUT.dmp.0.nc": "/tmp/run/BOUT.dmp.0.nc"},
        "dimensions": {"t": 1, "x": 3},
        "time_points": (0.0,),
        "dataset_scalars": {"Nnorm": 1.0},
        "compare_variables": ("Ne",),
        "variable_summaries": {
            "Ne": VariableSummary(
                name="Ne",
                dimensions=("t", "x"),
                shape=(1, 3),
                minimum=1.0,
                maximum=3.0,
                mean=2.0,
                max_abs_delta_last_first=None,
            )
        },
        "component_labels": ("e:evolve_density",),
        "nout": 0,
        "timestep": 1.0,
    }
    values.update(overrides)
    return ReferenceRunSummary(**values)


def test_reference_cases_support_optional_compare_variables() -> None:
    cases = load_reference_cases()
    evolve_density = next(case for case in cases if case.name == "evolve_density_rhs")
    assert evolve_density.compare_variables == ("Ne",)


def test_discover_reference_binary_accepts_explicit_env_and_build_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit-reference"
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")
    assert discover_reference_binary(reference_binary=explicit) == explicit.resolve()

    env_binary = tmp_path / "env-reference"
    env_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("JAX_DRB_REFERENCE_BINARY", os.fspath(env_binary))
    assert discover_reference_binary() == env_binary.resolve()
    monkeypatch.delenv("JAX_DRB_REFERENCE_BINARY")

    reference_root = tmp_path / "hermes-3"
    build_binary = reference_root / "build" / "hermes-3"
    build_binary.parent.mkdir(parents=True)
    build_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    assert discover_reference_binary(reference_root=reference_root) == build_binary


def test_discover_reference_binary_reports_missing_locations(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Reference binary not found"):
        discover_reference_binary(reference_binary=tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Could not discover"):
        discover_reference_binary(reference_root=tmp_path / "empty-reference-root")


def test_find_and_resolve_reference_cases_report_bad_inputs(tmp_path: Path) -> None:
    manifest = _write_reference_manifest(tmp_path / "manifest.toml")
    case = find_reference_case("toy_rhs", manifest_path=manifest)
    assert case.compare_variables == ("Ne",)

    with pytest.raises(KeyError, match="Unknown reference case"):
        find_reference_case("not-a-case", manifest_path=manifest)

    reference_root = tmp_path / "reference-root"
    with pytest.raises(FileNotFoundError, match="Reference case input not found"):
        resolve_reference_case("toy_rhs", reference_root=reference_root, manifest_path=manifest)

    input_path = reference_root / "cases" / "toy" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("nout = 0\n", encoding="utf-8")
    resolved_case, resolved_input = resolve_reference_case("toy_rhs", reference_root=reference_root, manifest_path=manifest)
    assert resolved_case.name == "toy_rhs"
    assert resolved_input == input_path


def test_make_default_overrides_respects_fast_parity_modes() -> None:
    assert make_default_overrides("one_rhs") == ("nout=0",)
    assert make_default_overrides("one_step") == ("nout=1",)
    assert make_default_overrides("short_window") == ()


def test_merge_overrides_replaces_duplicate_keys_with_latest_value() -> None:
    assert merge_overrides(("nout=0", "i:diagnose=false"), ("nout=1",), ("i:diagnose=true",)) == (
        "nout=1",
        "i:diagnose=true",
    )


def test_run_reference_case_sanitizes_and_removes_ephemeral_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _toy_case(process_count=3)
    input_path = tmp_path / "reference-root" / "cases" / "toy" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("nout = 0\n", encoding="utf-8")
    binary = tmp_path / "reference-binary"
    staged_workdir = tmp_path / "staged-run"
    called: dict[str, object] = {}

    monkeypatch.setattr(reference_module, "resolve_reference_case", lambda *args, **kwargs: (case, input_path))
    monkeypatch.setattr(reference_module, "discover_reference_binary", lambda *args, **kwargs: binary)

    def fake_prepare_workdir(*args: object, **kwargs: object) -> Path:
        staged_workdir.mkdir()
        return staged_workdir

    def fake_run_reference_binary(**kwargs: object) -> None:
        called["run_kwargs"] = kwargs
        Path(kwargs["stdout_path"]).write_text("ok\n", encoding="utf-8")

    def fake_summarize_run(**kwargs: object) -> ReferenceRunSummary:
        return _toy_summary(workdir=os.fspath(staged_workdir), overrides=kwargs["overrides"])

    monkeypatch.setattr(reference_module, "_prepare_workdir", fake_prepare_workdir)
    monkeypatch.setattr(reference_module, "_run_reference_binary", fake_run_reference_binary)
    monkeypatch.setattr(reference_module, "_summarize_run", fake_summarize_run)

    result = run_reference_case(
        "toy_rhs",
        reference_root=tmp_path / "reference-root",
        extra_overrides=("nout=2", "mesh:file={reference_root}/mesh.nc"),
        keep_workdir=False,
    )

    assert result.summary.workdir == os.fspath(staged_workdir)
    assert result.summary.overrides == ("nout=2", f"mesh:file={tmp_path / 'reference-root'}/mesh.nc")
    assert Path(result.stdout_path) == staged_workdir / "run.stdout"
    assert called["run_kwargs"]["process_count"] == 3
    assert not staged_workdir.exists()


def test_run_reference_case_removes_ephemeral_workdir_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _toy_case()
    input_path = tmp_path / "reference-root" / "cases" / "toy" / "BOUT.inp"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("nout = 0\n", encoding="utf-8")
    staged_workdir = tmp_path / "staged-run"

    monkeypatch.setattr(reference_module, "resolve_reference_case", lambda *args, **kwargs: (case, input_path))
    monkeypatch.setattr(reference_module, "discover_reference_binary", lambda *args, **kwargs: tmp_path / "binary")

    def fake_prepare_workdir(*args: object, **kwargs: object) -> Path:
        staged_workdir.mkdir()
        return staged_workdir

    def failing_run_reference_binary(**kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(reference_module, "_prepare_workdir", fake_prepare_workdir)
    monkeypatch.setattr(reference_module, "_run_reference_binary", failing_run_reference_binary)

    with pytest.raises(RuntimeError, match="boom"):
        run_reference_case("toy_rhs", reference_root=tmp_path / "reference-root", keep_workdir=False)

    assert not staged_workdir.exists()


def test_validate_reference_baselines_reports_missing_and_mismatched_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_reference_manifest(tmp_path / "manifest.toml")
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()

    missing = validate_reference_baselines(
        reference_root=tmp_path / "reference-root",
        manifest_path=manifest,
        case_names=("toy_rhs",),
        baseline_dir=baseline_dir,
    )
    assert missing == (
        reference_module.ReferenceSmokeResult(
            case_name="toy_rhs",
            ok=False,
            issues=(f"Missing committed baseline JSON: {baseline_dir / 'toy_rhs.json'}",),
        ),
    )

    baseline_payload = build_case_baseline_payload(_toy_summary())
    (baseline_dir / "toy_rhs.json").write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")

    def fake_run_reference_case(*args: object, **kwargs: object) -> ReferenceExecutionResult:
        return ReferenceExecutionResult(
            summary=_toy_summary(dataset_scalars={"Nnorm": 2.0}),
            stdout_path=os.fspath(tmp_path / "run.stdout"),
        )

    monkeypatch.setattr(reference_module, "run_reference_case", fake_run_reference_case)
    mismatch = validate_reference_baselines(
        reference_root=tmp_path / "reference-root",
        manifest_path=manifest,
        baseline_dir=baseline_dir,
    )
    assert mismatch[0].case_name == "toy_rhs"
    assert mismatch[0].ok is False
    assert mismatch[0].issues == ("dataset_scalars.Nnorm: expected 1.0, got 2.0",)


def test_summarize_dataset_extracts_scalars_and_deltas(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("x", 3)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0, 1.0])
        for name, value in {
            "Nnorm": 1.0e18,
            "Tnorm": 5.0,
            "Bnorm": 1.0,
            "Cs0": 2.0,
            "Omega_ci": 3.0,
            "rho_s0": 4.0,
        }.items():
            scalar = dataset.createVariable(name, "f8")
            scalar.assignValue(value)
        ne = dataset.createVariable("Ne", "f8", ("t", "x"))
        ne[:] = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]])

    variables, dimensions, time_points, scalars = _summarize_dataset(
        dataset_path,
        compare_variables=("Ne",),
        trim_x_guards=False,
        x_guards=0,
        trim_y_guards=False,
        y_guards=0,
    )

    assert dimensions == {"t": 2, "x": 3}
    assert time_points == (0.0, 1.0)
    assert scalars["Nnorm"] == 1.0e18
    assert variables["Ne"].shape == (2, 3)
    assert variables["Ne"].max_abs_delta_last_first == 0.5


def test_summarize_run_builds_trimmed_summary_from_required_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "BOUT.inp"
    input_path.write_text(
        """
        nout = 2
        timestep = 0.5
        MXG = 1
        MYG = 1

        [mesh]
        nx = 5
        ny = 5
        nz = 1
        J = 1

        [model]
        components = e

        [e]
        type = evolve_density
        charge = -1
        AA = 1/1836
        """,
        encoding="utf-8",
    )
    workdir = tmp_path / "run"
    workdir.mkdir()
    for artifact_name in ("BOUT.settings", "BOUT.log.0", "BOUT.restart.0.nc"):
        (workdir / artifact_name).write_text("present\n", encoding="utf-8")
    with Dataset(workdir / "BOUT.dmp.0.nc", "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("x", 5)
        dataset.createDimension("y", 5)
        dataset.createDimension("z", 1)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0, 0.5])
        for name, value in {
            "Nnorm": 1.0e18,
            "Tnorm": 5.0,
            "Bnorm": 2.0,
            "Cs0": 3.0,
            "Omega_ci": 4.0,
            "rho_s0": 5.0,
        }.items():
            scalar = dataset.createVariable(name, "f8")
            scalar.assignValue(value)
        ne = dataset.createVariable("Ne", "f8", ("t", "x", "y", "z"))
        ne[:] = np.arange(50, dtype=np.float64).reshape(2, 5, 5, 1)

    summary = _summarize_run(
        case=_toy_case(trim_x_guards=True, trim_y_guards=True),
        input_path=input_path,
        binary=tmp_path / "reference-binary",
        workdir=workdir,
        overrides=("nout=2",),
    )

    assert summary.case_name == "toy_rhs"
    assert summary.overrides == ("nout=2",)
    assert summary.artifacts["BOUT.dmp.0.nc"] == os.fspath(workdir / "BOUT.dmp.0.nc")
    assert summary.dimensions == {"t": 2, "x": 5, "y": 5, "z": 1}
    assert summary.time_points == (0.0, 0.5)
    assert summary.dataset_scalars["rho_s0"] == 5.0
    assert summary.component_labels == ("e:evolve_density",)
    assert summary.nout == 2
    assert summary.timestep == 0.5
    ne_summary = summary.variable_summaries["Ne"]
    assert ne_summary.shape == (2, 3, 3, 1)
    assert ne_summary.max_abs_delta_last_first == pytest.approx(25.0)


def test_summarize_variable_trims_guards_and_omits_delta_without_time_dimension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("x", 5)
        dataset.createDimension("y", 5)
        ne = dataset.createVariable("Ne", "f8", ("x", "y"))
        ne[:] = np.arange(25, dtype=np.float64).reshape(5, 5)

    with Dataset(dataset_path) as dataset:
        summary = _summarize_variable(
            dataset,
            "Ne",
            trim_x_guards=True,
            x_guards=1,
            trim_y_guards=True,
            y_guards=2,
        )

    assert summary.dimensions == ("x", "y")
    assert summary.shape == (3, 1)
    assert summary.minimum == 7.0
    assert summary.maximum == 17.0
    assert summary.max_abs_delta_last_first is None


def test_assert_artifacts_exist_reports_missing_required_outputs(tmp_path: Path) -> None:
    present = tmp_path / "BOUT.dmp.0.nc"
    present.write_text("data", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="BOUT.log.0"):
        _assert_artifacts_exist(
            {
                "BOUT.dmp.0.nc": os.fspath(present),
                "BOUT.log.0": os.fspath(tmp_path / "BOUT.log.0"),
            }
        )


def test_write_run_summary_json_serializes_payload(tmp_path: Path) -> None:
    summary = ReferenceRunSummary(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        reference_binary="/tmp/reference-binary",
        overrides=("nout=0",),
        workdir="/tmp/run",
        artifacts={"BOUT.dmp.0.nc": "/tmp/run/BOUT.dmp.0.nc"},
        dimensions={"t": 1},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        compare_variables=("Ne",),
        variable_summaries={
            "Ne": VariableSummary(
                name="Ne",
                dimensions=("t", "x"),
                shape=(1, 3),
                minimum=1.0,
                maximum=3.0,
                mean=2.0,
                max_abs_delta_last_first=None,
            )
        },
        component_labels=("e:evolve_density",),
        nout=5,
        timestep=20.0,
    )

    path = write_run_summary_json(summary, tmp_path / "summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case_name"] == "toy"
    assert payload["capability_tier"] == "native_exact"
    assert payload["variable_summaries"]["Ne"]["maximum"] == 3.0


def test_write_case_baseline_json_omits_machine_specific_paths(tmp_path: Path) -> None:
    summary = ReferenceRunSummary(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        reference_binary="/tmp/build/reference-binary",
        overrides=("nout=0",),
        workdir="/tmp/run",
        artifacts={"BOUT.dmp.0.nc": "/tmp/run/BOUT.dmp.0.nc", "BOUT.settings": "/tmp/run/BOUT.settings"},
        dimensions={"t": 1},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        compare_variables=("Ne",),
        variable_summaries={
            "Ne": VariableSummary(
                name="Ne",
                dimensions=("t", "x"),
                shape=(1, 3),
                minimum=1.0,
                maximum=3.0,
                mean=2.0,
                max_abs_delta_last_first=None,
            )
        },
        component_labels=("e:evolve_density",),
        nout=5,
        timestep=20.0,
    )

    path = write_case_baseline_json(summary, tmp_path / "baseline.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["capability_tier"] == "native_exact"
    assert payload["reference_runner"] == "external-reference"
    assert payload["required_artifacts"] == ["BOUT.dmp.0.nc", "BOUT.settings"]
    assert "workdir" not in payload


def test_build_case_baseline_payload_serializes_sequence_metadata_as_lists() -> None:
    summary = ReferenceRunSummary(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        reference_binary="/tmp/build/reference-binary",
        overrides=("nout=0",),
        workdir="/tmp/run",
        artifacts={"BOUT.dmp.0.nc": "/tmp/run/BOUT.dmp.0.nc"},
        dimensions={"t": 1},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        compare_variables=("Ne",),
        variable_summaries={
            "Ne": VariableSummary(
                name="Ne",
                dimensions=("t", "x"),
                shape=(1, 3),
                minimum=1.0,
                maximum=3.0,
                mean=2.0,
                max_abs_delta_last_first=None,
            )
        },
        component_labels=("e:evolve_density",),
        nout=5,
        timestep=20.0,
    )

    payload = build_case_baseline_payload(summary)
    assert payload["capability_tier"] == "native_exact"
    assert payload["compare_variables"] == ["Ne"]
    assert payload["component_labels"] == ["e:evolve_density"]
    assert payload["variable_summaries"]["Ne"]["dimensions"] == ["t", "x"]
    assert payload["variable_summaries"]["Ne"]["shape"] == [1, 3]


def test_run_reference_binary_executes_in_staged_workdir(tmp_path: Path) -> None:
    binary = tmp_path / "reference-stub"
    marker = tmp_path / "expected-cwd.txt"
    binary.write_text(
        "#!/bin/sh\n"
        "pwd > expected-cwd.txt\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    workdir = tmp_path / "run"
    workdir.mkdir()
    stdout_path = workdir / "run.stdout"

    _run_reference_binary(binary=binary, workdir=workdir, overrides=(), stdout_path=stdout_path)

    assert (workdir / "expected-cwd.txt").read_text(encoding="utf-8").strip() == os.fspath(workdir)


def test_run_reference_binary_writes_stdout_and_raises_on_failure(tmp_path: Path) -> None:
    binary = tmp_path / "reference-stub"
    binary.write_text(
        "#!/bin/sh\n"
        "echo failed-run\n"
        "exit 7\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    workdir = tmp_path / "run"
    workdir.mkdir()
    stdout_path = workdir / "run.stdout"

    with pytest.raises(RuntimeError, match="exit code 7"):
        _run_reference_binary(binary=binary, workdir=workdir, overrides=(), stdout_path=stdout_path)

    assert stdout_path.read_text(encoding="utf-8") == "failed-run\n"


def test_reference_command_uses_mpirun_for_multi_rank_cases(tmp_path: Path) -> None:
    binary = tmp_path / "reference-binary"
    workdir = tmp_path / "run"
    command = _reference_command(
        binary=binary,
        workdir=workdir,
        overrides=("nout=1", "NXPE=1"),
        process_count=6,
    )
    assert command == [
        "mpirun",
        "-np",
        "6",
        os.fspath(binary),
        "-d",
        os.fspath(workdir),
        "nout=1",
        "NXPE=1",
    ]


def test_prepare_workdir_extracts_required_bundle_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "tests" / "integrated" / "2D-recycling" / "data"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "BOUT.inp"
    input_path.write_text("nout = 1\n", encoding="utf-8")

    bundle_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("grid_test2.nc", "grid-data")
        archive.writestr("unused.txt", "ignore-me")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    case = ReferenceCase(
        name="integrated_2d_recycling_one_step",
        stage="stage7",
        reference_path="tests/integrated/2D-recycling/data/BOUT.inp",
        parity_mode="one_step",
        rationale="Stable 2D recycling staging target.",
        process_count=10,
        artifact_bundle_url=bundle_path.as_uri(),
        artifact_bundle_sha256=digest,
        artifact_bundle_files=("grid_test2.nc",),
    )

    workdir = _prepare_workdir(case, input_path, workdir=tmp_path / "run")

    assert (workdir / "BOUT.inp").is_symlink()
    assert (workdir / "grid_test2.nc").read_text(encoding="utf-8") == "grid-data"
    assert not (workdir / "unused.txt").exists()


def test_prepare_workdir_uses_temporary_directory_and_skips_reference_outputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "cases" / "toy"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "BOUT.inp"
    input_path.write_text("nout = 0\n", encoding="utf-8")
    (source_dir / "BOUT.dmp.0.nc").write_text("old-output", encoding="utf-8")
    (source_dir / "BOUT.settings").write_text("old-settings", encoding="utf-8")
    (source_dir / "run.stdout").write_text("old-stdout", encoding="utf-8")
    case = _toy_case()

    workdir = _prepare_workdir(case, input_path, workdir=None)
    try:
        assert workdir.name.startswith("jaxdrb-cases-")
        assert (workdir / "BOUT.inp").is_symlink()
        assert not (workdir / "BOUT.dmp.0.nc").exists()
        assert not (workdir / "BOUT.settings").exists()
        assert not (workdir / "run.stdout").exists()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_prepare_workdir_uses_bundle_artifact_for_declared_mesh_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "tests" / "integrated" / "2D-recycling" / "data"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "BOUT.inp"
    input_path.write_text(
        """
        nout = 1

        [mesh]
        file = grid_test2.nc
        """,
        encoding="utf-8",
    )

    bundle_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("grid_test2.nc", "grid-data")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    case = ReferenceCase(
        name="integrated_2d_recycling_one_step",
        stage="stage7",
        reference_path="tests/integrated/2D-recycling/data/BOUT.inp",
        parity_mode="one_step",
        rationale="Stable 2D recycling staging target.",
        process_count=10,
        artifact_bundle_url=bundle_path.as_uri(),
        artifact_bundle_sha256=digest,
        artifact_bundle_files=("grid_test2.nc",),
    )

    workdir = _prepare_workdir(case, input_path, workdir=tmp_path / "run")

    assert (workdir / "grid_test2.nc").read_text(encoding="utf-8") == "grid-data"


def test_prepare_workdir_handles_empty_absolute_local_and_nested_mesh_paths(tmp_path: Path) -> None:
    case = _toy_case()

    empty_dir = tmp_path / "empty-mesh"
    empty_dir.mkdir()
    empty_input = empty_dir / "BOUT.inp"
    empty_input.write_text("[mesh]\nfile = \"\"\n", encoding="utf-8")
    assert _prepare_workdir(case, empty_input, workdir=tmp_path / "run-empty").exists()

    missing_abs_dir = tmp_path / "missing-absolute"
    missing_abs_dir.mkdir()
    missing_abs_input = missing_abs_dir / "BOUT.inp"
    missing_abs_input.write_text(f"[mesh]\nfile = {tmp_path / 'missing.nc'}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Configured mesh file does not exist"):
        _prepare_workdir(case, missing_abs_input, workdir=tmp_path / "run-missing-absolute")

    local_dir = tmp_path / "local-mesh"
    local_dir.mkdir()
    local_mesh = local_dir / "grid.nc"
    local_mesh.write_text("local-grid", encoding="utf-8")
    local_input = local_dir / "BOUT.inp"
    local_input.write_text("[mesh]\nfile = grid.nc\n", encoding="utf-8")
    local_run = _prepare_workdir(case, local_input, workdir=tmp_path / "run-local")
    assert (local_run / "grid.nc").resolve() == local_mesh.resolve()

    nested_dir = tmp_path / "nested-mesh"
    nested_mesh = nested_dir / "meshes" / "grid.nc"
    nested_mesh.parent.mkdir(parents=True)
    nested_mesh.write_text("nested-grid", encoding="utf-8")
    nested_input = nested_dir / "BOUT.inp"
    nested_input.write_text("[mesh]\nfile = meshes/grid.nc\n", encoding="utf-8")
    nested_run = _prepare_workdir(case, nested_input, workdir=tmp_path / "run-nested")
    assert (nested_run / "meshes" / "grid.nc").resolve() == nested_mesh.resolve()

    missing_nested_dir = tmp_path / "missing-nested-mesh"
    missing_nested_dir.mkdir()
    missing_nested_input = missing_nested_dir / "BOUT.inp"
    missing_nested_input.write_text("[mesh]\nfile = meshes/grid.nc\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="meshes/grid.nc"):
        _prepare_workdir(case, missing_nested_input, workdir=tmp_path / "run-missing-nested")


def test_prepare_workdir_rejects_ambiguous_parent_mesh_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    case_dir = source_root / "level" / "case"
    case_dir.mkdir(parents=True)
    (source_root / "grid.nc").write_text("root-grid", encoding="utf-8")
    (source_root / "level" / "grid.nc").write_text("level-grid", encoding="utf-8")
    input_path = case_dir / "BOUT.inp"
    input_path.write_text("[mesh]\nfile = grid.nc\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Ambiguous mesh file"):
        _prepare_workdir(_toy_case(), input_path, workdir=tmp_path / "run-ambiguous")


def test_stage_mesh_file_accepts_same_target_and_rejects_different_source(tmp_path: Path) -> None:
    source = tmp_path / "source-grid.nc"
    other_source = tmp_path / "other-grid.nc"
    target = tmp_path / "run" / "grid.nc"
    source.write_text("grid", encoding="utf-8")
    other_source.write_text("other-grid", encoding="utf-8")

    _stage_mesh_file(source, target)
    _stage_mesh_file(source, target)
    with pytest.raises(RuntimeError, match="different source"):
        _stage_mesh_file(other_source, target)


def test_prepare_workdir_stages_shared_json_database_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "examples" / "tokamak-2D" / "recycling-dthene"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "BOUT.inp"
    input_path.write_text("nout = 0\n", encoding="utf-8")
    json_database = tmp_path / "source" / "json_database"
    json_database.mkdir()
    (json_database / "scd96_ne.json").write_text("{\"mock\": true}\n", encoding="utf-8")

    case = ReferenceCase(
        name="tokamak_recycling_dthene_rhs",
        stage="stage8",
        reference_path="examples/tokamak-2D/recycling-dthene/BOUT.inp",
        parity_mode="one_rhs",
        rationale="Tokamak multispecies recycling with neon.",
        process_count=6,
    )

    workdir = _prepare_workdir(case, input_path, workdir=tmp_path / "run")

    assert (workdir / "json_database").is_symlink()
    assert (workdir / "json_database" / "scd96_ne.json").read_text(encoding="utf-8") == "{\"mock\": true}\n"


def test_stage_case_artifacts_handles_local_files_existing_targets_and_missing_members(tmp_path: Path) -> None:
    no_bundle_case = _toy_case(artifact_bundle_url=None)
    _stage_case_artifacts(no_bundle_case, tmp_path / "run-no-bundle")

    bundle_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("grid.nc", "bundle-grid")
    run_dir = tmp_path / "run-bundle"
    run_dir.mkdir()
    (run_dir / "grid.nc").write_text("existing-grid", encoding="utf-8")
    case = _toy_case(artifact_bundle_url=os.fspath(bundle_path))
    _stage_case_artifacts(case, run_dir)
    assert (run_dir / "grid.nc").read_text(encoding="utf-8") == "existing-grid"

    missing_member_case = _toy_case(
        artifact_bundle_url=os.fspath(bundle_path),
        artifact_bundle_files=("missing.nc",),
    )
    with pytest.raises(FileNotFoundError, match="Artifact 'missing.nc' not found"):
        _stage_case_artifacts(missing_member_case, tmp_path / "run-missing-member")


def test_prepare_workdir_rejects_artifact_bundle_hash_mismatch(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "tests" / "integrated" / "2D-recycling" / "data"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "BOUT.inp"
    input_path.write_text("nout = 0\n", encoding="utf-8")

    bundle_path = tmp_path / "artifacts.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("grid_test2.nc", "grid-data")

    case = ReferenceCase(
        name="integrated_2d_recycling_rhs",
        stage="stage7",
        reference_path="tests/integrated/2D-recycling/data/BOUT.inp",
        parity_mode="one_rhs",
        rationale="Stable 2D recycling staging target.",
        process_count=10,
        artifact_bundle_url=bundle_path.as_uri(),
        artifact_bundle_sha256="deadbeef",
        artifact_bundle_files=("grid_test2.nc",),
    )

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _prepare_workdir(case, input_path, workdir=tmp_path / "run")


def test_prepare_workdir_stages_mesh_file_from_parent_directory(tmp_path: Path) -> None:
    tokamak_root = tmp_path / "source" / "examples" / "tokamak-2D"
    case_dir = tokamak_root / "diffusion-flow-evolveT"
    case_dir.mkdir(parents=True)
    mesh_path = tokamak_root / "tokamak.nc"
    mesh_path.write_text("mesh-data", encoding="utf-8")
    input_path = case_dir / "BOUT.inp"
    input_path.write_text(
        """
        nout = 1

        [mesh]
        file = tokamak.nc
        """,
        encoding="utf-8",
    )

    case = ReferenceCase(
        name="tokamak_diffusion_flow_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/diffusion-flow-evolveT/BOUT.inp",
        parity_mode="one_step",
        rationale="Stable direct tokamak staging target.",
        process_count=6,
    )

    workdir = _prepare_workdir(case, input_path, workdir=tmp_path / "run")

    assert (workdir / "BOUT.inp").is_symlink()
    assert (workdir / "tokamak.nc").is_symlink()
    assert (workdir / "tokamak.nc").resolve() == mesh_path.resolve()


def test_prepare_workdir_raises_when_parent_mesh_file_is_missing(tmp_path: Path) -> None:
    case_dir = tmp_path / "source" / "examples" / "tokamak-2D" / "diffusion-flow-evolveT"
    case_dir.mkdir(parents=True)
    input_path = case_dir / "BOUT.inp"
    input_path.write_text(
        """
        nout = 1

        [mesh]
        file = tokamak.nc
        """,
        encoding="utf-8",
    )

    case = ReferenceCase(
        name="tokamak_diffusion_flow_one_step",
        stage="stage7",
        reference_path="examples/tokamak-2D/diffusion-flow-evolveT/BOUT.inp",
        parity_mode="one_step",
        rationale="Stable direct tokamak staging target.",
        process_count=6,
    )

    with pytest.raises(FileNotFoundError, match="tokamak.nc"):
        _prepare_workdir(case, input_path, workdir=tmp_path / "run")
