from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_drb.parity.reference import (
    _prepare_workdir,
    _reference_command,
    _run_reference_binary,
    build_case_baseline_payload,
    _summarize_dataset,
    merge_overrides,
    make_default_overrides,
    write_case_baseline_json,
    write_run_summary_json,
)
from jax_drb.reference.cases import ReferenceCase, load_reference_cases
from jax_drb.parity.reference import ReferenceRunSummary, VariableSummary


def test_reference_cases_support_optional_compare_variables() -> None:
    cases = load_reference_cases()
    evolve_density = next(case for case in cases if case.name == "evolve_density_rhs")
    assert evolve_density.compare_variables == ("Ne",)


def test_make_default_overrides_respects_fast_parity_modes() -> None:
    assert make_default_overrides("one_rhs") == ("nout=0",)
    assert make_default_overrides("one_step") == ("nout=1",)
    assert make_default_overrides("short_window") == ()


def test_merge_overrides_replaces_duplicate_keys_with_latest_value() -> None:
    assert merge_overrides(("nout=0", "i:diagnose=false"), ("nout=1",), ("i:diagnose=true",)) == (
        "nout=1",
        "i:diagnose=true",
    )


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
