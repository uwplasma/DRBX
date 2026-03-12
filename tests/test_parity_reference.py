from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.parity.reference import (
    _summarize_dataset,
    merge_overrides,
    make_default_overrides,
    write_case_baseline_json,
    write_run_summary_json,
)
from jax_drb.reference.cases import load_reference_cases
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
    assert payload["variable_summaries"]["Ne"]["maximum"] == 3.0


def test_write_case_baseline_json_omits_machine_specific_paths(tmp_path: Path) -> None:
    summary = ReferenceRunSummary(
        case_name="toy",
        parity_mode="one_rhs",
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
    assert payload["reference_runner"] == "external-reference"
    assert payload["required_artifacts"] == ["BOUT.dmp.0.nc", "BOUT.settings"]
    assert "workdir" not in payload
