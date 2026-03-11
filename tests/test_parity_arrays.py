from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.parity.arrays import (
    build_dataset_array_payload,
    build_portable_array_payload,
    compare_array_payloads,
    load_portable_array_payload,
    write_portable_array_payload,
)


def test_portable_array_payload_round_trips_npz(tmp_path: Path) -> None:
    payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 3},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]])},
        overrides=("nout=1",),
        configured_nout=5,
        configured_timestep=1.0,
    )

    path = write_portable_array_payload(payload, tmp_path / "toy_arrays.npz")
    loaded = load_portable_array_payload(path)

    assert loaded["case_name"] == "toy"
    np.testing.assert_allclose(loaded["variables"]["Ne"], payload["variables"]["Ne"])


def test_compare_array_payloads_reports_array_mismatch() -> None:
    expected = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 2},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0], [1.5, 2.5]])},
    )
    actual = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 2},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0], [1.5, 2.8]])},
    )

    result = compare_array_payloads(expected, actual, array_rtol=1e-12, array_atol=1e-12)

    assert result.ok is False
    assert result.issues[0].field == "variables.Ne"


def test_build_dataset_array_payload_reads_full_variable_arrays(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("x", 3)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0, 1.0])
        scalar = dataset.createVariable("Nnorm", "f8")
        scalar.assignValue(1.0e18)
        ne = dataset.createVariable("Ne", "f8", ("t", "x"))
        ne[:] = np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]])

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_step",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        overrides=("nout=1",),
        configured_nout=5,
        configured_timestep=1.0,
    )

    assert payload["case_name"] == "toy"
    assert payload["time_points"] == [0.0, 1.0]
    assert payload["variable_dimensions"]["Ne"] == ["t", "x"]
    np.testing.assert_allclose(payload["variables"]["Ne"], np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]]))
