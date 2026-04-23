from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
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
        capability_tier="native_exact",
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
    assert loaded["capability_tier"] == "native_exact"
    np.testing.assert_allclose(loaded["variables"]["Ne"], payload["variables"]["Ne"])


def test_portable_array_payload_skips_missing_variables_and_generates_fallback_dimensions() -> None:
    payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne", "Te"),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 3},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.ones((2, 3, 4))},
    )

    assert set(payload["variables"]) == {"Ne"}
    assert payload["variable_dimensions"]["Ne"] == ["t", "dim_1", "dim_2"]


def test_array_payload_from_summary_payload_uses_defaults_for_older_metadata() -> None:
    payload = build_array_payload_from_summary_payload(
        {
            "case_name": "toy",
            "parity_mode": "one_rhs",
            "compare_variables": ["Ne"],
            "dimensions": {"t": 1, "x": 2},
            "time_points": ["0.0"],
            "dataset_scalars": {"Nnorm": 1.0},
        },
        {"Ne": np.array([[1.0, 2.0]])},
    )

    assert payload["capability_tier"] == "native_exact"
    assert payload["producer"] == "jax-drb"
    assert "configured_nout" not in payload
    assert payload["time_points"] == [0.0]


def test_compare_array_payloads_reports_array_mismatch() -> None:
    expected = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
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
        capability_tier="native_exact",
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


def test_compare_array_payloads_reports_metadata_time_scalar_and_shape_mismatches() -> None:
    expected = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 2},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0, "Tnorm": 2.0},
        variables={"Ne": np.ones((2, 2)), "Ni": np.ones((2, 2))},
        variable_dimensions={"Ne": ("t", "x"), "Ni": ("t", "x")},
    )
    actual = build_portable_array_payload(
        case_name="wrong-case",
        parity_mode="one_rhs",
        capability_tier="native_operational",
        compare_variables=("Ne",),
        component_labels=("different",),
        dimensions={"t": 3, "x": 2},
        time_points=(0.0, 1.1, 2.0),
        dataset_scalars={"Nnorm": 1.2, "Bnorm": 3.0},
        variables={"Ne": np.ones((2, 3)), "Te": np.ones((2, 2))},
        variable_dimensions={"Ne": ("t", "x")},
    )
    expected["variables"]["Ni"] = np.ones((2, 2))
    expected["variable_dimensions"]["Ni"] = ["t", "x"]
    actual["variables"]["Te"] = np.ones((2, 2))
    actual["variable_dimensions"]["Te"] = ["t", "x"]

    result = compare_array_payloads(expected, actual, scalar_rtol=1e-12, scalar_atol=1e-12)
    fields = {issue.field for issue in result.issues}

    assert result.ok is False
    assert {
        "case_name",
        "parity_mode",
        "component_labels",
        "dimensions",
        "variable_dimensions",
        "capability_tier",
        "time_points",
        "dataset_scalars.Bnorm",
        "dataset_scalars.Nnorm",
        "dataset_scalars.Tnorm",
        "variables.Ne.shape",
        "variables.Ni",
        "variables.Te",
    }.issubset(fields)


def test_compare_array_payloads_reports_individual_time_point_mismatch() -> None:
    expected = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 1},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.ones((2, 1))},
    )
    actual = dict(expected)
    actual["time_points"] = [0.0, 1.25]

    result = compare_array_payloads(expected, actual)

    assert result.ok is False
    assert result.issues[0].field == "time_points[1]"


def test_compare_array_payloads_tolerates_missing_capability_tier_in_older_baselines() -> None:
    expected = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 2},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0], [1.5, 2.5]])},
    )
    actual = dict(expected)
    actual["variables"] = dict(expected["variables"])
    actual["capability_tier"] = "native_exact"
    expected_without_tier = dict(expected)
    expected_without_tier.pop("capability_tier")

    result = compare_array_payloads(expected_without_tier, actual)

    assert result.ok is True
    assert result.issues == ()


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
    assert payload["capability_tier"] == "scaffolded_reference_backed"
    assert payload["time_points"] == [0.0, 1.0]
    assert payload["variable_dimensions"]["Ne"] == ["t", "x"]
    np.testing.assert_allclose(payload["variables"]["Ne"], np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]]))


def test_build_dataset_array_payload_handles_missing_time_scalars_and_requested_variables(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("x", 2)
        ne = dataset.createVariable("Ne", "f8", ("x",))
        ne[:] = np.array([1.0, 2.0])

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_rhs",
        compare_variables=("Ne", "Te"),
        component_labels=("e:evolve_density",),
    )

    assert payload["time_points"] == []
    assert payload["dataset_scalars"] == {}
    assert set(payload["variables"]) == {"Ne"}


def test_build_dataset_array_payload_accepts_explicit_capability_tier(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("t", 1)
        dataset.createDimension("x", 1)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0])
        ne = dataset.createVariable("Ne", "f8", ("t", "x"))
        ne[:] = np.array([[1.0]])

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_operational",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
    )

    assert payload["capability_tier"] == "native_operational"


def test_build_dataset_array_payload_can_trim_y_guards(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("t", 1)
        dataset.createDimension("x", 1)
        dataset.createDimension("y", 6)
        dataset.createDimension("z", 1)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0])
        ne = dataset.createVariable("Ne", "f8", ("t", "x", "y", "z"))
        ne[:] = np.arange(6, dtype=np.float64)[None, None, :, None]

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_rhs",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        trim_y_guards=True,
        y_guards=2,
    )

    np.testing.assert_allclose(payload["variables"]["Ne"], np.array([[[[2.0], [3.0]]]]))


def test_build_dataset_array_payload_can_trim_x_and_y_guards(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("t", 1)
        dataset.createDimension("x", 5)
        dataset.createDimension("y", 6)
        dataset.createDimension("z", 1)
        t = dataset.createVariable("t_array", "f8", ("t",))
        t[:] = np.array([0.0])
        ne = dataset.createVariable("Ne", "f8", ("t", "x", "y", "z"))
        ne[:] = np.arange(30, dtype=np.float64).reshape(1, 5, 6, 1)

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_rhs",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        trim_x_guards=True,
        x_guards=2,
        trim_y_guards=True,
        y_guards=2,
    )

    np.testing.assert_allclose(payload["variables"]["Ne"], np.array([[[[14.0], [15.0]]]]))


def test_build_dataset_array_payload_leaves_too_small_guard_dimensions_untrimmed(tmp_path: Path) -> None:
    dataset_path = tmp_path / "BOUT.dmp.0.nc"
    with Dataset(dataset_path, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 4)
        ne = dataset.createVariable("Ne", "f8", ("x", "y"))
        ne[:] = np.arange(16, dtype=np.float64).reshape(4, 4)

    payload = build_dataset_array_payload(
        dataset_path,
        case_name="toy",
        parity_mode="one_rhs",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        trim_x_guards=True,
        x_guards=2,
        trim_y_guards=True,
        y_guards=2,
    )

    np.testing.assert_allclose(payload["variables"]["Ne"], np.arange(16, dtype=np.float64).reshape(4, 4))
