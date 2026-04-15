from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation.tokamak_tcv_x21_scaffold import _write_synthetic_preview_workdir
from jax_drb.validation.tokamak_tcv_x21_selected_field import (
    compare_tcv_x21_selected_field_workdirs,
    create_tcv_x21_selected_field_parity_package,
)


def test_compare_tcv_x21_selected_fields_is_zero_for_identical_synthetic_workdirs() -> None:
    with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_test_") as temp_dir:
        root = Path(temp_dir)
        reference = _write_synthetic_preview_workdir(root / "reference", field_name="phi")
        candidate = _write_synthetic_preview_workdir(root / "candidate", field_name="phi")

        result = compare_tcv_x21_selected_field_workdirs(
            reference_workdir=reference.workdir,
            candidate_workdir=candidate.workdir,
        )

        assert tuple(result.field_names) == ("Ne", "Pe", "phi")
        for error in result.variable_errors.values():
            assert error.max_abs_error == 0.0
            assert error.rms_error == 0.0
            assert error.relative_l2_error == 0.0


def test_compare_tcv_x21_selected_fields_detects_candidate_field_offset() -> None:
    with tempfile.TemporaryDirectory(prefix="jax_drb_tcv_x21_selected_field_test_") as temp_dir:
        root = Path(temp_dir)
        reference = _write_synthetic_preview_workdir(root / "reference", field_name="phi")
        candidate = _write_synthetic_preview_workdir(root / "candidate", field_name="phi")
        for dump_path in sorted(candidate.workdir.glob("BOUT.dmp.*.nc")):
            with Dataset(dump_path, "r+") as dataset:
                values = np.asarray(dataset.variables["phi"][:], dtype=np.float64)
                dataset.variables["phi"][:] = values + 0.01

        result = compare_tcv_x21_selected_field_workdirs(
            reference_workdir=reference.workdir,
            candidate_workdir=candidate.workdir,
        )

        assert result.variable_errors["phi"].max_abs_error > 0.0
        assert result.variable_errors["phi"].relative_l2_error > 0.0
        assert result.variable_errors["Ne"].max_abs_error == 0.0


def test_create_tcv_x21_selected_field_parity_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_tcv_x21_selected_field_parity_package(
        reference_workdir=None,
        candidate_workdir=None,
        benchmark_data_root=None,
        output_root=tmp_path / "output",
    )

    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    assert artifacts.benchmark_data_report_json_path is None

    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["Ne", "Pe", "phi"]
    assert sorted(payload["variable_errors"]) == ["Ne", "Pe", "phi"]

    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["metadata"]["source_mode"] == "synthetic_preview"


def test_create_tcv_x21_selected_field_parity_package_supports_public_benchmark_root(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    _write_public_benchmark_snapshot_bundle(benchmark_root)

    artifacts = create_tcv_x21_selected_field_parity_package(
        reference_workdir=None,
        candidate_workdir=None,
        benchmark_data_root=benchmark_root,
        output_root=tmp_path / "output",
    )

    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    assert artifacts.benchmark_data_report_json_path is not None
    assert artifacts.benchmark_data_report_json_path.exists()

    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["Ne", "Pe", "phi"]
    assert payload["variable_errors"]["phi"]["max_abs_error"] > 0.0

    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["metadata"]["source_mode"] == "external_benchmark_reference_derived_candidate"
    assert observable["metadata"]["reference_source"] == "public_tcv_x21_benchmark_bundle"

    benchmark_report = json.loads(artifacts.benchmark_data_report_json_path.read_text(encoding="utf-8"))
    assert benchmark_report["requested_field_name"] == "Ne"


def _write_public_benchmark_snapshot_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with Dataset(root / "TCV_forward_field.nc", "w") as dataset:
        diagnostic = dataset.createGroup("FHRP")
        observables = diagnostic.createGroup("observables")
        density = observables.createGroup("density")
        density.createDimension("n", 3)
        value = density.createVariable("value", "f8", ("n",))
        error = density.createVariable("error", "f8", ("n",))
        coord = density.createVariable("Rsep_omp", "f8", ("n",))
        value[:] = np.array([1.0, 1.2, 1.1], dtype=np.float64)
        error[:] = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        coord[:] = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    with Dataset(root / "snaps00000.nc", "w") as dataset:
        dataset.createDimension("tau", 4)
        dataset.createDimension("point", 6)
        tau = dataset.createVariable("tau", "f8", ("tau",))
        tau[:] = np.linspace(0.0, 0.3, 4)
        base = np.linspace(0.0, 1.0, 24, dtype=np.float64).reshape(4, 6)
        for name, values in {
            "logne": 1.0 + base,
            "logte": 0.6 + 0.5 * base,
            "potxx": -0.1 + 0.2 * base,
        }.items():
            variable = dataset.createVariable(name, "f8", ("tau", "point"))
            variable[:] = values
    with Dataset(root / "vgrid.nc", "w") as dataset:
        dataset.createDimension("point", 6)
        li = dataset.createVariable("li", "i4", ("point",))
        lj = dataset.createVariable("lj", "i4", ("point",))
        li[:] = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
        lj[:] = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
