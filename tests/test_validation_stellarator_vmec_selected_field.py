from __future__ import annotations

import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from jax_drb.validation import (
    compare_stellarator_vmec_selected_fields,
    create_stellarator_vmec_selected_field_parity_package,
)


def _write_vmec_case(path: Path, *, scale: float) -> None:
    ns = 6
    mn_mode = 2
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ns", ns)
        dataset.createDimension("mn_mode", mn_mode)
        dataset.createVariable("iotaf", "f8", ("ns",))[:] = scale * np.linspace(0.35, 0.55, ns)
        dataset.createVariable("presf", "f8", ("ns",))[:] = scale * np.linspace(1000.0, 10.0, ns)
        dataset.createVariable("phi", "f8", ("ns",))[:] = scale * np.linspace(0.0, 2.0, ns)
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = np.asarray([0.0, 1.0])
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = np.asarray([0.0, 0.0])
        rmnc = dataset.createVariable("rmnc", "f8", ("ns", "mn_mode"))
        zmns = dataset.createVariable("zmns", "f8", ("ns", "mn_mode"))
        rmnc[:] = np.column_stack((np.full(ns, 4.1), np.linspace(0.05, 0.4, ns)))
        zmns[:] = np.column_stack((np.zeros(ns), np.linspace(0.08, 0.55, ns)))
        dataset.createVariable("nfp", "i4")[:] = 4


def test_compare_stellarator_vmec_selected_fields_zero_for_identical_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.nc"
    candidate = tmp_path / "candidate.nc"
    _write_vmec_case(reference, scale=1.0)
    _write_vmec_case(candidate, scale=1.0)
    result = compare_stellarator_vmec_selected_fields(
        reference_equilibrium_path=reference,
        candidate_equilibrium_path=candidate,
    )
    for error in result.variable_errors.values():
        assert error.max_abs_error == 0.0
        assert error.rms_error == 0.0


def test_create_stellarator_vmec_selected_field_parity_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_vmec_selected_field_parity_package(
        reference_equilibrium_path=None,
        candidate_equilibrium_path=None,
        output_root=tmp_path / "output",
    )
    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    assert artifacts.source_report_json_path.exists()
    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["iota", "pressure", "toroidal_flux"]
    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["metadata"]["source_mode"] == "synthetic_preview"
    source = json.loads(artifacts.source_report_json_path.read_text(encoding="utf-8"))
    assert source["candidate_origin"] == "synthetic_preview_pair"


def test_create_stellarator_vmec_selected_field_parity_package_materializes_external_pair(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.nc"
    _write_vmec_case(reference, scale=1.0)
    artifacts = create_stellarator_vmec_selected_field_parity_package(
        reference_equilibrium_path=reference,
        candidate_equilibrium_path=None,
        output_root=tmp_path / "output",
    )
    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["variable_errors"]["pressure"]["max_abs_error"] > 0.0
    source = json.loads(artifacts.source_report_json_path.read_text(encoding="utf-8"))
    assert source["source_mode"] == "external_explicit_pair"
    assert source["candidate_origin"] == "materialized_from_reference_input"
