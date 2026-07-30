from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from drbx.geometry import (
    essos_runtime_available,
    load_essos_field_line_bundle_npz,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
)
from drbx.validation import (
    create_essos_fieldline_import_package,
    create_essos_vmec_fieldline_surface_package,
)
import drbx.geometry.essos_import as essos_import


def _has_essos_landreman_runtime() -> bool:
    if os.environ.get("DRBX_RUN_ESSOS_TESTS") != "1":
        return False
    try:
        resolve_essos_landreman_qa_json()
        resolve_essos_landreman_qa_wout()
    except FileNotFoundError:
        return False
    return essos_runtime_available()


def test_target_exit_lengths_are_masked_to_fci_endpoint_cells() -> None:
    raw_exit_length = np.array([0.5, 1.5, np.nan, 3.0], dtype=np.float64)
    boundary = np.array([True, False, True, True])
    masked = essos_import._mask_exit_length_to_boundary(raw_exit_length, boundary)
    np.testing.assert_allclose(masked, np.array([0.5, np.nan, np.nan, 3.0]), equal_nan=True)
    with pytest.raises(ValueError, match="shapes must match"):
        essos_import._mask_exit_length_to_boundary(raw_exit_length, boundary[:-1])


@pytest.mark.skipif(
    not _has_essos_landreman_runtime(),
    reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available",
)
def test_essos_fieldline_import_generates_portable_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_fieldline_import_package(
        output_root=tmp_path / "essos_import", n_field_lines=3, times_to_trace=768, maxtime=150.0
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    bundle = load_essos_field_line_bundle_npz(artifacts.arrays_npz_path, metadata=report["metadata"])
    assert report["passed"] is True
    assert report["source"] == "ESSOS external field and field-line import"
    assert bundle.trajectories_xyz.shape == (3, 768, 3)
    assert bundle.field_sample_xyz.shape == bundle.field_sample_b_xyz.shape
    assert bundle.poincare_point_count == report["poincare_point_count"]
    assert bundle.poincare_point_count > 0
    assert np.all(np.isfinite(bundle.trajectories_xyz))
    assert np.all(np.isfinite(bundle.field_sample_b_xyz))
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(
    not _has_essos_landreman_runtime(),
    reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available",
)
def test_essos_fieldline_poincare_quantifies_scaled_vmec_surface_registration(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_fieldline_surface",
        rho_min=0.20, rho_max=0.82, n_surfaces=3, ntheta_surface=96,
        maxtime=180.0, times_to_trace=768, sections=(0.0, float(np.pi / 2.0)),
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS coil field-line tracing compared with scaled Landreman-Paul QA VMEC surfaces"
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert np.isfinite(report["same_surface_distance_normalized_p95"])
    assert np.isfinite(report["nearest_surface_distance_normalized_p95"])
    assert isinstance(report["fieldline_surface_match_passed"], bool)
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(
    not _has_essos_landreman_runtime(),
    reason="ESSOS runtime and Landreman-Paul QA VMEC wout are not available",
)
def test_essos_vmec_fieldline_poincare_preserves_scaled_vmec_surfaces(tmp_path: Path) -> None:
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=tmp_path / "essos_vmec_equilibrium_fieldline_surface",
        field_source="vmec", rho_min=0.20, rho_max=0.82, n_surfaces=3, ntheta_surface=96,
        maxtime=180.0, times_to_trace=768, sections=(0.0, float(np.pi / 2.0)),
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["field_source"] == "vmec"
    assert report["fieldline_surface_match_passed"] is True
    assert report["surface_nonaxisymmetric_major_rms"] > 5.0e-2
    assert report["poincare_point_count"] >= 2 * report["n_surfaces"]
    assert report["fieldline_s_drift_max"] < 1.0e-7
    assert report["same_surface_distance_normalized_p95"] < 5.0e-2
    assert report["nearest_surface_distance_normalized_p95"] < 5.0e-2
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
