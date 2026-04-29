from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jax_drb.geometry import (
    build_essos_imported_fci_geometry,
    essos_runtime_available,
    load_essos_field_line_bundle_npz,
    resolve_essos_landreman_qa_json,
)
from jax_drb.validation import create_essos_fieldline_import_package, create_essos_imported_fci_campaign_package


def _has_essos_landreman_runtime() -> bool:
    if os.environ.get("JAX_DRB_RUN_ESSOS_TESTS") != "1":
        return False
    try:
        resolve_essos_landreman_qa_json()
    except FileNotFoundError:
        return False
    return essos_runtime_available()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_fieldline_import_generates_portable_artifacts(tmp_path: Path) -> None:
    artifacts = create_essos_fieldline_import_package(
        output_root=tmp_path / "essos_import",
        n_field_lines=3,
        times_to_trace=768,
        maxtime=150.0,
    )

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    bundle = load_essos_field_line_bundle_npz(artifacts.arrays_npz_path, metadata=report["metadata"])

    assert report["passed"] is True
    assert report["source"] == "ESSOS external field and field-line import"
    assert report["metadata"]["field_model"] == "essos.fields.BiotSavart"
    assert report["metadata"]["tracing_model"] == "essos.dynamics.Tracing(FieldLineAdaptative)"
    assert bundle.trajectories_xyz.shape == (3, 768, 3)
    assert bundle.field_sample_xyz.shape == bundle.field_sample_b_xyz.shape
    assert bundle.poincare_point_count == report["poincare_point_count"]
    assert bundle.poincare_point_count > 0
    assert np.all(np.isfinite(bundle.trajectories_xyz))
    assert np.all(np.isfinite(bundle.field_sample_b_xyz))
    assert artifacts.plot_png_path.exists()


@pytest.mark.skipif(not _has_essos_landreman_runtime(), reason="ESSOS runtime and Landreman-Paul QA coil JSON are not available")
def test_essos_imported_fci_maps_feed_native_sheath_and_neutral_gates(tmp_path: Path) -> None:
    geometry = build_essos_imported_fci_geometry(
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )

    assert geometry.shape == (3, 4, 6)
    assert geometry.metadata["geometry_family"] == "essos_imported_annular_fci"
    assert np.all(np.isfinite(np.asarray(geometry.magnetic_field_magnitude)))
    assert np.all(np.isfinite(np.asarray(geometry.connection_length)))
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool))) < 0.95
    assert 0.05 < float(np.mean(np.asarray(geometry.maps.backward_boundary, dtype=bool))) < 0.95

    artifacts = create_essos_imported_fci_campaign_package(
        output_root=tmp_path / "essos_imported_fci",
        nx=3,
        ny=4,
        nz=6,
        rho_min=0.12,
        rho_max=0.34,
        maxtime=40.0,
        times_to_trace=160,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source"] == "ESSOS-imported field-line maps with jax_drb FCI closures"
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
