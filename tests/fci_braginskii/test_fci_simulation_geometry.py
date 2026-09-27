"""Consumer checks for FCI simulation geometry artifacts."""

from dataclasses import fields
import json
import os
from pathlib import Path

import numpy as np
import pytest

from drbx.fci_braginskii.geometry.fci_geometry import FciMaps3D
from drbx.fci_braginskii.geometry.fci_simulation_geometry import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_fci_simulation_geometry,
)


def _write_manifest(path: Path, **overrides) -> None:
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "topology_name": "toroidal",
        "nfp": 4,
        "components": {},
        **overrides,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_missing_manifest_fails_directly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fci_simulation_geometry(tmp_path)


@pytest.mark.parametrize("schema,version", [(SCHEMA_NAME, 2), ("other", SCHEMA_VERSION)])
def test_unsupported_schema_fails_before_array_loading(tmp_path, schema, version):
    _write_manifest(tmp_path, schema=schema, schema_version=version)
    with pytest.raises(ValueError, match="schema"):
        load_fci_simulation_geometry(tmp_path)


def test_missing_component_fails_directly(tmp_path):
    _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="base_geometry"):
        load_fci_simulation_geometry(tmp_path)


def test_canonical_artifact_preserves_consumed_arrays():
    path = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if path is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to replay a real HSX artifact")
    path = Path(path)
    artifact = load_fci_simulation_geometry(path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert artifact.geometry.shape == tuple(manifest["shape"])
    assert artifact.topology_name == manifest["topology_name"]
    assert artifact.nfp == manifest["nfp"]
    assert artifact.metadata == manifest["metadata"]

    with np.load(path / "center_maps.npz", allow_pickle=False) as data:
        for field in fields(FciMaps3D):
            np.testing.assert_array_equal(getattr(artifact.geometry.maps, field.name), data[field.name])
    owner = artifact.polar_angular_geometry
    with np.load(path / "rlp_topology.npz", allow_pickle=False) as data:
        for field in fields(owner.topology):
            if field.name != "shape":
                np.testing.assert_array_equal(getattr(owner.topology, field.name), data[field.name])
        for field in fields(owner):
            if field.name in data:
                np.testing.assert_array_equal(getattr(owner, field.name), data[field.name])
    with np.load(path / "base_geometry.npz", allow_pickle=False) as data:
        metrics = [("cell_metric", artifact.geometry.cell_metric)] + [
            (f"face_metric.{axis}", getattr(artifact.geometry.face_metric, axis)) for axis in "xyz"
        ]
        for prefix, metric in metrics:
            for field in fields(metric):
                np.testing.assert_array_equal(
                    getattr(metric, field.name), data[f"{prefix}.{field.name}"]
                )
