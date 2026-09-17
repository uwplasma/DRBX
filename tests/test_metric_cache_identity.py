"""Regression tests for stable HSX metric and derived-geometry cache keys."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drbx.geometry import hsx_fci_builder as hsx  # noqa: E402
from drbx.geometry.fci_geometry import (  # noqa: E402
    angular_agglomeration_host_geometry_cache_path,
)


def _new_spec(makegrid: Path, vessel: Path) -> dict[str, object]:
    return {
        "format_version": hsx.METRIC_CACHE_FORMAT_VERSION,
        "identity_version": hsx.METRIC_CACHE_IDENTITY_VERSION,
        "metric_builder_revision": hsx.METRIC_BUILDER_REVISION,
        "makegrid": hsx._metric_input_content_identity(makegrid),
        "makegrid_currents": [1.0, 2.0],
        "vessel": hsx._metric_input_content_identity(vessel),
        "resolution": [4, 6, 8],
        "topology": "toroidal",
    }


def test_metric_input_identity_ignores_path_and_mtime(tmp_path):
    first = tmp_path / "first.dat"
    second = tmp_path / "second.dat"
    first.write_bytes(b"same numerical input")
    second.write_bytes(first.read_bytes())
    second.touch()

    assert hsx._metric_input_content_identity(first) == (
        hsx._metric_input_content_identity(second)
    )

    second.write_bytes(b"same numerical inpuT")
    assert hsx._metric_input_content_identity(first) != (
        hsx._metric_input_content_identity(second)
    )


def test_legacy_metric_spec_ignores_paths_mtimes_and_source_mtimes(tmp_path):
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "wall.txt"
    makegrid.write_bytes(b"equilibrium")
    vessel.write_bytes(b"vessel")
    expected = _new_spec(makegrid, vessel)
    legacy = {
        key: value
        for key, value in expected.items()
        if key not in {"identity_version", "metric_builder_revision", "makegrid", "vessel"}
    }
    legacy.update(
        {
            "makegrid": {
                "path": "/an/old/checkout/mgrid.nc",
                "size": makegrid.stat().st_size,
                "mtime_ns": 1,
            },
            "vessel": {
                "path": "/an/old/checkout/wall.txt",
                "size": vessel.stat().st_size,
                "mtime_ns": 2,
            },
            "geometry_sources": {
                "drbx/geometry/MetricEvaluator.py": {
                    "size": 123,
                    "mtime_ns": 3,
                }
            },
        }
    )

    assert hsx._metric_cache_specs_compatible(legacy, expected)
    legacy["resolution"] = [8, 6, 8]
    assert not hsx._metric_cache_specs_compatible(legacy, expected)


def test_new_metric_spec_rejects_changed_content_or_builder_revision(tmp_path):
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "wall.txt"
    makegrid.write_bytes(b"equilibrium")
    vessel.write_bytes(b"vessel")
    expected = _new_spec(makegrid, vessel)

    changed_content = json.loads(json.dumps(expected))
    changed_content["makegrid"]["sha256"] = "0" * 64
    assert not hsx._metric_cache_specs_compatible(changed_content, expected)

    changed_revision = json.loads(json.dumps(expected))
    changed_revision["metric_builder_revision"] += 1
    assert not hsx._metric_cache_specs_compatible(changed_revision, expected)


def test_compatible_legacy_cache_is_discovered_without_loading_arrays(tmp_path):
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "wall.txt"
    makegrid.write_bytes(b"equilibrium")
    vessel.write_bytes(b"vessel")
    expected = _new_spec(makegrid, vessel)
    legacy = {
        key: value
        for key, value in expected.items()
        if key not in {"identity_version", "metric_builder_revision", "makegrid", "vessel"}
    }
    legacy["makegrid"] = {
        "path": "/old/mgrid.nc",
        "size": makegrid.stat().st_size,
        "mtime_ns": 1,
    }
    legacy["vessel"] = {
        "path": "/old/wall.txt",
        "size": vessel.stat().st_size,
        "mtime_ns": 1,
    }
    candidate = tmp_path / "hsx_metric_legacy.npz"
    np.savez(candidate, cache_spec=np.asarray(json.dumps(legacy)), huge=np.ones(4))

    selected = hsx._find_compatible_metric_cache(
        tmp_path, tmp_path / "hsx_metric_new.npz", expected
    )
    assert selected == candidate


def test_angular_host_key_uses_metric_contract_not_cache_path_or_mtime(tmp_path):
    spec_a = {
        "format_version": 7,
        "makegrid": {"path": "/old/a", "size": 10, "mtime_ns": 1},
        "vessel": {"path": "/old/b", "size": 20, "mtime_ns": 2},
        "resolution": [4, 6, 8],
    }
    spec_b = {
        "format_version": 7,
        "makegrid": {"path": "/new/a", "size": 10, "mtime_ns": 10},
        "vessel": {"path": "/new/b", "size": 20, "mtime_ns": 20},
        "resolution": [4, 6, 8],
    }
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "hsx_metric_a.npz"
    second = second_dir / "hsx_metric_b.npz"
    common = {
        "fci_maps_format_version": np.asarray(2),
        "fci_maps_source_fingerprint": np.asarray("fingerprint"),
        "fci_maps_trace_substeps": np.asarray(4),
    }
    np.savez(first, cache_spec=np.asarray(json.dumps(spec_a)), **common)
    np.savez(second, cache_spec=np.asarray(json.dumps(spec_b)), **common)
    faces = np.linspace(0.0, 1.0, 5)
    theta = np.linspace(0.0, 2.0 * np.pi, 7)
    eta = np.linspace(0.0, 2.0 * np.pi, 9)
    profile = np.asarray([6, 6, 3, 1])

    first_key = angular_agglomeration_host_geometry_cache_path(
        first, faces, theta, eta, profile
    )
    second_key = angular_agglomeration_host_geometry_cache_path(
        second, faces, theta, eta, profile
    )
    assert first_key is not None and second_key is not None
    assert first_key.name == second_key.name
