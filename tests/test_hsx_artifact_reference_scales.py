"""Physical reference scales recorded in the HSX simulation-geometry manifest."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drbx.geometry import hsx_fci_builder, hsx_simulation_geometry as producer
from drbx.geometry.fci_simulation_geometry import (
    audit_fci_simulation_geometry_checksums,
    load_fci_simulation_geometry,
    write_fci_simulation_geometry,
)
from tests.test_fci_simulation_geometry_artifact import _artifact

B0_TESLA = 1.0198142727751343


def _producer_run(monkeypatch, tmp_path: Path, payload_factory):
    shape = (3, 4, 4)
    grid = SimpleNamespace(
        x=SimpleNamespace(faces=np.linspace(0, 1, 4), centers=np.array([1 / 6, 0.5, 5 / 6])),
        y=SimpleNamespace(
            faces=np.linspace(0, 2 * np.pi, 5),
            centers=np.linspace(np.pi / 4, 7 * np.pi / 4, 4),
        ),
        z=SimpleNamespace(faces=np.linspace(0, 4, 5), centers=np.array([0.5, 1.5, 2.5, 3.5])),
    )
    geometry = SimpleNamespace(
        shape=shape,
        grid=grid,
        maps=SimpleNamespace(
            forward_boundary=np.zeros(shape, dtype=bool),
            backward_boundary=np.zeros(shape, dtype=bool),
        ),
    )
    positions = np.zeros(shape + (3,))
    monkeypatch.setattr(
        producer, "_call_builder", lambda _config: payload_factory(geometry, positions)
    )
    monkeypatch.setattr(producer, "_continuous_field_callback", lambda *_a: object())
    monkeypatch.setattr(producer, "_trace_atlas", lambda *_a, **_k: object())
    monkeypatch.setattr(producer, "_read_stage_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(producer, "_write_stage_checkpoint", lambda *_a, **_k: None)
    from drbx.geometry import fci_simulation_geometry as artifact_io

    for name in ("_vertex_payload", "_polar_payload", "_overlap_payload"):
        monkeypatch.setattr(artifact_io, name, lambda _value: {})
    monkeypatch.setattr(
        producer,
        "build_metric_aware_polar_angular_agglomeration_geometry",
        lambda *_a, **_k: (object(), 1.0),
    )
    monkeypatch.setattr(
        producer, "build_owner_boundary_overlap_geometry", lambda *_a, **_k: object()
    )

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        producer,
        "_artifact_api",
        lambda: (FakeArtifact, object, lambda *_a, **_k: {"valid": True}, lambda *_a, **_k: None),
    )
    makegrid = tmp_path / "mgrid.nc"
    vessel = tmp_path / "vessel.txt"
    makegrid.write_bytes(b"mgrid")
    vessel.write_text("vessel")
    config = producer.HsxSimulationGeometryConfig(
        makegrid_path=makegrid,
        vessel_path=vessel,
        resolution=shape,
        output=tmp_path / "geometry",
    )
    return producer.build_hsx_simulation_geometry(config)


def test_producer_records_b0_tesla_and_length_unit(monkeypatch, tmp_path: Path) -> None:
    result = _producer_run(
        monkeypatch,
        tmp_path,
        lambda geometry, positions: (
            geometry, positions, 4, None, object(), None, None, None, B0_TESLA
        ),
    )
    assert result.metadata["reference_magnetic_field_tesla"] == B0_TESLA
    assert result.metadata["length_unit"] == "m"
    # The (null) *requested* B0 keeps its old meaning and is not overwritten.
    assert result.metadata["reference_magnetic_field"] is None


@pytest.mark.parametrize("legacy_length", [6, 8])
def test_producer_accepts_builder_doubles_without_b0(
    monkeypatch, tmp_path: Path, legacy_length: int
) -> None:
    def payload(geometry, positions):
        full = (geometry, positions, 4, None, object(), None, None, None)
        if legacy_length == 6:
            return (geometry, positions, 4, None, object(), None)
        return full

    result = _producer_run(monkeypatch, tmp_path, payload)
    assert result.metadata["reference_magnetic_field_tesla"] is None
    assert result.metadata["length_unit"] == "m"


def test_call_builder_reads_b0_from_metric_context(monkeypatch) -> None:
    class Geometry:
        shape = (4, 4, 8)

    def fake_builder(**kwargs):
        context = hsx_fci_builder.HSXMetricContext(
            SimpleNamespace(), object(), 2, None, reference_magnetic_field=B0_TESLA
        )
        return Geometry(), np.zeros((4, 4, 8, 3)), 2, None, context

    monkeypatch.setattr(hsx_fci_builder, "build_hsx_fci_geometry", fake_builder)
    config = producer.HsxSimulationGeometryConfig(
        makegrid_path=Path("makegrid").resolve(),
        vessel_path=Path("vessel").resolve(),
        resolution=(4, 4, 8),
    )
    payload = producer._call_builder(config)
    assert len(payload) == 9
    assert payload[8] == B0_TESLA

    def builder_without_b0(**kwargs):
        context = hsx_fci_builder.HSXMetricContext(SimpleNamespace(), object(), 2)
        return Geometry(), np.zeros((4, 4, 8, 3)), 2, None, context

    monkeypatch.setattr(hsx_fci_builder, "build_hsx_fci_geometry", builder_without_b0)
    assert producer._call_builder(config)[8] is None

    for bad in (float("nan"), -1.0, 0.0, float("inf")):
        def builder_with_bad_b0(bad=bad, **kwargs):
            context = hsx_fci_builder.HSXMetricContext(
                SimpleNamespace(), object(), 2, reference_magnetic_field=bad
            )
            return Geometry(), np.zeros((4, 4, 8, 3)), 2, None, context

        monkeypatch.setattr(
            hsx_fci_builder, "build_hsx_fci_geometry", builder_with_bad_b0
        )
        assert producer._call_builder(config)[8] is None


def test_manifest_round_trip_new_keys_and_old_manifest_loads(tmp_path: Path) -> None:
    new = _artifact()
    new = replace(
        new,
        metadata={
            **new.metadata,
            "reference_magnetic_field_tesla": B0_TESLA,
            "length_unit": "m",
        },
    )
    target = write_fci_simulation_geometry(new, tmp_path / "new")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["metadata"]["reference_magnetic_field_tesla"] == B0_TESLA
    assert manifest["metadata"]["length_unit"] == "m"
    loaded = load_fci_simulation_geometry(target)
    assert loaded.metadata["reference_magnetic_field_tesla"] == B0_TESLA
    assert loaded.metadata["length_unit"] == "m"

    # An artifact written before the keys existed loads, with them unknown.
    old = write_fci_simulation_geometry(_artifact(), tmp_path / "old")
    old_loaded = load_fci_simulation_geometry(old)
    assert "reference_magnetic_field_tesla" not in old_loaded.metadata
    assert old_loaded.metadata.get("reference_magnetic_field_tesla") is None
    assert old_loaded.metadata.get("length_unit") is None

    # Stripping the keys from a new manifest in place keeps it loadable and
    # checksum-valid: the manifest metadata is not a hash input.
    manifest["metadata"].pop("reference_magnetic_field_tesla")
    manifest["metadata"].pop("length_unit")
    (target / "manifest.json").write_text(json.dumps(manifest))
    assert audit_fci_simulation_geometry_checksums(target)["valid"] is True
    assert "length_unit" not in load_fci_simulation_geometry(target).metadata


def test_component_checksums_do_not_depend_on_new_metadata(tmp_path: Path) -> None:
    plain = write_fci_simulation_geometry(_artifact(), tmp_path / "plain")
    annotated_artifact = _artifact()
    annotated_artifact = replace(
        annotated_artifact,
        metadata={
            **annotated_artifact.metadata,
            "reference_magnetic_field_tesla": B0_TESLA,
            "length_unit": "m",
        },
    )
    annotated = write_fci_simulation_geometry(annotated_artifact, tmp_path / "annotated")
    plain_sums = json.loads((plain / "manifest.json").read_text())["checksums"]
    annotated_sums = json.loads((annotated / "manifest.json").read_text())["checksums"]
    assert plain_sums == annotated_sums
