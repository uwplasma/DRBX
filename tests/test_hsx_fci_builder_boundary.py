"""Focused tests for the producer-side HSX builder import boundary."""

from pathlib import Path
import builtins

import numpy as np

from drbx.geometry import hsx_fci_builder, hsx_simulation_geometry


def test_package_builder_has_no_simulation_driver_dependency():
    source = Path(hsx_fci_builder.__file__).read_text(encoding="utf-8")
    assert "simulate_hsx_blob" not in source


def test_hsx_producer_calls_package_builder(monkeypatch, tmp_path):
    calls = []

    class Geometry:
        shape = (4, 4, 8)

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return Geometry(), np.zeros((4, 4, 8, 3)), 2, None, object()

    monkeypatch.setattr(hsx_fci_builder, "build_hsx_fci_geometry", fake_builder)
    real_import = builtins.__import__

    def block_driver_import(name, *args, **kwargs):
        if name == "simulate_hsx_blob":
            raise AssertionError("producer imported the simulation driver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_driver_import)
    config = hsx_simulation_geometry.HsxSimulationGeometryConfig(
        makegrid_path=Path("makegrid").resolve(),
        vessel_path=Path("vessel").resolve(),
        resolution=(4, 4, 8),
    )

    result = hsx_simulation_geometry._call_builder(config)

    assert result[2] == 2
    assert calls and calls[0]["fci_trace_substeps"] == 64
    assert calls[0]["construct_fci_maps"] is True
