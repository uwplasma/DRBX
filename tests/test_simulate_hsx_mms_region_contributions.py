"""Regression tests for dependency-aware Stage-7 regional accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "simulate_hsx_mms.py"


def _load():
    name = "simulate_hsx_mms_region_contribution_test"
    spec = importlib.util.spec_from_file_location(name, DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_raw_footprint_dilation_wraps_only_periodic_axes():
    driver = _load()
    seed = np.zeros((5, 6, 4), dtype=bool)
    seed[0, 0, 0] = True
    dilated = driver._dilate_raw_mask(seed, (1, 1, 1))
    assert not dilated[-1, 0, 0]
    assert dilated[0, -1, 0]
    assert dilated[0, 0, -1]
    assert dilated[1, 0, 0]


def test_region_statistics_reconstruct_global_squared_error_exactly():
    driver = _load()
    shape = (3, 4, 2)
    active = np.ones(shape, dtype=bool)
    volume = 1.0 + np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0
    host = type("Host", (), {
        "topology": type("Topology", (), {"is_active_owner": active})(),
        "aggregate_chart_volume": volume,
    })()
    first = np.zeros(shape, dtype=bool)
    first[:1] = True
    second = ~first
    error = np.linspace(-0.7, 1.1, np.prod(shape)).reshape(shape)
    statistics = driver._partitioned_error_statistics(
        error, host, {"first": first, "second": second}
    )
    total_volume = float(np.sum(volume))
    total_squared_error = float(np.sum(volume * error**2))
    assert np.isclose(
        sum(item["volume"] for item in statistics.values()), total_volume
    )
    assert np.isclose(
        sum(item["squared_error"] for item in statistics.values()),
        total_squared_error,
    )
    assert np.isclose(
        sum(
            item["global_mean_squared_error_contribution"]
            for item in statistics.values()
        ),
        total_squared_error / total_volume,
    )
    assert np.isclose(
        sum(item["volume_fraction"] for item in statistics.values()), 1.0
    )
    assert np.isclose(
        sum(item["squared_error_fraction"] for item in statistics.values()), 1.0
    )
