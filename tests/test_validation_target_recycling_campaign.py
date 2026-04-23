from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.target_recycling_campaign import (
    TargetRecyclingCampaignMetric,
    build_target_recycling_campaign,
    create_target_recycling_campaign_package,
)


_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "recycling_dthe_rhs_snapshot.npz"
)


def test_create_target_recycling_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.target_recycling_campaign.build_target_recycling_campaign",
        lambda **kwargs: (
            TargetRecyclingCampaignMetric(
                name="metric_a",
                kind="min_value",
                value=1.0,
                target=1.0e-4,
                passed=True,
                notes="ok",
            ),
        ),
    )
    fake_context = type("FakeContext", (), {})()
    fake_context.mesh = type(
        "FakeMesh",
        (),
        {"xstart": 0, "xend": 0, "ystart": 0, "yend": 2, "y": np.asarray([0.0, 1.0, 2.0], dtype=np.float64)},
    )()
    fake_context.target_terms = type(
        "FakeTerms",
        (),
        {
            "diagnostics": {
                "Sd_target_recycle": np.asarray([[[1.0], [0.5], [0.0]]], dtype=np.float64),
                "St_target_recycle": np.asarray([[[0.8], [0.4], [0.0]]], dtype=np.float64),
                "She_target_recycle": np.asarray([[[0.1], [0.05], [0.0]]], dtype=np.float64),
                "Ed_target_recycle": np.asarray([[[0.6], [0.2], [0.0]]], dtype=np.float64),
                "Et_target_recycle": np.asarray([[[0.4], [0.1], [0.0]]], dtype=np.float64),
                "Ehe_target_recycle": np.asarray([[[0.08], [0.02], [0.0]]], dtype=np.float64),
            },
        },
    )()
    fake_context.electron_boundary = type(
        "FakeElectronBoundary",
        (),
        {"energy_source": np.asarray([[[0.2], [0.1], [0.0]]], dtype=np.float64)},
    )()
    fake_context.electron_zero_current_velocity = np.asarray([[[0.0], [0.1], [0.2]]], dtype=np.float64)
    monkeypatch.setattr(
        "jax_drb.validation.target_recycling_campaign.build_target_recycling_context",
        lambda **kwargs: fake_context,
    )

    artifacts = create_target_recycling_campaign_package(
        output_root=tmp_path / "output",
        input_path=tmp_path / "BOUT.inp",
        snapshot_path=tmp_path / "snapshot.npz",
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "target_recycling"
    assert payload["passed_metric_count"] == 1
    assert payload["neutral_order"] == ["d", "t", "he"]


def test_build_target_recycling_campaign_passes_reference_checks() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    metrics = build_target_recycling_campaign(
        input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
        snapshot_path=_SNAPSHOT_CACHE_PATH,
    )
    assert [metric.name for metric in metrics] == [
        "d_target_recycling_density_peak",
        "t_target_recycling_density_peak",
        "he_target_recycling_density_peak",
        "electron_sheath_energy_peak",
        "electron_zero_current_velocity_finite_fraction",
    ]
    assert all(metric.passed for metric in metrics)


def test_create_target_recycling_campaign_package_writes_arrays() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_target_recycling_campaign_package(
            output_root=tmp_root,
            input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
            snapshot_path=_SNAPSHOT_CACHE_PATH,
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert payload["neutral_order"] == ["d", "t", "he"]
        arrays = np.load(artifacts.arrays_npz_path)
        assert "target_recycling_density_peak" in arrays
        assert "target_recycling_density_integral" in arrays
        assert "target_recycling_energy_integral" in arrays
        assert "d_target_energy_line" in arrays
        assert "electron_sheath_energy_line" in arrays
        assert "electron_zero_current_velocity_line" in arrays
