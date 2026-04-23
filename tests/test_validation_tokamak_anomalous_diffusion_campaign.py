from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.tokamak_anomalous_diffusion_campaign import (
    TokamakAnomalousDiffusionCampaignMetric,
    build_tokamak_anomalous_diffusion_campaign,
    create_tokamak_anomalous_diffusion_campaign_package,
)


_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "tokamak_recycling_dthe_rhs_snapshot.npz"
)
_ARRAY_HISTORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_arrays"
    / "tokamak_recycling_dthe_one_step.npz"
)
_OPTIONAL_HISTORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "tokamak_recycling_dthe_one_step_optional_history.npz"
)


def test_create_tokamak_anomalous_diffusion_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.tokamak_anomalous_diffusion_campaign.build_tokamak_anomalous_diffusion_campaign",
        lambda **kwargs: (
            TokamakAnomalousDiffusionCampaignMetric(
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
    fake_context.nonorthogonal_terms = type(
        "FakeTerms",
        (),
        {
            "diagnostics": {
                "anomalous_D_d+": np.full((1, 3, 1), 0.25, dtype=np.float64),
                "anomalous_D_t+": np.full((1, 3, 1), 0.25, dtype=np.float64),
                "anomalous_D_he+": np.full((1, 3, 1), 0.25, dtype=np.float64),
                "anomalous_D_e": np.full((1, 3, 1), 0.25, dtype=np.float64),
                "anomalous_Chi_d+": np.full((1, 3, 1), 0.5, dtype=np.float64),
                "anomalous_Chi_t+": np.full((1, 3, 1), 0.5, dtype=np.float64),
                "anomalous_Chi_he+": np.full((1, 3, 1), 0.5, dtype=np.float64),
                "anomalous_Chi_e": np.full((1, 3, 1), 0.5, dtype=np.float64),
            },
            "density_source": {
                "d+": np.full((1, 3, 1), 1.0e-6, dtype=np.float64),
                "t+": np.full((1, 3, 1), 2.0e-6, dtype=np.float64),
                "he+": np.full((1, 3, 1), 3.0e-6, dtype=np.float64),
                "e": np.full((1, 3, 1), 4.0e-6, dtype=np.float64),
            },
            "energy_source": {
                "d+": np.asarray([[[1.0e-4], [1.2e-4], [1.1e-4]]], dtype=np.float64),
                "t+": np.asarray([[[9.0e-5], [1.0e-4], [1.1e-4]]], dtype=np.float64),
                "he+": np.asarray([[[1.0e-6], [1.2e-6], [1.1e-6]]], dtype=np.float64),
                "e": np.asarray([[[2.0e-6], [2.2e-6], [2.1e-6]]], dtype=np.float64),
            },
            "momentum_source": {
                "d+": np.full((1, 3, 1), 1.0e-7, dtype=np.float64),
                "t+": np.full((1, 3, 1), 2.0e-7, dtype=np.float64),
                "he+": np.full((1, 3, 1), 3.0e-7, dtype=np.float64),
                "e": np.zeros((1, 3, 1), dtype=np.float64),
            },
        },
    )()
    fake_context.orthogonal_terms = type(
        "FakeTerms",
        (),
        {
            "density_source": {
                "d+": np.full((1, 3, 1), 0.5e-6, dtype=np.float64),
                "t+": np.full((1, 3, 1), 1.0e-6, dtype=np.float64),
                "he+": np.full((1, 3, 1), 1.5e-6, dtype=np.float64),
                "e": np.full((1, 3, 1), 2.0e-6, dtype=np.float64),
            },
            "energy_source": {
                "d+": np.asarray([[[8.0e-5], [8.5e-5], [8.2e-5]]], dtype=np.float64),
                "t+": np.asarray([[[7.0e-5], [7.5e-5], [7.3e-5]]], dtype=np.float64),
                "he+": np.asarray([[[0.8e-6], [0.9e-6], [0.85e-6]]], dtype=np.float64),
                "e": np.asarray([[[1.9e-6], [2.0e-6], [1.95e-6]]], dtype=np.float64),
            },
            "momentum_source": {
                "d+": np.full((1, 3, 1), 0.5e-7, dtype=np.float64),
                "t+": np.full((1, 3, 1), 1.0e-7, dtype=np.float64),
                "he+": np.full((1, 3, 1), 1.5e-7, dtype=np.float64),
                "e": np.zeros((1, 3, 1), dtype=np.float64),
            },
        },
    )()
    monkeypatch.setattr(
        "jax_drb.validation.tokamak_anomalous_diffusion_campaign.build_tokamak_anomalous_diffusion_context",
        lambda **kwargs: fake_context,
    )

    artifacts = create_tokamak_anomalous_diffusion_campaign_package(
        output_root=tmp_path / "output",
        input_path=tmp_path / "BOUT.inp",
        mesh_path=tmp_path / "tokamak.nc",
        snapshot_path=tmp_path / "snapshot.npz",
        array_history_path=tmp_path / "history.npz",
        optional_history_path=tmp_path / "optional_history.npz",
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "tokamak_anomalous_diffusion"
    assert payload["passed_metric_count"] == 1
    assert payload["species_order"] == ["d+", "t+", "he+", "e"]


def test_build_tokamak_anomalous_diffusion_campaign_passes_reference_checks() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    metrics = build_tokamak_anomalous_diffusion_campaign(
        input_path=reference_root / "examples" / "tokamak-2D" / "recycling-dthe" / "BOUT.inp",
        mesh_path=reference_root / "examples" / "tokamak-2D" / "recycling-dthe" / "tokamak.nc",
        snapshot_path=_SNAPSHOT_CACHE_PATH,
        array_history_path=_ARRAY_HISTORY_PATH,
        optional_history_path=_OPTIONAL_HISTORY_PATH,
    )
    assert [metric.name for metric in metrics] == [
        "electron_anomalous_d_matches_d_plus",
        "d_plus_energy_relative_contrast",
        "t_plus_energy_relative_contrast",
        "he_plus_energy_relative_contrast",
        "electron_density_contrast_peak",
    ]
    assert all(metric.passed for metric in metrics)


def test_create_tokamak_anomalous_diffusion_campaign_package_writes_arrays() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_tokamak_anomalous_diffusion_campaign_package(
            output_root=tmp_root,
            input_path=reference_root / "examples" / "tokamak-2D" / "recycling-dthe" / "BOUT.inp",
            mesh_path=reference_root / "examples" / "tokamak-2D" / "recycling-dthe" / "tokamak.nc",
            snapshot_path=_SNAPSHOT_CACHE_PATH,
            array_history_path=_ARRAY_HISTORY_PATH,
            optional_history_path=_OPTIONAL_HISTORY_PATH,
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert payload["species_order"] == ["d+", "t+", "he+", "e"]
        arrays = np.load(artifacts.arrays_npz_path)
        assert "anomalous_D_active_point" in arrays
        assert "anomalous_Chi_active_point" in arrays
        assert "density_contrast_peak" in arrays
        assert "energy_relative_contrast" in arrays
        assert "d_plus_energy_delta_line" in arrays
        assert "t_plus_energy_delta_line" in arrays
