from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.neutral_parallel_diffusion_campaign import (
    NeutralParallelDiffusionCampaignMetric,
    build_neutral_parallel_diffusion_campaign,
    create_neutral_parallel_diffusion_campaign_package,
)


_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "recycling_dthe_rhs_snapshot.npz"
)


def test_create_neutral_parallel_diffusion_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.neutral_parallel_diffusion_campaign.build_neutral_parallel_diffusion_campaign",
        lambda **kwargs: (
            NeutralParallelDiffusionCampaignMetric(
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
        {"xstart": 0, "xend": 0, "ystart": 0, "yend": 1, "y": np.asarray([0.0, 1.0], dtype=np.float64)},
    )()
    fake_context.species = {
        "d": type("FakeSpecies", (), {"density": np.ones((1, 2, 1), dtype=np.float64)})(),
        "t": type("FakeSpecies", (), {"density": np.ones((1, 2, 1), dtype=np.float64)})(),
        "he": type("FakeSpecies", (), {"density": np.ones((1, 2, 1), dtype=np.float64)})(),
    }
    fake_context.terms = type(
        "FakeTerms",
        (),
        {
            "diagnostics": {
                "Dd_Dpar": np.ones((1, 2, 1), dtype=np.float64),
                "Dt_Dpar": 2.0 * np.ones((1, 2, 1), dtype=np.float64),
                "Dhe_Dpar": 3.0 * np.ones((1, 2, 1), dtype=np.float64),
            },
        },
    )()
    fake_context.ionisation_rates = {
        "d": 4.0 * np.ones((1, 2, 1), dtype=np.float64),
        "t": 5.0 * np.ones((1, 2, 1), dtype=np.float64),
        "he": 6.0 * np.ones((1, 2, 1), dtype=np.float64),
    }
    fake_context.charge_exchange_rates = {
        "d": np.ones((1, 2, 1), dtype=np.float64),
        "t": 2.0 * np.ones((1, 2, 1), dtype=np.float64),
        "he": np.zeros((1, 2, 1), dtype=np.float64),
    }
    fake_context.multispecies_collision_totals = {
        "d": 3.0 * np.ones((1, 2, 1), dtype=np.float64),
        "t": 4.0 * np.ones((1, 2, 1), dtype=np.float64),
        "he": 5.0 * np.ones((1, 2, 1), dtype=np.float64),
    }
    monkeypatch.setattr(
        "jax_drb.validation.neutral_parallel_diffusion_campaign.build_neutral_parallel_diffusion_context",
        lambda *args, **kwargs: fake_context,
    )
    artifacts = create_neutral_parallel_diffusion_campaign_package(
        output_root=tmp_path / "output",
        input_path=tmp_path / "BOUT.inp",
        dump_path=tmp_path / "BOUT.dmp.0.nc",
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "neutral_parallel_diffusion"
    assert payload["metric_count"] == 6
    assert payload["passed_metric_count"] == 3
    assert payload["species_order"] == ["d", "t", "he"]
    assert sorted(payload["species_summaries"]) == [
        "afn_charge_exchange_rate",
        "afn_diffusion",
        "afn_ionisation_rate",
        "diffusion_ratio",
        "multispecies_charge_exchange_rate",
        "multispecies_collision_rate",
        "multispecies_diffusion",
    ]


def test_build_neutral_parallel_diffusion_campaign_passes_reference_checks() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    metrics = build_neutral_parallel_diffusion_campaign(
        input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
        dump_path=_SNAPSHOT_CACHE_PATH,
    )
    assert [metric.name for metric in metrics] == [
        "afn_diffusion_finite_fraction",
        "multispecies_diffusion_finite_fraction",
        "d_afn_to_multispecies_diffusion_contrast",
        "t_afn_to_multispecies_diffusion_contrast",
        "he_afn_to_multispecies_diffusion_contrast",
        "deuterium_afn_charge_exchange_fraction",
    ]
    assert all(metric.passed for metric in metrics)


def test_create_neutral_parallel_diffusion_campaign_package_writes_profile_arrays() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_neutral_parallel_diffusion_campaign_package(
            output_root=tmp_root,
            input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
            dump_path=_SNAPSHOT_CACHE_PATH,
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert payload["species_order"] == ["d", "t", "he"]
        arrays = np.load(artifacts.arrays_npz_path)
        assert "afn_diffusion" in arrays
        assert "multispecies_diffusion" in arrays
        assert "afn_ionisation_rate" in arrays
        assert "afn_charge_exchange_rate" in arrays
        assert "multispecies_collision_rate" in arrays
        assert "multispecies_charge_exchange_rate" in arrays
        assert "diffusion_ratio" in arrays
