from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation.collision_closure_campaign import (
    CollisionClosureCampaignMetric,
    build_collision_closure_campaign,
    create_collision_closure_campaign_package,
)


_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "recycling_dthe_rhs_snapshot.npz"
)


def test_create_collision_closure_campaign_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_drb.validation.collision_closure_campaign.build_collision_closure_campaign",
        lambda **kwargs: (
            CollisionClosureCampaignMetric(
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
    fake_context.mesh = type("FakeMesh", (), {"xstart": 0, "xend": 0, "ystart": 0, "yend": 0})()
    fake_context.closure_terms = type(
        "FakeTerms",
        (),
        {
            "diagnostics": {
                "DivPiPar_d+": np.ones((1, 1, 1), dtype=np.float64),
                "DivPiPar_t+": 2.0 * np.ones((1, 1, 1), dtype=np.float64),
                "DivPiPar_he+": 3.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fd+t+_coll": 4.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Ft+d+_coll": -4.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fd+e_coll": 5.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fed+_coll": -5.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Ft+e_coll": 6.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fet+_coll": -6.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fhe+e_coll": 7.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Fehe+_coll": -7.0 * np.ones((1, 1, 1), dtype=np.float64),
                "Ed+t+_coll_friction": 0.1 * np.ones((1, 1, 1), dtype=np.float64),
                "Ed+e_coll_friction": 0.2 * np.ones((1, 1, 1), dtype=np.float64),
                "Et+e_coll_friction": 0.3 * np.ones((1, 1, 1), dtype=np.float64),
                "Ehe+e_coll_friction": 0.4 * np.ones((1, 1, 1), dtype=np.float64),
            },
        },
    )()
    fake_context.conduction_times = {
        "e": 10.0 * np.ones((1, 1, 1), dtype=np.float64),
        "d+": 11.0 * np.ones((1, 1, 1), dtype=np.float64),
        "t+": 12.0 * np.ones((1, 1, 1), dtype=np.float64),
        "he+": 13.0 * np.ones((1, 1, 1), dtype=np.float64),
        "d": 14.0 * np.ones((1, 1, 1), dtype=np.float64),
        "t": 15.0 * np.ones((1, 1, 1), dtype=np.float64),
        "he": 16.0 * np.ones((1, 1, 1), dtype=np.float64),
    }
    monkeypatch.setattr(
        "jax_drb.validation.collision_closure_campaign.build_collision_closure_context",
        lambda **kwargs: fake_context,
    )

    artifacts = create_collision_closure_campaign_package(
        output_root=tmp_path / "output",
        input_path=tmp_path / "BOUT.inp",
        snapshot_path=tmp_path / "snapshot.npz",
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["family"] == "collision_closure"
    assert payload["passed_metric_count"] == 1
    assert payload["ion_species_order"] == ["d+", "t+", "he+"]
    assert payload["pressure_species_order"] == ["e", "d+", "t+", "he+", "d", "t", "he"]


def test_build_collision_closure_campaign_passes_reference_checks() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    metrics = build_collision_closure_campaign(
        input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
        snapshot_path=_SNAPSHOT_CACHE_PATH,
    )
    assert [metric.name for metric in metrics] == [
        "d_plus_ion_viscosity_peak",
        "t_plus_ion_viscosity_peak",
        "he_plus_ion_viscosity_peak",
        "d_plus_t_plus_action_reaction_residual",
        "d_plus_e_action_reaction_residual",
        "t_plus_e_action_reaction_residual",
        "he_plus_e_action_reaction_residual",
        "pressure_species_conduction_time_finite_fraction",
    ]
    assert all(metric.passed for metric in metrics)


def test_create_collision_closure_campaign_package_writes_arrays() -> None:
    reference_root = default_reference_root()
    if reference_root is None:
        pytest.skip("external reference decks are not available")
    with tempfile.TemporaryDirectory() as tmp_root:
        artifacts = create_collision_closure_campaign_package(
            output_root=tmp_root,
            input_path=reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
            snapshot_path=_SNAPSHOT_CACHE_PATH,
        )
        payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
        assert payload["friction_pair_order"] == [["d+", "t+"], ["d+", "e"], ["t+", "e"], ["he+", "e"]]
        arrays = np.load(artifacts.arrays_npz_path)
        assert "ion_viscosity_peak" in arrays
        assert "conduction_time_active_point" in arrays
        assert "friction_pair_peak" in arrays
        assert "friction_action_reaction_residual" in arrays
        assert "energy_exchange_peak" in arrays
