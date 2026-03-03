from __future__ import annotations

from pathlib import Path

from jaxdrb.benchmarking import compare_bundle_diagnostics, load_bundle_npz


def test_hermes_short_window_compact_fixture_loads_and_self_matches():
    fixture = Path(__file__).resolve().parent / "fixtures" / "hermes_short_window_compact.npz"
    bundle = load_bundle_npz(fixture)
    assert bundle.code == "hermes"
    assert bundle.geometry == "tokamak_open_field"
    assert "n_fluct_last_xz" in bundle.snapshots
    assert "rms_n_fluct" in bundle.diagnostics

    out = compare_bundle_diagnostics(bundle, bundle)
    assert out.mean_rel_l2 == 0.0
    assert out.max_rel_l2 == 0.0
