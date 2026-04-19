from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import build_hermes_capability_audit, write_hermes_capability_audit


def test_build_hermes_capability_audit_reports_major_open_families() -> None:
    audit = build_hermes_capability_audit()

    assert audit["reference_code"] == "hermes-3"
    assert audit["family_count"] >= 10
    families = {item["family"]: item for item in audit["families"]}
    assert families["neutral_mixed"]["jax_status"] == "partially_closed"
    assert families["neutral_mixed"]["jax_capability"] == "native_rhs_exact_bounded_full_short_window_metric_window"
    assert families["direct_tokamak_recycling"]["jax_status"] == "partially_closed"
    assert families["direct_tokamak_recycling"]["jax_capability"] == "native_exact_first_output_plus_bounded_two_output_and_neon_windows"
    assert families["impurity_radiation_and_detachment_control"]["jax_status"] == "partially_closed"
    assert families["non_tokamak_3d_geometry_adapters"]["jax_status"] == "closed_selected"
    assert "neutral_mixed" in audit["remaining_priority_families"]


def test_write_hermes_capability_audit_writes_json(tmp_path: Path) -> None:
    output = write_hermes_capability_audit(tmp_path / "audit.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["reference_code"] == "hermes-3"
    assert payload["closed_family_count"] >= 1
    assert payload["open_family_count"] >= 1
