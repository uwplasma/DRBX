from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_hermes_comparison_summary_package


def _write_parity(path: Path, errors: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "field_names": list(errors),
                "variable_errors": errors,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_create_hermes_comparison_summary_package_writes_summary_and_plot(tmp_path: Path) -> None:
    tokamak_one = tmp_path / "tokamak_one.json"
    tokamak_short = tmp_path / "tokamak_short.json"
    traced = tmp_path / "traced.json"
    stellarator = tmp_path / "stellarator.json"
    _write_parity(tokamak_one, {"Ne": {"relative_l2_error": 0.01, "max_abs_error": 0.2}})
    _write_parity(tokamak_short, {"phi": {"relative_l2_error": 0.02, "max_abs_error": 0.1}})
    _write_parity(traced, {"g33": {"relative_l2_error": 0.11, "max_abs_error": 0.3}})
    _write_parity(stellarator, {"pressure": {"relative_l2_error": 0.04, "max_abs_error": 0.25}})

    artifacts = create_hermes_comparison_summary_package(
        output_root=tmp_path / "output",
        tokamak_one_step_parity_json=tokamak_one,
        tokamak_short_window_parity_json=tokamak_short,
        traced_native_parity_json=traced,
        stellarator_native_parity_json=stellarator,
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["reference_code"] == "hermes-3"
    assert payload["lane_count"] == 4
