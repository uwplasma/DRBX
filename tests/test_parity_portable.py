from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.parity.portable import build_portable_summary_payload, write_portable_summary_payload


def test_build_portable_summary_payload_matches_expected_shape() -> None:
    payload = build_portable_summary_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 2, "x": 3},
        time_points=(0.0, 1.0),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 2.5]])},
        overrides=("nout=1",),
        configured_nout=5,
        configured_timestep=1.0,
    )

    assert payload["case_name"] == "toy"
    assert payload["capability_tier"] == "native_exact"
    assert payload["variable_summaries"]["Ne"]["max_abs_delta_last_first"] == 0.5
    assert payload["effective_output_points"] == 2


def test_build_portable_summary_payload_skips_missing_variables_and_uses_fallback_dimensions() -> None:
    payload = build_portable_summary_payload(
        case_name="toy",
        parity_mode="one_rhs",
        capability_tier="native_exact",
        compare_variables=("Ne", "Te"),
        component_labels=("e:evolve_density",),
        dimensions={"x": 3, "y": 4},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.ones((1, 3, 4))},
    )

    assert set(payload["variable_summaries"]) == {"Ne"}
    assert payload["variable_summaries"]["Ne"]["dimensions"] == ["t", "dim_1", "dim_2"]
    assert "configured_nout" not in payload
    assert "configured_timestep" not in payload


def test_write_portable_summary_payload_serializes_json(tmp_path: Path) -> None:
    path = write_portable_summary_payload({"case_name": "toy"}, tmp_path / "portable.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case_name"] == "toy"
