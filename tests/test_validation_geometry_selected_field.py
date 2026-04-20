from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation.geometry_selected_field import (
    compare_geometry_selected_fields,
    save_geometry_selected_field_parity_plot,
    write_geometry_selected_field_parity_arrays,
    write_geometry_selected_field_parity_json,
)


def test_compare_geometry_selected_fields_detects_offsets(tmp_path: Path) -> None:
    result = compare_geometry_selected_fields(
        reference_fields={"A": np.ones((2, 3)), "B": np.ones((2, 3)) * 2.0},
        candidate_fields={"A": np.ones((2, 3)) * 1.1, "B": np.ones((2, 3)) * 2.0},
        field_names=("A", "B"),
    )
    assert result.variable_errors["A"].max_abs_error > 0.0
    assert result.variable_errors["B"].max_abs_error == 0.0
    json_path = write_geometry_selected_field_parity_json(result, tmp_path / "parity.json")
    arrays_path = write_geometry_selected_field_parity_arrays(result, tmp_path / "parity.npz")
    plot_path = save_geometry_selected_field_parity_plot(result, tmp_path / "parity.png", title="demo")
    assert json.loads(json_path.read_text(encoding="utf-8"))["field_names"] == ["A", "B"]
    assert arrays_path.exists()
    assert plot_path.exists()
