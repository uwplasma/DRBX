from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_compare_modes_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_recycling_transient_modes.py"
    spec = importlib.util.spec_from_file_location("compare_recycling_transient_modes", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_compare_modes = _load_compare_modes_module()
_summarize_mode_errors = _compare_modes._summarize_mode_errors


def test_summarize_mode_errors_orders_fields_by_max_abs_diff() -> None:
    actual = {
        "Nd+": np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64),
        "Pe": np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64),
    }
    expected = {
        "Nd+": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "Pe": np.asarray([[1.0, 1.5], [2.0, 2.0]], dtype=np.float64),
    }

    rows = _summarize_mode_errors(actual, expected, fields=("Pe", "Nd+"))

    assert rows == [("Nd+", 1.0), ("Pe", 0.5)]


def test_summarize_mode_errors_marks_shape_mismatch_as_infinite() -> None:
    rows = _summarize_mode_errors(
        {"Nd+": np.zeros((2, 3), dtype=np.float64)},
        {"Nd+": np.zeros((2, 4), dtype=np.float64)},
        fields=("Nd+",),
    )

    assert len(rows) == 1
    assert rows[0][0] == "Nd+"
    assert np.isinf(rows[0][1])


def test_summarize_mode_errors_trims_guard_cells_with_mesh() -> None:
    class Mesh:
        xstart = 1
        xend = 1
        ystart = 1
        yend = 2

    actual = {
        "Nd+": np.arange(2 * 3 * 4 * 1, dtype=np.float64).reshape(2, 3, 4, 1),
    }
    expected = {
        "Nd+": actual["Nd+"][:, 1:2, 1:3, :].copy(),
    }

    rows = _summarize_mode_errors(actual, expected, fields=("Nd+",), mesh=Mesh())

    assert rows == [("Nd+", 0.0)]
