from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_drb.native.runner_compare import prepare_compare_variables, select_payload_variables


def test_prepare_compare_variables_can_trim_x_and_y_guards() -> None:
    mesh = SimpleNamespace(mxg=1, myg=2)
    values = {
        "A": np.arange(2 * 5 * 8 * 3, dtype=np.float64).reshape(2, 5, 8, 3),
        "B": np.arange(2 * 2 * 3 * 1, dtype=np.float64).reshape(2, 2, 3, 1),
    }

    prepared = prepare_compare_variables(values, mesh, trim_x_guards=True, trim_y_guards=True)

    np.testing.assert_allclose(prepared["A"], values["A"][:, 1:-1, 2:-2, :])
    np.testing.assert_allclose(prepared["B"], values["B"])


def test_select_payload_variables_respects_compare_variable_order_and_presence() -> None:
    values = {
        "Pe": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "Nd+": np.asarray([[3.0, 4.0]], dtype=np.float64),
    }

    selected = select_payload_variables(values, compare_variables=("Nd+", "missing", "Pe"))

    assert tuple(selected) == ("Nd+", "Pe")
    np.testing.assert_allclose(selected["Nd+"], values["Nd+"])
    np.testing.assert_allclose(selected["Pe"], values["Pe"])
