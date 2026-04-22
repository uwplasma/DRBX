from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def prepare_compare_variables(
    variables: Mapping[str, Any],
    mesh: Any,
    *,
    trim_x_guards: bool,
    trim_y_guards: bool,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for name, value in variables.items():
        array = np.asarray(value, dtype=np.float64)
        if trim_x_guards and array.ndim >= 4 and array.shape[1] > 2 * mesh.mxg:
            array = array[:, mesh.mxg : -mesh.mxg, ...]
        if trim_y_guards and array.ndim >= 4 and array.shape[2] > 2 * mesh.myg:
            array = array[:, :, mesh.myg : -mesh.myg, ...]
        prepared[name] = array
    return prepared


def select_payload_variables(
    variables: Mapping[str, Any],
    *,
    compare_variables: tuple[str, ...],
) -> dict[str, Any]:
    if not compare_variables:
        return {name: np.asarray(value, dtype=np.float64) for name, value in variables.items()}
    return {
        name: np.asarray(variables[name], dtype=np.float64)
        for name in compare_variables
        if name in variables
    }
