from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hermes import VariableSummary


def build_portable_summary_payload(
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...],
    component_labels: tuple[str, ...],
    dimensions: Mapping[str, int],
    time_points: tuple[float, ...],
    dataset_scalars: Mapping[str, float],
    variables: Mapping[str, Any],
    overrides: tuple[str, ...] = (),
    configured_nout: int | None = None,
    configured_timestep: float | None = None,
    producer: str = "jax-drb",
) -> dict[str, Any]:
    summaries = {
        name: _summarize_array(name, np.asarray(variables[name], dtype=np.float64))
        for name in compare_variables
        if name in variables
    }
    payload: dict[str, Any] = {
        "case_name": case_name,
        "parity_mode": parity_mode,
        "producer": producer,
        "overrides": list(overrides),
        "compare_variables": list(compare_variables),
        "component_labels": list(component_labels),
        "dimensions": dict(dimensions),
        "time_points": list(time_points),
        "dataset_scalars": {key: float(value) for key, value in dataset_scalars.items()},
        "variable_summaries": {name: asdict(summary) for name, summary in summaries.items()},
        "effective_output_points": len(time_points),
    }
    if configured_nout is not None:
        payload["configured_nout"] = configured_nout
    if configured_timestep is not None:
        payload["configured_timestep"] = configured_timestep
    return payload


def write_portable_summary_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return target


def _summarize_array(name: str, data: np.ndarray) -> VariableSummary:
    delta = None
    if data.ndim >= 1 and data.shape[0] >= 2:
        delta = float(np.max(np.abs(data[-1] - data[0])))
    dimensions = tuple(["t", *[f"dim_{index}" for index in range(1, data.ndim)]])
    return VariableSummary(
        name=name,
        dimensions=dimensions,
        shape=tuple(int(value) for value in data.shape),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        mean=float(np.mean(data)),
        max_abs_delta_last_first=delta,
    )
