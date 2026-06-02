from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..runtime import runtime_numpy_dtype
from ..runtime.output import RestartBundle
from ..runtime.run_config import RunConfiguration
from .metrics import StructuredMetrics
from .mesh import StructuredMesh
from .runner_execution import restart_variable_names


@dataclass(frozen=True)
class NativeRunResult:
    payload: Mapping[str, Any]
    variables: Mapping[str, Any]
    time_points: tuple[float, ...]
    run_config: RunConfiguration
    mesh: StructuredMesh
    metrics: StructuredMetrics
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeExecutionResult:
    time_points: tuple[float, ...]
    variables: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.time_points
        yield self.variables


def coerce_native_execution_result(result: object) -> NativeExecutionResult:
    if isinstance(result, NativeExecutionResult):
        return result
    time_points, variables = result  # type: ignore[misc]
    return NativeExecutionResult(
        time_points=tuple(time_points),
        variables=variables,
    )


@dataclass(frozen=True)
class NativeRestartState:
    time_offset: float
    completed_steps: int
    configured_timestep: float
    variables: Mapping[str, np.ndarray]


def build_restart_state(
    result: NativeRunResult,
    *,
    parity_mode: str,
) -> RestartBundle | None:
    names = restart_variable_names(result.run_config)
    if not names:
        return None
    dtype = runtime_numpy_dtype()
    final_state = {
        name: np.asarray(result.variables[name][-1], dtype=dtype)
        for name in names
        if name in result.variables
    }
    if tuple(final_state) != names:
        return None
    return RestartBundle(
        case_name=str(result.payload.get("case_name", "run")),
        parity_mode=parity_mode,
        component_labels=tuple(request.label for request in result.run_config.components),
        current_time=float(result.time_points[-1]) if result.time_points else 0.0,
        completed_steps=max(len(result.time_points) - 1, 0),
        configured_timestep=float(result.run_config.time.timestep),
        state_variables=final_state,
    )
