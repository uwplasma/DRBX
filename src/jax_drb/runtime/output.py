from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RestartBundle:
    case_name: str
    parity_mode: str
    component_labels: tuple[str, ...]
    current_time: float
    completed_steps: int
    configured_timestep: float
    state_variables: dict[str, np.ndarray]


def write_restart_bundle(
    bundle: RestartBundle,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "__metadata__": np.asarray(
            json.dumps(
                {
                    "case_name": bundle.case_name,
                    "parity_mode": bundle.parity_mode,
                    "component_labels": list(bundle.component_labels),
                    "current_time": bundle.current_time,
                    "completed_steps": bundle.completed_steps,
                    "configured_timestep": bundle.configured_timestep,
                    "state_variables": sorted(bundle.state_variables),
                },
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    }
    for name, value in bundle.state_variables.items():
        payload[f"state:{name}"] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(target, **payload)
    return target


def load_restart_bundle(path: str | Path) -> RestartBundle:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["__metadata__"]).item()))
        state_variables = {
            key.removeprefix("state:"): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key.startswith("state:")
        }
    return RestartBundle(
        case_name=str(metadata["case_name"]),
        parity_mode=str(metadata["parity_mode"]),
        component_labels=tuple(str(value) for value in metadata["component_labels"]),
        current_time=float(metadata["current_time"]),
        completed_steps=int(metadata["completed_steps"]),
        configured_timestep=float(metadata["configured_timestep"]),
        state_variables=state_variables,
    )


def build_run_log_payload(
    *,
    input_file: str | Path,
    case_name: str,
    parity_mode: str,
    component_labels: tuple[str, ...],
    time_points: tuple[float, ...],
    dimensions: Mapping[str, int],
    compare_variables: tuple[str, ...],
    restart_supported: bool,
    outputs: Mapping[str, str],
    variable_summaries: Mapping[str, Any],
    run_configuration: Mapping[str, Any] | None = None,
    restart_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_file": str(input_file),
        "case_name": case_name,
        "parity_mode": parity_mode,
        "component_labels": list(component_labels),
        "time_points": list(time_points),
        "dimensions": dict(dimensions),
        "compare_variables": list(compare_variables),
        "restart_supported": bool(restart_supported),
        "outputs": dict(outputs),
        "variable_summaries": dict(variable_summaries),
    }
    if run_configuration is not None:
        payload["run_configuration"] = dict(run_configuration)
    if restart_info is not None:
        payload["restart_info"] = dict(restart_info)
    return payload


def write_run_log_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return target


def format_run_log_text(payload: Mapping[str, Any]) -> str:
    outputs = payload.get("outputs", {})
    output_lines = "\n".join(f"  - {name}: {value}" for name, value in outputs.items()) if outputs else "  - (none)"
    time_points = payload.get("time_points", [])
    if time_points:
        time_line = f"{time_points[0]} -> {time_points[-1]} ({len(time_points)} stored states)"
    else:
        time_line = "(none)"
    components = ", ".join(payload.get("component_labels", [])) or "(none)"
    compare_variables = ", ".join(payload.get("compare_variables", [])) or "(none)"
    run_configuration = payload.get("run_configuration", {})
    restart_info = payload.get("restart_info", {})
    mesh = run_configuration.get("mesh", {})
    solver = run_configuration.get("solver", {})
    time_cfg = run_configuration.get("time", {})
    restart_lines = []
    if restart_info:
        for key in ("loaded_from", "start_time", "input_completed_steps", "requested_additional_steps", "saved_completed_steps"):
            if key in restart_info:
                restart_lines.append(f"  - {key}: {restart_info[key]}")
    restart_block = "\n".join(restart_lines) if restart_lines else "  - (fresh run)"
    return (
        f"Run Summary\n"
        f"  input: {payload.get('input_file')}\n"
        f"  case: {payload.get('case_name')}\n"
        f"  mode: {payload.get('parity_mode')}\n"
        f"  configured nout/timestep: {time_cfg.get('nout', '(unknown)')} / {time_cfg.get('timestep', '(unknown)')}\n"
        f"  mesh: nx={mesh.get('nx', '(unknown)')}, ny={mesh.get('ny', '(unknown)')}, nz={mesh.get('nz', '(unknown)')}, file={mesh.get('file', '<analytic mesh>')}\n"
        f"  solver: type={solver.get('type', '<native default>')}, mxstep={solver.get('mxstep', '(unknown)')}, rtol={solver.get('rtol', '(unknown)')}, atol={solver.get('atol', '(unknown)')}\n"
        f"  components: {components}\n"
        f"  compare variables: {compare_variables}\n"
        f"  dimensions: {payload.get('dimensions')}\n"
        f"  time: {time_line}\n"
        f"  restart supported: {'yes' if payload.get('restart_supported') else 'no'}\n"
        f"  restart:\n{restart_block}\n"
        f"  outputs:\n{output_lines}"
    )
