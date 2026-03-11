from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

from ..reference.cases import ReferenceCase, load_reference_cases
from ..runtime.run_config import RunConfiguration

DEFAULT_REQUIRED_ARTIFACTS = (
    "BOUT.settings",
    "BOUT.log.0",
    "BOUT.dmp.0.nc",
    "BOUT.restart.0.nc",
)
DEFAULT_DATASET_SCALARS = ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")


@dataclass(frozen=True)
class VariableSummary:
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    mean: float
    max_abs_delta_last_first: float | None


@dataclass(frozen=True)
class HermesRunSummary:
    case_name: str
    parity_mode: str
    hermes_binary: str
    overrides: tuple[str, ...]
    workdir: str
    artifacts: Mapping[str, str]
    dimensions: Mapping[str, int]
    time_points: tuple[float, ...]
    dataset_scalars: Mapping[str, float]
    compare_variables: tuple[str, ...]
    variable_summaries: Mapping[str, VariableSummary]
    component_labels: tuple[str, ...]
    nout: int
    timestep: float


@dataclass(frozen=True)
class HermesExecutionResult:
    summary: HermesRunSummary
    stdout_path: str


@dataclass(frozen=True)
class HermesCaseBaseline:
    case_name: str
    parity_mode: str
    compare_variables: tuple[str, ...]


def discover_hermes_binary(
    *,
    hermes_binary: str | Path | None = None,
    hermes_root: str | Path | None = None,
) -> Path:
    if hermes_binary is not None:
        path = Path(hermes_binary).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Hermes binary not found: {path}")
        return path

    env_binary = os.environ.get("JAX_DRB_HERMES_BINARY")
    if env_binary:
        return discover_hermes_binary(hermes_binary=env_binary)

    root = Path(hermes_root).expanduser().resolve() if hermes_root is not None else None
    if root is not None:
        candidate = root / "build" / "hermes-3"
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not discover a Hermes binary. Pass --hermes-binary or --hermes-root.")


def find_reference_case(case_name: str, *, manifest_path: str | Path | None = None) -> ReferenceCase:
    for case in load_reference_cases(manifest_path):
        if case.name == case_name:
            return case
    available = ", ".join(case.name for case in load_reference_cases(manifest_path))
    raise KeyError(f"Unknown reference case {case_name!r}. Available cases: {available}")


def resolve_reference_case(
    case_name: str,
    *,
    hermes_root: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[ReferenceCase, Path]:
    case = find_reference_case(case_name, manifest_path=manifest_path)
    input_path = case.input_path(hermes_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference case input not found: {input_path}")
    return case, input_path


def make_default_overrides(parity_mode: str) -> tuple[str, ...]:
    if parity_mode == "one_rhs":
        return ("nout=0",)
    if parity_mode == "one_step":
        return ("nout=1",)
    return ()


def run_reference_case(
    case_name: str,
    *,
    hermes_root: str | Path,
    hermes_binary: str | Path | None = None,
    manifest_path: str | Path | None = None,
    workdir: str | Path | None = None,
    extra_overrides: Iterable[str] = (),
    keep_workdir: bool = True,
) -> HermesExecutionResult:
    case, input_path = resolve_reference_case(case_name, hermes_root=hermes_root, manifest_path=manifest_path)
    binary = discover_hermes_binary(hermes_binary=hermes_binary, hermes_root=hermes_root)
    staged_workdir = _prepare_workdir(input_path, workdir=workdir)
    stdout_path = staged_workdir / "run.stdout"
    overrides = (*make_default_overrides(case.parity_mode), *tuple(extra_overrides))

    try:
        _run_hermes(binary=binary, workdir=staged_workdir, overrides=overrides, stdout_path=stdout_path)
        summary = _summarize_run(case=case, input_path=input_path, binary=binary, workdir=staged_workdir, overrides=overrides)
    except Exception:
        if workdir is None and not keep_workdir:
            shutil.rmtree(staged_workdir, ignore_errors=True)
        raise

    if workdir is None and not keep_workdir:
        summary_workdir = summary.workdir
        result = HermesExecutionResult(summary=summary, stdout_path=str(stdout_path))
        shutil.rmtree(staged_workdir, ignore_errors=True)
        sanitized_summary = HermesRunSummary(
            case_name=summary.case_name,
            parity_mode=summary.parity_mode,
            hermes_binary=summary.hermes_binary,
            overrides=summary.overrides,
            workdir=summary_workdir,
            artifacts=summary.artifacts,
            dimensions=summary.dimensions,
            time_points=summary.time_points,
            dataset_scalars=summary.dataset_scalars,
            compare_variables=summary.compare_variables,
            variable_summaries=summary.variable_summaries,
            component_labels=summary.component_labels,
            nout=summary.nout,
            timestep=summary.timestep,
        )
        return HermesExecutionResult(summary=sanitized_summary, stdout_path=str(stdout_path))

    return HermesExecutionResult(summary=summary, stdout_path=str(stdout_path))


def write_run_summary_json(summary: HermesRunSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_case_baseline_json(summary: HermesRunSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": summary.case_name,
        "parity_mode": summary.parity_mode,
        "hermes_binary": Path(summary.hermes_binary).name,
        "overrides": list(summary.overrides),
        "required_artifacts": sorted(summary.artifacts),
        "dimensions": dict(summary.dimensions),
        "time_points": list(summary.time_points),
        "dataset_scalars": dict(summary.dataset_scalars),
        "compare_variables": list(summary.compare_variables),
        "variable_summaries": {name: asdict(variable) for name, variable in summary.variable_summaries.items()},
        "component_labels": list(summary.component_labels),
        "configured_nout": summary.nout,
        "configured_timestep": summary.timestep,
        "effective_output_points": len(summary.time_points),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _prepare_workdir(input_path: Path, *, workdir: str | Path | None) -> Path:
    if workdir is None:
        staged = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{input_path.parent.parent.name}-"))
    else:
        staged = Path(workdir).expanduser().resolve()
        staged.mkdir(parents=True, exist_ok=True)
    _stage_case_directory(input_path.parent, staged)
    return staged


def _stage_case_directory(source_dir: Path, target_dir: Path) -> None:
    skip_prefixes = ("BOUT.dmp", "BOUT.log", "BOUT.restart", ".BOUT.pid")
    skip_names = {"BOUT.settings", "run.stdout"}
    for child in source_dir.iterdir():
        if any(child.name.startswith(prefix) for prefix in skip_prefixes) or child.name in skip_names:
            continue
        target = target_dir / child.name
        if target.exists():
            continue
        target.symlink_to(child, target_is_directory=child.is_dir())


def _run_hermes(*, binary: Path, workdir: Path, overrides: Iterable[str], stdout_path: Path) -> None:
    command = [str(binary), "-d", str(workdir), *overrides]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Hermes run failed with exit code {completed.returncode}. See {stdout_path}")


def _summarize_run(
    *,
    case: ReferenceCase,
    input_path: Path,
    binary: Path,
    workdir: Path,
    overrides: tuple[str, ...],
) -> HermesRunSummary:
    run_config = RunConfiguration.from_config(load_bout_input(input_path))
    dmp_path = workdir / "BOUT.dmp.0.nc"
    artifacts = {name: str(workdir / name) for name in DEFAULT_REQUIRED_ARTIFACTS}
    _assert_artifacts_exist(artifacts)
    variable_summaries, dimensions, time_points, dataset_scalars = _summarize_dataset(
        dmp_path,
        compare_variables=case.compare_variables,
    )
    return HermesRunSummary(
        case_name=case.name,
        parity_mode=case.parity_mode,
        hermes_binary=str(binary),
        overrides=overrides,
        workdir=str(workdir),
        artifacts=artifacts,
        dimensions=dimensions,
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        compare_variables=case.compare_variables,
        variable_summaries=variable_summaries,
        component_labels=tuple(request.label for request in run_config.components),
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
    )


def _assert_artifacts_exist(artifacts: Mapping[str, str]) -> None:
    missing = [name for name, path in artifacts.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected Hermes artifacts: {', '.join(missing)}")


def _summarize_dataset(
    path: Path,
    *,
    compare_variables: Iterable[str],
) -> tuple[dict[str, VariableSummary], dict[str, int], tuple[float, ...], dict[str, float]]:
    with Dataset(path) as dataset:
        dimensions = {name: len(dimension) for name, dimension in dataset.dimensions.items()}
        time_points = tuple(float(value) for value in dataset.variables["t_array"][:]) if "t_array" in dataset.variables else ()
        dataset_scalars = {
            name: float(dataset.variables[name][...].item())
            for name in DEFAULT_DATASET_SCALARS
            if name in dataset.variables
        }
        variable_summaries = {
            name: _summarize_variable(dataset, name)
            for name in compare_variables
            if name in dataset.variables
        }
    return variable_summaries, dimensions, time_points, dataset_scalars


def _summarize_variable(dataset: Dataset, name: str) -> VariableSummary:
    variable = dataset.variables[name]
    data = np.asarray(variable[:], dtype=np.float64)
    delta = None
    if "t" in variable.dimensions and data.shape[0] >= 2:
        delta = float(np.max(np.abs(data[-1] - data[0])))
    return VariableSummary(
        name=name,
        dimensions=tuple(variable.dimensions),
        shape=tuple(int(value) for value in data.shape),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        mean=float(np.mean(data)),
        max_abs_delta_last_first=delta,
    )


def load_bout_input(path: Path):
    from ..config.boutinp import load_bout_input as _load_bout_input

    return _load_bout_input(path)
