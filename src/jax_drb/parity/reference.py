from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

from ..reference.cases import ReferenceCase, load_reference_cases
from ..config.boutinp import apply_bout_overrides, load_bout_input
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
class ReferenceRunSummary:
    case_name: str
    parity_mode: str
    capability_tier: str
    reference_binary: str
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
class ReferenceExecutionResult:
    summary: ReferenceRunSummary
    stdout_path: str


@dataclass(frozen=True)
class ReferenceCaseBaseline:
    case_name: str
    parity_mode: str
    compare_variables: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceSmokeResult:
    case_name: str
    ok: bool
    issues: tuple[str, ...]


def discover_reference_binary(
    *,
    reference_binary: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> Path:
    if reference_binary is not None:
        path = Path(reference_binary).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reference binary not found: {path}")
        return path

    env_binary = os.environ.get("JAX_DRB_REFERENCE_BINARY")
    if env_binary:
        return discover_reference_binary(reference_binary=env_binary)

    root = Path(reference_root).expanduser().resolve() if reference_root is not None else None
    if root is not None:
        candidate = root / "build" / root.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not discover a reference binary. Pass --reference-binary or --reference-root.")


def find_reference_case(case_name: str, *, manifest_path: str | Path | None = None) -> ReferenceCase:
    for case in load_reference_cases(manifest_path):
        if case.name == case_name:
            return case
    available = ", ".join(case.name for case in load_reference_cases(manifest_path))
    raise KeyError(f"Unknown reference case {case_name!r}. Available cases: {available}")


def resolve_reference_case(
    case_name: str,
    *,
    reference_root: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[ReferenceCase, Path]:
    case = find_reference_case(case_name, manifest_path=manifest_path)
    input_path = case.input_path(reference_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference case input not found: {input_path}")
    return case, input_path


def make_default_overrides(parity_mode: str) -> tuple[str, ...]:
    if parity_mode == "one_rhs":
        return ("nout=0",)
    if parity_mode == "one_step":
        return ("nout=1",)
    return ()


def merge_overrides(*groups: Iterable[str]) -> tuple[str, ...]:
    merged: dict[str, str] = {}
    order: list[str] = []
    for group in groups:
        for override in group:
            key = _override_key(override)
            if key not in merged:
                order.append(key)
            merged[key] = override
    return tuple(merged[key] for key in order)


def run_reference_case(
    case_name: str,
    *,
    reference_root: str | Path,
    reference_binary: str | Path | None = None,
    manifest_path: str | Path | None = None,
    workdir: str | Path | None = None,
    extra_overrides: Iterable[str] = (),
    keep_workdir: bool = True,
) -> ReferenceExecutionResult:
    case, input_path = resolve_reference_case(case_name, reference_root=reference_root, manifest_path=manifest_path)
    binary = discover_reference_binary(reference_binary=reference_binary, reference_root=reference_root)
    staged_workdir = _prepare_workdir(case, input_path, workdir=workdir)
    stdout_path = staged_workdir / "run.stdout"
    reference_root_path = Path(reference_root).expanduser().resolve()
    overrides = merge_overrides(
        make_default_overrides(case.parity_mode),
        _resolve_override_placeholders(case.extra_overrides, reference_root=reference_root_path),
        _resolve_override_placeholders(tuple(extra_overrides), reference_root=reference_root_path),
    )

    try:
        _run_reference_binary(
            binary=binary,
            workdir=staged_workdir,
            overrides=overrides,
            stdout_path=stdout_path,
            process_count=case.process_count,
        )
        summary = _summarize_run(case=case, input_path=input_path, binary=binary, workdir=staged_workdir, overrides=overrides)
    except Exception:
        if workdir is None and not keep_workdir:
            shutil.rmtree(staged_workdir, ignore_errors=True)
        raise

    if workdir is None and not keep_workdir:
        summary_workdir = summary.workdir
        result = ReferenceExecutionResult(summary=summary, stdout_path=str(stdout_path))
        shutil.rmtree(staged_workdir, ignore_errors=True)
        sanitized_summary = ReferenceRunSummary(
            case_name=summary.case_name,
            parity_mode=summary.parity_mode,
            capability_tier=summary.capability_tier,
            reference_binary=summary.reference_binary,
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
        return ReferenceExecutionResult(summary=sanitized_summary, stdout_path=str(stdout_path))

    return ReferenceExecutionResult(summary=summary, stdout_path=str(stdout_path))


def write_run_summary_json(summary: ReferenceRunSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_case_baseline_json(summary: ReferenceRunSummary, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_case_baseline_payload(summary)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def build_case_baseline_payload(summary: ReferenceRunSummary) -> dict[str, Any]:
    return {
        "case_name": summary.case_name,
        "parity_mode": summary.parity_mode,
        "capability_tier": summary.capability_tier,
        "reference_runner": "external-reference",
        "overrides": list(summary.overrides),
        "required_artifacts": sorted(summary.artifacts),
        "dimensions": dict(summary.dimensions),
        "time_points": list(summary.time_points),
        "dataset_scalars": dict(summary.dataset_scalars),
        "compare_variables": list(summary.compare_variables),
        "variable_summaries": {
            name: {
                **asdict(variable),
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
            }
            for name, variable in summary.variable_summaries.items()
        },
        "component_labels": list(summary.component_labels),
        "configured_nout": summary.nout,
        "configured_timestep": summary.timestep,
        "effective_output_points": len(summary.time_points),
    }


def validate_reference_baselines(
    *,
    reference_root: str | Path,
    reference_binary: str | Path | None = None,
    manifest_path: str | Path | None = None,
    case_names: Iterable[str] | None = None,
    baseline_dir: str | Path,
) -> tuple[ReferenceSmokeResult, ...]:
    from .compare import compare_summary_payloads, load_summary_json

    requested = tuple(case_names) if case_names is not None else ()
    if requested:
        selected_cases = [find_reference_case(name, manifest_path=manifest_path) for name in requested]
    else:
        selected_cases = [
            case for case in load_reference_cases(manifest_path) if (Path(baseline_dir) / f"{case.name}.json").exists()
        ]
    results: list[ReferenceSmokeResult] = []
    for case in selected_cases:
        baseline_path = Path(baseline_dir) / f"{case.name}.json"
        if not baseline_path.exists():
            results.append(
                ReferenceSmokeResult(
                    case_name=case.name,
                    ok=False,
                    issues=(f"Missing committed baseline JSON: {baseline_path}",),
                )
            )
            continue
        execution = run_reference_case(
            case.name,
            reference_root=reference_root,
            reference_binary=reference_binary,
            manifest_path=manifest_path,
            keep_workdir=False,
        )
        expected = load_summary_json(baseline_path)
        actual = build_case_baseline_payload(execution.summary)
        comparison = compare_summary_payloads(expected, actual, scalar_rtol=1e-12, scalar_atol=1e-12)
        results.append(
            ReferenceSmokeResult(
                case_name=case.name,
                ok=comparison.ok,
                issues=tuple(f"{issue.field}: {issue.message}" for issue in comparison.issues),
            )
        )
    return tuple(results)


def _prepare_workdir(case: ReferenceCase, input_path: Path, *, workdir: str | Path | None) -> Path:
    if workdir is None:
        staged = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{input_path.parent.parent.name}-"))
    else:
        staged = Path(workdir).expanduser().resolve()
        staged.mkdir(parents=True, exist_ok=True)
    _stage_case_directory(input_path.parent, staged)
    _stage_case_artifacts(case, staged)
    _stage_referenced_mesh_files(input_path, staged)
    _stage_shared_reference_support_directories(input_path, staged)
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


def _stage_referenced_mesh_files(input_path: Path, target_dir: Path) -> None:
    config = load_bout_input(input_path)
    if not config.has_section("mesh") or not config.has_option("mesh", "file"):
        return

    mesh_file = str(config.parsed("mesh", "file")).strip().strip('"').strip("'")
    if not mesh_file:
        return

    mesh_path = Path(mesh_file)
    if mesh_path.is_absolute():
        if not mesh_path.exists():
            raise FileNotFoundError(f"Configured mesh file does not exist: {mesh_path}")
        return

    if len(mesh_path.parts) > 1:
        candidate = (input_path.parent / mesh_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Configured mesh file does not exist: {candidate}")
        _stage_mesh_file(candidate, target_dir / mesh_path)
        return

    candidate = (input_path.parent / mesh_path.name).resolve()
    if candidate.exists():
        _stage_mesh_file(candidate, target_dir / mesh_path.name)
        return

    staged_candidate = (target_dir / mesh_path.name).resolve()
    if staged_candidate.exists():
        return

    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in input_path.parents[1:]:
        resolved = (parent / mesh_path.name).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            candidates.append(resolved)

    if not candidates:
        raise FileNotFoundError(f"Could not locate mesh file {mesh_path.name!r} for {input_path}")
    if len(candidates) > 1:
        formatted = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(f"Ambiguous mesh file {mesh_path.name!r} for {input_path}: {formatted}")
    _stage_mesh_file(candidates[0], target_dir / mesh_path.name)


def _stage_mesh_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.resolve() == source.resolve():
            return
        raise RuntimeError(f"Staged mesh target already exists with a different source: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def _stage_shared_reference_support_directories(input_path: Path, target_dir: Path) -> None:
    for directory_name in ("json_database",):
        source = _find_shared_support_directory(input_path, directory_name)
        if source is None:
            continue
        _stage_mesh_file(source, target_dir / directory_name)


def _find_shared_support_directory(input_path: Path, directory_name: str) -> Path | None:
    seen: set[Path] = set()
    for parent in input_path.parents:
        candidate = (parent / directory_name).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _stage_case_artifacts(case: ReferenceCase, target_dir: Path) -> None:
    if case.artifact_bundle_url is None:
        return

    bundle_bytes = _read_artifact_bundle(case.artifact_bundle_url)
    if case.artifact_bundle_sha256 is not None:
        digest = hashlib.sha256(bundle_bytes).hexdigest()
        if digest != case.artifact_bundle_sha256:
            raise RuntimeError(
                f"Artifact bundle sha256 mismatch for {case.name}: expected {case.artifact_bundle_sha256}, got {digest}"
            )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        filenames = case.artifact_bundle_files or tuple(archive.namelist())
        for filename in filenames:
            target = target_dir / filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(filename) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Artifact {filename!r} not found in bundle for {case.name}: {case.artifact_bundle_url}"
                ) from exc


def _read_artifact_bundle(bundle_url: str) -> bytes:
    candidate = Path(bundle_url)
    if "://" not in bundle_url and candidate.exists():
        return candidate.read_bytes()

    with urllib.request.urlopen(bundle_url, timeout=60) as response:
        return response.read()


def _reference_command(
    *,
    binary: Path,
    workdir: Path,
    overrides: Iterable[str],
    process_count: int,
) -> list[str]:
    base = [str(binary), "-d", str(workdir), *overrides]
    if process_count <= 1:
        return base
    return ["mpirun", "-np", str(process_count), *base]


def _run_reference_binary(
    *,
    binary: Path,
    workdir: Path,
    overrides: Iterable[str],
    stdout_path: Path,
    process_count: int = 1,
) -> None:
    command = _reference_command(binary=binary, workdir=workdir, overrides=overrides, process_count=process_count)
    completed = subprocess.run(
        command,
        check=False,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Reference run failed with exit code {completed.returncode}. See {stdout_path}")


def _summarize_run(
    *,
    case: ReferenceCase,
    input_path: Path,
    binary: Path,
    workdir: Path,
    overrides: tuple[str, ...],
) -> ReferenceRunSummary:
    run_config = RunConfiguration.from_config(
        apply_bout_overrides(load_bout_input(input_path), overrides)
        if overrides
        else load_bout_input(input_path)
    )
    dmp_path = workdir / "BOUT.dmp.0.nc"
    artifacts = {name: str(workdir / name) for name in DEFAULT_REQUIRED_ARTIFACTS}
    _assert_artifacts_exist(artifacts)
    variable_summaries, dimensions, time_points, dataset_scalars = _summarize_dataset(
        dmp_path,
        compare_variables=case.compare_variables,
        trim_x_guards=case.trim_x_guards,
        x_guards=run_config.mesh.mxg,
        trim_y_guards=case.trim_y_guards,
        y_guards=run_config.mesh.myg,
    )
    return ReferenceRunSummary(
        case_name=case.name,
        parity_mode=case.parity_mode,
        capability_tier=case.capability_tier,
        reference_binary=str(binary),
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
        raise FileNotFoundError(f"Missing expected reference artifacts: {', '.join(missing)}")


def _summarize_dataset(
    path: Path,
    *,
    compare_variables: Iterable[str],
    trim_x_guards: bool,
    x_guards: int,
    trim_y_guards: bool,
    y_guards: int,
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
            name: _summarize_variable(
                dataset,
                name,
                trim_x_guards=trim_x_guards,
                x_guards=x_guards,
                trim_y_guards=trim_y_guards,
                y_guards=y_guards,
            )
            for name in compare_variables
            if name in dataset.variables
        }
    return variable_summaries, dimensions, time_points, dataset_scalars


def _summarize_variable(
    dataset: Dataset,
    name: str,
    *,
    trim_x_guards: bool,
    x_guards: int,
    trim_y_guards: bool,
    y_guards: int,
) -> VariableSummary:
    variable = dataset.variables[name]
    data = _maybe_trim_guards(
        np.asarray(variable[:], dtype=np.float64),
        dimensions=tuple(variable.dimensions),
        trim_x_guards=trim_x_guards,
        x_guards=x_guards,
        trim_y_guards=trim_y_guards,
        y_guards=y_guards,
    )
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


def _override_key(override: str) -> str:
    return override.split("=", 1)[0].strip()


def _resolve_override_placeholders(
    overrides: Iterable[str],
    *,
    reference_root: Path,
) -> tuple[str, ...]:
    return tuple(str(override).format(reference_root=str(reference_root)) for override in overrides)


def _maybe_trim_guards(
    array: np.ndarray,
    *,
    dimensions: tuple[str, ...],
    trim_x_guards: bool,
    x_guards: int,
    trim_y_guards: bool,
    y_guards: int,
) -> np.ndarray:
    result = array
    if trim_x_guards and x_guards > 0 and "x" in dimensions:
        axis = dimensions.index("x")
        if result.shape[axis] > 2 * x_guards:
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(x_guards, -x_guards)
            result = result[tuple(slicer)]
    if trim_y_guards and y_guards > 0 and "y" in dimensions:
        axis = dimensions.index("y")
        if result.shape[axis] > 2 * y_guards:
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(y_guards, -y_guards)
            result = result[tuple(slicer)]
    return result


def load_bout_input(path: Path):
    from ..config.boutinp import load_bout_input as _load_bout_input

    return _load_bout_input(path)
