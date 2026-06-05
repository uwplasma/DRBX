from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import load_bout_input
from ..native import run_curated_case
from ..native.mesh import build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..native.neutral_mixed import (
    _prepare_neutral_mixed_state,
    _sanitize_neutral_state,
    advance_neutral_mixed_implicit_history,
    compute_neutral_mixed_rhs,
    initialize_neutral_mixed_state,
)
from ..native.neutral_mixed_state import NeutralMixedState
from ..native.units import resolved_dataset_scalars
from ..parity.reference import resolve_reference_case
from ..reference.paths import default_reference_root, repo_root
from ..runtime.run_config import RunConfiguration
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class NeutralMixedTermBalanceCampaignArtifacts:
    report_json_path: Path
    report_npz_path: Path
    report_plot_png_path: Path


_ACCEPTED_TRACE_STAGE = "post_accepted"


def write_neutral_mixed_diagnostic_input(
    source_input: str | Path,
    target_input: str | Path,
    *,
    nout: int = 1,
) -> Path:
    """Write a one-step Hermès neutral-mixed deck with diagnostic outputs on."""

    source = Path(source_input).expanduser().resolve()
    target = Path(target_input).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    text = _set_root_option(text, "nout", str(int(nout)))
    text = _set_section_option(text, "h", "output_ddt", "true")
    text = _set_section_option(text, "h", "diagnose", "true")
    target.write_text(text, encoding="utf-8")
    return target


def write_neutral_mixed_accepted_step_trace_input(
    source_input: str | Path,
    target_input: str | Path,
    *,
    trace_jsonl_path: str | Path,
    nout: int = 1,
    species: str = "h",
) -> Path:
    """Write a reference deck configured for accepted-step neutral trace JSONL."""

    source = Path(source_input).expanduser().resolve()
    target = Path(target_input).expanduser().resolve()
    trace_path = Path(trace_jsonl_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    text = _set_root_option(text, "nout", str(int(nout)))
    text = _set_section_option(text, "solver", "monitor_timestep", "true")
    text = _set_section_option(text, str(species), "output_ddt", "true")
    text = _set_section_option(text, str(species), "diagnose", "true")
    text = _set_section_option(
        text, "hermes", "neutral_mixed_accepted_step_trace", "true"
    )
    text = _set_section_option(
        text,
        "hermes",
        "neutral_mixed_accepted_step_trace_file",
        trace_path.as_posix(),
    )
    text = _set_section_option(
        text, "hermes", "neutral_mixed_accepted_step_trace_species", str(species)
    )
    target.write_text(text, encoding="utf-8")
    return target


def run_neutral_mixed_hermes_diagnostic_rerun(
    *,
    reference_root: str | Path,
    workdir: str | Path,
    hermes_binary: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> Path:
    """Run the Hermès neutral-mixed one-step diagnostic case and return the dump."""

    root = Path(reference_root).expanduser().resolve()
    target_workdir = Path(workdir).expanduser().resolve()
    data_dir = target_workdir / "data"
    if target_workdir.exists():
        shutil.rmtree(target_workdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_input = root / "tests" / "integrated" / "neutral_mixed" / "data" / "BOUT.inp"
    if not source_input.exists():
        raise FileNotFoundError(f"Neutral mixed Hermès input not found: {source_input}")
    write_neutral_mixed_diagnostic_input(source_input, data_dir / "BOUT.inp")
    binary = (
        Path(hermes_binary).expanduser().resolve()
        if hermes_binary is not None
        else _default_hermes_binary(root)
    )
    completed = subprocess.run(
        [str(binary), "-d", "data"],
        cwd=target_workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=float(timeout_seconds),
    )
    (target_workdir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"Hermès neutral-mixed diagnostic rerun failed with exit code {completed.returncode}:\n{tail}"
        )
    dump_path = data_dir / "BOUT.dmp.0.nc"
    if not dump_path.exists():
        raise FileNotFoundError(
            f"Hermès neutral-mixed diagnostic rerun did not produce {dump_path}"
        )
    return dump_path


def run_neutral_mixed_hermes_accepted_step_trace(
    *,
    reference_root: str | Path,
    workdir: str | Path,
    hermes_binary: str | Path | None = None,
    trace_jsonl_path: str | Path | None = None,
    timeout_seconds: float = 120.0,
    species: str = "h",
) -> Path:
    """Run a patched reference neutral-mixed case and return accepted-step JSONL."""

    root = Path(reference_root).expanduser().resolve()
    target_workdir = Path(workdir).expanduser().resolve()
    data_dir = target_workdir / "data"
    if target_workdir.exists():
        shutil.rmtree(target_workdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_input = root / "tests" / "integrated" / "neutral_mixed" / "data" / "BOUT.inp"
    if not source_input.exists():
        raise FileNotFoundError(f"Neutral mixed Hermès input not found: {source_input}")
    trace_path = (
        Path(trace_jsonl_path).expanduser().resolve()
        if trace_jsonl_path is not None
        else data_dir / "neutral_mixed_reference_accepted_step_trace.jsonl"
    )
    write_neutral_mixed_accepted_step_trace_input(
        source_input,
        data_dir / "BOUT.inp",
        trace_jsonl_path=trace_path,
        species=species,
    )
    binary = (
        Path(hermes_binary).expanduser().resolve()
        if hermes_binary is not None
        else _default_hermes_binary(root)
    )
    completed = subprocess.run(
        [str(binary), "-d", "data"],
        cwd=target_workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=float(timeout_seconds),
    )
    (target_workdir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            "Hermès neutral-mixed accepted-step trace run failed with exit "
            f"code {completed.returncode}:\n{tail}"
        )
    if not trace_path.exists():
        raise FileNotFoundError(
            "Hermès neutral-mixed accepted-step trace JSONL was not produced. "
            "This diagnostic requires a reference binary with the gated accepted-step "
            f"trace monitor enabled. Expected: {trace_path}"
        )
    _validate_neutral_mixed_reference_accepted_step_trace_schema(
        trace_path, species=species
    )
    return trace_path


def create_neutral_mixed_term_balance_campaign_package(
    *,
    output_root: str | Path,
    reference_root: str | Path | None = None,
    case_name: str = "neutral_mixed_one_step",
    case_label: str = "neutral_mixed_term_balance_campaign",
    input_path: str | Path | None = None,
    reference_arrays_npz: str | Path | None = None,
    native_arrays_npz: str | Path | None = None,
    hermes_diagnostic_nc: str | Path | None = None,
) -> NeutralMixedTermBalanceCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_neutral_mixed_term_balance_campaign_report(
        reference_root=reference_root,
        case_name=case_name,
        input_path=input_path,
        reference_arrays_npz=reference_arrays_npz,
        native_arrays_npz=native_arrays_npz,
        hermes_diagnostic_nc=hermes_diagnostic_nc,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_npz_path = _write_neutral_mixed_term_balance_arrays(
        report, data_dir / f"{case_label}.npz"
    )
    report_plot_png_path = save_neutral_mixed_term_balance_campaign_plot(
        report,
        images_dir / f"{case_label}.png",
    )
    return NeutralMixedTermBalanceCampaignArtifacts(
        report_json_path=report_json_path,
        report_npz_path=report_npz_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_neutral_mixed_term_balance_campaign_report(
    *,
    reference_root: str | Path | None = None,
    case_name: str = "neutral_mixed_one_step",
    input_path: str | Path | None = None,
    reference_arrays_npz: str | Path | None = None,
    native_arrays_npz: str | Path | None = None,
    hermes_diagnostic_nc: str | Path | None = None,
) -> dict[str, object]:
    root = (
        Path(reference_root).expanduser().resolve()
        if reference_root is not None
        else default_reference_root()
    )
    if input_path is None:
        if root is None:
            raise FileNotFoundError(
                "reference_root or input_path is required for neutral mixed term-balance diagnostics."
            )
        _, resolved_input_path = resolve_reference_case(case_name, reference_root=root)
        input_path = resolved_input_path
    input_path = Path(input_path).expanduser().resolve()
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    template_state = initialize_neutral_mixed_state(config, section="h", mesh=mesh)
    reference_npz = (
        Path(reference_arrays_npz)
        if reference_arrays_npz is not None
        else repo_root()
        / "references"
        / "baselines"
        / "reference_arrays"
        / f"{case_name}.npz"
    )
    reference_history = _load_neutral_mixed_history_npz(reference_npz)
    native_history = (
        _load_neutral_mixed_history_npz(native_arrays_npz)
        if native_arrays_npz is not None
        else _native_history_from_curated_case(case_name, reference_root=root)
    )
    time_points = np.asarray(reference_history["time_points"], dtype=np.float64)
    if time_points.size < 2:
        raise ValueError(
            "Neutral mixed term-balance diagnostics require at least two stored time points."
        )
    timestep = float(time_points[-1] - time_points[0])
    reference_initial = _state_from_trimmed_history(
        reference_history, template_state, time_index=0, mesh=mesh
    )
    reference_final = _state_from_trimmed_history(
        reference_history, template_state, time_index=-1, mesh=mesh
    )
    native_final = _state_from_trimmed_history(
        native_history, template_state, time_index=-1, mesh=mesh
    )
    native_balance = _momentum_balance(
        config,
        native_final,
        reference_initial,
        mesh=mesh,
        metrics=metrics,
        scalars=scalars,
        timestep=timestep,
    )
    reference_balance = _momentum_balance(
        config,
        reference_final,
        reference_initial,
        mesh=mesh,
        metrics=metrics,
        scalars=scalars,
        timestep=timestep,
    )

    active_x = slice(mesh.xstart, mesh.xend + 1)
    active_y = slice(mesh.ystart, mesh.yend + 1)
    final_error = np.asarray(
        native_final.momentum[active_x, active_y, :]
        - reference_final.momentum[active_x, active_y, :],
        dtype=np.float64,
    )
    worst_x_active, worst_y_active, worst_z = np.unravel_index(
        np.argmax(np.abs(final_error)), final_error.shape
    )
    worst_x = int(mesh.xstart + worst_x_active)
    worst_y = int(mesh.ystart + worst_y_active)
    line_x = worst_x
    line_z = int(worst_z)
    x_indices = np.arange(mesh.xstart, mesh.xend + 1, dtype=np.int32)
    y_indices = np.arange(mesh.ystart, mesh.yend + 1, dtype=np.int32)
    target_y_indices = _target_adjacent_y_indices(mesh)

    report: dict[str, object] = {
        "case_name": case_name,
        "reference_code": "hermes-3",
        "input_path": _sanitize_public_path(input_path),
        "reference_arrays_npz": _sanitize_public_path(reference_npz),
        "timestep": timestep,
        "field": "NVh",
        "active_x_indices": x_indices.tolist(),
        "active_y_indices": y_indices.tolist(),
        "lineout_x_index": line_x,
        "lineout_z_index": line_z,
        "worst_final_error_index": {
            "x": line_x,
            "y": worst_y,
            "z": line_z,
        },
        "final_momentum_error": {
            "max_abs": float(np.max(np.abs(final_error))),
            "rms": _rms(final_error),
            "lineout": final_error[worst_x_active, :, line_z].tolist(),
        },
        "final_field_error_register": _final_field_error_register(
            native_final,
            reference_final,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            line_x=line_x,
            line_z=line_z,
        ),
        "state_driver_register": _state_driver_register(
            native_final,
            reference_final,
            native_balance,
            reference_balance,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            line_x=line_x,
            line_z=line_z,
            timestep=timestep,
        ),
        "native_balance": _balance_payload(
            native_balance,
            active_x=active_x,
            active_y=active_y,
            line_x=line_x,
            line_z=line_z,
        ),
        "reference_balance": _balance_payload(
            reference_balance,
            active_x=active_x,
            active_y=active_y,
            line_x=line_x,
            line_z=line_z,
        ),
        "term_delta": _term_delta_payload(
            native_balance,
            reference_balance,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            line_x=line_x,
            line_z=line_z,
        ),
        "offender_register": {
            "target_y_indices": list(target_y_indices),
            "native_final_residual_rate": _ranked_term_metrics(
                native_balance,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
            ),
            "hermes_final_residual_rate": _ranked_term_metrics(
                reference_balance,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
            ),
            "native_minus_hermes_term_delta": _ranked_term_delta_metrics(
                native_balance,
                reference_balance,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
            ),
            "dominant_residual_cells": _dominant_residual_cells(
                reference_balance,
                active_x=active_x,
                active_y=active_y,
                count=6,
            ),
            "interpretation": (
                "The ranked target-adjacent register separates large physical terms from terms whose "
                "native-vs-reference final-state delta is largest. A pressure-gradient or viscosity "
                "entry near the top is therefore a concrete parity target rather than an aggregate "
                "NVh mismatch."
            ),
        },
        "interpretation": {
            "balance_form": "backward_euler_rate_residual = (NVh_final - NVh_initial) / dt - native_rhs_terms(NVh_final)",
            "diagnostic_role": (
                "The Hermès final state is inserted into the native neutral momentum operator. "
                "A nonzero residual localizes the one-step mismatch to the terms that cannot "
                "balance the Hermès update under the current native closure and boundary rules."
            ),
            "next_action": "compare the largest residual-rate lineout terms against Hermès operator diagnostics or add a targeted boundary/closure unit test.",
        },
    }
    if hermes_diagnostic_nc is not None:
        report["hermes_diagnostic_outputs"] = _hermes_diagnostic_payload(
            hermes_diagnostic_nc,
            active_x=active_x,
            active_y=active_y,
            line_x=line_x,
            line_z=line_z,
            matched_sources={
                "SNVh_pressure_gradient": reference_balance.get("pressure_gradient"),
                "SNVh_parallel_viscosity": reference_balance.get("parallel_viscosity"),
                "SNVh_perpendicular_viscosity": reference_balance.get(
                    "perpendicular_viscosity"
                ),
            },
        )
    return report


def build_neutral_mixed_substep_hybrid_report(
    *,
    reference_root: str | Path | None = None,
    case_name: str = "neutral_mixed_one_step",
    input_path: str | Path | None = None,
    reference_arrays_npz: str | Path | None = None,
    native_arrays_by_substep: dict[int, str | Path | dict[str, object]] | None = None,
    substeps: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
) -> dict[str, object]:
    """Build a Hermès-free substep and hybrid-state diagnostic for neutral-mixed ``NVh``.

    The diagnostic reuses committed reference arrays and either supplied native
    histories or native curated runs with ``runtime:neutral_mixed_internal_substeps``.
    It is intended to rank the target-band state sequencing issue without
    claiming a new live-reference comparison.
    """

    root = (
        Path(reference_root).expanduser().resolve()
        if reference_root is not None
        else default_reference_root()
    )
    if input_path is None:
        if root is None:
            raise FileNotFoundError(
                "reference_root or input_path is required for neutral mixed substep diagnostics."
            )
        _, resolved_input_path = resolve_reference_case(case_name, reference_root=root)
        input_path = resolved_input_path
    input_path = Path(input_path).expanduser().resolve()
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    template_state = initialize_neutral_mixed_state(config, section="h", mesh=mesh)
    reference_npz = (
        Path(reference_arrays_npz)
        if reference_arrays_npz is not None
        else repo_root()
        / "references"
        / "baselines"
        / "reference_arrays"
        / f"{case_name}.npz"
    )
    reference_history = _load_neutral_mixed_history_npz(reference_npz)
    time_points = np.asarray(reference_history["time_points"], dtype=np.float64)
    if time_points.size < 2:
        raise ValueError(
            "Neutral mixed substep diagnostics require at least two stored time points."
        )
    timestep = float(time_points[-1] - time_points[0])
    reference_initial = _state_from_trimmed_history(
        reference_history, template_state, time_index=0, mesh=mesh
    )
    reference_final = _state_from_trimmed_history(
        reference_history, template_state, time_index=-1, mesh=mesh
    )
    reference_balance = _momentum_balance(
        config,
        reference_final,
        reference_initial,
        mesh=mesh,
        metrics=metrics,
        scalars=scalars,
        timestep=timestep,
    )

    active_x = slice(mesh.xstart, mesh.xend + 1)
    active_y = slice(mesh.ystart, mesh.yend + 1)
    x_indices = np.arange(mesh.xstart, mesh.xend + 1, dtype=np.int32)
    y_indices = np.arange(mesh.ystart, mesh.yend + 1, dtype=np.int32)
    target_y_indices = _target_adjacent_y_indices(mesh)
    native_inputs = native_arrays_by_substep or {}
    points: list[dict[str, object]] = []
    for substep_count in tuple(int(value) for value in substeps):
        point = _build_neutral_mixed_substep_point(
            substep_count,
            native_inputs.get(substep_count),
            case_name=case_name,
            reference_root=root,
            config=config,
            template_state=template_state,
            reference_initial=reference_initial,
            reference_final=reference_final,
            reference_balance=reference_balance,
            mesh=mesh,
            metrics=metrics,
            scalars=scalars,
            timestep=timestep,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
        )
        points.append(point)

    successful_points = [point for point in points if point.get("status") == "ok"]
    best = None
    if successful_points:
        best_point = min(
            successful_points,
            key=lambda point: float(
                point["final_field_error_register"]["fields"]["NVh"]["max_abs"]  # type: ignore[index]
            ),
        )
        best = {
            "metric": "NVh_final_max_abs",
            "internal_substeps": int(best_point["internal_substeps"]),
            "value": float(
                best_point["final_field_error_register"]["fields"]["NVh"]["max_abs"]
            ),  # type: ignore[index]
        }

    return {
        "diagnostic": "neutral_mixed_substep_hybrid_state",
        "requires_hermes": False,
        "case_name": case_name,
        "parity_mode": "one_step",
        "reference_code": "hermes-3",
        "input_path": _sanitize_public_path(input_path),
        "reference_arrays_npz": _sanitize_public_path(reference_npz),
        "field": "NVh",
        "substeps_requested": [int(value) for value in substeps],
        "time_points": time_points.tolist(),
        "timestep": timestep,
        "active_x_indices": x_indices.tolist(),
        "active_y_indices": y_indices.tolist(),
        "target_y_indices": list(target_y_indices),
        "reference_balance": _balance_payload(
            reference_balance,
            active_x=active_x,
            active_y=active_y,
            line_x=int(mesh.xstart),
            line_z=0,
        ),
        "sweep_points": points,
        "best": best,
        "interpretation": (
            "This committed-baseline diagnostic sweeps neutral-mixed internal substeps and swaps "
            "individual reference fields into the native final state. It is Hermès-free in CI: it "
            "uses stored reference arrays and supplied or native-generated histories. Its role is "
            "to rank target-band state sequencing errors before changing the production solver."
        ),
    }


def build_neutral_mixed_native_accepted_step_trace_report(
    *,
    reference_root: str | Path | None = None,
    case_name: str = "neutral_mixed_one_step",
    input_path: str | Path | None = None,
    internal_substeps: int = 8,
    steps: int = 1,
    reference_trace_json: str | Path | None = None,
    reference_stage: str = _ACCEPTED_TRACE_STAGE,
    time_tolerance: float = 1.0e-8,
) -> dict[str, object]:
    """Run JAXDRB neutral-mixed implicit history with accepted-step tracing enabled."""

    root = (
        Path(reference_root).expanduser().resolve()
        if reference_root is not None
        else default_reference_root()
    )
    if input_path is None:
        if root is None:
            raise FileNotFoundError(
                "reference_root or input_path is required for neutral mixed accepted-step traces."
            )
        _, resolved_input_path = resolve_reference_case(case_name, reference_root=root)
        input_path = resolved_input_path
    input_path = Path(input_path).expanduser().resolve()
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    if not run_config.components:
        raise ValueError(
            "Neutral mixed accepted-step trace requires one configured component."
        )
    section = run_config.components[0].section
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    reference_time_grid: np.ndarray | None = None
    reference_trace_point_count = 0
    time_grid_source = "uniform_internal_substeps"
    if reference_trace_json is not None:
        time_grid = _accepted_step_time_grid_from_reference_trace(
            reference_trace_json,
            preferred_stage=reference_stage,
            target_final_time=float(run_config.time.timestep) * float(steps),
            time_tolerance=float(time_tolerance),
        )
        reference_time_grid = time_grid["time_points"]
        reference_trace_point_count = int(time_grid["trace_point_count"])
        time_grid_source = "reference_accepted_steps"
        time_grid_final_time = float(time_grid["final_time"])
        target_final_time = float(time_grid["target_final_time"])
    else:
        time_grid_final_time = float(run_config.time.timestep) * float(steps)
        target_final_time = time_grid_final_time
    history = advance_neutral_mixed_implicit_history(
        config,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        timestep=float(run_config.time.timestep),
        steps=int(steps),
        internal_substeps=int(internal_substeps),
        solver_mode="matrix_free",
        residual_tolerance=1.0e-8,
        step_tolerance=1.0e-10,
        max_nonlinear_iterations=8,
        linear_restart=20,
        linear_maxiter=200,
        linear_rtol=1.0e-8,
        store_internal_substeps=True,
        accepted_step_time_points=reference_time_grid,
    )
    return _native_accepted_step_trace_report_from_history(
        history,
        config=config,
        input_path=input_path,
        case_name=case_name,
        section=section,
        timestep=float(run_config.time.timestep),
        internal_substeps=int(internal_substeps),
        steps=int(steps),
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
        time_grid_source=time_grid_source,
        reference_trace_json=reference_trace_json,
        reference_stage=reference_stage,
        reference_trace_point_count=reference_trace_point_count,
        time_grid_final_time=time_grid_final_time,
        target_final_time=target_final_time,
    )


def write_neutral_mixed_native_accepted_step_trace_json(
    report: dict[str, object], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target


def build_neutral_mixed_accepted_step_trace_parity_report(
    *,
    native_trace_json: str | Path,
    reference_trace_json: str | Path,
    reference_stage: str = "post_accepted",
    time_tolerance: float = 1.0e-8,
) -> dict[str, object]:
    """Compare native and reference accepted-step traces at matched internal times.

    The reference file may be either the same JSON report shape as the native
    trace or JSONL records emitted by a timestep-monitor reference run with a
    ``stages`` dictionary.
    """

    native_report = _load_accepted_step_trace_records(
        native_trace_json, preferred_stage="post_accepted"
    )
    reference_report = _load_accepted_step_trace_records(
        reference_trace_json, preferred_stage=reference_stage
    )
    native_points = native_report["trace_points"]
    reference_points = reference_report["trace_points"]
    matched_points, field_errors = _compare_accepted_step_trace_points(
        native_points,
        reference_points,
        time_tolerance=float(time_tolerance),
    )
    ranked = sorted(
        field_errors.values(),
        key=_accepted_trace_field_ranking_key,
        reverse=True,
    )
    parallel_viscosity_input_register = (
        _build_parallel_viscosity_input_register(field_errors)
    )
    return {
        "diagnostic": "neutral_mixed_accepted_step_trace_parity",
        "requires_hermes": True,
        "native_diagnostic": native_report["diagnostic"],
        "reference_diagnostic": reference_report["diagnostic"],
        "reference_stage": reference_stage,
        "native_trace_json": _sanitize_public_path(Path(native_trace_json)),
        "reference_trace_json": _sanitize_public_path(Path(reference_trace_json)),
        "trace_point_count": len(native_points),
        "matched_trace_point_count": len(matched_points),
        "time_tolerance": float(time_tolerance),
        "fields": field_errors,
        "ranked_fields": ranked,
        "parallel_viscosity_input_register": parallel_viscosity_input_register,
        "matched_points": matched_points,
        "interpretation": (
            "This accepted-internal-step parity report compares native and reference "
            "post-accepted neutral states before changing neutral-mixed boundary or "
            "source sequencing. A target-adjacent or guard-dominated NVh error points "
            "at sheath/guard reconstruction, while an active-domain error at the first "
            "matched time points at RHS/source assembly. State fields are ranked with "
            "guard metrics because their guard reconstruction is part of the boundary "
            "parity question; derivative and source fields are ranked by active and "
            "target-adjacent cells while still reporting guard deltas separately."
        ),
    }


def _build_parallel_viscosity_input_register(
    field_errors: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Rank neutral-viscosity source offenders against their operator inputs."""

    entries: list[dict[str, object]] = []
    for source_field in sorted(field_errors):
        if not (
            source_field.startswith("SNV")
            and source_field.endswith("_parallel_viscosity")
        ):
            continue
        section = source_field[len("SNV") : -len("_parallel_viscosity")]
        velocity_field = f"V{section}"
        viscosity_field = f"eta_{section}"
        source_error = field_errors[source_field]
        velocity_error = field_errors.get(velocity_field)
        viscosity_error = field_errors.get(viscosity_field)
        input_errors = [
            error
            for error in (velocity_error, viscosity_error)
            if isinstance(error, dict)
        ]
        max_input_active_delta = max(
            (float(error.get("max_active_delta", 0.0)) for error in input_errors),
            default=0.0,
        )
        max_input_target_delta = max(
            (
                float(error.get("max_target_adjacent_delta", 0.0))
                for error in input_errors
            ),
            default=0.0,
        )
        missing_inputs = [
            name
            for name, error in (
                (velocity_field, velocity_error),
                (viscosity_field, viscosity_error),
            )
            if error is None
        ]
        entries.append(
            {
                "source_field": source_field,
                "section": section,
                "source_max_active_delta": float(
                    source_error.get("max_active_delta", 0.0)
                ),
                "source_max_target_adjacent_delta": float(
                    source_error.get("max_target_adjacent_delta", 0.0)
                ),
                "velocity_field": velocity_field,
                "velocity_error": velocity_error,
                "viscosity_field": viscosity_field,
                "viscosity_error": viscosity_error,
                "missing_input_fields": missing_inputs,
                "input_fields_present": not missing_inputs,
                "max_input_active_delta": max_input_active_delta,
                "max_input_target_adjacent_delta": max_input_target_delta,
                "diagnosis": (
                    "input_drift_check_available"
                    if not missing_inputs
                    else "reference_input_trace_missing"
                ),
            }
        )
    entries.sort(
        key=lambda entry: (
            float(entry["source_max_target_adjacent_delta"]),
            float(entry["source_max_active_delta"]),
        ),
        reverse=True,
    )
    return {
        "description": (
            "Compares each accepted-step parallel-viscosity source offender "
            "against the matched operator inputs V and eta. When both input "
            "fields are present, a small input delta and large source delta "
            "points at the stencil/boundary operator; a large input delta "
            "points first at state/history or closure drift."
        ),
        "entries": entries,
        "missing_reference_input_fields": sorted(
            {
                missing
                for entry in entries
                for missing in entry["missing_input_fields"]
            }
        ),
    }


def write_neutral_mixed_accepted_step_trace_parity_json(
    report: dict[str, object], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_neutral_mixed_substep_hybrid_json(
    report: dict[str, object], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target


def save_neutral_mixed_term_balance_campaign_plot(
    report: dict[str, object], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(report["active_y_indices"], dtype=np.float64)
    native = report["native_balance"]
    reference = report["reference_balance"]
    term_order = [
        "time_derivative",
        "parallel_inertia",
        "pressure_gradient",
        "perpendicular_diffusion",
        "parallel_viscosity",
        "perpendicular_viscosity",
    ]
    colors = {
        "time_derivative": "#001219",
        "parallel_inertia": "#005f73",
        "pressure_gradient": "#9b2226",
        "perpendicular_diffusion": "#0a9396",
        "parallel_viscosity": "#ca6702",
        "perpendicular_viscosity": "#ee9b00",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16.8, 9.2), constrained_layout=True)

    error = np.asarray(report["final_momentum_error"]["lineout"], dtype=np.float64)
    axes[0, 0].plot(y, error, color="#9b2226", linewidth=2.2)
    style_axis(
        axes[0, 0],
        title="Final NVh error at worst x,z",
        xlabel="parallel index",
        ylabel="JAXDRB - Hermès-3",
        grid="both",
    )

    axes[0, 1].plot(
        y,
        np.asarray(reference["lineouts"]["residual_rate"], dtype=np.float64),
        color="#9b2226",
        linewidth=2.1,
        label="Hermès final in native residual",
    )
    axes[0, 1].plot(
        y,
        np.asarray(native["lineouts"]["residual_rate"], dtype=np.float64),
        color="#005f73",
        linewidth=1.9,
        linestyle="--",
        label="native final in native residual",
    )
    style_axis(
        axes[0, 1],
        title="Backward-Euler residual-rate lineout",
        xlabel="parallel index",
        ylabel="residual rate",
        grid="both",
    )
    axes[0, 1].legend(frameon=False, fontsize=8.8)

    for name in term_order:
        axes[1, 0].plot(
            y,
            np.asarray(reference["lineouts"][name], dtype=np.float64),
            color=colors[name],
            linewidth=1.7,
            label=name.replace("_", " "),
        )
    style_axis(
        axes[1, 0],
        title="Hermès final state in native momentum balance",
        xlabel="parallel index",
        ylabel="term value",
        grid="both",
    )
    axes[1, 0].legend(frameon=False, fontsize=8.0, ncol=2)

    bar_labels = [
        "parallel\ninertia",
        "pressure\ngradient",
        "perp.\ndiff.",
        "parallel\nviscosity",
        "perp.\nviscosity",
    ]
    values = np.asarray(
        [float(reference["term_metrics"][name]["max_abs"]) for name in term_order[1:]],
        dtype=np.float64,
    )
    x = np.arange(len(values))
    axes[1, 1].bar(
        x, np.maximum(values, 1.0e-16), color=[colors[name] for name in term_order[1:]]
    )
    axes[1, 1].set_xticks(x, bar_labels)
    style_axis(
        axes[1, 1],
        title="Max |native term| for Hermès final state",
        ylabel="max absolute value",
        yscale="log",
        grid="y",
    )
    annotate_bars(
        axes[1, 1], x, np.maximum(values, 1.0e-16), fmt="{:.1e}", fontsize=7.8
    )

    delta = report.get("term_delta", {})
    delta_metrics = delta.get("term_metrics", {}) if isinstance(delta, dict) else {}
    delta_values = np.asarray(
        [
            float(delta_metrics.get(name, {}).get("target_adjacent_max_abs", 0.0))
            for name in term_order[1:]
        ],
        dtype=np.float64,
    )
    axes[0, 2].bar(
        x,
        np.maximum(delta_values, 1.0e-16),
        color=[colors[name] for name in term_order[1:]],
    )
    axes[0, 2].set_xticks(x, bar_labels)
    style_axis(
        axes[0, 2],
        title="Target-adjacent |native - Hermès| term delta",
        ylabel="max absolute delta",
        yscale="log",
        grid="y",
    )
    annotate_bars(
        axes[0, 2], x, np.maximum(delta_values, 1.0e-16), fmt="{:.1e}", fontsize=7.5
    )

    diagnostics = report.get("hermes_diagnostic_outputs", {})
    direct_comparisons = (
        diagnostics.get("direct_comparisons", {})
        if isinstance(diagnostics, dict)
        else {}
    )
    pressure_direct = (
        direct_comparisons.get("SNVh_pressure_gradient", {})
        if isinstance(direct_comparisons, dict)
        else {}
    )
    closure_names = [
        ("SNVh_pressure_gradient", "pressure\ngrad."),
        ("SNVh_parallel_viscosity", "parallel\nvisc."),
        ("SNVh_perpendicular_viscosity", "perp.\nvisc."),
    ]
    closure_values: list[float] = []
    closure_labels: list[str] = []
    closure_source_names: list[str] = []
    for name, label in closure_names:
        comparison = direct_comparisons.get(name, {})
        if not isinstance(comparison, dict):
            continue
        metrics = comparison.get("scaled_difference_metrics", {})
        if not isinstance(metrics, dict) or "max_abs" not in metrics:
            continue
        closure_values.append(float(metrics["max_abs"]))
        closure_labels.append(label)
        closure_source_names.append(name)
    if closure_values:
        closure_x = np.arange(len(closure_values))
        closure_color_map = {
            "SNVh_pressure_gradient": colors["pressure_gradient"],
            "SNVh_parallel_viscosity": colors["parallel_viscosity"],
            "SNVh_perpendicular_viscosity": colors["perpendicular_viscosity"],
        }
        closure_colors = [closure_color_map[name] for name in closure_source_names]
        axes[1, 2].bar(
            closure_x,
            np.maximum(np.asarray(closure_values, dtype=np.float64), 1.0e-16),
            color=closure_colors,
        )
        axes[1, 2].set_xticks(closure_x, closure_labels)
        style_axis(
            axes[1, 2],
            title="Direct source diagnostic closure",
            ylabel="max |scaled direct - native|",
            yscale="log",
            grid="y",
        )
        annotate_bars(
            axes[1, 2],
            closure_x,
            np.maximum(np.asarray(closure_values, dtype=np.float64), 1.0e-16),
            fmt="{:.1e}",
            fontsize=7.5,
        )
    elif (
        isinstance(pressure_direct, dict) and "scaled_direct_lineout" in pressure_direct
    ):
        axes[1, 2].plot(
            y,
            np.asarray(
                pressure_direct["matched_reconstruction_lineout"], dtype=np.float64
            ),
            color=colors["pressure_gradient"],
            linewidth=2.0,
            label="matched native -Grad_par(Pn)",
        )
        axes[1, 2].plot(
            y,
            np.asarray(pressure_direct["scaled_direct_lineout"], dtype=np.float64),
            color="#005f73",
            linewidth=1.8,
            linestyle="--",
            label="scaled direct reference diagnostic",
        )
        style_axis(
            axes[1, 2],
            title="Direct pressure-gradient diagnostic check",
            xlabel="parallel index",
            ylabel="native-normalized term",
            grid="both",
        )
        axes[1, 2].legend(frameon=False, fontsize=7.8)
    else:
        for name in (
            "pressure_gradient",
            "parallel_viscosity",
            "perpendicular_viscosity",
        ):
            if isinstance(delta, dict) and name in delta.get("lineouts", {}):
                axes[1, 2].plot(
                    y,
                    np.asarray(delta["lineouts"][name], dtype=np.float64),
                    linewidth=1.8,
                    color=colors[name],
                    label=name.replace("_", " "),
                )
        style_axis(
            axes[1, 2],
            title="Native - Hermès final-state term deltas",
            xlabel="parallel index",
            ylabel="term delta",
            grid="both",
        )
        axes[1, 2].legend(frameon=False, fontsize=7.8)

    figure.suptitle(
        "Neutral mixed NVh term-balance audit",
        fontsize=13.5,
        fontweight="semibold",
    )
    save_publication_figure(figure, target)
    return target


def _load_neutral_mixed_history_npz(path: str | Path) -> dict[str, object]:
    with np.load(path) as payload:
        metadata = json.loads(str(payload["__metadata__"].item()))
        return {
            "time_points": np.asarray(metadata["time_points"], dtype=np.float64),
            "Nh": np.asarray(payload["var__Nh"], dtype=np.float64),
            "Ph": np.asarray(payload["var__Ph"], dtype=np.float64),
            "NVh": np.asarray(payload["var__NVh"], dtype=np.float64),
        }


def _native_history_from_curated_case(
    case_name: str, *, reference_root: Path | None
) -> dict[str, object]:
    if reference_root is None:
        raise FileNotFoundError(
            "reference_root is required when native_arrays_npz is not supplied."
        )
    result = run_curated_case(case_name, reference_root=reference_root)
    return {
        "time_points": np.asarray(result.time_points, dtype=np.float64),
        "Nh": np.asarray(result.variables["Nh"], dtype=np.float64),
        "Ph": np.asarray(result.variables["Ph"], dtype=np.float64),
        "NVh": np.asarray(result.variables["NVh"], dtype=np.float64),
    }


def _native_history_from_curated_case_with_substeps(
    case_name: str,
    *,
    reference_root: Path | None,
    internal_substeps: int,
) -> dict[str, object]:
    if reference_root is None:
        raise FileNotFoundError(
            "reference_root is required when native arrays are not supplied for a substep point."
        )
    result = run_curated_case(
        case_name,
        reference_root=reference_root,
        extra_overrides=(
            f"runtime:neutral_mixed_internal_substeps={int(internal_substeps)}",
        ),
    )
    return {
        "time_points": np.asarray(result.time_points, dtype=np.float64),
        "Nh": np.asarray(result.variables["Nh"], dtype=np.float64),
        "Ph": np.asarray(result.variables["Ph"], dtype=np.float64),
        "NVh": np.asarray(result.variables["NVh"], dtype=np.float64),
    }


def _coerce_neutral_mixed_history(
    source: str | Path | dict[str, object],
) -> dict[str, object]:
    if isinstance(source, (str, Path)):
        return _load_neutral_mixed_history_npz(source)
    return {
        "time_points": np.asarray(source["time_points"], dtype=np.float64),
        "Nh": np.asarray(source["Nh"], dtype=np.float64),
        "Ph": np.asarray(source["Ph"], dtype=np.float64),
        "NVh": np.asarray(source["NVh"], dtype=np.float64),
    }


def _build_neutral_mixed_substep_point(
    internal_substeps: int,
    native_source: str | Path | dict[str, object] | None,
    *,
    case_name: str,
    reference_root: Path | None,
    config,
    template_state: NeutralMixedState,
    reference_initial: NeutralMixedState,
    reference_final: NeutralMixedState,
    reference_balance: dict[str, np.ndarray],
    mesh,
    metrics,
    scalars: dict[str, float],
    timestep: float,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
) -> dict[str, object]:
    import time

    start = time.perf_counter()
    try:
        native_history = (
            _coerce_neutral_mixed_history(native_source)
            if native_source is not None
            else _native_history_from_curated_case_with_substeps(
                case_name,
                reference_root=reference_root,
                internal_substeps=internal_substeps,
            )
        )
        native_time_points = np.asarray(native_history["time_points"], dtype=np.float64)
        if native_time_points.size < 2:
            raise ValueError(
                "Native substep history must contain at least two stored time points."
            )
        native_final = _state_from_trimmed_history(
            native_history, template_state, time_index=-1, mesh=mesh
        )
        native_balance = _momentum_balance(
            config,
            native_final,
            reference_initial,
            mesh=mesh,
            metrics=metrics,
            scalars=scalars,
            timestep=timestep,
        )
        line_x, line_y, line_z = _worst_state_error_index(
            native_final, reference_final, "NVh", active_x=active_x, active_y=active_y
        )
        return {
            "internal_substeps": int(internal_substeps),
            "status": "ok",
            "elapsed_seconds": float(time.perf_counter() - start),
            "sub_timestep": float(timestep / max(int(internal_substeps), 1)),
            "native_time_points": native_time_points.tolist(),
            "probe_index": {
                "x": line_x,
                "y": line_y,
                "z": line_z,
                "trimmed_y": int(line_y - active_y.start),
            },
            "series_errors": _neutral_mixed_series_errors(
                native_history,
                {
                    "time_points": np.asarray([0.0, timestep], dtype=np.float64),
                    "Nh": np.stack(
                        [reference_initial.density, reference_final.density]
                    ),
                    "Ph": np.stack(
                        [reference_initial.pressure, reference_final.pressure]
                    ),
                    "NVh": np.stack(
                        [reference_initial.momentum, reference_final.momentum]
                    ),
                },
                mesh=mesh,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
            ),
            "final_field_error_register": _final_field_error_register(
                native_final,
                reference_final,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
                line_x=line_x,
                line_z=line_z,
            ),
            "native_balance": _balance_payload(
                native_balance,
                active_x=active_x,
                active_y=active_y,
                line_x=line_x,
                line_z=line_z,
            ),
            "term_delta": _term_delta_payload(
                native_balance,
                reference_balance,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
                line_x=line_x,
                line_z=line_z,
            ),
            "hybrid_state_register": _hybrid_state_register(
                config,
                native_final,
                reference_final,
                reference_initial,
                native_balance,
                reference_balance,
                mesh=mesh,
                metrics=metrics,
                scalars=scalars,
                timestep=timestep,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
                line_x=line_x,
                line_z=line_z,
            ),
        }
    except Exception as exc:
        payload = {
            "internal_substeps": int(internal_substeps),
            "status": "failed",
            "elapsed_seconds": float(time.perf_counter() - start),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        failure_vector = _exception_array_payload(exc)
        if failure_vector:
            payload["failure_vector"] = failure_vector
        return payload


def _exception_array_payload(exc: Exception) -> dict[str, object]:
    for arg in getattr(exc, "args", ()):
        try:
            values = np.asarray(arg, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if values.ndim == 0 or values.size == 0:
            continue
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size == 0:
            max_abs = float("nan")
            rms = float("nan")
        else:
            max_abs = float(np.max(np.abs(finite_values)))
            rms = float(np.sqrt(np.mean(np.square(finite_values))))
        return {
            "size": int(values.size),
            "finite_fraction": float(np.mean(finite)),
            "max_abs": max_abs,
            "rms": rms,
            "first_values": values.reshape(-1)[:8].tolist(),
            "last_values": values.reshape(-1)[-8:].tolist(),
        }
    return {}


def _state_from_trimmed_history(
    history: dict[str, object],
    template: NeutralMixedState,
    *,
    time_index: int,
    mesh,
) -> NeutralMixedState:
    density = _restore_trimmed_field(
        np.asarray(history["Nh"], dtype=np.float64)[time_index],
        template.density,
        mesh=mesh,
    )
    pressure = _restore_trimmed_field(
        np.asarray(history["Ph"], dtype=np.float64)[time_index],
        template.pressure,
        mesh=mesh,
    )
    momentum = _restore_trimmed_field(
        np.asarray(history["NVh"], dtype=np.float64)[time_index],
        template.momentum,
        mesh=mesh,
    )
    return _sanitize_neutral_state(
        NeutralMixedState(density=density, pressure=pressure, momentum=momentum),
        mesh,
    )


def _worst_state_error_index(
    native_final: NeutralMixedState,
    reference_final: NeutralMixedState,
    field_name: str,
    *,
    active_x: slice,
    active_y: slice,
) -> tuple[int, int, int]:
    field_map = {
        "Nh": (native_final.density, reference_final.density),
        "Ph": (native_final.pressure, reference_final.pressure),
        "NVh": (native_final.momentum, reference_final.momentum),
    }
    native_value, reference_value = field_map[field_name]
    delta = np.asarray(native_value, dtype=np.float64) - np.asarray(
        reference_value, dtype=np.float64
    )
    active = delta[active_x, active_y, :]
    x_offset, y_offset, z_index = np.unravel_index(
        int(np.argmax(np.abs(active))), active.shape
    )
    return int(active_x.start + x_offset), int(active_y.start + y_offset), int(z_index)


def _active_history_field(
    history: dict[str, object], field_name: str, *, mesh, active_y: slice
) -> np.ndarray:
    field = np.asarray(history[field_name], dtype=np.float64)
    if field.ndim != 4:
        raise ValueError(
            f"Expected {field_name} history to have shape (time, x, y, z), got {field.shape}."
        )
    active_y_size = int(active_y.stop - active_y.start)
    if field.shape[2] == active_y_size:
        return field
    if field.shape[2] == mesh.local_ny:
        return field[:, :, active_y, :]
    raise ValueError(
        f"Unsupported {field_name} history y-size {field.shape[2]}; expected {active_y_size} active cells or {mesh.local_ny} full cells."
    )


def _neutral_mixed_series_errors(
    native_history: dict[str, object],
    reference_history: dict[str, object],
    *,
    mesh,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
) -> dict[str, object]:
    target_offsets = _target_offsets(
        slice(0, active_y.stop - active_y.start),
        tuple(index - active_y.start for index in target_y_indices),
    )
    fields: dict[str, object] = {}
    ranked: list[dict[str, object]] = []
    for name in ("Nh", "Ph", "NVh"):
        native = _active_history_field(
            native_history, name, mesh=mesh, active_y=active_y
        )
        reference = _active_history_field(
            reference_history, name, mesh=mesh, active_y=active_y
        )
        time_count = min(native.shape[0], reference.shape[0])
        delta = (
            native[:time_count, active_x, :, :] - reference[:time_count, active_x, :, :]
        )
        final_delta = delta[-1]
        target = (
            final_delta[:, target_offsets, :] if target_offsets.size else final_delta
        )
        metrics = _array_metrics(delta)
        final_metrics = _array_metrics(final_delta)
        final_metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        final_metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        fields[name] = {
            "time_points_compared": int(time_count),
            "series_max_abs": metrics["max_abs"],
            "series_rms": metrics["rms"],
            "active_edge_history_trace": _active_edge_history_trace(
                delta, target_offsets=target_offsets
            ),
            **{f"final_{key}": value for key, value in final_metrics.items()},
        }
        ranked.append({"field": name, **fields[name]})  # type: ignore[arg-type]
    return {
        "fields": fields,
        "ranked_by_final_target_adjacent_max_abs": sorted(
            ranked,
            key=lambda item: float(item["final_target_adjacent_max_abs"]),
            reverse=True,
        ),
    }


def _active_edge_history_trace(
    delta: np.ndarray, *, target_offsets: np.ndarray
) -> dict[str, object]:
    if delta.ndim != 4:
        raise ValueError(
            f"Expected neutral mixed history delta to have shape (time, x, y, z), got {delta.shape}."
        )
    if target_offsets.size:
        target_delta = delta[:, :, target_offsets, :]
    else:
        target_delta = delta
    target_max_abs = (
        np.max(np.abs(target_delta), axis=(1, 2, 3))
        if target_delta.size
        else np.zeros(delta.shape[0])
    )
    target_rms = (
        np.sqrt(np.mean(np.square(target_delta), axis=(1, 2, 3)))
        if target_delta.size
        else np.zeros(delta.shape[0])
    )
    edge_traces: list[dict[str, object]] = []
    for offset in target_offsets:
        edge_delta = delta[:, :, int(offset), :]
        edge_traces.append(
            {
                "active_y_offset": int(offset),
                "max_abs_by_time": np.max(np.abs(edge_delta), axis=(1, 2)).tolist(),
                "rms_by_time": np.sqrt(
                    np.mean(np.square(edge_delta), axis=(1, 2))
                ).tolist(),
            }
        )
    return {
        "target_active_y_offsets": [int(value) for value in target_offsets],
        "target_adjacent_max_abs_by_time": target_max_abs.tolist(),
        "target_adjacent_rms_by_time": target_rms.tolist(),
        "edge_traces": edge_traces,
        "interpretation": (
            "These time-indexed active-edge deltas localize whether the remaining neutral-mixed "
            "difference is introduced at the first stored state or by the final target-band history update."
        ),
    }


def _native_accepted_step_trace_report_from_history(
    history,
    *,
    config,
    input_path: Path,
    case_name: str,
    section: str,
    timestep: float,
    internal_substeps: int,
    steps: int,
    mesh,
    metrics,
    meters_scale: float,
    tnorm: float,
    time_grid_source: str = "uniform_internal_substeps",
    reference_trace_json: str | Path | None = None,
    reference_stage: str | None = None,
    reference_trace_point_count: int = 0,
    time_grid_final_time: float | None = None,
    target_final_time: float | None = None,
) -> dict[str, object]:
    time_points = history.accepted_step_time_points
    accepted_dt = history.accepted_step_dt
    accepted_order = history.accepted_step_order
    density = history.accepted_step_density_history
    pressure = history.accepted_step_pressure_history
    momentum = history.accepted_step_momentum_history
    residual_norm = history.accepted_step_residual_inf_norm
    nonlinear_iterations = history.accepted_step_nonlinear_iterations
    if any(
        item is None
        for item in (
            time_points,
            accepted_dt,
            accepted_order,
            density,
            pressure,
            momentum,
            residual_norm,
            nonlinear_iterations,
        )
    ):
        raise ValueError(
            "Neutral mixed history was not generated with store_internal_substeps=True."
        )
    time_points = np.asarray(time_points, dtype=np.float64)
    accepted_dt = np.asarray(accepted_dt, dtype=np.float64)
    accepted_order = np.asarray(accepted_order, dtype=np.int32)
    residual_norm = np.asarray(residual_norm, dtype=np.float64)
    nonlinear_iterations = np.asarray(nonlinear_iterations, dtype=np.int32)
    field_histories = {
        f"N{section}": np.asarray(density, dtype=np.float64),
        f"P{section}": np.asarray(pressure, dtype=np.float64),
        f"NV{section}": np.asarray(momentum, dtype=np.float64),
    }
    active_x = slice(mesh.xstart, mesh.xend + 1)
    active_y = slice(mesh.ystart, mesh.yend + 1)
    target_y_indices = _target_adjacent_y_indices(mesh)
    guard_y_indices = _neutral_mixed_guard_y_indices(mesh)
    sample_y_indices = tuple(sorted(set(target_y_indices).union(guard_y_indices)))
    line_x = int(mesh.xstart + max((mesh.xend - mesh.xstart) // 2, 0))
    line_z = int(mesh.nz // 2)
    trace_points: list[dict[str, object]] = []
    for index, time_value in enumerate(time_points):
        accepted_state = NeutralMixedState(
            density=np.asarray(density[index], dtype=np.float64),
            pressure=np.asarray(pressure[index], dtype=np.float64),
            momentum=np.asarray(momentum[index], dtype=np.float64),
        )
        rhs_payloads = _native_accepted_step_rhs_field_payloads(
            config,
            accepted_state,
            section=section,
            mesh=mesh,
            metrics=metrics,
            meters_scale=meters_scale,
            tnorm=tnorm,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            guard_y_indices=guard_y_indices,
            sample_y_indices=sample_y_indices,
            line_x=line_x,
            line_z=line_z,
        )
        state_payloads = {
            name: _native_accepted_step_field_payload(
                values[index],
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
                guard_y_indices=guard_y_indices,
                sample_y_indices=sample_y_indices,
                line_x=line_x,
                line_z=line_z,
            )
            for name, values in field_histories.items()
        }
        state_payloads.update(rhs_payloads)
        trace_points.append(
            {
                "index": int(index),
                "time": float(time_value),
                "dt": float(accepted_dt[index]),
                "solver_order": int(accepted_order[index]),
                "stage": "post_accepted",
                "residual_inf_norm": float(residual_norm[index]),
                "nonlinear_iterations": int(nonlinear_iterations[index]),
                "fields": state_payloads,
            }
        )
    return {
        "diagnostic": "neutral_mixed_native_accepted_step_trace",
        "requires_hermes": False,
        "case_name": case_name,
        "reference_code": "none",
        "input_path": _sanitize_public_path(input_path),
        "section": section,
        "configured_timestep": float(timestep),
        "steps": int(steps),
        "internal_substeps": int(internal_substeps),
        "time_grid_source": str(time_grid_source),
        "reference_trace_json": _sanitize_public_path(Path(reference_trace_json))
        if reference_trace_json is not None
        else None,
        "reference_stage": reference_stage,
        "reference_trace_point_count": int(reference_trace_point_count),
        "time_grid_final_time": float(time_grid_final_time)
        if time_grid_final_time is not None
        else None,
        "target_final_time": float(target_final_time)
        if target_final_time is not None
        else None,
        "trace_point_count": int(time_points.size),
        "active_x_indices": list(range(int(mesh.xstart), int(mesh.xend) + 1)),
        "active_y_indices": list(range(int(mesh.ystart), int(mesh.yend) + 1)),
        "target_y_indices": list(target_y_indices),
        "guard_y_indices": list(guard_y_indices),
        "sample_y_indices": list(sample_y_indices),
        "lineout_x_index": line_x,
        "lineout_z_index": line_z,
        "time_points": time_points.tolist(),
        "trace_points": trace_points,
        "interpretation": (
            "This is the native post-accepted internal-step trace that should be compared with "
            "a reference accepted-step trace before changing neutral-mixed target or guard sequencing."
        ),
    }


def _neutral_mixed_guard_y_indices(mesh) -> tuple[int, ...]:
    return tuple(
        sorted(
            index
            for index in {
                int(mesh.ystart - 2),
                int(mesh.ystart - 1),
                int(mesh.yend + 1),
                int(mesh.yend + 2),
            }
            if 0 <= index < int(mesh.local_ny)
        )
    )


def _native_accepted_step_field_payload(
    values: np.ndarray,
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    guard_y_indices: tuple[int, ...],
    sample_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    active = array[active_x, active_y, :]
    target = (
        array[active_x, target_y_indices, :]
        if target_y_indices
        else np.asarray([], dtype=np.float64)
    )
    guard = (
        array[active_x, guard_y_indices, :]
        if guard_y_indices
        else np.asarray([], dtype=np.float64)
    )
    return {
        "active_metrics": _array_metrics(active),
        "target_adjacent_metrics": _array_metrics(target),
        "guard_metrics": _array_metrics(guard),
        "sample_lineout_y_indices": list(sample_y_indices),
        "sample_lineout": array[line_x, sample_y_indices, line_z].tolist()
        if sample_y_indices
        else [],
    }


def _native_accepted_step_rhs_field_payloads(
    config,
    state: NeutralMixedState,
    *,
    section: str,
    mesh,
    metrics,
    meters_scale: float,
    tnorm: float,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    guard_y_indices: tuple[int, ...],
    sample_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    rhs = compute_neutral_mixed_rhs(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    prepared = _prepare_neutral_mixed_state(
        config,
        state,
        section=section,
        mesh=mesh,
        metrics=metrics,
        meters_scale=meters_scale,
        tnorm=tnorm,
    )
    zeros = np.zeros_like(rhs.momentum, dtype=np.float64)
    fields = {
        f"V{section}": prepared.velocity,
        f"eta_{section}": prepared.viscosity,
        f"ddt(N{section})": rhs.density,
        f"ddt(P{section})": rhs.pressure,
        f"ddt(NV{section})": rhs.momentum,
        f"SNV{section}": rhs.momentum_terms.get("momentum_source", zeros),
        f"SNV{section}_pressure_gradient": rhs.momentum_terms.get(
            "pressure_gradient", zeros
        ),
        f"SNV{section}_parallel_viscosity": rhs.momentum_terms.get(
            "parallel_viscosity", zeros
        ),
        f"SNV{section}_perpendicular_viscosity": rhs.momentum_terms.get(
            "perpendicular_viscosity", zeros
        ),
    }
    return {
        name: _native_accepted_step_field_payload(
            values,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            guard_y_indices=guard_y_indices,
            sample_y_indices=sample_y_indices,
            line_x=line_x,
            line_z=line_z,
        )
        for name, values in fields.items()
    }


def _accepted_step_time_grid_from_reference_trace(
    path: str | Path,
    *,
    preferred_stage: str,
    target_final_time: float,
    time_tolerance: float,
) -> dict[str, object]:
    if time_tolerance <= 0.0:
        raise ValueError("time_tolerance must be positive")
    if target_final_time <= 0.0:
        raise ValueError("target_final_time must be positive")
    report = _load_accepted_step_trace_records(path, preferred_stage=preferred_stage)
    trace_points = report["trace_points"]
    accepted_times: list[float] = []
    for index, point in enumerate(trace_points):
        time_value = float(point["time"])
        if not np.isfinite(time_value):
            raise ValueError(
                f"Accepted-step reference trace contains a non-finite time at index {index}."
            )
        if abs(time_value) <= time_tolerance:
            continue
        if time_value <= 0.0:
            raise ValueError(
                f"Accepted-step reference trace time must be positive, got {time_value:g}."
            )
        if accepted_times and time_value <= accepted_times[-1]:
            raise ValueError(
                "Accepted-step reference trace times must be strictly increasing."
            )
        accepted_times.append(time_value)
    if not accepted_times:
        raise ValueError("Accepted-step reference trace contains no positive times.")
    final_time = accepted_times[-1]
    if final_time + time_tolerance < target_final_time:
        raise ValueError(
            "Accepted-step reference trace final time does not reach the requested "
            f"native trace window: final_time={final_time:g}, "
            f"target_final_time={target_final_time:g}, tolerance={time_tolerance:g}."
        )
    return {
        "time_points": np.asarray(accepted_times, dtype=np.float64),
        "trace_point_count": len(trace_points),
        "final_time": float(final_time),
        "target_final_time": float(target_final_time),
    }


def _load_accepted_step_trace_records(
    path: str | Path, *, preferred_stage: str
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Accepted-step trace file is empty: {source}")
    records: list[dict[str, object]]
    if text[0] == "{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            if "trace_points" in payload:
                return {
                    "diagnostic": str(payload.get("diagnostic", "accepted_step_trace")),
                    "trace_points": [
                        _normalize_accepted_trace_point(
                            point, preferred_stage=preferred_stage
                        )
                        for point in payload["trace_points"]
                    ],
                }
            records = [payload]
    else:
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return {
        "diagnostic": str(
            records[0].get("diagnostic", "neutral_mixed_reference_accepted_step_trace")
            if records
            else "accepted_step_trace"
        ),
        "trace_points": [
            _normalize_accepted_trace_point(record, preferred_stage=preferred_stage)
            for record in records
        ],
    }


def _validate_neutral_mixed_reference_accepted_step_trace_schema(
    path: str | Path,
    *,
    species: str = "h",
) -> None:
    """Validate the reference trace contains the term-level diagnostics we need."""

    source = Path(path).expanduser().resolve()
    records = _load_accepted_step_trace_raw_records(source)
    if not records:
        raise ValueError(f"Accepted-step trace file contains no records: {source}")
    suffix = str(species)
    state_fields = (f"N{suffix}", f"P{suffix}", f"NV{suffix}")
    rhs_fields = (f"ddt(N{suffix})", f"ddt(P{suffix})", f"ddt(NV{suffix})")
    source_fields = (
        f"SNV{suffix}",
        f"SNV{suffix}_pressure_gradient",
        f"SNV{suffix}_parallel_viscosity",
        f"SNV{suffix}_perpendicular_viscosity",
    )

    missing_state_records: list[int] = []
    all_fields: set[str] = set()
    available_stages: set[str] = set()
    for record_index, record in enumerate(records):
        stage_fields = _accepted_trace_fields_by_stage(record)
        available_stages.update(stage_fields)
        for fields in stage_fields.values():
            all_fields.update(fields)
        post_accepted_fields = stage_fields.get(_ACCEPTED_TRACE_STAGE, set())
        if any(name not in post_accepted_fields for name in state_fields):
            missing_state_records.append(record_index)

    missing_rhs = [name for name in rhs_fields if name not in all_fields]
    missing_sources = [name for name in source_fields if name not in all_fields]
    available_field_list = ", ".join(sorted(all_fields)) or "<none>"
    available_stage_list = ", ".join(sorted(available_stages)) or "<none>"
    errors: list[str] = []
    if missing_state_records:
        errors.append(
            f"{_ACCEPTED_TRACE_STAGE!r} is missing {state_fields} on record indices "
            f"{missing_state_records[:8]}"
        )
    if missing_rhs:
        errors.append(f"missing RHS fields: {missing_rhs}")
    if missing_sources:
        errors.append(f"missing source diagnostics: {missing_sources}")
    if errors:
        raise ValueError(
            "Hermès neutral-mixed accepted-step trace is missing required diagnostics. "
            + "; ".join(errors)
            + f". Available stages: {available_stage_list}. Available fields: {available_field_list}."
        )


def _load_accepted_step_trace_raw_records(path: str | Path) -> list[dict[str, object]]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [
                json.loads(line)
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if isinstance(payload, dict) and isinstance(payload.get("trace_points"), list):
            return [
                point for point in payload["trace_points"] if isinstance(point, dict)
            ]
        return [payload] if isinstance(payload, dict) else []
    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _accepted_trace_fields_by_stage(record: dict[str, object]) -> dict[str, set[str]]:
    stages = record.get("stages")
    if isinstance(stages, dict):
        return {
            str(stage_name): {
                str(field_name)
                for field_name, payload in stage_payload.items()
                if isinstance(payload, dict)
            }
            for stage_name, stage_payload in stages.items()
            if isinstance(stage_payload, dict)
        }
    fields = record.get("fields")
    if isinstance(fields, dict):
        stage_name = str(record.get("stage", _ACCEPTED_TRACE_STAGE))
        return {
            stage_name: {
                str(field_name)
                for field_name, payload in fields.items()
                if isinstance(payload, dict)
            }
        }
    return {}


def _normalize_accepted_trace_point(
    point: dict[str, object], *, preferred_stage: str
) -> dict[str, object]:
    stages = point.get("stages")
    if isinstance(stages, dict):
        stage_payload = stages.get(preferred_stage)
        if not isinstance(stage_payload, dict):
            available = ", ".join(str(name) for name in stages)
            raise KeyError(
                f"Accepted-step trace point does not contain stage {preferred_stage!r}; "
                f"available stages: {available}"
            )
        fields = {
            str(name): _normalize_accepted_trace_field_payload(payload)
            for name, payload in stage_payload.items()
            if isinstance(payload, dict)
        }
        return {
            "index": int(point.get("step_index", point.get("index", 0))),
            "time": float(point["time"]),
            "dt": float(point.get("dt", 0.0)),
            "solver_order": int(
                point.get(
                    "solver_order",
                    point.get("order", _solver_payload_value(point, "order", 0)),
                )
            ),
            "stage": preferred_stage,
            "fields": fields,
        }
    fields = point.get("fields")
    if not isinstance(fields, dict):
        raise KeyError("Accepted-step trace point must contain fields or stages.")
    return {
        "index": int(point.get("step_index", point.get("index", 0))),
        "time": float(point["time"]),
        "dt": float(point.get("dt", 0.0)),
        "solver_order": int(point.get("solver_order", point.get("order", 0))),
        "stage": str(point.get("stage", preferred_stage)),
        "fields": {
            str(name): _normalize_accepted_trace_field_payload(payload)
            for name, payload in fields.items()
            if isinstance(payload, dict)
        },
    }


def _solver_payload_value(
    point: dict[str, object], name: str, default: object
) -> object:
    solver = point.get("solver")
    if isinstance(solver, dict):
        return solver.get(name, default)
    return default


def _normalize_accepted_trace_field_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "active_metrics": _normalize_metric_payload(payload.get("active_metrics")),
        "target_adjacent_metrics": _normalize_metric_payload(
            payload.get("target_adjacent_metrics")
        ),
        "guard_metrics": _normalize_metric_payload(payload.get("guard_metrics")),
        "sample_lineout_y_indices": [
            int(value) for value in payload.get("sample_lineout_y_indices", [])
        ],
        "sample_lineout": [float(value) for value in payload.get("sample_lineout", [])],
    }


def _normalize_metric_payload(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {"max_abs": 0.0, "rms": 0.0}
    return {
        "max_abs": float(payload.get("max_abs", 0.0)),
        "rms": float(payload.get("rms", 0.0)),
    }


def _compare_accepted_step_trace_points(
    native_points: list[dict[str, object]],
    reference_points: list[dict[str, object]],
    *,
    time_tolerance: float,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    reference_by_time = {
        round(float(point["time"]) / max(time_tolerance, 1.0e-30)): point
        for point in reference_points
    }
    field_errors: dict[str, dict[str, object]] = {}
    matched_points: list[dict[str, object]] = []
    for native_point in native_points:
        native_time = float(native_point["time"])
        key = round(native_time / max(time_tolerance, 1.0e-30))
        reference_point = reference_by_time.get(key)
        if reference_point is None:
            reference_point = _nearest_trace_point(reference_points, native_time)
            if (
                reference_point is None
                or abs(float(reference_point["time"]) - native_time) > time_tolerance
            ):
                continue
        point_errors = _compare_accepted_step_fields(native_point, reference_point)
        matched_points.append(
            {
                "native_index": int(native_point["index"]),
                "reference_index": int(reference_point["index"]),
                "time": native_time,
                "reference_time": float(reference_point["time"]),
                "dt": float(native_point.get("dt", 0.0)),
                "reference_dt": float(reference_point.get("dt", 0.0)),
                "field_errors": point_errors,
            }
        )
        for name, error in point_errors.items():
            aggregate = field_errors.setdefault(
                name,
                {
                    "field": name,
                    "comparison_scope": _accepted_trace_field_scope(name),
                    "max_active_delta": 0.0,
                    "max_target_adjacent_delta": 0.0,
                    "max_guard_delta": 0.0,
                    "max_sample_lineout_delta": 0.0,
                    "worst_ranking_key": [0.0, 0.0, 0.0, 0.0],
                    "worst_time": native_time,
                },
            )
            _update_trace_error_aggregate(aggregate, error, native_time)
    return matched_points, field_errors


def _nearest_trace_point(
    points: list[dict[str, object]], time_value: float
) -> dict[str, object] | None:
    if not points:
        return None
    return min(points, key=lambda point: abs(float(point["time"]) - time_value))


def _compare_accepted_step_fields(
    native_point: dict[str, object], reference_point: dict[str, object]
) -> dict[str, dict[str, float]]:
    native_fields = native_point["fields"]
    reference_fields = reference_point["fields"]
    if not isinstance(native_fields, dict) or not isinstance(reference_fields, dict):
        raise TypeError("Accepted-step trace fields must be dictionaries.")
    common = sorted(set(native_fields).intersection(reference_fields))
    return {
        name: _compare_accepted_step_field_payload(
            native_fields[name], reference_fields[name]
        )
        for name in common
        if isinstance(native_fields[name], dict)
        and isinstance(reference_fields[name], dict)
    }


def _compare_accepted_step_field_payload(
    native_payload: dict[str, object], reference_payload: dict[str, object]
) -> dict[str, float]:
    return {
        "active_max_abs_delta": _metric_delta(
            native_payload, reference_payload, "active_metrics", "max_abs"
        ),
        "active_rms_delta": _metric_delta(
            native_payload, reference_payload, "active_metrics", "rms"
        ),
        "target_adjacent_max_abs_delta": _metric_delta(
            native_payload, reference_payload, "target_adjacent_metrics", "max_abs"
        ),
        "target_adjacent_rms_delta": _metric_delta(
            native_payload, reference_payload, "target_adjacent_metrics", "rms"
        ),
        "guard_max_abs_delta": _metric_delta(
            native_payload, reference_payload, "guard_metrics", "max_abs"
        ),
        "guard_rms_delta": _metric_delta(
            native_payload, reference_payload, "guard_metrics", "rms"
        ),
        "sample_lineout_max_abs_delta": _lineout_delta(
            native_payload, reference_payload
        ),
    }


def _metric_delta(
    native_payload: dict[str, object],
    reference_payload: dict[str, object],
    zone: str,
    metric: str,
) -> float:
    native_metrics = native_payload.get(zone, {})
    reference_metrics = reference_payload.get(zone, {})
    if not isinstance(native_metrics, dict) or not isinstance(reference_metrics, dict):
        return 0.0
    return abs(
        float(native_metrics.get(metric, 0.0))
        - float(reference_metrics.get(metric, 0.0))
    )


def _lineout_delta(
    native_payload: dict[str, object], reference_payload: dict[str, object]
) -> float:
    native_line = np.asarray(native_payload.get("sample_lineout", []), dtype=np.float64)
    reference_line = np.asarray(
        reference_payload.get("sample_lineout", []), dtype=np.float64
    )
    if native_line.size == 0 or reference_line.size == 0:
        return 0.0
    count = min(native_line.size, reference_line.size)
    return float(np.max(np.abs(native_line[:count] - reference_line[:count])))


def _update_trace_error_aggregate(
    aggregate: dict[str, object], point_error: dict[str, float], time_value: float
) -> None:
    updates = {
        "max_active_delta": point_error["active_max_abs_delta"],
        "max_target_adjacent_delta": point_error["target_adjacent_max_abs_delta"],
        "max_guard_delta": point_error["guard_max_abs_delta"],
        "max_sample_lineout_delta": point_error["sample_lineout_max_abs_delta"],
    }
    for key, value in updates.items():
        if float(value) > float(aggregate[key]):
            aggregate[key] = float(value)
    ranking_key = _accepted_trace_point_ranking_key(
        str(aggregate["comparison_scope"]),
        point_error,
    )
    if ranking_key > tuple(float(value) for value in aggregate["worst_ranking_key"]):
        aggregate["worst_ranking_key"] = [float(value) for value in ranking_key]
        aggregate["worst_time"] = float(time_value)


def _accepted_trace_field_scope(field_name: str) -> str:
    if field_name.startswith("ddt(") or field_name.startswith("SNV"):
        return "active_target_rhs_source"
    return "state_with_guard_boundary"


def _accepted_trace_field_ranking_key(item: dict[str, object]) -> tuple[float, ...]:
    if item.get("comparison_scope") == "active_target_rhs_source":
        return (
            float(item["max_target_adjacent_delta"]),
            float(item["max_active_delta"]),
            float(item["max_sample_lineout_delta"]),
            float(item["max_guard_delta"]),
        )
    return (
        float(item["max_target_adjacent_delta"]),
        float(item["max_guard_delta"]),
        float(item["max_active_delta"]),
        float(item["max_sample_lineout_delta"]),
    )


def _accepted_trace_point_ranking_key(
    comparison_scope: str, point_error: dict[str, float]
) -> tuple[float, ...]:
    if comparison_scope == "active_target_rhs_source":
        return (
            float(point_error["target_adjacent_max_abs_delta"]),
            float(point_error["active_max_abs_delta"]),
            float(point_error["sample_lineout_max_abs_delta"]),
            float(point_error["guard_max_abs_delta"]),
        )
    return (
        float(point_error["target_adjacent_max_abs_delta"]),
        float(point_error["guard_max_abs_delta"]),
        float(point_error["active_max_abs_delta"]),
        float(point_error["sample_lineout_max_abs_delta"]),
    )


def _state_with_reference_field(
    native_final: NeutralMixedState,
    reference_final: NeutralMixedState,
    field_name: str,
    *,
    mesh,
) -> NeutralMixedState:
    return _sanitize_neutral_state(
        NeutralMixedState(
            density=np.asarray(
                reference_final.density if field_name == "Nh" else native_final.density,
                dtype=np.float64,
            ).copy(),
            pressure=np.asarray(
                reference_final.pressure
                if field_name == "Ph"
                else native_final.pressure,
                dtype=np.float64,
            ).copy(),
            momentum=np.asarray(
                reference_final.momentum
                if field_name == "NVh"
                else native_final.momentum,
                dtype=np.float64,
            ).copy(),
        ),
        mesh,
    )


def _hybrid_state_register(
    config,
    native_final: NeutralMixedState,
    reference_final: NeutralMixedState,
    reference_initial: NeutralMixedState,
    native_balance: dict[str, np.ndarray],
    reference_balance: dict[str, np.ndarray],
    *,
    mesh,
    metrics,
    scalars: dict[str, float],
    timestep: float,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    swaps: dict[str, object] = {}
    ranked: list[dict[str, object]] = []
    for field_name in ("Nh", "Ph", "NVh"):
        hybrid = _state_with_reference_field(
            native_final, reference_final, field_name, mesh=mesh
        )
        hybrid_balance = _momentum_balance(
            config,
            hybrid,
            reference_initial,
            mesh=mesh,
            metrics=metrics,
            scalars=scalars,
            timestep=timestep,
        )
        term_delta = _term_delta_payload(
            hybrid_balance,
            reference_balance,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            line_x=line_x,
            line_z=line_z,
        )
        native_to_hybrid_term_delta = _term_delta_payload(
            hybrid_balance,
            native_balance,
            active_x=active_x,
            active_y=active_y,
            target_y_indices=target_y_indices,
            line_x=line_x,
            line_z=line_z,
        )
        residual = np.asarray(hybrid_balance["residual_rate"], dtype=np.float64)[
            active_x, active_y, :
        ]
        residual_metrics = _array_metrics(residual)
        term_metrics = (
            term_delta.get("term_metrics", {}) if isinstance(term_delta, dict) else {}
        )
        pressure_delta = float(
            term_metrics.get("pressure_gradient", {}).get(
                "target_adjacent_max_abs", 0.0
            )
        )  # type: ignore[union-attr]
        viscosity_delta = float(
            term_metrics.get("parallel_viscosity", {}).get(
                "target_adjacent_max_abs", 0.0
            )
        )  # type: ignore[union-attr]
        swaps[field_name] = {
            "swapped_field": field_name,
            "final_field_error_register": _final_field_error_register(
                hybrid,
                reference_final,
                active_x=active_x,
                active_y=active_y,
                target_y_indices=target_y_indices,
                line_x=line_x,
                line_z=line_z,
            ),
            "residual_rate_metrics": residual_metrics,
            "term_delta": term_delta,
            "native_to_hybrid_term_delta": native_to_hybrid_term_delta,
        }
        ranked.append(
            {
                "swapped_field": field_name,
                "residual_rate_max_abs": residual_metrics["max_abs"],
                "pressure_gradient_target_adjacent_max_abs": pressure_delta,
                "parallel_viscosity_target_adjacent_max_abs": viscosity_delta,
            }
        )
    return {
        "swaps": swaps,
        "ranked_by_pressure_gradient_target_adjacent_delta": sorted(
            ranked,
            key=lambda item: float(item["pressure_gradient_target_adjacent_max_abs"]),
        ),
        "ranked_by_parallel_viscosity_target_adjacent_delta": sorted(
            ranked,
            key=lambda item: float(item["parallel_viscosity_target_adjacent_max_abs"]),
        ),
        "interpretation": (
            "Each hybrid point replaces one native final field by the reference final field before "
            "reevaluating the native NVh balance. The ranking indicates which state variable most "
            "reduces the target-adjacent pressure-gradient or viscosity discrepancy."
        ),
    }


def _restore_trimmed_field(
    field: np.ndarray, template: np.ndarray, *, mesh
) -> np.ndarray:
    restored = np.asarray(template, dtype=np.float64).copy()
    field_array = np.asarray(field, dtype=np.float64)
    if field_array.shape == restored.shape:
        return field_array.copy()
    active_y = slice(mesh.ystart, mesh.yend + 1)
    if field_array.shape == (mesh.nx, mesh.yend - mesh.ystart + 1, mesh.nz):
        restored[:, active_y, :] = field_array
        return restored
    raise ValueError(
        f"Unsupported neutral mixed field shape {field_array.shape}; expected {restored.shape} or trimmed active-y shape."
    )


def _momentum_balance(
    config,
    state: NeutralMixedState,
    previous_state: NeutralMixedState,
    *,
    mesh,
    metrics,
    scalars: dict[str, float],
    timestep: float,
) -> dict[str, np.ndarray]:
    rhs = compute_neutral_mixed_rhs(
        config,
        state,
        section="h",
        mesh=mesh,
        metrics=metrics,
        meters_scale=float(scalars["rho_s0"]),
        tnorm=float(scalars["Tnorm"]),
    )
    time_derivative = (
        np.asarray(state.momentum, dtype=np.float64)
        - np.asarray(previous_state.momentum, dtype=np.float64)
    ) / float(timestep)
    terms = {"time_derivative": time_derivative}
    terms.update(rhs.momentum_terms)
    terms["rhs_sum"] = np.asarray(rhs.momentum, dtype=np.float64)
    terms["residual_rate"] = time_derivative - np.asarray(
        rhs.momentum, dtype=np.float64
    )
    return terms


def _balance_payload(
    balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    payload = {"lineouts": {}, "term_metrics": {}}
    for name, value in balance.items():
        array = np.asarray(value, dtype=np.float64)
        active = array[active_x, active_y, :]
        payload["lineouts"][name] = array[line_x, active_y, line_z].tolist()
        payload["term_metrics"][name] = {
            "max_abs": float(np.max(np.abs(active))),
            "rms": _rms(active),
        }
    return payload


def _target_adjacent_y_indices(mesh) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(mesh.ystart),
                int(min(mesh.ystart + 1, mesh.yend)),
                int(max(mesh.yend - 1, mesh.ystart)),
                int(mesh.yend),
            }
        )
    )


def _diagnostic_term_names(balance: dict[str, np.ndarray]) -> tuple[str, ...]:
    preferred = (
        "time_derivative",
        "parallel_inertia",
        "pressure_gradient",
        "perpendicular_diffusion",
        "parallel_viscosity",
        "perpendicular_viscosity",
        "rhs_sum",
        "residual_rate",
    )
    return tuple(name for name in preferred if name in balance)


def _array_metrics(array: np.ndarray) -> dict[str, float]:
    value = np.asarray(array, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(value))) if value.size else 0.0,
        "rms": _rms(value) if value.size else 0.0,
    }


def _target_offsets(active_y: slice, target_y_indices: tuple[int, ...]) -> np.ndarray:
    active_indices = np.arange(active_y.start, active_y.stop, dtype=np.int32)
    return np.flatnonzero(
        np.isin(active_indices, np.asarray(target_y_indices, dtype=np.int32))
    )


def _ranked_term_metrics(
    balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    target_offsets = _target_offsets(active_y, target_y_indices)
    ranked: list[dict[str, object]] = []
    for name in _diagnostic_term_names(balance):
        active = np.asarray(balance[name], dtype=np.float64)[active_x, active_y, :]
        target = active[:, target_offsets, :] if target_offsets.size else active
        metrics = _array_metrics(active)
        metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        ranked.append({"term": name, **metrics})
    return sorted(
        ranked, key=lambda item: float(item["target_adjacent_max_abs"]), reverse=True
    )


def _ranked_term_delta_metrics(
    native_balance: dict[str, np.ndarray],
    reference_balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    target_offsets = _target_offsets(active_y, target_y_indices)
    ranked: list[dict[str, object]] = []
    for name in _diagnostic_term_names(reference_balance):
        if name not in native_balance:
            continue
        delta = np.asarray(native_balance[name], dtype=np.float64) - np.asarray(
            reference_balance[name], dtype=np.float64
        )
        active = delta[active_x, active_y, :]
        target = active[:, target_offsets, :] if target_offsets.size else active
        metrics = _array_metrics(active)
        metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        ranked.append({"term": name, **metrics})
    return sorted(
        ranked, key=lambda item: float(item["target_adjacent_max_abs"]), reverse=True
    )


def _term_delta_payload(
    native_balance: dict[str, np.ndarray],
    reference_balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    payload: dict[str, object] = {"lineouts": {}, "term_metrics": {}}
    target_offsets = _target_offsets(active_y, target_y_indices)
    for name in _diagnostic_term_names(reference_balance):
        if name not in native_balance:
            continue
        delta = np.asarray(native_balance[name], dtype=np.float64) - np.asarray(
            reference_balance[name], dtype=np.float64
        )
        active = delta[active_x, active_y, :]
        target = active[:, target_offsets, :] if target_offsets.size else active
        metrics = _array_metrics(active)
        metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        payload["lineouts"][name] = delta[line_x, active_y, line_z].tolist()
        payload["term_metrics"][name] = metrics
    return payload


def _final_field_error_register(
    native_final: NeutralMixedState,
    reference_final: NeutralMixedState,
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    target_offsets = _target_offsets(active_y, target_y_indices)
    active_y_indices = np.arange(active_y.start, active_y.stop, dtype=np.int32)
    target_y_values = (
        active_y_indices[target_offsets] if target_offsets.size else active_y_indices
    )
    interior_offsets = np.asarray(
        [
            index
            for index in range(active_y_indices.size)
            if index not in set(int(value) for value in target_offsets)
        ],
        dtype=np.int32,
    )
    fields = {
        "Nh": (native_final.density, reference_final.density),
        "Ph": (native_final.pressure, reference_final.pressure),
        "NVh": (native_final.momentum, reference_final.momentum),
    }
    payload_fields: dict[str, object] = {}
    ranked: list[dict[str, object]] = []
    for name, (native_value, reference_value) in fields.items():
        delta = np.asarray(native_value, dtype=np.float64) - np.asarray(
            reference_value, dtype=np.float64
        )
        active = delta[active_x, active_y, :]
        target = active[:, target_offsets, :] if target_offsets.size else active
        interior = (
            active[:, interior_offsets, :]
            if interior_offsets.size
            else np.asarray([], dtype=np.float64)
        )
        metrics = _array_metrics(active)
        metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        metrics["interior_max_abs"] = (
            float(np.max(np.abs(interior))) if interior.size else 0.0
        )
        metrics["interior_rms"] = _rms(interior) if interior.size else 0.0
        metrics["target_to_interior_max_abs_ratio"] = (
            metrics["target_adjacent_max_abs"]
            / max(metrics["interior_max_abs"], 1.0e-30)
            if interior.size
            else None
        )
        payload_fields[name] = {
            **metrics,
            "lineout": delta[line_x, active_y, line_z].tolist(),
        }
        ranked.append({"field": name, **metrics})
    return {
        "fields": payload_fields,
        "ranked_by_target_adjacent_max_abs": sorted(
            ranked,
            key=lambda item: float(item["target_adjacent_max_abs"]),
            reverse=True,
        ),
        "target_y_indices": [int(value) for value in target_y_values],
        "interpretation": (
            "This register ranks final-state Nh, Ph, and NVh errors by target-adjacent and interior bands. "
            "Direct pressure-gradient and viscosity source formulas are closed by written reference "
            "diagnostics; the remaining entries track density, pressure, and momentum state drift "
            "that feeds those closed operators."
        ),
    }


def _state_driver_register(
    native_final: NeutralMixedState,
    reference_final: NeutralMixedState,
    native_balance: dict[str, np.ndarray],
    reference_balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    target_y_indices: tuple[int, ...],
    line_x: int,
    line_z: int,
    timestep: float,
) -> dict[str, object]:
    target_offsets = _target_offsets(active_y, target_y_indices)
    active_y_indices = np.arange(active_y.start, active_y.stop, dtype=np.int32)
    interior_offsets = np.asarray(
        [
            index
            for index in range(active_y_indices.size)
            if index not in set(int(value) for value in target_offsets)
        ],
        dtype=np.int32,
    )
    state_fields = {
        "Nh": (native_final.density, reference_final.density),
        "Ph": (native_final.pressure, reference_final.pressure),
        "NVh": (native_final.momentum, reference_final.momentum),
    }
    state_rate_errors: dict[str, object] = {}
    ranked_state_rates: list[dict[str, object]] = []
    for name, (native_value, reference_value) in state_fields.items():
        delta_rate = (
            np.asarray(native_value, dtype=np.float64)
            - np.asarray(reference_value, dtype=np.float64)
        ) / float(timestep)
        active = delta_rate[active_x, active_y, :]
        target = active[:, target_offsets, :] if target_offsets.size else active
        interior = (
            active[:, interior_offsets, :]
            if interior_offsets.size
            else np.asarray([], dtype=np.float64)
        )
        metrics = _array_metrics(active)
        metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target))) if target.size else 0.0
        )
        metrics["target_adjacent_rms"] = _rms(target) if target.size else 0.0
        metrics["interior_max_abs"] = (
            float(np.max(np.abs(interior))) if interior.size else 0.0
        )
        metrics["interior_rms"] = _rms(interior) if interior.size else 0.0
        state_rate_errors[name] = {
            **metrics,
            "lineout": delta_rate[line_x, active_y, line_z].tolist(),
        }
        ranked_state_rates.append({"field": name, **metrics})

    driver_pairs = (
        ("Nh", "parallel_inertia", "density_to_parallel_inertia"),
        ("Ph", "pressure_gradient", "pressure_to_pressure_gradient"),
        ("NVh", "parallel_viscosity", "momentum_to_parallel_viscosity"),
        ("NVh", "perpendicular_viscosity", "momentum_to_perpendicular_viscosity"),
    )
    driver_metrics: dict[str, object] = {}
    ranked_drivers: list[dict[str, object]] = []
    for field_name, term_name, label in driver_pairs:
        if term_name not in native_balance or term_name not in reference_balance:
            continue
        native_field, reference_field = state_fields[field_name]
        state_delta = np.asarray(native_field, dtype=np.float64) - np.asarray(
            reference_field, dtype=np.float64
        )
        term_delta = np.asarray(
            native_balance[term_name], dtype=np.float64
        ) - np.asarray(reference_balance[term_name], dtype=np.float64)
        active_state = state_delta[active_x, active_y, :]
        active_term = term_delta[active_x, active_y, :]
        target_state = (
            active_state[:, target_offsets, :] if target_offsets.size else active_state
        )
        target_term = (
            active_term[:, target_offsets, :] if target_offsets.size else active_term
        )
        interior_term = (
            active_term[:, interior_offsets, :]
            if interior_offsets.size
            else np.asarray([], dtype=np.float64)
        )
        target_state_flat = target_state.ravel()
        target_term_flat = target_term.ravel()
        term_metrics = _array_metrics(active_term)
        term_metrics["target_adjacent_max_abs"] = (
            float(np.max(np.abs(target_term))) if target_term.size else 0.0
        )
        term_metrics["target_adjacent_rms"] = (
            _rms(target_term) if target_term.size else 0.0
        )
        term_metrics["interior_max_abs"] = (
            float(np.max(np.abs(interior_term))) if interior_term.size else 0.0
        )
        term_metrics["interior_rms"] = (
            _rms(interior_term) if interior_term.size else 0.0
        )
        term_metrics["target_term_to_interior_term_max_abs_ratio"] = (
            term_metrics["target_adjacent_max_abs"]
            / max(term_metrics["interior_max_abs"], 1.0e-30)
            if interior_term.size
            else None
        )
        term_metrics["target_term_per_state_max_abs"] = (
            term_metrics["target_adjacent_max_abs"]
            / max(float(np.max(np.abs(target_state))), 1.0e-30)
            if target_state.size
            else None
        )
        term_metrics["target_state_term_correlation"] = _signed_correlation(
            target_state_flat, target_term_flat
        )
        driver_metrics[label] = {
            "field": field_name,
            "term": term_name,
            **term_metrics,
            "state_delta_lineout": state_delta[line_x, active_y, line_z].tolist(),
            "term_delta_lineout": term_delta[line_x, active_y, line_z].tolist(),
        }
        ranked_drivers.append(
            {"driver": label, "field": field_name, "term": term_name, **term_metrics}
        )

    return {
        "state_rate_errors": state_rate_errors,
        "ranked_state_rate_errors": sorted(
            ranked_state_rates,
            key=lambda item: float(item["target_adjacent_max_abs"]),
            reverse=True,
        ),
        "momentum_driver_deltas": driver_metrics,
        "ranked_momentum_driver_deltas": sorted(
            ranked_drivers,
            key=lambda item: float(item["target_adjacent_max_abs"]),
            reverse=True,
        ),
        "target_y_indices": [int(value) for value in active_y_indices[target_offsets]],
        "interpretation": (
            "This register links final-state Nh, Ph, and NVh drift to the named NVh driver-term deltas. "
            "Large pressure-gradient or viscosity deltas with machine-precision direct source closure "
            "indicate a boundary/history state mismatch feeding an otherwise closed operator."
        ),
    }


def _signed_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).ravel()
    right_array = np.asarray(right, dtype=np.float64).ravel()
    if left_array.size == 0 or right_array.size == 0:
        return None
    left_centered = left_array - float(np.mean(left_array))
    right_centered = right_array - float(np.mean(right_array))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _dominant_residual_cells(
    balance: dict[str, np.ndarray],
    *,
    active_x: slice,
    active_y: slice,
    count: int,
) -> list[dict[str, object]]:
    residual = np.asarray(
        balance["residual_rate"][active_x, active_y, :], dtype=np.float64
    )
    flat_count = min(int(count), residual.size)
    if flat_count == 0:
        return []
    flat_indices = np.argpartition(np.abs(residual).ravel(), -flat_count)[-flat_count:]
    flat_indices = flat_indices[
        np.argsort(np.abs(residual).ravel()[flat_indices])[::-1]
    ]
    terms = tuple(name for name in _diagnostic_term_names(balance) if name != "rhs_sum")
    cells: list[dict[str, object]] = []
    for flat_index in flat_indices:
        x_offset, y_offset, z = np.unravel_index(int(flat_index), residual.shape)
        x = int(active_x.start + x_offset)
        y = int(active_y.start + y_offset)
        cells.append(
            {
                "x": x,
                "y": y,
                "z": int(z),
                "terms": {
                    name: float(np.asarray(balance[name], dtype=np.float64)[x, y, z])
                    for name in terms
                },
            }
        )
    return cells


def _write_neutral_mixed_term_balance_arrays(
    report: dict[str, object], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "active_y_indices": np.asarray(report["active_y_indices"], dtype=np.float64),
        "final_momentum_error_lineout": np.asarray(
            report["final_momentum_error"]["lineout"], dtype=np.float64
        ),
    }
    final_field_register = report.get("final_field_error_register")
    if isinstance(final_field_register, dict):
        fields = final_field_register.get("fields", {})
        if isinstance(fields, dict):
            for field_name, payload in fields.items():
                if isinstance(payload, dict) and "lineout" in payload:
                    arrays[f"final_field_error_{field_name}_lineout"] = np.asarray(
                        payload["lineout"], dtype=np.float64
                    )
    state_driver_register = report.get("state_driver_register")
    if isinstance(state_driver_register, dict):
        state_rates = state_driver_register.get("state_rate_errors", {})
        if isinstance(state_rates, dict):
            for field_name, payload in state_rates.items():
                if isinstance(payload, dict) and "lineout" in payload:
                    arrays[f"state_rate_error_{field_name}_lineout"] = np.asarray(
                        payload["lineout"], dtype=np.float64
                    )
        driver_deltas = state_driver_register.get("momentum_driver_deltas", {})
        if isinstance(driver_deltas, dict):
            for driver_name, payload in driver_deltas.items():
                if not isinstance(payload, dict):
                    continue
                if "state_delta_lineout" in payload:
                    arrays[f"state_driver_{driver_name}_state_delta_lineout"] = (
                        np.asarray(
                            payload["state_delta_lineout"],
                            dtype=np.float64,
                        )
                    )
                if "term_delta_lineout" in payload:
                    arrays[f"state_driver_{driver_name}_term_delta_lineout"] = (
                        np.asarray(
                            payload["term_delta_lineout"],
                            dtype=np.float64,
                        )
                    )
    for group_name in ("native_balance", "reference_balance"):
        for term_name, lineout in report[group_name]["lineouts"].items():
            arrays[f"{group_name}_{term_name}_lineout"] = np.asarray(
                lineout, dtype=np.float64
            )
    term_delta = report.get("term_delta")
    if isinstance(term_delta, dict):
        for term_name, lineout in term_delta.get("lineouts", {}).items():
            arrays[f"term_delta_{term_name}_lineout"] = np.asarray(
                lineout, dtype=np.float64
            )
    diagnostics = report.get("hermes_diagnostic_outputs")
    if isinstance(diagnostics, dict):
        lineouts = diagnostics.get("lineouts", {})
        if isinstance(lineouts, dict):
            for term_name, lineout in lineouts.items():
                arrays[f"hermes_diagnostic_{term_name}_lineout"] = np.asarray(
                    lineout, dtype=np.float64
                )
        reconstructions = diagnostics.get("matched_reconstructions", {})
        if isinstance(reconstructions, dict):
            for term_name, reconstruction in reconstructions.items():
                if isinstance(reconstruction, dict) and "lineout" in reconstruction:
                    arrays[f"hermes_matched_{term_name}_lineout"] = np.asarray(
                        reconstruction["lineout"],
                        dtype=np.float64,
                    )
        direct_comparisons = diagnostics.get("direct_comparisons", {})
        if isinstance(direct_comparisons, dict):
            for term_name, comparison in direct_comparisons.items():
                if not isinstance(comparison, dict):
                    continue
                for line_name in (
                    "matched_reconstruction_lineout",
                    "scaled_direct_lineout",
                    "scaled_difference_lineout",
                ):
                    if line_name in comparison:
                        arrays[f"hermes_direct_comparison_{term_name}_{line_name}"] = (
                            np.asarray(
                                comparison[line_name],
                                dtype=np.float64,
                            )
                        )
    np.savez(target, **arrays)
    return target


def _hermes_diagnostic_payload(
    path: str | Path,
    *,
    active_x: slice,
    active_y: slice,
    line_x: int,
    line_z: int,
    matched_sources: dict[str, np.ndarray | None] | None = None,
) -> dict[str, object]:
    try:
        from netCDF4 import Dataset
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency is part of the runtime package
        raise ImportError(
            "netCDF4 is required to read Hermès diagnostic NetCDF output."
        ) from exc

    target = Path(path).expanduser().resolve()
    field_names = (
        "ddt(NVh)",
        "SNVh",
        "SNVh_pressure_gradient",
        "SNVh_parallel_viscosity",
        "SNVh_perpendicular_viscosity",
        "mfh_visc_par_ylow",
        "mfh_visc_perp_xlow",
        "mfh_visc_perp_ylow",
        "mfh_adv_perp_xlow",
        "mfh_adv_perp_ylow",
    )
    payload: dict[str, object] = {
        "source_nc": target.name,
        "lineouts": {},
        "field_metrics": {},
        "matched_reconstructions": {},
        "direct_comparisons": {},
        "variables_present": [],
        "variables_missing": [],
        "interpretation": {
            "direct_hermes_outputs": (
                "Hermès writes ddt(NVh), external/source terms, and selected "
                "momentum-source and momentum-flow diagnostics when "
                "neutral_mixed output_ddt=true and diagnose=true. A local "
                "Hermès diagnostic patch can also write SNVh_pressure_gradient "
                "for direct -Grad_par(Pn) parity."
            ),
            "pressure_gradient_limitation": (
                "The neutral pressure-gradient source appears in Hermès as "
                "-Grad_par(Pn) inside neutral_mixed.cxx. Stock Hermès output "
                "does not write that term as a named diagnostic, so this "
                "report always carries a matched postprocessed reconstruction "
                "when reference arrays are available. When the local Hermès "
                "diagnostic patch is present, SNVh_pressure_gradient provides "
                "the direct written-variable comparison."
            ),
            "viscosity_limitation": (
                "Stock Hermès writes viscosity flows but not the parallel and "
                "perpendicular viscosity source terms separately. When the "
                "local diagnostic patch is present, SNVh_parallel_viscosity "
                "and SNVh_perpendicular_viscosity provide direct source-level "
                "comparisons for the native closure."
            ),
        },
    }
    direct_fields: dict[str, np.ndarray] = {}
    matched_arrays: dict[str, np.ndarray] = {}
    matched_sources = matched_sources or {}
    reconstruction_names = {
        "SNVh_pressure_gradient": "pressure_gradient",
        "SNVh_parallel_viscosity": "parallel_viscosity",
        "SNVh_perpendicular_viscosity": "perpendicular_viscosity",
    }
    reconstruction_descriptions = {
        "SNVh_pressure_gradient": "matched postprocessed reconstruction of Hermès final Ph through native -Grad_par(Pn) term",
        "SNVh_parallel_viscosity": "matched postprocessed reconstruction of Hermès final NVh through native parallel-viscosity term",
        "SNVh_perpendicular_viscosity": "matched postprocessed reconstruction of Hermès final NVh through native perpendicular-viscosity term",
    }
    for diagnostic_name, matched_source in matched_sources.items():
        if matched_source is None:
            continue
        matched = np.asarray(matched_source, dtype=np.float64)
        matched_arrays[diagnostic_name] = matched
        active = matched[active_x, active_y, :]
        reconstruction_name = reconstruction_names.get(diagnostic_name, diagnostic_name)
        payload["matched_reconstructions"][reconstruction_name] = {
            "source": reconstruction_descriptions.get(
                diagnostic_name, "matched postprocessed native reconstruction"
            ),
            "lineout": matched[line_x, active_y, line_z].tolist(),
            "field_metrics": {
                "max_abs": float(np.max(np.abs(active))),
                "rms": _rms(active),
            },
            "parity_scope": (
                "This isolates the same mathematical source term on the Hermès final state. "
                "A direct Hermès variable is compared when the local diagnostic patch writes it."
            ),
        }
    with Dataset(target) as dataset:
        for name in field_names:
            if name not in dataset.variables:
                payload["variables_missing"].append(name)
                continue
            data = np.asarray(dataset.variables[name][-1], dtype=np.float64)
            active = data[active_x, active_y, :]
            payload["variables_present"].append(name)
            payload["lineouts"][name] = data[line_x, active_y, line_z].tolist()
            payload["field_metrics"][name] = {
                "max_abs": float(np.max(np.abs(active))),
                "rms": _rms(active),
            }
            if name in matched_arrays:
                direct_fields[name] = data
    for diagnostic_name, direct in direct_fields.items():
        matched = matched_arrays[diagnostic_name]
        matched_active = matched[active_x, active_y, :]
        direct_active = direct[active_x, active_y, :]
        denominator = float(np.sum(np.square(direct_active)))
        scale = (
            float(np.sum(matched_active * direct_active) / denominator)
            if denominator > 0.0
            else 0.0
        )
        scaled_direct = scale * direct
        difference = scaled_direct - matched
        active_difference = difference[active_x, active_y, :]
        payload["direct_comparisons"][diagnostic_name] = {
            "comparison": "least-squares scale of direct written reference diagnostic to native-normalized matched reconstruction",
            "least_squares_scale_to_native_units": scale,
            "scaled_difference_metrics": {
                "max_abs": float(np.max(np.abs(active_difference))),
                "rms": _rms(active_difference),
            },
            "matched_reconstruction_lineout": matched[
                line_x, active_y, line_z
            ].tolist(),
            "scaled_direct_lineout": scaled_direct[line_x, active_y, line_z].tolist(),
            "scaled_difference_lineout": difference[line_x, active_y, line_z].tolist(),
        }
    return payload


def _default_hermes_binary(reference_root: Path) -> Path:
    candidates = (
        reference_root / "build" / "hermes-3",
        reference_root / "hermes-3",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Hermès executable not found. Pass hermes_binary explicitly or build Hermès under reference_root/build/hermes-3."
    )


def _set_root_option(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text, count=1)
    return f"{key} = {value}\n{text}"


def _set_section_option(text: str, section: str, key: str, value: str) -> str:
    header = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
    if header is None:
        suffix = "" if text.endswith("\n") else "\n"
        return f"{text}{suffix}\n[{section}]\n{key} = {value}\n"
    next_header = re.search(r"(?m)^\[[^\]]+\]\s*$", text[header.end() :])
    section_end = (
        len(text) if next_header is None else header.end() + next_header.start()
    )
    body = text[header.end() : section_end]
    option_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if option_pattern.search(body):
        body = option_pattern.sub(rf"\g<1>{value}", body, count=1)
    else:
        insertion = f"{key} = {value}\n"
        body = (
            f"\n{insertion}{body.lstrip()}"
            if not body.startswith("\n")
            else f"\n{insertion}{body[1:]}"
        )
    return text[: header.end()] + body + text[section_end:]


def _rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _sanitize_public_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    if "hermes-3" in parts:
        index = parts.index("hermes-3")
        suffix = Path(*parts[index + 1 :]).as_posix() if parts[index + 1 :] else ""
        return "<reference-root>" if not suffix else f"<reference-root>/{suffix}"
    if "jax_drb" in parts:
        index = parts.index("jax_drb")
        suffix = Path(*parts[index + 1 :]).as_posix() if parts[index + 1 :] else ""
        return "<repo-root>" if not suffix else f"<repo-root>/{suffix}"
    home = Path.home().resolve()
    try:
        return f"~/{resolved.relative_to(home).as_posix()}"
    except ValueError:
        return resolved.as_posix()
