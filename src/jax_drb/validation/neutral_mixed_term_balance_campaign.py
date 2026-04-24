from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import load_bout_input
from ..native import run_curated_case
from ..native.mesh import build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..native.neutral_mixed import (
    _sanitize_neutral_state,
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
    binary = Path(hermes_binary).expanduser().resolve() if hermes_binary is not None else _default_hermes_binary(root)
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
        raise RuntimeError(f"Hermès neutral-mixed diagnostic rerun failed with exit code {completed.returncode}:\n{tail}")
    dump_path = data_dir / "BOUT.dmp.0.nc"
    if not dump_path.exists():
        raise FileNotFoundError(f"Hermès neutral-mixed diagnostic rerun did not produce {dump_path}")
    return dump_path


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
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_npz_path = _write_neutral_mixed_term_balance_arrays(report, data_dir / f"{case_label}.npz")
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
    root = Path(reference_root).expanduser().resolve() if reference_root is not None else default_reference_root()
    if input_path is None:
        if root is None:
            raise FileNotFoundError("reference_root or input_path is required for neutral mixed term-balance diagnostics.")
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
        else repo_root() / "references" / "baselines" / "reference_arrays" / f"{case_name}.npz"
    )
    reference_history = _load_neutral_mixed_history_npz(reference_npz)
    native_history = (
        _load_neutral_mixed_history_npz(native_arrays_npz)
        if native_arrays_npz is not None
        else _native_history_from_curated_case(case_name, reference_root=root)
    )
    time_points = np.asarray(reference_history["time_points"], dtype=np.float64)
    if time_points.size < 2:
        raise ValueError("Neutral mixed term-balance diagnostics require at least two stored time points.")
    timestep = float(time_points[-1] - time_points[0])
    reference_initial = _state_from_trimmed_history(reference_history, template_state, time_index=0, mesh=mesh)
    reference_final = _state_from_trimmed_history(reference_history, template_state, time_index=-1, mesh=mesh)
    native_final = _state_from_trimmed_history(native_history, template_state, time_index=-1, mesh=mesh)
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

    active_y = slice(mesh.ystart, mesh.yend + 1)
    final_error = np.asarray(native_final.momentum[:, active_y, :] - reference_final.momentum[:, active_y, :], dtype=np.float64)
    worst_x, worst_y_active, worst_z = np.unravel_index(np.argmax(np.abs(final_error)), final_error.shape)
    worst_y = int(mesh.ystart + worst_y_active)
    line_x = int(worst_x)
    line_z = int(worst_z)
    y_indices = np.arange(mesh.ystart, mesh.yend + 1, dtype=np.int32)
    reference_residual_active = np.asarray(reference_balance["residual_rate"][:, active_y, :], dtype=np.float64)
    native_residual_active = np.asarray(native_balance["residual_rate"][:, active_y, :], dtype=np.float64)

    report: dict[str, object] = {
        "case_name": case_name,
        "reference_code": "hermes-3",
        "input_path": _sanitize_public_path(input_path),
        "reference_arrays_npz": _sanitize_public_path(reference_npz),
        "timestep": timestep,
        "field": "NVh",
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
            "lineout": final_error[line_x, :, line_z].tolist(),
        },
        "native_balance": _balance_payload(native_balance, active_y=active_y, line_x=line_x, line_z=line_z),
        "reference_balance": _balance_payload(reference_balance, active_y=active_y, line_x=line_x, line_z=line_z),
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
            active_y=active_y,
            line_x=line_x,
            line_z=line_z,
        )
    return report


def save_neutral_mixed_term_balance_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
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
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)

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
    values = np.asarray([float(reference["term_metrics"][name]["max_abs"]) for name in term_order[1:]], dtype=np.float64)
    x = np.arange(len(values))
    axes[1, 1].bar(x, np.maximum(values, 1.0e-16), color=[colors[name] for name in term_order[1:]])
    axes[1, 1].set_xticks(x, bar_labels)
    style_axis(
        axes[1, 1],
        title="Max |native term| for Hermès final state",
        ylabel="max absolute value",
        yscale="log",
        grid="y",
    )
    annotate_bars(axes[1, 1], x, np.maximum(values, 1.0e-16), fmt="{:.1e}", fontsize=7.8)

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


def _native_history_from_curated_case(case_name: str, *, reference_root: Path | None) -> dict[str, object]:
    if reference_root is None:
        raise FileNotFoundError("reference_root is required when native_arrays_npz is not supplied.")
    result = run_curated_case(case_name, reference_root=reference_root)
    return {
        "time_points": np.asarray(result.time_points, dtype=np.float64),
        "Nh": np.asarray(result.variables["Nh"], dtype=np.float64),
        "Ph": np.asarray(result.variables["Ph"], dtype=np.float64),
        "NVh": np.asarray(result.variables["NVh"], dtype=np.float64),
    }


def _state_from_trimmed_history(
    history: dict[str, object],
    template: NeutralMixedState,
    *,
    time_index: int,
    mesh,
) -> NeutralMixedState:
    density = _restore_trimmed_field(np.asarray(history["Nh"], dtype=np.float64)[time_index], template.density, mesh=mesh)
    pressure = _restore_trimmed_field(np.asarray(history["Ph"], dtype=np.float64)[time_index], template.pressure, mesh=mesh)
    momentum = _restore_trimmed_field(np.asarray(history["NVh"], dtype=np.float64)[time_index], template.momentum, mesh=mesh)
    return _sanitize_neutral_state(
        NeutralMixedState(density=density, pressure=pressure, momentum=momentum),
        mesh,
    )


def _restore_trimmed_field(field: np.ndarray, template: np.ndarray, *, mesh) -> np.ndarray:
    restored = np.asarray(template, dtype=np.float64).copy()
    field_array = np.asarray(field, dtype=np.float64)
    if field_array.shape == restored.shape:
        return field_array.copy()
    active_y = slice(mesh.ystart, mesh.yend + 1)
    if field_array.shape == (mesh.nx, mesh.yend - mesh.ystart + 1, mesh.nz):
        restored[:, active_y, :] = field_array
        return restored
    raise ValueError(f"Unsupported neutral mixed field shape {field_array.shape}; expected {restored.shape} or trimmed active-y shape.")


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
    time_derivative = (np.asarray(state.momentum, dtype=np.float64) - np.asarray(previous_state.momentum, dtype=np.float64)) / float(timestep)
    terms = {"time_derivative": time_derivative}
    terms.update(rhs.momentum_terms)
    terms["rhs_sum"] = np.asarray(rhs.momentum, dtype=np.float64)
    terms["residual_rate"] = time_derivative - np.asarray(rhs.momentum, dtype=np.float64)
    return terms


def _balance_payload(
    balance: dict[str, np.ndarray],
    *,
    active_y: slice,
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    payload = {"lineouts": {}, "term_metrics": {}}
    for name, value in balance.items():
        array = np.asarray(value, dtype=np.float64)
        active = array[:, active_y, :]
        payload["lineouts"][name] = array[line_x, active_y, line_z].tolist()
        payload["term_metrics"][name] = {
            "max_abs": float(np.max(np.abs(active))),
            "rms": _rms(active),
        }
    return payload


def _write_neutral_mixed_term_balance_arrays(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "active_y_indices": np.asarray(report["active_y_indices"], dtype=np.float64),
        "final_momentum_error_lineout": np.asarray(report["final_momentum_error"]["lineout"], dtype=np.float64),
    }
    for group_name in ("native_balance", "reference_balance"):
        for term_name, lineout in report[group_name]["lineouts"].items():
            arrays[f"{group_name}_{term_name}_lineout"] = np.asarray(lineout, dtype=np.float64)
    diagnostics = report.get("hermes_diagnostic_outputs")
    if isinstance(diagnostics, dict):
        lineouts = diagnostics.get("lineouts", {})
        if isinstance(lineouts, dict):
            for term_name, lineout in lineouts.items():
                arrays[f"hermes_diagnostic_{term_name}_lineout"] = np.asarray(lineout, dtype=np.float64)
    np.savez(target, **arrays)
    return target


def _hermes_diagnostic_payload(
    path: str | Path,
    *,
    active_y: slice,
    line_x: int,
    line_z: int,
) -> dict[str, object]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - dependency is part of the runtime package
        raise ImportError("netCDF4 is required to read Hermès diagnostic NetCDF output.") from exc

    target = Path(path).expanduser().resolve()
    field_names = (
        "ddt(NVh)",
        "SNVh",
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
        "variables_present": [],
        "variables_missing": [],
        "interpretation": {
            "direct_hermes_outputs": (
                "Hermès writes ddt(NVh), external/source terms, and selected "
                "momentum-flow diagnostics when neutral_mixed output_ddt=true "
                "and diagnose=true."
            ),
            "pressure_gradient_limitation": (
                "The neutral pressure-gradient source appears in Hermès as "
                "-Grad_par(Pn) inside neutral_mixed.cxx, but it is not written "
                "as a named diagnostic in the stock output. Direct "
                "pressure-gradient parity therefore still requires either a "
                "small Hermès diagnostic patch or a matched postprocessed "
                "operator reconstruction."
            ),
        },
    }
    with Dataset(target) as dataset:
        for name in field_names:
            if name not in dataset.variables:
                payload["variables_missing"].append(name)
                continue
            data = np.asarray(dataset.variables[name][-1], dtype=np.float64)
            active = data[:, active_y, :]
            payload["variables_present"].append(name)
            payload["lineouts"][name] = data[line_x, active_y, line_z].tolist()
            payload["field_metrics"][name] = {
                "max_abs": float(np.max(np.abs(active))),
                "rms": _rms(active),
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
    section_end = len(text) if next_header is None else header.end() + next_header.start()
    body = text[header.end() : section_end]
    option_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if option_pattern.search(body):
        body = option_pattern.sub(rf"\g<1>{value}", body, count=1)
    else:
        insertion = f"{key} = {value}\n"
        body = f"\n{insertion}{body.lstrip()}" if not body.startswith("\n") else f"\n{insertion}{body[1:]}"
    return text[: header.end()] + body + text[section_end:]


def _rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _sanitize_public_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    try:
        return f"~/{resolved.relative_to(home).as_posix()}"
    except ValueError:
        return resolved.as_posix()
