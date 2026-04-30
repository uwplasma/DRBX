from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np

from ..native import run_curated_case
from ..parity.arrays import build_array_payload_from_summary_payload, build_dataset_array_payload
from ..parity.reference import discover_reference_binary, resolve_reference_case, run_reference_case
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class NeutralMixedBoundaryCampaignArtifacts:
    report_json_path: Path
    report_npz_path: Path
    report_plot_png_path: Path


def create_neutral_mixed_boundary_campaign_package(
    *,
    reference_root: str | Path,
    output_root: str | Path,
    case_name: str = "neutral_mixed_one_step",
    case_label: str = "neutral_mixed_boundary_campaign",
) -> NeutralMixedBoundaryCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_neutral_mixed_boundary_campaign_report(
        reference_root=reference_root,
        case_name=case_name,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_npz_path = _write_neutral_mixed_boundary_campaign_arrays(report, data_dir / f"{case_label}.npz")
    report_plot_png_path = save_neutral_mixed_boundary_campaign_plot(
        report,
        images_dir / f"{case_label}.png",
    )
    return NeutralMixedBoundaryCampaignArtifacts(
        report_json_path=report_json_path,
        report_npz_path=report_npz_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_neutral_mixed_boundary_campaign_report(
    *,
    reference_root: str | Path,
    case_name: str = "neutral_mixed_one_step",
) -> dict[str, object]:
    reference_root_path = Path(reference_root).expanduser().resolve()
    reference_binary = discover_reference_binary(reference_root=reference_root_path)
    reference_case, _ = resolve_reference_case(case_name, reference_root=reference_root_path)
    reference_input_path = (
        reference_case.input_path(reference_root_path)
        if callable(getattr(reference_case, "input_path", None))
        else Path(reference_case.input_path)
    )

    native_started_at = perf_counter()
    native_result = run_curated_case(case_name, reference_root=reference_root_path)
    native_elapsed_seconds = perf_counter() - native_started_at

    with tempfile.TemporaryDirectory(prefix=f"jaxdrb-{case_name}-") as workdir_name:
        workdir = Path(workdir_name)
        reference_started_at = perf_counter()
        reference_execution = run_reference_case(
            case_name,
            reference_root=reference_root_path,
            workdir=workdir,
        )
        reference_elapsed_seconds = perf_counter() - reference_started_at
        dataset_path = workdir / "BOUT.dmp.0.nc"
        reference_payload = build_dataset_array_payload(
            dataset_path,
            case_name=reference_execution.summary.case_name,
            parity_mode=reference_execution.summary.parity_mode,
            capability_tier=reference_execution.summary.capability_tier,
            compare_variables=reference_execution.summary.compare_variables,
            component_labels=reference_execution.summary.component_labels,
            overrides=reference_execution.summary.overrides,
            trim_x_guards=reference_case.trim_x_guards,
            x_guards=native_result.run_config.mesh.mxg,
            trim_y_guards=reference_case.trim_y_guards,
            y_guards=native_result.run_config.mesh.myg,
            configured_nout=reference_execution.summary.nout,
            configured_timestep=reference_execution.summary.timestep,
            producer="external-reference-rerun",
        )

    native_payload = build_array_payload_from_summary_payload(native_result.payload, native_result.variables)
    compare_variables = tuple(reference_execution.summary.compare_variables)
    density = np.asarray(reference_payload["variables"]["Nh"], dtype=np.float64)
    x_index = density.shape[1] // 2
    y_index = density.shape[2] // 2
    z_index = density.shape[3] // 2
    y_coordinate = np.arange(density.shape[2], dtype=np.int32)
    time_points = np.asarray(reference_payload["time_points"], dtype=np.float64)
    profile_index = int(time_points.size - 1)

    profiles: dict[str, dict[str, object]] = {}
    field_metrics: dict[str, dict[str, float]] = {}
    worst_field = ""
    worst_max_abs_error = -1.0
    boundary_band = 2

    for variable_name in compare_variables:
        reference_history = np.asarray(reference_payload["variables"][variable_name], dtype=np.float64)
        native_history = np.asarray(native_payload["variables"][variable_name], dtype=np.float64)
        absolute_error = np.abs(native_history - reference_history)
        worst_index = np.unravel_index(np.argmax(absolute_error[profile_index]), absolute_error[profile_index].shape)
        line_x = int(worst_index[0])
        line_z = int(worst_index[2])
        line_reference = reference_history[profile_index, line_x, :, line_z]
        line_native = native_history[profile_index, line_x, :, line_z]
        line_abs_diff = np.abs(line_native - line_reference)
        max_abs_profile_y = np.max(absolute_error[profile_index], axis=(0, 2))
        lower_boundary_abs_error = float(np.max(max_abs_profile_y[:boundary_band]))
        upper_boundary_abs_error = float(np.max(max_abs_profile_y[-boundary_band:]))
        interior_abs_error = (
            float(np.max(max_abs_profile_y[boundary_band:-boundary_band]))
            if max_abs_profile_y.size > 2 * boundary_band
            else float(np.max(max_abs_profile_y))
        )
        max_abs_error = float(np.max(absolute_error))
        rms_error = float(np.sqrt(np.mean(np.square(native_history - reference_history))))
        profiles[variable_name] = {
            "reference_lineout": line_reference.tolist(),
            "native_lineout": line_native.tolist(),
            "abs_diff_lineout": line_abs_diff.tolist(),
            "max_abs_profile_y": max_abs_profile_y.tolist(),
            "line_x": line_x,
            "line_z": line_z,
        }
        field_metrics[variable_name] = {
            "max_abs_error": max_abs_error,
            "rms_error": rms_error,
            "lineout_max_abs_error": float(np.max(line_abs_diff)),
            "lower_boundary_max_abs_error": lower_boundary_abs_error,
            "upper_boundary_max_abs_error": upper_boundary_abs_error,
            "interior_max_abs_error": interior_abs_error,
        }
        if max_abs_error > worst_max_abs_error:
            worst_field = variable_name
            worst_max_abs_error = max_abs_error

    return {
        "case_name": case_name,
        "reference_code": "hermes-3",
        "reference_root": _sanitize_public_path(reference_root_path),
        "reference_binary": _sanitize_public_path(reference_binary),
        "reference_path": _sanitize_public_path(reference_input_path),
        "compare_variables": list(compare_variables),
        "time_points": time_points.tolist(),
        "x_index": int(x_index),
        "y_index": int(y_index),
        "z_index": int(z_index),
        "y_coordinate": y_coordinate.tolist(),
        "profile_time_index": profile_index,
        "boundary_band_cells": int(boundary_band),
        "native_elapsed_seconds": float(native_elapsed_seconds),
        "reference_elapsed_seconds": float(reference_elapsed_seconds),
        "native_to_reference_runtime_ratio": (
            float(native_elapsed_seconds / reference_elapsed_seconds)
            if reference_elapsed_seconds > 0.0
            else 0.0
        ),
        "profiles": profiles,
        "field_metrics": field_metrics,
        "worst_field": worst_field,
        "worst_max_abs_error": float(worst_max_abs_error),
        "notes": {
            "comparison_surface": "live_native_vs_live_reference_neutral_mixed_centerline",
            "plot_note": "The lineouts follow the literature pattern of 1D open-field parallel-profile comparison at a fixed cross-field location, with a separate absolute-error panel to expose boundary-localized mismatch.",
        },
    }


def save_neutral_mixed_boundary_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    profiles = report["profiles"]
    y_coordinate = np.asarray(report["y_coordinate"], dtype=np.float64)
    boundary_band = int(report["boundary_band_cells"])
    runtime_ratio = float(report["native_to_reference_runtime_ratio"])
    worst_field = str(report["worst_field"])
    worst_error = float(report["worst_max_abs_error"])

    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
    variable_order = (
        ("Nh", "#1d3557"),
        ("Ph", "#2a9d8f"),
        ("NVh", "#d1495b"),
    )

    for axis, (variable_name, color) in zip(axes.flat[:3], variable_order, strict=True):
        reference_lineout = np.asarray(profiles[variable_name]["reference_lineout"], dtype=np.float64)
        native_lineout = np.asarray(profiles[variable_name]["native_lineout"], dtype=np.float64)
        line_x = int(profiles[variable_name]["line_x"])
        line_z = int(profiles[variable_name]["line_z"])
        axis.plot(y_coordinate, reference_lineout, color=color, linewidth=2.2, label="Hermès-3")
        axis.plot(y_coordinate, native_lineout, color=color, linewidth=1.9, linestyle="--", label="JAX-DRB")
        axis.axvspan(y_coordinate[0], y_coordinate[min(boundary_band - 1, y_coordinate.size - 1)], color="#adb5bd", alpha=0.14)
        axis.axvspan(y_coordinate[max(y_coordinate.size - boundary_band, 0)], y_coordinate[-1], color="#adb5bd", alpha=0.14)
        style_axis(
            axis,
            title=f"{variable_name} lineout at worst-error x={line_x}, z={line_z}",
            xlabel="parallel index",
            ylabel=variable_name,
            grid="both",
        )
        axis.legend(frameon=False, fontsize=9.6, loc="best")

    diff_axis = axes[1, 1]
    for variable_name, color in variable_order:
        max_abs_profile_y = np.asarray(profiles[variable_name]["max_abs_profile_y"], dtype=np.float64)
        diff_axis.plot(y_coordinate, np.maximum(max_abs_profile_y, 1.0e-16), color=color, linewidth=2.0, label=variable_name)
    diff_axis.axvspan(y_coordinate[0], y_coordinate[min(boundary_band - 1, y_coordinate.size - 1)], color="#adb5bd", alpha=0.14)
    diff_axis.axvspan(y_coordinate[max(y_coordinate.size - boundary_band, 0)], y_coordinate[-1], color="#adb5bd", alpha=0.14)
    style_axis(
        diff_axis,
        title="max |Δ| across x,z at each parallel index",
        xlabel="parallel index",
        ylabel="|JAX-DRB - Hermès-3|",
        yscale="log",
        grid="both",
    )
    diff_axis.legend(frameon=False, fontsize=9.6, loc="best")
    diff_axis.text(
        0.03,
        0.95,
        f"worst field: {worst_field}\nworst max |Δ|: {worst_error:.2e}\nruntime ratio: {runtime_ratio:.2f}x",
        transform=diff_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        bbox={"facecolor": "white", "edgecolor": "#ced4da", "alpha": 0.92},
    )

    figure.suptitle(
        "Neutral mixed one-step rerun audit: centerline profile agreement and boundary-localized error",
        fontsize=13.4,
        fontweight="semibold",
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.90, wspace=0.22, hspace=0.28)
    save_publication_figure(figure, target)
    return target


def _write_neutral_mixed_boundary_campaign_arrays(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "y_coordinate": np.asarray(report["y_coordinate"], dtype=np.float64),
        "time_points": np.asarray(report["time_points"], dtype=np.float64),
    }
    for variable_name, profile_group in report["profiles"].items():
        arrays[f"{variable_name}_reference_lineout"] = np.asarray(
            profile_group["reference_lineout"],
            dtype=np.float64,
        )
        arrays[f"{variable_name}_native_lineout"] = np.asarray(
            profile_group["native_lineout"],
            dtype=np.float64,
        )
        arrays[f"{variable_name}_abs_diff_lineout"] = np.asarray(
            profile_group["abs_diff_lineout"],
            dtype=np.float64,
        )
        arrays[f"{variable_name}_max_abs_profile_y"] = np.asarray(
            profile_group["max_abs_profile_y"],
            dtype=np.float64,
        )
    np.savez(target, **arrays)
    return target


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
