from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PublicationReady3DArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_publication_ready_3d_campaign_package(
    *,
    output_root: str | Path,
    tokamak_one_step_runtime_report: str | Path | None = None,
    tokamak_one_step_parity_json: str | Path | None = None,
    tokamak_short_window_runtime_report: str | Path | None = None,
    tokamak_short_window_parity_json: str | Path | None = None,
    traced_field_line_parity_json: str | Path | None = None,
    traced_field_line_source_report: str | Path | None = None,
    traced_field_line_native_runtime_report: str | Path | None = None,
    traced_field_line_native_parity_json: str | Path | None = None,
    stellarator_parity_json: str | Path | None = None,
    stellarator_source_report: str | Path | None = None,
    stellarator_native_runtime_report: str | Path | None = None,
    stellarator_native_parity_json: str | Path | None = None,
    convergence_report_json: str | Path | None = None,
    case_label: str = "publication_ready_3d_campaign",
) -> PublicationReady3DArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_publication_ready_3d_report(
        tokamak_one_step_runtime_report=_resolve_or_default(
            tokamak_one_step_runtime_report,
            "docs/data/tokamak_native_selected_field_artifacts/data/tokamak_native_selected_field_runtime_report.json",
        ),
        tokamak_one_step_parity_json=_resolve_or_default(
            tokamak_one_step_parity_json,
            "docs/data/tokamak_native_selected_field_artifacts/data/tokamak_native_selected_field.json",
        ),
        tokamak_short_window_runtime_report=_resolve_or_default(
            tokamak_short_window_runtime_report,
            "docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window_runtime_report.json",
        ),
        tokamak_short_window_parity_json=_resolve_or_default(
            tokamak_short_window_parity_json,
            "docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window.json",
        ),
        traced_field_line_parity_json=_resolve_or_default(
            traced_field_line_parity_json,
            "docs/data/traced_field_line_selected_field_artifacts/data/traced_field_line_selected_field_parity.json",
        ),
        traced_field_line_source_report=_resolve_or_default(
            traced_field_line_source_report,
            "docs/data/traced_field_line_selected_field_artifacts/data/traced_field_line_selected_field_parity_source_report.json",
        ),
        traced_field_line_native_runtime_report=_resolve_or_default(
            traced_field_line_native_runtime_report,
            "docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field_runtime_report.json",
        ),
        traced_field_line_native_parity_json=_resolve_or_default(
            traced_field_line_native_parity_json,
            "docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field.json",
        ),
        stellarator_parity_json=_resolve_or_default(
            stellarator_parity_json,
            "docs/data/stellarator_vmec_selected_field_artifacts/data/stellarator_vmec_selected_field_parity.json",
        ),
        stellarator_source_report=_resolve_or_default(
            stellarator_source_report,
            "docs/data/stellarator_vmec_selected_field_artifacts/data/stellarator_vmec_selected_field_parity_source_report.json",
        ),
        stellarator_native_runtime_report=_resolve_or_default(
            stellarator_native_runtime_report,
            "docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field_runtime_report.json",
        ),
        stellarator_native_parity_json=_resolve_or_default(
            stellarator_native_parity_json,
            "docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field.json",
        ),
        convergence_report_json=_resolve_or_default(
            convergence_report_json,
            "docs/data/fluid_1d_mms_convergence.json",
        ),
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_publication_ready_3d_summary_plot(
        report,
        images_dir / f"{case_label}.png",
    )
    return PublicationReady3DArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_publication_ready_3d_report(
    *,
    tokamak_one_step_runtime_report: str | Path,
    tokamak_one_step_parity_json: str | Path,
    tokamak_short_window_runtime_report: str | Path,
    tokamak_short_window_parity_json: str | Path,
    traced_field_line_parity_json: str | Path,
    traced_field_line_source_report: str | Path,
    traced_field_line_native_runtime_report: str | Path,
    traced_field_line_native_parity_json: str | Path,
    stellarator_parity_json: str | Path,
    stellarator_source_report: str | Path,
    stellarator_native_runtime_report: str | Path,
    stellarator_native_parity_json: str | Path,
    convergence_report_json: str | Path,
) -> dict[str, object]:
    tokamak_one_step_runtime = _load_json(tokamak_one_step_runtime_report)
    tokamak_one_step_parity = _load_json(tokamak_one_step_parity_json)
    tokamak_short_window_runtime = _load_json(tokamak_short_window_runtime_report)
    tokamak_short_window_parity = _load_json(tokamak_short_window_parity_json)
    traced_parity = _load_json(traced_field_line_parity_json)
    traced_source = _load_json(traced_field_line_source_report)
    traced_native_runtime = _load_json(traced_field_line_native_runtime_report)
    traced_native_parity = _load_json(traced_field_line_native_parity_json)
    stellarator_parity = _load_json(stellarator_parity_json)
    stellarator_source = _load_json(stellarator_source_report)
    stellarator_native_runtime = _load_json(stellarator_native_runtime_report)
    stellarator_native_parity = _load_json(stellarator_native_parity_json)
    convergence = _load_json(convergence_report_json)

    lane_summaries = [
        _native_lane_summary(
            lane_name="tokamak_native_one_step",
            geometry_family="diverted_tokamak_3d",
            runtime_report=tokamak_one_step_runtime,
            parity_payload=tokamak_one_step_parity,
        ),
        _native_lane_summary(
            lane_name="tokamak_native_short_window",
            geometry_family="diverted_tokamak_3d",
            runtime_report=tokamak_short_window_runtime,
            parity_payload=tokamak_short_window_parity,
        ),
        _external_lane_summary(
            lane_name="traced_field_line_selected_field",
            geometry_family="traced_field_line_3d",
            parity_payload=traced_parity,
            source_report=traced_source,
        ),
        _native_lane_summary(
            lane_name="traced_field_line_native_selected_field",
            geometry_family="traced_field_line_3d",
            runtime_report=traced_native_runtime,
            parity_payload=traced_native_parity,
        ),
        _external_lane_summary(
            lane_name="stellarator_vmec_selected_field",
            geometry_family="stellarator_vmec_3d",
            parity_payload=stellarator_parity,
            source_report=stellarator_source,
        ),
        _native_lane_summary(
            lane_name="stellarator_vmec_native_selected_field",
            geometry_family="stellarator_vmec_3d",
            runtime_report=stellarator_native_runtime,
            parity_payload=stellarator_native_parity,
        ),
    ]
    observed_orders = convergence.get("observed_orders", [])
    density_orders = [float(entry["density_order"]) for entry in observed_orders]
    momentum_orders = [float(entry["momentum_order"]) for entry in observed_orders]
    pressure_orders = [float(entry["pressure_order"]) for entry in observed_orders]
    campaign_status = {
        "native_tokamak_rungs": 2,
        "non_tokamak_external_pair_gates": 2,
        "native_non_tokamak_rungs": 2,
        "remaining_blockers": [
            "expanded_3d_native_convergence_and_scaling_campaign",
        ],
    }
    return {
        "case": "publication_ready_3d_campaign",
        "lane_summaries": lane_summaries,
        "convergence_summary": {
            "case": convergence.get("case"),
            "min_density_order": float(min(density_orders)),
            "min_momentum_order": float(min(momentum_orders)),
            "min_pressure_order": float(min(pressure_orders)),
            "resolutions": list(convergence.get("resolutions", [])),
        },
        "campaign_status": campaign_status,
    }


def save_publication_ready_3d_summary_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lanes = list(report["lane_summaries"])
    labels = [str(entry["lane_name"]) for entry in lanes]
    rel_l2 = [float(entry["worst_relative_l2_error"]) for entry in lanes]
    runtimes = [
        float(entry["elapsed_seconds"]) if entry.get("elapsed_seconds") is not None else np.nan
        for entry in lanes
    ]
    orders = report["convergence_summary"]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    colors = ["#0a9396", "#005f73", "#ca6702", "#bb3e03", "#6a4c93", "#3a86ff"]
    axes[0].bar(x, rel_l2, color=[colors[index % len(colors)] for index in range(len(labels))])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("worst relative L2 error")
    axes[0].set_title("Reduced 3D parity surfaces")
    axes[0].grid(alpha=0.25, axis="y")

    finite_runtime = np.isfinite(runtimes)
    if np.any(finite_runtime):
        x_runtime = np.arange(np.count_nonzero(finite_runtime))
        runtime_labels = [labels[index] for index, value in enumerate(finite_runtime) if value]
        runtime_values = [runtimes[index] for index, value in enumerate(finite_runtime) if value]
        axes[1].bar(x_runtime, runtime_values, color="#3a86ff")
        axes[1].set_xticks(x_runtime, runtime_labels, rotation=15, ha="right")
        axes[1].set_ylabel("elapsed seconds")
        axes[1].grid(alpha=0.25, axis="y")
    else:
        axes[1].text(0.5, 0.5, "No native runtime data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Promoted native runtime surface")

    figure.suptitle(
        "Publication-ready 3D campaign summary\n"
        f"MMS min orders: density={orders['min_density_order']:.2f}, "
        f"momentum={orders['min_momentum_order']:.2f}, "
        f"pressure={orders['min_pressure_order']:.2f}",
        fontsize=12,
    )
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _native_lane_summary(
    *,
    lane_name: str,
    geometry_family: str,
    runtime_report: dict[str, object],
    parity_payload: dict[str, object],
) -> dict[str, object]:
    worst_field, worst_error = _worst_relative_l2(parity_payload)
    return {
        "lane_name": lane_name,
        "geometry_family": geometry_family,
        "lane_kind": "native_selected_field",
        "capability_tier": runtime_report.get("capability_tier", runtime_report.get("native_capability_tier")),
        "selected_fields": list(runtime_report.get("selected_fields", [])),
        "elapsed_seconds": float(runtime_report.get("elapsed_seconds", 0.0)),
        "worst_field": worst_field,
        "worst_relative_l2_error": worst_error,
    }


def _external_lane_summary(
    *,
    lane_name: str,
    geometry_family: str,
    parity_payload: dict[str, object],
    source_report: dict[str, object],
) -> dict[str, object]:
    worst_field, worst_error = _worst_relative_l2(parity_payload)
    return {
        "lane_name": lane_name,
        "geometry_family": geometry_family,
        "lane_kind": "external_selected_field",
        "capability_tier": "external_pair_validation",
        "selected_fields": list(parity_payload.get("field_names", [])),
        "elapsed_seconds": None,
        "worst_field": worst_field,
        "worst_relative_l2_error": worst_error,
        "source_mode": source_report.get("source_mode"),
        "candidate_origin": source_report.get("candidate_origin"),
    }


def _worst_relative_l2(payload: dict[str, object]) -> tuple[str, float]:
    variable_errors = payload.get("variable_errors", {})
    items = [
        (str(name), float(entry.get("relative_l2_error", 0.0)))
        for name, entry in variable_errors.items()
    ]
    if not items:
        return ("(none)", 0.0)
    return max(items, key=lambda item: item[1])


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_or_default(value: str | Path | None, default_relative: str) -> Path:
    if value is not None:
        return Path(value)
    return Path(__file__).resolve().parents[3] / default_relative
