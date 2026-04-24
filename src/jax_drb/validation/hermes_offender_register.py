from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class HermesOffenderRegisterArtifacts:
    report_json_path: Path
    report_plot_png_path: Path


def create_hermes_offender_register_package(
    *,
    output_root: str | Path,
    live_rerun_json: str | Path | None = None,
    comparison_summary_json: str | Path | None = None,
    case_label: str = "hermes_offender_register",
) -> HermesOffenderRegisterArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_hermes_offender_register_report(
        live_rerun_json=_resolve_or_default(
            live_rerun_json,
            "docs/data/hermes_live_rerun_campaign_artifacts/data/hermes_live_rerun_campaign.json",
        ),
        comparison_summary_json=_resolve_or_default(
            comparison_summary_json,
            "docs/data/hermes_comparison_summary_artifacts/data/hermes_comparison_summary.json",
        ),
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_plot_png_path = save_hermes_offender_register_plot(report, images_dir / f"{case_label}.png")
    return HermesOffenderRegisterArtifacts(
        report_json_path=report_json_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_hermes_offender_register_report(
    *,
    live_rerun_json: str | Path,
    comparison_summary_json: str | Path | None = None,
) -> dict[str, object]:
    live_report = _load_json(live_rerun_json)
    comparison_report = _load_json(comparison_summary_json) if comparison_summary_json is not None else None
    live_cases = list(live_report.get("cases", []))
    parity_offenders = _build_parity_offenders(live_cases, comparison_report=comparison_report)
    runtime_offenders = _build_runtime_offenders(live_cases)
    memory_offenders = _build_memory_offenders(live_cases)
    return {
        "reference_code": "hermes-3",
        "register_scope": "live Hermes rerun matrix plus committed reduced geometry comparison summary",
        "source_artifacts": {
            "live_rerun_json": str(Path(live_rerun_json)),
            "comparison_summary_json": None if comparison_summary_json is None else str(Path(comparison_summary_json)),
        },
        "case_count": len(live_cases),
        "parity_offenders": parity_offenders,
        "runtime_offenders": runtime_offenders,
        "memory_offenders": memory_offenders,
        "top_offenders": {
            "parity": None if not parity_offenders else parity_offenders[0],
            "runtime": None if not runtime_offenders else runtime_offenders[0],
            "memory": None if not memory_offenders else memory_offenders[0],
        },
        "notes": {
            "ranking_policy": (
                "Parity ranks use the largest available relative/scaled error with absolute-error context; "
                "runtime ranks use native/Hermes wall-time ratio; memory ranks use native/Hermes peak-RSS "
                "ratio when measured and otherwise fall back to slow-case memory-risk proxies."
            ),
            "near_zero_policy": (
                "Normalization-sensitive cases are not automatically treated as physical failures; "
                "their absolute max error must be inspected before changing equations."
            ),
            "next_actions": [
                "re-run the top live offender with profiler and component diagnostics",
                "localize the dominant field to a closure, boundary rule, or compare-window convention",
                "add a direct operator test before changing the broad transient runner",
                "record memory peaks for the top slow recycling and neutral cases",
            ],
        },
    }


def save_hermes_offender_register_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parity = list(report.get("parity_offenders", []))[:8]
    runtime = list(report.get("runtime_offenders", []))[:8]
    memory = list(report.get("memory_offenders", []))[:8]
    if not parity and not runtime and not memory:
        figure, axis = plt.subplots(1, 1, figsize=(7.0, 3.8))
        axis.text(0.5, 0.5, "No Hermes offenders available", ha="center", va="center")
        axis.set_axis_off()
        save_publication_figure(figure, target)
        return target

    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))
    _plot_offender_bars(
        axes[0],
        parity,
        value_key="rank_metric",
        label_key="case_name",
        field_key="dominant_field",
        title="Largest live/reference parity offenders",
        ylabel="rank metric",
    )
    _plot_offender_bars(
        axes[1],
        runtime,
        value_key="native_to_reference_runtime_ratio",
        label_key="case_name",
        field_key="dominant_field",
        title="Slowest native/Hermes runtime ratios",
        ylabel="native / Hermès wall time",
        threshold=1.0,
    )
    _plot_offender_bars(
        axes[2],
        memory,
        value_key="rank_metric",
        label_key="case_name",
        field_key="dominant_field",
        title="Largest native/reference memory offenders",
        ylabel="peak RSS ratio or risk proxy",
        threshold=1.0,
    )
    figure.suptitle(
        "Hermès offender register: where to spend the next parity, runtime, and memory effort",
        fontsize=13.0,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.02,
        "Parity bars retain absolute-error context in the JSON report; hatched near-zero cases should be diagnosed before changing equations.",
        ha="center",
        va="center",
        fontsize=8.8,
    )
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.82, wspace=0.92)
    save_publication_figure(figure, target)
    return target


def _build_parity_offenders(
    live_cases: list[dict[str, Any]],
    *,
    comparison_report: dict[str, Any] | None,
) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for case in live_cases:
        rel_to_max = _optional_float(case.get("worst_relative_to_expected_max"))
        rel_l2 = _optional_float(case.get("worst_relative_l2_error"))
        rel_rms = _optional_float(case.get("worst_relative_rms_error"))
        rank_metric = max(value for value in (rel_to_max, rel_l2, rel_rms, 0.0) if value is not None)
        dominant_field = str(case.get("worst_max_abs_field") or case.get("worst_relative_l2_field") or "")
        offenders.append(
            {
                "source": "live_rerun",
                "case_name": str(case.get("case_name", "")),
                "family": str(case.get("family", "")),
                "capability_tier": str(case.get("capability_tier", "")),
                "parity_mode": str(case.get("parity_mode", "")),
                "dominant_field": dominant_field,
                "component_hint": _component_hint(str(case.get("case_name", "")), dominant_field),
                "rank_metric": float(rank_metric),
                "worst_relative_l2_error": float(rel_l2 or 0.0),
                "worst_relative_rms_error": float(rel_rms or 0.0),
                "worst_relative_to_expected_max": None if rel_to_max is None else float(rel_to_max),
                "worst_max_abs_diff": float(case.get("worst_max_abs_diff", 0.0)),
                "normalization_sensitive": bool(case.get("normalization_sensitive", False)),
                "recommended_next_action": _recommended_parity_action(case, dominant_field),
            }
        )
    if comparison_report is not None:
        for lane in comparison_report.get("lanes", []):
            dominant_field = str(lane.get("worst_relative_l2_field") or lane.get("worst_max_abs_field") or "")
            rel_l2 = float(lane.get("worst_relative_l2_error", 0.0))
            max_abs = float(lane.get("worst_max_abs_error", 0.0))
            offenders.append(
                {
                    "source": "committed_geometry_summary",
                    "case_name": str(lane.get("lane_name", "")),
                    "family": str(lane.get("geometry_family", "")),
                    "capability_tier": "reduced_geometry_comparison",
                    "parity_mode": "selected_field",
                    "dominant_field": dominant_field,
                    "component_hint": _component_hint(str(lane.get("lane_name", "")), dominant_field),
                    "rank_metric": rel_l2,
                    "worst_relative_l2_error": rel_l2,
                    "worst_relative_rms_error": 0.0,
                    "worst_relative_to_expected_max": None,
                    "worst_max_abs_diff": max_abs,
                    "normalization_sensitive": False,
                    "recommended_next_action": "compare selected-field geometry construction against the reference adapter inputs",
                }
            )
    offenders.sort(
        key=lambda entry: (
            float(entry["rank_metric"]),
            float(entry["worst_max_abs_diff"]),
        ),
        reverse=True,
    )
    return _with_ranks(offenders)


def _build_runtime_offenders(live_cases: list[dict[str, Any]]) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for case in live_cases:
        ratio = _optional_float(case.get("native_to_reference_runtime_ratio"))
        if ratio is None:
            continue
        dominant_field = str(case.get("worst_max_abs_field") or case.get("worst_relative_l2_field") or "")
        offenders.append(
            {
                "case_name": str(case.get("case_name", "")),
                "family": str(case.get("family", "")),
                "dominant_field": dominant_field,
                "component_hint": _component_hint(str(case.get("case_name", "")), dominant_field),
                "native_elapsed_seconds": float(case.get("native_elapsed_seconds", 0.0)),
                "reference_elapsed_seconds": float(case.get("reference_elapsed_seconds", 0.0)),
                "native_to_reference_runtime_ratio": float(ratio),
                "runtime_status": "native_slower" if ratio > 1.0 else "native_faster",
                "recommended_next_action": _recommended_runtime_action(str(case.get("case_name", ""))),
            }
        )
    offenders.sort(key=lambda entry: float(entry["native_to_reference_runtime_ratio"]), reverse=True)
    return _with_ranks(offenders)


def _build_memory_offenders(live_cases: list[dict[str, Any]]) -> list[dict[str, object]]:
    measured: list[dict[str, object]] = []
    for case in live_cases:
        native_peak = _optional_float(case.get("native_peak_rss_bytes"))
        reference_peak = _optional_float(case.get("reference_peak_rss_bytes"))
        native_delta = _optional_float(case.get("native_peak_rss_delta_bytes"))
        reference_delta = _optional_float(case.get("reference_peak_rss_delta_bytes"))
        if native_peak is None and reference_peak is None and native_delta is None:
            continue
        case_name = str(case.get("case_name", ""))
        ratio = _optional_float(case.get("native_to_reference_peak_rss_ratio"))
        delta_ratio = _optional_float(case.get("native_to_reference_peak_rss_delta_ratio"))
        rank_metric = ratio
        if rank_metric is None:
            rank_metric = native_peak
        if rank_metric is None:
            rank_metric = native_delta
        measured.append(
            {
                "case_name": case_name,
                "family": str(case.get("family", "")),
                "memory_measurement_status": str(case.get("native_memory_measurement_status", "measured")),
                "memory_risk": _memory_risk(case_name),
                "native_peak_rss_mebibytes": _optional_float(case.get("native_peak_rss_mebibytes")),
                "reference_peak_rss_mebibytes": _optional_float(case.get("reference_peak_rss_mebibytes")),
                "native_peak_rss_delta_mebibytes": _optional_float(case.get("native_peak_rss_delta_mebibytes")),
                "reference_peak_rss_delta_mebibytes": _optional_float(case.get("reference_peak_rss_delta_mebibytes")),
                "native_to_reference_peak_rss_ratio": ratio,
                "native_to_reference_peak_rss_delta_ratio": delta_ratio,
                "rank_metric": float(rank_metric or 0.0),
                "recommended_next_action": _recommended_memory_action(case_name, measured=True),
            }
        )
    if measured:
        measured.sort(key=lambda entry: float(entry["rank_metric"]), reverse=True)
        return _with_ranks(measured)

    slow_cases = _build_runtime_offenders(live_cases)[:5]
    offenders: list[dict[str, object]] = []
    for case in slow_cases:
        case_name = str(case["case_name"])
        offenders.append(
            {
                "case_name": case_name,
                "family": str(case["family"]),
                "memory_measurement_status": "not_measured_in_live_register",
                "memory_risk": _memory_risk(case_name),
                "recommended_next_action": _recommended_memory_action(case_name, measured=False),
                "linked_runtime_ratio": float(case["native_to_reference_runtime_ratio"]),
                "rank_metric": float(case["native_to_reference_runtime_ratio"]),
            }
        )
    return _with_ranks(offenders)


def _plot_offender_bars(
    axis,
    entries: list[dict[str, object]],
    *,
    value_key: str,
    label_key: str,
    field_key: str,
    title: str,
    ylabel: str,
    threshold: float | None = None,
) -> None:
    if not entries:
        axis.text(0.5, 0.5, "No entries", ha="center", va="center")
        axis.set_axis_off()
        return
    labels = [_offender_axis_label(entry, label_key=label_key, field_key=field_key) for entry in entries]
    values = np.asarray([max(float(entry[value_key]), 1.0e-16) for entry in entries], dtype=np.float64)
    y = np.arange(len(entries), dtype=np.float64)
    colors = ["#9b2226" if bool(entry.get("normalization_sensitive", False)) else "#005f73" for entry in entries]
    bars = axis.barh(y, values, color=colors, height=0.66)
    for bar, entry in zip(bars, entries, strict=True):
        if bool(entry.get("normalization_sensitive", False)):
            bar.set_hatch("//")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.7)
    if threshold is not None:
        axis.axvline(float(threshold), color="#6c757d", linestyle="--", linewidth=1.0)
    style_axis(axis, title=title, xlabel=ylabel, xscale="log", grid="x")
    axis.set_xlim(max(float(np.min(values)) * 0.45, 1.0e-16), float(np.max(values)) * 7.0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.tick_params(axis="y", labelsize=8.4)
    for yi, value in zip(y, values, strict=True):
        axis.text(float(value * 1.18), float(yi), f"{value:.2e}", ha="left", va="center", fontsize=8.0)


def _offender_axis_label(entry: dict[str, object], *, label_key: str, field_key: str) -> str:
    case_label = str(entry[label_key]).replace("_", " ")
    field_label = str(entry.get(field_key, ""))
    text = f"{case_label} [{field_label}]" if field_label else case_label
    return "\n".join(textwrap.wrap(text, width=28))


def _component_hint(case_name: str, field_name: str) -> str:
    case = case_name.lower()
    field = field_name.lower()
    if "neutral_mixed" in case or field in {"nvh", "nh", "ph"}:
        return "neutral mixed boundary and parallel momentum closure"
    if "recycling" in case and field.startswith("nv"):
        return "parallel momentum, recycling source, or near-zero compare normalization"
    if field.startswith("p") or field == "pe":
        return "pressure, conduction, heat-exchange, or sheath energy closure"
    if "ne" in field and "openadas" in case:
        return "OpenADAS rate/radiation table and source partition"
    if field in {"g33", "g11", "toroidal_flux", "iota"}:
        return "geometry metric or selected-field adapter"
    if "em" in case or "alfven" in case:
        return "electromagnetic selected-field reconstruction"
    return "case-level diagnostics needed"


def _recommended_parity_action(case: dict[str, Any], field_name: str) -> str:
    if bool(case.get("normalization_sensitive", False)):
        return "inspect absolute error and near-zero support before changing equations"
    return f"localize {field_name or 'dominant field'} mismatch to component-level source, boundary, or closure"


def _recommended_runtime_action(case_name: str) -> str:
    case = case_name.lower()
    if "recycling" in case:
        return "profile sparse Jacobian assembly, residual calls, pack/unpack, target recycling, and closure assembly"
    if "neutral_mixed" in case:
        return "profile neutral boundary reconstruction and implicit neutral residual assembly"
    return "profile curated native runner path and compare artifact extraction"


def _memory_risk(case_name: str) -> str:
    case = case_name.lower()
    if "recycling" in case:
        return "materialized sparse Jacobians, colored finite-difference states, full-field copies, and long histories"
    if "neutral_mixed" in case:
        return "implicit residual temporaries and boundary/gradient reconstruction arrays"
    return "history arrays, reference dumps, and artifact extraction temporaries"


def _recommended_memory_action(case_name: str, *, measured: bool) -> str:
    case = case_name.lower()
    if not measured:
        return (
            "profile with process-tree RSS sampling plus JAX device-memory profile "
            "where the lane is JAX-visible"
        )
    if "recycling" in case:
        return "split RSS peak into Jacobian assembly, residual evaluation, packing, and artifact extraction phases"
    if "neutral_mixed" in case:
        return "split RSS peak into boundary reconstruction, residual assembly, and dataset extraction phases"
    return "compare native and reference RSS peaks with artifact extraction disabled before optimizing"


def _with_ranks(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for index, entry in enumerate(entries, start=1):
        ranked.append({"rank": index, **entry})
    return ranked


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_or_default(path: str | Path | None, default_relative: str) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[3] / default_relative


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
