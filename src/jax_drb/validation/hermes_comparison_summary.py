from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class HermesComparisonSummaryArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_hermes_comparison_summary_package(
    *,
    output_root: str | Path,
    tokamak_one_step_parity_json: str | Path | None = None,
    tokamak_short_window_parity_json: str | Path | None = None,
    traced_native_parity_json: str | Path | None = None,
    stellarator_native_parity_json: str | Path | None = None,
    case_label: str = "hermes_comparison_summary",
) -> HermesComparisonSummaryArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_hermes_comparison_summary_report(
        tokamak_one_step_parity_json=_resolve_or_default(
            tokamak_one_step_parity_json,
            "docs/data/tokamak_native_selected_field_artifacts/data/tokamak_native_selected_field.json",
        ),
        tokamak_short_window_parity_json=_resolve_or_default(
            tokamak_short_window_parity_json,
            "docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window.json",
        ),
        traced_native_parity_json=_resolve_or_default(
            traced_native_parity_json,
            "docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field.json",
        ),
        stellarator_native_parity_json=_resolve_or_default(
            stellarator_native_parity_json,
            "docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field.json",
        ),
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_hermes_comparison_summary_plot(report, images_dir / f"{case_label}.png")
    return HermesComparisonSummaryArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_hermes_comparison_summary_report(
    *,
    tokamak_one_step_parity_json: str | Path,
    tokamak_short_window_parity_json: str | Path,
    traced_native_parity_json: str | Path,
    stellarator_native_parity_json: str | Path,
) -> dict[str, object]:
    lanes = [
        _parity_entry("tokamak_native_one_step", "diverted_tokamak_3d", _load_json(tokamak_one_step_parity_json)),
        _parity_entry("tokamak_native_short_window", "diverted_tokamak_3d", _load_json(tokamak_short_window_parity_json)),
        _parity_entry("traced_field_line_native_selected_field", "traced_field_line_3d", _load_json(traced_native_parity_json)),
        _parity_entry("stellarator_vmec_native_selected_field", "stellarator_vmec_3d", _load_json(stellarator_native_parity_json)),
    ]
    return {
        "reference_code": "hermes-3",
        "lane_count": len(lanes),
        "lanes": lanes,
    }


def save_hermes_comparison_summary_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lanes = list(report["lanes"])
    labels = [
        "tokamak\n1-step",
        "tokamak\nshort window",
        "traced-field-line\nreduced",
        "stellarator VMEC\nreduced",
    ]
    rel_l2 = [float(entry["worst_relative_l2_error"]) for entry in lanes]
    max_abs = [float(entry["worst_max_abs_error"]) for entry in lanes]
    rel_fields = [str(entry["worst_relative_l2_field"]) for entry in lanes]
    abs_fields = [str(entry["worst_max_abs_field"]) for entry in lanes]
    floor = 1.0e-12
    x = np.arange(len(labels), dtype=np.float64)
    rel_plot = np.maximum(np.asarray(rel_l2, dtype=np.float64), floor)
    abs_plot = np.maximum(np.asarray(max_abs, dtype=np.float64), floor)

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

    axes[0].bar(x, rel_plot, color="#005f73", width=0.62)
    style_axis(
        axes[0],
        title="Worst relative L2 error by lane",
        ylabel="relative L2 error",
        yscale="log",
    )
    axes[0].set_xticks(x, labels)
    annotate_bars(axes[0], x, np.asarray(rel_l2, dtype=np.float64), fmt="{:.2e}", fontsize=8.8)
    for xi, field_name in zip(x, rel_fields, strict=True):
        axes[0].text(float(xi), floor * 1.7, field_name, ha="center", va="bottom", fontsize=8.4, color="#33415c")

    axes[1].bar(x, abs_plot, color="#ca6702", width=0.62)
    style_axis(
        axes[1],
        title="Worst max-absolute error by lane",
        ylabel="max |Δ|",
        yscale="log",
    )
    axes[1].set_xticks(x, labels)
    annotate_bars(axes[1], x, np.asarray(max_abs, dtype=np.float64), fmt="{:.2e}", fontsize=8.8)
    for xi, field_name in zip(x, abs_fields, strict=True):
        axes[1].text(float(xi), floor * 1.7, field_name, ha="center", va="bottom", fontsize=8.4, color="#33415c")

    figure.suptitle(
        "Committed reduced native-vs-reference comparison summary",
        fontsize=14.0,
        fontweight="semibold",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.84, wspace=0.28)
    save_publication_figure(figure, target)
    return target


def _parity_entry(lane_name: str, geometry_family: str, payload: dict[str, object]) -> dict[str, object]:
    variable_errors = payload["variable_errors"]
    worst_rel_name, worst_rel_error = max(
        ((name, float(values["relative_l2_error"])) for name, values in variable_errors.items()),
        key=lambda item: item[1],
    )
    worst_abs_name, worst_abs_error = max(
        ((name, float(values["max_abs_error"])) for name, values in variable_errors.items()),
        key=lambda item: item[1],
    )
    return {
        "lane_name": lane_name,
        "geometry_family": geometry_family,
        "worst_relative_l2_field": worst_rel_name,
        "worst_relative_l2_error": worst_rel_error,
        "worst_max_abs_field": worst_abs_name,
        "worst_max_abs_error": worst_abs_error,
    }


def _resolve_or_default(path: str | Path | None, default_relative: str) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[3] / default_relative


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
