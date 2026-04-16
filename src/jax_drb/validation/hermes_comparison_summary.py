from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


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
    labels = [str(entry["lane_name"]) for entry in lanes]
    rel_l2 = [float(entry["worst_relative_l2_error"]) for entry in lanes]
    max_abs = [float(entry["worst_max_abs_error"]) for entry in lanes]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    axis.bar(x - width / 2.0, rel_l2, width=width, color="#005f73", label="worst rel L2")
    axis.bar(x + width / 2.0, max_abs, width=width, color="#ca6702", label="worst max|Δ|")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylabel("error metric")
    axis.set_title("Committed native-vs-reference comparison summary")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(frameon=False)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(target, dpi=180)
    plt.close(figure)
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
