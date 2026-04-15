from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry_observables import build_geometry_observable_report, write_geometry_observable_report
from .geometry_selected_field import (
    GeometrySelectedFieldParityResult,
    compare_geometry_selected_fields,
    save_geometry_selected_field_parity_plot,
    write_geometry_selected_field_parity_arrays,
    write_geometry_selected_field_parity_json,
)
from .traced_field_line_scaffold import _load_traced_field_line_source, _write_synthetic_mesh_spec


@dataclass(frozen=True)
class TracedFieldLineSelectedFieldParityArtifacts:
    parity_json_path: Path
    parity_arrays_npz_path: Path
    parity_plot_png_path: Path
    observable_report_json_path: Path


def compare_traced_field_line_selected_fields(
    *,
    reference_mesh_spec: str | Path,
    candidate_mesh_spec: str | Path,
    field_names: tuple[str, ...] = ("J", "g11", "g33"),
) -> GeometrySelectedFieldParityResult:
    reference = _load_metric_fields(reference_mesh_spec)
    candidate = _load_metric_fields(candidate_mesh_spec)
    resolved_field_names = tuple(_resolve_field_alias(reference, candidate, name) for name in field_names)
    return compare_geometry_selected_fields(
        reference_fields=reference,
        candidate_fields=candidate,
        field_names=resolved_field_names,
    )


def create_traced_field_line_selected_field_parity_package(
    *,
    reference_mesh_spec: str | Path | None,
    candidate_mesh_spec: str | Path | None,
    output_root: str | Path,
    case_label: str = "traced_field_line_selected_field_parity",
    field_names: tuple[str, ...] = ("J", "g11", "g33"),
) -> TracedFieldLineSelectedFieldParityArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if reference_mesh_spec is None or candidate_mesh_spec is None:
        with tempfile.TemporaryDirectory(prefix="jax_drb_traced_field_selected_") as temp_dir:
            temp_root = Path(temp_dir)
            reference_path = temp_root / "reference.json"
            candidate_path = temp_root / "candidate.json"
            _write_synthetic_mesh_spec(reference_path)
            _write_synthetic_mesh_spec(candidate_path)
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            profiles = payload["profiles"]
            for field_name, delta in (("J", 0.015), ("g_11", 0.01), ("g_33", -0.02)):
                if field_name in profiles:
                    values = np.asarray(profiles[field_name], dtype=np.float64)
                    profiles[field_name] = (values * (1.0 + delta)).tolist()
            candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            result = compare_traced_field_line_selected_fields(
                reference_mesh_spec=reference_path,
                candidate_mesh_spec=candidate_path,
                field_names=field_names,
            )
    else:
        result = compare_traced_field_line_selected_fields(
            reference_mesh_spec=reference_mesh_spec,
            candidate_mesh_spec=candidate_mesh_spec,
            field_names=field_names,
        )

    parity_json_path = write_geometry_selected_field_parity_json(result, data_dir / f"{case_label}.json")
    parity_arrays_npz_path = write_geometry_selected_field_parity_arrays(result, data_dir / f"{case_label}.npz")
    parity_plot_png_path = save_geometry_selected_field_parity_plot(
        result,
        images_dir / f"{case_label}.png",
        title="Traced-field-line reduced selected-field parity",
    )
    observable_report = build_geometry_observable_report(
        geometry_family="traced_field_line_3d",
        benchmark_adapter="stellarator_traced_field_line_scaffold",
        observable_groups=(
            {
                "name": "selected_metric_parity",
                "description": "Compact selected-field parity surface on traced-field-line metric fields.",
                "families": [
                    {
                        "name": "selected_metric_fields",
                        "kind": "selected_field_parity",
                        "coordinate_name": "full_domain",
                        "field_names": list(result.field_names),
                    }
                ],
            },
        ),
        metadata={"compare_surface": "static_metric_field_bundle"},
    )
    observable_report_json_path = write_geometry_observable_report(
        observable_report,
        data_dir / f"{case_label}_observable_report.json",
    )
    return TracedFieldLineSelectedFieldParityArtifacts(
        parity_json_path=parity_json_path,
        parity_arrays_npz_path=parity_arrays_npz_path,
        parity_plot_png_path=parity_plot_png_path,
        observable_report_json_path=observable_report_json_path,
    )


def _load_metric_fields(path: str | Path) -> dict[str, np.ndarray]:
    source = _load_traced_field_line_source(Path(path))
    profiles = source.payload.get("profiles", {})
    return {name: np.asarray(values, dtype=np.float64) for name, values in profiles.items()}


def _resolve_field_alias(
    reference_fields: dict[str, np.ndarray],
    candidate_fields: dict[str, np.ndarray],
    requested_name: str,
) -> str:
    alias_groups = {
        "J": ("J", "jacobian"),
        "jacobian": ("jacobian", "J"),
        "g11": ("g11", "g_11"),
        "g_11": ("g_11", "g11"),
        "g22": ("g22", "g_22"),
        "g_22": ("g_22", "g22"),
        "g33": ("g33", "g_33"),
        "g_33": ("g_33", "g33"),
    }
    for candidate_name in alias_groups.get(requested_name, (requested_name,)):
        if candidate_name in reference_fields and candidate_name in candidate_fields:
            return candidate_name
    raise KeyError(f"Missing selected field {requested_name!r} in traced-field-line parity comparison.")
