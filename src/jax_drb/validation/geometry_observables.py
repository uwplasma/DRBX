from __future__ import annotations

from pathlib import Path
import json


def build_geometry_observable_report(
    *,
    geometry_family: str,
    benchmark_adapter: str,
    observable_groups: tuple[dict[str, object], ...],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "available": True,
        "parse_status": "ok",
        "geometry_family": geometry_family,
        "benchmark_adapter": benchmark_adapter,
        "observable_groups": list(observable_groups),
        "metadata": dict(metadata or {}),
    }


def write_geometry_observable_report(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return target


def profile_group_from_report(report: dict[str, object], *, name: str, description: str) -> dict[str, object]:
    diagnostics = report.get("diagnostics", {})
    families = []
    if isinstance(diagnostics, dict):
        for diagnostic_name, fields in diagnostics.items():
            if not isinstance(fields, dict) or not fields:
                continue
            first_field = next(iter(fields.values()))
            coordinate_name = first_field.get("coordinate_name", "coord") if isinstance(first_field, dict) else "coord"
            families.append(
                {
                    "name": diagnostic_name,
                    "kind": "profile",
                    "coordinate_name": coordinate_name,
                    "field_names": sorted(fields.keys()),
                }
            )
    return {"name": name, "description": description, "families": families}


def line_group_from_report(report: dict[str, object], *, name: str, description: str) -> dict[str, object]:
    diagnostics = report.get("diagnostics", {})
    families = []
    if isinstance(diagnostics, dict):
        for diagnostic_name, fields in diagnostics.items():
            if not isinstance(fields, dict) or not fields:
                continue
            first_field = next(iter(fields.values()))
            coordinate_name = first_field.get("coordinate_name", "coord") if isinstance(first_field, dict) else "coord"
            families.append(
                {
                    "name": diagnostic_name,
                    "kind": "lineout",
                    "coordinate_name": coordinate_name,
                    "field_names": sorted(fields.keys()),
                }
            )
    return {"name": name, "description": description, "families": families}


def slice_group_from_report(report: dict[str, object], *, name: str, description: str) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "families": [
            {
                "name": str(report.get("slice_name", "slice_planes")),
                "kind": "slice",
                "coordinate_name": str(report.get("coordinate_name", "coord")),
                "field_names": [str(report.get("field_name", "field"))],
            }
        ],
    }
