from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset

from .geometry_adapter import build_geometry_adapter_contract, build_geometry_adapter_manifest


@dataclass(frozen=True)
class TracedFieldLineScaffoldArtifacts:
    manifest_json_path: Path
    input_report_json_path: Path
    validation_contract_json_path: Path
    metric_report_json_path: Path
    metric_arrays_npz_path: Path
    metric_plot_png_path: Path


@dataclass(frozen=True)
class TracedFieldLineMeshSource:
    payload: dict[str, object]
    source_format: str


def create_traced_field_line_scaffold_package(
    *,
    output_root: str | Path,
    case_label: str = "traced_field_line_scaffold",
    mesh_spec_path: str | Path | None = None,
) -> TracedFieldLineScaffoldArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_mesh_spec = Path(mesh_spec_path) if mesh_spec_path is not None else None
    preview_mode = resolved_mesh_spec is None
    with tempfile.TemporaryDirectory(prefix="jax_drb_traced_field_line_") as temp_dir:
        temp_root = Path(temp_dir)
        spec_path = resolved_mesh_spec
        if spec_path is None:
            spec_path = temp_root / "synthetic_traced_field_line_mesh.json"
            _write_synthetic_mesh_spec(spec_path)
        mesh_source = _load_traced_field_line_source(spec_path)
        mesh_spec = mesh_source.payload

    input_report = _build_input_report(
        mesh_spec=mesh_spec,
        preview_mode=preview_mode,
        source_format=mesh_source.source_format,
    )
    input_report_json_path = data_dir / f"{case_label}_input_report.json"
    input_report_json_path.write_text(json.dumps(input_report, indent=2, sort_keys=True), encoding="utf-8")

    validation_contract = _build_validation_contract()
    validation_contract_json_path = data_dir / f"{case_label}_validation_contract.json"
    validation_contract_json_path.write_text(
        json.dumps(validation_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metric_report, metric_arrays = _build_metric_report(mesh_spec, source_format=mesh_source.source_format)
    metric_report_json_path = data_dir / f"{case_label}_metric_report.json"
    metric_report_json_path.write_text(json.dumps(metric_report, indent=2, sort_keys=True), encoding="utf-8")
    metric_arrays_npz_path = data_dir / f"{case_label}_metric_arrays.npz"
    np.savez_compressed(metric_arrays_npz_path, **metric_arrays)
    metric_plot_png_path = _save_metric_summary_plot(metric_report, images_dir / f"{case_label}_metrics.png")

    manifest = build_geometry_adapter_manifest(
        case_label=case_label,
        geometry_family="traced_field_line_3d",
        benchmark_adapter="stellarator_traced_field_line_scaffold",
        preview_mode=preview_mode,
        artifacts={
            "input_report_json": str(input_report_json_path.relative_to(root)),
            "validation_contract_json": str(validation_contract_json_path.relative_to(root)),
            "metric_report_json": str(metric_report_json_path.relative_to(root)),
            "metric_arrays_npz": str(metric_arrays_npz_path.relative_to(root)),
            "metric_plot_png": str(metric_plot_png_path.relative_to(root)),
        },
        metadata={"source_format": mesh_source.source_format},
    )
    manifest_json_path = data_dir / f"{case_label}_manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return TracedFieldLineScaffoldArtifacts(
        manifest_json_path=manifest_json_path,
        input_report_json_path=input_report_json_path,
        validation_contract_json_path=validation_contract_json_path,
        metric_report_json_path=metric_report_json_path,
        metric_arrays_npz_path=metric_arrays_npz_path,
        metric_plot_png_path=metric_plot_png_path,
    )


def _write_synthetic_mesh_spec(path: Path) -> None:
    s = np.linspace(0.0, 1.0, 24)
    theta = np.linspace(-np.pi, np.pi, 32)
    phi = np.linspace(0.0, 2.0 * np.pi, 16)
    ss, tt, pp = np.meshgrid(s, theta, phi, indexing="ij")
    payload = {
        "geometry_name": "synthetic_traced_field_line_preview",
        "coordinate_system": "field_aligned",
        "dimensions": {"ns": int(s.size), "ntheta": int(theta.size), "nphi": int(phi.size)},
        "periodicity": {"poloidal": True, "toroidal": True},
        "profiles": {
            "Bmag": (1.0 + 0.2 * ss + 0.08 * np.cos(tt) + 0.03 * np.sin(pp)).tolist(),
            "jacobian": (0.9 + 0.15 * ss**2 + 0.02 * np.cos(tt - pp)).tolist(),
            "g_11": (0.8 + 0.1 * ss + 0.02 * np.cos(tt)).tolist(),
            "g_22": (1.2 + 0.2 * ss + 0.03 * np.sin(tt)).tolist(),
            "g_33": (1.5 + 0.1 * ss + 0.04 * np.cos(pp)).tolist(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_traced_field_line_source(path: Path) -> TracedFieldLineMeshSource:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return TracedFieldLineMeshSource(
            payload=json.loads(path.read_text(encoding="utf-8")),
            source_format="json_mesh_spec",
        )
    if suffix == ".nc":
        return TracedFieldLineMeshSource(
            payload=_load_netcdf_mesh_spec(path),
            source_format="netcdf_fci_grid",
        )
    raise ValueError(f"Unsupported traced-field-line mesh specification: {path}")


def _load_netcdf_mesh_spec(path: Path) -> dict[str, object]:
    with Dataset(path) as dataset:
        dims = {name: len(value) for name, value in dataset.dimensions.items()}
        profiles: dict[str, object] = {}
        for name in ("Bxy", "J", "g11", "g22", "g33", "g_11", "g_22", "g_33", "dx", "dy", "dz"):
            if name in dataset.variables:
                profiles[name] = np.asarray(dataset.variables[name][:], dtype=np.float64).tolist()
        return {
            "geometry_name": path.stem,
            "coordinate_system": "field_aligned",
            "dimensions": {
                "ns": int(dims.get("x", 0)),
                "ntheta": int(dims.get("z", 0)),
                "nphi": int(dims.get("y", 0)),
            },
            "periodicity": {"poloidal": True, "toroidal": True},
            "profiles": profiles,
        }


def _build_input_report(*, mesh_spec: dict[str, object], preview_mode: bool, source_format: str) -> dict[str, object]:
    dimensions = mesh_spec.get("dimensions", {})
    periodicity = mesh_spec.get("periodicity", {})
    return {
        "available": True,
        "parse_status": "ok",
        "preview_mode": preview_mode,
        "source_format": source_format,
        "geometry_name": mesh_spec.get("geometry_name", "unknown"),
        "geometry_family": "traced_field_line_3d",
        "coordinate_system": mesh_spec.get("coordinate_system", "unknown"),
        "dimensions": dimensions,
        "periodicity": periodicity,
        "declared_metric_fields": sorted(list(mesh_spec.get("profiles", {}).keys())),
    }


def _build_validation_contract() -> dict[str, object]:
    return build_geometry_adapter_contract(
        geometry_family="traced_field_line_3d",
        benchmark_adapter="stellarator_traced_field_line_scaffold",
        diagnostic_layer="geometry_adapter_on_general_3d_geometry",
        references=[
            {
                "label": "Zoidberg traced-field-line metrics branch",
                "url": "https://github.com/boutproject/zoidberg/tree/better-metric",
            },
            {
                "label": "Zoidberg metric pull request discussion",
                "url": "https://github.com/boutproject/zoidberg/pull/62",
            },
            {
                "label": "BSTING mesh/script bundle search",
                "url": "https://github.com/search?q=bsting_files&type=repositories",
            },
        ],
        promotion_gates=[
            "scaffold_metric_bundle",
            "external_metric_workdir_bundle",
            "selected_field_parity_bundle",
            "native_execution_bundle",
        ],
        metadata={
            "metric_checks": [
                "positive_jacobian",
                "finite_metric_tensors",
                "periodic_toroidal_indexing",
                "field_line_coordinate_metadata",
            ],
        },
    )


def _build_metric_report(
    mesh_spec: dict[str, object],
    *,
    source_format: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    profiles = mesh_spec.get("profiles", {})
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in profiles.items()}
    report = {
        "available": True,
        "parse_status": "ok",
        "source_format": source_format,
        "geometry_name": mesh_spec.get("geometry_name", "unknown"),
        "coordinate_system": mesh_spec.get("coordinate_system", "unknown"),
        "dimensions": mesh_spec.get("dimensions", {}),
        "metric_fields": {},
    }
    for name, values in arrays.items():
        report["metric_fields"][name] = {
            "shape": list(values.shape),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "finite": bool(np.isfinite(values).all()),
        }
    return report, arrays


def _save_metric_summary_plot(metric_report: dict[str, object], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = metric_report.get("metric_fields", {})
    names = list(fields.keys())
    means = [fields[name]["mean"] for name in names]
    mins = [fields[name]["minimum"] for name in names]
    maxs = [fields[name]["maximum"] for name in names]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0].bar(names, means, color="#005f73")
    axes[0].set_ylabel("Mean value")
    axes[0].set_title("Traced-field-line metric summary")
    axes[0].grid(alpha=0.25, axis="y")
    x = np.arange(len(names))
    axes[1].plot(x, mins, marker="o", color="#ae2012", label="min")
    axes[1].plot(x, maxs, marker="o", color="#0a9396", label="max")
    axes[1].set_xticks(x, names, rotation=20)
    axes[1].set_ylabel("Field range")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
