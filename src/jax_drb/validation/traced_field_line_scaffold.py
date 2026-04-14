from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset

from .geometry_adapter import build_geometry_adapter_contract, build_geometry_adapter_manifest
from .geometry_lineouts import LineoutSpec, build_lineout_report, save_lineout_summary_plot, write_lineout_arrays_npz
from .geometry_slices import SliceSpec, build_slice_report, save_slice_gif, save_slice_summary_plot, write_slice_arrays_npz, write_slice_report_json


@dataclass(frozen=True)
class TracedFieldLineScaffoldArtifacts:
    manifest_json_path: Path
    input_report_json_path: Path
    validation_contract_json_path: Path
    metric_report_json_path: Path
    metric_arrays_npz_path: Path
    metric_plot_png_path: Path
    line_report_json_path: Path
    line_arrays_npz_path: Path
    line_plot_png_path: Path
    slice_report_json_path: Path
    slice_arrays_npz_path: Path
    slice_plot_png_path: Path
    slice_gif_path: Path


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
    line_report = _build_line_report(metric_arrays)
    line_report_json_path = data_dir / f"{case_label}_line_report.json"
    line_report_json_path.write_text(json.dumps(line_report, indent=2, sort_keys=True), encoding="utf-8")
    line_arrays_npz_path = write_lineout_arrays_npz(line_report, data_dir / f"{case_label}_line_arrays.npz")
    preferred_line_fields = _select_line_fields(metric_arrays)
    line_plot_png_path = save_lineout_summary_plot(
        line_report,
        images_dir / f"{case_label}_lineouts.png",
        field_names=preferred_line_fields,
        title="Traced-field-line line diagnostics",
    )
    movie_field_name, movie_field_values, slice_spec = _select_slice_diagnostic(metric_arrays)
    slice_report = build_slice_report(field_name=movie_field_name, values=movie_field_values, spec=slice_spec)
    slice_report_json_path = write_slice_report_json(
        slice_report,
        data_dir / f"{case_label}_slice_report.json",
    )
    slice_arrays_npz_path = write_slice_arrays_npz(
        field_name=movie_field_name,
        values=movie_field_values,
        spec=slice_spec,
        path=data_dir / f"{case_label}_slice_arrays.npz",
    )
    slice_plot_png_path = save_slice_summary_plot(
        slice_report,
        images_dir / f"{case_label}_slice_summary.png",
        title=f"{movie_field_name} {_slice_display_title(slice_spec.name)}",
    )
    slice_gif_path = save_slice_gif(
        field_name=movie_field_name,
        values=movie_field_values,
        spec=slice_spec,
        path=images_dir / f"{case_label}_slice_movie.gif",
    )

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
            "line_report_json": str(line_report_json_path.relative_to(root)),
            "line_arrays_npz": str(line_arrays_npz_path.relative_to(root)),
            "line_plot_png": str(line_plot_png_path.relative_to(root)),
            "slice_report_json": str(slice_report_json_path.relative_to(root)),
            "slice_arrays_npz": str(slice_arrays_npz_path.relative_to(root)),
            "slice_plot_png": str(slice_plot_png_path.relative_to(root)),
            "slice_gif": str(slice_gif_path.relative_to(root)),
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
        line_report_json_path=line_report_json_path,
        line_arrays_npz_path=line_arrays_npz_path,
        line_plot_png_path=line_plot_png_path,
        slice_report_json_path=slice_report_json_path,
        slice_arrays_npz_path=slice_arrays_npz_path,
        slice_plot_png_path=slice_plot_png_path,
        slice_gif_path=slice_gif_path,
    )


def _write_synthetic_mesh_spec(path: Path) -> None:
    s = np.linspace(0.0, 1.0, 24)
    phi = np.linspace(0.0, 2.0 * np.pi, 16)
    theta = np.linspace(-np.pi, np.pi, 32)
    ss, pp, tt = np.meshgrid(s, phi, theta, indexing="ij")
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


def _build_line_report(metric_arrays: dict[str, np.ndarray]) -> dict[str, object]:
    sample_field = next(iter(metric_arrays.values()))
    shape = np.asarray(sample_field).shape
    coords = {
        0: np.linspace(0.0, 1.0, shape[0], dtype=np.float64),
        1: np.linspace(0.0, 1.0, shape[1], dtype=np.float64),
        2: np.linspace(0.0, 1.0, shape[2], dtype=np.float64),
    }
    center0 = shape[0] // 2
    center1 = shape[1] // 2
    center2 = shape[2] // 2
    specs = (
        LineoutSpec("radial_midplane", axis=0, coordinate_name="s", coordinate_values=coords[0], fixed_indices=(center1, center2)),
        LineoutSpec("toroidal_cut", axis=1, coordinate_name="phi_index", coordinate_values=coords[1], fixed_indices=(center0, center2)),
        LineoutSpec("poloidal_cut", axis=2, coordinate_name="theta_index", coordinate_values=coords[2], fixed_indices=(center0, center1)),
    )
    return build_lineout_report(fields=metric_arrays, specs=specs)


def _select_slice_diagnostic(metric_arrays: dict[str, np.ndarray]) -> tuple[str, np.ndarray, SliceSpec]:
    sample_field = next(iter(metric_arrays.values()))
    sample_shape = np.asarray(sample_field).shape
    slice_specs = (
        SliceSpec(
            name="radial_index_planes",
            axis=0,
            coordinate_name="s_index",
            coordinate_values=np.linspace(0.0, 1.0, sample_shape[0], dtype=np.float64),
        ),
        SliceSpec(
            name="toroidal_index_planes",
            axis=1,
            coordinate_name="phi_index",
            coordinate_values=np.linspace(0.0, 1.0, sample_shape[1], dtype=np.float64),
        ),
        SliceSpec(
            name="poloidal_index_planes",
            axis=2,
            coordinate_name="theta_index",
            coordinate_values=np.linspace(0.0, 1.0, sample_shape[2], dtype=np.float64),
        ),
    )
    preferred_names = ("J", "jacobian", "Bmag", "Bxy", "g33", "g11", "g22", "g_33", "g_11", "g_22")
    best_choice: tuple[float, int, int, str, SliceSpec] | None = None
    for preferred_order, name in enumerate(preferred_names):
        if name not in metric_arrays:
            continue
        values = np.asarray(metric_arrays[name], dtype=np.float64)
        for spec_order, spec in enumerate(slice_specs):
            score = _slice_variation_score(values, axis=spec.axis)
            choice = (score, -preferred_order, -spec_order, name, spec)
            if best_choice is None or choice > best_choice:
                best_choice = choice
    if best_choice is not None:
        _, _, _, field_name, spec = best_choice
        return field_name, metric_arrays[field_name], spec
    fallback_name, fallback_values = next(iter(metric_arrays.items()))
    fallback_spec = slice_specs[0]
    return fallback_name, fallback_values, fallback_spec


def _select_line_fields(metric_arrays: dict[str, np.ndarray], *, max_fields: int = 4) -> tuple[str, ...]:
    ranked = _rank_metric_fields(metric_arrays)
    if not ranked:
        return tuple(metric_arrays.keys())[:max_fields]
    return tuple(ranked[:max_fields])


def _rank_metric_fields(metric_arrays: dict[str, np.ndarray]) -> list[str]:
    preferred_names = ("J", "jacobian", "Bmag", "Bxy", "g33", "g11", "g22")
    scores: list[tuple[int, float, str]] = []
    for order, name in enumerate(preferred_names):
        if name not in metric_arrays:
            continue
        values = metric_arrays[name]
        array = np.asarray(values, dtype=np.float64)
        span = float(np.max(array) - np.min(array))
        scale = max(abs(float(np.mean(array))), 1.0e-12)
        relative_span = span / scale
        if relative_span <= 1.0e-12:
            continue
        scores.append((order, -relative_span, name))
    scores.sort()
    return [name for _, _, name in scores]


def _slice_variation_score(values: np.ndarray, *, axis: int) -> float:
    if values.ndim != 3 or values.shape[axis] <= 1:
        return 0.0
    reference = np.take(values, indices=0, axis=axis)
    deltas = []
    for index in range(values.shape[axis]):
        plane = np.take(values, indices=index, axis=axis)
        delta = plane - reference
        deltas.append(float(np.sqrt(np.mean(delta**2))))
    scale = max(float(np.sqrt(np.mean(reference**2))), 1.0e-12)
    return max(deltas) / scale


def _slice_display_title(name: str) -> str:
    titles = {
        "radial_index_planes": "radial slice summary",
        "toroidal_index_planes": "toroidal slice summary",
        "poloidal_index_planes": "poloidal slice summary",
    }
    return titles.get(name, name.replace("_", " "))


def _save_metric_summary_plot(metric_report: dict[str, object], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = metric_report.get("metric_fields", {})
    names = list(fields.keys())
    labels = [_metric_display_label(name) for name in names]
    means = [fields[name]["mean"] for name in names]
    mins = [fields[name]["minimum"] for name in names]
    maxs = [fields[name]["maximum"] for name in names]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0].bar(labels, means, color="#005f73")
    axes[0].set_ylabel("Mean value")
    axes[0].set_title("Traced-field-line metric summary")
    axes[0].grid(alpha=0.25, axis="y")
    x = np.arange(len(names))
    axes[1].plot(x, mins, marker="o", color="#ae2012", label="min")
    axes[1].plot(x, maxs, marker="o", color="#0a9396", label="max")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("Field range")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _metric_display_label(name: str) -> str:
    labels = {
        "Bxy": "B",
        "Bmag": "|B|",
        "J": "Jacobian",
        "jacobian": "Jacobian",
        "g11": "g11",
        "g22": "g22",
        "g33": "g33",
        "g_11": "g^11",
        "g_22": "g^22",
        "g_33": "g^33",
        "dx": "dx",
        "dy": "dy",
        "dz": "dz",
    }
    return labels.get(name, name)
