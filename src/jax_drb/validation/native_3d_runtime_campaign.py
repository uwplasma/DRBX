from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np

from .stellarator_vmec_native_selected_field import compare_native_stellarator_vmec_selected_fields
from .traced_field_line_native_selected_field import compare_native_traced_field_line_selected_fields


@dataclass(frozen=True)
class Native3DRuntimeCampaignArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_native_3d_runtime_campaign_package(
    *,
    output_root: str | Path,
    tokamak_one_step_runtime_report: str | Path | None = None,
    tokamak_short_window_runtime_report: str | Path | None = None,
    traced_native_runtime_report: str | Path | None = None,
    stellarator_native_runtime_report: str | Path | None = None,
    case_label: str = "native_3d_runtime_campaign",
) -> Native3DRuntimeCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_native_3d_runtime_campaign_report(
        tokamak_one_step_runtime_report=_resolve_or_default(
            tokamak_one_step_runtime_report,
            "docs/data/tokamak_native_selected_field_artifacts/data/tokamak_native_selected_field_runtime_report.json",
        ),
        tokamak_short_window_runtime_report=_resolve_or_default(
            tokamak_short_window_runtime_report,
            "docs/data/tokamak_native_selected_field_short_window_artifacts/data/tokamak_native_selected_field_short_window_runtime_report.json",
        ),
        traced_native_runtime_report=_resolve_or_default(
            traced_native_runtime_report,
            "docs/data/traced_field_line_native_selected_field_artifacts/data/traced_field_line_native_selected_field_runtime_report.json",
        ),
        stellarator_native_runtime_report=_resolve_or_default(
            stellarator_native_runtime_report,
            "docs/data/stellarator_vmec_native_selected_field_artifacts/data/stellarator_vmec_native_selected_field_runtime_report.json",
        ),
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_native_3d_runtime_campaign_plot(report, images_dir / f"{case_label}.png")
    return Native3DRuntimeCampaignArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_native_3d_runtime_campaign_report(
    *,
    tokamak_one_step_runtime_report: str | Path,
    tokamak_short_window_runtime_report: str | Path,
    traced_native_runtime_report: str | Path,
    stellarator_native_runtime_report: str | Path,
) -> dict[str, object]:
    tokamak_one_step = _load_json(tokamak_one_step_runtime_report)
    tokamak_short_window = _load_json(tokamak_short_window_runtime_report)
    traced_native = _load_json(traced_native_runtime_report)
    stellarator_native = _load_json(stellarator_native_runtime_report)

    traced_scaling = _benchmark_traced_field_line_scaling()
    stellarator_scaling = _benchmark_stellarator_vmec_scaling()
    return {
        "case": "native_3d_runtime_campaign",
        "native_lane_runtimes": [
            _runtime_entry("tokamak_native_one_step", tokamak_one_step),
            _runtime_entry("tokamak_native_short_window", tokamak_short_window),
            _runtime_entry("traced_field_line_native_selected_field", traced_native),
            _runtime_entry("stellarator_vmec_native_selected_field", stellarator_native),
        ],
        "scaling_sweeps": [
            traced_scaling,
            stellarator_scaling,
        ],
    }


def save_native_3d_runtime_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtimes = list(report["native_lane_runtimes"])
    sweeps = list(report["scaling_sweeps"])
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)

    labels = [str(entry["lane_name"]) for entry in runtimes]
    values = [float(entry["elapsed_seconds"]) for entry in runtimes]
    colors = ["#005f73", "#0a9396", "#ca6702", "#6a4c93"]
    x = np.arange(len(labels))
    axes[0].bar(x, values, color=colors[: len(labels)])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("elapsed seconds")
    axes[0].set_title("Committed native 3D rung runtimes")
    axes[0].grid(alpha=0.25, axis="y")

    for sweep, color in zip(sweeps, ("#bb3e03", "#3a86ff"), strict=False):
        sizes = np.asarray(sweep["problem_sizes"], dtype=np.float64)
        elapsed = np.asarray(sweep["elapsed_seconds"], dtype=np.float64)
        axes[1].plot(sizes, elapsed, marker="o", linewidth=2.0, color=color, label=str(sweep["lane_name"]))
    axes[1].set_xlabel("synthetic problem size")
    axes[1].set_ylabel("elapsed seconds")
    axes[1].set_title("Native non-tokamak reduction scaling")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _runtime_entry(name: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "lane_name": name,
        "geometry_family": payload.get("geometry_family"),
        "selected_fields": list(payload.get("selected_fields", [])),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        "capability_tier": payload.get("capability_tier", payload.get("native_capability_tier")),
    }


def _benchmark_traced_field_line_scaling() -> dict[str, object]:
    factors = (1, 4, 16)
    elapsed: list[float] = []
    for factor in factors:
        with tempfile.TemporaryDirectory(prefix="jax_drb_traced_scaling_") as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.fci.nc"
            candidate = root / "candidate.fci.nc"
            _write_metric_grid(reference, nx=4 * factor, ny=3, nz=2, offset=0.0)
            _write_metric_grid(candidate, nx=4 * factor, ny=3, nz=2, offset=0.25)
            compare_native_traced_field_line_selected_fields(
                reference_mesh_spec=reference,
                candidate_mesh_spec=candidate,
            )
            timer_start = perf_counter()
            compare_native_traced_field_line_selected_fields(
                reference_mesh_spec=reference,
                candidate_mesh_spec=candidate,
            )
            elapsed.append(perf_counter() - timer_start)
    return {
        "lane_name": "traced_field_line_native_selected_field",
        "problem_sizes": [4 * factor * 3 * 2 for factor in factors],
        "elapsed_seconds": elapsed,
    }


def _benchmark_stellarator_vmec_scaling() -> dict[str, object]:
    factors = (1, 4, 16)
    elapsed: list[float] = []
    for factor in factors:
        with tempfile.TemporaryDirectory(prefix="jax_drb_stellarator_scaling_") as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.nc"
            candidate = root / "candidate.nc"
            _write_vmec_case(reference, ns=6 * factor, scale=1.0)
            _write_vmec_case(candidate, ns=6 * factor, scale=1.1)
            compare_native_stellarator_vmec_selected_fields(
                reference_equilibrium_path=reference,
                candidate_equilibrium_path=candidate,
            )
            timer_start = perf_counter()
            compare_native_stellarator_vmec_selected_fields(
                reference_equilibrium_path=reference,
                candidate_equilibrium_path=candidate,
            )
            elapsed.append(perf_counter() - timer_start)
    return {
        "lane_name": "stellarator_vmec_native_selected_field",
        "problem_sizes": [6 * factor for factor in factors],
        "elapsed_seconds": elapsed,
    }


def _write_metric_grid(path: Path, *, nx: int, ny: int, nz: int, offset: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", nx)
        dataset.createDimension("y", ny)
        dataset.createDimension("z", nz)
        for name, scale in (("g11", 2.0), ("g33", 3.0)):
            variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
            values = np.arange(nx * ny * nz, dtype=np.float64).reshape(nx, ny, nz)
            variable[:] = scale + values + offset


def _write_vmec_case(path: Path, *, ns: int, scale: float) -> None:
    mn_mode = 2
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ns", ns)
        dataset.createDimension("mn_mode", mn_mode)
        dataset.createVariable("iotaf", "f8", ("ns",))[:] = scale * np.linspace(0.35, 0.55, ns)
        dataset.createVariable("presf", "f8", ("ns",))[:] = scale * np.linspace(1000.0, 10.0, ns)
        dataset.createVariable("phi", "f8", ("ns",))[:] = scale * np.linspace(0.0, 2.0, ns)
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = np.asarray([0.0, 1.0])
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = np.asarray([0.0, 0.0])
        rmnc = dataset.createVariable("rmnc", "f8", ("ns", "mn_mode"))
        zmns = dataset.createVariable("zmns", "f8", ("ns", "mn_mode"))
        rmnc[:] = np.column_stack((np.full(ns, 4.1), np.linspace(0.05, 0.4, ns)))
        zmns[:] = np.column_stack((np.zeros(ns), np.linspace(0.08, 0.55, ns)))
        dataset.createVariable("nfp", "i4")[:] = 4


def _resolve_or_default(path: str | Path | None, default_relative: str) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[3] / default_relative


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
