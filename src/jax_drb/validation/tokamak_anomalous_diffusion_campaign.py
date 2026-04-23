from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import apply_bout_overrides, load_bout_input
from ..native.metrics import StructuredMetrics
from ..native.recycling_anomalous_diffusion import apply_anomalous_diffusion
from ..native.recycling_setup import initialize_species
from ..native.reference_dump import (
    load_local_reference_snapshot_cache,
    synthesize_local_reference_snapshot_from_active_history,
)
from ..native.units import resolved_dataset_scalars
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration
from .publication_plotting import (
    annotate_bars,
    save_publication_figure,
    style_axis,
    support_window_slice,
)


_FIELD_NAMES = (
    "Nd+",
    "Pd+",
    "NVd+",
    "Nd",
    "Pd",
    "NVd",
    "Nt+",
    "Pt+",
    "NVt+",
    "Nt",
    "Pt",
    "NVt",
    "Nhe+",
    "Phe+",
    "NVhe+",
    "Nhe",
    "Phe",
    "NVhe",
    "Pe",
)
_SCALAR_NAMES = ("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0")
_SPECIES_ORDER = ("d+", "t+", "he+", "e")
_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "tokamak_recycling_dthe_rhs_snapshot.npz"
)
_ARRAY_HISTORY_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "baselines"
    / "reference_arrays"
    / "tokamak_recycling_dthe_one_step.npz"
)
_OPTIONAL_HISTORY_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "tokamak_recycling_dthe_one_step_optional_history.npz"
)


@dataclass(frozen=True)
class TokamakAnomalousDiffusionCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class TokamakAnomalousDiffusionCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _TokamakAnomalousDiffusionContext:
    input_path: Path
    mesh_path: Path
    snapshot_path: Path
    config: object
    mesh: object
    metrics_nonorthogonal: object
    metrics_orthogonal: object
    dataset_scalars: dict[str, float]
    species: dict[str, object]
    nonorthogonal_terms: object
    orthogonal_terms: object


def create_tokamak_anomalous_diffusion_campaign_package(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    mesh_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    array_history_path: str | Path | None = None,
    optional_history_path: str | Path | None = None,
    case_label: str = "tokamak_anomalous_diffusion_campaign",
) -> TokamakAnomalousDiffusionCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_path, resolved_mesh_path, resolved_snapshot_path, resolved_array_history_path, resolved_optional_history_path = _resolve_paths(
        input_path=input_path,
        mesh_path=mesh_path,
        snapshot_path=snapshot_path,
        array_history_path=array_history_path,
        optional_history_path=optional_history_path,
    )
    context = build_tokamak_anomalous_diffusion_context(
        input_path=resolved_input_path,
        mesh_path=resolved_mesh_path,
        snapshot_path=resolved_snapshot_path,
        array_history_path=resolved_array_history_path,
        optional_history_path=resolved_optional_history_path,
    )
    metrics = build_tokamak_anomalous_diffusion_campaign(
        input_path=resolved_input_path,
        mesh_path=resolved_mesh_path,
        snapshot_path=resolved_snapshot_path,
        array_history_path=resolved_array_history_path,
        optional_history_path=resolved_optional_history_path,
    )
    summaries = _build_tokamak_anomalous_diffusion_summaries(context)

    summary_payload = {
        "family": "tokamak_anomalous_diffusion",
        "input_name": resolved_input_path.name,
        "mesh_name": resolved_mesh_path.name,
        "snapshot_name": resolved_snapshot_path.name,
        "array_history_name": resolved_array_history_path.name,
        "literature_anchor": {
            "direct_tokamak_context": "Dudson et al. 2024, Hermes-3 multi-component plasma simulations",
            "geometry_context": "Bufferand et al. 2016, non-orthogonal tokamak edge/SOL transport on mapped meshes",
        },
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if metric.passed),
        "metrics": [
            {
                "name": metric.name,
                "kind": metric.kind,
                "value": float(metric.value),
                "target": float(metric.target),
                "passed": bool(metric.passed),
                "notes": metric.notes,
            }
            for metric in metrics
        ],
        "species_order": list(summaries["species_order"]),
        "representative_x_index": int(summaries["representative_x_index"]),
        "summaries": {
            name: [float(value) for value in values]
            for name, values in summaries.items()
            if name not in {"species_order", "representative_x_index"}
        },
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in summaries.items()
        if name != "species_order"
    }
    arrays_payload["metric_values"] = np.asarray([metric.value for metric in metrics], dtype=np.float64)
    arrays_payload["metric_targets"] = np.asarray([metric.target for metric in metrics], dtype=np.float64)
    arrays_payload["metric_pass"] = np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64)
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_tokamak_anomalous_diffusion_plot(summaries, plot_png_path)
    return TokamakAnomalousDiffusionCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_tokamak_anomalous_diffusion_campaign(
    *,
    input_path: str | Path,
    mesh_path: str | Path,
    snapshot_path: str | Path,
    array_history_path: str | Path,
    optional_history_path: str | Path,
) -> tuple[TokamakAnomalousDiffusionCampaignMetric, ...]:
    context = build_tokamak_anomalous_diffusion_context(
        input_path=input_path,
        mesh_path=mesh_path,
        snapshot_path=snapshot_path,
        array_history_path=array_history_path,
        optional_history_path=optional_history_path,
    )
    summaries = _build_tokamak_anomalous_diffusion_summaries(context)
    return (
        TokamakAnomalousDiffusionCampaignMetric(
            name="electron_anomalous_d_matches_d_plus",
            kind="max_value",
            value=float(abs(summaries["anomalous_D_active_point"][0] - summaries["anomalous_D_active_point"][3])),
            target=1.0e-12,
            passed=bool(abs(summaries["anomalous_D_active_point"][0] - summaries["anomalous_D_active_point"][3]) <= 1.0e-12),
            notes="The electron anomalous_D coefficient should inherit the d+ literal reference on the direct tokamak recycling deck.",
        ),
        *(
            TokamakAnomalousDiffusionCampaignMetric(
                name=f"{species_name.replace('+', '_plus')}_energy_relative_contrast",
                kind="min_value",
                value=float(value),
                target=5.0e-2,
                passed=bool(value >= 5.0e-2),
                notes=f"Non-orthogonal tokamak metrics should materially modify the anomalous energy transport for {species_name} on the evolved recycling state.",
            )
            for species_name, value in zip(_SPECIES_ORDER[:3], summaries["energy_relative_contrast"][:3], strict=True)
        ),
        TokamakAnomalousDiffusionCampaignMetric(
            name="electron_density_contrast_peak",
            kind="min_value",
            value=float(summaries["density_contrast_peak"][3]),
            target=1.0e-8,
            passed=bool(summaries["density_contrast_peak"][3] >= 1.0e-8),
            notes="Electron anomalous density transport should retain a measurable non-orthogonal contrast on the evolved tokamak state.",
        ),
    )


def build_tokamak_anomalous_diffusion_context(
    *,
    input_path: str | Path,
    mesh_path: str | Path,
    snapshot_path: str | Path,
    array_history_path: str | Path,
    optional_history_path: str | Path,
) -> _TokamakAnomalousDiffusionContext:
    input_path = Path(input_path)
    mesh_path = Path(mesh_path)
    config = apply_bout_overrides(
        load_bout_input(input_path),
        (
            "timestep=0.1",
            f"mesh:file={mesh_path}",
            "he+:diagnose=false",
            "input:error_on_unused_options=false",
        ),
    )
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot_cache(
        snapshot_path,
        field_names=_FIELD_NAMES,
        scalar_names=_SCALAR_NAMES,
    )
    evolved = synthesize_local_reference_snapshot_from_active_history(
        initial_snapshot=snapshot,
        array_history_path=array_history_path,
        optional_history_path=optional_history_path,
        timestep=0.1,
        state_field_names=_FIELD_NAMES,
        optional_field_names=(),
    )
    species = initialize_species(
        config,
        mesh=evolved.mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=evolved.fields,
    )

    netcdf4 = __import__("netCDF4")
    with netcdf4.Dataset(mesh_path) as mesh_dataset:
        g23 = np.asarray(mesh_dataset.variables["g23"][:], dtype=np.float64)[..., None]
        g_23 = np.asarray(mesh_dataset.variables["g_23"][:], dtype=np.float64)[..., None]

    metrics_nonorthogonal = StructuredMetrics(
        dx=evolved.metrics.dx,
        dy=evolved.metrics.dy,
        dz=evolved.metrics.dz,
        J=evolved.metrics.J,
        g11=evolved.metrics.g11,
        g22=evolved.metrics.g22,
        g33=evolved.metrics.g33,
        g_22=evolved.metrics.g_22,
        g23=g23,
        Bxy=evolved.metrics.Bxy,
        g_23=g_23,
    )
    metrics_orthogonal = StructuredMetrics(
        dx=evolved.metrics.dx,
        dy=evolved.metrics.dy,
        dz=evolved.metrics.dz,
        J=evolved.metrics.J,
        g11=evolved.metrics.g11,
        g22=evolved.metrics.g22,
        g33=evolved.metrics.g33,
        g_22=evolved.metrics.g_22,
        g23=np.zeros_like(g23),
        Bxy=evolved.metrics.Bxy,
        g_23=np.zeros_like(g_23),
    )

    nonorthogonal_terms = apply_anomalous_diffusion(
        config,
        species=species,
        mesh=evolved.mesh,
        metrics=metrics_nonorthogonal,
        dataset_scalars=dataset_scalars,
    )
    orthogonal_terms = apply_anomalous_diffusion(
        config,
        species=species,
        mesh=evolved.mesh,
        metrics=metrics_orthogonal,
        dataset_scalars=dataset_scalars,
    )
    return _TokamakAnomalousDiffusionContext(
        input_path=input_path,
        mesh_path=mesh_path,
        snapshot_path=Path(snapshot_path),
        config=config,
        mesh=evolved.mesh,
        metrics_nonorthogonal=metrics_nonorthogonal,
        metrics_orthogonal=metrics_orthogonal,
        dataset_scalars=dataset_scalars,
        species=species,
        nonorthogonal_terms=nonorthogonal_terms,
        orthogonal_terms=orthogonal_terms,
    )


def _resolve_paths(
    *,
    input_path: str | Path | None,
    mesh_path: str | Path | None,
    snapshot_path: str | Path | None,
    array_history_path: str | Path | None,
    optional_history_path: str | Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    if input_path is None:
        reference_root = require_reference_root()
        resolved_input_path = reference_root / "examples" / "tokamak-2D" / "recycling-dthe" / "BOUT.inp"
    else:
        resolved_input_path = Path(input_path)
    resolved_mesh_path = Path(mesh_path) if mesh_path is not None else resolved_input_path.parent / "tokamak.nc"
    resolved_snapshot_path = Path(snapshot_path) if snapshot_path is not None else _SNAPSHOT_CACHE_PATH
    resolved_array_history_path = Path(array_history_path) if array_history_path is not None else _ARRAY_HISTORY_PATH
    resolved_optional_history_path = Path(optional_history_path) if optional_history_path is not None else _OPTIONAL_HISTORY_PATH
    return (
        resolved_input_path,
        resolved_mesh_path,
        resolved_snapshot_path,
        resolved_array_history_path,
        resolved_optional_history_path,
    )


def _build_tokamak_anomalous_diffusion_summaries(
    context: _TokamakAnomalousDiffusionContext,
) -> dict[str, np.ndarray | int | tuple[str, ...]]:
    active = (
        slice(context.mesh.xstart, context.mesh.xend + 1),
        slice(context.mesh.ystart, context.mesh.yend + 1),
        slice(None),
    )

    anomalous_d_active_point = []
    anomalous_chi_active_point = []
    density_peak_nonorth = []
    density_peak_orth = []
    density_contrast_peak = []
    energy_peak_nonorth = []
    energy_peak_orth = []
    energy_contrast_peak = []
    energy_relative_contrast = []
    momentum_contrast_peak = []
    for name in _SPECIES_ORDER:
        anomalous_d_active_point.append(_active_point(context.nonorthogonal_terms.diagnostics.get(f"anomalous_D_{name}"), active))
        anomalous_chi_active_point.append(_active_point(context.nonorthogonal_terms.diagnostics.get(f"anomalous_Chi_{name}"), active))

        density_nonorth = np.asarray(context.nonorthogonal_terms.density_source[name], dtype=np.float64)[active]
        density_orth = np.asarray(context.orthogonal_terms.density_source[name], dtype=np.float64)[active]
        density_delta = density_nonorth - density_orth
        density_peak_nonorth.append(float(np.max(np.abs(density_nonorth))))
        density_peak_orth.append(float(np.max(np.abs(density_orth))))
        density_contrast_peak.append(float(np.max(np.abs(density_delta))))

        energy_nonorth = np.asarray(context.nonorthogonal_terms.energy_source[name], dtype=np.float64)[active]
        energy_orth = np.asarray(context.orthogonal_terms.energy_source[name], dtype=np.float64)[active]
        energy_delta = energy_nonorth - energy_orth
        peak_nonorth = float(np.max(np.abs(energy_nonorth)))
        peak_orth = float(np.max(np.abs(energy_orth)))
        peak_delta = float(np.max(np.abs(energy_delta)))
        energy_peak_nonorth.append(peak_nonorth)
        energy_peak_orth.append(peak_orth)
        energy_contrast_peak.append(peak_delta)
        energy_relative_contrast.append(peak_delta / max(peak_orth, 1.0e-30))

        momentum_nonorth = np.asarray(context.nonorthogonal_terms.momentum_source[name], dtype=np.float64)[active]
        momentum_orth = np.asarray(context.orthogonal_terms.momentum_source[name], dtype=np.float64)[active]
        momentum_contrast_peak.append(float(np.max(np.abs(momentum_nonorth - momentum_orth))))

    d_plus_delta = np.asarray(
        context.nonorthogonal_terms.energy_source["d+"] - context.orthogonal_terms.energy_source["d+"],
        dtype=np.float64,
    )[active][:, :, 0]
    representative_x_offset = int(np.argmax(np.max(np.abs(d_plus_delta), axis=1)))
    representative_x_index = context.mesh.xstart + representative_x_offset
    y_slice = slice(context.mesh.ystart, context.mesh.yend + 1)
    y_line = np.asarray(context.mesh.y[y_slice], dtype=np.float64)

    d_plus_nonorth_line = np.asarray(context.nonorthogonal_terms.energy_source["d+"], dtype=np.float64)[representative_x_index, y_slice, 0]
    d_plus_orth_line = np.asarray(context.orthogonal_terms.energy_source["d+"], dtype=np.float64)[representative_x_index, y_slice, 0]
    t_plus_nonorth_line = np.asarray(context.nonorthogonal_terms.energy_source["t+"], dtype=np.float64)[representative_x_index, y_slice, 0]
    t_plus_orth_line = np.asarray(context.orthogonal_terms.energy_source["t+"], dtype=np.float64)[representative_x_index, y_slice, 0]

    return {
        "species_order": _SPECIES_ORDER,
        "anomalous_D_active_point": np.asarray(anomalous_d_active_point, dtype=np.float64),
        "anomalous_Chi_active_point": np.asarray(anomalous_chi_active_point, dtype=np.float64),
        "density_peak_nonorth": np.asarray(density_peak_nonorth, dtype=np.float64),
        "density_peak_orth": np.asarray(density_peak_orth, dtype=np.float64),
        "density_contrast_peak": np.asarray(density_contrast_peak, dtype=np.float64),
        "energy_peak_nonorth": np.asarray(energy_peak_nonorth, dtype=np.float64),
        "energy_peak_orth": np.asarray(energy_peak_orth, dtype=np.float64),
        "energy_contrast_peak": np.asarray(energy_contrast_peak, dtype=np.float64),
        "energy_relative_contrast": np.asarray(energy_relative_contrast, dtype=np.float64),
        "momentum_contrast_peak": np.asarray(momentum_contrast_peak, dtype=np.float64),
        "representative_x_index": np.asarray(representative_x_index, dtype=np.int64),
        "y_line": y_line,
        "d_plus_energy_nonorth_line": np.asarray(d_plus_nonorth_line, dtype=np.float64),
        "d_plus_energy_orth_line": np.asarray(d_plus_orth_line, dtype=np.float64),
        "d_plus_energy_delta_line": np.asarray(d_plus_nonorth_line - d_plus_orth_line, dtype=np.float64),
        "t_plus_energy_nonorth_line": np.asarray(t_plus_nonorth_line, dtype=np.float64),
        "t_plus_energy_orth_line": np.asarray(t_plus_orth_line, dtype=np.float64),
        "t_plus_energy_delta_line": np.asarray(t_plus_nonorth_line - t_plus_orth_line, dtype=np.float64),
    }


def _active_point(values: np.ndarray | None, active: tuple[slice, slice, slice]) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    view = array[active]
    if view.size == 0:
        return 0.0
    return float(view.reshape(-1)[0])


def _save_tokamak_anomalous_diffusion_plot(
    summaries: dict[str, np.ndarray | int | tuple[str, ...]],
    output_path: Path,
) -> None:
    species_order = list(summaries["species_order"])
    x = np.arange(len(species_order), dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    anomalous_d = np.asarray(summaries["anomalous_D_active_point"], dtype=np.float64)
    anomalous_chi = np.asarray(summaries["anomalous_Chi_active_point"], dtype=np.float64)
    axes[0, 0].bar(x - 0.18, anomalous_d, width=0.36, label="D")
    axes[0, 0].bar(x + 0.18, anomalous_chi, width=0.36, label=r"$\chi$")
    axes[0, 0].set_xticks(x, species_order)
    style_axis(
        axes[0, 0],
        title="Configured anomalous coefficients",
        ylabel="normalized coefficient",
    )
    axes[0, 0].legend(frameon=False)

    energy_relative_contrast = np.asarray(summaries["energy_relative_contrast"], dtype=np.float64)
    axes[0, 1].bar(x, energy_relative_contrast, color="#c85c3a", width=0.6)
    axes[0, 1].set_xticks(x, species_order)
    style_axis(
        axes[0, 1],
        title="Non-orthogonal energy-transport contrast",
        ylabel=r"$\max|\Delta S_E| / \max|S_E^{\mathrm{orth}}|$",
    )
    annotate_bars(axes[0, 1], x, energy_relative_contrast, fmt="{:.3f}")

    y_line = np.asarray(summaries["y_line"], dtype=np.float64)
    d_nonorth = np.asarray(summaries["d_plus_energy_nonorth_line"], dtype=np.float64)
    d_orth = np.asarray(summaries["d_plus_energy_orth_line"], dtype=np.float64)
    d_delta = np.asarray(summaries["d_plus_energy_delta_line"], dtype=np.float64)
    d_slice = support_window_slice(d_nonorth, d_orth, d_delta, padding=1, threshold=1.0e-8)
    axes[1, 0].plot(y_line[d_slice], d_nonorth[d_slice], label="d+ nonorth")
    axes[1, 0].plot(y_line[d_slice], d_orth[d_slice], label="d+ orth")
    style_axis(
        axes[1, 0],
        title="Representative d+ anomalous-energy lineout",
        xlabel="parallel coordinate y",
        ylabel="energy source",
        grid="both",
    )
    delta_axis_d = axes[1, 0].twinx()
    delta_axis_d.plot(
        y_line[d_slice],
        d_delta[d_slice],
        color="#2c9a42",
        linestyle="--",
        label=r"$\Delta$",
    )
    delta_axis_d.set_ylabel(r"$\Delta$")
    delta_axis_d.ticklabel_format(axis="y", style="plain", useOffset=False)
    delta_axis_d.spines["top"].set_visible(False)
    d_handles, d_labels = axes[1, 0].get_legend_handles_labels()
    d_delta_handles, d_delta_labels = delta_axis_d.get_legend_handles_labels()
    axes[1, 0].legend(d_handles + d_delta_handles, d_labels + d_delta_labels, frameon=False, loc="upper right")

    t_nonorth = np.asarray(summaries["t_plus_energy_nonorth_line"], dtype=np.float64)
    t_orth = np.asarray(summaries["t_plus_energy_orth_line"], dtype=np.float64)
    t_delta = np.asarray(summaries["t_plus_energy_delta_line"], dtype=np.float64)
    t_slice = support_window_slice(t_nonorth, t_orth, t_delta, padding=1, threshold=1.0e-8)
    axes[1, 1].plot(y_line[t_slice], t_nonorth[t_slice], label="t+ nonorth")
    axes[1, 1].plot(y_line[t_slice], t_orth[t_slice], label="t+ orth")
    style_axis(
        axes[1, 1],
        title="Representative t+ anomalous-energy lineout",
        xlabel="parallel coordinate y",
        ylabel="energy source",
        grid="both",
    )
    delta_axis_t = axes[1, 1].twinx()
    delta_axis_t.plot(
        y_line[t_slice],
        t_delta[t_slice],
        color="#2c9a42",
        linestyle="--",
        label=r"$\Delta$",
    )
    delta_axis_t.set_ylabel(r"$\Delta$")
    delta_axis_t.ticklabel_format(axis="y", style="plain", useOffset=False)
    delta_axis_t.spines["top"].set_visible(False)
    t_handles, t_labels = axes[1, 1].get_legend_handles_labels()
    t_delta_handles, t_delta_labels = delta_axis_t.get_legend_handles_labels()
    axes[1, 1].legend(t_handles + t_delta_handles, t_labels + t_delta_labels, frameon=False, loc="upper right")
    figure.suptitle("Tokamak anomalous-transport geometry audit", fontsize=13.5, fontweight="semibold")
    save_publication_figure(figure, output_path)
