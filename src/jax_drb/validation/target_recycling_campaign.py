from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import load_bout_input
from ..native.reference_dump import load_local_reference_snapshot_cache
from ..native.recycling_1d import _initialize_species, _prepare_open_field_states
from ..native.recycling_targets import electron_zero_current_velocity, target_recycling_sources
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration
from ..native.units import resolved_dataset_scalars
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
_NEUTRAL_ORDER = ("d", "t", "he")
_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "recycling_dthe_rhs_snapshot.npz"
)


@dataclass(frozen=True)
class TargetRecyclingCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class TargetRecyclingCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _TargetRecyclingContext:
    input_path: Path
    snapshot_path: Path
    config: object
    mesh: object
    metrics: object
    species: dict[str, object]
    prepared: dict[str, object]
    ion_velocity: dict[str, np.ndarray]
    target_terms: object
    electron_boundary: object
    electron_zero_current_velocity: np.ndarray


def create_target_recycling_campaign_package(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    case_label: str = "target_recycling_campaign",
) -> TargetRecyclingCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_path, resolved_snapshot_path = _resolve_paths(input_path=input_path, snapshot_path=snapshot_path)
    context = build_target_recycling_context(
        input_path=resolved_input_path,
        snapshot_path=resolved_snapshot_path,
    )
    metrics = build_target_recycling_campaign(
        input_path=resolved_input_path,
        snapshot_path=resolved_snapshot_path,
    )
    summaries = _build_target_recycling_summaries(context)

    summary_payload = {
        "family": "target_recycling",
        "input_name": resolved_input_path.name,
        "snapshot_name": resolved_snapshot_path.name,
        "literature_anchor": {
            "sheath_boundary_conditions": "Cohen and Ryutov 2004 sheath physics and boundary conditions for edge plasmas",
            "edge_model_context": "Dudson et al. 2024, Hermes-3 multi-component plasma simulations",
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
        "neutral_order": list(summaries["neutral_order"]),
        "representative_x_index": int(summaries["representative_x_index"]),
        "summaries": {
            name: [float(value) for value in values]
            for name, values in summaries.items()
            if name not in {"neutral_order", "representative_x_index"}
        },
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in summaries.items()
        if name != "neutral_order"
    }
    arrays_payload["metric_values"] = np.asarray([metric.value for metric in metrics], dtype=np.float64)
    arrays_payload["metric_targets"] = np.asarray([metric.target for metric in metrics], dtype=np.float64)
    arrays_payload["metric_pass"] = np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64)
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_target_recycling_plot(summaries, plot_png_path)
    return TargetRecyclingCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_target_recycling_campaign(
    *,
    input_path: str | Path,
    snapshot_path: str | Path,
) -> tuple[TargetRecyclingCampaignMetric, ...]:
    context = build_target_recycling_context(input_path=input_path, snapshot_path=snapshot_path)
    summaries = _build_target_recycling_summaries(context)
    electron_sheath_energy_peak = float(
        np.asarray(summaries["electron_sheath_energy_peak"], dtype=np.float64).reshape(-1)[0]
    )
    electron_zero_current_velocity_finite_fraction = float(
        np.asarray(summaries["electron_zero_current_velocity_finite_fraction"], dtype=np.float64).reshape(-1)[0]
    )
    return (
        *(
            TargetRecyclingCampaignMetric(
                name=f"{neutral_name}_target_recycling_density_peak",
                kind="min_value",
                value=float(value),
                target=1.0e-5 if neutral_name in {"d", "t"} else 1.0e-7,
                passed=bool(value >= (1.0e-5 if neutral_name in {"d", "t"} else 1.0e-7)),
                notes=f"{neutral_name} target recycling should remain active on the prepared multispecies recycling state.",
            )
            for neutral_name, value in zip(_NEUTRAL_ORDER, summaries["target_recycling_density_peak"], strict=True)
        ),
        TargetRecyclingCampaignMetric(
            name="electron_sheath_energy_peak",
            kind="min_value",
            value=electron_sheath_energy_peak,
            target=1.0e-4,
            passed=bool(electron_sheath_energy_peak >= 1.0e-4),
            notes="The boundary-conditioned electron energy sink should remain active on the prepared state.",
        ),
        TargetRecyclingCampaignMetric(
            name="electron_zero_current_velocity_finite_fraction",
            kind="fraction",
            value=electron_zero_current_velocity_finite_fraction,
            target=1.0,
            passed=bool(electron_zero_current_velocity_finite_fraction == 1.0),
            notes="The zero-current electron velocity reconstruction should stay finite on the prepared state.",
        ),
    )


def build_target_recycling_context(
    *,
    input_path: str | Path,
    snapshot_path: str | Path,
) -> _TargetRecyclingContext:
    input_path = Path(input_path)
    snapshot_path = Path(snapshot_path)
    config = load_bout_input(input_path)
    snapshot = load_local_reference_snapshot_cache(
        snapshot_path,
        field_names=_FIELD_NAMES,
        scalar_names=_SCALAR_NAMES,
    )
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(
        config,
        mesh=snapshot.mesh,
        dataset_scalars=snapshot.scalar_values or dataset_scalars,
        field_overrides=snapshot.fields,
    )
    prepared, _, electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=snapshot.scalar_values or dataset_scalars,
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e")
    ion_velocity = {ion.name: prepared[ion.name].velocity for ion in ions}
    target_terms = target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        gamma_i=2.5,
    )
    electron_current_free = electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=prepared["e"].density,
    )
    return _TargetRecyclingContext(
        input_path=input_path,
        snapshot_path=snapshot_path,
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        species=species,
        prepared=prepared,
        ion_velocity=ion_velocity,
        target_terms=target_terms,
        electron_boundary=electron_boundary,
        electron_zero_current_velocity=electron_current_free,
    )


def _resolve_paths(
    *,
    input_path: str | Path | None,
    snapshot_path: str | Path | None,
) -> tuple[Path, Path]:
    if input_path is None:
        reference_root = require_reference_root()
        resolved_input_path = reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp"
    else:
        resolved_input_path = Path(input_path)
    resolved_snapshot_path = Path(snapshot_path) if snapshot_path is not None else _SNAPSHOT_CACHE_PATH
    return resolved_input_path, resolved_snapshot_path


def _build_target_recycling_summaries(context: _TargetRecyclingContext) -> dict[str, np.ndarray | int | tuple[str, ...]]:
    active = (
        slice(context.mesh.xstart, context.mesh.xend + 1),
        slice(context.mesh.ystart, context.mesh.yend + 1),
        slice(None),
    )
    target_density_peak = []
    target_density_integral = []
    target_energy_integral = []
    for neutral_name in _NEUTRAL_ORDER:
        density_line = np.asarray(context.target_terms.diagnostics[f"S{neutral_name}_target_recycle"], dtype=np.float64)[active]
        energy_line = np.asarray(context.target_terms.diagnostics[f"E{neutral_name}_target_recycle"], dtype=np.float64)[active]
        target_density_peak.append(float(np.max(np.abs(density_line))))
        target_density_integral.append(float(np.sum(density_line)))
        target_energy_integral.append(float(np.sum(energy_line)))

    d_density = np.asarray(context.target_terms.diagnostics["Sd_target_recycle"], dtype=np.float64)[active][:, :, 0]
    representative_x_offset = int(np.argmax(np.max(np.abs(d_density), axis=1)))
    representative_x_index = context.mesh.xstart + representative_x_offset
    y_slice = slice(context.mesh.ystart, context.mesh.yend + 1)
    y_line = np.asarray(context.mesh.y[y_slice], dtype=np.float64)

    density_lineouts = {
        neutral_name: np.asarray(context.target_terms.diagnostics[f"S{neutral_name}_target_recycle"], dtype=np.float64)[representative_x_index, y_slice, 0]
        for neutral_name in _NEUTRAL_ORDER
    }
    energy_lineouts = {
        neutral_name: np.asarray(context.target_terms.diagnostics[f"E{neutral_name}_target_recycle"], dtype=np.float64)[representative_x_index, y_slice, 0]
        for neutral_name in _NEUTRAL_ORDER
    }
    electron_sheath_energy_line = np.asarray(context.electron_boundary.energy_source, dtype=np.float64)[representative_x_index, y_slice, 0]
    electron_zero_current_velocity_line = np.asarray(context.electron_zero_current_velocity, dtype=np.float64)[representative_x_index, y_slice, 0]
    electron_zero_current_velocity_finite_fraction = float(
        np.mean(np.isfinite(np.asarray(context.electron_zero_current_velocity, dtype=np.float64)[active]))
    )

    return {
        "neutral_order": _NEUTRAL_ORDER,
        "representative_x_index": np.asarray(representative_x_index, dtype=np.int64),
        "target_recycling_density_peak": np.asarray(target_density_peak, dtype=np.float64),
        "target_recycling_density_integral": np.asarray(target_density_integral, dtype=np.float64),
        "target_recycling_energy_integral": np.asarray(target_energy_integral, dtype=np.float64),
        "y_line": y_line,
        "d_target_density_line": np.asarray(density_lineouts["d"], dtype=np.float64),
        "t_target_density_line": np.asarray(density_lineouts["t"], dtype=np.float64),
        "he_target_density_line": np.asarray(density_lineouts["he"], dtype=np.float64),
        "d_target_energy_line": np.asarray(energy_lineouts["d"], dtype=np.float64),
        "t_target_energy_line": np.asarray(energy_lineouts["t"], dtype=np.float64),
        "he_target_energy_line": np.asarray(energy_lineouts["he"], dtype=np.float64),
        "electron_sheath_energy_line": np.asarray(electron_sheath_energy_line, dtype=np.float64),
        "electron_sheath_energy_peak": np.asarray([float(np.max(np.abs(electron_sheath_energy_line)))], dtype=np.float64),
        "electron_zero_current_velocity_line": np.asarray(electron_zero_current_velocity_line, dtype=np.float64),
        "electron_zero_current_velocity_finite_fraction": np.asarray([electron_zero_current_velocity_finite_fraction], dtype=np.float64),
    }


def _save_target_recycling_plot(
    summaries: dict[str, np.ndarray | int | tuple[str, ...]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    y_line = np.asarray(summaries["y_line"], dtype=np.float64)
    d_density_line = np.asarray(summaries["d_target_density_line"], dtype=np.float64)
    t_density_line = np.asarray(summaries["t_target_density_line"], dtype=np.float64)
    he_density_line = np.asarray(summaries["he_target_density_line"], dtype=np.float64)
    electron_sheath_energy_line = np.asarray(summaries["electron_sheath_energy_line"], dtype=np.float64)
    plot_slice = support_window_slice(
        d_density_line,
        t_density_line,
        he_density_line,
        electron_sheath_energy_line,
        padding=4,
    )
    y_plot = y_line[plot_slice]

    axes[0, 0].plot(y_plot, d_density_line[plot_slice], label="d")
    axes[0, 0].plot(y_plot, t_density_line[plot_slice], label="t")
    axes[0, 0].plot(y_plot, he_density_line[plot_slice], label="he")
    style_axis(
        axes[0, 0],
        title="Target recycling density sources near the target",
        xlabel="parallel coordinate y",
        ylabel="density source",
        grid="both",
    )
    axes[0, 0].legend(frameon=False)

    neutral_order = list(summaries["neutral_order"])
    x = np.arange(len(neutral_order), dtype=np.float64)
    density_integral = np.asarray(summaries["target_recycling_density_integral"], dtype=np.float64)
    axes[0, 1].bar(x, density_integral, color="#c85c3a", width=0.6)
    axes[0, 1].set_xticks(x, neutral_order)
    style_axis(
        axes[0, 1],
        title="Integrated target recycling source",
        ylabel="integrated density source",
    )
    annotate_bars(axes[0, 1], x, density_integral, fmt="{:.2e}")

    axes[1, 0].plot(y_plot, electron_sheath_energy_line[plot_slice], color="#6b5ca5")
    style_axis(
        axes[1, 0],
        title="Boundary-conditioned electron energy sink near the target",
        xlabel="parallel coordinate y",
        ylabel="electron energy source",
        grid="both",
    )

    density_peak = np.asarray(summaries["target_recycling_density_peak"], dtype=np.float64)
    axes[1, 1].bar(x, density_peak, color="#2c9a42", width=0.6)
    axes[1, 1].set_xticks(x, neutral_order)
    style_axis(
        axes[1, 1],
        title="Peak target recycling strength",
        ylabel="peak density source",
    )
    annotate_bars(axes[1, 1], x, density_peak, fmt="{:.2e}")

    figure.suptitle("Target recycling closure audit", fontsize=15.0, fontweight="semibold")
    save_publication_figure(figure, output_path)
