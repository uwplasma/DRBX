from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import apply_bout_overrides, load_bout_input
from ..native.reference_dump import (
    load_local_reference_snapshot,
    load_local_reference_snapshot_cache,
)
from ..native.recycling_collisions import compute_collision_frequencies
from ..native.recycling_neutral_diffusion import apply_neutral_parallel_diffusion
from ..native.recycling_reactions import (
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
)
from ..native.recycling_setup import initialize_species
from ..native.recycling_1d import _prepare_open_field_states
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration
from ..native.units import resolved_dataset_scalars
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


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
_SNAPSHOT_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / "references"
    / "baselines"
    / "reference_snapshots"
    / "recycling_dthe_rhs_snapshot.npz"
)


@dataclass(frozen=True)
class NeutralParallelDiffusionCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class NeutralParallelDiffusionCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _NeutralParallelDiffusionContext:
    label: str
    input_path: Path
    dump_path: Path
    config: object
    mesh: object
    metrics: object
    dataset_scalars: dict[str, float]
    species: dict[str, object]
    prepared: dict[str, object]
    terms: object
    ionisation_rates: dict[str, np.ndarray]
    charge_exchange_rates: dict[str, np.ndarray]
    multispecies_collision_totals: dict[str, np.ndarray]


def create_neutral_parallel_diffusion_campaign_package(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    dump_path: str | Path | None = None,
    case_label: str = "neutral_parallel_diffusion_campaign",
) -> NeutralParallelDiffusionCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_path, resolved_dump_path = _resolve_paths(input_path=input_path, dump_path=dump_path)
    afn_context = build_neutral_parallel_diffusion_context(
        resolved_input_path,
        dump_path=resolved_dump_path,
        diffusion_mode="afn",
    )
    multispecies_context = build_neutral_parallel_diffusion_context(
        resolved_input_path,
        dump_path=resolved_dump_path,
        diffusion_mode="multispecies",
    )
    metrics = _build_neutral_parallel_diffusion_metrics(
        afn=afn_context,
        multispecies=multispecies_context,
    )
    species_summaries = _build_neutral_parallel_diffusion_species_summaries(
        afn=afn_context,
        multispecies=multispecies_context,
    )

    summary_payload = {
        "family": "neutral_parallel_diffusion",
        "input_name": resolved_input_path.name,
        "snapshot_name": resolved_dump_path.name,
        "literature_anchor": {
            "primary_code_paper": "Dudson et al. 2024, Hermes-3 multi-component plasma simulations",
            "closure_note": "Hermes-3 documents AFN as the recommended neutral model, with multispecies diffusion retained as a legacy comparison mode.",
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
        "species_order": list(species_summaries["species_order"]),
        "species_summaries": {
            name: [float(value) for value in values]
            for name, values in species_summaries.items()
            if name != "species_order"
        },
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {
        "metric_values": np.asarray([metric.value for metric in metrics], dtype=np.float64),
        "metric_targets": np.asarray([metric.target for metric in metrics], dtype=np.float64),
        "metric_pass": np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64),
    }
    for name, values in species_summaries.items():
        if name == "species_order":
            continue
        arrays_payload[name] = np.asarray(values, dtype=np.float64)
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_neutral_parallel_diffusion_plot(metrics, species_summaries, plot_png_path)
    return NeutralParallelDiffusionCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_neutral_parallel_diffusion_campaign(
    *,
    input_path: str | Path,
    dump_path: str | Path,
) -> tuple[NeutralParallelDiffusionCampaignMetric, ...]:
    afn = build_neutral_parallel_diffusion_context(input_path, dump_path=dump_path, diffusion_mode="afn")
    multispecies = build_neutral_parallel_diffusion_context(input_path, dump_path=dump_path, diffusion_mode="multispecies")
    return _build_neutral_parallel_diffusion_metrics(afn=afn, multispecies=multispecies)

def build_neutral_parallel_diffusion_context(
    input_path: str | Path,
    *,
    dump_path: str | Path,
    diffusion_mode: str,
) -> _NeutralParallelDiffusionContext:
    path = Path(input_path)
    config = apply_bout_overrides(
        load_bout_input(path),
        (
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            f"neutral_parallel_diffusion:diffusion_collisions_mode={diffusion_mode}",
            "neutral_parallel_diffusion:equation_fix=true",
            "neutral_parallel_diffusion:perpendicular_conduction=true",
            "neutral_parallel_diffusion:perpendicular_viscosity=true",
            "neutral_parallel_diffusion:diagnose=true",
        ),
    )
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot_path = Path(dump_path)
    if snapshot_path.suffix == ".npz":
        snapshot = load_local_reference_snapshot_cache(
            snapshot_path,
            field_names=_FIELD_NAMES,
            scalar_names=_SCALAR_NAMES,
        )
    else:
        snapshot = load_local_reference_snapshot(
            snapshot_path,
            field_names=_FIELD_NAMES,
            scalar_names=_SCALAR_NAMES,
        )
    species = initialize_species(
        config,
        mesh=snapshot.mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=snapshot.fields,
    )
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
    )
    collision_rates = compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=dataset_scalars,
    )
    ionisation_rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    charge_exchange_rates = neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    multispecies_collision_totals: dict[str, np.ndarray] = {}
    for name, sp in species.items():
        if name == "e" or sp.charge != 0.0:
            continue
        total = np.zeros_like(np.asarray(prepared[name].density, dtype=np.float64))
        for (left_name, _), rate in collision_rates.items():
            if left_name == name:
                total = total + np.asarray(rate, dtype=np.float64)
        multispecies_collision_totals[name] = total
    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
    )
    return _NeutralParallelDiffusionContext(
        label=diffusion_mode,
        input_path=path,
        dump_path=Path(dump_path),
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        species=species,
        prepared=prepared,
        terms=terms,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=charge_exchange_rates,
        multispecies_collision_totals=multispecies_collision_totals,
    )


def _resolve_paths(
    *,
    input_path: str | Path | None,
    dump_path: str | Path | None,
) -> tuple[Path, Path]:
    if input_path is not None:
        resolved_input = Path(input_path)
    else:
        reference_root = require_reference_root()
        resolved_input = reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp"
    if dump_path is not None:
        resolved_dump = Path(dump_path)
    else:
        candidate_dump = resolved_input.with_name("BOUT.dmp.0.nc")
        resolved_dump = candidate_dump if candidate_dump.exists() else _SNAPSHOT_CACHE_PATH
    return resolved_input, resolved_dump


def _active_mean(context: _NeutralParallelDiffusionContext, values: np.ndarray) -> float:
    mesh = context.mesh
    active = np.asarray(
        values[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :],
        dtype=np.float64,
    )
    return float(np.mean(active))


def _build_neutral_parallel_diffusion_metrics(
    *,
    afn: _NeutralParallelDiffusionContext,
    multispecies: _NeutralParallelDiffusionContext,
) -> tuple[NeutralParallelDiffusionCampaignMetric, ...]:
    species_order = ("d", "t", "he")
    afn_values = {name: _active_mean(afn, afn.terms.diagnostics[f"D{name}_Dpar"]) for name in species_order}
    multi_values = {
        name: _active_mean(multispecies, multispecies.terms.diagnostics[f"D{name}_Dpar"])
        for name in species_order
    }
    return (
        NeutralParallelDiffusionCampaignMetric(
            name="afn_diffusion_finite_fraction",
            kind="fraction",
            value=float(
                np.mean(
                    [
                        np.isfinite(_active_mean(afn, afn.terms.diagnostics[f"D{name}_Dpar"]))
                        for name in species_order
                    ]
                )
            ),
            target=1.0,
            passed=all(np.isfinite(value) for value in afn_values.values()),
            notes="AFN neutral parallel diffusivities should remain finite for the prepared D/T/He recycling state.",
        ),
        NeutralParallelDiffusionCampaignMetric(
            name="multispecies_diffusion_finite_fraction",
            kind="fraction",
            value=float(
                np.mean(
                    [
                        np.isfinite(_active_mean(multispecies, multispecies.terms.diagnostics[f"D{name}_Dpar"]))
                        for name in species_order
                    ]
                )
            ),
            target=1.0,
            passed=all(np.isfinite(value) for value in multi_values.values()),
            notes="Multispecies neutral parallel diffusivities should remain finite on the same prepared state.",
        ),
        *(
            NeutralParallelDiffusionCampaignMetric(
                name=f"{name}_afn_to_multispecies_diffusion_contrast",
                kind="min_value",
                value=float(
                    abs(multi_values[name] - afn_values[name])
                    / max(abs(multi_values[name]), 1.0e-30)
                ),
                target=1.0e-3,
                passed=bool(
                    abs(multi_values[name] - afn_values[name]) / max(abs(multi_values[name]), 1.0e-30) >= 1.0e-3
                ),
                notes=(
                    f"The {name} neutral should retain a measurable AFN-versus-multispecies diffusivity contrast "
                    "on the prepared multispecies state."
                ),
            )
            for name in species_order
        ),
        NeutralParallelDiffusionCampaignMetric(
            name="deuterium_afn_charge_exchange_fraction",
            kind="min_value",
            value=float(
                _active_mean(afn, afn.charge_exchange_rates["d"])
                / max(
                    _active_mean(afn, afn.charge_exchange_rates["d"])
                    + _active_mean(afn, afn.ionisation_rates["d"]),
                    1.0e-30,
                )
            ),
            target=0.1,
            passed=bool(
                _active_mean(afn, afn.charge_exchange_rates["d"])
                / max(
                    _active_mean(afn, afn.charge_exchange_rates["d"])
                    + _active_mean(afn, afn.ionisation_rates["d"]),
                    1.0e-30,
                )
                >= 0.1
            ),
            notes="Charge exchange should remain a non-negligible part of the AFN neutral collision budget for deuterium on this prepared state.",
        ),
    )


def _build_neutral_parallel_diffusion_species_summaries(
    *,
    afn: _NeutralParallelDiffusionContext,
    multispecies: _NeutralParallelDiffusionContext,
) -> dict[str, object]:
    species_order = ("d", "t", "he")
    return {
        "species_order": species_order,
        "afn_diffusion": [_active_mean(afn, afn.terms.diagnostics[f"D{name}_Dpar"]) for name in species_order],
        "multispecies_diffusion": [
            _active_mean(multispecies, multispecies.terms.diagnostics[f"D{name}_Dpar"]) for name in species_order
        ],
        "afn_ionisation_rate": [_active_mean(afn, afn.ionisation_rates[name]) for name in species_order],
        "afn_charge_exchange_rate": [
            _active_mean(afn, afn.charge_exchange_rates.get(name, np.zeros_like(afn.species[name].density)))
            for name in species_order
        ],
        "multispecies_collision_rate": [
            _active_mean(multispecies, multispecies.multispecies_collision_totals[name]) for name in species_order
        ],
        "multispecies_charge_exchange_rate": [
            _active_mean(
                multispecies,
                multispecies.charge_exchange_rates.get(name, np.zeros_like(multispecies.species[name].density)),
            )
            for name in species_order
        ],
        "diffusion_ratio": [
            _active_mean(multispecies, multispecies.terms.diagnostics[f"D{name}_Dpar"])
            / max(_active_mean(afn, afn.terms.diagnostics[f"D{name}_Dpar"]), 1.0e-30)
            for name in species_order
        ],
    }


def _save_neutral_parallel_diffusion_plot(
    metrics: tuple[NeutralParallelDiffusionCampaignMetric, ...],
    species_summaries: dict[str, object],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    species = list(species_summaries["species_order"])
    positions = np.arange(len(species), dtype=np.float64)
    width = 0.36
    color_afn = "#1f77b4"
    color_multi = "#d62728"

    axes[0, 0].bar(
        positions - width / 2.0,
        np.asarray(species_summaries["afn_diffusion"], dtype=np.float64),
        width=width,
        label="AFN",
        color=color_afn,
    )
    axes[0, 0].bar(
        positions + width / 2.0,
        np.asarray(species_summaries["multispecies_diffusion"], dtype=np.float64),
        width=width,
        label="Multispecies",
        color=color_multi,
    )
    axes[0, 0].set_xticks(positions, [name.upper() for name in species])
    style_axis(axes[0, 0], title="Effective neutral diffusivity", ylabel=r"$D_{\parallel,n}$")
    axes[0, 0].legend(frameon=False, fontsize=9)
    annotate_bars(axes[0, 0], positions - width / 2.0, np.asarray(species_summaries["afn_diffusion"], dtype=np.float64), fmt="{:.2f}", fontsize=8.5)
    annotate_bars(axes[0, 0], positions + width / 2.0, np.asarray(species_summaries["multispecies_diffusion"], dtype=np.float64), fmt="{:.2f}", fontsize=8.5)

    axes[0, 1].bar(
        positions - width / 2.0,
        np.asarray(species_summaries["afn_ionisation_rate"], dtype=np.float64),
        width=width,
        label="Ionisation",
        color="#2ca02c",
    )
    axes[0, 1].bar(
        positions - width / 2.0,
        np.asarray(species_summaries["afn_charge_exchange_rate"], dtype=np.float64),
        width=width,
        bottom=np.asarray(species_summaries["afn_ionisation_rate"], dtype=np.float64),
        label="Charge exchange",
        color="#9467bd",
    )
    axes[0, 1].set_xticks(positions, [name.upper() for name in species])
    style_axis(axes[0, 1], title="AFN collision budget", ylabel=r"$\nu_n$")
    axes[0, 1].legend(frameon=False, fontsize=9)

    axes[1, 0].bar(
        positions - width / 2.0,
        np.asarray(species_summaries["multispecies_collision_rate"], dtype=np.float64),
        width=width,
        label="Multispecies collisions",
        color="#ff7f0e",
    )
    axes[1, 0].bar(
        positions - width / 2.0,
        np.asarray(species_summaries["multispecies_charge_exchange_rate"], dtype=np.float64),
        width=width,
        bottom=np.asarray(species_summaries["multispecies_collision_rate"], dtype=np.float64),
        label="Charge exchange",
        color="#8c564b",
    )
    axes[1, 0].set_xticks(positions, [name.upper() for name in species])
    style_axis(axes[1, 0], title="Multispecies collision budget", ylabel=r"$\nu_n$")
    axes[1, 0].legend(frameon=False, fontsize=9)

    axes[1, 1].bar(
        positions,
        np.asarray(species_summaries["diffusion_ratio"], dtype=np.float64),
        width=0.55,
        color="#17becf",
    )
    axes[1, 1].axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    axes[1, 1].set_xticks(positions, [name.upper() for name in species])
    style_axis(
        axes[1, 1],
        title="Multispecies / AFN diffusivity",
        ylabel=r"$D_{\parallel,n}^{\mathrm{multi}} / D_{\parallel,n}^{\mathrm{AFN}}$",
    )
    annotate_bars(axes[1, 1], positions, np.asarray(species_summaries["diffusion_ratio"], dtype=np.float64), fmt="{:.2f}")

    passed = sum(1 for metric in metrics if metric.passed)
    figure.suptitle(
        f"Neutral parallel diffusion closure on a prepared D/T/He recycling state ({passed}/{len(metrics)} metrics passing)",
        fontsize=15.0,
        fontweight="semibold",
    )
    save_publication_figure(figure, path)
    passed = sum(1 for metric in metrics if metric.passed)
    figure.suptitle(
        f"Neutral parallel diffusion closure on a prepared D/T/He recycling state ({passed}/{len(metrics)} metrics passing)",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
