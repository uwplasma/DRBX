from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import apply_bout_overrides, load_bout_input
from ..native.mesh import build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..native.recycling_1d import (
    _charge_exchange_collision_rates,
    _compute_collision_frequencies,
    _electron_density,
    _hydrogen_cx_sigmav,
    _initialize_species,
    _ion_parallel_viscosity_inputs,
    _load_openadas_rate,
    _neutral_ionisation_collision_rates,
    _prepare_open_field_states,
    _reaction_sources,
)
from ..native.units import resolved_dataset_scalars
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class ReactionsCollisionsCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class ReactionsCollisionsCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_reactions_collisions_campaign_package(
    *,
    output_root: str | Path,
    single_species_input: str | Path | None = None,
    multispecies_input: str | Path | None = None,
    case_label: str = "reactions_collisions_campaign",
) -> ReactionsCollisionsCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_single_species_input, resolved_multispecies_input = _resolve_inputs(
        single_species_input=single_species_input,
        multispecies_input=multispecies_input,
    )
    metrics = build_reactions_collisions_campaign(
        single_species_input=resolved_single_species_input,
        multispecies_input=resolved_multispecies_input,
    )

    summary_payload = {
        "family": "reactions_collisions_and_atomic_data",
        "single_species_input_name": resolved_single_species_input.name,
        "multispecies_input_name": resolved_multispecies_input.name,
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
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {
        "metric_values": np.asarray([metric.value for metric in metrics], dtype=np.float64),
        "metric_targets": np.asarray([metric.target for metric in metrics], dtype=np.float64),
        "metric_pass": np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64),
    }
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_campaign_plot(metrics, plot_png_path)
    return ReactionsCollisionsCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_reactions_collisions_campaign(
    *,
    single_species_input: str | Path,
    multispecies_input: str | Path,
) -> tuple[ReactionsCollisionsCampaignMetric, ...]:
    single_species_path = Path(single_species_input)
    multispecies_path = Path(multispecies_input)

    single_species = _build_species_context(single_species_path)
    multispecies = _build_species_context(multispecies_path)

    single_active = (single_species["mesh"].xstart, single_species["mesh"].ystart, 0)
    multispecies_active = (multispecies["mesh"].xstart, multispecies["mesh"].ystart, 0)

    single_cx_rates = _charge_exchange_collision_rates(
        single_species["config"],
        species=single_species["species"],
        prepared=single_species["prepared"],
        dataset_scalars=single_species["dataset_scalars"],
    )
    multispecies_cx_rates = _charge_exchange_collision_rates(
        multispecies["config"],
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_collision_rates = _compute_collision_frequencies(
        multispecies["config"],
        multispecies["species"],
        multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_viscosity_inputs = _ion_parallel_viscosity_inputs(
        species_name="d+",
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        collision_rates=multispecies_collision_rates,
        cx_rates=multispecies_cx_rates,
    )
    ionisation_rates = _neutral_ionisation_collision_rates(
        single_species["config"],
        species=single_species["species"],
        prepared=single_species["prepared"],
        dataset_scalars=single_species["dataset_scalars"],
    )
    reaction_terms = _reaction_sources(
        single_species["config"],
        species=single_species["species"],
        electron_density=_electron_density(tuple(sp for sp in single_species["species"].values() if sp.charge > 0.0)),
        dataset_scalars=single_species["dataset_scalars"],
    )

    active_same = float(single_species["prepared"]["d+"].density[single_active] * _hydrogen_cx_sigmav(
        np.clip(
            (
                single_species["prepared"]["d"].temperature / single_species["species"]["d"].atomic_mass
                + single_species["prepared"]["d+"].temperature / single_species["species"]["d+"].atomic_mass
            ) * single_species["dataset_scalars"]["Tnorm"],
            0.01,
            10000.0,
        ),
        single_species["dataset_scalars"],
    )[single_active])
    active_atom_rate = float(single_cx_rates["d"][single_active])

    d_same = float(multispecies["prepared"]["d+"].density[multispecies_active] * _hydrogen_cx_sigmav(
        np.clip(
            (
                multispecies["prepared"]["d"].temperature / multispecies["species"]["d"].atomic_mass
                + multispecies["prepared"]["d+"].temperature / multispecies["species"]["d+"].atomic_mass
            ) * multispecies["dataset_scalars"]["Tnorm"],
            0.01,
            10000.0,
        ),
        multispecies["dataset_scalars"],
    )[multispecies_active])
    d_cross = float(multispecies["prepared"]["t+"].density[multispecies_active] * _hydrogen_cx_sigmav(
        np.clip(
            (
                multispecies["prepared"]["d"].temperature / multispecies["species"]["d"].atomic_mass
                + multispecies["prepared"]["t+"].temperature / multispecies["species"]["t+"].atomic_mass
            ) * multispecies["dataset_scalars"]["Tnorm"],
            0.01,
            10000.0,
        ),
        multispecies["dataset_scalars"],
    )[multispecies_active])

    scaled_config = apply_bout_overrides(load_bout_input(multispecies_path), ("d:K_cx_multiplier=3.0",))
    scaled_context = _build_species_context(multispecies_path, config=scaled_config)
    scaled_cx_rates = _charge_exchange_collision_rates(
        scaled_context["config"],
        species=scaled_context["species"],
        prepared=scaled_context["prepared"],
        dataset_scalars=scaled_context["dataset_scalars"],
    )
    scaled_active = (scaled_context["mesh"].xstart, scaled_context["mesh"].ystart, 0)
    multiplier_ratio = float(
        scaled_cx_rates["d"][scaled_active] / max(multispecies_cx_rates["d"][multispecies_active], np.finfo(np.float64).tiny)
    )

    expected_ionisation = float(
        reaction_terms.diagnostics["Sd+_iz"][single_species["mesh"].xstart, single_species["mesh"].yend, 0]
        / single_species["species"]["d"].density[single_species["mesh"].xstart, single_species["mesh"].yend, 0]
    )
    actual_ionisation = float(ionisation_rates["d"][single_species["mesh"].xstart, single_species["mesh"].yend, 0])

    expected_collisionality = np.zeros_like(multispecies["prepared"]["d+"].density, dtype=np.float64)
    for other_name in multispecies["species"]:
        rate = multispecies_collision_rates.get(("d+", other_name))
        if rate is not None:
            expected_collisionality = expected_collisionality + rate
    expected_collisionality = np.maximum(expected_collisionality + multispecies_cx_rates["d+"], 1.0e-12)
    collisionality_relative_error = float(
        np.max(
            np.abs(multispecies_viscosity_inputs.total_collisionality - expected_collisionality)
            / np.maximum(expected_collisionality, np.finfo(np.float64).tiny)
        )
    )

    ionisation_coeffs, radiation_coeffs, log_temperature, log_density, electron_heating = _load_openadas_rate("ne", "iz")
    neon_finite_fraction = float(
        (
            np.isfinite(ionisation_coeffs).mean()
            + np.isfinite(radiation_coeffs).mean()
            + np.isfinite(log_temperature).mean()
            + np.isfinite(log_density).mean()
            + np.isfinite(electron_heating).mean()
        )
        / 5.0
    )

    metrics = (
        ReactionsCollisionsCampaignMetric(
            name="single_species_atom_cx_matches_same_species_formula",
            kind="relative_error",
            value=abs(active_atom_rate - active_same) / max(abs(active_same), np.finfo(np.float64).tiny),
            target=1.0e-12,
            passed=abs(active_atom_rate - active_same) <= max(abs(active_same), 1.0) * 1.0e-12,
            notes="Single-species atom charge-exchange rate matches the closed-form hydrogen sigma-v expression.",
        ),
        ReactionsCollisionsCampaignMetric(
            name="multispecies_cross_isotope_cx_fraction",
            kind="fraction",
            value=d_cross / max(d_same + d_cross, np.finfo(np.float64).tiny),
            target=1.0e-2,
            passed=(d_cross / max(d_same + d_cross, np.finfo(np.float64).tiny)) > 1.0e-2,
            notes="Cross-isotope charge exchange is non-trivial on the multispecies recycling lane.",
        ),
        ReactionsCollisionsCampaignMetric(
            name="species_rate_multiplier_ratio",
            kind="ratio",
            value=multiplier_ratio,
            target=3.0,
            passed=np.isclose(multiplier_ratio, 3.0, rtol=1.0e-12, atol=1.0e-12),
            notes="Per-species charge-exchange multiplier is applied on the native rate assembly path.",
        ),
        ReactionsCollisionsCampaignMetric(
            name="ionisation_rate_matches_reaction_diagnostic",
            kind="relative_error",
            value=abs(actual_ionisation - expected_ionisation) / max(abs(expected_ionisation), np.finfo(np.float64).tiny),
            target=1.0e-12,
            passed=abs(actual_ionisation - expected_ionisation) <= max(abs(expected_ionisation), 1.0) * 1.0e-12,
            notes="Neutral ionisation collision rate matches the reaction diagnostic per neutral density.",
        ),
        ReactionsCollisionsCampaignMetric(
            name="ion_parallel_viscosity_collisionality_closure",
            kind="relative_error",
            value=collisionality_relative_error,
            target=1.0e-12,
            passed=collisionality_relative_error <= 1.0e-12,
            notes="Total collisionality used by the ion parallel viscosity closure matches the assembled rate stack.",
        ),
        ReactionsCollisionsCampaignMetric(
            name="openadas_neon_rate_bundle_finite_fraction",
            kind="fraction",
            value=neon_finite_fraction,
            target=1.0,
            passed=np.isclose(neon_finite_fraction, 1.0, rtol=0.0, atol=0.0),
            notes="Neon OpenADAS bundle loads with finite ionisation, radiation, temperature, density, and heating tables.",
        ),
    )
    return metrics


def _resolve_inputs(
    *,
    single_species_input: str | Path | None,
    multispecies_input: str | Path | None,
) -> tuple[Path, Path]:
    if single_species_input is not None and multispecies_input is not None:
        return Path(single_species_input), Path(multispecies_input)
    reference_root = require_reference_root()
    return (
        Path(single_species_input) if single_species_input is not None else reference_root / "tests" / "integrated" / "1D-recycling" / "data" / "BOUT.inp",
        Path(multispecies_input) if multispecies_input is not None else reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp",
    )


def _build_species_context(path: Path, *, config=None) -> dict[str, object]:
    if config is None:
        config = load_bout_input(path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
    )
    return {
        "config": config,
        "mesh": mesh,
        "metrics": metrics,
        "dataset_scalars": dataset_scalars,
        "species": species,
        "prepared": prepared,
    }


def _save_campaign_plot(metrics: tuple[ReactionsCollisionsCampaignMetric, ...], path: Path) -> None:
    labels = [metric.name.replace("_", "\n") for metric in metrics]
    values = [metric.value for metric in metrics]
    targets = [metric.target for metric in metrics]
    colors = ["#0a9396" if metric.passed else "#bb3e03" for metric in metrics]

    figure, axis = plt.subplots(figsize=(12.0, 6.5), constrained_layout=True)
    x = np.arange(len(metrics))
    axis.bar(x, values, color=colors, alpha=0.9)
    axis.plot(x, targets, color="#3a86ff", marker="o", linewidth=1.8, label="target")
    axis.set_xticks(x, labels)
    axis.set_ylabel("metric value")
    axis.set_title("Reactions and collisions verification campaign")
    axis.grid(alpha=0.25, axis="y")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
