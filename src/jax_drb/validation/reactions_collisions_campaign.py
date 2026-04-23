from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import apply_bout_overrides, load_bout_input
from ..native.mesh import build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..native.recycling_atomic import hydrogen_cx_sigmav, load_openadas_rate
from ..native.recycling_collisions import (
    compute_collision_frequencies,
    electron_density,
    ion_parallel_viscosity_inputs,
)
from ..native.recycling_1d import (
    _initialize_species,
    _prepare_open_field_states,
)
from ..native.recycling_reactions import (
    charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
    reaction_sources,
)
from ..native.units import resolved_dataset_scalars
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


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


def _profile_payload(context: dict[str, object], values: np.ndarray, *, x_index: int | None = None, z_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mesh = context["mesh"]
    xi = mesh.xstart if x_index is None else int(x_index)
    active_slice = slice(mesh.ystart, mesh.yend + 1)
    coordinates = np.asarray(mesh.y[active_slice], dtype=np.float64)
    profile = np.asarray(values[xi, active_slice, z_index], dtype=np.float64)
    return coordinates, profile


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

    profiles = _build_reactions_collisions_profiles(
        single_species=build_reactions_collisions_context(resolved_single_species_input),
        multispecies=build_reactions_collisions_context(resolved_multispecies_input),
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
        "profiles": {
            name: {
                "coordinate_name": payload["coordinate_name"],
                "coordinate": [float(value) for value in payload["coordinate"]],
                "series": {
                    series_name: [float(value) for value in series_values]
                    for series_name, series_values in payload["series"].items()
                },
                "notes": payload["notes"],
            }
            for name, payload in profiles.items()
        },
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {
        "metric_values": np.asarray([metric.value for metric in metrics], dtype=np.float64),
        "metric_targets": np.asarray([metric.target for metric in metrics], dtype=np.float64),
        "metric_pass": np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64),
    }
    for name, payload in profiles.items():
        arrays_payload[f"{name}_coordinate"] = np.asarray(payload["coordinate"], dtype=np.float64)
        for series_name, series_values in payload["series"].items():
            arrays_payload[f"{name}_{series_name}"] = np.asarray(series_values, dtype=np.float64)
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_campaign_plot(metrics, profiles, plot_png_path)
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

    single_species = build_reactions_collisions_context(single_species_path)
    multispecies = build_reactions_collisions_context(multispecies_path)

    single_active = (single_species["mesh"].xstart, single_species["mesh"].ystart, 0)
    multispecies_active = (multispecies["mesh"].xstart, multispecies["mesh"].ystart, 0)

    single_cx_rates = charge_exchange_collision_rates(
        single_species["config"],
        species=single_species["species"],
        prepared=single_species["prepared"],
        dataset_scalars=single_species["dataset_scalars"],
    )
    multispecies_cx_rates = charge_exchange_collision_rates(
        multispecies["config"],
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_collision_rates = compute_collision_frequencies(
        multispecies["config"],
        multispecies["species"],
        multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_viscosity_inputs = ion_parallel_viscosity_inputs(
        species_name="d+",
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        collision_rates=multispecies_collision_rates,
        cx_rates=multispecies_cx_rates,
    )
    ionisation_rates = neutral_ionisation_collision_rates(
        single_species["config"],
        species=single_species["species"],
        prepared=single_species["prepared"],
        dataset_scalars=single_species["dataset_scalars"],
    )
    reaction_terms = reaction_sources(
        single_species["config"],
        species=single_species["species"],
        electron_density=electron_density(tuple(sp for sp in single_species["species"].values() if sp.charge > 0.0)),
        dataset_scalars=single_species["dataset_scalars"],
    )

    active_same = float(single_species["prepared"]["d+"].density[single_active] * hydrogen_cx_sigmav(
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

    d_same = float(multispecies["prepared"]["d+"].density[multispecies_active] * hydrogen_cx_sigmav(
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
    d_cross = float(multispecies["prepared"]["t+"].density[multispecies_active] * hydrogen_cx_sigmav(
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
    scaled_context = build_reactions_collisions_context(multispecies_path, config=scaled_config)
    scaled_cx_rates = charge_exchange_collision_rates(
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

    ionisation_coeffs, radiation_coeffs, log_temperature, log_density, electron_heating = load_openadas_rate("ne", "iz")
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


def build_reactions_collisions_context(path: Path, *, config=None) -> dict[str, object]:
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


def _build_reactions_collisions_profiles(
    *,
    single_species: dict[str, object],
    multispecies: dict[str, object],
) -> dict[str, dict[str, object]]:
    single_mesh = single_species["mesh"]
    single_active_slice = slice(single_mesh.ystart, single_mesh.yend + 1)
    multispecies_mesh = multispecies["mesh"]
    multispecies_active_slice = slice(multispecies_mesh.ystart, multispecies_mesh.yend + 1)

    single_electron_density = electron_density(tuple(sp for sp in single_species["species"].values() if sp.charge > 0.0))
    single_reaction_terms = reaction_sources(
        single_species["config"],
        species=single_species["species"],
        electron_density=single_electron_density,
        dataset_scalars=single_species["dataset_scalars"],
    )
    single_ionisation_rates = neutral_ionisation_collision_rates(
        single_species["config"],
        species=single_species["species"],
        prepared=single_species["prepared"],
        dataset_scalars=single_species["dataset_scalars"],
    )
    single_coordinate, single_ionisation_actual = _profile_payload(single_species, single_ionisation_rates["d"])
    _, single_ionisation_expected = _profile_payload(
        single_species,
        single_reaction_terms.diagnostics["Sd+_iz"] / np.maximum(single_species["species"]["d"].density, np.finfo(np.float64).tiny),
    )

    multispecies_cx_rates = charge_exchange_collision_rates(
        multispecies["config"],
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_collision_rates = compute_collision_frequencies(
        multispecies["config"],
        multispecies["species"],
        multispecies["prepared"],
        dataset_scalars=multispecies["dataset_scalars"],
    )
    multispecies_viscosity_inputs = ion_parallel_viscosity_inputs(
        species_name="d+",
        species=multispecies["species"],
        prepared=multispecies["prepared"],
        collision_rates=multispecies_collision_rates,
        cx_rates=multispecies_cx_rates,
    )

    d_same = multispecies["prepared"]["d+"].density * hydrogen_cx_sigmav(
        np.clip(
            (
                multispecies["prepared"]["d"].temperature / multispecies["species"]["d"].atomic_mass
                + multispecies["prepared"]["d+"].temperature / multispecies["species"]["d+"].atomic_mass
            ) * multispecies["dataset_scalars"]["Tnorm"],
            0.01,
            10000.0,
        ),
        multispecies["dataset_scalars"],
    )
    d_cross = multispecies["prepared"]["t+"].density * hydrogen_cx_sigmav(
        np.clip(
            (
                multispecies["prepared"]["d"].temperature / multispecies["species"]["d"].atomic_mass
                + multispecies["prepared"]["t+"].temperature / multispecies["species"]["t+"].atomic_mass
            ) * multispecies["dataset_scalars"]["Tnorm"],
            0.01,
            10000.0,
        ),
        multispecies["dataset_scalars"],
    )
    multispecies_coordinate, d_same_profile = _profile_payload(multispecies, d_same)
    _, d_cross_profile = _profile_payload(multispecies, d_cross)
    _, d_total_profile = _profile_payload(multispecies, multispecies_cx_rates["d"])

    expected_collisionality = np.zeros_like(multispecies["prepared"]["d+"].density, dtype=np.float64)
    for other_name in multispecies["species"]:
        rate = multispecies_collision_rates.get(("d+", other_name))
        if rate is not None:
            expected_collisionality = expected_collisionality + rate
    expected_collisionality = np.maximum(expected_collisionality + multispecies_cx_rates["d+"], 1.0e-12)
    _, collisionality_expected_profile = _profile_payload(multispecies, expected_collisionality)
    _, collisionality_actual_profile = _profile_payload(multispecies, multispecies_viscosity_inputs.total_collisionality)

    return {
        "ionisation_profile": {
            "coordinate_name": "normalized_parallel_coordinate",
            "coordinate": single_coordinate,
            "series": {
                "diagnostic_per_density": single_ionisation_expected,
                "assembled_collision_rate": single_ionisation_actual,
            },
            "notes": "Single-species ionisation rate profile compared against the reaction diagnostic normalized by neutral density.",
        },
        "d_atom_charge_exchange_profile": {
            "coordinate_name": "normalized_parallel_coordinate",
            "coordinate": multispecies_coordinate,
            "series": {
                "same_isotope_d_plus": d_same_profile,
                "cross_isotope_t_plus": d_cross_profile,
                "assembled_total": d_total_profile,
            },
            "notes": "Multispecies D neutral charge-exchange profile decomposed into same-isotope and cross-isotope ion contributions.",
        },
        "d_plus_collisionality_profile": {
            "coordinate_name": "normalized_parallel_coordinate",
            "coordinate": multispecies_coordinate,
            "series": {
                "assembled_total_collisionality": collisionality_actual_profile,
                "expected_collision_stack": collisionality_expected_profile,
            },
            "notes": "Total collisionality used by the ion-parallel-viscosity closure compared against the explicit assembled collision stack.",
        },
    }


def _save_campaign_plot(
    metrics: tuple[ReactionsCollisionsCampaignMetric, ...],
    profiles: dict[str, dict[str, object]],
    path: Path,
) -> None:
    if not profiles:
        figure, axis = plt.subplots(figsize=(12.0, 6.5), constrained_layout=True)
        x = np.arange(len(metrics), dtype=np.float64)
        labels = [metric.name.replace("_", "\n") for metric in metrics]
        margins = np.asarray([_metric_margin(metric) for metric in metrics], dtype=np.float64)
        margins = np.minimum(margins, 10.0)
        colors = ["#0a9396" if metric.passed else "#bb3e03" for metric in metrics]
        axis.bar(x, margins, color=colors, alpha=0.9, width=0.65)
        axis.axhline(1.0, color="#3a86ff", linestyle="--", linewidth=1.5)
        axis.set_xticks(x, labels)
        style_axis(
            axis,
            title="Verification margin to gate",
            ylabel="margin to gate (capped at 10×)",
            yscale="log",
        )
        figure.suptitle(
            "Reactions, collisions, and atomic-data verification campaign",
            fontsize=15.0,
            fontweight="semibold",
        )
        save_publication_figure(figure, path)
        return

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)

    axis = axes[0, 0]
    x = np.arange(len(metrics), dtype=np.float64)
    labels = [metric.name.replace("_", "\n") for metric in metrics]
    margins = np.asarray([_metric_margin(metric) for metric in metrics], dtype=np.float64)
    margins = np.minimum(margins, 10.0)
    colors = ["#0a9396" if metric.passed else "#bb3e03" for metric in metrics]
    axis.bar(x, margins, color=colors, alpha=0.9, width=0.65)
    axis.axhline(1.0, color="#3a86ff", linestyle="--", linewidth=1.5)
    axis.set_xticks(x, labels)
    style_axis(
        axis,
        title="Verification margin to gate",
        ylabel="margin to gate (capped at 10×)",
        yscale="log",
    )

    ionisation = profiles["ionisation_profile"]
    axis = axes[0, 1]
    ion_expected = np.asarray(ionisation["series"]["diagnostic_per_density"], dtype=np.float64)
    ion_actual = np.asarray(ionisation["series"]["assembled_collision_rate"], dtype=np.float64)
    ion_values = np.asarray([float(np.mean(ion_expected)), float(np.mean(ion_actual))], dtype=np.float64)
    ion_x = np.arange(2, dtype=np.float64)
    axis.bar(ion_x, ion_values, color=["#1d3557", "#e76f51"], width=0.6)
    axis.set_xticks(ion_x, ["diagnostic / density", "assembled rate"])
    style_axis(axis, title="Ionisation rate agreement", ylabel="mean rate")
    annotate_bars(axis, ion_x, ion_values, fmt="{:.3e}")
    ion_rel = float(np.max(np.abs(ion_actual - ion_expected) / np.maximum(np.abs(ion_expected), np.finfo(np.float64).tiny)))
    axis.text(
        0.03,
        0.95,
        f"max relative residual = {ion_rel:.2e}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.0,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    charge_exchange = profiles["d_atom_charge_exchange_profile"]
    axis = axes[1, 0]
    same = np.asarray(charge_exchange["series"]["same_isotope_d_plus"], dtype=np.float64)
    cross = np.asarray(charge_exchange["series"]["cross_isotope_t_plus"], dtype=np.float64)
    total = np.asarray(charge_exchange["series"]["assembled_total"], dtype=np.float64)
    same_mean = float(np.mean(same))
    cross_mean = float(np.mean(cross))
    total_mean = float(np.mean(total))
    axis.bar([0.0], [same_mean], color="#2a9d8f", width=0.55, label="same-isotope D+")
    axis.bar([0.0], [cross_mean], bottom=[same_mean], color="#f4a261", width=0.55, label="cross-isotope T+")
    axis.plot([0.0], [total_mean], marker="_", markersize=26, color="#264653", linewidth=0.0, label="assembled total")
    axis.set_xticks([0.0], ["D neutral active-profile mean"])
    style_axis(axis, title="D neutral charge exchange decomposition", ylabel="collision frequency")
    annotate_bars(axis, np.asarray([0.0]), np.asarray([same_mean + cross_mean]), fmt="{:.3e}", fontsize=8.5)
    cross_fraction = cross_mean / max(same_mean + cross_mean, np.finfo(np.float64).tiny)
    axis.text(
        0.03,
        0.95,
        f"cross-isotope fraction = {cross_fraction:.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.0,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    collisionality = profiles["d_plus_collisionality_profile"]
    axis = axes[1, 1]
    coll_expected = np.asarray(collisionality["series"]["expected_collision_stack"], dtype=np.float64)
    coll_actual = np.asarray(collisionality["series"]["assembled_total_collisionality"], dtype=np.float64)
    coll_values = np.asarray([float(np.mean(coll_expected)), float(np.mean(coll_actual))], dtype=np.float64)
    coll_x = np.arange(2, dtype=np.float64)
    axis.bar(coll_x, coll_values, color=["#6a4c93", "#1982c4"], width=0.6)
    axis.set_xticks(coll_x, ["expected stack", "closure input"])
    style_axis(axis, title="Ion parallel viscosity collisionality", ylabel="mean collisionality")
    annotate_bars(axis, coll_x, coll_values, fmt="{:.3e}")
    coll_rel = float(
        np.max(
            np.abs(coll_actual - coll_expected)
            / np.maximum(np.abs(coll_expected), np.finfo(np.float64).tiny)
        )
    )
    axis.text(
        0.03,
        0.95,
        f"max relative residual = {coll_rel:.2e}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.0,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    figure.suptitle(
        "Reactions, collisions, and atomic-data verification campaign",
        fontsize=15.0,
        fontweight="semibold",
    )
    save_publication_figure(figure, path)


def _metric_margin(metric: ReactionsCollisionsCampaignMetric) -> float:
    tiny = np.finfo(np.float64).tiny
    if metric.kind in {"relative_error", "max_value"}:
        return metric.target / max(metric.value, tiny)
    return metric.value / max(metric.target, tiny)
