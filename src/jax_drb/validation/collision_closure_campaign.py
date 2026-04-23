from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..config.boutinp import apply_bout_overrides, load_bout_input
from ..native.reference_dump import load_local_reference_snapshot_cache
from ..native.recycling_1d import _prepare_open_field_states
from ..native.recycling_collision_closure import (
    apply_collision_closure,
    conduction_collision_time,
)
from ..native.recycling_collisions import compute_collision_frequencies
from ..native.recycling_reactions import charge_exchange_collision_rates
from ..native.recycling_setup import initialize_species
from ..reference.paths import require_reference_root
from ..runtime.run_config import RunConfiguration
from ..native.units import resolved_dataset_scalars


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
_ION_SPECIES = ("d+", "t+", "he+")
_PRESSURE_SPECIES = ("e", "d+", "t+", "he+", "d", "t", "he")
_FRICTION_PAIRS = (("d+", "t+"), ("d+", "e"), ("t+", "e"), ("he+", "e"))


@dataclass(frozen=True)
class CollisionClosureCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class CollisionClosureCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class _CollisionClosureContext:
    input_path: Path
    snapshot_path: Path
    config: object
    mesh: object
    metrics: object
    dataset_scalars: dict[str, float]
    species: dict[str, object]
    prepared: dict[str, object]
    collision_rates: dict[tuple[str, str], np.ndarray]
    charge_exchange_rates: dict[str, np.ndarray]
    closure_terms: object
    conduction_times: dict[str, np.ndarray]


def create_collision_closure_campaign_package(
    *,
    output_root: str | Path,
    input_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    case_label: str = "collision_closure_campaign",
) -> CollisionClosureCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_path, resolved_snapshot_path = _resolve_paths(input_path=input_path, snapshot_path=snapshot_path)
    context = build_collision_closure_context(
        input_path=resolved_input_path,
        snapshot_path=resolved_snapshot_path,
    )
    metrics = build_collision_closure_campaign(
        input_path=resolved_input_path,
        snapshot_path=resolved_snapshot_path,
    )
    summaries = _build_collision_closure_summaries(context)

    summary_payload = {
        "family": "collision_closure",
        "input_name": resolved_input_path.name,
        "snapshot_name": resolved_snapshot_path.name,
        "literature_anchor": {
            "transport_review": "Braginskii 1965 transport formulas for collisional magnetized plasmas",
            "implementation_anchor": "Hermes-3 multi-component edge/SOL collisional closure family",
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
        "ion_species_order": list(summaries["ion_species_order"]),
        "pressure_species_order": list(summaries["pressure_species_order"]),
        "friction_pair_order": list(summaries["friction_pair_order"]),
        "summaries": {
            name: [float(value) for value in values]
            for name, values in summaries.items()
            if not name.endswith("_order")
        },
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_payload = {}
    for name, values in summaries.items():
        if name.endswith("_order"):
            continue
        arrays_payload[name] = np.asarray(values, dtype=np.float64)
    arrays_payload["metric_values"] = np.asarray([metric.value for metric in metrics], dtype=np.float64)
    arrays_payload["metric_targets"] = np.asarray([metric.target for metric in metrics], dtype=np.float64)
    arrays_payload["metric_pass"] = np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64)
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays_payload)

    plot_png_path = images_dir / f"{case_label}.png"
    _save_collision_closure_plot(metrics, summaries, plot_png_path)
    return CollisionClosureCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_collision_closure_campaign(
    *,
    input_path: str | Path,
    snapshot_path: str | Path,
) -> tuple[CollisionClosureCampaignMetric, ...]:
    context = build_collision_closure_context(input_path=input_path, snapshot_path=snapshot_path)
    summaries = _build_collision_closure_summaries(context)
    pressure_finite_fraction = float(np.mean(np.isfinite(np.asarray(summaries["conduction_time_active_point"], dtype=np.float64))))
    return (
        *(
            CollisionClosureCampaignMetric(
                name=f"{name.replace('+', '_plus')}_ion_viscosity_peak",
                kind="min_value",
                value=float(value),
                target=1.0e-8,
                passed=bool(value > 1.0e-8),
                notes=f"{name} should retain active parallel ion-viscosity forcing on the prepared recycling state.",
            )
            for name, value in zip(_ION_SPECIES, summaries["ion_viscosity_peak"], strict=True)
        ),
        *(
            CollisionClosureCampaignMetric(
                name=f"{_pair_label(pair)}_action_reaction_residual",
                kind="max_value",
                value=float(value),
                target=1.0e-12,
                passed=bool(value <= 1.0e-12),
                notes=f"The collisional friction diagnostics for pair {_pair_display(pair)} should satisfy action-reaction balance.",
            )
            for pair, value in zip(_FRICTION_PAIRS, summaries["friction_action_reaction_residual"], strict=True)
        ),
        CollisionClosureCampaignMetric(
            name="pressure_species_conduction_time_finite_fraction",
            kind="fraction",
            value=pressure_finite_fraction,
            target=1.0,
            passed=bool(pressure_finite_fraction == 1.0),
            notes="All pressure-evolving species should retain finite conduction collision times on the prepared state.",
        ),
    )


def build_collision_closure_context(
    *,
    input_path: str | Path,
    snapshot_path: str | Path,
) -> _CollisionClosureContext:
    path = Path(input_path)
    config = apply_bout_overrides(
        load_bout_input(path),
        (
            "model:components=braginskii_friction,braginskii_heat_exchange,braginskii_thermal_force,braginskii_ion_viscosity,braginskii_conduction",
            "braginskii_thermal_force:override_ion_mass_restrictions=true",
            "e:conduction_collisions_mode=multispecies",
            "d+:conduction_collisions_mode=multispecies",
            "t+:conduction_collisions_mode=multispecies",
            "he+:conduction_collisions_mode=multispecies",
            "d:conduction_collisions_mode=multispecies",
            "t:conduction_collisions_mode=multispecies",
            "he:conduction_collisions_mode=multispecies",
        ),
    )
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot_cache(
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
    cx_rates = charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    closure_terms = apply_collision_closure(
        config,
        species,
        prepared,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
    )
    conduction_times = {
        name: conduction_collision_time(
            config,
            species=species,
            prepared=prepared,
            collision_rates=collision_rates,
            cx_rates=cx_rates,
            species_name=name,
        )
        for name in _PRESSURE_SPECIES
    }
    return _CollisionClosureContext(
        input_path=path,
        snapshot_path=Path(snapshot_path),
        config=config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        species=species,
        prepared=prepared,
        collision_rates=collision_rates,
        charge_exchange_rates=cx_rates,
        closure_terms=closure_terms,
        conduction_times=conduction_times,
    )


def _resolve_paths(
    *,
    input_path: str | Path | None,
    snapshot_path: str | Path | None,
) -> tuple[Path, Path]:
    if input_path is not None:
        resolved_input = Path(input_path)
    else:
        reference_root = require_reference_root()
        resolved_input = reference_root / "tests" / "integrated" / "1D-recycling-dthe" / "data" / "BOUT.inp"
    resolved_snapshot = Path(snapshot_path) if snapshot_path is not None else _SNAPSHOT_CACHE_PATH
    return resolved_input, resolved_snapshot


def _active_max(context: _CollisionClosureContext, values: np.ndarray) -> float:
    mesh = context.mesh
    active = np.asarray(values[mesh.xstart : mesh.xend + 1, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    return float(np.max(np.abs(active)))


def _domain_max(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _active_point(context: _CollisionClosureContext, values: np.ndarray) -> float:
    mesh = context.mesh
    return float(np.asarray(values, dtype=np.float64)[mesh.xstart, mesh.ystart, 0])


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0].replace('+', '_plus')}_{pair[1].replace('+', '_plus')}"


def _pair_display(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _build_collision_closure_summaries(context: _CollisionClosureContext) -> dict[str, object]:
    return {
        "ion_species_order": _ION_SPECIES,
        "pressure_species_order": _PRESSURE_SPECIES,
        "friction_pair_order": _FRICTION_PAIRS,
        "ion_viscosity_peak": [
            _active_max(context, context.closure_terms.diagnostics[f"DivPiPar_{name}"])
            for name in _ION_SPECIES
        ],
        "conduction_time_active_point": [
            _active_point(context, context.conduction_times[name])
            for name in _PRESSURE_SPECIES
        ],
        "friction_pair_peak": [
            _domain_max(context.closure_terms.diagnostics[f"F{left_name}{right_name}_coll"])
            for pair in _FRICTION_PAIRS
            for left_name, right_name in (pair,)
        ],
        "friction_action_reaction_residual": [
            _domain_max(
                np.asarray(context.closure_terms.diagnostics[f"F{left_name}{right_name}_coll"], dtype=np.float64)
                + np.asarray(
                    context.closure_terms.diagnostics[f"F{right_name}{left_name}_coll"],
                    dtype=np.float64,
                ),
            )
            for pair in _FRICTION_PAIRS
            for left_name, right_name in (pair,)
        ],
        "energy_exchange_peak": [
            _domain_max(context.closure_terms.diagnostics[f"E{left_name}{right_name}_coll_friction"])
            for pair in _FRICTION_PAIRS
            for left_name, right_name in (pair,)
        ],
    }


def _save_collision_closure_plot(
    metrics: tuple[CollisionClosureCampaignMetric, ...],
    summaries: dict[str, object],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ion_species = list(summaries["ion_species_order"])
    pressure_species = list(summaries["pressure_species_order"])
    friction_pairs = list(summaries["friction_pair_order"])

    axes[0, 0].bar(
        np.arange(len(ion_species), dtype=np.float64),
        np.asarray(summaries["ion_viscosity_peak"], dtype=np.float64),
        color="#1f77b4",
        width=0.6,
    )
    axes[0, 0].set_title("Ion viscosity activity")
    axes[0, 0].set_ylabel(r"$\max |\nabla_\parallel \Pi_{\parallel,i}|$")
    axes[0, 0].set_xticks(np.arange(len(ion_species), dtype=np.float64), [name.upper() for name in ion_species])
    axes[0, 0].grid(alpha=0.25, axis="y")

    axes[0, 1].bar(
        np.arange(len(friction_pairs), dtype=np.float64),
        np.asarray(summaries["friction_pair_peak"], dtype=np.float64),
        color="#d62728",
        width=0.6,
    )
    axes[0, 1].set_title("Collisional friction activity")
    axes[0, 1].set_ylabel(r"$\max |F_{ab}^{\mathrm{coll}}|$")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(
        np.arange(len(friction_pairs), dtype=np.float64),
        [_pair_display(label) for label in friction_pairs],
        rotation=20,
    )
    axes[0, 1].grid(alpha=0.25, axis="y")

    axes[1, 0].bar(
        np.arange(len(pressure_species), dtype=np.float64),
        np.asarray(summaries["conduction_time_active_point"], dtype=np.float64),
        color="#2ca02c",
        width=0.6,
    )
    axes[1, 0].set_title("Active-point conduction time")
    axes[1, 0].set_ylabel(r"$\tau_{\mathrm{cond}}$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(np.arange(len(pressure_species), dtype=np.float64), [name.upper() for name in pressure_species], rotation=20)
    axes[1, 0].grid(alpha=0.25, axis="y")

    axes[1, 1].bar(
        np.arange(len(friction_pairs), dtype=np.float64),
        np.asarray(summaries["energy_exchange_peak"], dtype=np.float64),
        color="#9467bd",
        width=0.6,
    )
    axes[1, 1].set_title("Frictional heating activity")
    axes[1, 1].set_ylabel(r"$\max |Q_{ab}^{\mathrm{fric}}|$")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(
        np.arange(len(friction_pairs), dtype=np.float64),
        [_pair_display(label) for label in friction_pairs],
        rotation=20,
    )
    axes[1, 1].grid(alpha=0.25, axis="y")

    passed = sum(1 for metric in metrics if metric.passed)
    figure.suptitle(
        f"Collision closure activity on a prepared D/T/He recycling state ({passed}/{len(metrics)} metrics passing)",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
