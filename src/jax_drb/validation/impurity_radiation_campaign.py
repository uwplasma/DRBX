from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..native import run_curated_case
from ..native.recycling_1d import _load_openadas_rate, _openadas_energy_loss, _openadas_reaction_rate
from ..parity.arrays import load_portable_array_payload
from ..reference.paths import require_reference_root

_REFERENCE_ARRAY_BASELINE_DIR = Path(__file__).resolve().parents[3] / "references" / "baselines" / "reference_arrays"


@dataclass(frozen=True)
class ImpurityRadiationCampaignMetric:
    name: str
    kind: str
    value: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class ImpurityRadiationCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_impurity_radiation_campaign_package(
    *,
    output_root: str | Path,
    reference_root: str | Path | None = None,
    case_label: str = "impurity_radiation_campaign",
) -> ImpurityRadiationCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_impurity_radiation_campaign(reference_root=reference_root)
    summary_payload = {
        "family": "impurity_radiation_and_detachment_control",
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

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        metric_values=np.asarray([metric.value for metric in metrics], dtype=np.float64),
        metric_targets=np.asarray([metric.target for metric in metrics], dtype=np.float64),
        metric_pass=np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64),
    )

    plot_png_path = images_dir / f"{case_label}.png"
    _save_campaign_plot(metrics, plot_png_path)
    return ImpurityRadiationCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_impurity_radiation_campaign(
    *,
    reference_root: str | Path | None = None,
) -> tuple[ImpurityRadiationCampaignMetric, ...]:
    resolved_reference_root = Path(reference_root) if reference_root is not None else require_reference_root()

    ionisation_coeffs, ionisation_radiation, log_temperature, log_density, electron_heating_iz = _load_openadas_rate("ne", "iz")
    recombination_coeffs, recombination_radiation, _, _, electron_heating_rec = _load_openadas_rate("ne", "rec")
    finite_fraction = float(
        (
            np.isfinite(ionisation_coeffs).mean()
            + np.isfinite(ionisation_radiation).mean()
            + np.isfinite(recombination_coeffs).mean()
            + np.isfinite(recombination_radiation).mean()
            + np.isfinite(log_temperature).mean()
            + np.isfinite(log_density).mean()
            + float(np.isfinite(electron_heating_iz))
            + float(np.isfinite(electron_heating_rec))
        )
        / 8.0
    )

    sample_density = np.asarray([2.5e-3], dtype=np.float64)
    sample_electron_density = np.asarray([3.5e-3], dtype=np.float64)
    sample_electron_temperature = np.asarray([4.0e-2], dtype=np.float64)
    dataset_scalars = {"Nnorm": 1.0e19, "Tnorm": 100.0, "Omega_ci": 1.0e6}
    ionisation_rate = float(
        _openadas_reaction_rate(
            sample_density,
            sample_electron_density,
            sample_electron_temperature,
            "ne",
            "iz",
            dataset_scalars,
        )[0]
    )
    recombination_rate = float(
        _openadas_reaction_rate(
            sample_density,
            sample_electron_density,
            sample_electron_temperature,
            "ne",
            "rec",
            dataset_scalars,
        )[0]
    )
    ionisation_radiation_loss = float(
        _openadas_energy_loss(
            sample_density,
            sample_electron_density,
            sample_electron_temperature,
            "ne",
            "iz",
            reaction_rate=np.asarray([ionisation_rate], dtype=np.float64),
            dataset_scalars=dataset_scalars,
        )[0]
    )
    recombination_radiation_loss = float(
        _openadas_energy_loss(
            sample_density,
            sample_electron_density,
            sample_electron_temperature,
            "ne",
            "rec",
            reaction_rate=np.asarray([recombination_rate], dtype=np.float64),
            dataset_scalars=dataset_scalars,
        )[0]
    )

    expected_payload = load_portable_array_payload(_REFERENCE_ARRAY_BASELINE_DIR / "tokamak_recycling_dthene_rhs.npz")
    actual_payload = run_curated_case("tokamak_recycling_dthene_rhs", reference_root=resolved_reference_root)
    expected_neon_density = np.asarray(expected_payload["variables"]["Nne+"], dtype=np.float64)
    actual_neon_density = np.asarray(actual_payload.variables["Nne+"], dtype=np.float64)
    expected_neon_pressure = np.asarray(expected_payload["variables"]["Pne+"], dtype=np.float64)
    actual_neon_pressure = np.asarray(actual_payload.variables["Pne+"], dtype=np.float64)
    expected_electron_pressure = np.asarray(expected_payload["variables"]["Pe"], dtype=np.float64)
    actual_electron_pressure = np.asarray(actual_payload.variables["Pe"], dtype=np.float64)
    neon_density_error = float(np.max(np.abs(actual_neon_density - expected_neon_density)))
    neon_pressure_error = float(np.max(np.abs(actual_neon_pressure - expected_neon_pressure)))
    electron_pressure_error = float(np.max(np.abs(actual_electron_pressure - expected_electron_pressure)))

    return (
        ImpurityRadiationCampaignMetric(
            name="openadas_neon_full_bundle_finite_fraction",
            kind="fraction",
            value=finite_fraction,
            target=1.0,
            passed=np.isclose(finite_fraction, 1.0, rtol=0.0, atol=0.0),
            notes="Neon OpenADAS ionisation/recombination plus radiation tables load fully and finitely.",
        ),
        ImpurityRadiationCampaignMetric(
            name="neon_ionisation_rate_positive",
            kind="scalar",
            value=ionisation_rate,
            target=0.0,
            passed=ionisation_rate > 0.0,
            notes="Neon ionisation reaction rate is finite and positive on a representative edge sample state.",
        ),
        ImpurityRadiationCampaignMetric(
            name="neon_recombination_rate_positive",
            kind="scalar",
            value=recombination_rate,
            target=0.0,
            passed=recombination_rate > 0.0,
            notes="Neon recombination reaction rate is finite and positive on a representative edge sample state.",
        ),
        ImpurityRadiationCampaignMetric(
            name="neon_openadas_radiation_terms_finite",
            kind="fraction",
            value=float(np.isfinite([ionisation_radiation_loss, recombination_radiation_loss]).mean()),
            target=1.0,
            passed=np.isfinite(ionisation_radiation_loss) and np.isfinite(recombination_radiation_loss),
            notes="Ionisation and recombination energy-loss channels are finite on the native OpenADAS evaluation path.",
        ),
        ImpurityRadiationCampaignMetric(
            name="tokamak_dthene_rhs_neon_density_exact",
            kind="max_abs_error",
            value=neon_density_error,
            target=1.0e-12,
            passed=neon_density_error <= 1.0e-12,
            notes="Direct tokamak D/T/He/Ne RHS rung matches the committed neon density baseline exactly.",
        ),
        ImpurityRadiationCampaignMetric(
            name="tokamak_dthene_rhs_neon_pressure_exact",
            kind="max_abs_error",
            value=neon_pressure_error,
            target=1.0e-12,
            passed=neon_pressure_error <= 1.0e-12,
            notes="Direct tokamak D/T/He/Ne RHS rung matches the committed neon pressure baseline exactly.",
        ),
        ImpurityRadiationCampaignMetric(
            name="tokamak_dthene_rhs_electron_pressure_exact",
            kind="max_abs_error",
            value=electron_pressure_error,
            target=1.0e-12,
            passed=electron_pressure_error <= 1.0e-12,
            notes="Electron pressure remains exact on the same neon-enabled direct tokamak RHS compare surface.",
        ),
    )


def _save_campaign_plot(metrics: tuple[ImpurityRadiationCampaignMetric, ...], path: Path) -> None:
    labels = [metric.name.replace("_", "\n") for metric in metrics]
    values = [metric.value for metric in metrics]
    colors = ["#0a9396" if metric.passed else "#bb3e03" for metric in metrics]
    figure, axis = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    x = np.arange(len(metrics))
    axis.bar(x, values, color=colors, alpha=0.92)
    axis.set_xticks(x, labels)
    axis.set_ylabel("metric value")
    axis.set_title("Impurity / radiation validation campaign")
    axis.grid(alpha=0.25, axis="y")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
