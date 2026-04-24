from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..native import run_curated_case
from ..reference.paths import default_reference_root, repo_root
from .publication_plotting import save_publication_figure, style_axis


DEFAULT_TOKAMAK_RECYCLING_OBSERVABLE_CASE = "tokamak_recycling_dthe_one_step"
_ION_SPECIES = ("d", "t", "he")
_ION_LABELS = {"d": "D", "t": "T", "he": "He"}
_SPECIES_COLORS = {"d": "#005f73", "t": "#9b2226", "he": "#ca6702"}


@dataclass(frozen=True)
class TokamakRecyclingObservableCampaignArtifacts:
    report_json_path: Path
    report_npz_path: Path
    report_plot_png_path: Path


def create_tokamak_recycling_observable_campaign_package(
    *,
    output_root: str | Path,
    case_name: str = DEFAULT_TOKAMAK_RECYCLING_OBSERVABLE_CASE,
    case_label: str = "tokamak_recycling_observable_campaign",
    reference_arrays_npz: str | Path | None = None,
    native_arrays_npz: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> TokamakRecyclingObservableCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_tokamak_recycling_observable_campaign_report(
        case_name=case_name,
        reference_arrays_npz=reference_arrays_npz,
        native_arrays_npz=native_arrays_npz,
        reference_root=reference_root,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_npz_path = _write_tokamak_recycling_observable_arrays(report, data_dir / f"{case_label}.npz")
    report_plot_png_path = save_tokamak_recycling_observable_campaign_plot(
        report,
        images_dir / f"{case_label}.png",
    )
    return TokamakRecyclingObservableCampaignArtifacts(
        report_json_path=report_json_path,
        report_npz_path=report_npz_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_tokamak_recycling_observable_campaign_report(
    *,
    case_name: str = DEFAULT_TOKAMAK_RECYCLING_OBSERVABLE_CASE,
    reference_arrays_npz: str | Path | None = None,
    native_arrays_npz: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> dict[str, object]:
    reference_path = (
        Path(reference_arrays_npz)
        if reference_arrays_npz is not None
        else repo_root() / "references" / "baselines" / "reference_arrays" / f"{case_name}.npz"
    )
    reference_payload = _load_array_payload(reference_path)
    native_payload, native_source = _load_or_run_native_payload(
        case_name=case_name,
        native_arrays_npz=native_arrays_npz,
        reference_root=reference_root,
    )
    metadata = dict(reference_payload.get("metadata", {}))
    time_points = tuple(float(value) for value in metadata.get("time_points", range(_first_field(reference_payload).shape[0])))
    target_indices = {"lower": 0, "upper": int(_first_field(reference_payload).shape[2] - 1)}

    profiles: dict[str, Any] = {
        "x_indices": list(range(int(_first_field(reference_payload).shape[1]))),
        "y_indices": list(range(int(_first_field(reference_payload).shape[2]))),
        "target_indices": target_indices,
        "species": {},
    }
    metrics: list[dict[str, object]] = []
    for species in _ION_SPECIES:
        species_payload = _build_species_observables(
            species,
            reference_payload=reference_payload,
            native_payload=native_payload,
            target_indices=target_indices,
        )
        profiles["species"][species] = species_payload
        metrics.extend(_species_metrics(species, species_payload))

    electron_temperature = _build_electron_temperature_observable(
        reference_payload=reference_payload,
        native_payload=native_payload,
        target_indices=target_indices,
    )
    profiles["electron_temperature"] = electron_temperature
    metrics.extend(_profile_metrics("electron_temperature", electron_temperature, target_indices=target_indices))

    return {
        "case_name": case_name,
        "reference_code": "hermes-3",
        "reference_arrays_npz": _public_path(reference_path),
        "native_source": native_source,
        "time_points": list(time_points),
        "final_time": float(time_points[-1]) if time_points else None,
        "component_labels": list(metadata.get("component_labels", [])),
        "literature_anchor": {
            "tcv_x21": (
                "Oliveira and Body et al. introduced the TCV-X21 reference case "
                "as a profile- and target-observable validation surface for edge turbulence codes."
            ),
            "solps_iter_tcv_x21": (
                "Wang et al. extended TCV-X21 validation with neutral observables and ionisation-source interpretation."
            ),
            "hermes_3_tcv_x21": (
                "Recent Hermes-3 TCV-X21 work emphasizes target-profile shifts, divertor profile agreement, "
                "and the role of missing or simplified neutral dynamics."
            ),
        },
        "observable_contract": {
            "target_density_profiles": "final charged-species density profiles at lower and upper target-index rows",
            "target_momentum_flux_proxy": "final |NV_s+| profiles at target-index rows; normalization follows the committed benchmark arrays",
            "neutral_parallel_profiles": "final neutral density averaged over radial and toroidal indices as a parallel-coordinate buildup proxy",
            "electron_temperature_proxy": "final Pe divided by summed charged density at target-index rows",
        },
        "profiles": profiles,
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if bool(metric["passed"])),
        "metrics": metrics,
    }


def save_tokamak_recycling_observable_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    profiles = report["profiles"]
    x = np.asarray(profiles["x_indices"], dtype=np.float64)
    y = np.asarray(profiles["y_indices"], dtype=np.float64)
    species_profiles = profiles["species"]

    figure, axes = plt.subplots(2, 2, figsize=(14.4, 9.4))
    lower_key = "lower"
    upper_key = "upper"

    for species in _ION_SPECIES:
        payload = species_profiles[species]
        color = _SPECIES_COLORS[species]
        label = _ION_LABELS[species]
        axes[0, 0].plot(
            x,
            np.asarray(payload["target_density"][lower_key]["reference"], dtype=np.float64),
            color=color,
            linewidth=2.0,
            label=f"{label} Hermès",
        )
        axes[0, 0].plot(
            x,
            np.asarray(payload["target_density"][lower_key]["native"], dtype=np.float64),
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"{label} JAXDRB",
        )
        axes[0, 1].plot(
            x,
            np.asarray(payload["target_flux_proxy"][upper_key]["reference"], dtype=np.float64),
            color=color,
            linewidth=2.0,
            label=f"{label} Hermès",
        )
        axes[0, 1].plot(
            x,
            np.asarray(payload["target_flux_proxy"][upper_key]["native"], dtype=np.float64),
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"{label} JAXDRB",
        )
        axes[1, 0].plot(
            y,
            np.asarray(payload["neutral_parallel_density"]["reference"], dtype=np.float64),
            color=color,
            linewidth=2.0,
            label=f"{label} Hermès",
        )
        axes[1, 0].plot(
            y,
            np.asarray(payload["neutral_parallel_density"]["native"], dtype=np.float64),
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"{label} JAXDRB",
        )

    style_axis(
        axes[0, 0],
        title="Lower target-index charged-density profiles",
        xlabel="radial index",
        ylabel="normalized density",
        grid="both",
    )
    style_axis(
        axes[0, 1],
        title="Upper target-index momentum-flux proxy",
        xlabel="radial index",
        ylabel=r"$|NV_{s+}|$",
        yscale="log",
        grid="both",
    )
    style_axis(
        axes[1, 0],
        title="Neutral buildup along the parallel index",
        xlabel="parallel index",
        ylabel="mean neutral density",
        yscale="log",
        grid="both",
    )

    metric_names = [
        "D density",
        "T density",
        "He density",
        "D lower flux",
        "T lower flux",
        "He lower flux",
        "upper flux max",
        "Te proxy",
    ]
    metric_values = np.asarray(
        [
            _metric_value(report, "d_target_density_max_relative_error"),
            _metric_value(report, "t_target_density_max_relative_error"),
            _metric_value(report, "he_target_density_max_relative_error"),
            _metric_value(report, "d_target_flux_proxy_lower_max_relative_error"),
            _metric_value(report, "t_target_flux_proxy_lower_max_relative_error"),
            _metric_value(report, "he_target_flux_proxy_lower_max_relative_error"),
            max(
                _metric_value(report, "d_target_flux_proxy_upper_max_relative_error"),
                _metric_value(report, "t_target_flux_proxy_upper_max_relative_error"),
                _metric_value(report, "he_target_flux_proxy_upper_max_relative_error"),
            ),
            _metric_value(report, "electron_temperature_upper_max_relative_error"),
        ],
        dtype=np.float64,
    )
    bar_y = np.arange(metric_values.size)
    axes[1, 1].barh(
        bar_y,
        np.maximum(metric_values, 1.0e-16),
        color=["#005f73", "#9b2226", "#ca6702", "#0a9396", "#bb3e03", "#ee9b00", "#001219", "#4d908e"],
    )
    axes[1, 1].set_yticks(bar_y, metric_names)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(5.0e-2, color="#9b2226", linestyle=":", linewidth=1.5, label="5e-2 gate")
    style_axis(
        axes[1, 1],
        title="Observable-level native/Hermès differences",
        xlabel="max relative error",
        xscale="log",
        grid="x",
    )
    for yi, value in zip(bar_y, metric_values, strict=True):
        axes[1, 1].text(
            float(max(value, 1.0e-16) * 1.12),
            float(yi),
            f"{value:.1e}",
            va="center",
            ha="left",
            fontsize=8.0,
        )
    axes[1, 1].legend(frameon=False, fontsize=8.0, loc="lower right")

    axes[0, 0].legend(frameon=False, fontsize=8.0, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=8.0, ncol=2)
    axes[1, 0].legend(frameon=False, fontsize=8.0, ncol=2)
    figure.suptitle(
        "Tokamak recycling observable campaign on the D/T/He direct-tokamak validation surface",
        fontsize=13.2,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.018,
        "Profiles follow the target and neutral-observable validation style used in TCV-X21, SOLPS-ITER, and Hermes-3 studies; indices are the active committed benchmark grid.",
        ha="center",
        va="center",
        fontsize=8.8,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.90, wspace=0.30, hspace=0.36)
    save_publication_figure(figure, target)
    return target


def _load_or_run_native_payload(
    *,
    case_name: str,
    native_arrays_npz: str | Path | None,
    reference_root: str | Path | None,
) -> tuple[dict[str, Any], str]:
    if native_arrays_npz is not None:
        return _load_array_payload(Path(native_arrays_npz)), _public_path(Path(native_arrays_npz))
    resolved_root = Path(reference_root).expanduser().resolve() if reference_root is not None else default_reference_root()
    if resolved_root is None:
        reference_path = repo_root() / "references" / "baselines" / "reference_arrays" / f"{case_name}.npz"
        return _load_array_payload(reference_path), "reference_arrays_self_check_no_reference_root"
    result = run_curated_case(case_name, reference_root=resolved_root)
    variables = {name: np.asarray(value, dtype=np.float64) for name, value in result.variables.items()}
    metadata = {
        "case_name": case_name,
        "producer": "jax_drb_native_run",
        "time_points": list(result.time_points),
        "component_labels": [request.label for request in result.run_config.components],
    }
    return {"metadata": metadata, "variables": variables}, f"native_run:{_public_path(resolved_root)}"


def _load_array_payload(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    metadata: dict[str, Any] = {}
    if "__metadata__" in payload.files:
        metadata = json.loads(str(payload["__metadata__"].item()))
    variables = {
        name.removeprefix("var__"): np.asarray(payload[name], dtype=np.float64)
        for name in payload.files
        if name.startswith("var__")
    }
    if not variables:
        raise ValueError(f"No var__ arrays found in {path}")
    return {"metadata": metadata, "variables": variables}


def _build_species_observables(
    species: str,
    *,
    reference_payload: dict[str, Any],
    native_payload: dict[str, Any],
    target_indices: dict[str, int],
) -> dict[str, object]:
    charged_density = f"N{species}+"
    charged_momentum = f"NV{species}+"
    neutral_density = f"N{species}"
    reference_variables = reference_payload["variables"]
    native_variables = native_payload["variables"]
    return {
        "target_density": {
            target_name: _profile_pair(
                reference_variables[charged_density],
                native_variables[charged_density],
                y_index=y_index,
            )
            for target_name, y_index in target_indices.items()
        },
        "target_flux_proxy": {
            target_name: _profile_pair(
                np.abs(reference_variables[charged_momentum]),
                np.abs(native_variables[charged_momentum]),
                y_index=y_index,
            )
            for target_name, y_index in target_indices.items()
        },
        "neutral_parallel_density": {
            "reference": _parallel_profile(reference_variables[neutral_density]).tolist(),
            "native": _parallel_profile(native_variables[neutral_density]).tolist(),
        },
        "integrals": {
            "reference_neutral_inventory": float(np.sum(np.asarray(reference_variables[neutral_density][-1], dtype=np.float64))),
            "native_neutral_inventory": float(np.sum(np.asarray(native_variables[neutral_density][-1], dtype=np.float64))),
            "reference_target_flux_proxy_total": float(
                sum(np.sum(np.abs(reference_variables[charged_momentum][-1, :, y_index, :])) for y_index in target_indices.values())
            ),
            "native_target_flux_proxy_total": float(
                sum(np.sum(np.abs(native_variables[charged_momentum][-1, :, y_index, :])) for y_index in target_indices.values())
            ),
        },
    }


def _build_electron_temperature_observable(
    *,
    reference_payload: dict[str, Any],
    native_payload: dict[str, Any],
    target_indices: dict[str, int],
) -> dict[str, object]:
    reference_temperature = _electron_temperature_proxy(reference_payload["variables"])
    native_temperature = _electron_temperature_proxy(native_payload["variables"])
    return {
        target_name: _profile_pair(reference_temperature, native_temperature, y_index=y_index)
        for target_name, y_index in target_indices.items()
    }


def _electron_temperature_proxy(variables: dict[str, np.ndarray]) -> np.ndarray:
    density = np.zeros_like(variables["Pe"], dtype=np.float64)
    for species in _ION_SPECIES:
        density = density + np.asarray(variables[f"N{species}+"], dtype=np.float64)
    return np.asarray(variables["Pe"], dtype=np.float64) / np.maximum(density, 1.0e-30)


def _profile_pair(reference: np.ndarray, native: np.ndarray, *, y_index: int) -> dict[str, list[float]]:
    return {
        "reference": _target_profile(reference, y_index=y_index).tolist(),
        "native": _target_profile(native, y_index=y_index).tolist(),
    }


def _target_profile(array: np.ndarray, *, y_index: int) -> np.ndarray:
    return np.mean(np.asarray(array, dtype=np.float64)[-1, :, y_index, :], axis=-1)


def _parallel_profile(array: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(array, dtype=np.float64)[-1, :, :, :], axis=(0, 2))


def _species_metrics(species: str, payload: dict[str, object]) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    metrics.extend(_profile_metrics(f"{species}_target_density", payload["target_density"], target_indices={"lower": 0, "upper": -1}))
    metrics.extend(_profile_metrics(f"{species}_target_flux_proxy", payload["target_flux_proxy"], target_indices={"lower": 0, "upper": -1}))
    neutral_error = _relative_error(
        np.asarray(payload["neutral_parallel_density"]["reference"], dtype=np.float64),
        np.asarray(payload["neutral_parallel_density"]["native"], dtype=np.float64),
    )
    metrics.append(
        {
            "name": f"{species}_neutral_parallel_density_max_relative_error",
            "kind": "max_relative_error",
            "value": neutral_error,
            "target": 5.0e-2,
            "passed": bool(neutral_error <= 5.0e-2),
            "notes": "Neutral-density buildup should stay inside the current promoted direct-tokamak operational band.",
        }
    )
    return metrics


def _profile_metrics(
    prefix: str,
    payload: dict[str, object],
    *,
    target_indices: dict[str, int],
) -> list[dict[str, object]]:
    metrics = []
    for target_name in target_indices:
        reference = np.asarray(payload[target_name]["reference"], dtype=np.float64)
        native = np.asarray(payload[target_name]["native"], dtype=np.float64)
        error = _relative_error(reference, native)
        metrics.append(
            {
                "name": f"{prefix}_{target_name}_max_relative_error",
                "kind": "max_relative_error",
                "value": error,
                "target": 5.0e-2,
                "passed": bool(error <= 5.0e-2),
                "notes": "Profile-level observable parity gate on the committed direct-tokamak validation surface.",
            }
        )
    combined = max(float(metric["value"]) for metric in metrics)
    metrics.append(
        {
            "name": f"{prefix}_max_relative_error",
            "kind": "max_relative_error",
            "value": combined,
            "target": 5.0e-2,
            "passed": bool(combined <= 5.0e-2),
            "notes": "Maximum of lower and upper target-index profile errors.",
        }
    )
    return metrics


def _relative_error(reference: np.ndarray, native: np.ndarray) -> float:
    denominator = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(native - reference)) / denominator)


def _metric_value(report: dict[str, object], name: str) -> float:
    for metric in report["metrics"]:
        if metric["name"] == name:
            return float(metric["value"])
    raise KeyError(name)


def _write_tokamak_recycling_observable_arrays(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    profiles = report["profiles"]
    arrays: dict[str, np.ndarray] = {
        "x_indices": np.asarray(profiles["x_indices"], dtype=np.float64),
        "y_indices": np.asarray(profiles["y_indices"], dtype=np.float64),
        "metric_values": np.asarray([metric["value"] for metric in report["metrics"]], dtype=np.float64),
        "metric_targets": np.asarray([metric["target"] for metric in report["metrics"]], dtype=np.float64),
        "metric_passed": np.asarray([1.0 if metric["passed"] else 0.0 for metric in report["metrics"]], dtype=np.float64),
    }
    for species, payload in profiles["species"].items():
        for family in ("target_density", "target_flux_proxy"):
            for target_name, pair in payload[family].items():
                arrays[f"{species}_{family}_{target_name}_reference"] = np.asarray(pair["reference"], dtype=np.float64)
                arrays[f"{species}_{family}_{target_name}_native"] = np.asarray(pair["native"], dtype=np.float64)
        arrays[f"{species}_neutral_parallel_density_reference"] = np.asarray(
            payload["neutral_parallel_density"]["reference"],
            dtype=np.float64,
        )
        arrays[f"{species}_neutral_parallel_density_native"] = np.asarray(
            payload["neutral_parallel_density"]["native"],
            dtype=np.float64,
        )
    for target_name, pair in profiles["electron_temperature"].items():
        arrays[f"electron_temperature_{target_name}_reference"] = np.asarray(pair["reference"], dtype=np.float64)
        arrays[f"electron_temperature_{target_name}_native"] = np.asarray(pair["native"], dtype=np.float64)
    np.savez_compressed(target, **arrays)
    return target


def _first_field(payload: dict[str, Any]) -> np.ndarray:
    return next(iter(payload["variables"].values()))


def _public_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.expanduser().resolve())
