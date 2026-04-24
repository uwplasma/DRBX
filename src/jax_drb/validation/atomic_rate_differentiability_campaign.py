from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jax import grad, jit, vmap
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from ..native.recycling_atomic import (
    eval_amjuel_fit,
    eval_openadas_rate,
    hydrogen_cx_sigmav,
    load_amjuel_rate,
    load_openadas_rate,
)
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class AtomicRateDifferentiabilityCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_atomic_rate_differentiability_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "atomic_rate_differentiability_campaign",
) -> AtomicRateDifferentiabilityCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_atomic_rate_differentiability_campaign_report()
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = save_atomic_rate_differentiability_campaign_plot(
        report,
        arrays,
        images_dir / f"{case_label}.png",
    )
    return AtomicRateDifferentiabilityCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_atomic_rate_differentiability_campaign_report(
    *,
    point_count: int = 96,
    density_m3: float = 2.0e18,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Build an AMJUEL/OpenADAS/CX differentiability audit for documentation."""

    log_temperature = jnp.linspace(jnp.log(0.2), jnp.log(300.0), int(point_count), dtype=jnp.float64)
    density = jnp.asarray(float(density_m3), dtype=jnp.float64)
    sigma_v_coeffs, _, _ = load_amjuel_rate("d", "iz")
    openadas_coeffs, _, openadas_log_temperature, openadas_log_density, _ = load_openadas_rate("ne", "iz")
    dataset_scalars = {"Nnorm": 1.0e19, "Omega_ci": 2.0e6}

    def amjuel_log_rate(log_te):
        te = jnp.exp(log_te)
        return jnp.log(eval_amjuel_fit(te, density, sigma_v_coeffs))

    def openadas_log_rate(log_te):
        te = jnp.exp(log_te)
        return jnp.log(
            eval_openadas_rate(
                te,
                density,
                openadas_coeffs,
                log_temperature=openadas_log_temperature,
                log_density=openadas_log_density,
            )
        )

    def cx_log_rate(log_te):
        te = jnp.exp(log_te)
        return jnp.log(hydrogen_cx_sigmav(te, dataset_scalars))

    log_rate_functions = {
        "amjuel_d_ionisation": amjuel_log_rate,
        "openadas_ne_ionisation": openadas_log_rate,
        "hydrogen_charge_exchange": cx_log_rate,
    }
    arrays: dict[str, np.ndarray] = {
        "temperature_ev": np.asarray(jnp.exp(log_temperature), dtype=np.float64),
    }
    metrics: list[dict[str, object]] = []
    epsilon = jnp.asarray(1.0e-4, dtype=jnp.float64)
    for name, scalar_fn in log_rate_functions.items():
        compiled_fn = jit(vmap(scalar_fn))
        compiled_grad = jit(vmap(grad(scalar_fn)))
        autodiff_derivative = compiled_grad(log_temperature)
        finite_difference_derivative = (
            compiled_fn(log_temperature + epsilon) - compiled_fn(log_temperature - epsilon)
        ) / (2.0 * epsilon)
        log_rate = compiled_fn(log_temperature)
        derivative_abs_error = jnp.abs(autodiff_derivative - finite_difference_derivative)
        derivative_rel_error = derivative_abs_error / jnp.maximum(jnp.abs(finite_difference_derivative), 1.0e-30)

        arrays[f"{name}_rate"] = np.asarray(jnp.exp(log_rate), dtype=np.float64)
        arrays[f"{name}_autodiff_dlograte_dlogte"] = np.asarray(autodiff_derivative, dtype=np.float64)
        arrays[f"{name}_finite_difference_dlograte_dlogte"] = np.asarray(finite_difference_derivative, dtype=np.float64)
        arrays[f"{name}_derivative_abs_error"] = np.asarray(derivative_abs_error, dtype=np.float64)
        arrays[f"{name}_derivative_rel_error"] = np.asarray(derivative_rel_error, dtype=np.float64)
        max_abs_error = float(jnp.max(derivative_abs_error))
        max_rel_error = float(jnp.max(derivative_rel_error))
        metrics.append(
            {
                "name": name,
                "max_abs_derivative_error": max_abs_error,
                "max_relative_derivative_error": max_rel_error,
                "passed": bool(max_abs_error < 1.0e-6 and max_rel_error < 1.0e-5),
            }
        )

    report = {
        "case": "atomic_rate_differentiability_campaign",
        "density_m3": float(density_m3),
        "temperature_min_ev": float(arrays["temperature_ev"][0]),
        "temperature_max_ev": float(arrays["temperature_ev"][-1]),
        "point_count": int(point_count),
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if metric["passed"]),
        "metrics": metrics,
        "literature_context": {
            "hermes_3_reactions": "Hermes-3 documents AMJUEL H.4 ionisation/recombination, AMJUEL H.3 charge exchange, and OpenADAS impurity reactions as reaction-source components.",
            "autodiff_context": "JAX recommends composing grad/jvp/vjp with vmap for batched derivative products; this campaign checks the atomic-rate derivative surface before full residual promotion.",
        },
    }
    return report, arrays


def save_atomic_rate_differentiability_campaign_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temperature = arrays["temperature_ev"]
    series = (
        ("amjuel_d_ionisation", "AMJUEL D ionisation", "#005f73"),
        ("openadas_ne_ionisation", "OpenADAS Ne ionisation", "#9b2226"),
        ("hydrogen_charge_exchange", "H charge exchange", "#ee9b00"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.2), constrained_layout=True)
    for key, label, color in series:
        rate = arrays[f"{key}_rate"]
        normalized_rate = np.maximum(rate / np.max(rate), 1.0e-14)
        axes[0, 0].plot(temperature, normalized_rate, label=label, color=color, linewidth=2.0)
        axes[0, 1].plot(
            temperature,
            arrays[f"{key}_autodiff_dlograte_dlogte"],
            label=label,
            color=color,
            linewidth=2.0,
        )
        axes[1, 0].plot(
            temperature,
            arrays[f"{key}_derivative_abs_error"],
            label=label,
            color=color,
            linewidth=2.0,
        )

    style_axis(
        axes[0, 0],
        title="Packaged atomic-rate surfaces normalized by peak value",
        xlabel="electron/effective temperature [eV]",
        ylabel="rate / max(rate)",
        xscale="log",
        yscale="log",
        grid="both",
    )
    axes[0, 0].legend(frameon=False, fontsize=8.4)

    style_axis(
        axes[0, 1],
        title="Autodiff slope with respect to log temperature",
        xlabel="electron/effective temperature [eV]",
        ylabel=r"$d \log(k) / d \log(T)$",
        xscale="log",
        grid="both",
    )

    style_axis(
        axes[1, 0],
        title="Autodiff versus centered finite difference",
        xlabel="electron/effective temperature [eV]",
        ylabel="absolute derivative error",
        xscale="log",
        yscale="log",
        grid="both",
    )

    metric_labels = [str(metric["name"]).replace("_", "\n") for metric in report["metrics"]]
    metric_values = np.asarray([float(metric["max_abs_derivative_error"]) for metric in report["metrics"]], dtype=np.float64)
    x = np.arange(metric_values.size)
    axes[1, 1].bar(x, np.maximum(metric_values, 1.0e-16), color=["#005f73", "#9b2226", "#ee9b00"])
    axes[1, 1].set_xticks(x, metric_labels)
    style_axis(
        axes[1, 1],
        title="Maximum derivative-parity error",
        ylabel="max absolute error",
        yscale="log",
        grid="y",
    )
    axes[1, 1].axhline(1.0e-6, color="#6c757d", linestyle="--", linewidth=1.0, label="gate")
    axes[1, 1].legend(frameon=False, fontsize=8.4)

    figure.suptitle(
        "Atomic-rate differentiability audit for reaction-source residuals",
        fontsize=13.2,
        fontweight="semibold",
    )
    save_publication_figure(figure, target)
    return target
