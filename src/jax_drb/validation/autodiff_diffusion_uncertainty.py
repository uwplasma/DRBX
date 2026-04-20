from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from .autodiff_diffusion import (
    active_density_slice,
    build_diffusion_autodiff_setup,
    objective_for_physical_parameters,
    simulate_density_history_from_physical,
)


@dataclass(frozen=True)
class AutodiffDiffusionUncertaintyArtifacts:
    analysis_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_autodiff_diffusion_uncertainty_package(
    *,
    output_root: str | Path,
    case_label: str = "autodiff_diffusion_uncertainty",
    sample_count: int = 96,
    random_seed: int = 7,
) -> AutodiffDiffusionUncertaintyArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_autodiff_diffusion_uncertainty_report(
        sample_count=sample_count,
        random_seed=random_seed,
    )
    analysis_json_path = data_dir / f"{case_label}_analysis.json"
    arrays_npz_path = data_dir / f"{case_label}_arrays.npz"
    plot_png_path = images_dir / f"{case_label}.png"
    analysis_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(arrays_npz_path, **arrays)
    save_autodiff_diffusion_uncertainty_plot(report, arrays, plot_png_path)
    return AutodiffDiffusionUncertaintyArtifacts(
        analysis_json_path=analysis_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_autodiff_diffusion_uncertainty_report(
    *,
    sample_count: int = 96,
    random_seed: int = 7,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    setup = build_diffusion_autodiff_setup(nx=160, ny=24, timestep=3.0, steps=8)

    mean_parameters = jnp.asarray([0.34, 0.15, 0.48, 0.14], dtype=jnp.float64)
    parameter_std = jnp.asarray([0.03, 0.025, 0.04, 0.02], dtype=jnp.float64)
    covariance = jnp.diag(parameter_std**2)

    def radial_profile(parameters: jnp.ndarray) -> jnp.ndarray:
        history = simulate_density_history_from_physical(
            setup,
            anomalous_D=parameters[0],
            amplitude=parameters[1],
            center=parameters[2],
            width=parameters[3],
        )
        final_density = active_density_slice(setup, history[-1])
        return jnp.mean(final_density, axis=1)

    def scalar_qoi(parameters: jnp.ndarray) -> jnp.ndarray:
        return objective_for_physical_parameters(
            parameters,
            setup,
            objective_kind="variance",
        )

    mean_profile = radial_profile(mean_parameters)
    mean_qoi = scalar_qoi(mean_parameters)
    profile_jacobian = jax.jacobian(radial_profile)(mean_parameters)
    qoi_gradient = jax.grad(scalar_qoi)(mean_parameters)

    linearized_profile_variance = jnp.einsum("ik,kl,il->i", profile_jacobian, covariance, profile_jacobian)
    linearized_profile_sigma = jnp.sqrt(jnp.maximum(linearized_profile_variance, 0.0))
    linearized_qoi_sigma = jnp.sqrt(jnp.maximum(qoi_gradient @ covariance @ qoi_gradient, 0.0))

    rng = np.random.default_rng(random_seed)
    raw_samples = rng.normal(
        loc=np.asarray(mean_parameters, dtype=np.float64),
        scale=np.asarray(parameter_std, dtype=np.float64),
        size=(sample_count, mean_parameters.size),
    )
    sampled_parameters = np.asarray(raw_samples, dtype=np.float64)
    sampled_parameters[:, 0] = np.clip(sampled_parameters[:, 0], 0.08, None)
    sampled_parameters[:, 1] = np.clip(sampled_parameters[:, 1], 0.03, None)
    sampled_parameters[:, 2] = np.clip(sampled_parameters[:, 2], 0.05, 0.95)
    sampled_parameters[:, 3] = np.clip(sampled_parameters[:, 3], 0.05, 0.28)

    batched_profile = jax.jit(jax.vmap(radial_profile))
    batched_qoi = jax.jit(jax.vmap(scalar_qoi))
    sampled_profiles = np.asarray(batched_profile(jnp.asarray(sampled_parameters, dtype=jnp.float64)), dtype=np.float64)
    sampled_qoi = np.asarray(batched_qoi(jnp.asarray(sampled_parameters, dtype=jnp.float64)), dtype=np.float64)

    monte_carlo_profile_mean = sampled_profiles.mean(axis=0)
    monte_carlo_profile_sigma = sampled_profiles.std(axis=0, ddof=1)
    monte_carlo_qoi_mean = float(sampled_qoi.mean())
    monte_carlo_qoi_sigma = float(sampled_qoi.std(ddof=1))

    profile_sigma_gap = float(np.max(np.abs(monte_carlo_profile_sigma - np.asarray(linearized_profile_sigma))))
    qoi_sigma_relative_error = float(
        abs(float(linearized_qoi_sigma) - monte_carlo_qoi_sigma) / max(monte_carlo_qoi_sigma, 1.0e-12)
    )

    report = {
        "case": "autodiff_diffusion_uncertainty",
        "sample_count": sample_count,
        "random_seed": random_seed,
        "mean_parameters": np.asarray(mean_parameters, dtype=np.float64).tolist(),
        "parameter_std": np.asarray(parameter_std, dtype=np.float64).tolist(),
        "scalar_qoi_name": "final_density_variance",
        "scalar_qoi_at_mean": float(mean_qoi),
        "linearized_qoi_sigma": float(linearized_qoi_sigma),
        "monte_carlo_qoi_mean": monte_carlo_qoi_mean,
        "monte_carlo_qoi_sigma": monte_carlo_qoi_sigma,
        "qoi_sigma_relative_error": qoi_sigma_relative_error,
        "profile_sigma_max_abs_gap": profile_sigma_gap,
        "interpretation": [
            "The scalar QoI uses the variance of the final active-domain density field on the compact native diffusion lane.",
            "The profile uncertainty uses the radial mean of the final active-domain density field.",
            "The comparison is between a first-order autodiff pushforward of the parameter covariance and a vectorized Monte Carlo estimate on the same native solve path.",
        ],
    }
    arrays = {
        "mean_parameters": np.asarray(mean_parameters, dtype=np.float64),
        "parameter_std": np.asarray(parameter_std, dtype=np.float64),
        "radial_coordinate": np.linspace(0.0, 1.0, mean_profile.shape[0], dtype=np.float64),
        "mean_profile": np.asarray(mean_profile, dtype=np.float64),
        "linearized_profile_sigma": np.asarray(linearized_profile_sigma, dtype=np.float64),
        "monte_carlo_profile_mean": monte_carlo_profile_mean,
        "monte_carlo_profile_sigma": monte_carlo_profile_sigma,
        "sampled_qoi": sampled_qoi,
        "sampled_parameters": sampled_parameters,
        "qoi_gradient": np.asarray(qoi_gradient, dtype=np.float64),
    }
    return report, arrays


def save_autodiff_diffusion_uncertainty_plot(
    report: dict[str, object],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), constrained_layout=True)

    labels = ["D", "A", "center", "width"]
    x = np.arange(len(labels))
    axes[0].bar(x, arrays["mean_parameters"], color="#4c78a8", alpha=0.85)
    axes[0].errorbar(
        x,
        arrays["mean_parameters"],
        yerr=arrays["parameter_std"],
        fmt="none",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=5,
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("parameter value")
    axes[0].set_title("Input uncertainty model")
    axes[0].grid(alpha=0.25, axis="y")

    sampled_qoi = arrays["sampled_qoi"]
    axes[1].hist(sampled_qoi, bins=18, color="#72b7b2", alpha=0.8, density=True, label="Monte Carlo")
    qoi_mean = float(report["scalar_qoi_at_mean"])
    qoi_sigma = float(report["linearized_qoi_sigma"])
    qoi_x = np.linspace(sampled_qoi.min(), sampled_qoi.max(), 240, dtype=np.float64)
    normal_pdf = np.exp(-0.5 * ((qoi_x - qoi_mean) / max(qoi_sigma, 1.0e-12)) ** 2) / max(
        qoi_sigma * np.sqrt(2.0 * np.pi),
        1.0e-12,
    )
    axes[1].plot(qoi_x, normal_pdf, color="#d1495b", linewidth=2.4, label="Linearized Gaussian")
    axes[1].axvline(float(report["monte_carlo_qoi_mean"]), color="#111111", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("final-density variance QoI")
    axes[1].set_ylabel("density")
    axes[1].set_title("Scalar uncertainty propagation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    radial = arrays["radial_coordinate"]
    mean_profile = arrays["mean_profile"]
    linear_sigma = arrays["linearized_profile_sigma"]
    mc_mean = arrays["monte_carlo_profile_mean"]
    mc_sigma = arrays["monte_carlo_profile_sigma"]
    axes[2].plot(radial, mean_profile, color="#111111", linewidth=2.5, label="mean profile")
    axes[2].fill_between(
        radial,
        mean_profile - 2.0 * linear_sigma,
        mean_profile + 2.0 * linear_sigma,
        color="#f58518",
        alpha=0.28,
        label="linearized 95% band",
    )
    axes[2].plot(radial, mc_mean, color="#2a9d8f", linewidth=2.0, linestyle="--", label="MC mean profile")
    axes[2].fill_between(
        radial,
        mc_mean - 2.0 * mc_sigma,
        mc_mean + 2.0 * mc_sigma,
        color="#54a24b",
        alpha=0.18,
        label="Monte Carlo 95% band",
    )
    axes[2].set_xlabel("normalized radial coordinate")
    axes[2].set_ylabel("radial mean density")
    axes[2].set_title("Profile uncertainty pushforward")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    figure.savefig(target, dpi=220)
    plt.close(figure)
    return target
