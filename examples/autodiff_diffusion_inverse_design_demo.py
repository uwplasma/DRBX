from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from jax_drb.validation.autodiff_diffusion import (
    active_density_slice,
    build_diffusion_autodiff_setup,
    optimize_inverse_design,
    objective_for_physical_parameters,
    physical_to_theta,
    simulate_density_history_from_physical,
    theta_to_physical,
)


@dataclass(frozen=True)
class InverseDesignSettings:
    output_root: Path
    quiet: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a gradient-based inverse design demonstration on the native diffusion lane "
            "and save publication-ready plots."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "autodiff_diffusion_inverse_design_artifacts",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> InverseDesignSettings:
    return InverseDesignSettings(
        output_root=args.output_root,
        quiet=args.quiet,
    )


def log(settings: InverseDesignSettings, title: str, mapping: dict[str, Any]) -> None:
    if settings.quiet:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def radial_mean(field: jnp.ndarray) -> np.ndarray:
    return np.asarray(jnp.mean(field, axis=1), dtype=np.float64)


def build_inverse_design_problem():
    setup = build_diffusion_autodiff_setup(nx=128, ny=20, timestep=3.0, steps=6)
    target_parameters = jnp.asarray([0.46, 0.20, 0.60, 0.10], dtype=jnp.float64)
    initial_guess_parameters = jnp.asarray([0.22, 0.08, 0.34, 0.18], dtype=jnp.float64)
    target_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=target_parameters[0],
        amplitude=target_parameters[1],
        center=target_parameters[2],
        width=target_parameters[3],
    )
    target_final = active_density_slice(setup, target_history[-1])
    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        physical = theta_to_physical(theta)
        return objective_for_physical_parameters(
            jnp.asarray(
                [
                    physical["anomalous_D"],
                    physical["amplitude"],
                    physical["center"],
                    physical["width"],
                ],
                dtype=jnp.float64,
            ),
            setup,
            target_final_density=target_final,
        )

    theta0 = physical_to_theta(
        anomalous_D=float(initial_guess_parameters[0]),
        amplitude=float(initial_guess_parameters[1]),
        center=float(initial_guess_parameters[2]),
        width=float(initial_guess_parameters[3]),
    )
    return setup, target_parameters, initial_guess_parameters, target_final, theta0, objective


def build_inverse_design_payload():
    setup, target_parameters, initial_guess_parameters, target_final, theta0, objective = build_inverse_design_problem()
    result = optimize_inverse_design(objective, theta0, iterations=55, learning_rate=0.05)
    optimized_physical = theta_to_physical(result["theta"])

    initial_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=initial_guess_parameters[0],
        amplitude=initial_guess_parameters[1],
        center=initial_guess_parameters[2],
        width=initial_guess_parameters[3],
    )
    optimized_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=optimized_physical["anomalous_D"],
        amplitude=optimized_physical["amplitude"],
        center=optimized_physical["center"],
        width=optimized_physical["width"],
    )

    return {
        "target_parameters": np.asarray(target_parameters, dtype=np.float64),
        "initial_guess_parameters": np.asarray(initial_guess_parameters, dtype=np.float64),
        "optimized_parameters": np.asarray(
            [
                optimized_physical["anomalous_D"],
                optimized_physical["amplitude"],
                optimized_physical["center"],
                optimized_physical["width"],
            ],
            dtype=np.float64,
        ),
        "loss_history": np.asarray(result["loss_history"], dtype=np.float64),
        "target_final": np.asarray(target_final, dtype=np.float64),
        "initial_final": np.asarray(active_density_slice(setup, initial_history[-1]), dtype=np.float64),
        "optimized_final": np.asarray(active_density_slice(setup, optimized_history[-1]), dtype=np.float64),
        "final_loss": float(result["final_loss"]),
    }


def write_analysis_json(settings: InverseDesignSettings, payload: dict[str, Any]) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "autodiff_diffusion_inverse_design_analysis.json"
    path.write_text(
        json.dumps(
            {
                "target_parameters": payload["target_parameters"].tolist(),
                "initial_guess_parameters": payload["initial_guess_parameters"].tolist(),
                "optimized_parameters": payload["optimized_parameters"].tolist(),
                "loss_history": payload["loss_history"].tolist(),
                "final_loss": payload["final_loss"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def save_publication_plot(settings: InverseDesignSettings, payload: dict[str, Any]) -> Path:
    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)

    axes[0].plot(payload["loss_history"], color="#7b2cbf", linewidth=2.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimization iteration")
    axes[0].set_ylabel("objective")
    axes[0].set_title("Inverse-design convergence")
    axes[0].grid(alpha=0.25)

    x_coords = np.linspace(0.0, 1.0, payload["target_final"].shape[0], dtype=np.float64)
    axes[1].plot(x_coords, radial_mean(payload["target_final"]), color="#111111", linewidth=2.8, label="target")
    axes[1].plot(x_coords, radial_mean(payload["initial_final"]), color="#9d4edd", linewidth=2.2, linestyle="--", label="initial guess")
    axes[1].plot(x_coords, radial_mean(payload["optimized_final"]), color="#2a9d8f", linewidth=2.4, label="optimized")
    axes[1].set_xlabel("normalized radial coordinate")
    axes[1].set_ylabel("radial mean density")
    axes[1].set_title("Recovered final-state profile")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    labels = ["D", "A", "center", "width"]
    x = np.arange(len(labels))
    axes[2].plot(x, payload["target_parameters"], marker="o", linewidth=2.5, color="#111111", label="target")
    axes[2].plot(x, payload["initial_guess_parameters"], marker="s", linewidth=2.0, linestyle="--", color="#9d4edd", label="initial")
    axes[2].plot(x, payload["optimized_parameters"], marker="^", linewidth=2.3, color="#2a9d8f", label="optimized")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("parameter value")
    axes[2].set_title("Recovered design parameters")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    path = images_dir / "autodiff_diffusion_inverse_design.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def main() -> int:
    settings = build_settings(parse_args())
    payload = build_inverse_design_payload()
    json_path = write_analysis_json(settings, payload)
    plot_path = save_publication_plot(settings, payload)
    log(
        settings,
        "Autodiff Inverse Design Artifacts",
        {
            "analysis_json": json_path,
            "plot": plot_path,
            "final_loss": f"{payload['final_loss']:.6e}",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
