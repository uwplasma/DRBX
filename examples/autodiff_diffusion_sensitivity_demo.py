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
from jax import value_and_grad
import jax.numpy as jnp

from jax_drb.validation.autodiff_diffusion import (
    active_density_slice,
    build_diffusion_autodiff_setup,
    finite_difference_gradient,
    objective_for_physical_parameters,
    simulate_density_history_from_physical,
)


@dataclass(frozen=True)
class SensitivitySettings:
    output_root: Path
    quiet: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a summary automatic-differentiation sensitivity study on the "
            "native diffusion lane and save detailed plots."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "autodiff_diffusion_sensitivity_artifacts",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> SensitivitySettings:
    return SensitivitySettings(
        output_root=args.output_root,
        quiet=args.quiet,
    )


def log(settings: SensitivitySettings, title: str, mapping: dict[str, Any]) -> None:
    if settings.quiet:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def radial_mean(field: jnp.ndarray) -> np.ndarray:
    return np.asarray(jnp.mean(field, axis=1), dtype=np.float64)


def build_demo_problem():
    setup = build_diffusion_autodiff_setup(nx=160, ny=24, timestep=3.0, steps=8)
    target_parameters = jnp.asarray([0.42, 0.22, 0.56, 0.11], dtype=jnp.float64)
    nominal_parameters = jnp.asarray([0.30, 0.16, 0.46, 0.16], dtype=jnp.float64)
    target_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=target_parameters[0],
        amplitude=target_parameters[1],
        center=target_parameters[2],
        width=target_parameters[3],
    )
    target_final = active_density_slice(setup, target_history[-1])
    objective = lambda parameters: objective_for_physical_parameters(
        parameters,
        setup,
        target_final_density=target_final,
    )
    return setup, target_parameters, nominal_parameters, target_final, objective


def build_sensitivity_payload():
    setup, target_parameters, nominal_parameters, target_final, objective = build_demo_problem()
    objective_value, autodiff_gradient = value_and_grad(objective)(nominal_parameters)
    finite_difference = finite_difference_gradient(objective, nominal_parameters, epsilon=5.0e-4)

    nominal_history = simulate_density_history_from_physical(
        setup,
        anomalous_D=nominal_parameters[0],
        amplitude=nominal_parameters[1],
        center=nominal_parameters[2],
        width=nominal_parameters[3],
    )
    nominal_final = active_density_slice(setup, nominal_history[-1])

    diffusivity_sweep = np.linspace(0.2, 0.6, 60, dtype=np.float64)
    sweep_objective = np.asarray(
        [float(objective(nominal_parameters.at[0].set(value))) for value in diffusivity_sweep],
        dtype=np.float64,
    )
    tangent = float(objective_value) + float(autodiff_gradient[0]) * (diffusivity_sweep - float(nominal_parameters[0]))

    return {
        "setup": setup,
        "target_parameters": np.asarray(target_parameters, dtype=np.float64),
        "nominal_parameters": np.asarray(nominal_parameters, dtype=np.float64),
        "target_final": np.asarray(target_final, dtype=np.float64),
        "nominal_final": np.asarray(nominal_final, dtype=np.float64),
        "objective_value": float(objective_value),
        "autodiff_gradient": np.asarray(autodiff_gradient, dtype=np.float64),
        "finite_difference_gradient": np.asarray(finite_difference, dtype=np.float64),
        "diffusivity_sweep": diffusivity_sweep,
        "sweep_objective": sweep_objective,
        "tangent": tangent,
    }


def write_analysis_json(settings: SensitivitySettings, payload: dict[str, Any]) -> Path:
    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "autodiff_diffusion_sensitivity_analysis.json"
    path.write_text(
        json.dumps(
            {
                "target_parameters": payload["target_parameters"].tolist(),
                "nominal_parameters": payload["nominal_parameters"].tolist(),
                "objective_value": payload["objective_value"],
                "autodiff_gradient": payload["autodiff_gradient"].tolist(),
                "finite_difference_gradient": payload["finite_difference_gradient"].tolist(),
                "diffusivity_sweep": payload["diffusivity_sweep"].tolist(),
                "sweep_objective": payload["sweep_objective"].tolist(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def save_summary_plot(settings: SensitivitySettings, payload: dict[str, Any]) -> Path:
    images_dir = settings.output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    axes[0].plot(payload["diffusivity_sweep"], payload["sweep_objective"], color="#0b6e4f", linewidth=2.6, label="objective sweep")
    axes[0].plot(payload["diffusivity_sweep"], payload["tangent"], color="#d1495b", linestyle="--", linewidth=2.0, label="autodiff tangent")
    axes[0].axvline(float(payload["nominal_parameters"][0]), color="#1b1b1b", linewidth=1.2, alpha=0.7)
    axes[0].set_xlabel("anomalous diffusivity")
    axes[0].set_ylabel("objective")
    axes[0].set_title("Sensitivity around nominal diffusivity")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    labels = ["D", "A", "center", "width"]
    x = np.arange(len(labels))
    width = 0.34
    axes[1].bar(x - width / 2.0, payload["autodiff_gradient"], width=width, color="#0077b6", label="autodiff")
    axes[1].bar(x + width / 2.0, payload["finite_difference_gradient"], width=width, color="#f4a261", label="finite diff")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("gradient")
    axes[1].set_title("Gradient verification")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    x_coords = np.linspace(0.0, 1.0, payload["target_final"].shape[0], dtype=np.float64)
    axes[2].plot(x_coords, radial_mean(payload["target_final"]), color="#111111", linewidth=2.8, label="target final state")
    axes[2].plot(x_coords, radial_mean(payload["nominal_final"]), color="#6c757d", linewidth=2.3, linestyle="--", label="nominal final state")
    axes[2].set_xlabel("normalized radial coordinate")
    axes[2].set_ylabel("radial mean density")
    axes[2].set_title("Final-state profile sensitivity context")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    path = images_dir / "autodiff_diffusion_sensitivity.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def main() -> int:
    settings = build_settings(parse_args())
    payload = build_sensitivity_payload()
    json_path = write_analysis_json(settings, payload)
    plot_path = save_summary_plot(settings, payload)
    log(
        settings,
        "Autodiff Sensitivity Artifacts",
        {
            "analysis_json": json_path,
            "plot": plot_path,
            "objective_value": f"{payload['objective_value']:.6e}",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
