"""Gradient-based inverse design on the native diffusion lane.

The script builds a small differentiable diffusion setup, simulates a synthetic
"target" final density from known physical parameters, and then recovers those
parameters by gradient descent from a deliberately poor initial guess:

1. the design vector ``theta`` lives in an unconstrained space and is mapped to
   positive physical parameters with ``theta_to_physical``;
2. ``optimize_inverse_design`` runs Adam-style gradient steps on the mismatch
   objective, differentiating through the full diffusion rollout;
3. the recovered parameters, loss history, and final-state profiles are saved.

It prints the optimization progress and final loss, writes the analysis JSON to
``docs/data/autodiff_diffusion_inverse_design_artifacts/data/autodiff_diffusion_inverse_design_analysis.json``
and the three-panel summary figure to
``docs/data/autodiff_diffusion_inverse_design_artifacts/images/autodiff_diffusion_inverse_design.png``
(both relative to the current working directory).

Run from the repository root:

    PYTHONPATH=src python examples/autodiff_diffusion_inverse_design.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from drbx.validation.autodiff_diffusion import (
    active_density_slice,
    build_diffusion_autodiff_setup,
    optimize_inverse_design,
    objective_for_physical_parameters,
    physical_to_theta,
    simulate_density_history_from_physical,
    theta_to_physical,
)
from drbx.validation.publication_plotting import annotate_bars, save_publication_figure, style_axis

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/autodiff_diffusion_inverse_design_artifacts")  # artifact root (cwd-relative)
NX = 128                  # radial grid points; lower for a faster demo
NY = 20                   # poloidal grid points
TIMESTEP = 3.0            # output interval of the diffusion rollout
STEPS = 6                 # number of rollout steps the gradient flows through
# Physical parameters are [anomalous_D, amplitude, center, width].
TARGET_PARAMETERS = (0.46, 0.20, 0.60, 0.10)         # generates the synthetic target
INITIAL_GUESS_PARAMETERS = (0.22, 0.08, 0.34, 0.18)  # deliberately poor starting design
ITERATIONS = 55           # optimizer iterations; raise for a tighter recovery
LEARNING_RATE = 0.05      # optimizer learning rate

# --- problem setup through the public autodiff-diffusion API ----------------------
print("building differentiable diffusion setup "
      f"(nx={NX}, ny={NY}, timestep={TIMESTEP}, steps={STEPS})...")
setup = build_diffusion_autodiff_setup(nx=NX, ny=NY, timestep=TIMESTEP, steps=STEPS)
target_parameters = jnp.asarray(TARGET_PARAMETERS, dtype=jnp.float64)
initial_guess_parameters = jnp.asarray(INITIAL_GUESS_PARAMETERS, dtype=jnp.float64)

print("simulating the synthetic target final state...")
target_history = simulate_density_history_from_physical(
    setup,
    anomalous_D=target_parameters[0],
    amplitude=target_parameters[1],
    center=target_parameters[2],
    width=target_parameters[3],
)
target_final = active_density_slice(setup, target_history[-1])


def objective(theta: jnp.ndarray) -> jnp.ndarray:
    """Mismatch objective in the unconstrained design space."""

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

# --- gradient-based optimization --------------------------------------------------
print(f"optimizing the design ({ITERATIONS} iterations, learning rate {LEARNING_RATE})...")
result = optimize_inverse_design(objective, theta0, iterations=ITERATIONS, learning_rate=LEARNING_RATE)
optimized_physical = theta_to_physical(result["theta"])
optimized_parameters = np.asarray(
    [
        optimized_physical["anomalous_D"],
        optimized_physical["amplitude"],
        optimized_physical["center"],
        optimized_physical["width"],
    ],
    dtype=np.float64,
)
final_loss = float(result["final_loss"])
loss_history = np.asarray(result["loss_history"], dtype=np.float64)
print(f"  final loss: {final_loss:.6e}")
print(f"  target parameters:    {np.asarray(target_parameters)}")
print(f"  optimized parameters: {optimized_parameters}")

print("simulating initial-guess and optimized final states for the profile panel...")
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
initial_final = active_density_slice(setup, initial_history[-1])
optimized_final = active_density_slice(setup, optimized_history[-1])

# --- save the analysis JSON -------------------------------------------------------
data_dir = OUTPUT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
analysis_path = data_dir / "autodiff_diffusion_inverse_design_analysis.json"
analysis_path.write_text(
    json.dumps(
        {
            "target_parameters": np.asarray(target_parameters).tolist(),
            "initial_guess_parameters": np.asarray(initial_guess_parameters).tolist(),
            "optimized_parameters": optimized_parameters.tolist(),
            "loss_history": loss_history.tolist(),
            "final_loss": final_loss,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
print(f"wrote analysis JSON: {analysis_path}")

# --- three-panel summary figure ---------------------------------------------------
def radial_mean(field: jnp.ndarray) -> np.ndarray:
    return np.asarray(jnp.mean(field, axis=1), dtype=np.float64)


images_dir = OUTPUT_ROOT / "images"
images_dir.mkdir(parents=True, exist_ok=True)
figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)

axes[0].plot(loss_history, color="#7b2cbf", linewidth=2.6)
style_axis(
    axes[0],
    title="Inverse-design convergence",
    xlabel="optimization iteration",
    ylabel="objective",
    yscale="log",
    grid="both",
)

x_coords = np.linspace(0.0, 1.0, np.asarray(target_final).shape[0], dtype=np.float64)
axes[1].plot(x_coords, radial_mean(target_final), color="#111111", linewidth=2.8, label="target")
axes[1].plot(x_coords, radial_mean(initial_final), color="#9d4edd", linewidth=2.2, linestyle="--", label="initial guess")
axes[1].plot(x_coords, radial_mean(optimized_final), color="#2a9d8f", linewidth=2.4, label="optimized")
style_axis(
    axes[1],
    title="Recovered final-state profile",
    xlabel="normalized radial coordinate",
    ylabel="radial mean density",
    grid="both",
)
axes[1].legend(frameon=False)

labels = ["D", "A", "center", "width"]
x = np.arange(len(labels))
axes[2].plot(x, np.asarray(target_parameters), marker="o", linewidth=2.5, color="#111111", label="target")
axes[2].plot(x, np.asarray(initial_guess_parameters), marker="s", linewidth=2.0, linestyle="--", color="#9d4edd", label="initial")
axes[2].plot(x, optimized_parameters, marker="^", linewidth=2.3, color="#2a9d8f", label="optimized")
axes[2].set_xticks(x, labels)
style_axis(axes[2], title="Recovered design parameters", ylabel="parameter value", grid="both")
axes[2].legend(frameon=False)
annotate_bars(axes[2], x, optimized_parameters, fmt="{:.2f}", fontsize=8.2)

plot_path = images_dir / "autodiff_diffusion_inverse_design.png"
save_publication_figure(figure, plot_path)
print(f"wrote summary plot: {plot_path}")
