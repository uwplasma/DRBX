"""Automatic-differentiation sensitivity study on the native diffusion lane.

The script builds a small differentiable diffusion setup, defines a scalar
mismatch objective against a synthetic "target" final density, and then:

1. evaluates the objective and its gradient with ``jax.value_and_grad``;
2. cross-checks the four-parameter gradient against central finite differences;
3. sweeps the anomalous diffusivity to show the objective landscape and the
   autodiff tangent line at the nominal point.

It prints the objective value and both gradients, writes the analysis JSON to
``docs/data/autodiff_diffusion_sensitivity_artifacts/data/autodiff_diffusion_sensitivity_analysis.json``
and the three-panel summary figure to
``docs/data/autodiff_diffusion_sensitivity_artifacts/images/autodiff_diffusion_sensitivity.png``
(both relative to the current working directory).

Run from the repository root:

    PYTHONPATH=src python examples/autodiff_diffusion_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from jax import value_and_grad
import jax.numpy as jnp

from drbx.validation.autodiff_diffusion import (
    active_density_slice,
    build_diffusion_autodiff_setup,
    finite_difference_gradient,
    objective_for_physical_parameters,
    simulate_density_history_from_physical,
)
from drbx.validation.publication_plotting import annotate_bars, save_publication_figure, style_axis

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/autodiff_diffusion_sensitivity_artifacts")  # artifact root (cwd-relative)
NX = 160                  # radial grid points; lower for a faster demo
NY = 24                   # poloidal grid points
TIMESTEP = 3.0            # output interval of the diffusion rollout
STEPS = 8                 # number of rollout steps the gradient flows through
# Physical parameters are [anomalous_D, amplitude, center, width].
TARGET_PARAMETERS = (0.42, 0.22, 0.56, 0.11)    # generates the synthetic target state
NOMINAL_PARAMETERS = (0.30, 0.16, 0.46, 0.16)   # point where the sensitivity is evaluated
FD_EPSILON = 5.0e-4       # central finite-difference step for the gradient check
SWEEP_RANGE = (0.2, 0.6)  # anomalous-diffusivity range for the objective sweep
SWEEP_POINTS = 60         # sweep resolution (each point is one objective evaluation)

# --- problem setup through the public autodiff-diffusion API ----------------------
print("building differentiable diffusion setup "
      f"(nx={NX}, ny={NY}, timestep={TIMESTEP}, steps={STEPS})...")
setup = build_diffusion_autodiff_setup(nx=NX, ny=NY, timestep=TIMESTEP, steps=STEPS)
target_parameters = jnp.asarray(TARGET_PARAMETERS, dtype=jnp.float64)
nominal_parameters = jnp.asarray(NOMINAL_PARAMETERS, dtype=jnp.float64)

print("simulating the synthetic target final state...")
target_history = simulate_density_history_from_physical(
    setup,
    anomalous_D=target_parameters[0],
    amplitude=target_parameters[1],
    center=target_parameters[2],
    width=target_parameters[3],
)
target_final = active_density_slice(setup, target_history[-1])


def objective(parameters: jnp.ndarray) -> jnp.ndarray:
    """Scalar mismatch of the evolved final density against the target state."""

    return objective_for_physical_parameters(
        parameters,
        setup,
        target_final_density=target_final,
    )


# --- gradients: reverse-mode autodiff vs central finite differences ---------------
print("evaluating objective and reverse-mode gradient at the nominal parameters...")
objective_value, autodiff_gradient = value_and_grad(objective)(nominal_parameters)
print(f"  objective value: {float(objective_value):.6e}")
print(f"  autodiff gradient: {np.asarray(autodiff_gradient)}")

print(f"cross-checking with central finite differences (epsilon={FD_EPSILON:g})...")
finite_difference = finite_difference_gradient(objective, nominal_parameters, epsilon=FD_EPSILON)
print(f"  finite-difference gradient: {np.asarray(finite_difference)}")

print("simulating the nominal final state for the profile panel...")
nominal_history = simulate_density_history_from_physical(
    setup,
    anomalous_D=nominal_parameters[0],
    amplitude=nominal_parameters[1],
    center=nominal_parameters[2],
    width=nominal_parameters[3],
)
nominal_final = active_density_slice(setup, nominal_history[-1])

# --- diffusivity sweep with the autodiff tangent line -----------------------------
print(f"sweeping the anomalous diffusivity over {SWEEP_RANGE} ({SWEEP_POINTS} points)...")
diffusivity_sweep = np.linspace(SWEEP_RANGE[0], SWEEP_RANGE[1], SWEEP_POINTS, dtype=np.float64)
sweep_objective = np.asarray(
    [float(objective(nominal_parameters.at[0].set(value))) for value in diffusivity_sweep],
    dtype=np.float64,
)
tangent = float(objective_value) + float(autodiff_gradient[0]) * (
    diffusivity_sweep - float(nominal_parameters[0])
)

# --- save the analysis JSON -------------------------------------------------------
data_dir = OUTPUT_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
analysis_path = data_dir / "autodiff_diffusion_sensitivity_analysis.json"
analysis_path.write_text(
    json.dumps(
        {
            "target_parameters": np.asarray(target_parameters).tolist(),
            "nominal_parameters": np.asarray(nominal_parameters).tolist(),
            "objective_value": float(objective_value),
            "autodiff_gradient": np.asarray(autodiff_gradient, dtype=np.float64).tolist(),
            "finite_difference_gradient": np.asarray(finite_difference, dtype=np.float64).tolist(),
            "diffusivity_sweep": diffusivity_sweep.tolist(),
            "sweep_objective": sweep_objective.tolist(),
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
figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

axes[0].plot(diffusivity_sweep, sweep_objective, color="#0b6e4f", linewidth=2.6, label="objective sweep")
axes[0].plot(diffusivity_sweep, tangent, color="#d1495b", linestyle="--", linewidth=2.0, label="autodiff tangent")
axes[0].axvline(float(nominal_parameters[0]), color="#1b1b1b", linewidth=1.2, alpha=0.7)
style_axis(
    axes[0],
    title="Sensitivity around nominal diffusivity",
    xlabel="anomalous diffusivity",
    ylabel="objective",
    grid="both",
)
axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
axes[0].legend(frameon=False)

labels = ["D", "A", "center", "width"]
x = np.arange(len(labels))
bar_width = 0.34
autodiff_values = np.asarray(autodiff_gradient, dtype=np.float64)
axes[1].bar(x - bar_width / 2.0, autodiff_values, width=bar_width, color="#0077b6", label="autodiff")
axes[1].bar(
    x + bar_width / 2.0,
    np.asarray(finite_difference, dtype=np.float64),
    width=bar_width,
    color="#f4a261",
    label="finite diff",
)
axes[1].set_xticks(x, labels)
style_axis(axes[1], title="Gradient verification", ylabel="gradient", grid="y")
axes[1].legend(frameon=False)
annotate_bars(axes[1], x - bar_width / 2.0, autodiff_values, fmt="{:.2e}", fontsize=8.2)

x_coords = np.linspace(0.0, 1.0, np.asarray(target_final).shape[0], dtype=np.float64)
axes[2].plot(x_coords, radial_mean(target_final), color="#111111", linewidth=2.8, label="target final state")
axes[2].plot(x_coords, radial_mean(nominal_final), color="#6c757d", linewidth=2.3, linestyle="--", label="nominal final state")
style_axis(
    axes[2],
    title="Final-state profile sensitivity context",
    xlabel="normalized radial coordinate",
    ylabel="radial mean density",
    grid="both",
)
axes[2].legend(frameon=False)

plot_path = images_dir / "autodiff_diffusion_sensitivity.png"
save_publication_figure(figure, plot_path)
print(f"wrote summary plot: {plot_path}")
