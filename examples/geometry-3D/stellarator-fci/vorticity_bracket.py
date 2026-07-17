"""Vorticity/potential dynamics with logical E x B brackets on the stellarator.

The script demonstrates the physics-backed nonlinear coupling path of the FCI
DRB right-hand side: ``compute_fci_drb_rhs`` supplies the sheath/recycling,
neutral, charge-exchange, vorticity-diffusion, and potential-inversion pieces
on the synthetic stellarator geometry, and the example adds explicit
``logical_exb_bracket_xz`` advection of density, pressure, and vorticity plus a
curvature-driven vorticity source. A short explicit rollout is integrated,
tracking the ion-density fluctuation and the potential-solve residual.

It prints the pass flag, final RMS fluctuation, final potential residual, and
radial flux proxy, and saves under
``docs/data/stellarator_fci_example_artifacts/vorticity_bracket`` (relative to
the current working directory): ``stellarator_vorticity_bracket.npz``,
``.json`` summary, ``_snapshots.png``, and ``_diagnostics.png``.

Edit the PARAMETERS constants below (grid, frames, dt, advection/drive/damping
coefficients) and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/vorticity_bracket.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_drb.geometry import build_synthetic_stellarator_geometry
from jax_drb.native.fci import logical_exb_bracket_xz
from jax_drb.native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs
from jax_drb.validation import save_stellarator_sol_diagnostics_panel, save_stellarator_sol_snapshot_panel

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_example_artifacts/vorticity_bracket")  # artifact root (cwd-relative)
CASE_LABEL = "stellarator_vorticity_bracket"

NX = 14
NY = 12
NZ = 28
FRAMES = 12
SUBSTEPS_PER_FRAME = 2
DT = 0.004

FIELD_PERIODS = 5
ISLAND_MODE = 2
ISLAND_AMPLITUDE = 0.030
MIRROR_AMPLITUDE = 0.16

POTENTIAL_ITERATIONS = 35
SOURCE_STRENGTH = 0.010
CURVATURE_DRIVE = 0.016
VORTICITY_DAMPING = 0.030
EXB_DENSITY_ADVECTION = 0.025
EXB_PRESSURE_ADVECTION = 0.018
EXB_VORTICITY_ADVECTION = 0.030


def build_initial_state(geometry) -> FciDrbState:
    """Create a compact two-fluid/neutral state on the non-axisymmetric grid.

    The evolved fields are ion/electron/neutral densities, ion/electron/neutral
    pressures, parallel ion/neutral momenta, and vorticity. The helical
    perturbation seeds an interchange-like fluctuation while preserving
    positive density and pressure.
    """

    radial = geometry.radial
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    envelope = jnp.exp(-jnp.square((radial - 0.68) / 0.17))
    helical = jnp.cos(2.0 * theta - FIELD_PERIODS * phi) + 0.35 * jnp.sin(
        3.0 * theta + FIELD_PERIODS * phi
    )
    fluctuation = envelope * helical
    ion_density = 1.0 + 0.10 * radial + 0.030 * fluctuation
    electron_density = 1.0 + 0.10 * radial + 0.028 * fluctuation
    neutral_density = 0.22 + 0.08 * radial
    return FciDrbState(
        ion_density=ion_density,
        electron_density=electron_density,
        neutral_density=neutral_density,
        ion_pressure=0.08 * ion_density,
        electron_pressure=0.11 * electron_density,
        neutral_pressure=0.012 * neutral_density,
        ion_momentum=0.010 * ion_density * fluctuation,
        neutral_momentum=0.003 * neutral_density * fluctuation,
        vorticity=0.040 * fluctuation,
    )


def radial_derivative(field, geometry):
    """One-sided radial-edge derivative for the curvature-drive proxy."""

    values = jnp.asarray(field, dtype=jnp.float64)
    spacing = jnp.asarray(geometry.metric.dx, dtype=jnp.float64)
    centered = (jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0)) / jnp.maximum(
        2.0 * spacing,
        1.0e-30,
    )
    first = (values[1, :, :] - values[0, :, :]) / jnp.maximum(spacing[0, :, :], 1.0e-30)
    last = (values[-1, :, :] - values[-2, :, :]) / jnp.maximum(spacing[-1, :, :], 1.0e-30)
    return centered.at[0, :, :].set(first).at[-1, :, :].set(last)


def add_state(state: FciDrbState, rhs: FciDrbState, scale: float) -> FciDrbState:
    """Explicit Euler update for the compact pedagogical example."""

    state_children = state.tree_flatten()[0]
    rhs_children = rhs.tree_flatten()[0]
    return FciDrbState(
        *(current + scale * increment for current, increment in zip(state_children, rhs_children, strict=True))
    )


def clip_state(state: FciDrbState) -> FciDrbState:
    """Keep the explicit demonstration run in the positive-state regime."""

    return FciDrbState(
        ion_density=jnp.maximum(state.ion_density, 1.0e-6),
        electron_density=jnp.maximum(state.electron_density, 1.0e-6),
        neutral_density=jnp.maximum(state.neutral_density, 1.0e-8),
        ion_pressure=jnp.maximum(state.ion_pressure, 1.0e-8),
        electron_pressure=jnp.maximum(state.electron_pressure, 1.0e-8),
        neutral_pressure=jnp.maximum(state.neutral_pressure, 1.0e-10),
        ion_momentum=state.ion_momentum,
        neutral_momentum=state.neutral_momentum,
        vorticity=jnp.clip(state.vorticity, -2.0, 2.0),
    )


def bracket_rhs(state: FciDrbState, geometry, parameters: FciDrbRhsParameters, scalar_time: float):
    """Assemble a physics-backed nonlinear RHS around the native FCI DRB pieces.

    `compute_fci_drb_rhs` supplies the JAX-native sheath/recycling, neutral,
    charge-exchange, vorticity-diffusion, and potential-inversion pieces.
    The nonlinear terms are then written as logical perpendicular brackets
    `{phi, f} = (partial_theta phi partial_s f - partial_s phi partial_theta f) / B`.
    """

    result = compute_fci_drb_rhs(state, maps=geometry.maps, metric=geometry.metric, parameters=parameters)
    potential = result.potential
    total_pressure = state.ion_pressure + state.electron_pressure
    pressure_gradient = radial_derivative(total_pressure, geometry)
    curvature = geometry.curvature / jnp.maximum(jnp.max(jnp.abs(geometry.curvature)), 1.0e-12)
    source = jnp.exp(-jnp.square((geometry.radial - 0.55) / 0.14)) * (
        1.0 + 0.12 * jnp.sin(9.0 * scalar_time)
    )

    ion_advection = logical_exb_bracket_xz(potential, state.ion_density, geometry.metric)
    electron_advection = logical_exb_bracket_xz(potential, state.electron_density, geometry.metric)
    pressure_advection = logical_exb_bracket_xz(potential, total_pressure, geometry.metric)
    vorticity_advection = logical_exb_bracket_xz(potential, state.vorticity, geometry.metric)

    return (
        FciDrbState(
            ion_density=result.rhs.ion_density - EXB_DENSITY_ADVECTION * ion_advection + SOURCE_STRENGTH * source,
            electron_density=(
                result.rhs.electron_density - EXB_DENSITY_ADVECTION * electron_advection + SOURCE_STRENGTH * source
            ),
            neutral_density=result.rhs.neutral_density,
            ion_pressure=(
                result.rhs.ion_pressure - EXB_PRESSURE_ADVECTION * pressure_advection + 0.5 * SOURCE_STRENGTH * source
            ),
            electron_pressure=(
                result.rhs.electron_pressure
                - EXB_PRESSURE_ADVECTION * pressure_advection
                + 0.6 * SOURCE_STRENGTH * source
            ),
            neutral_pressure=result.rhs.neutral_pressure,
            ion_momentum=result.rhs.ion_momentum,
            neutral_momentum=result.rhs.neutral_momentum,
            vorticity=(
                result.rhs.vorticity
                - EXB_VORTICITY_ADVECTION * vorticity_advection
                + CURVATURE_DRIVE * curvature * pressure_gradient
                - VORTICITY_DAMPING * state.vorticity
            ),
        ),
        potential,
        result.potential_residual_l2,
    )


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

geometry = build_synthetic_stellarator_geometry(
    nx=NX,
    ny=NY,
    nz=NZ,
    field_periods=FIELD_PERIODS,
    island_mode=ISLAND_MODE,
    island_amplitude=ISLAND_AMPLITUDE,
    mirror_amplitude=MIRROR_AMPLITUDE,
)
parameters = FciDrbRhsParameters(potential_iterations=POTENTIAL_ITERATIONS)
state = build_initial_state(geometry)

history = []
potential_history = []
potential_residual_history = []
time = []
for frame in range(FRAMES):
    history.append(np.asarray(state.ion_density - jnp.mean(state.ion_density), dtype=np.float64))
    rhs, potential, residual = bracket_rhs(state, geometry, parameters, frame * SUBSTEPS_PER_FRAME * DT)
    potential_history.append(np.asarray(potential, dtype=np.float64))
    potential_residual_history.append(float(residual))
    time.append(frame * SUBSTEPS_PER_FRAME * DT)
    state = clip_state(add_state(state, rhs, DT))
    for substep in range(1, SUBSTEPS_PER_FRAME):
        scalar_time = (frame * SUBSTEPS_PER_FRAME + substep) * DT
        rhs, _, _ = bracket_rhs(state, geometry, parameters, scalar_time)
        state = clip_state(add_state(state, rhs, DT))

history_array = np.asarray(history, dtype=np.float64)
potential_array = np.asarray(potential_history, dtype=np.float64)
time_array = np.asarray(time, dtype=np.float64)
potential_residual_array = np.asarray(potential_residual_history, dtype=np.float64)
energy = np.mean(np.square(history_array), axis=(1, 2, 3))
radial_flux_proxy = np.mean(history_array * np.asarray(geometry.curvature, dtype=np.float64), axis=(1, 2, 3))

arrays_path = OUTPUT_ROOT / f"{CASE_LABEL}.npz"
summary_path = OUTPUT_ROOT / f"{CASE_LABEL}.json"
snapshot_path = OUTPUT_ROOT / f"{CASE_LABEL}_snapshots.png"
diagnostics_path = OUTPUT_ROOT / f"{CASE_LABEL}_diagnostics.png"

np.savez_compressed(
    arrays_path,
    density_fluctuation=history_array.astype(np.float32),
    potential=potential_array.astype(np.float32),
    time=time_array.astype(np.float32),
    potential_residual=potential_residual_array.astype(np.float64),
    energy=energy.astype(np.float64),
    radial_flux_proxy=radial_flux_proxy.astype(np.float64),
)
summary = {
    "case": CASE_LABEL,
    "model": "fci_drb_vorticity_potential_exb_bracket",
    "grid": [NX, NY, NZ],
    "frames": FRAMES,
    "substeps_per_frame": SUBSTEPS_PER_FRAME,
    "dt": DT,
    "potential_residual_l2_final": float(potential_residual_array[-1]),
    "density_fluctuation_rms_final": float(np.sqrt(np.mean(np.square(history_array[-1])))),
    "energy_initial": float(energy[0]),
    "energy_final": float(energy[-1]),
    "radial_flux_proxy_final": float(radial_flux_proxy[-1]),
    "passed": bool(np.all(np.isfinite(history_array)) and np.all(np.isfinite(potential_array))),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

save_stellarator_sol_snapshot_panel(geometry, history_array, time_array, snapshot_path)
save_stellarator_sol_diagnostics_panel(geometry, history_array, time_array, diagnostics_path)

print(f"passed: {summary['passed']}")
print(f"final RMS fluctuation: {summary['density_fluctuation_rms_final']:.4e}")
print(f"final potential residual L2: {summary['potential_residual_l2_final']:.4e}")
print(f"radial flux proxy: {summary['radial_flux_proxy_final']:.4e}")
print(f"wrote arrays:      {arrays_path}")
print(f"wrote summary:     {summary_path}")
print(f"wrote snapshots:   {snapshot_path}")
print(f"wrote diagnostics: {diagnostics_path}")
