"""Evolve a single linear mode on the synthetic stellarator FCI geometry.

The script seeds a helical mode envelope on the synthetic stellarator grid and
advances it with an explicit linear right-hand side built from the public FCI
operators: field-aligned parallel diffusion (``laplace_parallel_fci``),
perpendicular diffusion (``laplace_perp_xz``), a curvature-localized drive, and
a uniform damping. It fits the resulting energy trace for a growth rate and
prints it, then saves the history NPZ plus snapshot and diagnostics panels.

Artifacts land under ``docs/data/stellarator_fci_example_artifacts/linear_mode``
(relative to the current working directory): ``stellarator_linear_mode.npz``,
``stellarator_linear_mode_snapshots.png``, and
``stellarator_linear_mode_diagnostics.png``.

Edit the PARAMETERS constants below (grid, frames, time step, drive/damping
coefficients, mode numbers) and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/linear_mode.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dkx.geometry import build_synthetic_stellarator_geometry
from dkx.native.fci import laplace_parallel_fci, laplace_perp_xz
from dkx.validation import save_stellarator_sol_diagnostics_panel, save_stellarator_sol_snapshot_panel

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_example_artifacts/linear_mode")  # artifact root (cwd-relative)
CASE_LABEL = "stellarator_linear_mode"

NX = 24
NY = 24
NZ = 48
FRAMES = 18
SUBSTEPS_PER_FRAME = 5
DT = 0.004
CHI_PARALLEL = 1.2e-2
CHI_PERP = 2.0e-5
LINEAR_DRIVE = 0.020
LINEAR_DAMPING = 0.028
RADIAL_MODE_CENTER = 0.72
RADIAL_MODE_WIDTH = 0.18
POLOIDAL_MODE = 3
TOROIDAL_MODE = 5


def linear_rhs(state: jnp.ndarray) -> jnp.ndarray:
    parallel_diffusion = CHI_PARALLEL * laplace_parallel_fci(state, geometry.maps)
    perpendicular_diffusion = CHI_PERP * laplace_perp_xz(state, dx=dx, dz=dz)
    drive = LINEAR_DRIVE * curvature * envelope * state
    damping = LINEAR_DAMPING * state
    return parallel_diffusion + perpendicular_diffusion + drive - damping


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

geometry = build_synthetic_stellarator_geometry(nx=NX, ny=NY, nz=NZ)
radial = np.asarray(geometry.radial, dtype=np.float64)
theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
curvature = jnp.asarray(geometry.curvature / jnp.max(jnp.abs(geometry.curvature)), dtype=jnp.float64)
envelope = jnp.asarray(np.exp(-((radial - RADIAL_MODE_CENTER) / RADIAL_MODE_WIDTH) ** 2), dtype=jnp.float64)
initial_state = envelope * np.cos(POLOIDAL_MODE * theta - TOROIDAL_MODE * phi)
state = jnp.asarray(initial_state, dtype=jnp.float64)
dx = float(1.0 / (geometry.shape[0] - 1))
dz = float(2.0 * np.pi / geometry.shape[2])

history = []
time = []
for frame in range(FRAMES):
    history.append(np.asarray(state, dtype=np.float64))
    time.append(frame * SUBSTEPS_PER_FRAME * DT)
    for _ in range(SUBSTEPS_PER_FRAME):
        state = state + DT * linear_rhs(state)

history_array = np.asarray(history, dtype=np.float64)
time_array = np.asarray(time, dtype=np.float64)
energy = np.mean(history_array * history_array, axis=(1, 2, 3))
growth_rate = 0.5 * np.polyfit(time_array, np.log(np.maximum(energy, 1.0e-30)), deg=1)[0]

arrays_path = OUTPUT_ROOT / f"{CASE_LABEL}.npz"
snapshot_path = OUTPUT_ROOT / f"{CASE_LABEL}_snapshots.png"
diagnostics_path = OUTPUT_ROOT / f"{CASE_LABEL}_diagnostics.png"
np.savez_compressed(
    arrays_path,
    history=history_array.astype(np.float32),
    time=time_array.astype(np.float64),
    energy=energy.astype(np.float64),
)
save_stellarator_sol_snapshot_panel(geometry, history_array, time_array, snapshot_path)
save_stellarator_sol_diagnostics_panel(geometry, history_array, time_array, diagnostics_path)

print(f"linear mode growth rate: {growth_rate:.4e}")
print(f"energy initial/final: {energy[0]:.4e} -> {energy[-1]:.4e}")
print(f"wrote arrays:      {arrays_path}")
print(f"wrote snapshots:   {snapshot_path}")
print(f"wrote diagnostics: {diagnostics_path}")
