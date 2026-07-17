"""Compact nonlinear reduced-SOL turbulence run on the synthetic stellarator.

The script builds the synthetic stellarator FCI geometry, runs the public
``simulate_reduced_stellarator_sol_dynamics`` reduced nonlinear model for
``FRAMES`` output frames, and evaluates the showcase QA report (RMS
fluctuation, radial flux proxy, dominant mode numbers). It prints the report
summary and saves the history NPZ, snapshot and diagnostics panels, a 3D
poster frame, and a GIF movie.

Artifacts land under
``docs/data/stellarator_fci_example_artifacts/nonlinear_turbulence`` (relative
to the current working directory), all prefixed
``stellarator_nonlinear_turbulence``; the NPZ feeds
``turbulent_profile_analysis.py``.

Edit the PARAMETERS constants below (grid, frames, dt, island/mirror settings)
and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/stellarator-fci/nonlinear_turbulence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drbx.geometry import build_synthetic_stellarator_geometry
from drbx.validation import (
    build_stellarator_sol_showcase_report,
    save_stellarator_sol_3d_frame,
    save_stellarator_sol_3d_movie,
    save_stellarator_sol_diagnostics_panel,
    save_stellarator_sol_snapshot_panel,
    simulate_reduced_stellarator_sol_dynamics,
)

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/stellarator_fci_example_artifacts/nonlinear_turbulence")  # artifact root (cwd-relative)
CASE_LABEL = "stellarator_nonlinear_turbulence"

NX = 28
NY = 28
NZ = 56
FRAMES = 24
SUBSTEPS_PER_FRAME = 4
DT = 0.008
FIELD_PERIODS = 5
ISLAND_MODE = 2
ISLAND_AMPLITUDE = 0.034
MIRROR_AMPLITUDE = 0.18

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
history, time = simulate_reduced_stellarator_sol_dynamics(
    geometry,
    frames=FRAMES,
    substeps_per_frame=SUBSTEPS_PER_FRAME,
    dt=DT,
)
report = build_stellarator_sol_showcase_report(geometry, history, time)

arrays_path = OUTPUT_ROOT / f"{CASE_LABEL}.npz"
snapshot_path = OUTPUT_ROOT / f"{CASE_LABEL}_snapshots.png"
diagnostics_path = OUTPUT_ROOT / f"{CASE_LABEL}_diagnostics.png"
poster_path = OUTPUT_ROOT / f"{CASE_LABEL}_poster.png"
movie_path = OUTPUT_ROOT / f"{CASE_LABEL}.gif"
np.savez_compressed(
    arrays_path,
    history=history.astype(np.float16),
    time=time.astype(np.float32),
    curvature=np.asarray(geometry.curvature, dtype=np.float32),
    connection_length=np.asarray(geometry.connection_length, dtype=np.float32),
)
save_stellarator_sol_snapshot_panel(geometry, history, time, snapshot_path)
save_stellarator_sol_diagnostics_panel(geometry, history, time, diagnostics_path)
save_stellarator_sol_3d_frame(geometry, history[-1], float(time[-1]), poster_path)
save_stellarator_sol_3d_movie(geometry, history, time, movie_path)

print(f"passed: {report['passed']}")
print(f"final RMS fluctuation: {report['final_rms_fluctuation']:.4e}")
print(f"radial flux proxy: {report['radial_flux_proxy']:.4e}")
print(f"dominant modes (m,n): {report['dominant_poloidal_mode_index']}, {report['dominant_toroidal_mode_index']}")
print(f"wrote arrays:      {arrays_path}")
print(f"wrote snapshots:   {snapshot_path}")
print(f"wrote diagnostics: {diagnostics_path}")
print(f"wrote poster:      {poster_path}")
print(f"wrote movie:       {movie_path}")
