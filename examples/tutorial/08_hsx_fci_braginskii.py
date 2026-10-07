"""Tutorial 08 -- advanced: a drift-reduced Braginskii filament in the HSX stellarator.

The ``fci_braginskii`` backend evolves n, Te, Ti, Vi, Ve, vorticity and phi on
a precomputed FCI geometry of HSX (32^3 cells, field lines traced from the
vacuum coils, vessel walls as boundaries). Time stepping is IMEX-SSP222; in each
implicit stage the potential and the parallel current are solved together
(the "implicit current/potential pair"), which lifts the explicit limit
(dt ~ 1e-4 at rho* = 5e-4): the default dt is 7.5e-4 and 1.5e-3 is stable.

This script reads the shipped deck ``examples/inputs/hsx_fci_blob.toml`` (5
steps to t = 3.75e-3), points it at your local geometry bundle, runs it exactly
as ``drbx run`` does, and plots density and potential on one toroidal plane.

The geometry bundle is not in the repository (ask the maintainers; see
docs/fci_braginskii_hsx_backend.md). Expect about 2 minutes of one-time
compilation, then about 10 s per step on a laptop CPU.

Run from the repository root (page: docs/tutorial/08_hsx_fci_braginskii.md):

    PYTHONPATH=src python examples/tutorial/08_hsx_fci_braginskii.py
"""

# 1. Imports ------------------------------------------------------------------
import dataclasses
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.fci_braginskii.run import load_hsx_blob_deck, run

# 2. Input parameters -----------------------------------------------------------
DECK = Path("examples/inputs/hsx_fci_blob.toml")
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")   # where you placed the bundle
OUTPUT = Path("output/tutorial/08_hsx/history.npz")
NUM_STEPS = 5              # IMEX steps (deck value; 200 reaches the documented t = 0.15)
FINAL_TIME = 3.75e-3       # dt = FINAL_TIME / NUM_STEPS = 7.5e-4 (time unit: 1 m / c_s ~ 21 us)
FIGURE = Path("docs/media/tutorial_08_hsx_fci_braginskii.png")

# 3. Run -------------------------------------------------------------------------
if not (GEOMETRY / "manifest.json").is_file():
    raise SystemExit(f"[08] geometry bundle not found at {GEOMETRY}; obtain hsx_fci_32x32x32 from the "
                     "maintainers (docs/fci_braginskii_hsx_backend.md) and set GEOMETRY")
config = dataclasses.replace(load_hsx_blob_deck(DECK), geometry=GEOMETRY, output=OUTPUT,
                             num_steps=NUM_STEPS, final_time=FINAL_TIME, save_every=NUM_STEPS,
                             filament_cache_dir=OUTPUT.parent / "filament_cache")
print(f"[08] deck {DECK}: {NUM_STEPS} steps to t={FINAL_TIME}, rho*={config.rho_star}, "
      f"implicit current/phi pair={config.implicit_current_phi_pair}")
start = time.perf_counter()
run(config)   # prints geometry, compilation and one progress line per step (GMRES iterations, residual)
print(f"[08] run finished in {time.perf_counter() - start:.0f}s")

# 4. Results ----------------------------------------------------------------------
history = np.load(OUTPUT)
density, phi = history["density"].astype(float), history["phi"].astype(float)
plane = int(np.argmax(np.abs(density[0] - 1.0).max(axis=(0, 1))))   # toroidal plane of the blob peak
xyz = history["cartesian"][:, :, plane]
R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
print(f"[08] times saved: {history['times'].tolist()}")
print(f"[08] density: initial peak {density[0].max():.5f}, final peak {density[-1].max():.5f} "
      "(the filament starts to move and relax along the field)")
print(f"[08] potential: max |phi| {np.abs(phi[-1]).max():.3e} (from the vorticity / polarization balance)")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
panels = ((density[0, :, :, plane], "initial density", "viridis"),
          (density[-1, :, :, plane] - density[0, :, :, plane], f"density change at t={history['times'][-1]:.2e}", "RdBu_r"),
          (phi[-1, :, :, plane], "potential phi", "RdBu_r"))
for ax, (field, title, cmap) in zip(axes, panels):
    limit = np.abs(field).max() if cmap == "RdBu_r" else None
    mesh = ax.pcolormesh(R, Z, field, cmap=cmap, shading="auto",
                         vmin=-limit if limit else None, vmax=limit)
    fig.colorbar(mesh, ax=ax, shrink=0.8)
    ax.set(title=title, xlabel="R (m)", ylabel="Z (m)", aspect="equal")
fig.savefig(FIGURE, dpi=80)
print(f"[08] wrote {FIGURE}")
