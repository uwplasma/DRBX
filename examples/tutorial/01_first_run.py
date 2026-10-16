"""Tutorial 01 -- first run: one input deck, from the CLI and from Python, with a restart.

The deck ``examples/tutorial/01_diffusion.toml`` evolves one hydrogen species
(density Nh, pressure Ph) under anomalous radial diffusion,
dN/dt = d/dx (D dN/dx), on a 16 x 24 grid. This script

1. runs the deck from Python (``drbx.native.run_input_case``) for NOUT outputs,
2. runs the first FIRST_NOUT outputs through the CLI entry point, then resumes
   from the restart bundle for the remaining outputs (exactly what
   ``drbx run deck.toml --restart-in ... --resume-steps ...`` does), and
3. checks that the restarted history equals the uninterrupted Python run and
   plots the density at start and end plus radial cuts at every output.

Run from the repository root (page: docs/tutorial/01_first_run.md):

    PYTHONPATH=src python examples/tutorial/01_first_run.py
"""

# 1. Imports ------------------------------------------------------------------
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.cli import main as drbx_cli
from drbx.native import run_input_case
from drbx.native.deck_runner import load_portable_array_payload

# 2. Input parameters -----------------------------------------------------------
DECK = Path("examples/tutorial/01_diffusion.toml")  # the input file (D = 200; the shipped
                                                   # examples/inputs/restartable_diffusion.toml has D = 2)
NOUT = 5            # output intervals of the uninterrupted Python run
FIRST_NOUT = 3      # intervals of the first CLI segment (set by [time] nout in the deck)
OUTPUT_DIR = Path("output/tutorial/01_first_run")
FIGURE = Path("docs/media/tutorial_01_first_run.png")

# 3. Run -------------------------------------------------------------------------
t0 = time.perf_counter()
print(f"[01] deck: {DECK}")
print(f"[01] step A: Python run_input_case for {NOUT} output intervals (verbose)")
python_run = run_input_case(DECK, case_name="python", output_steps=NOUT, verbose=True)
density_python = np.asarray(python_run.variables["Nh"], dtype=float)
print(f"[01] step A done in {time.perf_counter() - t0:.1f}s; Nh history shape {density_python.shape}")

t1 = time.perf_counter()
first, resumed = OUTPUT_DIR / "first", OUTPUT_DIR / "resumed"
print(f"[01] step B: CLI run of the first {FIRST_NOUT} intervals")
argv = ["run", str(DECK), "--case-name", "first", "--output-dir", str(first), "--verbose"]
print("[01]   $ drbx " + " ".join(argv))
assert drbx_cli(argv) == 0
print(f"[01] step C: CLI restart for {NOUT - FIRST_NOUT} more intervals")
argv = ["run", str(DECK), "--case-name", "resumed", "--output-dir", str(resumed),
        "--restart-in", str(first / "first_restart.npz"), "--resume-steps", str(NOUT - FIRST_NOUT)]
print("[01]   $ drbx " + " ".join(argv))
assert drbx_cli(argv) == 0
print(f"[01] steps B+C done in {time.perf_counter() - t1:.1f}s")

# 4. Results ----------------------------------------------------------------------
a = load_portable_array_payload(first / "first_arrays.npz")
b = load_portable_array_payload(resumed / "resumed_arrays.npz")
times = np.concatenate([a["time_points"], np.asarray(b["time_points"])[1:]])
density_cli = np.concatenate([np.asarray(a["variables"]["Nh"]), np.asarray(b["variables"]["Nh"])[1:]])
mismatch = float(np.max(np.abs(density_cli - density_python)))
# The initial blob is a top hat in x (H(x-0.25) H(0.75-x)) times a Gaussian in y.
# The deck's diffusion is radial (x), so the sharp x-edges smooth out while the
# y-shape is untouched: watch the steepest radial gradient fall.
mid_y = density_python.shape[2] // 2
profiles = density_python[:, :, mid_y, 0]
steepest = np.max(np.abs(np.diff(profiles, axis=1)), axis=1)
print(f"[01] output times: {np.round(times, 3).tolist()}")
print(f"[01] restart check: max |N_cli+restart - N_python| = {mismatch:.2e} (expect ~0: restart is exact)")
print(f"[01] steepest radial jump in Nh: {steepest[0]:.3f} -> {steepest[-1]:.3f} (radial diffusion smooths the edges)")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
for ax, index, title in ((axes[0], 0, "initial Nh"), (axes[1], -1, f"Nh at t = {times[-1]:g}")):
    image = ax.imshow(density_python[index][:, :, 0].T, origin="lower", aspect="auto",
                      vmin=1.0, vmax=float(density_python.max()), cmap="viridis")
    ax.set(title=title, xlabel="x index", ylabel="y index")
fig.colorbar(image, ax=axes[:2], shrink=0.85)
for index, t in enumerate(times):
    axes[2].plot(profiles[index], color=plt.cm.viridis(index / len(times)), label=f"t={t:g}")
axes[2].plot(density_cli[-1][:, mid_y, 0], "k.", ms=4, label="CLI+restart, end")
axes[2].set(title="radial cut through the blob", xlabel="x index", ylabel="Nh")
axes[2].legend(fontsize=6)
fig.savefig(FIGURE, dpi=90)
print(f"[01] wrote {FIGURE}; total {time.perf_counter() - t0:.1f}s")
