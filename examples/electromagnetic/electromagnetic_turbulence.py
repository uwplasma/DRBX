"""Electromagnetic four-field turbulence on a tokamak-like torus, compared with the electrostatic model.

Runs examples/inputs/electromagnetic_turbulence.toml twice through the deck API (`drbx.run`), larger than the deck
and once with [physics] electromagnetic = false, then writes
  OUTPUT_DIR/electromagnetic_turbulence.png  diagnostics: energies, A_par, vorticity and density spread, both models
  OUTPUT_DIR/electromagnetic_turbulence.gif  density fluctuation, vorticity and A_par in one toroidal plane

Run:  PYTHONPATH=src python examples/electromagnetic/electromagnetic_turbulence.py   (about 2 minutes)
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import drbx

# ----------------------------- PARAMETERS -----------------------------------
DECK = Path(__file__).resolve().parents[1] / "inputs" / "electromagnetic_turbulence.toml"
OUTPUT_DIR = Path("output/electromagnetic")
SHAPE = [10, 16, 8]     # (radial, poloidal, toroidal) cells
BETA = 0.1              # mu0 n T / B^2
STEPS, DT = 150, 1.0e-3 # RK4 steps and step
STRIDE = 3              # frames every STRIDE steps
PLANE = 0               # toroidal plane of the movie
# ----------------------------------------------------------------------------

common = dict(shifted_torus={"shape": SHAPE}, run={"steps": STEPS, "dt": DT, "amplitude": 0.1, "seed": 1, "frame_stride": STRIDE})
runs = {}
for name, physics in (("electrostatic", {"electromagnetic": False}), ("electromagnetic", {"electromagnetic": True, "beta": BETA})):
    result = drbx.run(DECK, output_dir=OUTPUT_DIR / name, case_name=name, quiet=False, physics=physics, **common)
    runs[name] = dict(np.load(result["paths"][1]))
es, em = runs["electrostatic"], runs["electromagnetic"]
t = em["times"]
print("final energies (ExB, magnetic, electron, ion):", np.round(em["energies"][-1], 4))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
energies = em["energies"]
for k, label in enumerate(("E x B", "magnetic", "electron kinetic", "ion kinetic")):
    axes[0].semilogy(t, np.maximum(energies[:, k], 1e-12), label=label)
axes[0].set(title="electromagnetic energies", xlabel="t")
axes[0].legend(frameon=False, fontsize=7)
axes[1].plot(t, np.abs(em["apar_frames"]).max(axis=(1, 2, 3)))
axes[1].set(title=r"max $|A_\parallel|$", xlabel="t")
axes[2].plot(t, np.abs(es["omega_frames"]).max(axis=(1, 2, 3)), "k--", label="electrostatic")
axes[2].plot(t, np.abs(em["omega_frames"]).max(axis=(1, 2, 3)), label=f"electromagnetic, beta = {BETA}")
axes[2].set(title=r"max $|\omega|$", xlabel="t")
axes[2].legend(frameon=False, fontsize=7)
axes[3].plot(t, (es["density_frames"] - 1).std(axis=(1, 2, 3)), "k--", label="electrostatic")
axes[3].plot(t, (em["density_frames"] - 1).std(axis=(1, 2, 3)), label="electromagnetic")
axes[3].set(title="density fluctuation rms", xlabel="t")
fig.savefig(OUTPUT_DIR / "electromagnetic_turbulence.png", dpi=100)

panels = (("density fluctuation", em["density_frames"] - 1, "RdBu_r"), ("vorticity", em["omega_frames"], "RdBu_r"),
          (r"$A_\parallel$", em["apar_frames"], "PiYG"))
fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
images = []
for ax, (title, frames, cmap) in zip(axes, panels):
    limit = np.abs(frames[:, :, :, PLANE]).max() or 1.0
    images.append(ax.imshow(frames[0, :, :, PLANE].T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit, aspect="auto"))
    ax.set(title=title, xlabel="radial", ylabel="poloidal")
    fig.colorbar(images[-1], ax=ax, shrink=0.8)
label = fig.suptitle("t = 0")


def update(i):
    for image, (_, frames, _) in zip(images, panels):
        image.set_array(frames[i, :, :, PLANE].T)
    label.set_text(f"t = {t[i]:.3f}")
    return images


movie = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
movie.save(OUTPUT_DIR / "electromagnetic_turbulence.gif", writer=animation.PillowWriter(fps=8), dpi=60)
print(f"wrote {OUTPUT_DIR}/electromagnetic_turbulence.png and .gif")
