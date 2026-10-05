"""Render the HSX FCI blob: R-Z cross-sections of n - 1 at several eta planes.

Usage: python examples/stellarator/hsx_fci_blob_render.py history.npz [outdir]
where history.npz is written by simulate_hsx_blob.py (--save-every 10).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

plt.rcParams["svg.hashsalt"] = "drbx"
data = np.load(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/media")
xyz, times = data["cartesian"], data["times"]
dn = data["density"].astype(np.float64) - 1.0
active = data["owner_active"]
R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
planes = np.linspace(0, dn.shape[3], 4, endpoint=False).astype(int)
vmax = float(np.abs(dn).max())


def wrap(a):  # close the periodic second logical coordinate
    return np.concatenate([a, a[:, :1]], axis=1)


def draw(axes, it):
    for ax, k in zip(axes, planes):
        ax.clear()
        field = np.where(active[:, :, k], dn[it, :, :, k], np.nan)
        ax.pcolormesh(wrap(R[:, :, k]), wrap(Z[:, :, k]), wrap(field),
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
        ax.plot(R[-1, :, k], Z[-1, :, k], "k-", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        phi = np.degrees(np.arctan2(xyz[0, 0, k, 1], xyz[0, 0, k, 0]))
        ax.set_title(f"eta plane {k} (phi~{phi:.0f} deg)", fontsize=8)


shown = [0, len(times) // 2, len(times) - 1]
fig, axs = plt.subplots(3, len(planes), figsize=(2.2 * len(planes), 6.6), dpi=110)
for row, it in zip(axs, shown):
    draw(row, it)
    row[0].set_ylabel(f"t = {times[it]:.3f}")
fig.suptitle("HSX FCI filament: density perturbation n - 1", fontsize=10)
fig.tight_layout()
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "hsx_fci_blob.png", metadata={"Software": None})

fig, axs = plt.subplots(1, len(planes), figsize=(2.2 * len(planes), 2.6), dpi=80)


def frame(it):
    draw(axs, it)
    fig.suptitle(f"HSX FCI filament, n - 1, t = {times[it]:.3f}", fontsize=9)


FuncAnimation(fig, frame, frames=len(times)).save(
    out / "hsx_fci_blob.gif", writer=PillowWriter(fps=5)
)
