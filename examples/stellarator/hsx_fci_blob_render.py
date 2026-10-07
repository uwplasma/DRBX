"""Render the HSX FCI filament: 3-D context plus R-Z cross-sections.

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

data = np.load(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/media")
xyz, times, active = data["cartesian"], data["times"], data["owner_active"]
dn = data["density"].astype(np.float64) - 1.0
phi = data["phi"].astype(np.float64)
R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
planes = np.linspace(0, dn.shape[3], 4, endpoint=False).astype(int)
last = len(times) - 1


def wrap(a):  # close the periodic second logical coordinate
    return np.concatenate([a, a[:, :1]], axis=1)


def section(ax, field, k, vmax, cmap):
    masked = np.where(active[:, :, k], field[:, :, k], np.nan)
    mesh = ax.pcolormesh(wrap(R[:, :, k]), wrap(Z[:, :, k]), wrap(masked),
                         cmap=cmap, vmin=-vmax, vmax=vmax, shading="gouraud")
    ax.plot(np.append(R[-1, :, k], R[-1, 0, k]), np.append(Z[-1, :, k], Z[-1, 0, k]), "k-", lw=0.6)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    return mesh


def toroidal_angle(k):
    return np.degrees(np.arctan2(xyz[0, 0, k, 1], xyz[0, 0, k, 0]))


# Figure: 3-D context (boundary coloured by |B|, filament points) + sections.
fig = plt.figure(figsize=(11.0, 6.4), dpi=110)
grid = fig.add_gridspec(3, 5, width_ratios=[2.2, 1, 1, 1, 1], left=0.0, right=0.97, wspace=0.35)
ax3 = fig.add_subplot(grid[:, 0], projection="3d")
edge = xyz[-1]
bmag = data["Bmag"][-1]
ax3.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], rstride=1, cstride=1, linewidth=0,
                 facecolors=plt.cm.viridis((bmag - bmag.min()) / np.ptp(bmag)), alpha=0.25, shade=False)
blob = dn[0] > 0.3 * dn[0].max()
ax3.scatter(xyz[..., 0][blob], xyz[..., 1][blob], xyz[..., 2][blob], s=4, c="crimson")
ax3.set_title("HSX boundary (|B|) and initial filament", fontsize=8)
ax3.set_axis_off()
ax3.view_init(elev=35, azim=-60)
span = np.ptp(edge.reshape(-1, 3), axis=0)
ax3.set_box_aspect(span, zoom=1.35)
rows = [("n - 1, t = 0", dn[0], "RdBu_r"), (f"n - 1, t = {times[last]:.2f}", dn[last], "RdBu_r"),
        (f"phi, t = {times[last]:.2f}", phi[last], "PuOr_r")]
for r, (label, field, cmap) in enumerate(rows):
    vmax = float(np.nanmax(np.abs(np.where(active, field, np.nan))))
    for c, k in enumerate(planes):
        ax = fig.add_subplot(grid[r, c + 1])
        mesh = section(ax, field, k, vmax, cmap)
        if r == 0:
            ax.set_title(f"phi_tor ~ {toroidal_angle(k):.0f} deg", fontsize=7)
        if c == 0:
            ax.set_ylabel(f"{label}\nZ [m]", fontsize=7)
        if r == 2:
            ax.set_xlabel("R [m]", fontsize=7)
    fig.colorbar(mesh, ax=fig.axes[-4:], fraction=0.02, pad=0.02).ax.tick_params(labelsize=6)
fig.suptitle("HSX FCI Braginskii backend: seeded filament on the canonical 32^3 geometry", fontsize=10)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "hsx_fci_blob.png", metadata={"Software": None})

# Movie: n - 1 and phi on the same four planes, fixed colour scales.
fig, axs = plt.subplots(2, len(planes), figsize=(2.2 * len(planes), 4.6), dpi=80)
scales = [float(np.abs(dn).max()), float(np.abs(phi).max())]


def frame(it):
    for row, (field, cmap, vmax) in zip(axs, [(dn, "RdBu_r", scales[0]), (phi, "PuOr_r", scales[1])]):
        for ax, k in zip(row, planes):
            ax.clear()
            section(ax, field[it], k, vmax, cmap)
            ax.set_xticks([])
            ax.set_yticks([])
    axs[0, 0].set_ylabel("n - 1", fontsize=8)
    axs[1, 0].set_ylabel("phi", fontsize=8)
    fig.suptitle(f"HSX FCI filament, t = {times[it]:.3f}", fontsize=9)


FuncAnimation(fig, frame, frames=len(times)).save(out / "hsx_fci_blob.gif", writer=PillowWriter(fps=5))
