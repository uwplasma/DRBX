"""README figure 1: fluid neutrals on the phi = 0 (bean) poloidal section of the 32^3 HSX artifact.

Input: the history of a 240-step run (t = 0.18) of the default fluid-neutral deck with recycling, saved as
output/neutral_figures/hsx_neutrals_run.npz (examples/inputs/hsx_fci_neutrals.toml with geometry = the 32^3 bundle,
final_time = 0.18, num_steps = 240, save_every = 12, neutral_wall_absorption = 0.1, recycling_coefficient = 0.9).
Reference density n0 = 1e19 m^-3. Fields are interpolated linearly along the toroidal index to the cylindrical angle PHI0.

Run from the repository root:

    PYTHONPATH=src python examples/neutrals/fig1_hsx_neutrals.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PARAMETERS: run from the repository root; run histories and ledger logs are read from DATA, figures written to OUT.
DATA, OUT = pathlib.Path("output/neutral_figures"), pathlib.Path("docs/media")
DATA.mkdir(parents=True, exist_ok=True)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9", grey="#555555", black="#222222")
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 100, "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans",
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

def save(fig, name):
    """Save as an 8-bit palette PNG (<150 KB)."""
    from PIL import Image
    p = OUT / f"{name}.png"
    fig.savefig(p, facecolor="white")
    im = Image.open(p).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(p, optimize=True)
    print(name, p.stat().st_size // 1024, "KB", Image.open(p).size)

import json
import json, numpy as np
from matplotlib.colors import LogNorm
PHI0 = 0.0
d = np.load(DATA / "hsx_neutrals_run.npz", allow_pickle=True)
meta = json.loads(d["run_metadata_json"].item()); n0, T0 = 1e19, meta["neutral_reference_temperature_ev"]  # single reference density
X = d["cartesian"]; R = np.hypot(X[..., 0], X[..., 1]); Z = X[..., 2]
phi = np.unwrap(np.arctan2(X[..., 1], X[..., 0]), axis=2)

def at_phi(f, p0):
    """Periodic linear interpolation along the toroidal index to the angle p0, per (u, theta) column."""
    out = np.empty(f.shape[:2])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            p = phi[i, j]
            pe = np.concatenate([p - 2 * np.pi, p, p + 2 * np.pi]); fe = np.tile(f[i, j], 3)
            out[i, j] = np.interp(p0, pe, fe)
    return out

def wrap(a):  # close the poloidal ring
    return np.concatenate([a, a[:, :1]], axis=1)

last = -1
nn = d["neutral_density"][last] * n0 / 1e17; Tn = d["neutral_temperature"][last] * T0
kn = d["neutral_knudsen_gradient"]
panels = []
for name, f, cmap, label in (("n", nn, "viridis", r"neutral density $n_n$ ($10^{17}$ m$^{-3}$)"), ("T", Tn, "magma", r"neutral temperature $T_n$ (eV)"), ("kn", kn, "RdBu_r", r"Knudsen number $Kn$")):
    panels.append((name, wrap(at_phi(f, PHI0)), cmap, label))
NMIN = 1e-2  # 1e15 m^-3: below this T_n and Kn are set by the vacuum floor, not by neutrals, and are masked
mask = panels[0][1] < NMIN
panels = [(n_, np.ma.masked_where(mask, f), c_, l_) for n_, f, c_, l_ in panels]
bearing = nn >= NMIN
r_, z_ = wrap(at_phi(R, PHI0)), wrap(at_phi(Z, PHI0))
act = d["owner_active"].astype(float)
Nv = (d["neutral_density"] * d["jacobian"]).sum((1, 2, 3)); print("inventory change over last tenth (t=%.3f..%.3f) %.2f%%" % (d["times"][-3], d["times"][-1], 100 * (Nv[-1] / Nv[-3] - 1)), "t_final", d["times"][-1])
print("time", d["times"][last], "n_n max", nn.max(), "T_n range", Tn.min(), Tn.max(), "Kn min/max", kn.min(), kn.max(), "frac Kn<0.3 among n_n>=1e15", (kn[bearing] < 0.3).mean(), "frac Kn>=1", (kn[bearing] >= 1).mean(), "n_n>=1e15 cells", bearing.mean(), "Kn median there", np.median(kn[bearing]))

fig, axs = plt.subplots(1, 3, figsize=(8.8, 5.2), constrained_layout=True)
for ax, (name, f, cmap, label) in zip(axs, panels):
    kw = dict(norm=LogNorm(vmin=0.1, vmax=10)) if name == "kn" else dict(norm=LogNorm(vmin=NMIN, vmax=20)) if name == "n" else dict(vmin=f.min(), vmax=f.max())
    m = ax.pcolormesh(r_, z_, f, cmap=cmap, shading="gouraud", rasterized=True, **kw)
    ax.plot(r_[-1], z_[-1], color=C["black"], lw=1.8)  # outermost radial surface = domain / wall boundary
    if name == "kn":
        ax.contour(r_, z_, f, levels=[1.0], colors=C["black"], linewidths=1.0, linestyles="--")
        if f.min() < 0.3: ax.contour(r_, z_, f, levels=[0.3], colors=C["black"], linewidths=1.6)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("$R$ (m)")
    ax.set_title({"n": "Neutral density", "T": "Neutral temperature", "kn": "Knudsen number"}[name], fontsize=11.5)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.04, shrink=0.95); cb.set_label(label, fontsize=10.5); cb.ax.tick_params(labelsize=10.5)
    if name == "kn": cb.set_ticks([0.1, 0.3, 1, 3, 10]); cb.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
    if name == "T": cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
axs[0].set_ylabel("$Z$ (m)")
for ax in axs[1:]: ax.set_yticklabels([])
fig.suptitle(f"Fluid neutrals fill the HSX edge shell (peak {nn.max():.1f}e17 m$^{{-3}}$), but Kn > 1 in {100*(kn[bearing]>=1).mean():.0f}% of them", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "fig1_hsx_neutrals")
