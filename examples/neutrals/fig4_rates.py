"""Fig 4: AMJUEL 2020 rate coefficients used by the FCI path vs the SD1D fits (shared rates module).

Run from the repository root:

    PYTHONPATH=src python examples/neutrals/fig4_rates.py
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

import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from drbx.native.neutrals import atomic_rates as A

T = np.logspace(np.log10(0.3), np.log10(300), 200)
N = 1e19
iz = np.asarray(A.rate_coefficient("d", "iz", T, N)); rec = np.asarray(A.rate_coefficient("d", "rec", T, N))
cx = np.asarray(A.charge_exchange_rate_coefficient(T))
siz, scx, srec = (np.asarray(f) for f in (A.sd1d_ionisation_rate(T), A.sd1d_charge_exchange_rate(T), A.sd1d_recombination_rate(N, T)))
np.savez(DATA / "fig4_rates.npz", T=T, n=N, iz=iz, rec=rec, cx=cx, siz=siz, scx=scx, srec=srec)

fig, ax = plt.subplots(figsize=(8.8, 5.0))
for y, ys, c, lab in ((iz, siz, C["blue"], "ionisation"), (cx, scx, C["orange"], "charge exchange"), (rec, srec, C["red"], "recombination")):
    ax.loglog(T, y, color=c, lw=2.4, label=lab)
    ax.loglog(T, ys, color=c, lw=1.6, ls=(0, (4, 2.5)))
ax.set_xlabel("electron temperature $T_e$ (eV)"); ax.set_ylabel(r"rate coefficient $\langle\sigma v\rangle$ (m$^3$ s$^{-1}$)")
ax.set_ylim(1e-22, 1e-10); ax.set_xlim(0.3, 300)
f = lambda y: np.interp(10, T, y)
ax.set_title(f"At 10 eV the SD1D fits differ from AMJUEL by {f(siz)/f(iz):.2f}x (ionisation), {f(scx)/f(cx):.2f}x (CX)")
for x, y, c, lab, va in ((0.45, 3e-14, C["orange"], "charge exchange", "bottom"), (3.5, 2e-18, C["blue"], "ionisation", "bottom"), (25, 2e-19, C["red"], "recombination", "bottom")):
    ax.text(x, y, lab, color=c, fontsize=12, fontweight="bold", va=va)
from matplotlib.lines import Line2D
h = [Line2D([], [], color="k", lw=2.4, label="AMJUEL 2020, used by the 3D FCI model ($n_e$ = 10$^{19}$ m$^{-3}$; CX at $T_{eff}=T_e$)"),
     Line2D([], [], color="k", lw=1.6, ls=(0, (4, 2.5)), label="SD1D fits, used by the 1D model (CX at 10 eV)")]
ax.legend(handles=h, loc="upper left")
fig.tight_layout(); save(fig, "fig4_rates")
