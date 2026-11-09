"""README figure 2: slab penetration of fluid neutrals against the analytic exp(-x/lambda), and lambda_fit/lambda against Kn
(plasma temperature sweep). Same slab as tests/fci_braginskii/test_fci_neutral_fluid.py::_penetration, returning the profiles.

Run from the repository root:

    PYTHONPATH=src:. python examples/neutrals/fig2_slab.py
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
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from drbx.fci_braginskii.native import fci_neutral_fluid as FL, fci_neutrals as F
from tests.fci_braginskii.test_fci_neutrals import PARAMS, _geometry

def slab(Te_v, steps=121):
    shape = (96, 4, 4)
    def cov(U, V, W):
        e = np.zeros(U.shape + (3, 3)); e[..., 0, 0] = e[..., 1, 1] = e[..., 2, 2] = 1.0
        return e
    n, Te, Ti, Tn = (jnp.full(shape, v) for v in (1.0, Te_v, 1.25 * Te_v, 1.25 * Te_v))
    nu_cx, nu_iz = (float(x[0, 0, 0]) for x in F.collision_frequencies(n, Te, Ti, PARAMS, Tn))
    Tn0 = float(Tn[0, 0, 0]); lam = float(np.sqrt(Tn0 / (nu_iz * (nu_cx + nu_iz)))); lam_mfp = float(np.sqrt(Tn0) / (nu_cx + nu_iz))
    length = 8.0 * lam
    geometry = _geometry(cov, shape, (length, 1.0, 1.0))
    grid, x = F.build_neutral_grid(geometry), np.asarray(geometry.grid.x.centers)
    dt, wall = 0.03 / (nu_cx + nu_iz), 0.3
    flux = wall * nu_iz * lam
    source = jnp.zeros(shape).at[-1].set(flux / (length / shape[0]))
    nn = jnp.asarray(wall * np.exp(-(length - x) / lam)[:, None, None] * np.ones(shape)); un, tn = jnp.zeros(shape), Tn
    step = jax.jit(lambda nn, un, tn: FL.neutral_fluid_step(n, jnp.zeros(shape), Te, Ti, nn, un, tn, dt, grid, PARAMS, volume_source=source))
    for _ in range(steps):
        out = step(nn, un, tn); nn, un, tn = out[4:7]
    line = np.asarray(nn[:, 0, 0]); w = (x > length - 4 * lam) & (x < length - 0.5 * lam)
    slope = np.polyfit(x[w], np.log(line[w]), 1)[0]
    return dict(x=x, n=line, lam=lam, lam_mfp=lam_mfp, length=length, wall=wall, ratio=1 / slope / lam, kn=lam_mfp / lam, nu_iz_over_nu=nu_iz / (nu_cx + nu_iz))

runs = [slab(t) for t in (0.12, 0.03, 0.05, 0.08, 0.2, 0.35, 0.6, 1.0, 2.0)]
d = dict(te=np.array([0.12, 0.03, 0.05, 0.08, 0.2, 0.35, 0.6, 1.0, 2.0]), **{k: np.array([r[k] for r in runs]) for k in runs[0]})
for i, t in enumerate(d["te"]): print(f"Te={t}: Kn={d['kn'][i]:.3f} ratio={d['ratio'][i]:.4f}")

keep = d["te"] > 0.04
i0 = int(np.where(d["te"] == 0.12)[0][0])
x, n, lam, L, wall = d["x"][i0], d["n"][i0], d["lam"][i0], d["length"][i0], d["wall"][i0]
dist = (L - x) / lam
fig, (a, b) = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
a.semilogy(dist, wall * np.exp(-dist), color=C["black"], lw=2, label=r"analytic $e^{-x/\lambda}$")
a.semilogy(dist[::-1][::2], n[::-1][::2] , "o", color=C["blue"], ms=5.5, mfc="none", mew=1.6, label="DRBX fluid neutrals")
a.set_xlim(0, 4.2); a.set_ylim(3e-3, 0.5)
a.set_xlabel(r"distance from wall $x/\lambda$"); a.set_ylabel(r"neutral density $n_n/n_0$")
a.set_title("Slab profile", fontsize=12)
a.legend(loc="upper right")
a.text(0.97, 0.52, rf"fitted $\lambda$ / analytic = {d['ratio'][i0]:.3f}" + "\n" + rf"at $Kn$ = {d['kn'][i0]:.2f}", transform=a.transAxes, ha="right", fontsize=11)
o = np.argsort(d["kn"][keep])
kn, r = d["kn"][keep][o], d["ratio"][keep][o]
b.axvspan(0.3, 0.7, color="#f1f1f1", zorder=0); b.text(0.5, 0.875, "Kn > 0.3:\nclosure\nlimit", ha="center", va="bottom", fontsize=11, color=C["grey"])
b.axhline(1, color=C["black"], lw=1.2); b.plot(kn, r, "o-", color=C["blue"], lw=2, ms=6)
b.set_xlim(0, 0.7); b.set_ylim(0.85, 1.03)
b.set_xlabel(r"Knudsen number $Kn=\lambda_{mfp}/\lambda$"); b.set_ylabel(r"fitted $\lambda$ / analytic $\lambda$")
b.set_title("Penetration length vs Kn", fontsize=12)
fig.suptitle(f"Neutral penetration matches the mean-free-path law to {abs(d['ratio'][i0]-1)*100:.1f}% at Kn = {d['kn'][i0]:.2f}", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "fig2_slab")
