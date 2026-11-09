"""README figure 3: independent wall-face conservation ledger on the 16^3 HSX run (800 steps, t = 0.6).

Input: the [wall], [wall-step] and [wall-run] lines printed by the run, saved as output/neutral_figures/<name>.wall.txt for the
run with flux-form perpendicular diffusion ("perpdiff") and the run with the previous perpendicular energy diffusion ("ledger").
Energy residual = (W_e + W_i + K_i)(t) - W(0) - integral of the wall-face energy flux, in the run's normalised energy units.

Run from the repository root:

    PYTHONPATH=src python examples/neutrals/fig3_conservation.py
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

import re, numpy as np

def load(name):
    wall, step, run = [], [], []
    for line in open(DATA / f"{name}.wall.txt"):
        f = lambda k: float(re.search(rf"\b{k}=([-+0-9.e]+)", line).group(1))
        if line.startswith("[wall]"):
            wall.append([f("t"), f("q_e"), f("q_i"), f("W_e"), f("W_i"), f("K_i"), f("N"), f("particle")])
        elif line.startswith("[wall-step]"):
            step.append([f("n"), f("W_e"), f("W_i")])
        elif line.startswith("[wall-run]"):
            m = re.search(r"\) W_e=([-+0-9.e]+) W_i=([-+0-9.e]+)|face\) W_e=([-+0-9.e]+) W_i=([-+0-9.e]+)", line)
            run.append([f("t"), f("closure"), float(m.group(1) or m.group(3)), float(m.group(2) or m.group(4))])
    return np.array(wall), np.array(step), np.array(run)

R = {k: load(k) for k in ("ledger", "perpdiff")}
before, after = R["ledger"], R["perpdiff"]

def totals(w, run):
    t = run[:, 0]; K = np.interp(t, w[:, 0], w[:, 5])
    loss = np.concatenate([[0], np.cumsum(0.5 * np.diff(w[:, 0]) * ((w[1:, 1] + w[1:, 2]) + (w[:-1, 1] + w[:-1, 2])))])
    return t, run[:, 2] + run[:, 3] + K, np.interp(t, w[:, 0], loss)

fig = plt.figure(figsize=(8.8, 4.3), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 1.0], wspace=0.1)
a, b, c = (fig.add_subplot(g) for g in gs)
w, s, run = after
t = np.arange(1, len(s) + 1) * 7.5e-4
for j, (lab, col) in enumerate((("particles", C["black"]), ("electron energy", C["red"]), ("ion energy", C["blue"]))):
    a.semilogy(t, np.abs(s[:, j]), color=col, lw=1.6, label=lab)
a.set_ylim(1e-5, 3e-3); a.set_xlim(0, 0.6)
a.set_xlabel(r"time $t$ ($L_{ref}/c_{s0}$)"); a.set_ylabel("|applied − wall face| / max")
a.set_title("Per step (after)", fontsize=12); a.legend(loc="lower left", fontsize=10, handlelength=1.2)

for (w, s, run), col, lab, ls in ((before, C["orange"], "before", "-"), (after, C["blue"], "after (flux-form)", "-")):
    t, tot, loss = totals(w, run)
    b.plot(t, tot * 1e3, color=col, lw=2.2, label=lab)
b.set_xlim(0, 0.6); b.set_xlabel(r"time $t$ ($L_{ref}/c_{s0}$)"); b.set_ylabel("energy residual ($10^{-3}$)")
b.set_title("Energy vs time", fontsize=12); b.legend(loc="lower left", fontsize=10, handlelength=1.2)

labels = ["particles", "$W_e$", "$W_i$", "total\nenergy"]
vals = {}
for key, (w, s, run) in R.items():
    t, tot, loss = totals(w, run)
    vals[key] = [run[-1, 1], abs(run[-1, 2]), abs(run[-1, 3]), abs(tot[-1])]
    print(key, "particle closure", run[-1, 1], "W_e", run[-1, 2], "W_i", run[-1, 3], "total", tot[-1], "wall energy loss", loss[-1], "total/loss %", 100 * tot[-1] / loss[-1])
np.savez(DATA / "fig3_values.npz", **{k: np.array(v) for k, v in vals.items()})
# energies in absolute normalised units, particles in fraction of wall loss -> two groups on separate axes
x = np.arange(3); wd = 0.38
for i, (key, col) in enumerate((("ledger", C["orange"]), ("perpdiff", C["blue"]))):
    v = vals[key]
    c.bar(x + (i - 0.5) * wd, np.array(v[1:]) * 1e3, wd, color=col)
    for xx, y in zip(x + (i - 0.5) * wd, np.array(v[1:]) * 1e3):
        c.text(xx, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
c.set_xticks(x, labels[1:]); c.set_ylabel("|residual| at $t$ = 0.6 ($10^{-3}$)"); c.set_ylim(0, 4.6)
c.set_title("Final energy", fontsize=12); c.grid(axis="x", visible=False)
pb, pa = vals["ledger"][0], vals["perpdiff"][0]
fig.suptitle(f"Flux-form diffusion cuts the energy residual {vals['ledger'][3]/vals['perpdiff'][3]:.1f}x; particles close to {pa*1e5:.1f}e-5", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "fig3_conservation")
