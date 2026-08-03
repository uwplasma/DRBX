"""Figures for the tokamak island-chain example.

Two standalone deliverables:

1. ``poincare`` (default, no data needed): Poincare sections of the reduced
   island field at three perturbation amplitudes, with the measured island
   width tested against the pendulum-model prediction

       W = 4 sqrt(eps / (m |iota'|))

   (the standard magnetic-island width, e.g. J. Wesson, *Tokamaks*, ch. 7;
   R. B. White, *The Theory of Toroidally Confined Plasmas*). This is the
   geometry-verification anchor: the traced chains must reproduce both the
   resonance location iota = n/m and the sqrt(eps) width scaling.

2. ``evolution`` (needs the production npz from
   ``island_tokamak_profiles.py``): the profile-evolution summary -- density
   cross-section with the analytic separatrix overlaid, the space-time map
   of the mean profile, the particle-balance trace -- plus an animated GIF
   of the zeta = 0 density cross-section (the movie for the README).

    python examples/island_tokamak_figure.py poincare
    python examples/island_tokamak_figure.py evolution output/island_tokamak/island_media.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The reduced field of the example (island_tokamak_profiles.py).
X_MIN, X_MAX = 0.2, 1.0
IOTA_AXIS, IOTA_EDGE = 0.56, 0.44
M_RES, N_RES = 2, 1
IOTA_PRIME = (IOTA_EDGE - IOTA_AXIS) / (X_MAX - X_MIN)
X_RES = X_MIN + (N_RES / M_RES - IOTA_AXIS) / IOTA_PRIME
MEDIA = Path("docs/media")


def iota(x):
    return IOTA_AXIS + IOTA_PRIME * (x - X_MIN)


def trace_full(x0, theta0, eps, transits=400, steps_per=64):
    """Trace with explicit zeta dependence (exact reduced system)."""
    h = 2.0 * np.pi / steps_per
    x, th, z = x0, theta0, 0.0
    px, pth = [], []
    for i in range(transits * steps_per):
        def rhs(x_, th_, z_):
            return eps * np.sin(M_RES * th_ - N_RES * z_), iota(x_)
        k1 = rhs(x, th, z)
        k2 = rhs(x + 0.5 * h * k1[0], th + 0.5 * h * k1[1], z + 0.5 * h)
        k3 = rhs(x + 0.5 * h * k2[0], th + 0.5 * h * k2[1], z + 0.5 * h)
        k4 = rhs(x + h * k3[0], th + h * k3[1], z + h)
        x += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        th += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += h
        if (i + 1) % steps_per == 0:
            px.append(x), pth.append(np.mod(th, 2.0 * np.pi))
    return np.asarray(px), np.asarray(pth)


def measured_width(eps):
    """Island full width from a near-separatrix trace.

    A field line launched just inside the X-point traces the separatrix; its
    radial extent is the island width. In the co-rotating angle
    psi = m theta - n zeta the X-point sits at psi = pi (iota' < 0 here), so
    at zeta = 0 launch at theta = pi / m with a small radial offset.
    """
    px, _ = trace_full(X_RES + 1e-3, np.pi / M_RES, eps, transits=600)
    # a single near-separatrix orbit covers ONE lobe (X-point to O-column and
    # back), i.e. the island half-width; the full width is twice that.
    return 2.0 * float(px.max() - px.min())


def fig_poincare():
    eps_list = (0.006, 0.012, 0.030)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, eps in zip(axes[:3], eps_list):
        rng = np.random.default_rng(1)
        for x0 in np.linspace(X_MIN + 0.05, X_MAX - 0.05, 26):
            px, pth = trace_full(x0, rng.uniform(0, 2 * np.pi), eps, transits=300)
            ax.plot(pth, px, ".", ms=0.4, color="#1f77b4", alpha=0.6)
        # analytic separatrix of the pendulum reduction:
        # Delta x(psi) = +/- sqrt(2 eps (1 + cos psi) / (m |iota'|)), psi = m theta
        th_sep = np.linspace(0, 2 * np.pi, 600)
        dx_sep = np.sqrt(2.0 * eps * (1.0 + np.cos(M_RES * th_sep))
                         / (M_RES * abs(IOTA_PRIME)))
        ax.plot(th_sep, X_RES + dx_sep, color="#d62728", lw=1.2)
        ax.plot(th_sep, X_RES - dx_sep, color="#d62728", lw=1.2)
        ax.axhline(X_RES, color="#d62728", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(rf"$\epsilon$ = {eps:g}", fontsize=10)
        ax.set_xlabel(r"$\theta$"); ax.set_ylim(X_MIN, X_MAX)
        ax.set_xlim(0, 2 * np.pi)
    axes[0].set_ylabel(r"$x$")

    b = axes[3]
    eps_scan = np.array([0.0005, 0.001, 0.002, 0.004, 0.006, 0.012, 0.030])
    Wm = np.array([measured_width(e) for e in eps_scan])
    Wp = 4.0 * np.sqrt(eps_scan / (M_RES * abs(IOTA_PRIME)))
    b.loglog(eps_scan, Wp, "-", color="#d62728", lw=1.8,
             label=r"$W = 4\sqrt{\epsilon/(m|\iota'|)}$")
    b.loglog(eps_scan, Wm, "o", color="#1f77b4", ms=7, label="traced separatrix")
    b.axhline((X_MAX - X_MIN), color="k", ls=":", lw=1, alpha=0.7)
    b.text(eps_scan[0] * 1.1, (X_MAX - X_MIN) * 1.04, "domain width",
           fontsize=7.5)
    b.set_xlabel(r"$\epsilon$"); b.set_ylabel(r"island width $W$")
    b.set_title("width scaling: pendulum regime\nand finite-window saturation",
                fontsize=9)
    b.legend(fontsize=8, loc="lower right"); b.grid(alpha=0.3, which="both")

    fig.suptitle(r"The $(m,n)=(2,1)$ internal island chain at $\iota = 1/2$: "
                 "Poincare sections and the pendulum-model width",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(MEDIA / "island_tokamak_poincare.png", dpi=150)
    small = eps_scan <= 0.004
    rms = float(np.sqrt(np.mean((Wm[small] / Wp[small] - 1) ** 2)))
    print(f"poincare figure written; pendulum-regime rms dev {100*rms:.0f}%")


def fig_evolution(npz_path):
    d = np.load(npz_path)
    xn = np.asarray(d["xn"]); times = np.asarray(d["times"])
    prof = np.asarray(d["profile"])
    sl_n = np.asarray(d["slice_density"])
    sinks = np.asarray(d["sheath_loss"]) + np.asarray(d["wall_loss"])
    theta = np.linspace(0, 2 * np.pi, sl_n.shape[2], endpoint=False)

    EPS_RUN = 0.03                    # the production run's island amplitude
    xres_n = (X_RES - X_MIN) / (X_MAX - X_MIN)

    def separatrix_xn(th):
        """Pendulum separatrix in normalized radius, psi = m theta at zeta=0."""
        dx = np.sqrt(2.0 * EPS_RUN * (1.0 + np.cos(M_RES * th))
                     / (M_RES * abs(IOTA_PRIME)))
        return (np.clip(X_RES + dx, X_MIN, X_MAX) - X_MIN) / (X_MAX - X_MIN), \
               (np.clip(X_RES - dx, X_MIN, X_MAX) - X_MIN) / (X_MAX - X_MIN)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    a, b, c = axes
    im = a.pcolormesh(theta, xn, sl_n[-1], cmap="inferno", shading="auto")
    th_sep = np.linspace(0, 2 * np.pi, 400)
    up, lo = separatrix_xn(th_sep)
    a.plot(th_sep, up, color="cyan", lw=1.1, alpha=0.9)
    a.plot(th_sep, lo, color="cyan", lw=1.1, alpha=0.9)
    a.set_ylim(0, 1)
    a.set_xlabel(r"$\theta$"); a.set_ylabel(r"$x_n$")
    a.set_title(r"(a) density at $\zeta = 0$, final state (separatrix overlaid)",
                fontsize=10, loc="left")
    plt.colorbar(im, ax=a, pad=0.02)

    im2 = b.pcolormesh(times, xn, prof.T, cmap="viridis", shading="auto")
    b.axhline(xres_n, color="w", ls="--", lw=1)
    b.set_xlabel(r"$t\,[a/c_s]$"); b.set_ylabel(r"$x_n$")
    b.set_title(r"(b) mean-profile evolution $\langle n\rangle(x, t)$",
                fontsize=10, loc="left")
    plt.colorbar(im2, ax=b, pad=0.02)

    import json
    bal = Path(str(npz_path).replace(".npz", "_balance.json"))
    src_total = None
    if bal.exists():
        rec = json.loads(bal.read_text())
        # sinks_over_source at steady state recovers the source normalization
        src_total = sinks[-len(sinks) // 4:].mean() / max(
            rec.get("sinks_over_source", 1.0), 1e-30)
    if src_total:
        c.plot(times, sinks / src_total, color="#1f77b4", lw=1.5)
        c.axhline(1.0, color="k", ls=":", lw=1, label="sinks = source")
        c.set_ylabel("(sheath + wall) / source")
        c.legend(fontsize=8)
    else:
        c.plot(times, sinks, color="#1f77b4", lw=1.5)
        c.set_ylabel("total sinks")
    c.set_xlabel(r"$t\,[a/c_s]$")
    c.set_title("(c) particle balance approaching steady state",
                fontsize=10, loc="left")
    c.grid(alpha=0.3)

    fig.suptitle("Source-driven profile evolution across the internal island chain",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(MEDIA / "island_tokamak_evolution.png", dpi=150)
    print("evolution figure written")

    # the movie: zeta = 0 density cross-section over time
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio missing -- skipping the GIF")
        return
    frames = []
    vmin, vmax = np.percentile(sl_n[len(sl_n) // 4:], [1, 99])
    for i in range(0, len(sl_n), max(1, len(sl_n) // 120)):
        f, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.pcolormesh(theta, xn, sl_n[i], cmap="inferno", shading="auto",
                      vmin=vmin, vmax=vmax)
        up_g, lo_g = separatrix_xn(th_sep)
        ax.plot(th_sep, up_g, color="cyan", lw=0.9, alpha=0.85)
        ax.plot(th_sep, lo_g, color="cyan", lw=0.9, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$x_n$")
        ax.set_title(rf"$n(x,\theta)$ at $\zeta=0$,  $t$ = {times[i]:.1f} $a/c_s$",
                     fontsize=10)
        f.tight_layout()
        f.canvas.draw()
        frames.append(np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(f)
    imageio.mimsave(MEDIA / "island_tokamak_evolution.gif", frames, fps=12, loop=0)
    print(f"movie written ({len(frames)} frames)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "poincare"
    if mode == "poincare":
        fig_poincare()
    else:
        fig_evolution(sys.argv[2] if len(sys.argv) > 2 else
                      "output/island_tokamak/island_media.npz")
