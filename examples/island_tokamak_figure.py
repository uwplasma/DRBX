"""Figures for the tokamak island-chain example.

Four modes:

* ``poincare`` (default, no data needed): Poincare sections of the reduced
  island field at three perturbation amplitudes, with the measured island
  width tested against the pendulum-model prediction

      W = 4 sqrt(eps / (m |iota'|))

  (the standard magnetic-island width, e.g. J. Wesson, *Tokamaks*, ch. 7;
  R. B. White, *The Theory of Toroidally Confined Plasmas*). This is the
  geometry-verification anchor: the traced chains must reproduce both the
  resonance location iota = n/m and the sqrt(eps) width scaling.

* ``evolution`` (needs the production npz from
  ``island_tokamak_profiles.py``): the profile-evolution summary -- density
  cross-section with the analytic separatrix overlaid, the space-time map
  of the mean profile, the particle-balance trace -- plus an animated GIF
  of the zeta = 0 density cross-section.

* ``3d``: static 3-D snapshot -- island field lines, density rings, and the
  q = 1/iota profile.

* ``dashboard``: the README movie. Top row: the animated 3-D state inside a
  low-opacity plasma-boundary surface, the safety-factor profile, and the
  (R, Z) Poincare section at a single toroidal angle. Bottom row: the
  evolving mean profile, the evolving turbulent radial particle flux, and
  the fluctuation-energy / sinks-over-source time traces with a cursor.

    python examples/island_tokamak_figure.py poincare
    python examples/island_tokamak_figure.py dashboard output/island_tokamak/island_tokamak.npz
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
    if sl_n.ndim == 4:                       # four-plane format: take zeta = 0
        sl_n = sl_n[:, :, :, 0]
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



# --- 3-D rendering -----------------------------------------------------------
R0_TORUS, ELONG = 3.0, 1.35


def to_xyz(x, th, z):
    """(x, theta, zeta) -> cartesian on the elongated torus."""
    R = R0_TORUS + x * np.cos(th)
    return R * np.cos(z), R * np.sin(z), ELONG * x * np.sin(th)


def line3d(x0, th0, eps, transits=40, steps_per=180):
    """Continuous 3-D field line (not the Poincare subsample)."""
    h = 2.0 * np.pi / steps_per
    x, th, z = x0, th0, 0.0
    pts = []
    for _ in range(transits * steps_per):
        def rhs(x_, th_, z_):
            return eps * np.sin(M_RES * th_ - N_RES * z_), iota(x_)
        k1 = rhs(x, th, z)
        k2 = rhs(x + 0.5 * h * k1[0], th + 0.5 * h * k1[1], z + 0.5 * h)
        k3 = rhs(x + 0.5 * h * k2[0], th + 0.5 * h * k2[1], z + 0.5 * h)
        k4 = rhs(x + h * k3[0], th + h * k3[1], z + h)
        x += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        th += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += h
        pts.append((x, th, z))
    a = np.asarray(pts)
    return to_xyz(a[:, 0], a[:, 1], a[:, 2])


def _annulus(ax, dens2d, xn, zeta, cmap, norm_):
    """Poloidal cross-section ring at toroidal angle zeta, colored by density."""
    x_phys = X_MIN + xn * (X_MAX - X_MIN)
    th = np.linspace(0, 2 * np.pi, dens2d.shape[1] + 1)
    Xg, THg = np.meshgrid(x_phys, th, indexing="ij")
    dens = np.concatenate([dens2d, dens2d[:, :1]], axis=1)
    R = R0_TORUS + Xg * np.cos(THg)
    Xc = R * np.cos(zeta); Yc = R * np.sin(zeta); Zc = ELONG * Xg * np.sin(THg)
    ax.plot_surface(Xc, Yc, Zc, facecolors=cmap(norm_(dens)), shade=False,
                    rstride=1, cstride=1, antialiased=False, alpha=0.95)


def fig_3d(npz_path, eps_run=0.03):
    """3-D view: island field lines + density cross-section + q profile."""
    from matplotlib import cm, colors, gridspec

    d = np.load(npz_path)
    xn = np.asarray(d["xn"])
    dens3 = np.asarray(d["density"])          # final full 3-D snapshot

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1.0], figure=fig)
    ax = fig.add_subplot(gs[0], projection="3d")

    cmap = cm.inferno
    norm_ = colors.Normalize(*np.percentile(dens3, [2, 98]))
    _annulus(ax, dens3[:, :, 0], xn, 0.0, cmap, norm_)
    _annulus(ax, dens3[:, :, dens3.shape[2] // 2], xn, np.pi, cmap, norm_)

    # confined surfaces inside and outside the chain
    for x0 in (0.35, 0.90):
        X, Y, Z = line3d(x0, 0.3, eps_run, transits=9)
        ax.plot(X, Y, Z, color="#4c72b0", lw=0.6, alpha=0.55)
    # the island: separatrix (red) and a core line inside one island tube
    X, Y, Z = line3d(X_RES + 1e-3, np.pi / M_RES, eps_run, transits=14)
    ax.plot(X, Y, Z, color="#d62728", lw=0.9, alpha=0.95)
    X, Y, Z = line3d(X_RES + 0.05, 0.0, eps_run, transits=14)
    ax.plot(X, Y, Z, color="#ff7f0e", lw=1.0, alpha=0.95)

    lim = (R0_TORUS + X_MAX) * 0.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(-ELONG * X_MAX * 1.1, ELONG * X_MAX * 1.1)
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_axis_off()
    ax.view_init(elev=26, azim=-68)
    ax.set_title("blue: confined surfaces   red: island separatrix   "
                 "orange: island core\n"
                 "cross-sections colored by the final density", fontsize=9.5)

    b = fig.add_subplot(gs[1])
    xg = np.linspace(X_MIN, X_MAX, 200)
    b.plot((xg - X_MIN) / (X_MAX - X_MIN), 1.0 / iota(xg), lw=2, color="#4c72b0")
    b.axhline(2.0, color="#d62728", ls="--", lw=1.2)
    xres_n = (X_RES - X_MIN) / (X_MAX - X_MIN)
    b.plot([xres_n], [2.0], "o", color="#d62728", ms=9)
    b.annotate("(2,1) island chain", (xres_n + 0.04, 2.005),
               color="#d62728", fontsize=9.5, va="bottom")
    b.set_xlabel(r"$x_n$"); b.set_ylabel(r"safety factor $q = 1/\iota$")
    b.set_title("q profile crossing the rational surface", fontsize=10)
    b.grid(alpha=0.3)

    fig.suptitle("The tokamak with an internal (2,1) island chain",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(MEDIA / "island_tokamak_3d.png", dpi=150)
    print("3-D figure written")

def fig_dashboard(npz_path, eps_run=0.03):
    """The README movie: 3-D view + geometry context + evolving diagnostics.

    Row 1: the animated 3-D state (field lines, density cross-section rings,
    and a low-opacity plasma-boundary surface), the safety-factor profile,
    and the (R, Z) Poincare section at a single toroidal angle extending a
    little beyond the plasma boundary so the island structure is visible.
    Row 2, the standard flux-driven diagnostics (GBS / TOKAM3X practice):
    the evolving mean profile against the initial one with the island band
    marked, the evolving turbulent radial particle-flux profile, and the
    time traces (fluctuation energy, sinks/source) with a time cursor.
    """
    import json as _json

    from matplotlib import cm, colors, gridspec

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio missing -- cannot build the dashboard movie")
        return
    d = np.load(npz_path)
    xn = np.asarray(d["xn"]); times = np.asarray(d["times"])
    prof = np.asarray(d["profile"])
    flux = np.asarray(d["flux_profile"])
    energy = np.asarray(d["energy"])
    sinks = np.asarray(d["sheath_loss"]) + np.asarray(d["wall_loss"])
    sl = np.asarray(d["slice_density"])
    if sl.ndim != 4:
        print("npz lacks 4-plane slices"); return
    bal = Path(str(npz_path).replace(".npz", "_balance.json"))
    src_norm = None
    if bal.exists():
        rec = _json.loads(bal.read_text())
        src_norm = sinks[-len(sinks) // 4:].mean() / max(
            rec.get("sinks_over_source", 1.0), 1e-30)
    sinks_n = sinks / src_norm if src_norm else sinks / max(sinks.max(), 1e-30)

    zetas = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    cmap = cm.inferno
    norm_ = colors.Normalize(*np.percentile(sl[len(sl) // 4:], [2, 98]))
    xres_n = (X_RES - X_MIN) / (X_MAX - X_MIN)
    # at this amplitude the island fills most of the radius (see the
    # Poincare panel), so mark only the resonant surface, not a band

    def island_band(ax_):
        ax_.axvline(xres_n, color="#d62728", ls="--", lw=1.0, alpha=0.75)
        ax_.set_xlim(0.0, 1.0)

    # static geometry, computed once ---------------------------------------
    lines = []
    for x0 in (0.35, 0.90):
        lines.append((line3d(x0, 0.3, eps_run, transits=9), "#4c72b0", 0.6, 0.5))
    lines.append((line3d(X_RES + 1e-3, np.pi / M_RES, eps_run, transits=14),
                  "#d62728", 0.9, 0.9))
    lines.append((line3d(X_RES + 0.05, 0.0, eps_run, transits=14),
                  "#ff7f0e", 1.0, 0.9))
    # low-opacity plasma-boundary surface at x = X_MAX
    thb = np.linspace(0, 2 * np.pi, 40)
    zb = np.linspace(0, 2 * np.pi, 80)
    THB, ZB = np.meshgrid(thb, zb, indexing="ij")
    RB = R0_TORUS + X_MAX * np.cos(THB)
    BX, BY, BZ = RB * np.cos(ZB), RB * np.sin(ZB), ELONG * X_MAX * np.sin(THB)

    # (R, Z) Poincare at zeta = 0, extending past the boundary
    rng = np.random.default_rng(3)
    pr, pz = [], []
    for x0 in np.linspace(X_MIN + 0.03, X_MAX + 0.12, 30):
        px, pth = trace_full(x0, rng.uniform(0, 2 * np.pi), eps_run, transits=250)
        pr.append(R0_TORUS + px * np.cos(pth))
        pz.append(ELONG * px * np.sin(pth))
    pr = np.concatenate(pr); pz = np.concatenate(pz)
    thc = np.linspace(0, 2 * np.pi, 200)

    frames = []
    step = max(1, len(sl) // 100)
    for i in range(0, len(sl), step):
        fig = plt.figure(figsize=(12.6, 7.6))
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0],
                               width_ratios=[1.25, 1.0, 1.0],
                               hspace=0.34, wspace=0.30,
                               left=0.055, right=0.985, top=0.90, bottom=0.08)
        fig.suptitle("Source-driven turbulence on a tokamak with a (2,1) "
                     f"island chain      $t$ = {times[i]:.1f} $a/c_s$",
                     fontsize=12, fontweight="bold")

        ax = fig.add_subplot(gs[0, 0], projection="3d")
        for (X, Y, Z), col, lw, al in lines:
            ax.plot(X, Y, Z, color=col, lw=lw, alpha=al)
        for k, z in enumerate(zetas):
            _annulus(ax, sl[i][:, :, k], xn, z, cmap, norm_)
        ax.plot_surface(BX, BY, BZ, color="#7799bb", alpha=0.07,
                        linewidth=0, antialiased=False, shade=False)
        lim = (R0_TORUS + X_MAX) * 0.72
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_zlim(-ELONG * X_MAX * 1.1, ELONG * X_MAX * 1.1)
        ax.set_box_aspect((1, 1, 0.42)); ax.set_axis_off()
        ax.view_init(elev=26, azim=-68)
        ax.set_title("density cross-sections, island field lines,\n"
                     "plasma boundary", fontsize=9)

        b = fig.add_subplot(gs[0, 1])
        xg = np.linspace(X_MIN, X_MAX, 200)
        b.plot((xg - X_MIN) / (X_MAX - X_MIN), 1.0 / iota(xg), lw=2,
               color="#4c72b0")
        b.axhline(2.0, color="#d62728", ls="--", lw=1.1)
        b.plot([xres_n], [2.0], "o", color="#d62728", ms=8)
        b.annotate("(2,1) island", (xres_n + 0.05, 2.005), color="#d62728",
                   fontsize=8.5, va="bottom")
        b.set_xlabel(r"$x_n$", fontsize=9)
        b.set_ylabel(r"safety factor $q$", fontsize=9)
        b.set_title("q profile", fontsize=9)
        b.tick_params(labelsize=8); b.grid(alpha=0.3)

        c = fig.add_subplot(gs[0, 2])
        c.plot(pr, pz, ".", ms=0.5, color="#1f77b4", alpha=0.55)
        c.plot(R0_TORUS + X_MAX * np.cos(thc), ELONG * X_MAX * np.sin(thc),
               color="k", lw=1.4, alpha=0.75)
        c.plot(R0_TORUS + X_MIN * np.cos(thc), ELONG * X_MIN * np.sin(thc),
               color="k", lw=0.8, ls=":", alpha=0.6)
        c.set_aspect("equal")
        c.set_xlabel(r"$R$", fontsize=9); c.set_ylabel(r"$Z$", fontsize=9)
        c.set_title(r"Poincare section at $\phi = 0$"
                    "\n(solid: plasma boundary)", fontsize=9)
        c.tick_params(labelsize=8)

        e = fig.add_subplot(gs[1, 0])
        e.plot(xn, prof[0], color="gray", ls="--", lw=1.2, label="initial")
        e.plot(xn, prof[i], color="#1f77b4", lw=2, label="now")
        island_band(e)
        e.set_ylim(0, max(prof.max() * 1.05, 1e-3))
        e.set_xlabel(r"$x_n$", fontsize=9)
        e.set_ylabel(r"$\langle n \rangle$", fontsize=9)
        e.set_title("mean density profile (dashed: resonant surface)",
                    fontsize=9)
        e.legend(fontsize=7.5, loc="upper right")
        e.tick_params(labelsize=8); e.grid(alpha=0.3)

        f = fig.add_subplot(gs[1, 1])
        f.plot(xn, flux[i], color="#6a0dad", lw=1.8)
        island_band(f)
        fmax = np.percentile(np.abs(flux[len(flux) // 4:]), 99.5)
        f.set_ylim(-0.25 * fmax, 1.1 * fmax)
        f.axhline(0, color="k", lw=0.6)
        f.set_xlabel(r"$x_n$", fontsize=9)
        f.set_ylabel(r"$\Gamma_r = \langle \tilde n\, \tilde v_r \rangle$",
                     fontsize=9)
        f.set_title("turbulent radial particle flux", fontsize=9)
        f.tick_params(labelsize=8); f.grid(alpha=0.3)

        g = fig.add_subplot(gs[1, 2])
        g.plot(times, energy, color="#1f77b4", lw=1.5,
               label=r"fluctuation energy $\langle \tilde n^2 \rangle$")
        g.plot(times, sinks_n, color="#2ca02c", lw=1.5, label="sinks / source")
        g.axvline(times[i], color="k", lw=1.0)
        g.set_xlabel(r"$t\,[a/c_s]$", fontsize=9)
        g.set_title("time traces", fontsize=9)
        g.legend(fontsize=7.5, loc="center right")
        g.tick_params(labelsize=8); g.grid(alpha=0.3)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(MEDIA / "island_tokamak_3d.gif", frames, fps=10, loop=0)
    print(f"dashboard movie written ({len(frames)} frames)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "poincare"
    if mode == "poincare":
        fig_poincare()
    elif mode == "3d":
        fig_3d(sys.argv[2] if len(sys.argv) > 2 else
               "output/island_tokamak/island_media.npz")
    elif mode == "dashboard":
        fig_dashboard(sys.argv[2] if len(sys.argv) > 2 else
                      "output/island_tokamak/island_turb.npz")
    else:
        fig_evolution(sys.argv[2] if len(sys.argv) > 2 else
                      "output/island_tokamak/island_media.npz")
