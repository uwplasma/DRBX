from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def set_mpl_style() -> None:
    """Set a lightweight, publication-friendly Matplotlib style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_scan_panels(
    out_dir: str | Path,
    *,
    ky: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    gamma_iv: np.ndarray | None = None,
    title: str = "Linear scan",
    annotate_ky_star: bool = True,
    filename: str = "scan_panel.png",
) -> Path:
    """Save a compact multi-panel diagnostic for a ky scan."""

    out_dir = _ensure_dir(out_dir)
    ky = np.asarray(ky, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    omega = np.asarray(omega, dtype=float)

    ratio = np.maximum(gamma, 0.0) / ky
    i_star = int(np.argmax(ratio))
    ky_star = float(ky[i_star])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(ky, gamma, "o-", label="Re(eig)")
    if gamma_iv is not None:
        ax.plot(ky, np.asarray(gamma_iv, dtype=float), "s--", label="init-value")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Growth rate")
    if annotate_ky_star:
        ax.axvline(ky_star, color="k", alpha=0.25, linestyle="--")
    ax.legend()

    ax = axs[0, 1]
    ax.plot(ky, omega, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("Frequency (Im eigenvalue)")
    if annotate_ky_star:
        ax.axvline(ky_star, color="k", alpha=0.25, linestyle="--")

    ax = axs[1, 0]
    ax.plot(ky, ratio, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma,0)/k_y$")
    ax.set_title(r"Transport proxy $\max(\gamma,0)/k_y$")
    if annotate_ky_star:
        ax.axvline(ky_star, color="k", alpha=0.25, linestyle="--")
        ax.scatter([ky_star], [ratio[i_star]], color="k", zorder=5)

    ax = axs[1, 1]
    ax.plot(ky, np.maximum(gamma, 0.0), "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma, 0)$")
    ax.set_title("Unstable part")
    if annotate_ky_star:
        ax.axvline(ky_star, color="k", alpha=0.25, linestyle="--")

    fig.suptitle(title)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def save_kxky_heatmap(
    out_dir: str | Path,
    *,
    kx: np.ndarray,
    ky: np.ndarray,
    z: np.ndarray,
    zlabel: str,
    title: str,
    filename: str = "kxky_heatmap.png",
    cmap: str = "magma",
) -> Path:
    """Save a publication-friendly heatmap on a (kx, ky) grid."""

    out_dir = _ensure_dir(out_dir)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.shape != (kx.size, ky.size):
        raise ValueError("Expected z.shape == (len(kx), len(ky)).")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    im = ax.pcolormesh(ky, kx, z, shading="auto", cmap=cmap)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=zlabel)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def save_lp_history(
    out_dir: str | Path,
    *,
    history: np.ndarray,
    q: float | None = None,
    filename: str = "lp_history.png",
) -> Path:
    """Plot the fixed-point iteration history produced by solve_lp_fixed_point()."""

    out_dir = _ensure_dir(out_dir)
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2 or hist.shape[1] < 5:
        raise ValueError("Expected history with columns [Lp, ky*, gamma*, (gamma/ky)*, Lp_target].")

    it = np.arange(hist.shape[0])
    Lp = hist[:, 0]
    ky_star = hist[:, 1]
    ratio_star = hist[:, 3]
    Lp_target = hist[:, 4]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(it, Lp, "o-", label=r"$L_p$")
    ax.plot(it, Lp_target, "s--", label=r"target $q(\gamma/k_y)_{\max}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$L_p$")
    ax.legend()

    ax = axs[0, 1]
    ax.plot(it, ky_star, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$k_{y,*}$")
    ax.set_title(r"$k_{y,*}$ maximizing $\gamma/k_y$")

    ax = axs[1, 0]
    ax.plot(it, ratio_star, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$(\gamma/k_y)_{\max}$")

    ax = axs[1, 1]
    if q is None:
        ax.plot(ratio_star, Lp, "o-")
        ax.set_xlabel(r"$(\gamma/k_y)_{\max}$")
        ax.set_ylabel(r"$L_p$")
    else:
        ax.plot(ratio_star, Lp / q, "o-")
        ax.plot(ratio_star, ratio_star, "--", color="k", alpha=0.4)
        ax.set_xlabel(r"$(\gamma/k_y)_{\max}$")
        ax.set_ylabel(r"$L_p/q$")
        ax.set_title("Fixed-point condition")

    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def save_geometry_overview(
    out_dir: str | Path,
    *,
    geom,
    kx: float,
    ky: float,
    filename: str = "geometry_overview.png",
) -> Path:
    """Plot basic 1D geometry coefficients along the field line."""

    out_dir = _ensure_dir(out_dir)
    l = np.asarray(geom.l, dtype=float)
    kperp2 = np.asarray(geom.kperp2(float(kx), float(ky)), dtype=float)

    ones = np.ones_like(l, dtype=np.complex128)
    try:
        C1 = geom.curvature(float(kx), float(ky), ones)
        omega_d = np.asarray((-1j * C1).real, dtype=float)
    except Exception:
        omega_d = None

    B = None
    if hasattr(geom, "B") and callable(getattr(geom, "B")):
        try:
            B = np.asarray(geom.B(), dtype=float)
        except Exception:
            B = None

    import matplotlib.pyplot as plt

    nrows = 3 if B is not None else 2
    fig, axs = plt.subplots(
        nrows, 1, figsize=(9.0, 2.5 * nrows), sharex=True, constrained_layout=True
    )
    if nrows == 1:
        axs = [axs]

    ax = axs[0]
    ax.plot(l, kperp2, "-", color="C0")
    ax.set_ylabel(r"$k_\perp^2(l)$")
    ax.set_title(rf"$k_x={kx:g},\ k_y={ky:g}$")

    ax = axs[1]
    if omega_d is None:
        ax.text(0.5, 0.5, "curvature coeff unavailable", ha="center", va="center")
    else:
        ax.plot(l, omega_d, "-", color="C1")
    ax.set_ylabel(r"$\omega_d(l)$ (from curvature)")

    if B is not None:
        ax = axs[2]
        ax.plot(l, B, "-", color="C2")
        ax.set_ylabel(r"$B(l)$")

    axs[-1].set_xlabel(r"$l$")

    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def save_eigenfunction_panel(
    out_dir: str | Path,
    *,
    geom,
    state,
    eigenvalue: complex,
    kx: float,
    ky: float,
    kperp2_min: float = 1e-6,
    filename: str = "eigenfunctions.png",
) -> Path:
    """Plot eigenfunction structure along the field line.

    Works for all current model states by plotting any fields that are present on `state`.
    """

    out_dir = _ensure_dir(out_dir)
    l = np.asarray(geom.l, dtype=float)
    k2 = np.asarray(geom.kperp2(float(kx), float(ky)), dtype=float)

    omega = np.asarray(getattr(state, "omega"))
    k2_safe = np.maximum(k2, float(kperp2_min))
    phi = -omega / k2_safe

    fields: list[tuple[str, np.ndarray]] = []
    for name in ("n", "phi", "Te", "Ti", "vpar_e", "vpar_i", "psi", "omega"):
        if name == "phi":
            fields.append(("phi", np.asarray(phi)))
        elif hasattr(state, name):
            fields.append((name, np.asarray(getattr(state, name))))

    import matplotlib.pyplot as plt

    nplots = len(fields)
    ncols = 2
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(11.0, 2.6 * nrows), sharex=True, constrained_layout=True
    )
    axs = np.asarray(axs).reshape(-1)
    for ax, (name, f) in zip(axs, fields, strict=False):
        ax.plot(l, np.abs(f), "-", label="abs")
        ax.plot(l, np.real(f), "--", label="Re")
        ax.set_title(name)
        ax.legend(loc="upper right", frameon=False)

    for ax in axs[nplots:]:
        ax.axis("off")
    for ax in axs[max(0, nplots - ncols) : nplots]:
        ax.set_xlabel(r"$l$")

    fig.suptitle(rf"Leading mode: $\lambda={eigenvalue.real:+.3e}{eigenvalue.imag:+.3e}i$")
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def save_eigenvalue_spectrum(
    out_dir: str | Path,
    *,
    eigenvalues: np.ndarray,
    highlight: complex | None = None,
    filename: str = "spectrum.png",
) -> Path:
    """Plot a complex-plane eigenvalue spectrum scatter."""

    out_dir = _ensure_dir(out_dir)
    eig = np.asarray(eigenvalues, dtype=np.complex128).ravel()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.scatter(eig.real, eig.imag, s=18, alpha=0.8)
    if highlight is not None:
        ax.scatter([highlight.real], [highlight.imag], s=60, color="k", marker="x")
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    ax.set_title("Ritz spectrum (Arnoldi)")
    ax.axvline(0.0, color="k", alpha=0.2)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path
