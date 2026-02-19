"""Generate the README schematic: flux-tube reduction to 1D along a field line.

This script is intentionally self-contained (Matplotlib only) so it can be re-run
to regenerate `docs/assets/images/flux_tube_coordinates.png`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    _set_style()

    import matplotlib.pyplot as plt

    out = Path("docs/assets/images/flux_tube_coordinates.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25])

    # --- Panel A: perpendicular Fourier representation (psi, alpha).
    axL = fig.add_subplot(gs[0, 0])
    axL.set_title("Perpendicular plane: Fourier mode in (ψ, α)", pad=8)

    # Background plane-wave patch (purely illustrative).
    n = 160
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.cos(3.0 * X + 2.0 * Y)
    axL.imshow(
        Z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    axL.set_xticks([])
    axL.set_yticks([])
    for sp in axL.spines.values():
        sp.set_visible(False)

    # Compact axis labels kept inside the panel (prevents clipping in README renderers).
    axL.text(0.98, 0.04, "ψ", transform=axL.transAxes, ha="right", va="bottom")
    axL.text(0.02, 0.96, "α", transform=axL.transAxes, ha="left", va="top")

    # kx/ky direction hints.
    axL.annotate(
        "",
        xy=(0.55, -0.55),
        xytext=(-0.2, -0.9),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "k"},
    )
    axL.text(0.58, -0.55, r"$k_\perp$", ha="left", va="center")

    axL.text(
        0.03,
        0.03,
        r"$\tilde f(\psi,\alpha,l,t)=\hat f(l,t)\,e^{i(k_x\psi+k_y\alpha)}$"
        "\n"
        r"$k_\perp^2(l)=k_x^2\,g^{\psi\psi}(l)+2k_xk_y\,g^{\psi\alpha}(l)+k_y^2\,g^{\alpha\alpha}(l)$",
        transform=axL.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
    )

    # --- Panel B: field-line coordinate (l) with geometry-provided coefficients.
    axR = fig.add_subplot(gs[0, 1])
    axR.set_title("Field line: evolve 1D profiles in ℓ", pad=8)
    axR.set_xticks([])
    axR.set_yticks([])
    axR.set_xlim(0.0, 1.0)
    axR.set_ylim(-1.05, 1.05)
    for sp in axR.spines.values():
        sp.set_visible(False)

    s = np.linspace(0.0, 1.0, 400)
    curve = 0.75 * np.sin(2.0 * np.pi * (s - 0.05))
    axR.plot(s, curve, color="#c62828", lw=3)
    axR.scatter([0.0, 1.0], [curve[0], curve[-1]], s=55, color="k", zorder=5)
    axR.text(0.0, curve[0] - 0.12, "sheath end", ha="left", va="top")
    axR.text(1.0, curve[-1] - 0.12, "sheath end", ha="right", va="top")
    axR.annotate(
        "",
        xy=(0.97, curve[-1]),
        xytext=(0.90, curve[-1] + 0.12),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "#c62828"},
    )

    axR.text(
        0.04,
        0.92,
        "Geometry provides along ℓ:\n"
        r"• $\nabla_\parallel=b\cdot\nabla$"
        "\n"
        r"• curvature drive $C(\cdot)$"
        "\n"
        r"• metric → $k_\perp^2(\ell)$",
        transform=axR.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75"},
    )

    # Small inset showing k_perp^2(l) varying along the line.
    ax_in = axR.inset_axes([0.58, 0.62, 0.38, 0.30])
    l = np.linspace(-np.pi, np.pi, 200)
    k2 = 1.0 + 0.35 * np.cos(l) + 0.15 * np.cos(2 * l)
    ax_in.plot(l, k2, color="C0", lw=2)
    ax_in.set_title(r"$k_\perp^2(\ell)$", fontsize=10)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_alpha(0.4)

    # Polarization closure callout (kept separate, no overlap).
    fig.text(
        0.52,
        0.03,
        r"Polarization closure (Boussinesq flux-tube):  $\Omega(\ell)=-k_\perp^2(\ell)\,\phi(\ell)$  ⇒  "
        r"$\phi(\ell)=-\Omega(\ell)/k_\perp^2(\ell)$",
        ha="center",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.80"},
    )

    fig.suptitle("How jaxdrb reduces 3D SOL/edge physics to 1D along a field line", y=0.98)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.20, wspace=0.22)
    fig.savefig(out)
    plt.close(fig)

    print(f"[make_flux_tube_coordinates] wrote {out}")


if __name__ == "__main__":
    main()
