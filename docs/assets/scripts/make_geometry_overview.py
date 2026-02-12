"""Generate a geometry overview schematic for the README and docs.

Writes:
  - docs/assets/images/geometry_overview.png

Design goals
------------
- Explain (at a glance) what "1D field-line / flux-tube", "2D perpendicular box",
  and "3D FCI plane stack" mean in `jaxdrb`.
- Avoid overlapping text and keep the figure readable in GitHub markdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
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

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    out = Path("docs/assets/images/geometry_overview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.6, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25], wspace=0.25)

    # -----------------------------------------------------------------------------
    # Panel A: 2D perpendicular plane (x,y) with B out of plane.
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("2D: perpendicular box (x,y)", pad=6)
    ax0.set_aspect("equal")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_alpha(0.5)

    # Domain box with periodic arrows.
    ax0.add_patch(patches.Rectangle((0.08, 0.08), 0.84, 0.84, fill=False, lw=2, ec="0.2"))
    for x0, y0, x1, y1 in [
        (0.08, 0.50, 0.02, 0.50),
        (0.92, 0.50, 0.98, 0.50),
        (0.50, 0.08, 0.50, 0.02),
        (0.50, 0.92, 0.50, 0.98),
    ]:
        ax0.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "0.35"},
        )
    ax0.text(0.5, -0.02, "periodic BCs (default)", ha="center", va="top", transform=ax0.transAxes)

    # Coordinate arrows.
    ax0.annotate(
        "",
        xy=(0.50, 0.08),
        xytext=(0.15, 0.08),
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "k"},
    )
    ax0.text(0.16, 0.11, "x (radial-like)", ha="left", va="bottom")
    ax0.annotate(
        "",
        xy=(0.08, 0.50),
        xytext=(0.08, 0.15),
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "k"},
    )
    ax0.text(0.11, 0.16, "y (binormal/poloidal-like)", rotation=90, ha="left", va="bottom")

    # B out-of-plane symbol.
    ax0.add_patch(patches.Circle((0.82, 0.82), 0.06, fill=False, lw=2.0, ec="#1565c0"))
    ax0.add_patch(patches.Circle((0.82, 0.82), 0.015, color="#1565c0"))
    ax0.text(0.82, 0.90, r"$\mathbf{B}$", ha="center", va="bottom", color="#1565c0")
    ax0.text(
        0.10,
        0.92,
        "Used for:\n• conservative kernels\n• budgets/gates\n• solver tests",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8"},
    )

    # -----------------------------------------------------------------------------
    # Panel B: 1D field-line (flux-tube) model.
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("1D: flux-tube / field line (ℓ)", pad=6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-1.05, 1.05)

    s = np.linspace(0, 1, 250)
    y = 0.7 * np.sin(2 * np.pi * (s - 0.08))
    ax1.plot(s, y, color="#c62828", lw=3)
    ax1.scatter([0, 1], [y[0], y[-1]], s=50, color="k", zorder=3)
    ax1.text(0, y[0] - 0.12, "end", ha="left", va="top")
    ax1.text(1, y[-1] - 0.12, "end", ha="right", va="top")

    ax1.text(
        0.05,
        0.92,
        r"$\tilde f(\psi,\alpha,\ell,t)=\hat f(\ell,t)\,e^{i(k_x\psi+k_y\alpha)}$"
        "\n"
        r"$k_\perp^2(\ell)$ from metric;  $\nabla_\parallel=b\cdot\nabla$",
        ha="left",
        va="top",
        transform=ax1.transAxes,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8"},
    )
    ax1.text(
        0.5,
        -0.02,
        "Used for: linear stability + matrix-free J·v",
        ha="center",
        va="top",
        transform=ax1.transAxes,
    )

    # -----------------------------------------------------------------------------
    # Panel C: 3D FCI plane stack with target intersections.
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("3D: FCI plane stack (targets/plates)", pad=6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Planes along toroidal angle (schematic).
    plane_x = [0.12, 0.36, 0.60, 0.84]
    for px in plane_x:
        ax2.plot([px, px], [0.15, 0.85], color="0.65", lw=2)
    ax2.text(0.12, 0.88, r"$\varphi_0$", ha="center", va="bottom", color="0.35")
    ax2.text(0.84, 0.88, r"$\varphi_{k}$", ha="center", va="bottom", color="0.35")
    ax2.annotate(
        "",
        xy=(0.84, 0.10),
        xytext=(0.12, 0.10),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "0.35"},
    )
    ax2.text(0.48, 0.06, "toroidal angle / plane index", ha="center", va="top", color="0.35")

    # Field-line mapping curve between planes.
    tt = np.linspace(0, 1, 220)
    x = 0.12 + 0.72 * tt
    curve = 0.55 + 0.22 * np.sin(2 * np.pi * (tt - 0.1))
    ax2.plot(x, curve, color="#1565c0", lw=3)
    ax2.scatter([x[0], x[-1]], [curve[0], curve[-1]], s=35, color="#1565c0")

    # Target plates at the ends (schematic).
    ax2.add_patch(patches.Rectangle((0.05, 0.20), 0.03, 0.60, color="0.15"))
    ax2.add_patch(patches.Rectangle((0.92, 0.20), 0.03, 0.60, color="0.15"))
    ax2.text(0.05, 0.82, "plate", ha="left", va="bottom", color="0.15")
    ax2.text(0.92, 0.82, "plate", ha="left", va="bottom", color="0.15")

    ax2.text(
        0.52,
        0.92,
        "Parallel operator uses maps + Δℓ + hit metadata\n(one-sided near targets)",
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8"},
    )
    ax2.text(
        0.52,
        -0.02,
        "Used for: 3D open-field-line DRB + sheath budgets",
        ha="center",
        va="top",
        transform=ax2.transAxes,
    )

    fig.suptitle("Geometry conventions in jaxdrb (linear + nonlinear)", y=0.99)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.16)
    fig.savefig(out)
    plt.close(fig)
    print(f"[make_geometry_overview] wrote {out}")


if __name__ == "__main__":
    main()
