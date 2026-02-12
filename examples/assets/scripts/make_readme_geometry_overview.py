"""Generate a clean geometry overview figure for the README.

This intentionally avoids dense text overlays. The figure is meant to quickly convey:

1) 1D field-line (flux-tube) representation along the parallel coordinate ℓ
2) 2D perpendicular-plane turbulence testbeds (HW2D/DRB2D), with B out of plane
3) 3D FCI representation as a stack of toroidal planes + field-line mapping between planes

Output:
  examples/assets/readme/geometry_overview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "examples/assets/readme/geometry_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.close("all")
    fig = plt.figure(figsize=(12.0, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])

    # --- Panel 1: 1D along a field line ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("1D field-line (flux-tube)", pad=10)
    l = np.linspace(-np.pi, np.pi, 400)
    ax1.plot(l, 0.25 * np.sin(l), lw=3, color="#d62728")
    ax1.scatter([-np.pi, np.pi], [0.0, 0.0], s=60, color="k", zorder=3)
    ax1.text(-np.pi, -0.12, "target", ha="left", va="top", fontsize=10)
    ax1.text(np.pi, -0.12, "target", ha="right", va="top", fontsize=10)
    ax1.annotate(
        "",
        xy=(0.8 * np.pi, 0.0),
        xytext=(0.3 * np.pi, 0.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="#d62728"),
    )
    ax1.text(0.0, 0.33, "evolve fields vs ℓ", ha="center", va="bottom", fontsize=10)
    ax1.set_xlabel("parallel coordinate ℓ")
    ax1.set_yticks([])
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-0.35, 0.45)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # --- Panel 2: 2D perpendicular plane ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("2D perpendicular plane (HW2D / DRB2D)", pad=10)
    ax2.set_aspect("equal", "box")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    # Simple "eddy" texture
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y, indexing="xy")
    tex = np.sin(2 * np.pi * (1.2 * X + 0.7 * Y)) + 0.5 * np.sin(2 * np.pi * (0.4 * X - 1.1 * Y))
    ax2.imshow(tex, origin="lower", cmap="coolwarm", extent=[0, 1, 0, 1], alpha=0.85)
    ax2.annotate("", xy=(0.9, 0.1), xytext=(0.1, 0.1), arrowprops=dict(arrowstyle="->", lw=2))
    ax2.annotate("", xy=(0.1, 0.9), xytext=(0.1, 0.1), arrowprops=dict(arrowstyle="->", lw=2))
    ax2.text(0.92, 0.1, "x (radial-like)", ha="left", va="center", fontsize=10)
    ax2.text(0.1, 0.92, "y (poloidal-like)", ha="left", va="bottom", fontsize=10, rotation=90)
    ax2.text(0.82, 0.83, "B ⊙", fontsize=14, ha="center", va="center")
    ax2.text(0.82, 0.74, "(out of plane)", fontsize=9, ha="center", va="top")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # --- Panel 3: 3D FCI toroidal planes ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("3D FCI: toroidal planes + maps", pad=10)
    ax3.set_aspect("equal", "box")
    ax3.axis("off")
    # Draw a torus centerline (projected)
    t = np.linspace(0, 2 * np.pi, 500)
    cx, cy = 0.5, 0.52
    Xc = cx + 0.36 * np.cos(t)
    Yc = cy + 0.18 * np.sin(t)
    ax3.plot(Xc, Yc, color="0.7", lw=3)
    # Place plane "cards" along the torus
    plane_t = np.linspace(0.2 * np.pi, 1.8 * np.pi, 10)
    for k, tk in enumerate(plane_t):
        px = cx + 0.36 * np.cos(tk)
        py = cy + 0.18 * np.sin(tk)
        w = 0.06
        h = 0.20
        # Slight rotation effect
        angle = -20 * np.cos(tk)
        rect = plt.Rectangle(
            (px - w / 2, py - h / 2),
            w,
            h,
            facecolor=plt.cm.coolwarm(0.2 + 0.6 * (k / (len(plane_t) - 1))),
            edgecolor="0.2",
            lw=0.8,
            alpha=0.85,
        )
        tr = plt.matplotlib.transforms.Affine2D().rotate_deg_around(px, py, angle) + ax3.transData
        rect.set_transform(tr)
        ax3.add_patch(rect)
    # Mapping arrow
    ax3.annotate(
        "field-line map",
        xy=(cx - 0.12, cy + 0.18),
        xytext=(cx + 0.18, cy + 0.30),
        arrowprops=dict(arrowstyle="->", lw=2, color="#d62728"),
        ha="left",
        va="center",
        fontsize=10,
        color="#d62728",
    )
    ax3.text(0.5, 0.08, "evolve fields on planes; build ∂∥ from maps", ha="center", fontsize=10)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)

    fig.suptitle("How jaxdrb represents edge/SOL physics in 1D, 2D, and 3D", fontsize=14)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[make_readme_geometry_overview] wrote {out}")


if __name__ == "__main__":
    main()
