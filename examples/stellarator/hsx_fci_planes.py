"""HSX FCI planes: tilted logical planes versus the fixed toroidal planes of a classic FCI grid.

Classic flux-coordinate-independent (FCI) codes place poloidal planes at constant toroidal
angle and trace field lines from plane to plane. In the HSX geometry bundle the 32 logical
planes (the eta index) are wall-fitted surfaces whose cell centres do not share one toroidal
angle: each plane is tilted and curved. Left: three consecutive planes in 3D. Right: the toroidal
angle of their cell centres against height, with the fixed-angle plane of a classic grid dashed. Needs the 32^3 geometry bundle (see docs/fci_braginskii_hsx_backend.md).

    PYTHONPATH=src python examples/stellarator/hsx_fci_planes.py
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from drbx.viz import add_surface, image, scene  # noqa: E402

# ---- PARAMETERS ----------------------------------------------------------
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")   # canonical geometry bundle
PLANES = (8, 9, 10)                                      # consecutive plane indices to draw
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
OUTPUT = Path("output/hsx_fci_planes/hsx_fci_planes.png")
# ---------------------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cells = np.load(GEOMETRY / "cell_positions.npz")["cell_positions"]            # (u, theta, eta, 3), metres
angle = np.degrees(np.unwrap(np.arctan2(cells[..., 1], cells[..., 0]), axis=2))
spread = angle.max(axis=(0, 1)) - angle.min(axis=(0, 1))
print(f"{cells.shape[2]} planes, {360 / cells.shape[2]:.2f} deg apart; toroidal angle spread within a plane: "
      f"{spread.mean():.1f} deg on average, up to {spread.max():.1f} deg")


plotter = scene((900, 760))
add_surface(plotter, np.concatenate([cells[-1], cells[-1][:1]], axis=0)[:, :, :].transpose(1, 0, 2).copy(), color="#cccccc", opacity=0.10, close_second_axis=False)
for plane, color in zip(PLANES, COLORS):
    add_surface(plotter, cells[:, :, plane], color=color, opacity=1.0, ambient=0.6, show_edges=True, edge_color="white", line_width=0.3)
centre = cells[:, :, PLANES[1]].reshape(-1, 3).mean(axis=0)
plotter.camera_position = [tuple(centre + np.array([0.85, -0.55, 0.65])), tuple(centre), (0, 0, 1)]
picture = image(plotter)

plt.rcParams.update({"font.size": 10})
fig, (ax3, ax) = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw={"width_ratios": [1.0, 1.0]})
ax3.imshow(picture); ax3.set_axis_off(); ax3.set_title(f"planes {PLANES[0]}-{PLANES[-1]} of the HSX bundle", fontsize=10)
for plane, color in zip(PLANES, COLORS):
    a = angle[:, :, plane]
    ax.scatter(np.mod(a, 360.0).ravel(), cells[:, :, plane, 2].ravel(), s=2, color=color, label=f"plane {plane}")
    ax.axvline(np.mod(a.mean(), 360.0), color=color, ls="--", lw=1.0)
ax.set_xlabel("toroidal angle of the cell centres [deg]"); ax.set_ylabel("Z [m]")
ax.set_title("dashed: a classic fixed-angle plane", fontsize=10)
ax.legend(markerscale=4, fontsize=9, loc="upper left")
fig.suptitle(f"Logical planes are tilted: {spread.mean():.0f} deg of toroidal angle across a plane, {360 / cells.shape[2]:.2f} deg between planes", fontsize=10)
buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
Image.open(buf).convert("RGB").quantize(160, dither=Image.Dither.NONE).save(OUTPUT, optimize=True)
print(f"wrote {OUTPUT}")
