"""HSX FCI planes: tilted, vessel-fitted planes versus the fixed toroidal planes of a classic FCI grid.

Classic flux-coordinate-independent (FCI) codes put poloidal planes at constant toroidal angle phi and
trace field lines from plane to plane. In the HSX geometry bundle the 32 logical planes (the eta index)
are surfaces fitted to the vessel whose cell centres do not share one toroidal angle: each plane is
tilted and curved. The scene shows a short segment of the boundary (translucent), three consecutive
bundle planes as coloured surfaces, the phi = const planes of a classic grid as flat outlines at the
same toroidal positions, and field lines traced through the coil field. Needs the 32^3 geometry bundle
(docs/fci_braginskii_hsx_backend.md) and ``drbx[viz]``.

    PYTHONPATH=src python examples/stellarator/hsx_fci_planes.py
"""

import io
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from drbx.geometry.field_provider import FilamentCoilField  # noqa: E402
from drbx.viz import add_surface, add_tubes, image, scene  # noqa: E402

# ---- PARAMETERS ----------------------------------------------------------
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")   # canonical geometry bundle
COILS = Path("examples/inputs/hsx/coils.hsx")            # coil filaments, QHS currents
PLANES = (8, 9, 10)                                      # consecutive plane indices to draw
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
N_LINES, LINE_LENGTH, LINE_STEP = 7, 1.4, 0.01           # field lines: count, arc length [m], RK4 step [m]
OUTPUT = Path("output/hsx_fci_planes/hsx_fci_planes.png")
# ---------------------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cells = np.load(GEOMETRY / "cell_positions.npz")["cell_positions"]            # (u, theta, eta, 3), metres
angle = np.degrees(np.unwrap(np.arctan2(cells[..., 1], cells[..., 0]), axis=2))
spread = angle.max(axis=(0, 1)) - angle.min(axis=(0, 1))
print(f"{cells.shape[2]} planes, {360 / cells.shape[2]:.2f} deg apart; toroidal angle spread within a plane: "
      f"{spread.mean():.1f} deg on average, up to {spread.max():.1f} deg", flush=True)

coils = FilamentCoilField.from_makegrid(COILS, {i + 1: 10722.0 for i in range(6)})


def direction(p):
    b = coils.B(p)
    return b / jnp.linalg.norm(b, axis=-1, keepdims=True)


print("tracing field lines through the coil field ...", flush=True)
seeds = cells[-3, ::cells.shape[1] // N_LINES, PLANES[0]][:N_LINES]            # near the boundary, on the first plane
p, line = jnp.asarray(seeds), [np.asarray(seeds)]
sign = 1.0 if np.mean(angle[-3, :, PLANES[1]] - angle[-3, :, PLANES[0]]) > 0 else -1.0
for _ in range(int(LINE_LENGTH / LINE_STEP)):
    h = sign * LINE_STEP
    k1 = direction(p); k2 = direction(p + 0.5 * h * k1); k3 = direction(p + 0.5 * h * k2); k4 = direction(p + h * k3)
    p = p + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    line.append(np.asarray(p))
lines = np.stack(line, axis=1)                                                  # (line, step, 3)

plotter = scene((1500, 900))
k0, k1 = PLANES[0] - 2, PLANES[-1] + 3
edge = np.concatenate([cells[-1, :, k0:k1], cells[-1, :1, k0:k1]], axis=0)       # (theta+1, eta, 3) boundary segment
add_surface(plotter, edge, color="#9aa5b1", opacity=0.14, close_second_axis=False)
for plane, color in zip(PLANES, COLORS):
    add_surface(plotter, cells[:, :, plane], color=color, opacity=1.0, ambient=0.5)
    ring = np.vstack([cells[-1, :, plane], cells[-1, :1, plane]])               # classic plane: outer ring at the mean angle
    dphi = np.radians(angle[:, :, plane].mean() - angle[-1, :, plane])
    c, s = np.cos(dphi), np.sin(dphi)
    flat = np.stack([c * cells[-1, :, plane, 0] - s * cells[-1, :, plane, 1],
                     s * cells[-1, :, plane, 0] + c * cells[-1, :, plane, 1], cells[-1, :, plane, 2]], axis=-1)
    add_tubes(plotter, [flat], radius=0.004, color="black")
    # a flat translucent disc for the classic plane
    flat_all = np.stack([np.cos(np.radians(angle[:, :, plane].mean() - angle[:, :, plane])) * cells[:, :, plane, 0]
                         - np.sin(np.radians(angle[:, :, plane].mean() - angle[:, :, plane])) * cells[:, :, plane, 1],
                         np.sin(np.radians(angle[:, :, plane].mean() - angle[:, :, plane])) * cells[:, :, plane, 0]
                         + np.cos(np.radians(angle[:, :, plane].mean() - angle[:, :, plane])) * cells[:, :, plane, 1],
                         cells[:, :, plane, 2]], axis=-1)
    add_surface(plotter, flat_all, color="black", opacity=0.10)
add_tubes(plotter, lines, radius=0.004, color="#d62728", closed=False)
centre = cells[:, :, PLANES[1]].reshape(-1, 3).mean(axis=0)
radial = np.array([centre[0], centre[1], 0.0]) / np.hypot(centre[0], centre[1])
tangent = np.array([-radial[1], radial[0], 0.0])
plotter.camera_position = [tuple(centre + 1.3 * radial + 0.9 * tangent + np.array([0.0, 0.0, 0.7])), tuple(centre), (0, 0, 1)]
picture = image(plotter)

plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(9.0, 5.4))
ax = fig.add_axes([0.0, 0.0, 1.0, 0.9])
ax.imshow(picture); ax.set_axis_off()
handles = [Line2D([0], [0], color=c, lw=6, label=f"DRBX plane {k}: tilted, vessel-fitted") for k, c in zip(PLANES, COLORS)]
handles += [Line2D([0], [0], color="black", lw=2, label=r"classic FCI: $\phi$ = const planes (flat, black)"),
            Line2D([0], [0], color="#d62728", lw=2, label="field lines through the planes")]
ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
fig.suptitle(f"Toroidal angle varies by {spread[list(PLANES)].mean():.0f} deg across a DRBX plane; "
             f"planes are {360 / cells.shape[2]:.2f} deg apart", fontsize=11)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
Image.open(buf).convert("RGB").quantize(160, dither=Image.Dither.NONE).save(OUTPUT, optimize=True)
print(f"wrote {OUTPUT}")
