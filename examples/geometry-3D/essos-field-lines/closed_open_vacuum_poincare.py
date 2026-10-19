"""Landreman-Paul coils in vacuum: closed and open field lines side by side.

README figure: the defaults below (30 closed + 30 open lines, MAXTIME = 4000, 16000 samples each).

The Landreman-Paul quasi-axisymmetric configuration is produced by real coils;
their vacuum Biot-Savart field has nested closed flux surfaces in the core and
open, wandering field lines outside the confinement region. This example
traces both, directly through the ESSOS coil field:

1. load the coil set and locate the magnetic axis;
2. seed field lines inside the confinement region (closed) and outside it;
3. trace every line with ESSOS's adaptive integrator, printing progress;
4. classify each line as closed or open from its trajectory -- a line is open
   when it escapes the annulus around the axis (or leaves the integration
   domain entirely); and
5. draw the Poincare section at phi = 0: nested closed surfaces in blue, the
   escaping open lines in red.

Requires the installed ``essos`` package (``pip install 'drbx[essos]'``):

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/closed_open_vacuum_poincare.py

prints per-line classifications and writes
``output/closed_open_vacuum_poincare/closed_open_vacuum_poincare.png`` (relative to the
current working directory). If ESSOS is not importable the script says how to
install it and exits.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from drbx.geometry import (
    essos_runtime_available,
    load_essos_coil_field_axis,
    trace_essos_coil_initial_conditions,
)

# --- PARAMETERS -----------------------------------------------------------------
N_CLOSED = 30           # seeds inside the confinement region
N_OPEN = 30             # seeds outside it
MAXTIME = 4000.0        # integration time per line (ESSOS units)
TIMES_TO_TRACE = 16000   # trajectory samples per line
SHOW_POINTS = 900       # samples per line drawn in the 3D view
RHO_WALL = 0.45         # escape radius around the axis [m]: beyond this = open
OUTPUT_DIR = Path("output/closed_open_vacuum_poincare")   # artifact directory (cwd-relative)


def seed_points(axis_r: float, axis_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Seed lines on the outboard midplane, inside and outside the core."""

    closed_r = axis_r + np.linspace(0.06, 0.16, N_CLOSED)
    open_r = axis_r + np.linspace(0.21, 0.32, N_OPEN)
    make = lambda radii: np.stack([radii, np.zeros_like(radii), np.full_like(radii, axis_z)], axis=1)
    return make(closed_r), make(open_r)


def classify_line(trajectory_xyz: np.ndarray, axis_r: float, axis_z: float) -> tuple[bool, float]:
    """Return ``(is_open, max_rho)`` for one traced line.

    ``rho`` is the distance from the magnetic axis in the (R, Z) plane. A line
    is open when it leaves the integration domain (non-finite samples after an
    escape) or wanders beyond ``RHO_WALL``.
    """

    finite = np.all(np.isfinite(trajectory_xyz), axis=1)
    points = trajectory_xyz[finite]
    major_radius = np.hypot(points[:, 0], points[:, 1])
    rho = np.hypot(major_radius - axis_r, points[:, 2] - axis_z)
    max_rho = float(rho.max()) if len(rho) else np.inf
    return (not finite.all()) or (max_rho > RHO_WALL), max_rho


def poincare_points(trajectory_xyz: np.ndarray) -> np.ndarray:
    """(R, Z) crossings of the phi = 0 half-plane (y sign change with x > 0)."""

    x, y, z = trajectory_xyz[:, 0], trajectory_xyz[:, 1], trajectory_xyz[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    # Any y sign change with x > 0 is a crossing of the phi = 0 half-plane
    # (the toroidal circulation direction depends on the coil currents).
    crossing = (y[:-1] * y[1:] < 0) & (x[1:] > 0)
    weight = -y[:-1][crossing] / (y[1:][crossing] - y[:-1][crossing])
    r_cross = np.hypot(x[:-1][crossing] + weight * np.diff(x)[crossing],
                       y[:-1][crossing] + weight * np.diff(y)[crossing])
    z_cross = z[:-1][crossing] + weight * np.diff(z)[crossing]
    return np.stack([r_cross, z_cross], axis=1)


# --- trace, classify, and plot ----------------------------------------------------
if not essos_runtime_available():
    raise SystemExit(
        "ESSOS is not importable. Install it: pip install 'drbx[essos]'"
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("loading the Landreman-Paul coil set and locating the magnetic axis...")
axis_r, axis_z = load_essos_coil_field_axis()
print(f"  magnetic axis: R = {axis_r:.4f} m, Z = {axis_z:.4f} m")

closed_seeds, open_seeds = seed_points(axis_r, axis_z)
print(f"seeding {N_CLOSED} lines inside the core (R - R_axis = "
      f"{closed_seeds[0, 0] - axis_r:.2f}..{closed_seeds[-1, 0] - axis_r:.2f} m) and "
      f"{N_OPEN} outside ({open_seeds[0, 0] - axis_r:.2f}..{open_seeds[-1, 0] - axis_r:.2f} m)")

print(f"tracing {N_CLOSED + N_OPEN} field lines (maxtime={MAXTIME:g}, "
      f"{TIMES_TO_TRACE} samples each) through the Biot-Savart coil field...")
start = time.perf_counter()
trajectories = trace_essos_coil_initial_conditions(
    np.vstack([closed_seeds, open_seeds]), maxtime=MAXTIME, times_to_trace=TIMES_TO_TRACE
)
print(f"  traced array: {trajectories.shape} in {time.perf_counter() - start:.1f} s")

print(f"classifying each line (open = escapes rho > {RHO_WALL} m from the axis):")
sections, labels = [], []
for index, trajectory in enumerate(trajectories):
    is_open, max_rho = classify_line(trajectory, axis_r, axis_z)
    crossings = poincare_points(trajectory)
    sections.append(crossings)
    labels.append(is_open)
    kind = "OPEN  " if is_open else "closed"
    print(f"  line {index:2d}: {kind}  max rho = {max_rho:6.3f} m, "
          f"{len(crossings):4d} Poincare crossings")

n_open = sum(labels)
print(f"result: {len(labels) - n_open} closed lines, {n_open} open lines")
assert not any(labels[:N_CLOSED]), "a core-seeded line escaped: move RHO_WALL or seeds"
assert any(labels[N_CLOSED:]), "no edge-seeded line escaped: extend MAXTIME"

# 3D view (PyVista, offscreen): coils, a few closed lines (blue) and open lines (red)
import jax.numpy as jnp  # noqa: E402

from drbx.geometry import EssosCoilField, resolve_essos_landreman_qa_json  # noqa: E402
from drbx.viz import add_tubes, image, scene  # noqa: E402

coils = np.asarray(EssosCoilField.from_json(resolve_essos_landreman_qa_json()).coils.gamma)
plotter = scene((1000, 720))
add_tubes(plotter, coils, radius=0.012, color="#c9a27a")
for index in range(0, N_CLOSED, 3):
    line = trajectories[index][np.isfinite(trajectories[index]).all(axis=1)][:SHOW_POINTS]
    add_tubes(plotter, [line], radius=0.004, color="#1f77b4", closed=False)
for index in range(N_CLOSED, N_CLOSED + N_OPEN, 4):
    line = trajectories[index][np.isfinite(trajectories[index]).all(axis=1)]
    line = line[np.hypot(np.hypot(line[:, 0], line[:, 1]) - axis_r, line[:, 2] - axis_z) < 0.9][:SHOW_POINTS // 2]
    if labels[index] and len(line) > 10:
        add_tubes(plotter, [line], radius=0.004, color="#d62728", closed=False)
plotter.camera_position = [(3.4, -3.4, 2.6), (0.0, 0.0, 0.0), (0, 0, 1)]
picture = image(plotter)

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(9.0, 4.2))
ax3 = fig.add_axes([0.0, 0.0, 0.62, 0.9])
ax3.imshow(picture); ax3.set_axis_off()
ax3.text(0.02, 0.02, "coils (tan), closed (blue) and open (red) field lines", transform=ax3.transAxes, fontsize=9, color="0.3")
ax = fig.add_axes([0.70, 0.13, 0.28, 0.72])
for crossings, is_open in sorted(zip(sections, labels), key=lambda t: t[1]):
    if len(crossings):
        ax.scatter(crossings[:, 0], crossings[:, 1], s=2.5 if is_open else 1.0, color="#d62728" if is_open else "#1f77b4", linewidths=0)
ax.scatter([axis_r], [axis_z], marker="+", s=80, color="k")
ax.set_xlabel("R [m]"), ax.set_ylabel("Z [m]")
ax.set_aspect("equal")
ax.set_title(f"Poincare section, phi = 0 ({N_CLOSED + N_OPEN} lines)", fontsize=10)
fig.suptitle("Landreman-Paul coils in vacuum: closed surfaces and open edge lines", fontsize=11, y=0.99)
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=100)
plt.close(fig)
Image.open(buffer).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(OUTPUT_DIR / "closed_open_vacuum_poincare.png", optimize=True)
print(f"wrote {OUTPUT_DIR / 'closed_open_vacuum_poincare.png'}")
