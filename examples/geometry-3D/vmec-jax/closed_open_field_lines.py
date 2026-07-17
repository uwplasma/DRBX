"""Closed and open field lines: VMEC LCFS from vmec_jax over the coil field.

A VMEC wout file only describes the plasma inside the last closed flux
surface (LCFS): outside it the equilibrium field simply does not exist, so an
honest closed-plus-open picture needs the field of the actual coils.  The
Landreman-Paul 2021 precise QA configuration gives us both ingredients: the
ESSOS checkout carries a Biot-Savart coil set optimized for it, and the
matching (reactor-scale) VMEC wout.  This example combines them:

1. load the wout with vmec_jax and print the equilibrium summary;
2. locate the vacuum magnetic axis of the coil field at phi = 0 by tracing a
   probe line and taking the center of its Poincare crossings;
3. rescale the reactor-scale wout LCFS to the coil-set size by matching the
   phi = 0 magnetic-axis major radii (the wout and the coils describe the
   same configuration at different scales);
4. seed field lines on the outboard midplane inside the LCFS and well
   outside it, trace them all through the ESSOS Biot-Savart field, and
   classify each as closed (stays confined) or open (escapes toward the
   wall);
5. draw the Poincare section at phi = 0 with the LCFS overlaid: the closed
   lines fill the VMEC-confined region enclosed by the LCFS, the open lines
   are the scrape-off-layer field lines that leave the device.

A vacuum field keeps closed surfaces slightly outside the design LCFS before
they break up, so the edge seeds span that whole transition: the innermost of
them land on closed vacuum surfaces just beyond the LCFS (drawn in gray), the
outer ones are genuinely open scrape-off-layer lines that escape to the wall
(red, drawn up to the moment they leave). No field is extrapolated from the
wout.

Requires ESSOS and vmec_jax checkouts:

    DRBX_ESSOS_ROOT=~/local/ESSOS_test \
        PYTHONPATH=src python examples/geometry-3D/vmec-jax/closed_open_field_lines.py

writes ``output/vmec_jax_closed_open/closed_open_field_lines.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath

from drbx.geometry import (
    essos_runtime_available,
    load_vmec_jax_wout,
    trace_essos_coil_initial_conditions,
    vmec_jax_boundary_rz,
    vmec_jax_runtime_available,
    vmec_jax_surface_rz,
    vmec_jax_wout_summary,
)

# PARAMETERS ---------------------------------------------------------------
# Landreman-Paul 2021 precise QA (reactor scale) wout from the local
# ESSOS_test checkout; wout files are external inputs, never committed here.
# The ESSOS coil JSON used for tracing is resolved by the ESSOS adapter from
# the same checkout (DRBX_ESSOS_ROOT).
WOUT_PATH = Path.home() / "local" / "ESSOS_test" / "examples" / "input_files" / "wout_LandremanPaul2021_QA_reactorScale_lowres.nc"
AXIS_PROBE_R = 1.21        # midplane seed radius [m] used to locate the vacuum axis
AXIS_PROBE_MAXTIME = 800.0  # integration time for the axis probe line
CLOSED_FRACTIONS = (0.15, 0.35, 0.55, 0.75, 0.90)  # seed positions as fractions of the LCFS outboard minor radius
# Edge seeds span the vacuum transition beyond the LCFS: the inner ones sit on
# residual closed vacuum surfaces, the outer ones are open SOL lines.
EDGE_OFFSETS = (0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23)  # distances beyond the LCFS outboard edge [m]
MAXTIME = 1500.0           # integration time per field line (ESSOS units)
TIMES_TO_TRACE = 6000      # trajectory samples per line
RHO_WALL = 0.8             # escape distance from the vacuum axis circle [m]: beyond this = open
TRAIL_RHO = 0.45           # plotted escape trails stop here (classification still uses RHO_WALL)
OUTPUT_DIR = Path("output/vmec_jax_closed_open")

# setup --------------------------------------------------------------------
if not vmec_jax_runtime_available():
    raise SystemExit(
        "vmec_jax is not importable. Point DRBX_VMEC_JAX_ROOT at a checkout, e.g.\n"
        "    DRBX_VMEC_JAX_ROOT=~/local/vmec_jax DRBX_ESSOS_ROOT=~/local/ESSOS_test "
        "PYTHONPATH=src python examples/geometry-3D/vmec-jax/closed_open_field_lines.py"
    )
if not essos_runtime_available():
    raise SystemExit(
        "ESSOS is not importable. Point DRBX_ESSOS_ROOT at a checkout, e.g.\n"
        "    DRBX_ESSOS_ROOT=~/local/ESSOS_test PYTHONPATH=src python "
        "examples/geometry-3D/vmec-jax/closed_open_field_lines.py"
    )
if not WOUT_PATH.exists():
    raise SystemExit(
        f"VMEC wout file not found: {WOUT_PATH}\n"
        "Edit the WOUT_PATH parameter to point at the Landreman-Paul QA wout NetCDF "
        "from an ESSOS checkout; wout files are not committed here."
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# stage 1: the VMEC side (vmec_jax) ----------------------------------------
print(f"loading VMEC equilibrium via vmec_jax: {WOUT_PATH.name}")
wout = load_vmec_jax_wout(WOUT_PATH)
summary = vmec_jax_wout_summary(wout)
print(f"  nfp = {summary['nfp']}, aspect ratio = {summary['aspect']:.3f}, "
      f"B0 = {summary['b0']:.3f} T, iota = {summary['iota_axis']:.4f} -> "
      f"{summary['iota_edge']:.4f}")
wout_axis_r0 = float(vmec_jax_surface_rz(wout, s=0.0, theta=np.array(0.0), phi=np.array(0.0))[0])
lcfs_r_raw, lcfs_z_raw = vmec_jax_boundary_rz(wout, phi=0.0, n_theta=256)
print(f"  wout phi = 0 magnetic axis R = {wout_axis_r0:.3f} m, LCFS spans "
      f"R = {lcfs_r_raw.min():.3f}..{lcfs_r_raw.max():.3f} m (reactor scale)")

# stage 2: locate the vacuum axis of the coil field at phi = 0 -------------
print(f"probing the ESSOS coil field for its magnetic axis (seed R = {AXIS_PROBE_R} m)...")
probe = np.array([[AXIS_PROBE_R, 0.0, 0.0]])
probe_trajectory = trace_essos_coil_initial_conditions(
    probe, maxtime=AXIS_PROBE_MAXTIME, times_to_trace=3000
)[0]
x, y, z = probe_trajectory[:, 0], probe_trajectory[:, 1], probe_trajectory[:, 2]
crossing = (y[:-1] * y[1:] < 0) & (x[1:] > 0)
weight = -y[:-1][crossing] / (y[1:][crossing] - y[:-1][crossing])
probe_r = np.hypot(x[:-1][crossing] + weight * np.diff(x)[crossing],
                   y[:-1][crossing] + weight * np.diff(y)[crossing])
axis_r = 0.5 * (probe_r.min() + probe_r.max())
axis_z = 0.0  # phi = 0 is the stellarator-symmetry plane of the coil set
print(f"  vacuum magnetic axis at phi = 0: R = {axis_r:.4f} m")

# stage 3: rescale the wout LCFS to the coil-set size ----------------------
# Same configuration, different scale: match the phi = 0 axis major radii.
scale = axis_r / wout_axis_r0
lcfs_r = axis_r + scale * (lcfs_r_raw - wout_axis_r0)
lcfs_z = scale * lcfs_z_raw
lcfs_polygon = PolygonPath(np.stack([lcfs_r, lcfs_z], axis=1))
lcfs_outboard = float(lcfs_r.max())
minor_outboard = lcfs_outboard - axis_r
print(f"  length ratio coil-set/wout = {scale:.4f}; scaled LCFS spans "
      f"R = {lcfs_r.min():.3f}..{lcfs_outboard:.3f} m, Z = +-{lcfs_z.max():.3f} m")

# stage 4: seed, trace and classify field lines ----------------------------
closed_seeds_r = axis_r + minor_outboard * np.asarray(CLOSED_FRACTIONS)
edge_seeds_r = lcfs_outboard + np.asarray(EDGE_OFFSETS)
seeds_r = np.concatenate([closed_seeds_r, edge_seeds_r])
seeds = np.stack([seeds_r, np.zeros_like(seeds_r), np.zeros_like(seeds_r)], axis=1)
n_closed_seeded = len(closed_seeds_r)
print(f"seeding {n_closed_seeded} lines inside the LCFS "
      f"(R = {closed_seeds_r.min():.3f}..{closed_seeds_r.max():.3f} m) and "
      f"{len(edge_seeds_r)} across the edge transition (R = {edge_seeds_r.min():.3f}.."
      f"{edge_seeds_r.max():.3f} m)")
print(f"tracing {len(seeds_r)} field lines through the ESSOS Biot-Savart coil field "
      f"(maxtime = {MAXTIME:g}, {TIMES_TO_TRACE} samples each)...")
trajectories = trace_essos_coil_initial_conditions(
    seeds, maxtime=MAXTIME, times_to_trace=TIMES_TO_TRACE
)
print(f"  traced array: {trajectories.shape}")

print(f"classifying each line (open = escapes {RHO_WALL} m from the axis circle):")
sections, line_is_open, inside_fractions, escape_paths = [], [], [], []
for index, trajectory in enumerate(trajectories):
    finite = np.all(np.isfinite(trajectory), axis=1)
    points = trajectory[finite]
    major_radius = np.hypot(points[:, 0], points[:, 1])
    rho = np.hypot(major_radius - axis_r, points[:, 2] - axis_z)
    max_rho = float(rho.max()) if len(rho) else np.inf
    is_open = (not finite.all()) or (max_rho > RHO_WALL)
    if is_open and len(rho) and (rho > TRAIL_RHO).any():
        # Keep only the trajectory up to the escape onset: an open line leaves
        # within about one toroidal transit here, so its story is the escaping
        # path itself (plotted below as an R-Z trail), not Poincare crossings.
        points = points[: int(np.argmax(rho > TRAIL_RHO)) + 1]
    # Poincare crossings of the phi = 0 half-plane (y sign change, x > 0).
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    crossing = (y[:-1] * y[1:] < 0) & (x[1:] > 0)
    weight = -y[:-1][crossing] / (y[1:][crossing] - y[:-1][crossing])
    r_cross = np.hypot(x[:-1][crossing] + weight * np.diff(x)[crossing],
                       y[:-1][crossing] + weight * np.diff(y)[crossing])
    z_cross = z[:-1][crossing] + weight * np.diff(z)[crossing]
    crossings = np.stack([r_cross, z_cross], axis=1)
    inside_fraction = float(lcfs_polygon.contains_points(crossings).mean()) if len(crossings) else 0.0
    sections.append(crossings)
    line_is_open.append(is_open)
    inside_fractions.append(inside_fraction)
    escape_paths.append(np.stack([np.hypot(points[:, 0], points[:, 1]), points[:, 2]], axis=1) if is_open else None)
    kind = "OPEN  " if is_open else "closed"
    print(f"  line {index:2d} (seed R = {seeds_r[index]:.3f} m): {kind} "
          f"max rho = {max_rho:6.3f} m, {len(crossings):3d} crossings, "
          f"{100.0 * inside_fraction:5.1f}% inside the LCFS")

n_open = sum(line_is_open)
n_edge_closed = sum(not flag for flag in line_is_open[n_closed_seeded:])
print(f"result: {n_closed_seeded} closed inside the LCFS, {n_edge_closed} closed "
      f"vacuum surfaces beyond it, {n_open} open SOL lines")
assert not any(line_is_open[:n_closed_seeded]), "an LCFS-interior seed escaped: check the scaling"
assert n_open >= 2, "no edge seed escaped: extend EDGE_OFFSETS or MAXTIME"
assert all(fraction > 0.98 for fraction in inside_fractions[:n_closed_seeded]), \
    "closed-line crossings leak outside the LCFS: the configurations do not match"
assert all(fraction < 0.02 for fraction in inside_fractions[n_closed_seeded:]), \
    "edge-line crossings entered the LCFS"
print("verified: the VMEC LCFS encloses every interior line; every edge line stays outside it")

# stage 5: plot ------------------------------------------------------------
print("drawing the phi = 0 Poincare section with the VMEC LCFS overlay...")
fig, axis = plt.subplots(figsize=(7.2, 6.2))
for index, (crossings, is_open) in enumerate(zip(sections, line_is_open)):
    if is_open:
        # Open SOL line: draw the R-Z projection of the field line up to its
        # escape -- a red trail leaving the confined region.
        path_rz = escape_paths[index]
        axis.plot(path_rz[:, 0], path_rz[:, 1], color="#d62728", linewidth=0.6, alpha=0.55)
        axis.scatter([path_rz[-1, 0]], [path_rz[-1, 1]], marker="x", s=34, color="#d62728", linewidths=1.4)
        continue
    if len(crossings) == 0:
        continue
    color = "#1f77b4" if index < n_closed_seeded else "#8fa8bf"
    axis.scatter(crossings[:, 0], crossings[:, 1], s=1.5, color=color, linewidths=0)
axis.plot(lcfs_r, lcfs_z, "k--", linewidth=1.6, label="VMEC LCFS (vmec_jax wout)")
axis.scatter([axis_r], [axis_z], marker="+", s=90, color="k", label="vacuum magnetic axis")
axis.scatter([], [], s=10, color="#1f77b4", label="closed lines (VMEC-confined region)")
axis.scatter([], [], s=10, color="#8fa8bf", label="closed vacuum surfaces beyond the LCFS")
axis.plot([], [], color="#d62728", linewidth=1.2, label="open SOL lines (traced to escape, x = exit)")
axis.set_xlabel("R [m]")
axis.set_ylabel("Z [m]")
axis.set_aspect("equal")
# Frame the LCFS with room on the outboard side for the escape trails to
# visibly leave the picture.
axis.set_xlim(float(lcfs_r.min()) - 0.12, float(lcfs_outboard) + 0.45)
z_span = float(lcfs_z.max()) + 0.30
axis.set_ylim(-z_span, z_span)
axis.legend(loc="upper right", fontsize=8)
axis.set_title("Landreman-Paul precise QA: coil-field Poincare section at phi = 0\n"
               "closed field lines inside the VMEC LCFS, open SOL lines outside")
fig.tight_layout()
figure_path = OUTPUT_DIR / "closed_open_field_lines.png"
fig.savefig(figure_path, dpi=170)
plt.close(fig)
print(f"wrote {figure_path}")
