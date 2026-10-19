"""HSX FCI Braginskii filament: run the backend, then render the geometry (coils, vessel, Poincare, connection length) and a 3-D movie.

README figures: N_POINCARE = 90 seeds x POINCARE_STEPS = 120000 (about 320 toroidal turns), CL_SHAPE = (72, 108);
filament run: FINAL_TIME = 0.45, NUM_STEPS = 600, SAVE_EVERY = 20 with DENSITY_AMPLITUDE = 0.8,
BLOB_CENTER = (0.8, 0.5), BLOB_WIDTH = 0.12 and RHO_STAR = 5e-3 (ten times the default rho*, a smaller or
lower-field device) so the potential dipole develops clearly. The 600 steps took about 35 min on one A4000.

Runs ``drbx.fci_braginskii.run.run`` (the driver behind
``drbx run examples/inputs/hsx_fci_blob.toml``) on the canonical 32^3 HSX
geometry bundle, then renders the saved history.

Run from the repository root:

    PYTHONPATH=src python examples/stellarator/hsx_fci_blob_render.py

The driver prints the geometry, model parameters, one progress line per step
(step, time, timings, field extrema) and where the history is written. Outputs
go to ``output/hsx_fci_blob/``: ``history.npz``, ``hsx_geometry.png`` (coils, vessel,
Poincare section and connection-length map, traced through the coil field) and
``hsx_fci_blob.gif`` (the filament in 3D plus an R-Z section). The geometry figure
needs the vessel file (``VESSEL``), the coils (``COILS``) and a few minutes of CPU.

The defaults are the README run (about 35 min on a GPU, hours on a laptop CPU); a quick preset is
FINAL_TIME = 0.00375, NUM_STEPS = 5, SAVE_EVERY = 1.
Set RUN_SIMULATION = False to only re-render an existing HISTORY_PATH.
"""

import time
from pathlib import Path

import io

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from PIL import Image  # noqa: E402

from drbx.fci_braginskii.geometry_build.provider_bfield import CoilFieldTable  # noqa: E402
from drbx.fci_braginskii.run import HsxBlobConfig, run  # noqa: E402
from drbx.geometry.field_provider import FilamentCoilField  # noqa: E402
from drbx.viz import add_surface, add_tubes, image, scene  # noqa: E402

# ---- PARAMETERS ----------------------------------------------------------
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")  # canonical geometry bundle
FINAL_TIME = 0.45           # simulated time [L_ref / c_s]
NUM_STEPS = 600             # time steps (dt = 7.5e-4)
SAVE_EVERY = 20             # steps between history frames (movie frames)
DENSITY_AMPLITUDE = 0.8     # peak n - 1 of the seeded filament (backend default 0.05)
BLOB_CENTER = (0.8, 0.5)    # filament centre in the logical (radial, poloidal) square (default (0.65, 0.5))
BLOB_WIDTH = 0.12           # filament width (default 0.10)
RHO_STAR = 5.0e-3           # rho* = rho_s / L_ref; ExB and curvature drifts scale with it (default 5e-4)
COILS = Path("examples/inputs/hsx/coils.hsx")            # MAKEGRID filaments (QHS: six main groups, 10722 A)
VESSEL = Path("examples/inputs/hsx/vessel_hsx_flare.txt")  # vessel outline: 161 planes over one period
TRACE_STEP = 0.02           # field-line step [m]
N_POINCARE, POINCARE_STEPS = 90, 120000  # seeds on the midplane (R = 1.38..1.54 m), steps per seed (~320 toroidal turns)
CL_SHAPE, CL_MAX_STEPS = (72, 108), 15000  # connection-length grid (R, Z) at phi = 0 and step cap (300 m)
OUTPUT_DIR = Path("output/hsx_fci_blob")
HISTORY_PATH = OUTPUT_DIR / "history.npz"
RUN_SIMULATION = True       # False: re-render an existing HISTORY_PATH only
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if RUN_SIMULATION:
    print(f"HSX FCI blob: geometry={GEOMETRY} final_time={FINAL_TIME} "
          f"num_steps={NUM_STEPS} save_every={SAVE_EVERY} -> {HISTORY_PATH}", flush=True)
    start = time.perf_counter()
    run(HsxBlobConfig(geometry=GEOMETRY, final_time=FINAL_TIME, num_steps=NUM_STEPS,
                      save_every=SAVE_EVERY, diagnostic_every=10, output=HISTORY_PATH,
                      density_amplitude=DENSITY_AMPLITUDE, blob_center=BLOB_CENTER, blob_width=BLOB_WIDTH, rho_star=RHO_STAR))
    print(f"simulation finished in {time.perf_counter() - start:.1f} s", flush=True)


start = time.perf_counter()
jax.config.update("jax_enable_x64", True)
data = np.load(HISTORY_PATH)
out = OUTPUT_DIR
xyz, times, active = data["cartesian"], data["times"], data["owner_active"]
dn = data["density"].astype(np.float64) - 1.0
R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
print(f"{len(times)} history frames (t = {times[0]:.4g} .. {times[-1]:.4g})", flush=True)

# ---- vessel (161 planes, one field period) and the coil field -------------------------------
tok = np.array(VESSEL.read_text().split()[4:], float)   # skip "#name", "161 201 4"
n_plane, n_pt = 161, 201
blocks = tok.reshape(n_plane, 1 + 2 * n_pt)
vphi, vrz = np.radians(blocks[:, 0]), blocks[:, 1:].reshape(n_plane, n_pt, 2)
coils = FilamentCoilField.from_makegrid(COILS, {i + 1: 10722.0 for i in range(6)})
table = CoilFieldTable.from_provider(coils, R_bounds=(vrz[..., 0].min() - 0.05, vrz[..., 0].max() + 0.05),
                                     Z_bounds=(vrz[..., 1].min() - 0.05, vrz[..., 1].max() + 0.05), shape=(40, 40, 40))
period = 2.0 * np.pi / 4
# star-shaped wall about each plane's centroid: radius versus polar angle (inside test for the tracer)
centre = vrz.mean(axis=1)
ang_bins = np.linspace(-np.pi, np.pi, 361)
wall_r = np.zeros((n_plane, 360))
for i in range(n_plane):
    d = vrz[i] - centre[i]
    ang, rad = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(ang)
    wall_r[i] = np.interp(0.5 * (ang_bins[1:] + ang_bins[:-1]), ang[order], rad[order], period=2 * np.pi)
wall_r, centre_j = jnp.asarray(wall_r), jnp.asarray(centre)


def inside(p):
    ph = jnp.mod(jnp.arctan2(p[:, 1], p[:, 0]), period)
    plane = jnp.clip(jnp.round(ph / period * (n_plane - 1)).astype(int), 0, n_plane - 1)
    dr, dz = jnp.hypot(p[:, 0], p[:, 1]) - centre_j[plane, 0], p[:, 2] - centre_j[plane, 1]
    ibin = jnp.clip(((jnp.arctan2(dz, dr) + jnp.pi) / (2 * jnp.pi) * 360).astype(int), 0, 359)
    return jnp.hypot(dr, dz) < wall_r[plane, ibin]


def direction(p):
    b = table.B(p)
    return b / jnp.linalg.norm(b, axis=-1, keepdims=True)


@jax.jit
def _step(carry, _, sign):
    p, alive, length = carry
    h = sign * TRACE_STEP
    k1 = direction(p); k2 = direction(p + 0.5 * h * k1); k3 = direction(p + 0.5 * h * k2); k4 = direction(p + h * k3)
    q = p + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    alive = alive & inside(q)
    return (jnp.where(alive[:, None], q, p), alive, length + alive * TRACE_STEP), p


def trace(seeds, sign, steps, record=False):
    carry = (jnp.asarray(seeds), inside(jnp.asarray(seeds)), jnp.zeros(len(seeds)))
    (p, alive, length), path = jax.lax.scan(lambda c, x: _step(c, x, sign), carry, None, length=steps)
    return np.asarray(length), (np.asarray(path) if record else None)


print("tracing the Poincare section ...", flush=True)
r_seed = np.linspace(1.38, 1.54, N_POINCARE)
seeds = np.stack([r_seed, np.zeros(N_POINCARE), np.zeros(N_POINCARE)], axis=1)
seed_len, path = trace(seeds, 1.0, POINCARE_STEPS, record=True)       # (steps, seeds, 3)
x, y, z = path[..., 0], path[..., 1], path[..., 2]
cross = (y[:-1] * y[1:] < 0) & (x[1:] > 0)
t_c = -y[:-1] / (y[1:] - y[:-1] + 1e-30)
pr = np.hypot(x[:-1] + t_c * (x[1:] - x[:-1]), 0.0)[cross]
pz = (z[:-1] + t_c * (z[1:] - z[:-1]))[cross]
pseed = np.nonzero(cross)[1]
confined = seed_len > 0.98 * POINCARE_STEPS * TRACE_STEP

print("tracing the connection length ...", flush=True)
Rg = np.linspace(vrz[0, :, 0].min(), vrz[0, :, 0].max(), CL_SHAPE[0])
Zg = np.linspace(vrz[0, :, 1].min(), vrz[0, :, 1].max(), CL_SHAPE[1])
RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
grid_pts = np.stack([RR.ravel(), np.zeros(RR.size), ZZ.ravel()], axis=1)
l_fwd, _ = trace(grid_pts, 1.0, CL_MAX_STEPS)
l_bwd, _ = trace(grid_pts, -1.0, CL_MAX_STEPS)
conn = (l_fwd + l_bwd).reshape(RR.shape)
inside0 = np.asarray(inside(jnp.asarray(grid_pts))).reshape(RR.shape)
conn = np.where(inside0, np.maximum(conn, TRACE_STEP), np.nan)
print(f"geometry traced in {time.perf_counter() - start:.1f} s", flush=True)

# ---- figure 1: coils + vessel + boundary in 3D, Poincare section, connection length ------------
plt.rcParams.update({"font.size": 10})
edge, bmag = xyz[-1], data["Bmag"][-1]
fig = plt.figure(figsize=(9.0, 3.5))

def closed_surface(a):
    """Close an ``(theta, eta, ...)`` surface in both periodic directions."""

    a = np.concatenate([a, a[:1]], axis=0)
    return np.concatenate([a, a[:, :1]], axis=1)


plotter = scene((1100, 900))
vp = np.concatenate([vphi + k * period for k in range(4)])      # the vessel over the full torus (4 periods)
vR, vZ = np.concatenate([vrz[..., 0]] * 4), np.concatenate([vrz[..., 1]] * 4)
vessel = np.stack([vR * np.cos(vp)[:, None], vR * np.sin(vp)[:, None], vZ], axis=-1)
vessel = vessel[np.mod(np.degrees(vp), 360.0) <= 270.0]          # cutaway: drop the last quarter turn
add_surface(plotter, vessel, color="#bfbfbf", opacity=0.18, close_second_axis=True)
add_surface(plotter, closed_surface(edge)[:, :, :], scalars=closed_surface(bmag / bmag.max()), cmap="viridis",
            clim=(0.64, 1.0), show_scalar_bar=False, close_second_axis=False)
add_tubes(plotter, np.asarray(coils.nodes)[::2], radius=0.004, color="#8c564b")
plotter.camera_position = [(0.0, -2.6, 6.6), (0.0, 0.0, 0.0), (0, 0, 1)]
ax3 = fig.add_axes([-0.02, 0.14, 0.5, 0.8])
ax3.imshow(image(plotter)); ax3.set_axis_off()
ax3.set_title("coils, boundary (|B|), vessel cutaway", fontsize=10)
fig.colorbar(cm.ScalarMappable(colors.Normalize(bmag.min() / bmag.max(), 1.0), "viridis"),
             cax=fig.add_axes([0.08, 0.12, 0.26, 0.03]), orientation="horizontal", label=r"boundary $|B| / |B|_{max}$")
ax = fig.add_axes([0.52, 0.18, 0.19, 0.68])
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
hit = ~confined[pseed]
ax.scatter(pr[~hit], pz[~hit], s=0.35, c="#1f4e9c", linewidths=0, label="stay in the vessel")
ax.scatter(pr[hit], pz[hit], s=5, c="#d62728", marker="D", linewidths=0, label="reach the wall")
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_title(r"Poincare, $\phi$ = 0", fontsize=10)
ax = fig.add_axes([0.75, 0.18, 0.15, 0.68])
mesh = ax.pcolormesh(RR, ZZ, conn, cmap="magma_r", norm=colors.LogNorm(1.0, 300.0), shading="nearest", rasterized=True)
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_yticklabels([])
ax.set_title("connection length", fontsize=10)
fig.colorbar(mesh, cax=fig.add_axes([0.915, 0.18, 0.012, 0.68])).set_label("[m]")
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
Image.open(buf).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(out / "hsx_geometry.png", optimize=True)
print(f"saved {out / 'hsx_geometry.png'}", flush=True)

# ---- figure 2: the filament along the field (3D) and R-Z sections of n - 1 and phi ------------------
# Frames with a blank potential (t = 0) are skipped. Colour limits are symmetric, set per field from the run.
phi = data["phi"].astype(np.float64)
first = int(np.argmax(np.abs(phi).reshape(len(times), -1).max(axis=1) > 0))
k0 = int(np.argmax(np.abs(dn[first]).max(axis=(0, 1))))            # toroidal plane through the seeded filament
SCALE = 100.0                                                      # colour values in units of 1e-2
nlim = SCALE * float(np.percentile(np.abs(dn[first:]), 99.9))
plim = SCALE * float(np.percentile(np.abs(phi[first:]), 99.9))
norm, pnorm = colors.Normalize(-nlim, nlim), colors.Normalize(-plim, plim)


import pyvista as pv  # noqa: E402



def close_volume(a):
    """Close an ``(u, theta, eta, ...)`` array periodically in theta and eta."""

    a = np.concatenate([a, a[:, :1]], axis=1)
    return np.concatenate([a, a[:, :, :1]], axis=2)


cells = close_volume(xyz)
cell_grid = pv.StructuredGrid(cells[..., 0], cells[..., 1], cells[..., 2])
LEVELS = (0.2, 0.45, 0.7)                                          # isosurface levels, fractions of the colour limit


def filament_frame(it):
    cell_grid["dn"] = close_volume(SCALE * dn[it]).ravel(order="F")
    plotter = scene((900, 760))
    add_surface(plotter, closed_surface(edge), color="#cccccc", opacity=0.10, close_second_axis=False)
    iso = cell_grid.contour([lv * nlim for lv in LEVELS], scalars="dn")
    if iso.n_points:
        plotter.add_mesh(iso, scalars="dn", cmap="RdBu_r", clim=(-nlim, nlim), smooth_shading=True, show_scalar_bar=False, opacity=0.85)
    plotter.camera_position = [(0.0, -2.6, 6.6), (0.0, 0.0, 0.0), (0, 0, 1)]
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = fig.add_axes([0.0, 0.0, 0.54, 0.92])
    ax.imshow(image(plotter)); ax.set_axis_off()
    ax.text(0.02, 0.04, r"isosurfaces of $n - 1$ over the boundary", transform=ax.transAxes, fontsize=9, color="0.3")
    wrap = lambda a: np.concatenate([a, a[:, :1]], axis=1)
    for row, (name, fld, cmap, nrm) in enumerate(((r"$n - 1$", dn[it], "RdBu_r", norm), (r"$\phi$", phi[it], "PuOr_r", pnorm))):
        ax2 = fig.add_axes([0.60, 0.11 + 0.43 * (1 - row), 0.28, 0.34])
        mesh = ax2.pcolormesh(wrap(R[:, :, k0]), wrap(Z[:, :, k0]), wrap(SCALE * np.where(active[:, :, k0], fld[:, :, k0], np.nan)),
                              cmap=cmap, norm=nrm, shading="gouraud", rasterized=True)
        ax2.plot(np.append(R[-1, :, k0], R[-1, 0, k0]), np.append(Z[-1, :, k0], Z[-1, 0, k0]), "k", lw=0.8)
        ax2.set_aspect("equal"); ax2.set_ylabel("Z [m]")
        ax2.set_title(name + r" at toroidal plane %d ($\times 10^{-2}$)" % k0, fontsize=9)
        if row:
            ax2.set_xlabel("R [m]")
        else:
            ax2.tick_params(labelbottom=False)
        fig.colorbar(mesh, ax=ax2, fraction=0.05, pad=0.03)
    fig.suptitle(f"HSX scrape-off-layer filament, t = {times[it]:.3f} $L_{{ref}}/c_s$", fontsize=11)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); plt.close(fig)
    return Image.open(buf).convert("RGB").quantize(96, dither=Image.Dither.NONE)


frames = [filament_frame(it) for it in range(first, len(times))]
frames[0].save(out / "hsx_fci_blob.gif", save_all=True, append_images=frames[1:], duration=250, loop=0, optimize=True)
print(f"saved {out / 'hsx_fci_blob.gif'}; max |n-1| = {np.abs(dn[-1]).max():.3e} at t = {times[-1]:.4g}; render {time.perf_counter() - start:.1f} s")
