"""HSX FCI Braginskii filament: run the backend, then render the geometry (coils, vessel, Poincare, connection length) and a 3-D movie.

Runs ``drbx.fci_braginskii.run.run`` (the same driver as ``simulate_hsx_blob.py``
and ``drbx run examples/inputs/hsx_fci_blob.toml``) on the canonical 32^3 HSX
geometry bundle, then renders the saved history.

Run from the repository root:

    PYTHONPATH=src python examples/stellarator/hsx_fci_blob_render.py

The driver prints the geometry, model parameters, one progress line per step
(step, time, timings, field extrema) and where the history is written. Outputs
go to ``output/hsx_fci_blob/``: ``history.npz``, ``hsx_geometry.png`` (coils, vessel,
Poincare section and connection-length map, traced through the coil field) and
``hsx_fci_blob.gif`` (the filament in 3D plus an R-Z section). The geometry figure
needs the vessel file (``VESSEL``), the coils (``COILS``) and a few minutes of CPU.

The default is a quick 5-step preset (t = 0.00375), not the converged case. The
figure in ``docs/media`` uses the documented default run: FINAL_TIME = 0.6,
NUM_STEPS = 800, SAVE_EVERY = 40 (see docs/fci_braginskii_hsx_backend.md).
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

from drbx.fci_braginskii.geometry_build.provider_bfield import CoilFieldTable  # noqa: E402
from drbx.fci_braginskii.run import HsxBlobConfig, run  # noqa: E402
from drbx.geometry.field_provider import FilamentCoilField  # noqa: E402

# ---- PARAMETERS ----------------------------------------------------------
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")  # canonical geometry bundle
FINAL_TIME = 0.00375        # quick preset; documented run uses 0.6
NUM_STEPS = 5               # quick preset; documented run uses 800 (dt = 7.5e-4)
SAVE_EVERY = 1              # history frames (movie frames); documented run uses 40
COILS = Path("examples/inputs/hsx/coils.hsx")            # MAKEGRID filaments (QHS: six main groups, 10722 A)
VESSEL = Path("examples/inputs/hsx/vessel_hsx_flare.txt")  # vessel outline: 161 planes over one period
TRACE_STEP = 0.02           # field-line step [m]
N_POINCARE, POINCARE_STEPS = 40, 50000   # seeds on the midplane (R = 1.38..1.54 m), steps per seed (~130 toroidal turns)
CL_SHAPE, CL_MAX_STEPS = (44, 64), 15000  # connection-length grid (R, Z) at phi = 0 and step cap (300 m)
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
                      save_every=SAVE_EVERY, diagnostic_every=1, output=HISTORY_PATH))
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
ax3 = fig.add_axes([-0.04, 0.08, 0.5, 0.85], projection="3d")
vp = np.concatenate([vphi + k * period for k in range(3)])     # vessel: three of the four periods (cutaway)
vR, vZ = np.concatenate([vrz[..., 0]] * 3), np.concatenate([vrz[..., 1]] * 3)
ax3.plot_surface(vR * np.cos(vp)[:, None], vR * np.sin(vp)[:, None], vZ, color="0.5", alpha=0.18, shade=False,
                 rstride=4, cstride=4, linewidth=0, antialiased=False)
ax3.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], facecolors=cm.viridis((bmag - bmag.min()) / np.ptp(bmag)),
                 shade=False, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.9)
for c in np.asarray(coils.nodes):
    ax3.plot(*np.vstack([c, c[:1]]).T, color="#8c564b", lw=0.5, alpha=0.8)
ax3.set_box_aspect((1, 1, 0.42), zoom=1.55); ax3.view_init(elev=48, azim=-50); ax3.set_axis_off()
ax3.set_title("coils, boundary, vessel", fontsize=10, y=0.97)
fig.colorbar(cm.ScalarMappable(colors.Normalize(bmag.min() / bmag.max(), 1.0), "viridis"),
             cax=fig.add_axes([0.08, 0.13, 0.26, 0.03]), orientation="horizontal", label=r"boundary $|B| / |B|_{max}$")
ax = fig.add_axes([0.52, 0.18, 0.19, 0.68])
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
hit = ~confined[pseed]
ax.scatter(pr[~hit], pz[~hit], s=1.5, c="#1f77b4", linewidths=0, label="stay in the vessel")
ax.scatter(pr[hit], pz[hit], s=9, c="#d62728", marker="D", linewidths=0, label="reach the wall")
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_title(r"Poincare, $\phi$ = 0", fontsize=10)
ax = fig.add_axes([0.75, 0.18, 0.15, 0.68])
mesh = ax.pcolormesh(RR, ZZ, conn, cmap="magma_r", norm=colors.LogNorm(1.0, 300.0), shading="nearest", rasterized=True)
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_yticklabels([])
ax.set_title("connection length", fontsize=10)
fig.colorbar(mesh, cax=fig.add_axes([0.915, 0.18, 0.012, 0.68])).set_label("[m]")
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
Image.open(buf).convert("RGB").quantize(160).save(out / "hsx_geometry.png", optimize=True)
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


def filament_frame(it):
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = fig.add_axes([-0.02, 0.0, 0.52, 0.95], projection="3d")
    ax.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], color="0.75", alpha=0.12, shade=False, rstride=2, cstride=2, linewidth=0)
    f3 = SCALE * dn[it]
    # field-aligned columns (fixed radial and poloidal index, all toroidal planes): draw the segments where |n - 1| is large
    p = np.concatenate([xyz, xyz[:, :, :1]], axis=2)                 # close the toroidal circuit
    f = np.concatenate([f3, f3[:, :, :1]], axis=2)
    seg = np.stack([p[:, :, :-1], p[:, :, 1:]], axis=-2).reshape(-1, 2, 3)
    val = (0.5 * (f[:, :, :-1] + f[:, :, 1:])).ravel()
    keep = np.abs(val) > 0.12 * nlim
    lc = Line3DCollection(seg[keep], cmap="RdBu_r", norm=norm, linewidths=1.0 + 3.0 * np.minimum(np.abs(val[keep]) / nlim, 1.0))
    lc.set_array(val[keep]); ax.add_collection3d(lc)
    ax.set_box_aspect((1, 1, 0.42), zoom=1.5); ax.view_init(elev=48, azim=-50); ax.set_axis_off()
    ax.set_axis_off()
    wrap = lambda a: np.concatenate([a, a[:, :1]], axis=1)
    for row, (name, fld, cmap, nrm) in enumerate(((r"$n - 1$", dn[it], "RdBu_r", norm), (r"$\phi$", phi[it], "PuOr_r", pnorm))):
        ax2 = fig.add_axes([0.58, 0.12 + 0.43 * (1 - row), 0.26, 0.33])
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
