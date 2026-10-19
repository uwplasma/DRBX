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
figure in ``docs/media`` uses the documented default run: FINAL_TIME = 0.15,
NUM_STEPS = 200, SAVE_EVERY = 10 (see docs/fci_braginskii_hsx_backend.md).
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

# ---- PARAMETERS ----------------------------------------------------------
GEOMETRY = Path("artifacts/geometry/hsx_fci_32x32x32")  # canonical geometry bundle
FINAL_TIME = 0.00375        # quick preset; documented run uses 0.15
NUM_STEPS = 5               # quick preset; documented run uses 200 (dt = 7.5e-4)
SAVE_EVERY = 1              # history frames (movie frames); documented run uses 10
COILS = Path("examples/inputs/hsx/coils.hsx")            # MAKEGRID filaments (QHS: six main groups, 10722 A)
VESSEL = Path("artifacts/inputs/hsx/vessel_hsx_flare.txt")  # vessel outline: 161 planes over one period
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
edge, bmag = xyz[-1], data["Bmag"][-1]
fig = plt.figure(figsize=(11.0, 4.4))
ax3 = fig.add_axes([0.0, 0.0, 0.46, 0.95], projection="3d")
# vessel: three of the four periods (cutaway)
vp = np.concatenate([vphi + k * period for k in range(3)])
vR, vZ = np.concatenate([vrz[..., 0]] * 3), np.concatenate([vrz[..., 1]] * 3)
ax3.plot_surface(vR * np.cos(vp)[:, None], vR * np.sin(vp)[:, None], vZ, color="0.5", alpha=0.18, shade=False,
                 rstride=4, cstride=4, linewidth=0, antialiased=False)
bn = (bmag - bmag.min()) / np.ptp(bmag)
ax3.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], facecolors=cm.viridis(bn), shade=False,
                 rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.9)
nodes = np.asarray(coils.nodes)
for c in nodes:
    ax3.plot(*np.vstack([c, c[:1]]).T, color="#8c564b", lw=0.5, alpha=0.8)
ax3.set_box_aspect((1, 1, 0.3), zoom=1.45); ax3.view_init(elev=38, azim=-50); ax3.set_axis_off()
ax3.text2D(0.1, 0.93, "HSX: coils (brown), boundary coloured by |B|, vessel (grey)", transform=ax3.transAxes, fontsize=9)
fig.colorbar(cm.ScalarMappable(colors.Normalize(bmag.min() / bmag.max(), 1.0), "viridis"),
             cax=fig.add_axes([0.05, 0.12, 0.2, 0.02]), orientation="horizontal", label=r"$|B| / |B|_{max}$ on the boundary")
ax = fig.add_axes([0.50, 0.14, 0.19, 0.7])
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
ax.scatter(pr, pz, s=1.2, c=np.where(confined[pseed], "#1f77b4", "#d62728"), linewidths=0)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_title("Poincare section, phi = 0\n(blue: confined, red: reaches wall)", fontsize=9)
ax = fig.add_axes([0.74, 0.14, 0.16, 0.7])
mesh = ax.pcolormesh(RR, ZZ, conn, cmap="magma_r", norm=colors.LogNorm(0.1, 300.0), shading="nearest", rasterized=True)
ax.plot(vrz[0, :, 0], vrz[0, :, 1], "k", lw=1.0)
ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_yticklabels([])
ax.set_title("connection length to the wall", fontsize=9)
fig.colorbar(mesh, cax=fig.add_axes([0.915, 0.14, 0.01, 0.7]), label="[m] (capped at 300)")
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
Image.open(buf).convert("RGB").quantize(160).save(out / "hsx_geometry.png", optimize=True)
print(f"saved {out / 'hsx_geometry.png'}", flush=True)

# ---- figure 2: the filament in 3D (cells with large |n - 1|) and an R-Z section ------------------
k0 = int(np.argmax(np.abs(dn[0]).max(axis=(0, 1))))                # toroidal plane of the seeded filament
vmax = float(np.abs(dn).max())
norm = colors.Normalize(-vmax, vmax)
phi = data["phi"].astype(np.float64)
pnorm = colors.Normalize(-np.abs(phi).max(), np.abs(phi).max())


def section_frame(it):
    fig = plt.figure(figsize=(8.0, 4.4))
    ax = fig.add_axes([-0.06, -0.05, 0.62, 1.05], projection="3d")
    ax.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], color="0.8", alpha=0.10, shade=False, rstride=2, cstride=2, linewidth=0)
    sel = (np.abs(dn[it]) > 0.06 * vmax) & active
    ax.scatter(xyz[..., 0][sel], xyz[..., 1][sel], xyz[..., 2][sel], c=dn[it][sel], cmap="RdBu_r", norm=norm,
               s=10 + 90 * np.abs(dn[it][sel]) / vmax, depthshade=False, linewidths=0)
    ax.set_box_aspect((1, 1, 0.3), zoom=1.35); ax.view_init(elev=40, azim=-50); ax.set_axis_off()
    wrap = lambda a: np.concatenate([a, a[:, :1]], axis=1)
    for row, (name, fld, cmap, nrm, label) in enumerate((("density perturbation", dn[it], "RdBu_r", norm, r"$n - 1$"),
                                                         ("electrostatic potential", phi[it], "PuOr_r", pnorm, r"$\phi$"))):
        ax2 = fig.add_axes([0.60, 0.12 + 0.43 * (1 - row), 0.22, 0.31])
        mesh = ax2.pcolormesh(wrap(R[:, :, k0]), wrap(Z[:, :, k0]), wrap(np.where(active[:, :, k0], fld[:, :, k0], np.nan)),
                              cmap=cmap, norm=nrm, shading="gouraud", rasterized=True)
        ax2.plot(np.append(R[-1, :, k0], R[-1, 0, k0]), np.append(Z[-1, :, k0], Z[-1, 0, k0]), "k", lw=0.8)
        ax2.set_aspect("equal"); ax2.set_title(name + " (R-Z section)", fontsize=8); ax2.tick_params(labelsize=7)
        ax2.set_ylabel("Z [m]", fontsize=8)
        if row:
            ax2.set_xlabel("R [m]", fontsize=8)
        else:
            ax2.tick_params(labelbottom=False)
        fig.colorbar(mesh, cax=fig.add_axes([0.84, 0.12 + 0.43 * (1 - row) + 0.03, 0.012, 0.25]), label=label)
    fig.suptitle(f"HSX scrape-off-layer filament: t = {times[it]:.3f} (L_ref / c_s)", fontsize=10)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=72); plt.close(fig)
    return Image.open(buf).convert("RGB").quantize(96, dither=Image.Dither.NONE)


frames = [section_frame(it) for it in range(len(times))]
frames[0].save(out / "hsx_fci_blob.gif", save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
print(f"saved {out / 'hsx_fci_blob.gif'}; max |n-1| = {np.abs(dn[-1]).max():.3e} at t = {times[-1]:.4g}; render {time.perf_counter() - start:.1f} s")
