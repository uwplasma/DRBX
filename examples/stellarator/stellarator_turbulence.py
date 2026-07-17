"""Stellarator turbulence movies: closed and open field lines.

Runs the multi-mode seeded four-field interchange model (density, vorticity,
ion/electron parallel velocity) on the rotating-ellipse stellarator flux tube
twice -- once with all field lines closed, once with a toroidal limiter opening
the outer flux surfaces into a scrape-off layer drained by a Bohm sheath -- and
renders a compressed GIF of the density fluctuations in four rotating physical
cross-sections for each, plus a summary figure comparing particle content and
limiter flux.

The whole anatomy is written out below: the rotating-ellipse geometry, the
four-field parameter dataclass, the free-decay boundary conditions, the GMRES
phi solver, the seeded multi-mode initial state, and a plain RK4 stepping loop
with the sheath sink applied explicitly on the open geometry. Completed runs
are cached in ``closed_frames.npz`` / ``open_frames.npz`` so the movies can be
re-rendered without re-running the physics.

Run:

    PYTHONPATH=src python examples/stellarator/stellarator_turbulence.py

writes ``output/stellarator_turbulence/`` with two GIFs and a PNG. Movies are
release-hosted, not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_rotating_ellipse_geometry,
    rotating_ellipse_position,
)
from drbx.native import build_perp_laplacian_face_projectors  # noqa: E402
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState  # noqa: E402
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402
from drbx.native.stellarator_turbulence import (  # noqa: E402
    TEMPERATURE,
    TurbulenceRun,
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
)

# ----------------------------- PARAMETERS -----------------------------------
# Rotating-ellipse geometry (arguments of build_rotating_ellipse_geometry):
R0 = 3.0                 # torus major-radius offset
ELONGATION = 0.35        # ellipse deformation delta; aspect ratio (1+d)/(1-d)
NFP = 1                  # field periods: ellipse rotations per toroidal turn
IOTA = 0.9               # rotational transform of the helical field lines
LIMITER_RADIUS = 0.6     # flux surfaces with x > this hit the toroidal limiter
                         # (only the "open" run uses it; None keeps all closed)
SHAPE = (20, 32, 12)     # (radial, poloidal, toroidal) cell-centered grid

# Four-field model (Fci4FieldBlobParameters fields):
RHO_STAR = 1.0           # drift scale; sets the interchange drive strength
PHI_TOL = 5.0e-5         # GMRES tolerance of the phi (vorticity) inversion
PHI_MAXITER = 100        # GMRES max iterations per phi inversion
PHI_RESTART = 200        # GMRES restart length

# Time stepping:
DT = 2.0e-3              # RK4 timestep (normalized time units)
STEPS = 144              # number of RK4 steps
FRAME_STRIDE = 4         # record a movie frame every this many steps

# Seeded multi-mode density perturbation:
SEED = 1                 # RNG seed for the random mode phases/amplitudes
AMPLITUDE = 0.08         # peak relative density perturbation
MODES = ((2, 1), (3, 2), (4, 1), (5, 3))  # (poloidal m, toroidal n) seed modes

# Movie settings:
UPSAMPLE = 4             # smooth rendering: interpolate the coarse grid
MOVIE_FPS = 10           # GIF frames per second
MOVIE_DPI = 88           # GIF resolution
OUTPUT_DIR = Path("output/stellarator_turbulence")
# ----------------------------------------------------------------------------


def _fine_axes(geometry):
    x_coarse = np.asarray(geometry.grid.x.centers)
    theta_coarse = np.asarray(geometry.grid.y.centers)
    x_fine = np.linspace(x_coarse[0], x_coarse[-1], len(x_coarse) * UPSAMPLE)
    theta_fine = np.linspace(0.0, 2.0 * np.pi, len(theta_coarse) * UPSAMPLE + 1)
    return x_coarse, theta_coarse, x_fine, theta_fine


def _upsample_plane(field_xt, x_coarse, theta_coarse, x_fine, theta_fine):
    """Periodic interpolation in theta, linear in x, onto the fine grid."""

    fine_theta = np.apply_along_axis(
        lambda row: np.interp(theta_fine, theta_coarse, row, period=2.0 * np.pi), 1, field_xt)
    return np.apply_along_axis(lambda col: np.interp(x_fine, x_coarse, col), 0, fine_theta)


def _physical_plane_fine(geometry, zeta, x_fine, theta_fine):
    xx, tt = np.meshgrid(x_fine, theta_fine, indexing="ij")
    position = np.asarray(rotating_ellipse_position(
        jnp.asarray(xx), jnp.asarray(tt), jnp.asarray(zeta),
        r0=R0, elongation=ELONGATION, n_field_periods=NFP))
    return np.hypot(position[..., 0], position[..., 1]), position[..., 2]


def save_movie(run, geometry, title, path):
    """One row of four toroidal cross-sections, smoothly interpolated."""

    nz = geometry.shape[2]
    z_indices = [0, nz // 4, nz // 2, (3 * nz) // 4]
    zeta_values = [float(geometry.grid.z.centers[k]) for k in z_indices]
    fluctuation = run.density_frames - 1.0
    vmax = float(np.abs(fluctuation).max()) or 1.0
    x_coarse, theta_coarse, x_fine, theta_fine = _fine_axes(geometry)

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5), constrained_layout=True)
    meshes = []
    for ax, z_index, zeta in zip(axes, z_indices, zeta_values):
        R, Z = _physical_plane_fine(geometry, zeta, x_fine, theta_fine)
        field = _upsample_plane(fluctuation[0][:, :, z_index], x_coarse, theta_coarse, x_fine, theta_fine)
        mesh = ax.pcolormesh(R, Z, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"zeta = {zeta:.2f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        meshes.append((mesh, z_index))
    fig.colorbar(meshes[-1][0], ax=axes, shrink=0.85, label="density fluctuation")
    suptitle = fig.suptitle(f"{title}   t = 0.000")

    def update(frame_index):
        for mesh, z_index in meshes:
            field = _upsample_plane(fluctuation[frame_index][:, :, z_index],
                                    x_coarse, theta_coarse, x_fine, theta_fine)
            mesh.set_array(field.ravel())
        suptitle.set_text(f"{title}   t = {run.times[frame_index]:.3f}")
        return [m for m, _ in meshes]

    movie = animation.FuncAnimation(fig, update, frames=len(run.times), interval=100, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=MOVIE_FPS), dpi=MOVIE_DPI)
    plt.close(fig)
    print(f"[movie] wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


# --------------------- Geometry: closed and limiter SOL ---------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[geometry] building rotating-ellipse geometries (closed + limiter SOL)...")
closed_geometry = build_rotating_ellipse_geometry(
    SHAPE, r0=R0, elongation=ELONGATION, n_field_periods=NFP, iota=IOTA)
open_geometry = build_rotating_ellipse_geometry(
    SHAPE, r0=R0, elongation=ELONGATION, n_field_periods=NFP, iota=IOTA,
    limiter_radius=LIMITER_RADIUS)
print(f"[geometry] shape={SHAPE}, open SOL beyond x > {LIMITER_RADIUS}")

# ------------------- Run (or load) the two turbulence runs ------------------
runs: dict[str, TurbulenceRun] = {}
for name, geometry, use_sheath in (("closed", closed_geometry, False),
                                   ("open", open_geometry, True)):
    cache = OUTPUT_DIR / f"{name}_frames.npz"
    if cache.exists():
        data = np.load(cache)
        runs[name] = TurbulenceRun(data["density"], data["omega"], data["times"],
                                   data["content"], data["flux"])
        print(f"[run:{name}] loaded cached frames from {cache}")
        continue

    # Four-field model parameters and the operator scaffold, built explicitly.
    print(f"[run:{name}] building four-field operator scaffold (stencils, curvature, phi solver)...")
    parameters = Fci4FieldBlobParameters(rho_star=RHO_STAR, phi_inversion_tol=PHI_TOL,
                                         phi_inversion_maxiter=PHI_MAXITER,
                                         phi_inversion_restart=PHI_RESTART)
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_conditions = build_free_decay_boundary_conditions(geometry)
    curvature = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    projectors = build_perp_laplacian_face_projectors(geometry)
    phi_solver = build_four_field_phi_solver(
        geometry, parameters,
        conservative_stencil_builder=conservative_builder, face_projectors=projectors)

    # Seeded multi-mode initial state: random-phase modes, radial envelope,
    # pure density (the curvature drive must generate the vorticity itself).
    rng = np.random.default_rng(SEED)
    x = np.asarray(geometry.grid.x.centers)[:, None, None]
    theta = np.asarray(geometry.grid.y.centers)[None, :, None]
    zeta = np.asarray(geometry.grid.z.centers)[None, None, :]
    envelope = np.sin(np.pi * (x - x.min()) / (x.max() - x.min()))
    perturbation = np.zeros(geometry.shape)
    for m, n in MODES:
        perturbation += rng.uniform(0.5, 1.0) * np.cos(m * theta + n * zeta + rng.uniform(0, 2 * np.pi))
    state = Fci4FieldState(
        density=jnp.asarray(1.0 + AMPLITUDE * envelope * perturbation),
        omega=jnp.zeros(geometry.shape, dtype=jnp.float64),
        v_ion_parallel=jnp.zeros(geometry.shape, dtype=jnp.float64),
        v_electron_parallel=jnp.zeros(geometry.shape, dtype=jnp.float64))

    temperature = jnp.full(geometry.shape, TEMPERATURE)
    jacobian = np.asarray(geometry.cell_metric.J)
    density_frames = [np.asarray(state.density, dtype=np.float32)]
    omega_frames = [np.asarray(state.omega, dtype=np.float32)]
    times = [0.0]
    content = [float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian))]
    if use_sheath:
        sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
        flux = [float(sheath.total_ion_particle_loss)]
    else:
        flux = [0.0]

    # The stepping loop: classic RK4 on the four-field RHS (phi guess carried
    # between steps to keep the GMRES inversion warm), then -- on the open
    # geometry -- the explicit Bohm sheath density sink on the limiter cells.
    print(f"[run:{name}] stepping {STEPS} RK4 steps, dt={DT:g} (first step compiles)...")
    phi_guess = None
    for step_index in range(1, STEPS + 1):
        state, phi_guess = four_field_rk4_step(
            state, geometry=geometry, timestep=DT, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=phi_guess)

        sheath_loss = 0.0
        if use_sheath:
            sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
            state = Fci4FieldState(
                density=jnp.maximum(state.density - DT * sheath.ion_particle_loss, 1.0e-4),
                omega=state.omega, v_ion_parallel=state.v_ion_parallel,
                v_electron_parallel=state.v_electron_parallel)
            sheath_loss = float(sheath.total_ion_particle_loss)

        fluct = np.asarray(state.density, dtype=np.float64) - 1.0
        energy = float(np.mean(fluct**2 + np.asarray(state.omega, dtype=np.float64) ** 2))
        progress = f"[run:{name}] step {step_index:3d}/{STEPS}  t={step_index * DT:.4f}  field energy={energy:.4e}"
        if use_sheath:
            progress += f"  sheath loss={sheath_loss:.4e}"
        print(progress)

        if step_index % FRAME_STRIDE == 0 or step_index == STEPS:
            density_frames.append(np.asarray(state.density, dtype=np.float32))
            omega_frames.append(np.asarray(state.omega, dtype=np.float32))
            times.append(step_index * DT)
            content.append(float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian)))
            if use_sheath:
                sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
                flux.append(float(sheath.total_ion_particle_loss))
            else:
                flux.append(0.0)

    runs[name] = TurbulenceRun(np.stack(density_frames), np.stack(omega_frames),
                               np.asarray(times), np.asarray(content), np.asarray(flux))
    np.savez_compressed(cache, density=runs[name].density_frames, omega=runs[name].omega_frames,
                        times=runs[name].times, content=runs[name].particle_content,
                        flux=runs[name].target_flux)
    print(f"[run:{name}] done; cached frames in {cache}")

closed = runs["closed"]
open_run = runs["open"]

# ------------------------------- Movies -------------------------------------
print("[movie] rendering closed-field-line movie...")
save_movie(closed, closed_geometry, "Stellarator turbulence (closed field lines)",
           OUTPUT_DIR / "stellarator_turbulence_closed.gif")
print("[movie] rendering open-field-line movie...")
save_movie(open_run, open_geometry, "Stellarator SOL turbulence (open field lines)",
           OUTPUT_DIR / "stellarator_turbulence_open.gif")

# --------------------------- Summary figure ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
axes[0].plot(closed.times, closed.particle_content / closed.particle_content[0],
             label="closed", color="#1f77b4")
axes[0].plot(open_run.times, open_run.particle_content / open_run.particle_content[0],
             label="open (limiter SOL)", color="#d62728")
axes[0].set_xlabel("time"), axes[0].set_ylabel("particle content (norm.)")
axes[0].set_title("Open field lines drain to the limiter")
axes[0].legend(fontsize=8), axes[0].grid(True, ls=":", alpha=0.4)
axes[1].plot(open_run.times, open_run.target_flux, color="#d62728")
axes[1].set_xlabel("time"), axes[1].set_ylabel("total limiter ion flux")
axes[1].set_title("Bohm sheath flux at the limiter")
axes[1].grid(True, ls=":", alpha=0.4)
fig.suptitle("Stellarator turbulence: closed vs open field lines (rotating ellipse)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "stellarator_turbulence_summary.png", dpi=170)
plt.close(fig)

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "steps": STEPS, "dt": DT, "shape": list(SHAPE), "limiter_radius": LIMITER_RADIUS,
    "closed_content_change": float(closed.particle_content[-1] - closed.particle_content[0]),
    "open_content_change": float(open_run.particle_content[-1] - open_run.particle_content[0]),
    "final_limiter_flux": float(open_run.target_flux[-1]),
    "closed_omega_max": float(np.abs(closed.omega_frames[-1]).max()),
}, indent=2))
print(f"[done] wrote {OUTPUT_DIR / 'stellarator_turbulence_summary.png'} and summary.json")
