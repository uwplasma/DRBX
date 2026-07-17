"""3D stellarator renders: cutaway turbulence movie + closed/open field lines.

Two engaging three-dimensional views of the rotating-ellipse stellarator:

1. ``stellarator_3d_turbulence.gif`` -- a cutaway torus whose surface is colored
   by the evolving turbulent density fluctuation from a multi-mode four-field
   run, with poloidal cross-section caps showing the interior and a slowly
   orbiting camera. The rotating elliptical cross-section is visible directly.
2. ``stellarator_3d_field_lines.png`` -- helical field lines on the same
   geometry: closed core field lines wind around the torus indefinitely, while
   scrape-off-layer field lines beyond the limiter radius are open -- they end
   on the toroidal limiter after one transit.

The turbulence run below is written out explicitly -- geometry, four-field
parameters, boundary conditions, phi solver, multi-mode seed, and a plain RK4
stepping loop -- and its frames are cached in ``frames.npz`` so the movie can
be re-rendered without re-running the physics.

Run:

    PYTHONPATH=src python examples/stellarator/stellarator_3d_render.py

writes ``output/stellarator_3d/``. Both are release-hosted, not committed.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

from jax_drb.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_rotating_ellipse_geometry,
    rotating_ellipse_position,
)
from jax_drb.native import build_perp_laplacian_face_projectors  # noqa: E402
from jax_drb.native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState  # noqa: E402
from jax_drb.native.stellarator_turbulence import (  # noqa: E402
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
X_MIN, X_MAX = 0.2, 1.0  # minor-radius label bounds of the flux tube
LIMITER_RADIUS = 0.6     # SOL boundary drawn in the field-line figure
SHAPE = (16, 32, 16)     # (radial, poloidal, toroidal) cell-centered grid

# Four-field model (Fci4FieldBlobParameters fields):
RHO_STAR = 1.0           # drift scale; sets the interchange drive strength
PHI_TOL = 5.0e-5         # GMRES tolerance of the phi (vorticity) inversion
PHI_MAXITER = 100        # GMRES max iterations per phi inversion
PHI_RESTART = 200        # GMRES restart length

# Time stepping:
DT = 2.0e-3              # RK4 timestep (normalized time units)
STEPS = 120              # number of RK4 steps
FRAME_STRIDE = 4         # record a movie frame every this many steps

# Seeded multi-mode density perturbation:
SEED = 1                 # RNG seed for the random mode phases/amplitudes
AMPLITUDE = 0.08         # peak relative density perturbation
MODES = ((2, 1), (3, 2), (4, 1), (5, 3))  # (poloidal m, toroidal n) seed modes

# Movie / render settings:
WEDGE = 0.25 * 2.0 * np.pi          # cutaway opening angle
FINE_THETA, FINE_ZETA = 96, 120     # upsampled surface resolution
MOVIE_FPS = 9                       # GIF frames per second
MOVIE_DPI = 100                     # GIF resolution
OUTPUT_DIR = Path("output/stellarator_3d")
# ----------------------------------------------------------------------------


def _positions(x, theta, zeta):
    p = np.asarray(rotating_ellipse_position(
        jnp.asarray(x), jnp.asarray(theta), jnp.asarray(zeta),
        r0=R0, elongation=ELONGATION, n_field_periods=NFP))
    return p[..., 0], p[..., 1], p[..., 2]


def _periodic_upsample(field, coarse, fine):
    """Periodic linear interpolation along one axis (last axis of ``field``)."""

    return np.apply_along_axis(
        lambda row: np.interp(fine, coarse, row, period=2.0 * np.pi), -1, field)


def render_turbulence_movie(run, geometry, path):
    x_centers = np.asarray(geometry.grid.x.centers)
    theta_c = np.asarray(geometry.grid.y.centers)
    zeta_c = np.asarray(geometry.grid.z.centers)
    fluctuation = run.density_frames - 1.0

    # Color a substantial outer flux surface (fluctuations carry a radial
    # envelope, so the outermost surfaces are quiet; 0.8 of the radius is both
    # active and visually representative of the rotating ellipse).
    x_target = X_MIN + 0.8 * (X_MAX - X_MIN)
    surface_index = int(np.argmin(np.abs(x_centers - x_target)))
    x_surface = float(x_centers[surface_index])

    theta_f = np.linspace(0.0, 2.0 * np.pi, FINE_THETA)
    zeta_f = np.linspace(WEDGE, 2.0 * np.pi, FINE_ZETA)  # cutaway: skip the wedge
    theta_grid, zeta_grid = np.meshgrid(theta_f, zeta_f, indexing="ij")
    Xs, Ys, Zs = _positions(np.full_like(theta_grid, x_surface), theta_grid, zeta_grid)

    # Cross-section caps at the two wedge faces.
    x_cap = np.linspace(X_MIN, x_surface, 24)
    cap_x, cap_theta = np.meshgrid(x_cap, theta_f, indexing="ij")
    caps = [_positions(cap_x, cap_theta, np.full_like(cap_x, z)) for z in (WEDGE, 2.0 * np.pi)]

    vmax = float(np.abs(fluctuation).max()) or 1.0
    norm = colors.Normalize(-vmax, vmax)
    cmap = cm.RdBu_r

    def _bilinear(field_tz):
        """Upsample a (theta, zeta) field to the fine surface grid, periodically."""

        fine_zeta = _periodic_upsample(field_tz, zeta_c, zeta_f)        # (ny, FZ)
        return _periodic_upsample(fine_zeta.T, theta_c, theta_f).T      # (FT, FZ)

    def cap_colors(frame, z_index):
        field_xt = fluctuation[frame][:, :, z_index]      # (nx, ny)
        fine = _periodic_upsample(field_xt, theta_c, theta_f)   # (nx, FT)
        return cmap(norm(np.stack([np.interp(x_cap, x_centers, fine[:, j]) for j in range(FINE_THETA)], axis=1)))

    cap_z_indices = (int(np.argmin(np.abs(zeta_c - WEDGE))), 0)

    fig = plt.figure(figsize=(6.0, 3.6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame):
        ax.clear()
        ax.set_axis_off()
        colors_s = _bilinear(fluctuation[frame][surface_index])
        ax.plot_surface(Xs, Ys, Zs, facecolors=cmap(norm(colors_s)), shade=False,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
        for (cx, cy, cz), z_index in zip(caps, cap_z_indices):
            ax.plot_surface(cx, cy, cz, facecolors=cap_colors(frame, z_index), shade=False,
                            rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_box_aspect((1, 1, 0.34))
        ax.set_xlim(-3.85, 3.85), ax.set_ylim(-3.85, 3.85), ax.set_zlim(-1.35, 1.35)
        fig.subplots_adjust(left=-0.22, right=1.22, bottom=-0.32, top=1.28)
        ax.view_init(elev=24, azim=-55 + 1.2 * frame)
        ax.set_title(f"Stellarator turbulence (rotating ellipse)   t = {run.times[frame]:.3f}",
                     fontsize=9, y=0.96)

    movie = animation.FuncAnimation(fig, draw, frames=len(run.times), interval=110, blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=MOVIE_FPS), dpi=MOVIE_DPI)
    plt.close(fig)
    print(f"[movie] wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def render_field_lines(path):
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    # Faint boundary surface for context (with the same cutaway wedge).
    theta_f = np.linspace(0.0, 2.0 * np.pi, 60)
    zeta_f = np.linspace(WEDGE, 2.0 * np.pi, 90)
    tg, zg = np.meshgrid(theta_f, zeta_f, indexing="ij")
    Xb, Yb, Zb = _positions(np.full_like(tg, X_MAX), tg, zg)
    ax.plot_surface(Xb, Yb, Zb, color="lightsteelblue", alpha=0.10, shade=False,
                    rstride=2, cstride=2, linewidth=0, antialiased=False)

    # Closed core field lines: theta = theta0 + iota * zeta, three toroidal turns.
    zeta_line = np.linspace(0.0, 6.0 * np.pi, 900)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
        X, Y, Z = _positions(np.full_like(zeta_line, 0.45), theta0 + IOTA * zeta_line, zeta_line)
        ax.plot(X, Y, Z, color="#1f77b4", lw=0.9, alpha=0.9)

    # Open SOL field lines: one transit, ending on the limiter at zeta = 0 / 2 pi.
    zeta_open = np.linspace(0.0, 2.0 * np.pi, 300)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False):
        X, Y, Z = _positions(np.full_like(zeta_open, 0.82), theta0 + IOTA * zeta_open, zeta_open)
        ax.plot(X, Y, Z, color="#d62728", lw=1.3)
        ax.scatter(X[[0, -1]], Y[[0, -1]], Z[[0, -1]], color="#d62728", s=14)

    # The limiter: a translucent annulus at the zeta = 0 poloidal plane.
    x_ann = np.linspace(LIMITER_RADIUS, X_MAX, 8)
    ta, xa = np.meshgrid(np.linspace(0.0, 2.0 * np.pi, 60), x_ann, indexing="ij")
    Xl, Yl, Zl = _positions(xa, ta, np.zeros_like(xa))
    ax.plot_surface(Xl, Yl, Zl, color="dimgray", alpha=0.45, shade=False, linewidth=0)

    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=26, azim=-50)
    ax.set_xlim(-3.4, 3.4), ax.set_ylim(-3.4, 3.4), ax.set_zlim(-1.5, 1.5)
    ax.set_title("Rotating-ellipse stellarator: closed core field lines (blue) and\n"
                 "open scrape-off-layer field lines ending on the limiter (red)",
                 fontsize=10, y=0.92)
    fig.subplots_adjust(left=0, right=1, bottom=-0.12, top=1.05)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[figure] wrote {path}")


# ----------------------- Field-line geometry figure -------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[figure] rendering closed/open field-line figure...")
render_field_lines(OUTPUT_DIR / "stellarator_3d_field_lines.png")

# --------------------- Turbulence run (or cached frames) --------------------
print("[geometry] building rotating-ellipse geometry...")
geometry = build_rotating_ellipse_geometry(
    SHAPE, r0=R0, x_min=X_MIN, x_max=X_MAX, elongation=ELONGATION,
    n_field_periods=NFP, iota=IOTA)

cache = OUTPUT_DIR / "frames.npz"
if cache.exists():
    data = np.load(cache)
    run = TurbulenceRun(data["density"], data["omega"], data["times"],
                        data["content"], data["flux"])
    print(f"[run] loaded cached turbulence frames from {cache}")
else:
    # Four-field model parameters and the operator scaffold, built explicitly.
    print("[run] building four-field operator scaffold (stencils, curvature, phi solver)...")
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

    jacobian = np.asarray(geometry.cell_metric.J)
    density_frames = [np.asarray(state.density, dtype=np.float32)]
    omega_frames = [np.asarray(state.omega, dtype=np.float32)]
    times = [0.0]
    content = [float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian))]
    flux = [0.0]  # closed field lines: no sheath sink anywhere

    # The stepping loop: classic RK4 on the four-field RHS, phi guess carried
    # between steps to keep the GMRES inversion warm.
    print(f"[run] stepping {STEPS} RK4 steps, dt={DT:g} (first step compiles)...")
    phi_guess = None
    for step_index in range(1, STEPS + 1):
        state, phi_guess = four_field_rk4_step(
            state, geometry=geometry, timestep=DT, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=phi_guess)

        fluct = np.asarray(state.density, dtype=np.float64) - 1.0
        energy = float(np.mean(fluct**2 + np.asarray(state.omega, dtype=np.float64) ** 2))
        print(f"[run] step {step_index:3d}/{STEPS}  t={step_index * DT:.4f}  field energy={energy:.4e}")

        if step_index % FRAME_STRIDE == 0 or step_index == STEPS:
            density_frames.append(np.asarray(state.density, dtype=np.float32))
            omega_frames.append(np.asarray(state.omega, dtype=np.float32))
            times.append(step_index * DT)
            content.append(float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian)))
            flux.append(0.0)

    run = TurbulenceRun(np.stack(density_frames), np.stack(omega_frames),
                        np.asarray(times), np.asarray(content), np.asarray(flux))
    np.savez_compressed(cache, density=run.density_frames, omega=run.omega_frames,
                        times=run.times, content=run.particle_content, flux=run.target_flux)
    print(f"[run] turbulence run done; cached frames in {cache}")

# ------------------------------ Cutaway movie -------------------------------
print("[movie] rendering cutaway turbulence movie...")
render_turbulence_movie(run, geometry, OUTPUT_DIR / "stellarator_3d_turbulence.gif")
