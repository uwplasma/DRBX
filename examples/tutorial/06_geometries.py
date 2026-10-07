"""Tutorial 06 -- 3-D geometries: rotating-ellipse FCI torus, a limiter SOL, an island divertor.

FCI (flux-coordinate independent) geometry: each toroidal plane has its own
(x, theta) grid and the parallel derivative follows traced field lines from
one plane to the next. This script

1. builds the rotating-ellipse stellarator (elliptical cross-section turning
   with the toroidal angle zeta; metric from autodiff of the embedding) and
   draws flux surfaces at four zeta,
2. runs the 4-field interchange model (density n, vorticity Omega, ion and
   electron parallel velocities) from the same seed twice: all field lines
   closed, and with a toroidal limiter opening x > LIMITER_RADIUS (Bohm sheath
   sink on the cells whose field-line maps hit the limiter), and
3. maps the connection length of the analytic island-divertor field: closed
   core, island chains, and an open stochastic edge, all emerging from tracing.

Run from the repository root (page: docs/tutorial/06_geometries.md):

    PYTHONPATH=src python examples/tutorial/06_geometries.py
"""

# 1. Imports ------------------------------------------------------------------
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.geometry import (
    ConservativeStencilBuilder,
    IslandDivertorField,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_rotating_ellipse_geometry,
    island_divertor_connection_length,
    rotating_ellipse_position,
)
from drbx.native import build_perp_laplacian_face_projectors
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters
from drbx.native.stellarator_turbulence import (
    apply_sheath_sink,
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
    multi_mode_state,
)

# 2. Input parameters -----------------------------------------------------------
SHAPE = (12, 16, 8)        # (radial x, poloidal theta, toroidal zeta) cells
ELONGATION = 0.35          # ellipse deformation: aspect ratio (1 + d) / (1 - d)
LIMITER_RADIUS = 0.6       # field lines with x > this end on a limiter at zeta = 0
DT = 2.0e-3                # RK4 step of the 4-field model
STEPS = 100                # steps per run
REPORT_EVERY = 20
CONNECTION_GRID = (60, 64)  # (x, theta) starting points of the island-divertor map
MAX_TRANSITS = 30           # tracing cutoff (toroidal turns)
FIGURE = Path("docs/media/tutorial_06_geometries.png")


def run_four_field(geometry, sheath, label):
    """Seeded 4-field run; returns times and Jacobian-weighted particle content."""
    parameters = Fci4FieldBlobParameters(rho_star=1.0, phi_inversion_tol=5.0e-5,
                                         phi_inversion_maxiter=100, phi_inversion_restart=200)
    conservative = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    projectors = build_perp_laplacian_face_projectors(geometry)
    operators = dict(
        geometry=geometry, timestep=DT, parameters=parameters,
        curvature_coefficients=build_curvature_coefficients(geometry, periodic_axes=(False, True, True)),
        stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
        conservative_stencil_builder=conservative,
        boundary_conditions=build_free_decay_boundary_conditions(geometry),
        phi_face_projectors=projectors,
        phi_inverse_solver=build_four_field_phi_solver(geometry, parameters, conservative_stencil_builder=conservative,
                                                       face_projectors=projectors))
    step = jax.jit(lambda state, guess: four_field_rk4_step(state, phi_guess=guess, **operators))
    state, guess = multi_mode_state(geometry, seed=1), jnp.zeros(geometry.shape)
    jacobian = np.asarray(geometry.cell_metric.J)
    times, content = [0.0], [float(np.sum(np.asarray(state.density) * jacobian))]
    start = time.perf_counter()
    for index in range(1, STEPS + 1):
        state, guess = step(state, guess)
        loss = 0.0
        if sheath:
            state, loss = apply_sheath_sink(state, geometry, DT)
        times.append(index * DT)
        content.append(float(np.sum(np.asarray(state.density) * jacobian)))
        if index % REPORT_EVERY == 0 or index == 1:
            print(f"[06]   {label} step={index:3d} t={index * DT:.3f} max|Omega|={float(jnp.max(jnp.abs(state.omega))):.3e} "
                  f"sheath loss={loss:.3e} ({time.perf_counter() - start:.1f}s, step 1 compiles)")
    return np.array(times), np.array(content) / content[0]


# 3. Run -------------------------------------------------------------------------
start = time.perf_counter()
print(f"[06] rotating ellipse {SHAPE}, elongation {ELONGATION}; building closed and limiter geometries")
closed = build_rotating_ellipse_geometry(SHAPE, elongation=ELONGATION)
limited = build_rotating_ellipse_geometry(SHAPE, elongation=ELONGATION, limiter_radius=LIMITER_RADIUS)
open_cells = int(np.asarray(limited.maps.forward_boundary).sum() + np.asarray(limited.maps.backward_boundary).sum())
print(f"[06] limiter geometry: {open_cells} open field-line endpoints (closed geometry: 0)")
runs = {"closed": run_four_field(closed, False, "closed"),
        "limiter": run_four_field(limited, True, "limiter")}

print(f"[06] island divertor: tracing {CONNECTION_GRID[0] * CONNECTION_GRID[1]} field lines "
      f"for up to {MAX_TRANSITS} transits")
field = IslandDivertorField()
x = jnp.linspace(field.x_min + 0.01, field.x_max - 0.01, CONNECTION_GRID[0])
theta = jnp.linspace(0.0, 2.0 * jnp.pi, CONNECTION_GRID[1])
xx, tt = jnp.meshgrid(x, theta, indexing="ij")
transits, is_open = (np.asarray(a) for a in island_divertor_connection_length(
    field, xx, tt, jnp.zeros_like(xx), max_transits=MAX_TRANSITS))

# 4. Results ----------------------------------------------------------------------
# Closed field lines: the curvature drive turns the density seed into vorticity,
# and the particle content changes only slightly (free-decay radial walls). Limiter: the same turbulence, plus a sheath drain
# on the open SOL endpoints, so the particle content falls.
for label, (_, content) in runs.items():
    print(f"[06] {label}: particle content {content[0]:.4f} -> {content[-1]:.6f}")
x_np = np.asarray(x)
print(f"[06] island divertor: open fraction {is_open[x_np < 0.5].mean():.2f} in the core (x<0.5), "
      f"{is_open[x_np > 0.9].mean():.2f} at the edge (x>0.9)")
print(f"[06] total {time.perf_counter() - start:.1f}s")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), layout="constrained")
angles = np.linspace(0.0, 2.0 * np.pi, 200)
for zeta, color in zip(np.linspace(0, np.pi / 2, 4), plt.cm.viridis(np.linspace(0.1, 0.9, 4))):
    for radius in (0.4, 0.7, 1.0):
        position = np.asarray(rotating_ellipse_position(jnp.asarray(radius), jnp.asarray(angles), jnp.asarray(zeta),
                                                        r0=3.0, elongation=ELONGATION, n_field_periods=1))
        axes[0].plot(np.hypot(position[:, 0], position[:, 1]) - 3.0, position[:, 2], color=color,
                     lw=1.5 if radius == 1.0 else 0.6, label=f"zeta={zeta:.2f}" if radius == 1.0 else None)
axes[0].set(title="rotating-ellipse flux surfaces", xlabel="R - R0", ylabel="Z", aspect="equal")
axes[0].legend(fontsize=6)
for label, (times, content) in runs.items():
    axes[1].plot(times, content, label=label)
axes[1].set(title="particle content (normalized)", xlabel="time")
axes[1].legend(fontsize=7)
mesh = axes[2].pcolormesh(np.asarray(theta), x_np, np.where(is_open, transits, np.nan), cmap="magma_r", shading="auto")
axes[2].set_facecolor("#dce6f2")
fig.colorbar(mesh, ax=axes[2], label="transits to the wall")
axes[2].set(title="island divertor: connection length\n(blue = closed)", xlabel="theta", ylabel="x")
fig.savefig(FIGURE, dpi=80)
print(f"[06] wrote {FIGURE}")
