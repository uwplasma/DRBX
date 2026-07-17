"""Seeded-filament dynamics on the rotating ellipse (B7, second half).

Seeds a localized density blob on the genuinely non-axisymmetric rotating-ellipse
geometry and evolves the four-field drift-reduced FCI model (density, vorticity,
ion/electron parallel velocity) as a short free run. The curvature drive spins up
vorticity from the pressure blob -- the interchange mechanism that moves a
filament -- and because the flux surfaces rotate with the toroidal angle, the
filament evolves differently in each toroidal plane.

Everything is written out below: the rotating-ellipse geometry, the Gaussian
blob seed, the free-decay boundary conditions, the GMRES phi solver, and a plain
RK4 stepping loop -- the same four-field machinery gated in
``tests/test_rotating_ellipse_filament.py``. The figure is drawn in physical
(R, Z) space so the rotating elliptical cross-section is visible.

Run:

    PYTHONPATH=src python examples/stellarator/rotating_ellipse_filament.py

writes ``output/rotating_ellipse_filament/`` with a PNG (density + vorticity in
the rotating cross-sections, initial vs evolved) and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_rotating_ellipse_geometry,
    rotating_ellipse_position,
)
from dkx.native import build_perp_laplacian_face_projectors  # noqa: E402
from dkx.native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState  # noqa: E402
from dkx.native.stellarator_turbulence import (  # noqa: E402
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
)

# ----------------------------- PARAMETERS -----------------------------------
# Rotating-ellipse geometry (arguments of build_rotating_ellipse_geometry):
R0 = 3.0                 # torus major-radius offset
X_MIN, X_MAX = 0.2, 1.0  # minor-radius label bounds of the flux tube
ELONGATION = 0.35        # ellipse deformation delta; aspect ratio (1+d)/(1-d)
N_FIELD_PERIODS = 1      # field periods: ellipse rotations per toroidal turn
IOTA = 0.9               # rotational transform of the helical field lines
C_PHI = 3.0              # toroidal-angle scale factor of the embedding
SHAPE = (20, 28, 12)     # (radial, poloidal, toroidal) cell-centered grid

# Four-field model (Fci4FieldBlobParameters fields; perpendicular diffusion
# defaults of 1e-2 for every field are kept):
RHO_STAR = 1.0           # drift scale; sets the interchange drive strength
PHI_TOL = 5.0e-5         # GMRES tolerance of the phi (vorticity) inversion
PHI_MAXITER = 100        # GMRES max iterations per phi inversion
PHI_RESTART = 200        # GMRES restart length

# Time stepping:
DT = 2.0e-3              # RK4 timestep (normalized time units)
N_STEPS = 40             # number of RK4 steps

# Gaussian blob seed (values shared with the shifted-torus blob harness):
BLOB_BACKGROUND = 1.0    # background density n_bg
BLOB_AMPLITUDE = 0.1     # relative blob amplitude A_blob
BLOB_X0 = 0.575          # radial blob center
BLOB_THETA0 = np.pi      # poloidal blob center
BLOB_ZETA0 = np.pi       # toroidal blob center
BLOB_SIGMA_X = 0.085     # radial blob width
BLOB_SIGMA_THETA = 0.25  # poloidal blob width
BLOB_SIGMA_ZETA = 0.25   # toroidal blob width

OUTPUT_DIR = Path("output/rotating_ellipse_filament")
# ----------------------------------------------------------------------------


def _physical_plane(geometry, z_index):
    """(R, Z) coordinates of the cell centers in one toroidal plane."""

    x = np.asarray(geometry.grid.x.centers)
    theta = np.asarray(geometry.grid.y.centers)
    zeta = float(geometry.grid.z.centers[z_index])
    xx, tt = np.meshgrid(x, theta, indexing="ij")
    position = np.asarray(
        rotating_ellipse_position(
            jnp.asarray(xx), jnp.asarray(tt), jnp.asarray(zeta),
            r0=R0, elongation=ELONGATION, n_field_periods=N_FIELD_PERIODS,
        )
    )
    major_radius = np.hypot(position[..., 0], position[..., 1])
    return major_radius, position[..., 2], zeta


# ------------------------------- Geometry -----------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[geometry] building rotating-ellipse geometry...")
geometry = build_rotating_ellipse_geometry(
    SHAPE, r0=R0, x_min=X_MIN, x_max=X_MAX, elongation=ELONGATION,
    n_field_periods=N_FIELD_PERIODS, iota=IOTA, c_phi=C_PHI,
)

# --------------- Four-field parameters and operator scaffold ----------------
print("[setup] building four-field operator scaffold (stencils, curvature, phi solver)...")
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

# --------------------------- Gaussian blob seed -----------------------------
# Pure density blob: no vorticity and no parallel flow in the seed -- the
# curvature drive must generate them (the interchange filament mechanism).
print("[setup] seeding the Gaussian density blob...")
x = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[:, None, None]
theta = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[None, :, None]
zeta = jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)[None, None, :]
d_theta = jnp.arctan2(jnp.sin(theta - BLOB_THETA0), jnp.cos(theta - BLOB_THETA0))
d_zeta = jnp.arctan2(jnp.sin(zeta - BLOB_ZETA0), jnp.cos(zeta - BLOB_ZETA0))
blob = (
    jnp.exp(-((x - BLOB_X0) ** 2) / BLOB_SIGMA_X**2)
    * jnp.exp(-(d_theta**2) / BLOB_SIGMA_THETA**2)
    * jnp.exp(-(d_zeta**2) / BLOB_SIGMA_ZETA**2)
)
density = BLOB_BACKGROUND * (1.0 + BLOB_AMPLITUDE * blob)
density = jnp.broadcast_to(density, geometry.shape)
zeros = jnp.zeros(geometry.shape, dtype=jnp.float64)
initial_state = Fci4FieldState(density=density, omega=zeros,
                               v_ion_parallel=zeros, v_electron_parallel=zeros)

# ----------------------------- RK4 stepping ---------------------------------
print(f"[run] stepping {N_STEPS} RK4 steps, dt={DT:g} (first step compiles)...")
state = initial_state
phi_guess = None
for step_index in range(1, N_STEPS + 1):
    state, phi_guess = four_field_rk4_step(
        state, geometry=geometry, timestep=DT, parameters=parameters,
        curvature_coefficients=curvature, stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_builder,
        boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
        phi_inverse_solver=phi_solver, phi_guess=phi_guess)
    fluct = np.asarray(state.density, dtype=np.float64) - BLOB_BACKGROUND
    energy = float(np.mean(fluct**2 + np.asarray(state.omega, dtype=np.float64) ** 2))
    print(f"[run] step {step_index:3d}/{N_STEPS}  t={step_index * DT:.4f}  field energy={energy:.4e}")
final_state = state

# ------------------------------- Summary ------------------------------------
density = np.asarray(final_state.density)
omega = np.asarray(final_state.omega)
summary = {
    "shape": list(SHAPE),
    "n_steps": N_STEPS,
    "dt": DT,
    "finite": bool(np.all(np.isfinite(density)) and np.all(np.isfinite(omega))),
    "density_min": float(density.min()),
    "density_max": float(density.max()),
    "omega_max_abs": float(np.max(np.abs(omega))),
    "v_ion_max_abs": float(np.max(np.abs(np.asarray(final_state.v_ion_parallel)))),
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# ------------------------------- Figure -------------------------------------
# Bracket the blob's toroidal centre (zeta0 = pi) so each column shows the
# filament in a differently-oriented cross-section.
print("[figure] rendering the filament cross-sections...")
z_indices = [SHAPE[2] // 3, SHAPE[2] // 2, (2 * SHAPE[2]) // 3]

density0 = np.asarray(initial_state.density)
density1 = np.asarray(final_state.density)
omega1 = np.asarray(final_state.omega)
d_vmax = float(np.max(np.abs(np.concatenate([density0 - 1.0, density1 - 1.0]).ravel())))
w_vmax = float(np.max(np.abs(omega1))) or 1.0

fig, axes = plt.subplots(3, len(z_indices), figsize=(4.2 * len(z_indices), 11.0), squeeze=False)
for col, z_index in enumerate(z_indices):
    R, Z, zeta_plane = _physical_plane(geometry, z_index)
    panels = [
        (density0[:, :, z_index] - 1.0, "RdBu_r", d_vmax, f"density seed, zeta={zeta_plane:.2f}"),
        (density1[:, :, z_index] - 1.0, "RdBu_r", d_vmax, f"density evolved, zeta={zeta_plane:.2f}"),
        (omega1[:, :, z_index], "PuOr", w_vmax, f"vorticity evolved, zeta={zeta_plane:.2f}"),
    ]
    for row, (field, cmap, vmax, title) in enumerate(panels):
        ax = axes[row][col]
        mesh = ax.pcolormesh(R, Z, field, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        fig.colorbar(mesh, ax=ax, shrink=0.85)

fig.suptitle("Seeded filament on the rotating ellipse: vorticity generated in rotating cross-sections")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "rotating_ellipse_filament.png", dpi=170)
plt.close(fig)
print(f"[done] wrote {OUTPUT_DIR / 'rotating_ellipse_filament.png'}")
