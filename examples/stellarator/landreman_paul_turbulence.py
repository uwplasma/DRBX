"""Four-field turbulence on the Landreman-Paul stellarator: closed core + open SOL.

This example runs a drift-reduced four-field interchange turbulence simulation
on the real Landreman-Paul 2021 quasi-axisymmetric configuration, with both
closed and open field lines:

1. build a flux-coordinate-independent (FCI) geometry from the actual
   Landreman-Paul VMEC equilibrium -- the metric, the magnetic-field
   magnitude, and the surface-preserving parallel field-line maps all come
   from the real equilibrium (via the ESSOS import machinery), so the
   rotational transform recovered from the maps matches the equilibrium value
   (iota ~ 0.42);
2. open a scrape-off layer with a radial limiter: cells outside ``LIMITER_RHO``
   (the last closed flux surface, in the island-divertor / limiter sense) drain
   through a Bohm sheath at the toroidal target planes, while the core stays
   closed;
3. seed a multi-mode density perturbation and advance the four-field model with
   a plain RK4 loop, printing the field energy and the sheath particle loss
   each step;
4. compare against a closed reference run (no limiter, no sink) to show the SOL
   draining, and draw the final density fluctuations at one toroidal plane with
   the closed/open boundary marked.

The setup is written out explicitly -- geometry, four-field parameters,
boundary conditions, phi solver, seed, and the stepping loop -- so it can be
retargeted to a different equilibrium, grid, or limiter position by editing the
PARAMETERS block.

Requires an ESSOS checkout (coil set + matching VMEC wout for the
Landreman-Paul QA configuration resolve from it):

    DRBX_ESSOS_ROOT=~/local/ESSOS_test \
        PYTHONPATH=src python examples/stellarator/landreman_paul_turbulence.py

writes ``output/landreman_paul_turbulence/landreman_paul_turbulence.png``.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_essos_imported_fci_geometry,
    build_local_stencil_from_field,
    essos_imported_geometry_to_fci,
    essos_runtime_available,
)
from drbx.native import build_perp_laplacian_face_projectors  # noqa: E402
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters  # noqa: E402
from drbx.native.stellarator_turbulence import (  # noqa: E402
    apply_sheath_sink,
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
    multi_mode_state,
)

# PARAMETERS ---------------------------------------------------------------
# Geometry: an FCI grid traced on the Landreman-Paul QA VMEC equilibrium. The
# coil JSON and VMEC wout resolve from the ESSOS checkout (DRBX_ESSOS_ROOT).
# map_source="vmec" gives clean surface-preserving parallel maps (closed
# everywhere); the open scrape-off layer is added by the limiter cut below.
NX = 12                 # radial (rho) cells
NY_PHI = 8              # toroidal cells (field-line map planes)
NZ_THETA = 16           # poloidal cells
RHO_MIN = 0.12          # innermost normalized flux-surface label
RHO_MAX = 1.20          # outermost label (rho = 1 is the design LCFS)
TRACE_MAXTIME = 400.0   # ESSOS field-line integration time for the maps
TRACE_SAMPLES = 2048    # trajectory samples per traced map segment

# Limiter: cells with rho > LIMITER_RHO form the open SOL and drain through the
# Bohm sheath at the toroidal target planes; rho <= LIMITER_RHO stays closed.
# Set at the last closed flux surface (~rho 1.0); here slightly inside so the
# coarse radial grid resolves a few SOL shells.
LIMITER_RHO = 0.90

# Four-field interchange model (normalized units).
RHO_STAR = 1.0                  # drift scale / machine size
PHI_INVERSION_TOL = 5.0e-5      # GMRES tolerance for the per-stage phi solve
PHI_INVERSION_MAXITER = 100
PHI_INVERSION_RESTART = 200

# Time stepping and seed.
DT = 1.5e-3                     # RK4 timestep (normalized)
N_STEPS = 40                    # number of steps to advance
SEED = 1                        # RNG seed for the multi-mode density perturbation
SEED_AMPLITUDE = 0.08           # density perturbation amplitude
SEED_MODES = ((2, 1), (3, 2), (4, 1), (5, 3))  # (poloidal, toroidal) mode numbers

OUTPUT_DIR = Path("output/landreman_paul_turbulence")

# setup --------------------------------------------------------------------
if not essos_runtime_available():
    raise SystemExit(
        "ESSOS is not importable. Point DRBX_ESSOS_ROOT at a checkout, e.g.\n"
        "    DRBX_ESSOS_ROOT=~/local/ESSOS_test PYTHONPATH=src python "
        "examples/stellarator/landreman_paul_turbulence.py"
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("building the Landreman-Paul FCI geometry from the VMEC equilibrium...")
imported = build_essos_imported_fci_geometry(
    map_source="vmec",
    nx=NX,
    ny=NY_PHI,
    nz=NZ_THETA,
    rho_min=RHO_MIN,
    rho_max=RHO_MAX,
    maxtime=TRACE_MAXTIME,
    times_to_trace=TRACE_SAMPLES,
)


def build_geometry(*, limiter_rho):
    """Convert the imported payload to a native FciGeometry3D, opening the SOL."""

    return essos_imported_geometry_to_fci(imported, limiter_rho=limiter_rho)


open_geometry = build_geometry(limiter_rho=LIMITER_RHO)
closed_geometry = build_geometry(limiter_rho=None)  # reference: fully closed

rho_axis = np.asarray(open_geometry.grid.x.centers)
theta_axis = np.asarray(open_geometry.grid.y.centers)
bmag = np.asarray(open_geometry.cell_bfield.Bmag)
open_mask = np.asarray(open_geometry.maps.forward_boundary, dtype=bool) | np.asarray(
    open_geometry.maps.backward_boundary, dtype=bool
)
b_contra = np.asarray(open_geometry.cell_bfield.B_contra)
iota_estimate = float(np.nanmedian(b_contra[..., 1] / np.maximum(b_contra[..., 2], 1e-30)))
sol_shells = int(np.sum(rho_axis > LIMITER_RHO))
print(f"  grid (rho, theta, phi) = {open_geometry.shape}, |B| ~ {bmag.mean():.2f} T, "
      f"iota ~ {iota_estimate:.3f}")
print(f"  limiter at rho = {LIMITER_RHO}: {NX - sol_shells} closed radial shells, "
      f"{sol_shells} open SOL shells ({open_mask.mean() * 100:.0f}% of cells open on targets)")

# Shared numerical pieces: stencil builders, boundary conditions, phi solver.
parameters = Fci4FieldBlobParameters(
    rho_star=RHO_STAR,
    phi_inversion_tol=PHI_INVERSION_TOL,
    phi_inversion_maxiter=PHI_INVERSION_MAXITER,
    phi_inversion_restart=PHI_INVERSION_RESTART,
)
stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
conservative_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)


def run_turbulence(geometry, *, sheath_sink, label):
    """Advance the seeded four-field state; drain the open SOL when sheath_sink."""

    boundary_conditions = build_free_decay_boundary_conditions(geometry)
    curvature = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    projectors = build_perp_laplacian_face_projectors(geometry)
    phi_solver = build_four_field_phi_solver(
        geometry, parameters,
        conservative_stencil_builder=conservative_builder, face_projectors=projectors,
    )
    jacobian = np.asarray(geometry.cell_metric.J)

    state = multi_mode_state(geometry, amplitude=SEED_AMPLITUDE, seed=SEED, modes=SEED_MODES)
    phi_guess = jnp.zeros(geometry.shape, dtype=jnp.float64)
    content, flux = [], []

    def record(current):
        content.append(float(np.sum(np.asarray(current.density, dtype=np.float64) * jacobian)))

    record(state)
    print(f"\n[{label}] advancing {N_STEPS} RK4 steps (dt = {DT})...")
    for step in range(1, N_STEPS + 1):
        state, phi_guess = four_field_rk4_step(
            state, geometry=geometry, timestep=DT, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=phi_guess,
        )
        step_flux = 0.0
        if sheath_sink:
            state, step_flux = apply_sheath_sink(state, geometry, DT)
        record(state)
        flux.append(step_flux)
        if step % 8 == 0 or step == N_STEPS:
            energy = float(jnp.sum(jnp.asarray(state.omega) ** 2))
            print(f"  step {step:3d}  t = {step * DT:.3f}  vorticity energy = {energy:.4e}"
                  f"  sheath loss = {step_flux:.4e}")
    return state, np.asarray(content), np.asarray(flux)

# run ----------------------------------------------------------------------
open_state, open_content, open_flux = run_turbulence(open_geometry, sheath_sink=True, label="open SOL")
closed_state, closed_content, _ = run_turbulence(closed_geometry, sheath_sink=False, label="closed reference")

# physics summary ----------------------------------------------------------
open_drop = 100.0 * (open_content[0] - open_content[-1]) / open_content[0]
closed_drop = 100.0 * (closed_content[0] - closed_content[-1]) / closed_content[0]
final_vorticity = float(jnp.max(jnp.abs(jnp.asarray(open_state.omega))))
print("\nphysics summary:")
print(f"  interchange vorticity generated from the pure-density seed: max |omega| = {final_vorticity:.3e}")
print(f"  particle content change: open SOL {open_drop:+.2f}%  vs closed reference {closed_drop:+.2f}%")
print(f"  total sheath particle loss over the run: {float(np.sum(open_flux)):.4e}")
assert np.all(np.isfinite(np.asarray(open_state.density))), "open run went non-finite"
assert final_vorticity > 0.0, "no interchange vorticity was generated"
assert open_drop > closed_drop, "the open SOL did not drain faster than the closed reference"
print("  verified: the open SOL drains through the sheath while the closed core is retained")

# plot ---------------------------------------------------------------------
print("\ndrawing the closed/open turbulence figure...")
final_density = np.asarray(open_state.density, dtype=np.float64)
phi_plane = final_density.shape[2] // 2
density_slice = final_density[:, :, phi_plane] - 1.0  # fluctuation about the background

fig, (ax_field, ax_traces) = plt.subplots(1, 2, figsize=(12.0, 5.0))

theta_grid, rho_grid = np.meshgrid(theta_axis, rho_axis)
limit = float(np.max(np.abs(density_slice)))
mesh = ax_field.pcolormesh(theta_grid, rho_grid, density_slice, cmap="RdBu_r",
                           vmin=-limit, vmax=limit, shading="gouraud")
ax_field.axhline(LIMITER_RHO, color="k", linestyle="--", linewidth=1.4)
ax_field.text(theta_axis.mean(), LIMITER_RHO + 0.01, "limiter (last closed surface)",
              ha="center", va="bottom", fontsize=8)
ax_field.text(theta_axis.mean(), rho_axis.max() - 0.02, "open SOL", ha="center", va="top", fontsize=9)
ax_field.text(theta_axis.mean(), rho_axis.min() + 0.02, "closed core", ha="center", va="bottom", fontsize=9)
ax_field.set_xlabel("poloidal angle theta [rad]")
ax_field.set_ylabel("flux-surface label rho")
ax_field.set_title(f"density fluctuation at phi plane {phi_plane}")
fig.colorbar(mesh, ax=ax_field, label="n - n0")

times = np.arange(len(open_content)) * DT
ax_traces.plot(times, open_content / open_content[0], label="open SOL (sheath drain)", color="#d62728")
ax_traces.plot(times, closed_content / closed_content[0], label="closed reference", color="#1f77b4")
ax_flux = ax_traces.twinx()
ax_flux.plot(times[1:], open_flux, color="#7f7f7f", linewidth=0.9, alpha=0.7, label="sheath flux")
ax_flux.set_ylabel("sheath particle loss per step", color="#7f7f7f")
ax_traces.set_xlabel("time [normalized]")
ax_traces.set_ylabel("particle content (normalized to t = 0)")
ax_traces.set_title("SOL drains through the sheath; core is retained")
ax_traces.legend(loc="lower left", fontsize=9)
fig.suptitle("Landreman-Paul QA: four-field turbulence on closed + open field lines", fontsize=13)
fig.tight_layout()
figure_path = OUTPUT_DIR / "landreman_paul_turbulence.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print(f"wrote {figure_path}")
