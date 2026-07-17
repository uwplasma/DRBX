"""B8 island divertor: magnetic topology and turbulence draining through it.

Three views of the analytic island-divertor field (sheared iota crossing the
2/3, 3/4, 4/5 rational surfaces with resonant perturbations):

1. a Poincare section -- closed core surfaces, island chains, and the
   stochastic edge, with open field lines (those that reach the wall) in red;
2. the connection-length map at the outboard midplane -- the classic
   island-divertor footprint: infinite (closed) core, finite and structured in
   the stochastic scrape-off layer; and
3. four-field turbulence on this geometry: the multi-mode seed drains through
   the *emergent* open-endpoint masks (no hand-placed limiter), shown by the
   divertor sheath flux and the particle content vs a closed reference.

The turbulence runs are written out explicitly below -- geometry, four-field
parameters, boundary conditions, phi solver, multi-mode seed, and a plain RK4
stepping loop with the Bohm sheath sink applied on the traced open endpoints.

Run:

    PYTHONPATH=src python examples/stellarator/island_divertor.py

writes ``output/island_divertor/island_divertor.png`` (release-hosted).
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
from matplotlib import colors  # noqa: E402

from drbx.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    IslandDivertorField,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_island_divertor_geometry,
    build_local_stencil_from_field,
    island_divertor_connection_length,
    island_divertor_field_line_rhs,
)
from drbx.native import build_perp_laplacian_face_projectors  # noqa: E402
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState  # noqa: E402
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402
from drbx.native.stellarator_turbulence import (  # noqa: E402
    TEMPERATURE,
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
)

# ----------------------------- PARAMETERS -----------------------------------
# The analytic island-divertor field (sheared iota + resonant perturbations);
# its dataclass fields (iota profile, perturbation amplitudes, x bounds) are
# the knobs for the magnetic topology.
FIELD = IslandDivertorField()

# Geometry / grid (arguments of build_island_divertor_geometry):
SHAPE = (16, 24, 12)     # (radial, poloidal, toroidal) cell-centered grid
MASK_MAX_TRANSITS = 25   # field lines leaving within this many transits are open

# Four-field model (Fci4FieldBlobParameters fields):
RHO_STAR = 1.0           # drift scale; sets the interchange drive strength
PHI_TOL = 5.0e-5         # GMRES tolerance of the phi (vorticity) inversion
PHI_MAXITER = 100        # GMRES max iterations per phi inversion
PHI_RESTART = 200        # GMRES restart length

# Time stepping:
DT = 2.0e-3              # RK4 timestep (normalized time units)
STEPS = 30               # number of RK4 steps per turbulence run

# Seeded multi-mode density perturbation:
SEED = 2                 # RNG seed for the random mode phases/amplitudes
AMPLITUDE = 0.08         # peak relative density perturbation
MODES = ((2, 1), (3, 2), (4, 1), (5, 3))  # (poloidal m, toroidal n) seed modes

# Poincare / connection-length diagnostics:
POINCARE_LINES = 36      # traced field lines in the Poincare section
POINCARE_TRANSITS = 200  # toroidal transits per traced line
POINCARE_STEPS = 64      # RK4 substeps per transit
CONNECTION_NX, CONNECTION_NTHETA = 90, 96  # connection-length map resolution
CONNECTION_MAX_TRANSITS = 40               # tracing cutoff for the map

OUTPUT_DIR = Path("output/island_divertor")
# ----------------------------------------------------------------------------


def poincare_section(n_lines=POINCARE_LINES, transits=POINCARE_TRANSITS, steps=POINCARE_STEPS):
    """Trace lines and collect (x, theta) at every zeta = 0 crossing."""

    rng = np.random.default_rng(0)
    starts = [(x0, rng.uniform(0, 2 * np.pi)) for x0 in np.linspace(0.3, 0.98, n_lines)]
    dz = 2.0 * np.pi / steps
    sections = []
    for x0, th0 in starts:
        x, th = x0, th0
        points = [(x, th % (2 * np.pi))]
        escaped = False
        for _ in range(transits):
            for k in range(steps):
                ze = k * dz

                def f(xv, tv, zv):
                    dx, dt = island_divertor_field_line_rhs(FIELD, xv, tv, zv)
                    return float(dx), float(dt)

                k1 = f(x, th, ze)
                k2 = f(x + dz / 2 * k1[0], th + dz / 2 * k1[1], ze + dz / 2)
                k3 = f(x + dz / 2 * k2[0], th + dz / 2 * k2[1], ze + dz / 2)
                k4 = f(x + dz * k3[0], th + dz * k3[1], ze + dz)
                x += dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                th += dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                x = max(x, FIELD.x_min)
                if x >= FIELD.x_max:
                    escaped = True
                    break
            if escaped:
                break
            points.append((x, th % (2 * np.pi)))
        sections.append((np.asarray(points), escaped))
    return sections


def connection_length_map(nx=CONNECTION_NX, ntheta=CONNECTION_NTHETA,
                          max_transits=CONNECTION_MAX_TRANSITS):
    x = jnp.linspace(FIELD.x_min + 0.01, FIELD.x_max - 0.01, nx)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta)
    xx, tt = jnp.meshgrid(x, theta, indexing="ij")
    transits, is_open = island_divertor_connection_length(
        FIELD, xx, tt, jnp.zeros_like(xx), max_transits=max_transits
    )
    return np.asarray(x), np.asarray(theta), np.asarray(transits), np.asarray(is_open)


# ------------------------- Magnetic topology views --------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[topology] tracing Poincare section...")
sections = poincare_section()
print("[topology] computing connection-length map...")
x_map, theta_map, transits, is_open = connection_length_map()

# ------------------ Turbulence: closed reference vs divertor ----------------
print("[geometry] building island-divertor geometries (closed reference + traced masks)...")
closed_geometry = build_island_divertor_geometry(SHAPE, field=FIELD)
open_geometry = build_island_divertor_geometry(SHAPE, field=FIELD, open_field_line_masks=True,
                                               mask_max_transits=MASK_MAX_TRANSITS)

runs = {}
for name, geometry, use_sheath in (("closed", closed_geometry, False),
                                   ("open", open_geometry, True)):
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
    times = [0.0]
    content = [float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian))]
    if use_sheath:
        sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
        flux = [float(sheath.total_ion_particle_loss)]
    else:
        flux = [0.0]

    # The stepping loop: classic RK4 on the four-field RHS (phi guess carried
    # between steps), then -- on the open geometry -- the explicit Bohm sheath
    # density sink on the emergent open-endpoint cells.
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

        times.append(step_index * DT)
        content.append(float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian)))
        if use_sheath:
            sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
            flux.append(float(sheath.total_ion_particle_loss))
        else:
            flux.append(0.0)

    runs[name] = (np.asarray(times), np.asarray(content), np.asarray(flux))

closed_times, closed_content, _closed_flux = runs["closed"]
open_times, open_content, open_flux = runs["open"]

# ------------------------------- Figure -------------------------------------
print("[figure] rendering the three-panel island-divertor figure...")
fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))

ax = axes[0]
for points, escaped in sections:
    color = "#d62728" if escaped else "#1f77b4"
    size = 1.2 if escaped else 0.6
    ax.scatter(points[:, 1], points[:, 0], s=size, color=color, linewidths=0)
ax.set_xlabel("poloidal angle theta"), ax.set_ylabel("radius x")
ax.set_title("Poincare section: closed core (blue), open lines (red)")
ax.set_xlim(0, 2 * np.pi), ax.set_ylim(FIELD.x_min, FIELD.x_max)

ax = axes[1]
shown = np.where(is_open, transits, np.nan)  # closed cells blank
mesh = ax.pcolormesh(theta_map, x_map, shown, cmap="magma_r",
                     norm=colors.LogNorm(vmin=1.0, vmax=40.0), shading="auto")
ax.set_facecolor("#dce6f2")
fig.colorbar(mesh, ax=ax, label="connection length (transits)")
ax.set_xlabel("poloidal angle theta"), ax.set_ylabel("radius x")
ax.set_title("Connection length (closed region shaded blue)")

ax = axes[2]
ax.plot(closed_times, closed_content / closed_content[0],
        color="#1f77b4", label="closed reference")
ax.plot(open_times, open_content / open_content[0],
        color="#d62728", label="island divertor")
ax.set_xlabel("time"), ax.set_ylabel("particle content (norm.)")
ax.set_title("Turbulence drains through the emergent divertor")
ax.legend(fontsize=8), ax.grid(True, ls=":", alpha=0.4)
twin = ax.twinx()
twin.plot(open_times, open_flux, color="#d62728", ls=":", alpha=0.7)
twin.set_ylabel("divertor sheath flux", color="#d62728")

fig.suptitle("B8 island divertor: island chains, stochastic edge, and turbulence draining to the wall")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "island_divertor.png", dpi=165)
plt.close(fig)

(OUTPUT_DIR / "summary.json").write_text(json.dumps({
    "open_fraction_edge": float(is_open[x_map > 0.9].mean()),
    "open_fraction_core": float(is_open[x_map < 0.5].mean()),
    "median_edge_connection_transits": float(np.median(transits[(x_map[:, None] > 0.9) & is_open])),
    "closed_content_change": float(closed_content[-1] - closed_content[0]),
    "open_content_change": float(open_content[-1] - open_content[0]),
    "final_divertor_flux": float(open_flux[-1]),
}, indent=2))
print(f"[done] wrote {OUTPUT_DIR / 'island_divertor.png'} and summary.json")
