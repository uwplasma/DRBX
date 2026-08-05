"""Source-driven profile evolution across an internal island chain (tokamak).

A tokamak-like configuration: the rotational transform decreases from
``iota = 0.56`` on the inner boundary to ``0.44`` on the outer one, crossing
the rational ``iota = 1/2`` in the middle of the domain. A single ``(m, n) =
(2, 1)`` resonant perturbation opens an internal island chain there -- the
reduced analytic field of :class:`drbx.geometry.IslandDivertorField` with one
resonance and ``n_field_periods = 1``. All field lines stay confined (the
open-field-line mask comes out empty); this is an *internal* island, not a
divertor.

The four-field drift-reduced Braginskii model then evolves the profiles
flux-driven:

* a Gaussian-in-radius particle **source shell** near the inner boundary is
  the only drive (its amplitude sets the throughput),
* a smooth **wall buffer** in the outermost few percent of the radius is the
  sink (a limiter/wall proxy),
* no Dirichlet clamping anywhere -- the density profile is emergent and
  reaches a steady source -> wall balance at production resolution
  (48x96x32, fp32, ~1 h on a 16 GB GPU).

The classic island observable -- mean-profile flattening across the chain
-- appears once cross-island transport is turbulence-dominated: at the
default weak drive the state is laminar and the profile does NOT flatten
(gradient ratio inside/outside ~1.3), while ``S0 = 24, D = 0.005,
HYPER = 0.3`` gives a turbulent state (fluctuation energy ~2) whose profile
flattens across the separatrix (ratio ~0.8). The parallel-flow friction
``mu`` (``DRBX_ISLAND_MU``) stays at 8 in both regimes; lowering it far
(0.5) under-damps the drift-acoustic channel and the run goes unstable.

The whole step -- the RK4 advance of ``four_field_rk4_step`` plus source,
wall sink, and hyperviscosity -- is compiled as ONE XLA program, which is
what makes long profile-evolution runs practical (the same pattern as
:func:`drbx.native.stellarator_turbulence.run_stellarator_turbulence`; a raw
Python composition costs ~50x more in dispatch overhead).

Run it:

    # laptop smoke test (~1 minute)
    DRBX_ISLAND_SHAPE=16,32,12 DRBX_ISLAND_STEPS=400 \
      python examples/island_tokamak_profiles.py

    # GPU production (fp32; fp64 is 1/64-rate on consumer GPUs)
    DRBX_PRECISION=float32 DRBX_ISLAND_SHAPE=48,96,32 \
      DRBX_ISLAND_DT=2.5e-4 DRBX_ISLAND_STEPS=40000 \
      python examples/island_tokamak_profiles.py

Writes ``output/island_tokamak/island_tokamak.npz`` (profile history, flux
profiles, 2-D snapshots, particle-balance trace) and prints the balance
table; ``examples/island_tokamak_figure.py`` renders the summary figure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# CONFIG ----------------------------------------------------------------------
SHAPE = tuple(int(v) for v in os.environ.get("DRBX_ISLAND_SHAPE", "28,56,20").split(","))
DT = float(os.environ.get("DRBX_ISLAND_DT", "5e-4"))
N_STEPS = int(os.environ.get("DRBX_ISLAND_STEPS", "8000"))
FRAME_STRIDE = int(os.environ.get("DRBX_ISLAND_STRIDE", "100"))
SPINUP = int(os.environ.get("DRBX_ISLAND_SPINUP", str(N_STEPS // 2)))

# The tokamak-like field: iota crosses 1/2 mid-domain; one (2,1) resonance
# opens the internal island chain. EPS sets the island width.
IOTA_AXIS, IOTA_EDGE = 0.56, 0.44
RESONANCE_M, RESONANCE_N = 2, 1
EPS = float(os.environ.get("DRBX_ISLAND_EPS", "0.012"))
R0, ELONGATION = 3.0, 0.35

# Flux-driven knobs: source shell near the inner boundary, wall buffer at the
# outer one. S0 (the particle throughput) is the only physical drive.
SRC_X0, SRC_W = 0.12, 0.07
SRC_S0 = float(os.environ.get("DRBX_ISLAND_S0", "1.5"))
WALL_X0, NU_WALL = 0.92, 60.0
DENS_DIFF = float(os.environ.get("DRBX_ISLAND_D", "0.04"))
OMEGA_DIFF = 0.15
# Parallel-flow friction: heavy damping suppresses the parallel
# equilibration that flattens density along the island flux surfaces --
# use small values when the island response is the observable.
FRICTION_MU = float(os.environ.get("DRBX_ISLAND_MU", "8.0"))
NU_HYPER = float(os.environ.get("DRBX_ISLAND_HYPER", "1.5"))
N_FLOOR = 0.05
SEED, AMP = 7, 0.05
MODES = ((2, 1), (3, 1), (4, 2), (5, 2))

LABEL = os.environ.get("DRBX_ISLAND_LABEL", "island_tokamak")
OUT = Path("output/island_tokamak")
OUT.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------

from drbx.geometry import (  # noqa: E402
    ConservativeStencilBuilder, IslandDivertorField, LocalStencilBuilder,
    build_conservative_stencil_from_field, build_curvature_coefficients,
    build_island_divertor_geometry, build_local_stencil_from_field)
from drbx.native import build_perp_laplacian_face_projectors  # noqa: E402
from drbx.native.fci_4_field_rhs import (  # noqa: E402
    Fci4FieldBlobParameters, Fci4FieldState)
from drbx.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402
from drbx.native.stellarator_turbulence import (  # noqa: E402
    TEMPERATURE, build_four_field_phi_solver,
    build_free_decay_boundary_conditions, four_field_rk4_step)

field = IslandDivertorField(iota_axis=IOTA_AXIS, iota_edge=IOTA_EDGE,
                            resonances=((RESONANCE_M, RESONANCE_N, EPS),))
print(f"building tokamak + (m,n)=({RESONANCE_M},{RESONANCE_N}) island geometry {SHAPE} ...")
geom = build_island_divertor_geometry(
    SHAPE, field=field, r0=R0, elongation=ELONGATION, n_field_periods=1,
    open_field_line_masks=True, mask_max_transits=25)
open_frac = float(np.asarray(geom.maps.forward_boundary, dtype=bool).mean())
print(f"  open-field-line fraction: {100 * open_frac:.1f}% "
      "(inner region confined; outermost lines may intersect the wall, "
      "forming a limiter-like SOL)")

x = np.asarray(geom.grid.x.centers)
xn = (x - x.min()) / (x.max() - x.min())
DX = float(x[1] - x[0])
# radial location of the island chain: iota(x) = n/m
x_res = float(np.interp(RESONANCE_N / RESONANCE_M,
                        np.linspace(IOTA_AXIS, IOTA_EDGE, len(xn))[::-1], xn[::-1]))
print(f"  island chain at xn = {x_res:.2f} (iota = {RESONANCE_N}/{RESONANCE_M})")

DTHETA = float(np.asarray(geom.grid.y.centers)[1] - np.asarray(geom.grid.y.centers)[0])


@jax.jit
def lap_perp(f):
    fx = jnp.zeros_like(f)
    fx = fx.at[1:-1].set(f[2:] - 2 * f[1:-1] + f[:-2])
    fx = fx.at[0].set(f[1] - f[0]); fx = fx.at[-1].set(f[-2] - f[-1])
    fy = jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)
    return fx + fy


@jax.jit
def hyper4(f):
    return lap_perp(lap_perp(f))


def nonaxi(a):
    return a - a.mean(axis=(1, 2), keepdims=True)


src_x = SRC_S0 * np.exp(-0.5 * ((xn - SRC_X0) / SRC_W) ** 2)
source = jnp.asarray(src_x[:, None, None]) * jnp.ones(geom.shape)
src_total = float(np.sum(np.asarray(source)))
wall_x = NU_WALL * np.clip((xn - WALL_X0) / (1.0 - WALL_X0), 0.0, 1.0) ** 2
wall_nu = jnp.asarray(wall_x[:, None, None])

bcs = build_free_decay_boundary_conditions(geom)
par = Fci4FieldBlobParameters(rho_star=1.0, phi_inversion_tol=5e-5, phi_inversion_maxiter=120,
                              phi_inversion_restart=200, density_perp_diffusion=DENS_DIFF,
                              omega_perp_diffusion=OMEGA_DIFF)
stencil = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
conservative = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
curvature = build_curvature_coefficients(geom, periodic_axes=(False, True, True))
projectors = build_perp_laplacian_face_projectors(geom)
phi_solver = build_four_field_phi_solver(geom, par, conservative_stencil_builder=conservative,
                                         face_projectors=projectors)

rng = np.random.default_rng(SEED)
theta = np.asarray(geom.grid.y.centers)[None, :, None]
zeta = np.asarray(geom.grid.z.centers)[None, None, :]
pert = np.zeros(geom.shape)
for m, n in MODES:
    pert += rng.uniform(0.5, 1.0) * np.cos(m * theta + n * zeta + rng.uniform(0, 2 * np.pi))
env = np.sin(np.pi * xn)[:, None, None]
state = Fci4FieldState(density=jnp.asarray(0.3 + AMP * env * pert),
                       omega=jnp.zeros(geom.shape),
                       v_ion_parallel=jnp.zeros(geom.shape),
                       v_electron_parallel=jnp.zeros(geom.shape))
temperature = jnp.full(geom.shape, TEMPERATURE)
phi_guess = jnp.zeros(geom.shape)


@jax.jit
def fused_step(current, guess):
    """One XLA program per step: RK4 advance + source + sinks + filters."""
    nxt, phi = four_field_rk4_step(
        current, geometry=geom, timestep=DT, parameters=par,
        curvature_coefficients=curvature, stencil_builder=stencil,
        conservative_stencil_builder=conservative, boundary_conditions=bcs,
        phi_face_projectors=projectors, phi_inverse_solver=phi_solver,
        phi_guess=guess)
    sheath = compute_fci_sheath_recycling(nxt.density, temperature, temperature, geom.maps)
    wall_loss = wall_nu * jnp.maximum(nxt.density - N_FLOOR, 0.0)
    dens = (nxt.density
            + DT * source
            - DT * sheath.ion_particle_loss
            - DT * wall_loss
            - DT * NU_HYPER * hyper4(nonaxi(nxt.density)))
    om = nxt.omega * (1.0 - DT * FRICTION_MU) - DT * NU_HYPER * hyper4(nxt.omega)
    vpi = nxt.v_ion_parallel * (1.0 - DT * FRICTION_MU) - DT * NU_HYPER * hyper4(nxt.v_ion_parallel)
    vpe = nxt.v_electron_parallel * (1.0 - DT * FRICTION_MU) - DT * NU_HYPER * hyper4(nxt.v_electron_parallel)
    out = Fci4FieldState(density=jnp.maximum(dens, N_FLOOR), omega=om,
                         v_ion_parallel=vpi, v_electron_parallel=vpe)
    return out, phi, sheath.total_ion_particle_loss, jnp.sum(wall_loss)


frames = {k: [] for k in ("profile", "energy", "flux_profile", "wall_loss",
                          "sheath_loss", "times", "slice_density", "slice_phi")}
snap = {}
print(f"stepping {N_STEPS} flux-driven steps (dt={DT}, D_perp={DENS_DIFF}, "
      f"island eps={EPS}) ...")
for step in range(1, N_STEPS + 1):
    state, phi_guess, sheath_total, wall_total = fused_step(state, phi_guess)
    if step % FRAME_STRIDE == 0:
        d = np.asarray(state.density, dtype=np.float32)
        p = np.asarray(phi_guess, dtype=np.float32)
        vr = -(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * DTHETA)
        nt = d - d.mean(axis=(1, 2), keepdims=True)
        frames["profile"].append(d.mean(axis=(1, 2)))
        # four toroidal cross-sections (zeta = 0, pi/2, pi, 3pi/2) for the
        # 3-D rendering of the evolving state
        zi = [0, d.shape[2] // 4, d.shape[2] // 2, 3 * d.shape[2] // 4]
        frames["slice_density"].append(d[:, :, zi])
        frames["slice_phi"].append(p[:, :, zi])
        frames["flux_profile"].append(np.mean(nt * vr, axis=(1, 2)).astype(np.float32))
        frames["energy"].append(float(np.mean(nonaxi(d) ** 2)))
        frames["sheath_loss"].append(float(sheath_total))
        frames["wall_loss"].append(float(wall_total))
        frames["times"].append(step * DT)
    if step == N_STEPS:                       # final 2-D snapshots for the figure
        snap["density"] = np.asarray(state.density, dtype=np.float32)
        snap["phi"] = np.asarray(phi_guess, dtype=np.float32)
    if step % max(500, FRAME_STRIDE) == 0:
        sink = float(sheath_total + wall_total) / src_total
        print(f"  step {step}/{N_STEPS}  E_fluct={frames['energy'][-1]:.3f}  "
              f"sinks/source = {sink:.2f}", flush=True)

n_win = max(1, (N_STEPS - SPINUP) // FRAME_STRIDE)
prof = np.stack(frames["profile"])
prof_bar = prof[-n_win:].mean(axis=0)
# island-flattening metric: mean |d<n>/dx| inside vs outside the island band
band = np.abs(xn - x_res) < 0.08
outside = (np.abs(xn - x_res) > 0.15) & (xn > SRC_X0 + 2 * SRC_W) & (xn < WALL_X0)
grad = np.abs(np.gradient(prof_bar, DX))
flattening = float(grad[band].mean() / max(grad[outside].mean(), 1e-30))
balance = {
    "sinks_over_source": float(np.mean(np.asarray(frames["sheath_loss"][-n_win:])
                                       + np.asarray(frames["wall_loss"][-n_win:])) / src_total),
    "island_flattening_ratio": flattening,
    "x_res": x_res, "eps": EPS, "S0": SRC_S0, "dens_diff": DENS_DIFF,
    "shape": list(SHAPE), "dt": DT, "n_steps": N_STEPS, "open_fraction": open_frac,
}
print(f"balance: sinks/source = {balance['sinks_over_source']:.2f}")
print(f"island flattening: |grad n| inside / outside = {flattening:.2f} "
      f"(< 1 means the chain flattens the profile)")

np.savez_compressed(OUT / f"{LABEL}.npz",
                    **{k: np.asarray(v) for k, v in frames.items()},
                    **snap, x=x, xn=xn)
(OUT / f"{LABEL}_balance.json").write_text(json.dumps(balance, indent=1))
print(f"wrote {OUT}/{LABEL}.npz and {LABEL}_balance.json")
