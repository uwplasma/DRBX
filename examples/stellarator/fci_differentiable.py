"""Stellarator flagship: an end-to-end differentiable drift-reduced FCI model.

This is the Phase 6 non-axisymmetric flux-coordinate-independent (FCI) flagship.
It demonstrates the paper's two core claims on a *non-axisymmetric* flux-tube
geometry:

1. The reduced drift-reduced two-field FCI model (``compute_2field_rhs``) runs on
   a shifted-torus metric with genuine off-diagonal (``g12`` / ``g_12``) cross
   terms and helical field lines, staying finite over a short bounded evolution.
2. The whole rollout is end-to-end differentiable: we take ``jax.grad`` of a
   scalar diagnostic of the *evolved* state (the total density variance) with
   respect to a scalar knob (the initial perturbation amplitude), and verify it
   against a central finite difference. The gradient flows through every RK4
   stage and every FCI operator (Poisson bracket, curvature, parallel gradient).

What this is NOT: a claim of saturated stellarator turbulence. The run is a
short, bounded, seeded free evolution (no manufactured-solution source terms, no
exact-state boundary conditions). Its purpose is to show a differentiable DRB
model on non-axisymmetric FCI geometry, with honest numbers.

Differentiation path used: MULTI-STEP FREE ROLLOUT. The seeded free run stays
finite and its gradient matches the finite difference to ~1e-10 relative error,
so we differentiate through the full RK4 rollout rather than falling back to a
single-RHS evaluation. The single-RHS gradient check is still exercised as a
secondary, cheaper witness.

Geometry note (honest scope): the shifted-torus metric depends on ``(x, theta)``
and the field lines are helical (rotational transform ``iota``); a nonzero
poloidal shear ``sigma`` activates the off-diagonal metric terms. This is a
stellarator-relevant non-orthogonal flux tube, not a full 3D equilibrium. As in
the verified MMS scaffold, the field-following FCI maps are placeholders
(``construct_fci_maps=False``): the Poisson-bracket and curvature operators use
the full non-orthogonal metric (including ``g12`` / ``g_12``), and the parallel
gradient is the direct ``b^i partial_i`` operator built from the helical
contravariant field. So the differentiability shown here is of the FCI operator
stack on non-orthogonal helical geometry, not of a field-line interpolation map.

The reusable machinery (context builder, seeded state, RHS, differentiable
rollout, gradient reports) lives in ``dkx.native.fci_differentiable_case``
and is gated by ``tests/test_fci_differentiable.py``.

Run:

    PYTHONPATH=src python examples/stellarator/fci_differentiable.py

writes ``output/fci_differentiable/`` with a PNG (density slice + grad-vs-FD) and
a JSON summary. Edit the constants below to change resolution, drive, shear, or
run length.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.native.fci_differentiable_case import (  # noqa: E402
    build_context,
    differentiability_report,
    evolved_density_variance,
    seeded_initial_state,
    single_rhs,
    single_rhs_grad_and_fd,
)

# ----------------------------- PARAMETERS -----------------------------------
# Geometry (shifted-torus, non-axisymmetric flux tube):
SHAPE = (16, 16, 8)          # (radial, poloidal, toroidal) cell-centered grid
SIGMA = 0.6                  # poloidal shear -> activates off-diagonal metric terms
X_MIN = 0.15                 # minor-radius label bounds
X_MAX = 1.0
R0 = 3.0                     # torus major-radius offset
ALPHA_VALUE = 0.25           # shifted-torus shaping parameter
IOTA = 1.1                   # rotational transform of the helical field lines
C_PHI = 3.0                  # toroidal-angle scale factor of the embedding

# Model / evolution:
RHO_STAR = 1.0               # drift scale of the two-field model
AMP0 = 0.1                   # initial perturbation amplitude (the differentiation knob)
N_STEPS = 24                 # short bounded rollout (RK4 steps)
DT = 1.0e-3                  # RK4 timestep

# Finite-difference check:
FD_STEP = 1.0e-5             # central-difference step in the amplitude

# Seeded perturbation mode numbers (poloidal, toroidal):
M1, N1 = 2, 1
M2, N2 = 3, 2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "fci_differentiable"
# ----------------------------------------------------------------------------

# --------------- Geometry + fixed-wall FCI operator scaffold ----------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[setup] building shifted-torus geometry and FCI operator scaffold...")
ctx = build_context(
    SHAPE, sigma=SIGMA, rho_star=RHO_STAR, x_min=X_MIN, x_max=X_MAX,
    r0=R0, alpha_value=ALPHA_VALUE, iota=IOTA, c_phi=C_PHI,
)

# Sanity: a single RHS evaluation must be finite before we trust a rollout.
print("[setup] evaluating one two-field FCI RHS on the seeded state...")
seed_state = seeded_initial_state(ctx, AMP0, m1=M1, n1=N1, m2=M2, n2=N2)
rhs0 = single_rhs(ctx, seed_state)
rhs_finite = bool(
    np.all(np.isfinite(np.asarray(rhs0.density)))
    and np.all(np.isfinite(np.asarray(rhs0.v_parallel)))
)

# --------------- Differentiable rollout + gradient checks -------------------
print(f"[rollout] compiling and differentiating the {N_STEPS}-step RK4 rollout...")
report = differentiability_report(ctx, amp0=AMP0, n_steps=N_STEPS, dt=DT, fd_step=FD_STEP)
print("[rollout] single-RHS gradient witness...")
single = single_rhs_grad_and_fd(ctx, amp0=AMP0, fd_step=FD_STEP)

print("=" * 72)
print("Differentiable drift-reduced FCI model on a non-axisymmetric flux tube")
print("=" * 72)
print(f"geometry shape         : {tuple(int(s) for s in ctx.geometry.shape)}  (sigma={SIGMA})")
print(f"rollout                : {N_STEPS} RK4 steps, dt={DT:g}, rho_star={RHO_STAR:g}")
print(f"single-RHS finite      : {rhs_finite}")
print(f"rollout state finite   : {report['finite']}  (max|n|={report['density_max']:.4e}, max|v|={report['v_parallel_max']:.4e})")
print("-" * 72)
print("MULTI-STEP ROLLOUT  d(density variance)/d(amp):")
print(f"  jax.grad             : {report['grad']:.10e}")
print(f"  central FD (h={FD_STEP:g}) : {report['fd']:.10e}")
print(f"  relative error       : {report['rel_error']:.3e}")
print("-" * 72)
print("SINGLE-RHS witness  d(density variance)/d(amp):")
print(f"  jax.grad             : {single['grad']:.10e}")
print(f"  central FD           : {single['fd']:.10e}")
print(f"  relative error       : {single['rel_error']:.3e}")
print("=" * 72)

# ------------------------------- Summary ------------------------------------
summary = {
    "geometry_shape": [int(s) for s in ctx.geometry.shape],
    "sigma": float(SIGMA),
    "rho_star": float(RHO_STAR),
    "n_steps": int(N_STEPS),
    "dt": float(DT),
    "amp0": float(AMP0),
    "fd_step": float(FD_STEP),
    "single_rhs_finite": rhs_finite,
    "rollout_finite": report["finite"],
    "density_max": report["density_max"],
    "v_parallel_max": report["v_parallel_max"],
    "differentiation_path": "multi_step_rollout",
    "rollout_grad": report["grad"],
    "rollout_fd": report["fd"],
    "rollout_rel_error": report["rel_error"],
    "single_rhs_grad": single["grad"],
    "single_rhs_fd": single["fd"],
    "single_rhs_rel_error": single["rel_error"],
}
summary_path = OUTPUT_DIR / "fci_differentiable_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[done] wrote {summary_path}")

# ------------------------------- Figure -------------------------------------
# Left: evolved density fluctuation in one poloidal plane. Right: the objective
# J(amp) sampled around amp0 with the autodiff tangent overlaid.
print("[figure] rendering density slice and grad-vs-FD panel...")
final_state = report["final_state"]
density = np.asarray(final_state.density, dtype=np.float64)

x_values = np.asarray(ctx.geometry.grid.x.centers, dtype=np.float64)
theta_values = np.asarray(ctx.geometry.grid.y.centers, dtype=np.float64)
z_index = density.shape[2] // 2
density_slice = density[:, :, z_index]

theta_grid, radius_grid = np.meshgrid(theta_values, x_values)
vmax = float(np.max(np.abs(density_slice - 1.0))) or 1.0

fig = plt.figure(figsize=(12.0, 5.0), constrained_layout=True)
ax0 = fig.add_subplot(1, 2, 1, projection="polar")
mesh = ax0.pcolormesh(
    theta_grid,
    radius_grid,
    density_slice - 1.0,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
ax0.set_theta_zero_location("E")
ax0.set_theta_direction(-1)
ax0.set_ylim(0.0, float(x_values[-1]))
ax0.set_yticklabels([])
ax0.set_title(
    f"Evolved density fluctuation (n - n0)\n"
    f"shifted torus, sigma={SIGMA}, zeta index {z_index}, after {report['n_steps']} RK4 steps"
)
fig.colorbar(mesh, ax=ax0, shrink=0.85, pad=0.08)

# Right panel: objective J(amp) sampled by FD vs the autodiff tangent line.
objective_jit = jax.jit(partial(evolved_density_variance, ctx, n_steps=N_STEPS, dt=DT))
span = 12.0 * FD_STEP
amps = np.linspace(AMP0 - span, AMP0 + span, 7)
j_values = np.asarray([float(objective_jit(float(a))) for a in amps], dtype=np.float64)
tangent = report["objective_value"] + report["grad"] * (amps - AMP0)

ax1 = fig.add_subplot(1, 2, 2)
ax1.plot((amps - AMP0) / FD_STEP, j_values, "o", color="#1f77b4", label="J(amp) samples")
ax1.plot((amps - AMP0) / FD_STEP, tangent, "-", color="#d62728", label="autodiff tangent (jax.grad)")
ax1.axvline(0.0, color="0.7", lw=0.8, ls=":")
ax1.set_xlabel("(amp - amp0) / fd_step")
ax1.set_ylabel("density variance J")
ax1.set_title(
    "End-to-end differentiability of the FCI rollout\n"
    f"grad={report['grad']:.6e}  FD={report['fd']:.6e}  rel.err={report['rel_error']:.2e}"
)
ax1.legend(loc="best", fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.5)

figure_path = OUTPUT_DIR / "fci_differentiable.png"
fig.savefig(figure_path, dpi=200)
plt.close(fig)
print(f"[done] wrote {figure_path}")
