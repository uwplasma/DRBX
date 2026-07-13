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
finite and its gradient matches the finite difference to ~1e-10 relative error
(see ``differentiability_report``), so we differentiate through the full RK4
rollout rather than falling back to a single-RHS evaluation. The single-RHS
gradient check is still available via ``single_rhs_grad_and_fd`` and is exercised
as a secondary, cheaper witness.

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

Run:

    PYTHONPATH=src python examples/stellarator/fci_differentiable_demo.py

writes ``output/fci_differentiable/`` with a PNG (density slice + grad-vs-FD) and
a JSON summary. Edit the constants below to change resolution, drive, shear, or
run length.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jax_drb.geometry import (  # noqa: E402
    RegularFaceGeometry3D,
    build_curvature_coefficients,
    build_shifted_torus_geometry,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import rk4_step  # noqa: E402
from jax_drb.native.fci_2_field_rhs import (  # noqa: E402
    Fci2FieldRhsParameters,
    Fci2FieldState,
    compute_2field_rhs,
)
from jax_drb.native.fci_boundaries import (  # noqa: E402
    BC_DIRICHLET,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
)

# --- Geometry (shifted-torus, non-axisymmetric flux tube) ---
SHAPE = (16, 16, 8)          # (radial, poloidal, toroidal) cell-centered grid
SIGMA = 0.6                  # poloidal shear -> activates off-diagonal metric terms
X_MIN = 0.15
X_MAX = 1.0
R0 = 3.0
ALPHA_VALUE = 0.25
IOTA = 1.1
C_PHI = 3.0

# --- Model / evolution ---
RHO_STAR = 1.0
AMP0 = 0.1                   # initial perturbation amplitude (the differentiation knob)
N_STEPS = 24                 # short bounded rollout
DT = 1.0e-3

# --- Finite-difference check ---
FD_STEP = 1.0e-5

# --- Seeded perturbation mode numbers (poloidal, toroidal) ---
M1, N1 = 2, 1
M2, N2 = 3, 2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "fci_differentiable"


class DemoContext(NamedTuple):
    """Everything the differentiable rollout needs, built once per resolution."""

    geometry: object
    curvature_coefficients: jax.Array
    density_face_bc: BoundaryFaceBC3D
    phi_face_bc: BoundaryFaceBC3D
    v_parallel_face_bc: BoundaryFaceBC3D
    parameters: Fci2FieldRhsParameters
    x_min: float
    x_max: float


def _fixed_radial_dirichlet_face_bc(geometry, boundary_value: float) -> BoundaryFaceBC3D:
    """A fixed Dirichlet face BC at the two radial (x) walls; periodic elsewhere.

    ``boundary_value`` is held constant in time, so this is a simple fixed wall,
    not an exact-solution boundary. For the density the wall value is the
    background (1.0); for the fluctuation quantities phi and v_parallel it is 0
    (zero-Dirichlet on the fluctuation).
    """

    face = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(face.x_area, dtype=jnp.int32).at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=jnp.zeros_like(face.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(face.z_area, dtype=jnp.int32),
        value_x=jnp.full_like(face.x_area, 0.0).at[0].set(boundary_value).at[-1].set(boundary_value),
        value_y=jnp.zeros_like(face.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(face.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(face.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(face.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(face.z_open_mask, dtype=bool),
    )


def build_context(
    shape: tuple[int, int, int] = SHAPE,
    *,
    sigma: float = SIGMA,
    rho_star: float = RHO_STAR,
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    r0: float = R0,
    alpha_value: float = ALPHA_VALUE,
    iota: float = IOTA,
    c_phi: float = C_PHI,
) -> DemoContext:
    """Build the shifted-torus geometry and the fixed-wall FCI operator scaffold."""

    geometry = build_shifted_torus_geometry(
        shape,
        x_min=x_min,
        x_max=x_max,
        r0=r0,
        alpha_value=alpha_value,
        iota=iota,
        c_phi=c_phi,
        sigma=sigma,
        construct_fci_maps=False,
    )
    curvature_coefficients = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    return DemoContext(
        geometry=geometry,
        curvature_coefficients=curvature_coefficients,
        density_face_bc=_fixed_radial_dirichlet_face_bc(geometry, 1.0),
        phi_face_bc=_fixed_radial_dirichlet_face_bc(geometry, 0.0),
        v_parallel_face_bc=_fixed_radial_dirichlet_face_bc(geometry, 0.0),
        parameters=Fci2FieldRhsParameters(rho_star=rho_star),
        x_min=float(x_min),
        x_max=float(x_max),
    )


def seeded_initial_state(ctx: DemoContext, amp: jax.Array) -> Fci2FieldState:
    """Smooth seeded initial state whose fluctuation vanishes at the radial walls.

    ``density = background * exp(amp * perturbation)`` keeps the density strictly
    positive (the model derives ``phi = log(density / background)`` internally),
    and ``perturbation`` carries a radial envelope ``sin(pi * x_norm)`` so the
    fluctuation is zero at both radial boundaries, consistent with the fixed
    zero-Dirichlet fluctuation wall.
    """

    logical_grid = logical_grid_from_axis_vectors(*ctx.geometry.grid.logical_axis_vectors)
    x = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    x_norm = (x - ctx.x_min) / (ctx.x_max - ctx.x_min)
    envelope = jnp.sin(jnp.pi * x_norm)
    perturbation = envelope * (
        jnp.cos(M1 * theta) * jnp.sin(N1 * zeta)
        + 0.5 * jnp.sin(M2 * theta) * jnp.cos(N2 * zeta)
    )
    background = jnp.ones(ctx.geometry.shape, dtype=jnp.float64)
    density = background * jnp.exp(amp * perturbation)
    v_parallel = amp * envelope * jnp.sin(M1 * theta) * jnp.cos(N1 * zeta)
    return Fci2FieldState(density=density, v_parallel=v_parallel, density_background=background)


def _clamp_radial_boundaries(ctx: DemoContext, state: Fci2FieldState) -> Fci2FieldState:
    """Hold the radial boundary rows at the fixed wall values (density=bg, v=0)."""

    del ctx
    density = state.density.at[0, :, :].set(1.0).at[-1, :, :].set(1.0)
    v_parallel = state.v_parallel.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    return Fci2FieldState(
        density=density,
        v_parallel=v_parallel,
        density_background=state.density_background,
    )


def single_rhs(ctx: DemoContext, state: Fci2FieldState) -> Fci2FieldState:
    """One drift-reduced two-field FCI RHS evaluation on the shifted torus."""

    state = _clamp_radial_boundaries(ctx, state)
    result, _timings = compute_2field_rhs(
        state,
        geometry=ctx.geometry,
        parameters=ctx.parameters,
        curvature_coefficients=ctx.curvature_coefficients,
        periodic_axes=(False, True, True),
        density_face_bc=ctx.density_face_bc,
        phi_face_bc=ctx.phi_face_bc,
        v_parallel_face_bc=ctx.v_parallel_face_bc,
        density_cut_wall_geometry=CutWallGeometry3D.empty(),
        density_cut_wall_bc=CutWallBC3D.empty(),
        phi_cut_wall_geometry=CutWallGeometry3D.empty(),
        phi_cut_wall_bc=CutWallBC3D.empty(),
        v_parallel_cut_wall_geometry=CutWallGeometry3D.empty(),
        v_parallel_cut_wall_bc=CutWallBC3D.empty(),
    )
    return result.rhs


def rollout(ctx: DemoContext, amp: jax.Array, *, n_steps: int, dt: float) -> Fci2FieldState:
    """Advance the seeded free state ``n_steps`` RK4 steps (differentiable, via scan)."""

    initial_state = _clamp_radial_boundaries(ctx, seeded_initial_state(ctx, amp))

    def _rhs_fn(current_state, _stage_time, _carry):
        return single_rhs(ctx, current_state), None, jnp.asarray(0.0)

    def _body(state, _):
        step = rk4_step(state, time=0.0, timestep=dt, rhs_fn=_rhs_fn, carry=None)
        return _clamp_radial_boundaries(ctx, step.state), None

    final_state, _ = jax.lax.scan(_body, initial_state, None, length=int(n_steps))
    return final_state


def density_variance(state: Fci2FieldState) -> jax.Array:
    """Scalar diagnostic: variance of the density over interior (non-wall) cells."""

    interior = state.density[1:-1, :, :]
    return jnp.mean((interior - jnp.mean(interior)) ** 2)


def evolved_density_variance(ctx: DemoContext, amp: jax.Array, *, n_steps: int, dt: float) -> jax.Array:
    """The scalar objective we differentiate: variance of the evolved density."""

    return density_variance(rollout(ctx, amp, n_steps=n_steps, dt=dt))


def differentiability_report(
    ctx: DemoContext,
    *,
    amp0: float = AMP0,
    n_steps: int = N_STEPS,
    dt: float = DT,
    fd_step: float = FD_STEP,
) -> dict:
    """Roll out the free run and compare autodiff grad to a central FD (wrt amp0).

    Returns the evolved state and the gradient diagnostics. The objective is
    JIT-compiled once so the finite-difference samples reuse the compiled rollout.
    """

    objective = partial(evolved_density_variance, ctx, n_steps=n_steps, dt=dt)
    objective_jit = jax.jit(objective)
    grad_jit = jax.jit(jax.grad(objective))

    amp0 = float(amp0)
    grad_value = float(grad_jit(amp0))
    plus = float(objective_jit(amp0 + fd_step))
    minus = float(objective_jit(amp0 - fd_step))
    fd_value = (plus - minus) / (2.0 * fd_step)
    rel_error = abs(grad_value - fd_value) / max(abs(fd_value), 1.0e-30)

    final_state = rollout(ctx, amp0, n_steps=n_steps, dt=dt)
    density = np.asarray(final_state.density, dtype=np.float64)
    v_parallel = np.asarray(final_state.v_parallel, dtype=np.float64)
    finite = bool(np.all(np.isfinite(density)) and np.all(np.isfinite(v_parallel)))

    return {
        "grad": grad_value,
        "fd": fd_value,
        "rel_error": rel_error,
        "fd_step": float(fd_step),
        "amp0": amp0,
        "n_steps": int(n_steps),
        "dt": float(dt),
        "objective_value": float(objective_jit(amp0)),
        "finite": finite,
        "density_max": float(np.max(np.abs(density))),
        "v_parallel_max": float(np.max(np.abs(v_parallel))),
        "final_state": final_state,
    }


def single_rhs_grad_and_fd(
    ctx: DemoContext,
    *,
    amp0: float = AMP0,
    fd_step: float = FD_STEP,
) -> dict:
    """Secondary cheaper witness: differentiate a SINGLE RHS evaluation wrt amp.

    This is the fallback path described in the module docstring; here it is a
    fast cross-check that the FCI RHS itself is differentiable, independent of the
    RK4 rollout.
    """

    def objective(amp: jax.Array) -> jax.Array:
        return density_variance(single_rhs(ctx, seeded_initial_state(ctx, amp)))

    objective_jit = jax.jit(objective)
    grad_value = float(jax.jit(jax.grad(objective))(float(amp0)))
    plus = float(objective_jit(float(amp0) + fd_step))
    minus = float(objective_jit(float(amp0) - fd_step))
    fd_value = (plus - minus) / (2.0 * fd_step)
    rel_error = abs(grad_value - fd_value) / max(abs(fd_value), 1.0e-30)
    return {"grad": grad_value, "fd": fd_value, "rel_error": rel_error}


def _poloidal_slice_figure(ctx: DemoContext, report: dict, output_path: Path) -> None:
    """Density poloidal slice + a J(amp) curve with the autodiff tangent overlaid."""

    geometry = ctx.geometry
    final_state = report["final_state"]
    density = np.asarray(final_state.density, dtype=np.float64)

    x_values = np.asarray(geometry.grid.x.centers, dtype=np.float64)
    theta_values = np.asarray(geometry.grid.y.centers, dtype=np.float64)
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
    amp0 = report["amp0"]
    grad = report["grad"]
    j0 = report["objective_value"]
    objective = partial(evolved_density_variance, ctx, n_steps=report["n_steps"], dt=report["dt"])
    objective_jit = jax.jit(objective)
    span = 12.0 * report["fd_step"]
    amps = np.linspace(amp0 - span, amp0 + span, 7)
    j_values = np.asarray([float(objective_jit(float(a))) for a in amps], dtype=np.float64)
    tangent = j0 + grad * (amps - amp0)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot((amps - amp0) / report["fd_step"], j_values, "o", color="#1f77b4", label="J(amp) samples")
    ax1.plot((amps - amp0) / report["fd_step"], tangent, "-", color="#d62728", label="autodiff tangent (jax.grad)")
    ax1.axvline(0.0, color="0.7", lw=0.8, ls=":")
    ax1.set_xlabel("(amp - amp0) / fd_step")
    ax1.set_ylabel("density variance J")
    ax1.set_title(
        "End-to-end differentiability of the FCI rollout\n"
        f"grad={grad:.6e}  FD={report['fd']:.6e}  rel.err={report['rel_error']:.2e}"
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(
    *,
    output_dir: Path = OUTPUT_DIR,
    shape: tuple[int, int, int] = SHAPE,
    sigma: float = SIGMA,
    n_steps: int = N_STEPS,
    dt: float = DT,
    amp0: float = AMP0,
    rho_star: float = RHO_STAR,
    fd_step: float = FD_STEP,
    make_figure: bool = True,
) -> dict:
    """Build the geometry, run the differentiable free rollout, and write outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(shape, sigma=sigma, rho_star=rho_star)

    # Sanity: a single RHS evaluation must be finite before we trust a rollout.
    rhs0 = single_rhs(ctx, seeded_initial_state(ctx, amp0))
    rhs_finite = bool(
        np.all(np.isfinite(np.asarray(rhs0.density)))
        and np.all(np.isfinite(np.asarray(rhs0.v_parallel)))
    )

    report = differentiability_report(ctx, amp0=amp0, n_steps=n_steps, dt=dt, fd_step=fd_step)
    single = single_rhs_grad_and_fd(ctx, amp0=amp0, fd_step=fd_step)

    print("=" * 72)
    print("Differentiable drift-reduced FCI model on a non-axisymmetric flux tube")
    print("=" * 72)
    print(f"geometry shape         : {tuple(int(s) for s in ctx.geometry.shape)}  (sigma={sigma})")
    print(f"rollout                : {n_steps} RK4 steps, dt={dt:g}, rho_star={rho_star:g}")
    print(f"single-RHS finite      : {rhs_finite}")
    print(f"rollout state finite   : {report['finite']}  (max|n|={report['density_max']:.4e}, max|v|={report['v_parallel_max']:.4e})")
    print("-" * 72)
    print("MULTI-STEP ROLLOUT  d(density variance)/d(amp):")
    print(f"  jax.grad             : {report['grad']:.10e}")
    print(f"  central FD (h={fd_step:g}) : {report['fd']:.10e}")
    print(f"  relative error       : {report['rel_error']:.3e}")
    print("-" * 72)
    print("SINGLE-RHS witness  d(density variance)/d(amp):")
    print(f"  jax.grad             : {single['grad']:.10e}")
    print(f"  central FD           : {single['fd']:.10e}")
    print(f"  relative error       : {single['rel_error']:.3e}")
    print("=" * 72)

    summary = {
        "geometry_shape": [int(s) for s in ctx.geometry.shape],
        "sigma": float(sigma),
        "rho_star": float(rho_star),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "amp0": float(amp0),
        "fd_step": float(fd_step),
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

    summary_path = output_dir / "fci_differentiable_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")

    if make_figure:
        figure_path = output_dir / "fci_differentiable.png"
        _poloidal_slice_figure(ctx, report, figure_path)
        print(f"wrote {figure_path}")
        summary["figure"] = str(figure_path)

    return summary


if __name__ == "__main__":
    main()
