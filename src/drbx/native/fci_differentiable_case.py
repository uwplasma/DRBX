"""Differentiable drift-reduced two-field FCI rollout on the shifted torus.

Reusable pieces behind the Phase 6 non-axisymmetric differentiability flagship
(``examples/stellarator/fci_differentiable.py``) and its gate
(``tests/test_fci_differentiable.py``):

- :func:`build_context` -- shifted-torus geometry plus the fixed-wall FCI
  operator scaffold, bundled as a :class:`DemoContext`;
- :func:`seeded_initial_state` -- smooth multi-mode seed whose fluctuation
  vanishes at the radial walls; the amplitude is the differentiation knob;
- :func:`single_rhs` -- one drift-reduced two-field FCI RHS evaluation;
- :func:`rollout` / :func:`evolved_density_variance` -- the differentiable
  multi-step RK4 rollout (via ``jax.lax.scan``) and the scalar objective;
- :func:`differentiability_report` -- ``jax.grad`` of the evolved density
  variance vs a central finite difference, with finiteness diagnostics;
- :func:`single_rhs_grad_and_fd` -- the cheaper single-RHS gradient witness.

Everything is pure JAX and ``jit``/``grad``-transparent; no files are written
here. Plotting and JSON summaries live in the example script.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import (
    RegularFaceGeometry3D,
    build_curvature_coefficients,
    build_shifted_torus_geometry,
    logical_grid_from_axis_vectors,
)
from .fci_2_field_rhs import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
    compute_2field_rhs,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
)
from .fci_time_integrator import rk4_step

__all__ = [
    "DemoContext",
    "build_context",
    "density_variance",
    "differentiability_report",
    "evolved_density_variance",
    "rollout",
    "seeded_initial_state",
    "single_rhs",
    "single_rhs_grad_and_fd",
]


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
    shape: tuple[int, int, int] = (16, 16, 8),
    *,
    sigma: float = 0.6,
    rho_star: float = 1.0,
    x_min: float = 0.15,
    x_max: float = 1.0,
    r0: float = 3.0,
    alpha_value: float = 0.25,
    iota: float = 1.1,
    c_phi: float = 3.0,
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


def seeded_initial_state(
    ctx: DemoContext,
    amp: jax.Array,
    *,
    m1: int = 2,
    n1: int = 1,
    m2: int = 3,
    n2: int = 2,
) -> Fci2FieldState:
    """Smooth seeded initial state whose fluctuation vanishes at the radial walls.

    ``density = background * exp(amp * perturbation)`` keeps the density strictly
    positive (the model derives ``phi = log(density / background)`` internally),
    and ``perturbation`` carries a radial envelope ``sin(pi * x_norm)`` so the
    fluctuation is zero at both radial boundaries, consistent with the fixed
    zero-Dirichlet fluctuation wall. ``(m1, n1)`` and ``(m2, n2)`` are the seeded
    (poloidal, toroidal) mode numbers.
    """

    logical_grid = logical_grid_from_axis_vectors(*ctx.geometry.grid.logical_axis_vectors)
    x = logical_grid[..., 0]
    theta = logical_grid[..., 1]
    zeta = logical_grid[..., 2]
    x_norm = (x - ctx.x_min) / (ctx.x_max - ctx.x_min)
    envelope = jnp.sin(jnp.pi * x_norm)
    perturbation = envelope * (
        jnp.cos(m1 * theta) * jnp.sin(n1 * zeta)
        + 0.5 * jnp.sin(m2 * theta) * jnp.cos(n2 * zeta)
    )
    background = jnp.ones(ctx.geometry.shape, dtype=jnp.float64)
    density = background * jnp.exp(amp * perturbation)
    v_parallel = amp * envelope * jnp.sin(m1 * theta) * jnp.cos(n1 * zeta)
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
    amp0: float = 0.1,
    n_steps: int = 24,
    dt: float = 1.0e-3,
    fd_step: float = 1.0e-5,
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
    amp0: float = 0.1,
    fd_step: float = 1.0e-5,
) -> dict:
    """Secondary cheaper witness: differentiate a SINGLE RHS evaluation wrt amp.

    A fast cross-check that the FCI RHS itself is differentiable, independent of
    the RK4 rollout.
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
