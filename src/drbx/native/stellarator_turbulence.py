"""Reusable four-field stellarator turbulence driver (closed and open field lines).

A thin, reusable harness around the validated four-field interchange RK4 step:
seed a multi-mode density perturbation on a stellarator FCI geometry (rotating
ellipse, island divertor, ...) and let the curvature drive develop
turbulence-like interchange dynamics. On an open geometry (endpoint masks set,
e.g. via ``limiter_radius`` or traced island-divertor masks) a Bohm sheath
density sink drains the open-endpoint cells each step, so the scrape-off layer
empties onto the target while the core stays closed.

Public pieces, each reusable on its own:

- :func:`build_free_decay_boundary_conditions` -- Dirichlet phi / Neumann
  fields at the radial walls, bundled as :class:`FourFieldBoundaryConditions`;
- :func:`build_four_field_phi_solver` -- the GMRES perpendicular-Laplacian
  inverter used by the phi inversion;
- :func:`four_field_rk4_step` -- one classic RK4 step of the four-field blob
  RHS (``compute_4field_blob_rhs``), carrying the phi guess between stages;
- :func:`multi_mode_state` -- the seeded multi-mode initial state;
- :func:`run_stellarator_turbulence` -- the full loop, returning a
  :class:`TurbulenceRun` with frames and particle/sheath-flux traces.

Used by ``tests/test_stellarator_turbulence.py`` and
``tests/test_island_divertor.py`` (fast physics gates). The example scripts in
``examples/stellarator/`` show the same anatomy written out step by step.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import (
    ConservativeStencilBuilder,
    FciGeometry3D,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
)
from .fci_4_field_rhs import (
    Fci4FieldBlobParameters,
    Fci4FieldState,
    compute_4field_blob_rhs,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BoundaryFaceBC3D,
    CutWallBC3D,
    CutWallGeometry3D,
    FourFieldBoundaryConditions,
)
from ..fci._shared import _take_stencil_finite_difference
from .fci_operators import PerpLaplacianInverseSolver, build_perp_laplacian_face_projectors
from .electromagnetic import EmPhysics, FciEmClosure, build_fci_em_closure, em_energies
from .fci_sheath_recycling import compute_fci_sheath_recycling

TEMPERATURE = 1.0  # normalized Te = Ti for the sheath sound speed
SHEATH_DENSITY_FLOOR = 1.0e-4  # positivity floor after the sheath sink

__all__ = [
    "TEMPERATURE",
    "SHEATH_DENSITY_FLOOR",
    "FourFieldBoundaryConditions",
    "TurbulenceRun",
    "advance_checked",
    "apply_sheath_sink",
    "build_free_decay_boundary_conditions",
    "StepLimit",
    "build_four_field_phi_solver",
    "four_field_rk4_step",
    "four_field_step_limit",
    "multi_mode_state",
    "radial_dirichlet_face_bc",
    "radial_neumann_face_bc",
    "run_stellarator_turbulence",
]


def radial_dirichlet_face_bc(geometry: FciGeometry3D) -> BoundaryFaceBC3D:
    """Zero-Dirichlet at the two radial (x) walls; periodic elsewhere."""

    face = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(face.x_area, dtype=jnp.int32).at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        kind_y=jnp.zeros_like(face.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(face.z_area, dtype=jnp.int32),
        value_x=jnp.zeros_like(face.x_area, dtype=jnp.float64),
        value_y=jnp.zeros_like(face.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(face.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(face.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(face.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(face.z_open_mask, dtype=bool),
    )


def radial_neumann_face_bc(geometry: FciGeometry3D) -> BoundaryFaceBC3D:
    """Zero-Neumann at the two radial (x) walls; periodic elsewhere."""

    face = RegularFaceGeometry3D.unit(geometry)
    return BoundaryFaceBC3D(
        kind_x=jnp.zeros_like(face.x_area, dtype=jnp.int32).at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        kind_y=jnp.zeros_like(face.y_area, dtype=jnp.int32),
        kind_z=jnp.zeros_like(face.z_area, dtype=jnp.int32),
        value_x=jnp.zeros_like(face.x_area, dtype=jnp.float64),
        value_y=jnp.zeros_like(face.y_area, dtype=jnp.float64),
        value_z=jnp.zeros_like(face.z_area, dtype=jnp.float64),
        mask_x=jnp.zeros_like(face.x_open_mask, dtype=bool).at[0].set(True).at[-1].set(True),
        mask_y=jnp.zeros_like(face.y_open_mask, dtype=bool),
        mask_z=jnp.zeros_like(face.z_open_mask, dtype=bool),
    )


def build_free_decay_boundary_conditions(geometry: FciGeometry3D) -> FourFieldBoundaryConditions:
    """Free-decay walls: Dirichlet phi = 0, zero-Neumann for the evolved fields."""

    neumann = radial_neumann_face_bc(geometry)
    return FourFieldBoundaryConditions(
        phi_face_bc=radial_dirichlet_face_bc(geometry),
        density_face_bc=neumann,
        omega_face_bc=neumann,
        v_ion_parallel_face_bc=neumann,
        v_electron_parallel_face_bc=neumann,
    )


def build_four_field_phi_solver(
    geometry: FciGeometry3D,
    parameters: Fci4FieldBlobParameters,
    *,
    conservative_stencil_builder: ConservativeStencilBuilder,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
) -> PerpLaplacianInverseSolver:
    """GMRES inverter for the perpendicular Laplacian used by the phi inversion."""

    return PerpLaplacianInverseSolver(
        geometry,
        conservative_stencil_builder,
        tol=float(parameters.phi_inversion_tol),
        maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart),
        face_projectors=face_projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(),
        cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True),
        pin_point=None,
        pin_value=0.0,
        project_mean_zero=False,
        target_mean_phi=None,
        regularization_epsilon=0.0,
        gmres_debug=False,
        # Hot-loop configuration: skip the per-solve host-synced residual
        # check so the phi inversion (and any step built on it) can run fully
        # inside jit. The validation harnesses build their own solvers with
        # check_residual=True.
        check_residual=False,
    )


def four_field_rk4_step(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    timestep: float,
    parameters: Fci4FieldBlobParameters,
    curvature_coefficients: jnp.ndarray,
    stencil_builder: LocalStencilBuilder,
    conservative_stencil_builder: ConservativeStencilBuilder,
    boundary_conditions: FourFieldBoundaryConditions,
    phi_face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    phi_inverse_solver: PerpLaplacianInverseSolver,
    phi_guess: jnp.ndarray | None = None,
    electromagnetic: FciEmClosure | None = None,
) -> tuple[Fci4FieldState, jnp.ndarray]:
    """One classic RK4 step of the four-field blob RHS.

    The electrostatic potential from each stage seeds the GMRES phi inversion of
    the next, so passing the returned phi back in as ``phi_guess`` keeps the
    inversion warm across steps. Returns ``(next_state, phi_of_last_stage)``.
    """

    common_kwargs = dict(
        geometry=geometry,
        stencil_builder=stencil_builder,
        conservative_stencil_builder=conservative_stencil_builder,
        parameters=parameters,
        curvature_coefficients=curvature_coefficients,
        boundary_conditions=boundary_conditions,
        phi_face_projectors=phi_face_projectors,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=False,
        return_phi=True,
        electromagnetic=electromagnetic,
    )

    rhs_1, _timings, phi_1 = compute_4field_blob_rhs(state, phi_guess=phi_guess, **common_kwargs)
    k1 = rhs_1.rhs
    stage_1 = state.axpy(k1, scale=0.5 * timestep)

    rhs_2, _timings, phi_2 = compute_4field_blob_rhs(stage_1, phi_guess=phi_1, **common_kwargs)
    k2 = rhs_2.rhs
    stage_2 = state.axpy(k2, scale=0.5 * timestep)

    rhs_3, _timings, phi_3 = compute_4field_blob_rhs(stage_2, phi_guess=phi_2, **common_kwargs)
    k3 = rhs_3.rhs
    stage_3 = state.axpy(k3, scale=timestep)

    rhs_4, _timings, phi_4 = compute_4field_blob_rhs(stage_3, phi_guess=phi_3, **common_kwargs)
    k4 = rhs_4.rhs

    increment = Fci4FieldState(
        density=(k1.density + 2.0 * k2.density + 2.0 * k3.density + k4.density) / 6.0,
        omega=(k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega) / 6.0,
        v_ion_parallel=(
            k1.v_ion_parallel + 2.0 * k2.v_ion_parallel + 2.0 * k3.v_ion_parallel + k4.v_ion_parallel
        )
        / 6.0,
        v_electron_parallel=(
            k1.v_electron_parallel
            + 2.0 * k2.v_electron_parallel
            + 2.0 * k3.v_electron_parallel
            + k4.v_electron_parallel
        )
        / 6.0,
    )
    return state.axpy(increment, scale=timestep), phi_4


class StepLimit(NamedTuple):
    """Largest stable classic-RK4 step of the four-field model, overall and per mechanism."""

    dt: jnp.ndarray
    parallel_electron_wave: jnp.ndarray
    exb_advection: jnp.ndarray
    diffusion: jnp.ndarray


RK4_IMAGINARY_AXIS = 2.0 * np.sqrt(2.0)  # RK4 stability interval on the imaginary axis
RK4_REAL_AXIS = 2.785                    # and on the negative real axis


def four_field_step_limit(
    state: Fci4FieldState,
    phi: jnp.ndarray,
    *,
    geometry: FciGeometry3D,
    parameters: Fci4FieldBlobParameters,
    stencil_builder: LocalStencilBuilder,
    boundary_conditions: FourFieldBoundaryConditions,
) -> StepLimit:
    """Explicit RK4 step limit from bounds on the spectral radius of each term.

    With ``k_i = sum |stencil weights|`` along axis ``i`` (``1/dx_i`` for a
    central difference) the rates are: parallel electron wave
    ``sqrt(mi/me Te) sum_i |b^i| k_i``, E x B advection ``sum_i |v_E^i| k_i``
    and perpendicular diffusion ``4 D_max sum_i g^ii k_i^2``. Waves and
    advection are imaginary, diffusion real, so
    ``dt = 1 / ((wave + exb) / 2.83 + diffusion / 2.79)``. The wave bound is
    the one that limits the usual runs: it scales as ``dz / sqrt(mi/me)``.
    """

    stencil = stencil_builder(phi, geometry, periodic_axes=(False, True, True),
                              face_bc=boundary_conditions.phi_face_bc)
    axes = (stencil.x, stencil.y, stencil.z)
    k = jnp.stack([jnp.abs(a.derivative_minus_weight) + jnp.abs(a.derivative_center_weight)
                   + jnp.abs(a.derivative_plus_weight) for a in axes], axis=-1)
    dphi = jnp.stack([_take_stencil_finite_difference(a) for a in axes], axis=-1)
    bmag = jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64)
    b_unit = jnp.asarray(geometry.cell_bfield.B_contra, dtype=jnp.float64) / bmag[..., None]
    b_covariant = jnp.einsum("...ij,...j->...i", geometry.cell_metric.g_cov, b_unit)
    metric = geometry.cell_metric
    v_exb = jnp.cross(b_covariant, dphi) / (metric.J * parameters.rho_star * bmag)[..., None]
    g_contra = jnp.stack([metric.g11, metric.g22, metric.g33], axis=-1)
    diffusivity = max(parameters.density_perp_diffusion, parameters.omega_perp_diffusion,
                      parameters.v_ion_parallel_perp_diffusion, parameters.v_electron_parallel_perp_diffusion)
    wave = jnp.sqrt(parameters.mi_over_me * parameters.Te) * jnp.max(jnp.sum(jnp.abs(b_unit) * k, axis=-1))
    exb = jnp.max(jnp.sum(jnp.abs(v_exb) * k, axis=-1))
    diffusion = 4.0 * diffusivity * jnp.max(jnp.sum(g_contra * k * k, axis=-1))
    tiny = jnp.finfo(jnp.float64).tiny
    return StepLimit(
        dt=1.0 / ((wave + exb) / RK4_IMAGINARY_AXIS + diffusion / RK4_REAL_AXIS),
        parallel_electron_wave=RK4_IMAGINARY_AXIS / jnp.maximum(wave, tiny),
        exb_advection=RK4_IMAGINARY_AXIS / jnp.maximum(exb, tiny),
        diffusion=RK4_REAL_AXIS / jnp.maximum(diffusion, tiny),
    )


def apply_sheath_sink(
    state: Fci4FieldState,
    geometry: FciGeometry3D,
    dt: float,
    *,
    temperature: float = TEMPERATURE,
) -> tuple[Fci4FieldState, jnp.ndarray]:
    """Bohm sheath density sink on the open-endpoint cells (explicit Euler).

    Returns the drained state and the total ion particle loss rate of the
    *pre-sink* state (the instantaneous sheath flux to the target).
    """

    temperature_field = jnp.full(geometry.shape, float(temperature))
    sheath = compute_fci_sheath_recycling(state.density, temperature_field, temperature_field, geometry.maps)
    drained = Fci4FieldState(
        density=jnp.maximum(state.density - dt * sheath.ion_particle_loss, SHEATH_DENSITY_FLOOR),
        omega=state.omega,
        v_ion_parallel=state.v_ion_parallel,
        v_electron_parallel=state.v_electron_parallel,
    )
    return drained, sheath.total_ion_particle_loss


def _state_is_valid(state: Fci4FieldState, phi: jnp.ndarray) -> jnp.ndarray:
    fields = [value for _name, value in state.field_items()] + [phi]
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in fields]))
    return finite & (jnp.min(state.density) > 0.0)


def advance_checked(
    step: Callable[[Fci4FieldState, jnp.ndarray], tuple[Fci4FieldState, jnp.ndarray]],
    state: Fci4FieldState,
    phi: jnp.ndarray,
    steps: jnp.ndarray | int,
) -> tuple[Fci4FieldState, jnp.ndarray, jnp.ndarray]:
    """Take up to ``steps`` steps of ``step(state, phi) -> (state, phi)`` in one ``while_loop``.

    The loop stops at the first step whose result has a non-finite field or a
    non-positive density and returns the last valid state, its potential and
    the number of steps completed. ``steps`` may be traced, so one compiled
    chunk serves every chunk length. Jit it and compare the count with
    ``steps`` on the host to detect a failure.
    """

    def cond(carry):
        done, _state, _phi, ok = carry
        return ok & (done < steps)

    def body(carry):
        done, current, current_phi, _ok = carry
        candidate, candidate_phi = step(current, current_phi)
        ok = _state_is_valid(candidate, candidate_phi)
        keep = lambda new, old: jnp.where(ok, new, old)  # noqa: E731
        return (done + ok.astype(done.dtype), jax.tree_util.tree_map(keep, candidate, current),
                keep(candidate_phi, current_phi), ok)

    done, state, phi, _ok = jax.lax.while_loop(cond, body, (jnp.asarray(0), state, phi, jnp.asarray(True)))
    return state, phi, done


def _summary(state: Fci4FieldState) -> str:
    return (f"min n = {float(jnp.min(state.density)):.3e}, max|omega| = {float(jnp.max(jnp.abs(state.omega))):.3e}, "
            f"max|V_e| = {float(jnp.max(jnp.abs(state.v_electron_parallel))):.3e}")


def _invalid_step_message(last_valid: Fci4FieldState, failed: Fci4FieldState, failed_phi: jnp.ndarray,
                          step_index: int, time_value: float, dt: float) -> str:
    fields = [*failed.field_items(), ("phi", failed_phi)]
    bad = [name for name, field in fields if not bool(jnp.all(jnp.isfinite(field)))]
    reason = f"non-finite {', '.join(bad)}" if bad else f"min density {float(jnp.min(failed.density)):.3e} <= 0"
    return (f"four-field run stopped at step {step_index} (t = {time_value:.6g}, dt = {dt:.3g}): {reason}. "
            f"Last valid step: {_summary(last_valid)}. Reduce dt.")


def multi_mode_state(
    geometry: FciGeometry3D,
    *,
    amplitude: float = 0.08,
    seed: int = 0,
    modes: tuple[tuple[int, int], ...] = ((2, 1), (3, 2), (4, 1), (5, 3)),
) -> Fci4FieldState:
    """Seed a random-phase multi-mode density perturbation with a radial envelope."""

    rng = np.random.default_rng(seed)
    x = np.asarray(geometry.grid.x.centers)[:, None, None]
    theta = np.asarray(geometry.grid.y.centers)[None, :, None]
    zeta = np.asarray(geometry.grid.z.centers)[None, None, :]
    x_norm = (x - x.min()) / (x.max() - x.min())
    envelope = np.sin(np.pi * x_norm)
    perturbation = np.zeros(geometry.shape)
    for m, n in modes:
        perturbation += rng.uniform(0.5, 1.0) * np.cos(m * theta + n * zeta + rng.uniform(0, 2 * np.pi))
    density = 1.0 + amplitude * envelope * perturbation
    zeros = jnp.zeros(geometry.shape, dtype=jnp.float64)
    return Fci4FieldState(density=jnp.asarray(density), omega=zeros,
                          v_ion_parallel=zeros, v_electron_parallel=zeros)


class TurbulenceRun(NamedTuple):
    """Density/vorticity frames plus particle-content and sheath-flux traces."""

    density_frames: np.ndarray      # (n_frames, nx, ny, nz), float32
    omega_frames: np.ndarray
    times: np.ndarray
    particle_content: np.ndarray    # Jacobian-weighted total density per frame
    target_flux: np.ndarray         # total sheath ion loss per frame (0 when closed)
    apar_frames: np.ndarray | None = None      # electromagnetic runs: A_par per frame
    energies: np.ndarray | None = None         # electromagnetic runs: (ExB, magnetic, electron, ion) per frame


def run_stellarator_turbulence(geometry: FciGeometry3D, *, steps: int, dt: float,
                               amplitude: float = 0.08, seed: int = 0,
                               sheath_sink: bool = False, frame_stride: int = 1,
                               physics: EmPhysics = EmPhysics()) -> TurbulenceRun:
    """Advance the seeded four-field state; optionally drain open endpoints.

    ``dt`` is the largest step: before every chunk of ``frame_stride`` steps it
    is capped at :func:`four_field_step_limit` of the current state (a printed
    line reports each change), so ``times`` holds the actual frame times.
    ``physics.electromagnetic`` adds A_par, magnetic flutter and electron inertia (the evolved
    ``v_electron_parallel`` is then the canonical momentum); the default leaves the electrostatic run unchanged.
    """

    parameters = Fci4FieldBlobParameters(rho_star=1.0, mi_over_me=physics.mi_over_me, phi_inversion_tol=5.0e-5,
                                         phi_inversion_maxiter=100, phi_inversion_restart=200)
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_conditions = build_free_decay_boundary_conditions(geometry)
    curvature = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    projectors = build_perp_laplacian_face_projectors(geometry)
    phi_solver = build_four_field_phi_solver(
        geometry, parameters,
        conservative_stencil_builder=conservative_builder, face_projectors=projectors,
    )

    closure = build_fci_em_closure(geometry, physics, conservative_stencil_builder=conservative_builder,
                                   face_projectors=projectors)
    temperature = jnp.full(geometry.shape, TEMPERATURE)
    jacobian = np.asarray(geometry.cell_metric.J)
    state = multi_mode_state(geometry, amplitude=amplitude, seed=seed)
    phi_guess = jnp.zeros(geometry.shape, dtype=jnp.float64)
    density_frames, omega_frames, times, content, flux = [], [], [], [], []

    def step(current, guess, step_dt):
        # The whole RK4 step (four RHS evaluations, each with a GMRES phi
        # inversion) and the sheath sink are sync-free.
        current, guess = four_field_rk4_step(
            current, geometry=geometry, timestep=step_dt, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=guess, electromagnetic=closure,
        )
        return (apply_sheath_sink(current, geometry, step_dt)[0], guess) if sheath_sink else (current, guess)

    # One compiled chunk of up to frame_stride steps at a traced dt; between
    # chunks the host checks the step count, caps dt at the explicit limit,
    # records a frame and prints progress.
    chunk = jax.jit(lambda current, guess, count, step_dt: advance_checked(
        lambda s, p: step(s, p, step_dt), current, guess, count))
    step_limit = jax.jit(lambda current, guess: four_field_step_limit(
        current, guess, geometry=geometry, parameters=parameters, stencil_builder=stencil_builder,
        boundary_conditions=boundary_conditions))
    apar_frames, energies = [], []

    def record(time_value, current):
        density = np.asarray(current.density, dtype=np.float32)
        density_frames.append(density)
        omega_frames.append(np.asarray(current.omega, dtype=np.float32))
        times.append(time_value)
        content.append(float(np.sum(np.asarray(current.density, dtype=np.float64) * jacobian)))
        if sheath_sink:
            sheath = compute_fci_sheath_recycling(current.density, temperature, temperature, geometry.maps)
            flux.append(float(sheath.total_ion_particle_loss))
        else:
            flux.append(0.0)
        if closure is not None:
            apar = closure.solve_apar(current.density, current.v_ion_parallel, current.v_electron_parallel,
                                      face_bc=boundary_conditions.v_electron_parallel_face_bc)
            phi = phi_solver(-current.omega, face_bc=boundary_conditions.phi_face_bc)
            apar_frames.append(np.asarray(apar, dtype=np.float32))
            energies.append([float(e) for e in em_energies(closure, current, phi, apar, jacobian)])

    record(0.0, state)
    start, last_print, step_index, time_value, step_dt = time.perf_counter(), 0.0, 0, 0.0, dt
    while step_index < steps:
        count = min(frame_stride, steps - step_index)
        limit = step_limit(state, phi_guess)
        capped = min(dt, float(limit.dt))
        if capped != step_dt:
            print(f"[turbulence] step {step_index}: dt = {capped:.4g} (requested {dt:.4g}; RK4 limits: parallel "
                  f"electron wave {float(limit.parallel_electron_wave):.3g}, ExB {float(limit.exb_advection):.3g}, "
                  f"diffusion {float(limit.diffusion):.3g})", flush=True)
            step_dt = capped
        state_next, phi_next, done = chunk(state, phi_guess, count, step_dt)
        if int(done) < count:
            failed, failed_phi = jax.jit(step)(state_next, phi_next, step_dt)
            raise FloatingPointError(_invalid_step_message(state_next, failed, failed_phi, step_index + int(done) + 1,
                                                           time_value + (int(done) + 1) * step_dt, step_dt))
        state, phi_guess = state_next, phi_next
        step_index, time_value = step_index + count, time_value + count * step_dt
        record(time_value, state)
        elapsed = time.perf_counter() - start
        if step_index in (count, steps) or elapsed - last_print >= 10.0:
            print(f"[turbulence] step {step_index}/{steps} t = {time_value:.4g} {_summary(state)} "
                  f"({elapsed:.1f} s)", flush=True)
            last_print = elapsed

    return TurbulenceRun(np.stack(density_frames), np.stack(omega_frames),
                         np.asarray(times), np.asarray(content), np.asarray(flux),
                         np.stack(apar_frames) if closure is not None else None,
                         np.asarray(energies) if closure is not None else None)
