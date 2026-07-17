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

from dataclasses import dataclass, field
from typing import NamedTuple

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
)
from .fci_operators import PerpLaplacianInverseSolver, build_perp_laplacian_face_projectors
from .fci_sheath_recycling import compute_fci_sheath_recycling

TEMPERATURE = 1.0  # normalized Te = Ti for the sheath sound speed
SHEATH_DENSITY_FLOOR = 1.0e-4  # positivity floor after the sheath sink

__all__ = [
    "TEMPERATURE",
    "SHEATH_DENSITY_FLOOR",
    "FourFieldBoundaryConditions",
    "TurbulenceRun",
    "apply_sheath_sink",
    "build_free_decay_boundary_conditions",
    "build_four_field_phi_solver",
    "four_field_rk4_step",
    "multi_mode_state",
    "radial_dirichlet_face_bc",
    "radial_neumann_face_bc",
    "run_stellarator_turbulence",
]


@dataclass(frozen=True)
class FourFieldBoundaryConditions:
    """Per-field face BCs (plus optional cut-wall data) for the four-field model."""

    phi_face_bc: BoundaryFaceBC3D
    density_face_bc: BoundaryFaceBC3D
    omega_face_bc: BoundaryFaceBC3D
    v_ion_parallel_face_bc: BoundaryFaceBC3D
    v_electron_parallel_face_bc: BoundaryFaceBC3D
    phi_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    phi_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    density_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    density_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    omega_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    omega_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    v_ion_parallel_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    v_ion_parallel_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)
    v_electron_parallel_cut_wall_geometry: CutWallGeometry3D = field(default_factory=CutWallGeometry3D.empty)
    v_electron_parallel_cut_wall_bc: CutWallBC3D = field(default_factory=CutWallBC3D.empty)


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
        phi_face_bc=boundary_conditions.phi_face_bc,
        density_face_bc=boundary_conditions.density_face_bc,
        omega_face_bc=boundary_conditions.omega_face_bc,
        v_ion_parallel_face_bc=boundary_conditions.v_ion_parallel_face_bc,
        v_electron_parallel_face_bc=boundary_conditions.v_electron_parallel_face_bc,
        phi_cut_wall_geometry=boundary_conditions.phi_cut_wall_geometry,
        phi_cut_wall_bc=boundary_conditions.phi_cut_wall_bc,
        density_cut_wall_geometry=boundary_conditions.density_cut_wall_geometry,
        density_cut_wall_bc=boundary_conditions.density_cut_wall_bc,
        omega_cut_wall_geometry=boundary_conditions.omega_cut_wall_geometry,
        omega_cut_wall_bc=boundary_conditions.omega_cut_wall_bc,
        v_ion_parallel_cut_wall_geometry=boundary_conditions.v_ion_parallel_cut_wall_geometry,
        v_ion_parallel_cut_wall_bc=boundary_conditions.v_ion_parallel_cut_wall_bc,
        v_electron_parallel_cut_wall_geometry=boundary_conditions.v_electron_parallel_cut_wall_geometry,
        v_electron_parallel_cut_wall_bc=boundary_conditions.v_electron_parallel_cut_wall_bc,
        phi_face_projectors=phi_face_projectors,
        phi_inverse_solver=phi_inverse_solver,
        gmres_debug=False,
        return_phi=True,
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


def apply_sheath_sink(
    state: Fci4FieldState,
    geometry: FciGeometry3D,
    dt: float,
    *,
    temperature: float = TEMPERATURE,
) -> tuple[Fci4FieldState, float]:
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
    return drained, float(sheath.total_ion_particle_loss)


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


def run_stellarator_turbulence(geometry: FciGeometry3D, *, steps: int, dt: float,
                               amplitude: float = 0.08, seed: int = 0,
                               sheath_sink: bool = False, frame_stride: int = 1) -> TurbulenceRun:
    """Advance the seeded four-field state; optionally drain open endpoints."""

    parameters = Fci4FieldBlobParameters(rho_star=1.0, phi_inversion_tol=5.0e-5,
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

    temperature = jnp.full(geometry.shape, TEMPERATURE)
    jacobian = np.asarray(geometry.cell_metric.J)
    state = multi_mode_state(geometry, amplitude=amplitude, seed=seed)
    phi_guess = jnp.zeros(geometry.shape, dtype=jnp.float64)
    density_frames, omega_frames, times, content, flux = [], [], [], [], []

    # The whole RK4 step (four RHS evaluations, each with a GMRES phi
    # inversion) is sync-free, so compile it once and run it as a single
    # XLA program.
    jitted_step = jax.jit(
        lambda current, guess: four_field_rk4_step(
            current, geometry=geometry, timestep=dt, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=guess,
        )
    )

    def record(step_index, current):
        density = np.asarray(current.density, dtype=np.float32)
        density_frames.append(density)
        omega_frames.append(np.asarray(current.omega, dtype=np.float32))
        times.append(step_index * dt)
        content.append(float(np.sum(np.asarray(current.density, dtype=np.float64) * jacobian)))
        if sheath_sink:
            sheath = compute_fci_sheath_recycling(current.density, temperature, temperature, geometry.maps)
            flux.append(float(sheath.total_ion_particle_loss))
        else:
            flux.append(0.0)

    record(0, state)
    for step_index in range(1, steps + 1):
        state, phi_guess = jitted_step(state, phi_guess)
        if sheath_sink:
            state, _loss = apply_sheath_sink(state, geometry, dt)
        if step_index % frame_stride == 0 or step_index == steps:
            record(step_index, state)

    return TurbulenceRun(np.stack(density_frames), np.stack(omega_frames),
                         np.asarray(times), np.asarray(content), np.asarray(flux))
