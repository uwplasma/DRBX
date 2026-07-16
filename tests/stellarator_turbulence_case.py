"""Shared driver for stellarator turbulence on closed and open field lines.

A thin, reusable loop around the validated four-field interchange RK4 step:
seed a multi-mode density perturbation on a rotating-ellipse flux tube and let
the curvature drive develop turbulence-like interchange dynamics. On an open
geometry (``limiter_radius`` set) a Bohm sheath density sink acts on the
open-endpoint cells each step, so the scrape-off layer drains to the limiter
while the core stays closed.

Used by ``tests/test_stellarator_turbulence.py`` (fast physics gate) and
``examples/stellarator/stellarator_turbulence_demo.py`` (movies). The physics
lives in ``jax_drb.native``; this module only wires it together.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_drb.geometry import FciGeometry3D, build_curvature_coefficients  # noqa: E402
from jax_drb.native.fci_4_field_rhs import Fci4FieldBlobParameters, Fci4FieldState  # noqa: E402
from jax_drb.native.fci_sheath_recycling import compute_fci_sheath_recycling  # noqa: E402

from test_shifted_torus_4_field_blob import shifted_torus_4field_blob_rk4  # noqa: E402
from test_shifted_torus_4_field_free_decay import _build_free_decay_boundary_conditions  # noqa: E402
from jax_drb.geometry import (  # noqa: E402
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from jax_drb.native import build_perp_laplacian_face_projectors  # noqa: E402
from jax_drb.native.fci_boundaries import CutWallBC3D, CutWallGeometry3D  # noqa: E402
from jax_drb.native.fci_operators import PerpLaplacianInverseSolver  # noqa: E402

TEMPERATURE = 1.0  # normalized Te = Ti for the sheath sound speed


def multi_mode_state(geometry: FciGeometry3D, *, amplitude: float = 0.08, seed: int = 0,
                     modes: tuple[tuple[int, int], ...] = ((2, 1), (3, 2), (4, 1), (5, 3))) -> Fci4FieldState:
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
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    curvature = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    projectors = build_perp_laplacian_face_projectors(geometry)
    phi_solver = PerpLaplacianInverseSolver(
        geometry, conservative_builder,
        tol=float(parameters.phi_inversion_tol), maxiter=int(parameters.phi_inversion_maxiter),
        restart=int(parameters.phi_inversion_restart), face_projectors=projectors,
        regular_face_geometry=RegularFaceGeometry3D.unit(geometry),
        cut_wall_geometry=CutWallGeometry3D.empty(), cut_wall_bc=CutWallBC3D.empty(),
        periodic_axes=(False, True, True), pin_point=None, pin_value=0.0,
        project_mean_zero=False, target_mean_phi=None, regularization_epsilon=0.0,
        gmres_debug=False,
    )

    temperature = jnp.full(geometry.shape, TEMPERATURE)
    jacobian = np.asarray(geometry.cell_metric.J)
    state = multi_mode_state(geometry, amplitude=amplitude, seed=seed)
    phi_guess = None
    density_frames, omega_frames, times, content, flux = [], [], [], [], []

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
        state, _timings, phi_guess = shifted_torus_4field_blob_rk4(
            state, geometry=geometry, timestep=dt, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=phi_guess,
        )
        if sheath_sink:
            sheath = compute_fci_sheath_recycling(state.density, temperature, temperature, geometry.maps)
            state = Fci4FieldState(
                density=jnp.maximum(state.density - dt * sheath.ion_particle_loss, 1.0e-4),
                omega=state.omega, v_ion_parallel=state.v_ion_parallel,
                v_electron_parallel=state.v_electron_parallel,
            )
        if step_index % frame_stride == 0 or step_index == steps:
            record(step_index, state)

    return TurbulenceRun(np.stack(density_frames), np.stack(omega_frames),
                         np.asarray(times), np.asarray(content), np.asarray(flux))
