"""Cost of the electromagnetic closure: time per RK4 step of the four-field model, electrostatic and electromagnetic.

Run:  PYTHONPATH=src python examples/electromagnetic/step_time.py
Prints milliseconds per step (median and minimum of REPEATS jitted steps after a warm-up step) for each grid in SHAPES.
"""

import statistics
import time

import jax
import jax.numpy as jnp

from drbx.geometry import (
    ConservativeStencilBuilder, LocalStencilBuilder, build_conservative_stencil_from_field,
    build_curvature_coefficients, build_local_stencil_from_field, build_shifted_torus_geometry,
)
from drbx.native.electromagnetic import EmPhysics, build_fci_em_closure
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters
from drbx.native.fci_operators import build_perp_laplacian_face_projectors
from drbx.native.stellarator_turbulence import (
    build_four_field_phi_solver, build_free_decay_boundary_conditions, four_field_rk4_step, multi_mode_state,
)

jax.config.update("jax_enable_x64", True)

# ----------------------------- PARAMETERS -----------------------------------
SHAPES = [(8, 12, 6), (16, 24, 12)]  # (radial, poloidal, toroidal) cells
BETA = 0.1                           # electromagnetic runs only
DT = 1.0e-3
REPEATS = 16
# ----------------------------------------------------------------------------

for shape in SHAPES:
    geometry = build_shifted_torus_geometry(shape)
    parameters = Fci4FieldBlobParameters(phi_inversion_tol=5.0e-5, phi_inversion_maxiter=100, phi_inversion_restart=200)
    local = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    projectors = build_perp_laplacian_face_projectors(geometry)
    for electromagnetic in (False, True):
        closure = build_fci_em_closure(geometry, EmPhysics(electromagnetic=electromagnetic, beta=BETA),
                                       conservative_stencil_builder=conservative, face_projectors=projectors)
        step = jax.jit(lambda state, guess: four_field_rk4_step(
            state, geometry=geometry, timestep=DT, parameters=parameters,
            curvature_coefficients=build_curvature_coefficients(geometry, periodic_axes=(False, True, True)),
            stencil_builder=local, conservative_stencil_builder=conservative,
            boundary_conditions=build_free_decay_boundary_conditions(geometry), phi_face_projectors=projectors,
            phi_inverse_solver=build_four_field_phi_solver(geometry, parameters, conservative_stencil_builder=conservative,
                                                           face_projectors=projectors),
            phi_guess=guess, electromagnetic=closure))
        state, guess, times = multi_mode_state(geometry), jnp.zeros(shape), []
        state, guess = jax.block_until_ready(step(state, guess))
        for _ in range(REPEATS):
            start = time.perf_counter()
            state, guess = jax.block_until_ready(step(state, guess))
            times.append(time.perf_counter() - start)
        print(f"{jax.default_backend()} {shape} {'electromagnetic' if electromagnetic else 'electrostatic  '}: "
              f"median {1e3 * statistics.median(times):8.1f} ms/step, min {1e3 * min(times):8.1f}")
