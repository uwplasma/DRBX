"""Tutorial: four-field drift-reduced model (n, vorticity, V_par,i, V_par,e) on a circular tokamak flux tube.

A rotating-ellipse geometry with elongation 0 is an axisymmetric circular torus
(elongation > 0 gives a stellarator). A multi-mode density seed is advanced with
RK4; every step solves the polarization equation for the potential. Run:

    PYTHONPATH=src python examples/tutorial_four_field.py
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drbx.geometry import (
    ConservativeStencilBuilder,
    LocalStencilBuilder,
    build_conservative_stencil_from_field,
    build_curvature_coefficients,
    build_local_stencil_from_field,
    build_rotating_ellipse_geometry,
)
from drbx.native import build_perp_laplacian_face_projectors
from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters
from drbx.native.stellarator_turbulence import (
    build_four_field_phi_solver,
    build_free_decay_boundary_conditions,
    four_field_rk4_step,
    multi_mode_state,
)

# --- PARAMETERS (edit these) ---
SHAPE = (12, 16, 8)   # (radial x, poloidal theta, toroidal zeta) cells
ELONGATION = 0.0      # 0 = circular tokamak; try 0.35 for a rotating-ellipse stellarator
DT = 2.0e-3
STEPS = 100
OUTPUT = Path("output/tutorial_four_field.png")

geometry = build_rotating_ellipse_geometry(SHAPE, elongation=ELONGATION)
parameters = Fci4FieldBlobParameters(rho_star=1.0, phi_inversion_tol=5.0e-5,
                                     phi_inversion_maxiter=100, phi_inversion_restart=200)
conservative = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
projectors = build_perp_laplacian_face_projectors(geometry)
operators = dict(
    geometry=geometry, timestep=DT, parameters=parameters,
    curvature_coefficients=build_curvature_coefficients(geometry, periodic_axes=(False, True, True)),
    stencil_builder=LocalStencilBuilder(build_local_stencil_from_field.build_fn),
    conservative_stencil_builder=conservative,
    boundary_conditions=build_free_decay_boundary_conditions(geometry),
    phi_face_projectors=projectors,
    phi_inverse_solver=build_four_field_phi_solver(geometry, parameters, conservative_stencil_builder=conservative,
                                                   face_projectors=projectors))
step = jax.jit(lambda state, guess: four_field_rk4_step(state, phi_guess=guess, **operators))

state, phi = multi_mode_state(geometry, seed=1), jnp.zeros(geometry.shape)
start = time.perf_counter()
for i in range(1, STEPS + 1):
    state, phi = step(state, phi)
    if i % 20 == 0 or i == 1:
        print(f"step {i:3d}  t={i * DT:.3f}  max|vorticity|={float(jnp.max(jnp.abs(state.omega))):.3e}  "
              f"({time.perf_counter() - start:.1f}s)")

mid = SHAPE[2] // 2   # toroidal plane to draw
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
for ax, (name, field) in zip(axes, (("density", state.density), ("vorticity", state.omega), ("potential", phi))):
    mesh = ax.pcolormesh(np.asarray(field)[:, :, mid].T, shading="auto")
    fig.colorbar(mesh, ax=ax)
    ax.set(title=name, xlabel="x index", ylabel="theta index")
fig.savefig(OUTPUT, dpi=90)
print(f"wrote {OUTPUT}")
