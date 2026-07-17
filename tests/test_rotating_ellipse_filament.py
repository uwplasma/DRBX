"""Seeded-filament dynamics on the rotating ellipse (B7, second half).

Seeds a localized density blob on the genuinely non-axisymmetric rotating-ellipse
geometry and evolves the four-field drift-reduced FCI model (density, vorticity,
ion/electron parallel velocity) as a short free run. The gate checks that the
interchange machinery actually runs on this geometry: the run stays finite, the
density stays positive, and vorticity is generated from the seeded pressure blob
by the curvature drive (the mechanism that makes a filament move). This is also
the first pytest coverage of the four-field blob lane, whose conservative
perpendicular-diffusion assembly is exercised here.

The driver (boundary conditions, phi inversion, RK4 stepping) is reused from the
shifted-torus four-field harness; only the geometry is swapped.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dkx.geometry import build_rotating_ellipse_geometry
from dkx.native.fci_4_field_rhs import Fci4FieldBlobParameters

# The four-field blob driver lives in sibling harness modules; make sure the test
# directory is importable regardless of collection order.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_shifted_torus_4_field_blob import (  # noqa: E402
    _build_blob_initial_state,
    simulate_shifted_torus_4field_blob,
)
from test_shifted_torus_4_field_free_decay import _build_free_decay_boundary_conditions  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _rotating_ellipse_geometry():
    return build_rotating_ellipse_geometry(
        (16, 16, 8),
        x_min=0.2,
        x_max=1.0,
        elongation=0.35,
        n_field_periods=1,
        iota=0.9,
        c_phi=3.0,
    )


def test_four_field_filament_runs_and_drives_interchange() -> None:
    geometry = _rotating_ellipse_geometry()
    initial_state = _build_blob_initial_state(geometry)
    boundary_conditions = _build_free_decay_boundary_conditions(geometry, 0.0)
    parameters = Fci4FieldBlobParameters(
        rho_star=1.0,
        phi_inversion_tol=5.0e-5,
        phi_inversion_maxiter=100,
        phi_inversion_restart=200,
    )

    # The seed carries no vorticity and no parallel flow; the model must generate
    # them from the density blob.
    assert float(jnp.max(jnp.abs(initial_state.omega))) == 0.0
    assert float(jnp.max(jnp.abs(initial_state.v_ion_parallel))) == 0.0

    n_steps = 10
    dt = 2.0e-3
    final_state, *_history = simulate_shifted_torus_4field_blob(
        geometry,
        initial_state,
        boundary_conditions,
        parameters=parameters,
        final_time=n_steps * dt,
        timestep=dt,
    )

    density = np.asarray(final_state.density, dtype=np.float64)
    omega = np.asarray(final_state.omega, dtype=np.float64)
    v_ion = np.asarray(final_state.v_ion_parallel, dtype=np.float64)

    # Runs finite on the genuinely non-axisymmetric geometry.
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(omega))
    assert np.all(np.isfinite(v_ion))

    # Density stays positive and bounded (no numerical blow-up over the window).
    assert float(density.min()) > 0.0
    initial_peak = float(np.asarray(initial_state.density).max())
    assert float(density.max()) < 3.0 * initial_peak

    # The curvature drive generated vorticity from the pressure blob and spun up a
    # parallel ion flow through the four-field coupling: the interchange mechanism
    # that moves a filament is active on this geometry.
    assert float(np.max(np.abs(omega))) > 1.0e-4
    assert float(np.max(np.abs(v_ion))) > 1.0e-6
