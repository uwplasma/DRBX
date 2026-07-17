"""Gate for four-field turbulence on the imported Landreman-Paul geometry.

Skips cleanly when ESSOS is unavailable. Otherwise it exercises the same public
API path as ``examples/stellarator/landreman_paul_turbulence.py`` at tiny size:
the vmec-source imported geometry converts to a native FciGeometry3D with a
limiter-defined open SOL, the four-field step stays finite and generates
interchange vorticity, and the Bohm sheath sink drains the open SOL cells only.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from drbx.geometry import essos_runtime_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not essos_runtime_available(), reason="ESSOS runtime not importable"
)

LIMITER_RHO = 0.85


@pytest.fixture(scope="module")
def imported_payload():
    from drbx.geometry import build_essos_imported_fci_geometry

    return build_essos_imported_fci_geometry(
        map_source="vmec", nx=8, ny=8, nz=12,
        rho_min=0.12, rho_max=1.15, maxtime=400.0, times_to_trace=1024,
    )


def test_limiter_opens_only_the_scrape_off_layer(imported_payload) -> None:
    from drbx.geometry import essos_imported_geometry_to_fci

    geometry = essos_imported_geometry_to_fci(imported_payload, limiter_rho=LIMITER_RHO)
    rho = np.asarray(geometry.grid.x.centers)
    forward = np.asarray(geometry.maps.forward_boundary, dtype=bool)
    backward = np.asarray(geometry.maps.backward_boundary, dtype=bool)
    sol = rho > LIMITER_RHO

    # Open endpoints appear exactly on the SOL shells at the target planes.
    assert np.array_equal(forward[:, :, -1], np.broadcast_to(sol[:, None], forward[:, :, -1].shape))
    assert np.array_equal(backward[:, :, 0], np.broadcast_to(sol[:, None], backward[:, :, 0].shape))
    # Interior toroidal planes stay closed, and the core has no open cells.
    assert not forward[:, :, :-1].any()
    assert not backward[:, :, 1:].any()
    assert not (forward | backward)[~sol].any()

    # Fully closed when no limiter is set (clean surface-preserving vmec maps).
    closed = essos_imported_geometry_to_fci(imported_payload, limiter_rho=None)
    assert not np.asarray(closed.maps.forward_boundary, dtype=bool).any()
    assert not np.asarray(closed.maps.backward_boundary, dtype=bool).any()


def test_iota_recovers_the_landreman_paul_value(imported_payload) -> None:
    from drbx.geometry import essos_imported_geometry_to_fci

    geometry = essos_imported_geometry_to_fci(imported_payload, limiter_rho=None)
    b_contra = np.asarray(geometry.cell_bfield.B_contra)
    iota = float(np.nanmedian(b_contra[..., 1] / np.maximum(b_contra[..., 2], 1e-30)))
    # The Landreman-Paul QA rotational transform is ~0.42.
    assert 0.35 < iota < 0.50


def test_turbulence_runs_and_drains_the_sol(imported_payload) -> None:
    from drbx.geometry import (
        ConservativeStencilBuilder,
        LocalStencilBuilder,
        build_conservative_stencil_from_field,
        build_curvature_coefficients,
        build_local_stencil_from_field,
        essos_imported_geometry_to_fci,
    )
    from drbx.native import build_perp_laplacian_face_projectors
    from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters
    from drbx.native.stellarator_turbulence import (
        apply_sheath_sink,
        build_four_field_phi_solver,
        build_free_decay_boundary_conditions,
        four_field_rk4_step,
        multi_mode_state,
    )

    geometry = essos_imported_geometry_to_fci(imported_payload, limiter_rho=LIMITER_RHO)
    parameters = Fci4FieldBlobParameters(
        rho_star=1.0, phi_inversion_tol=5.0e-5,
        phi_inversion_maxiter=100, phi_inversion_restart=200,
    )
    stencil_builder = LocalStencilBuilder(build_local_stencil_from_field.build_fn)
    conservative_builder = ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn)
    boundary_conditions = build_free_decay_boundary_conditions(geometry)
    curvature = build_curvature_coefficients(geometry, periodic_axes=(False, True, True))
    projectors = build_perp_laplacian_face_projectors(geometry)
    phi_solver = build_four_field_phi_solver(
        geometry, parameters,
        conservative_stencil_builder=conservative_builder, face_projectors=projectors,
    )

    state = multi_mode_state(geometry, amplitude=0.08, seed=1)
    phi_guess = jnp.zeros(geometry.shape, dtype=jnp.float64)
    jacobian = np.asarray(geometry.cell_metric.J)
    content0 = float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian))
    total_flux = 0.0
    for _ in range(3):
        state, phi_guess = four_field_rk4_step(
            state, geometry=geometry, timestep=1.5e-3, parameters=parameters,
            curvature_coefficients=curvature, stencil_builder=stencil_builder,
            conservative_stencil_builder=conservative_builder,
            boundary_conditions=boundary_conditions, phi_face_projectors=projectors,
            phi_inverse_solver=phi_solver, phi_guess=phi_guess,
        )
        state, step_flux = apply_sheath_sink(state, geometry, 1.5e-3)
        total_flux += step_flux

    assert np.all(np.isfinite(np.asarray(state.density)))
    assert float(state.density.min()) > 0.0
    # Interchange vorticity generated from the pure-density seed.
    assert float(jnp.max(jnp.abs(state.omega))) > 0.0
    # The sheath removed particles from the open SOL.
    assert total_flux > 0.0
    content1 = float(np.sum(np.asarray(state.density, dtype=np.float64) * jacobian))
    assert content1 < content0
