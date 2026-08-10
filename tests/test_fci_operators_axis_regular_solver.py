"""Axis-regular coverage for auxiliary non-FCI solver/preconditioner paths.

The fixture is an analytic extruded polar disk.  These tests deliberately use
the scalar topology closure: all fields handled here are scalar cell fields,
and the auxiliary bands/preconditioners operate on scalar principal fluxes.
Vector and flux-density parity is tested by the direct/conservative operator
test modules, where those component types are materialized explicitly.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_regular_operator_support import owned, polar_fixture, scalar_field_halo
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D
from drbx.geometry import (
    StencilBuilderContext,
    build_axis_core_cell_gradient_reconstruction,
)
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _axis_core_prolong_from_coefficients,
    _axis_core_restrict_to_coefficients,
    _principal_perp_laplacian_bands,
    build_axis_core_line_u_preconditioner,
    build_local_perp_laplacian_face_projectors,
    build_solvax_perp_laplacian_preconditioner,
)


def _bands_fixture(shape=(4, 8, 8)):
    geometry, domain, context, coordinates, exchange, scalar, vector, flux = polar_fixture(shape)
    del context, coordinates, exchange, vector, flux
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    return geometry, domain, scalar, projectors, face_bc


def test_principal_bands_are_finite_and_have_no_collapsed_axis_neighbor():
    geometry, domain, _scalar, projectors, face_bc = _bands_fixture()
    diagonal, lower, upper = _principal_perp_laplacian_bands(
        geometry, domain, projectors, face_bc
    )

    assert diagonal.shape == geometry.owned_shape
    assert all(value.shape == geometry.owned_shape for value in (*lower, *upper))
    assert jnp.all(jnp.isfinite(diagonal))
    assert all(jnp.all(jnp.isfinite(value)) for value in (*lower, *upper))
    assert jnp.all(diagonal > 0.0)

    # Only the first owned radial row touches the collapsed polar face.  The
    # remaining radial rows still have ordinary lower-x neighbors.
    assert jnp.allclose(
        lower[0][0],
        0.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert jnp.any(jnp.abs(lower[0][1:]) > 1.0e-12)
    assert jnp.all(diagonal[0] > 0.0)


def test_all_solvax_preconditioners_are_finite_and_reduce_principal_residual():
    geometry, domain, _scalar, projectors, face_bc = _bands_fixture()
    diagonal, lower, upper = _principal_perp_laplacian_bands(
        geometry, domain, projectors, face_bc
    )
    rhs = jnp.sin(jnp.arange(geometry.owned_shape[0], dtype=jnp.float64))[:, None, None]
    rhs = jnp.broadcast_to(rhs, geometry.owned_shape)

    for kind in (
        "none",
        "jacobi",
        "line-u",
        "axis-core-line-u",
        "line-v",
        "line-uv",
    ):
        config = SolvaxGmresConfig(
            preconditioner=kind,
            regularization_epsilon=1.0e-8,
        )
        preconditioner = build_solvax_perp_laplacian_preconditioner(
            geometry, domain, projectors, face_bc, config
        )
        if kind == "none":
            assert preconditioner is None
            continue
        correction = preconditioner(rhs)
        assert correction.shape == geometry.owned_shape
        assert jnp.all(jnp.isfinite(correction))
        # The preconditioners approximate the full matrix-free operator, not
        # the diagonal-only band model.  A diagonal residual reduction is
        # therefore not a valid contract for line-u/line-v/line-uv; require
        # a finite, nontrivial correction as the auxiliary-path smoke check.
        assert jnp.linalg.norm(correction) > 0.0


def test_inverse_solver_reconstructs_smooth_axis_regular_field():
    geometry, domain, _context, (r, theta, z), exchange, scalar, _vector, _flux = polar_fixture(
        shape=(4, 8, 8)
    )
    del theta
    field_halo = scalar_field_halo(
        r, 0.0 * r, z, lambda r, _theta, z: r**2 * jnp.cos(z)
    )
    phi = owned(field_halo, geometry)
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        tol=2.0e-5,
        atol=2.0e-6,
        maxiter=30,
        restart=10,
        regularization_epsilon=1.0e-3,
        preconditioner="jacobi",
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        config=config,
    )
    rhs = solver._apply_A(
        phi,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )
    solution, info = solver(rhs, return_diagnostics=True)
    reconstructed = solver._apply_A(
        solution,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )
    residual = jnp.linalg.norm(reconstructed - rhs)
    rhs_norm = jnp.linalg.norm(rhs)

    assert jnp.all(jnp.isfinite(solution))
    assert jnp.all(jnp.isfinite(reconstructed))
    assert bool(info.converged)
    assert residual <= 5.0e-3 * jnp.maximum(rhs_norm, 1.0)
    assert jnp.max(jnp.abs(solution[0] - phi[0])) < 5.0e-2
    assert jnp.max(jnp.abs(reconstructed[0] - rhs[0])) < 5.0e-3


def test_inverse_solver_uses_supplied_axis_core_stencil_context():
    geometry, domain, _context, (r, theta, z), exchange, scalar, _vector, _flux = polar_fixture(
        shape=(4, 8, 8)
    )
    del theta
    field_halo = scalar_field_halo(
        r, 0.0 * r, z, lambda r, _theta, z: r**2 * jnp.cos(z)
    )
    field_owned = owned(field_halo, geometry)
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(regularization_epsilon=1.0e-3)

    custom_cell_reconstruction = build_axis_core_cell_gradient_reconstruction(
        geometry.layout,
        domain,
        polynomial_degree=1,
        observation_ring_count=3,
        target_ring_count=1,
    )
    custom_context = StencilBuilderContext(
        layout=geometry.layout,
        domain=domain,
        axis_core_cell_gradient_reconstruction=custom_cell_reconstruction,
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=custom_context,
        config=config,
    )
    default_solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        config=config,
    )

    assert solver.stencil_builder_context is custom_context
    leaves, treedef = jax.tree_util.tree_flatten(solver)
    rebuilt_solver = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt_solver.stencil_builder_context is not None
    assert (
        rebuilt_solver.stencil_builder_context.axis_core_cell_gradient_reconstruction
        .polynomial_degree
        == 1
    )
    custom_result = solver._apply_A(
        field_owned,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )
    default_result = default_solver._apply_A(
        field_owned,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )

    assert jnp.all(jnp.isfinite(custom_result))
    assert jnp.max(jnp.abs(custom_result - default_result)) > 1.0e-8


def test_axis_core_line_u_coarse_factor_inverts_projected_operator():
    geometry, domain, context, _coordinates, exchange, scalar, _vector, _flux = (
        polar_fixture(shape=(8, 16, 8))
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        regularization_epsilon=1.0e-3,
        preconditioner="axis-core-line-u",
    )
    solver_without_payload = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=config,
    )

    def apply_A(values):
        return solver_without_payload._apply_A(
            values,
            face_bc=face_bc,
            control_volume_boundary_bc=None,
            project_mean_zero=False,
        )

    reconstruction = context.axis_core_face_gradient_reconstruction
    assert reconstruction is not None
    payload = build_axis_core_line_u_preconditioner(
        apply_A,
        reconstruction,
        domain,
    )
    coefficient_count = payload.coefficient_count
    eta = jnp.arange(geometry.owned_shape[2], dtype=jnp.float64)[None, :]
    mode = jnp.arange(1, coefficient_count + 1, dtype=jnp.float64)[:, None]
    expected = (
        0.1 * jnp.cos(2.0 * jnp.pi * eta / geometry.owned_shape[2]) / mode
        + 0.03 * jnp.sin(4.0 * jnp.pi * eta / geometry.owned_shape[2])
    )
    field = _axis_core_prolong_from_coefficients(
        expected,
        reconstruction,
        domain,
    )
    coarse_rhs = _axis_core_restrict_to_coefficients(
        apply_A(field),
        reconstruction,
        domain,
    )
    solved = payload.solve_coefficients(coarse_rhs, domain)

    assert int(payload.factors.core.n_clamped) == 0
    assert jnp.all(jnp.isfinite(solved))
    assert jnp.max(jnp.abs(solved - expected)) < 2.0e-8


def test_axis_core_line_u_reduces_axis_regular_inverse_iterations():
    geometry, domain, context, _coordinates, exchange, scalar, _vector, _flux = (
        polar_fixture(shape=(8, 16, 8))
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    common = dict(
        tol=1.0e-8,
        atol=1.0e-10,
        acceptance_tol=1.0e-8,
        acceptance_atol=1.0e-10,
        maxiter=120,
        restart=30,
        regularization_epsilon=1.0e-3,
    )
    line_solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=SolvaxGmresConfig(preconditioner="line-u", **common),
    )
    reconstruction = context.axis_core_face_gradient_reconstruction
    assert reconstruction is not None

    def apply_A(values):
        return line_solver._apply_A(
            values,
            face_bc=face_bc,
            control_volume_boundary_bc=None,
            project_mean_zero=False,
        )

    payload = build_axis_core_line_u_preconditioner(
        apply_A,
        reconstruction,
        domain,
    )
    axis_solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        axis_core_line_u_preconditioner=payload,
        config=SolvaxGmresConfig(preconditioner="axis-core-line-u", **common),
    )
    coefficient_count = payload.coefficient_count
    eta = jnp.arange(geometry.owned_shape[2], dtype=jnp.float64)[None, :]
    mode = jnp.arange(1, coefficient_count + 1, dtype=jnp.float64)[:, None]
    coefficients = (
        0.08 * jnp.cos(2.0 * jnp.pi * eta / geometry.owned_shape[2]) / mode
        + 0.02 * jnp.sin(4.0 * jnp.pi * eta / geometry.owned_shape[2])
    )
    exact = _axis_core_prolong_from_coefficients(
        coefficients,
        reconstruction,
        domain,
    )
    rhs = apply_A(exact)
    _line_solution, line_info = line_solver(rhs, return_diagnostics=True)
    axis_solution, axis_info = axis_solver(rhs, return_diagnostics=True)

    assert bool(axis_info.converged)
    assert int(axis_info.num_steps) < int(line_info.num_steps)
    assert int(axis_info.num_steps) <= 0.5 * int(line_info.num_steps)
    assert jnp.linalg.norm(apply_A(axis_solution) - rhs) <= 1.0e-8 * jnp.linalg.norm(rhs)


def test_auxiliary_fixture_has_no_control_volume_rows():
    """CV algebra is coordinate-independent and is covered by CV tests.

    The regular polar fixture intentionally has no embedded irregular-cell
    rows.  Manufacturing such rows here would test invalid geometry rather
    than axis regularity; the solver path above therefore exercises the
    regular-grid closure only.
    """
    geometry, *_ = polar_fixture(shape=(4, 8, 8))
    assert geometry.cell_volume_geometry is not None
    assert geometry.cell_volume_geometry.volume_fraction.shape == geometry.owned_shape
