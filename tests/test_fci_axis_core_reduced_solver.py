"""Reduced Cartesian-axis / polar-outer phi-space coverage."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import (
    build_axis_core_cell_gradient_reconstruction,
    build_axis_core_face_gradient_reconstruction,
)
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import (
    AxisCoreReducedVector3D,
    LocalPerpLaplacianInverseSolver,
    build_axis_core_reduced_space,
)
from drbx.native.fci_drb_EB_rhs import FciDrbEBState, project_axis_core_state
from drbx.native.fci_sharding import make_shard_mesh


def _space(shape=(8, 16, 8)):
    geometry, domain, context, coordinates, exchange, scalar, vector, flux = polar_fixture(shape)
    del coordinates, vector, flux
    reconstruction = context.axis_core_face_gradient_reconstruction
    assert reconstruction is not None
    return geometry, domain, context, exchange, scalar, build_axis_core_reduced_space(
        reconstruction, geometry, domain
    )


def test_reduced_space_restriction_prolongation_and_inner_product_isometry():
    geometry, _domain, _context, _exchange, _scalar, space = _space()
    p = space.coefficient_count
    nz = geometry.owned_shape[2]
    coefficients = jnp.arange(p * nz, dtype=jnp.float64).reshape(p, nz) / 17.0
    outer = jnp.arange(jnp.prod(jnp.asarray(space.outer_shape)), dtype=jnp.float64).reshape(
        space.outer_shape
    ) / 23.0
    vector = AxisCoreReducedVector3D(coefficients, outer)
    reconstructed = space.prolong(vector)
    recovered = space.restrict(reconstructed)

    assert jnp.max(jnp.abs(recovered.coefficients - coefficients)) < 2.0e-11
    assert jnp.max(jnp.abs(recovered.phi_outer - outer)) < 2.0e-12
    assert jnp.abs(
        space.inner_product(vector, vector)
        - space.full_inner_product(reconstructed, reconstructed)
    ) < 2.0e-10


def test_cartesian_polynomials_are_exact_and_m4_is_not_representable():
    geometry, domain, context, _exchange, _scalar, space = _space()
    reconstruction = context.axis_core_face_gradient_reconstruction
    assert reconstruction is not None
    p = space.coefficient_count
    nz = geometry.owned_shape[2]
    coefficients = jnp.sin(jnp.arange(p * nz, dtype=jnp.float64)).reshape(p, nz)
    vector = AxisCoreReducedVector3D(
        coefficients,
        jnp.zeros(space.outer_shape, dtype=jnp.float64),
    )
    assert jnp.max(jnp.abs(space.prolong(space.restrict(space.prolong(vector))) - space.prolong(vector))) < 2.0e-11

    theta = 2.0 * jnp.pi * jnp.arange(geometry.owned_shape[1]) / geometry.owned_shape[1]
    m4 = jnp.cos(4.0 * theta)[None, :, None]
    nonregular = jnp.broadcast_to(m4, geometry.owned_shape)
    projected = space.compatible_residual(nonregular)
    residual = space.full_norm(nonregular - projected)
    assert residual > 1.0e-4 * space.full_norm(nonregular)
    del domain, reconstruction


def test_projector_preserves_weighted_integral_outer_values_and_polynomials():
    geometry, _domain, _context, _exchange, _scalar, space = _space((8, 16, 8))
    field = jnp.sin(jnp.arange(jnp.prod(jnp.asarray(geometry.owned_shape)), dtype=jnp.float64)).reshape(
        geometry.owned_shape
    )
    projected = space.project(field)
    ones = jnp.ones(geometry.owned_shape, dtype=jnp.float64)
    assert jnp.abs(
        space.full_inner_product(ones, projected)
        - space.full_inner_product(ones, field)
    ) < 2.0e-10
    assert jnp.array_equal(projected[space.core_ring_count :], field[space.core_ring_count :])

    coefficients = jnp.cos(
        jnp.arange(space.coefficient_count * geometry.owned_shape[2], dtype=jnp.float64)
    ).reshape(space.coefficient_count, geometry.owned_shape[2])
    polynomial = space.prolong(
        AxisCoreReducedVector3D(
            coefficients,
            jnp.zeros(space.outer_shape, dtype=jnp.float64),
        )
    )
    assert jnp.max(jnp.abs(space.project(polynomial) - polynomial)) < 2.0e-11


def test_state_projector_full_grid_is_identity_and_phi_is_not_a_leaf():
    geometry, _domain, _context, _exchange, _scalar, space = _space((8, 16, 8))
    zeros = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
    state = FciDrbEBState(
        density=zeros,
        phi=jnp.full_like(zeros, 3.0),
        Te=zeros,
        Ti=zeros,
        Vi=zeros,
        Ve=zeros,
        vorticity=zeros,
    )
    assert project_axis_core_state(state, None, "full-grid") is state
    projected = project_axis_core_state(state, space, "galerkin")
    assert jnp.array_equal(projected.phi, state.phi)


def test_reduced_manufactured_phi_solve_converges():
    geometry, domain, context, exchange, scalar, space = _space()
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        tol=1.0e-7,
        atol=1.0e-8,
        maxiter=80,
        restart=20,
        regularization_epsilon=1.0e-3,
        preconditioner="line-u",
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=config,
    )
    coefficients = jnp.cos(jnp.arange(space.coefficient_count * geometry.owned_shape[2], dtype=jnp.float64)).reshape(
        space.coefficient_count, geometry.owned_shape[2]
    ) / 3.0
    exact = space.prolong(
        AxisCoreReducedVector3D(
            coefficients,
            jnp.zeros(space.outer_shape, dtype=jnp.float64),
        )
    )
    rhs = solver._apply_A(
        exact,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )
    solution, info = solver.solve_axis_core_reduced(
        rhs,
        space=space,
        return_diagnostics=True,
    )
    assert bool(info.converged)
    assert int(info.num_steps) < config.maxiter
    # The matrix-free full operator can emit a small complement component
    # even for a reduced trial field.  The reduced solve controls the
    # compatible residual; the diagnostic makes that distinction explicit.
    assert info.final_residual.incompatible_l2 < info.rhs.incompatible_l2
    assert info.final_residual.total_l2 < 1.0e-3
    assert space.full_norm(solution - exact) < 2.0e-2 * jnp.maximum(space.full_norm(exact), 1.0)


def test_reduced_and_full_gmres_iteration_comparison():
    geometry, domain, context, exchange, scalar, space = _space((8, 16, 8))
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        tol=1.0e-7,
        atol=1.0e-8,
        maxiter=80,
        restart=20,
        regularization_epsilon=1.0e-3,
        preconditioner="line-u",
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(True, False, False),
        stencil_builder_context=context,
        config=config,
    )
    coefficients = jnp.sin(
        jnp.arange(space.coefficient_count * geometry.owned_shape[2], dtype=jnp.float64)
    ).reshape(space.coefficient_count, geometry.owned_shape[2]) / 5.0
    exact = space.prolong(
        AxisCoreReducedVector3D(coefficients, jnp.zeros(space.outer_shape))
    )
    rhs = solver._apply_A(
        exact,
        face_bc=face_bc,
        control_volume_boundary_bc=None,
        project_mean_zero=False,
    )
    _full_solution, full_info = solver(rhs, return_diagnostics=True)
    _reduced_solution, reduced_info = solver.solve_axis_core_reduced(
        rhs, space=space, return_diagnostics=True
    )
    assert bool(reduced_info.converged)
    assert int(reduced_info.num_steps) <= 0.75 * int(full_info.num_steps)


def test_reduced_constraint_removes_complement_conditioning():
    geometry, _domain, _context, _exchange, _scalar, space = _space((8, 12, 3))
    nx, ny, nz = geometry.owned_shape
    p = space.coefficient_count
    basis = jnp.asarray(space.reconstruction.coefficient_to_observation_basis)
    weights = space.core_weights.transpose(1, 0, 2).reshape(space.core_ring_count * ny, nz)
    gram = space.gram_eta[:, :, 0]
    weighted_basis = basis.T @ jnp.diag(weights[:, 0]) @ basis
    del weighted_basis
    v = basis
    m = jnp.diag(weights[:, 0])
    q = v @ jnp.linalg.solve(gram, v.T @ m)
    eps = 1.0e-6
    full_matrix = q + eps * (jnp.eye(space.core_ring_count * ny) - q)
    reduced_matrix = jnp.linalg.solve(gram, v.T @ m) @ full_matrix @ v
    full_singular = jnp.linalg.svd(full_matrix, compute_uv=False)
    reduced_singular = jnp.linalg.svd(reduced_matrix, compute_uv=False)
    assert float(full_singular[0] / full_singular[-1]) > 1.0e5
    assert float(reduced_singular[0] / reduced_singular[-1]) < 10.0
    del nx, nz, p


def test_reduced_space_pytree_and_jit_smoke():
    _geometry, _domain, _context, _exchange, _scalar, space = _space()
    leaves, treedef = jax.tree_util.tree_flatten(space)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    vector = AxisCoreReducedVector3D(
        jnp.ones((space.coefficient_count, space.geometry.owned_shape[2])),
        jnp.zeros(space.outer_shape),
    )
    prolong = jax.jit(lambda value: rebuilt.prolong(value))(vector)
    inner = jax.jit(lambda value: rebuilt.inner_product(value, value))(vector)
    assert prolong.shape == space.geometry.owned_shape
    assert jnp.isfinite(inner)


def test_reduced_space_rejects_radial_sharding():
    geometry, domain, context, _exchange, _scalar, _space_payload = _space()
    with pytest.raises(ValueError, match="radial shard count"):
        build_axis_core_reduced_space(
            context.axis_core_face_gradient_reconstruction,
            geometry,
            domain.__class__(
                layout=domain.layout,
                shard_spec=domain.shard_spec.__class__(
                    global_shape=domain.shard_spec.global_shape,
                    owned_start=domain.shard_spec.owned_start,
                    owned_stop=domain.shard_spec.owned_stop,
                    shard_index=(0, 0, 0),
                    shard_counts=(2, 1, 1),
                    periodic_axes=domain.shard_spec.periodic_axes,
                    axis_regular_axes=domain.shard_spec.axis_regular_axes,
                    halo_width=domain.shard_spec.halo_width,
                    side_kind_lower=domain.shard_spec.side_kind_lower,
                    side_kind_upper=domain.shard_spec.side_kind_upper,
                ),
                mesh_axis_names=domain.mesh_axis_names,
            ),
        )


def test_reduced_space_rejects_theta_sharding_until_replicated_gram_is_wired():
    geometry, domain, context, _exchange, _scalar, _space_payload = _space()
    sharded_spec = domain.shard_spec.__class__(
        global_shape=domain.shard_spec.global_shape,
        owned_start=domain.shard_spec.owned_start,
        owned_stop=domain.shard_spec.owned_stop,
        shard_index=(0, 0, 0),
        shard_counts=(1, 2, 1),
        periodic_axes=domain.shard_spec.periodic_axes,
        axis_regular_axes=domain.shard_spec.axis_regular_axes,
        halo_width=domain.shard_spec.halo_width,
        side_kind_lower=domain.shard_spec.side_kind_lower,
        side_kind_upper=domain.shard_spec.side_kind_upper,
    )
    sharded_domain = domain.__class__(
        layout=domain.layout,
        shard_spec=sharded_spec,
        mesh_axis_names=domain.mesh_axis_names,
    )
    with pytest.raises(ValueError, match="theta shard count"):
        build_axis_core_reduced_space(
            context.axis_core_face_gradient_reconstruction,
            geometry,
            sharded_domain,
        )


def test_reduced_space_eta_sharding_matches_unsharded_prolong_restrict_and_inner_product():
    """Four eta shards preserve the reduced-space algebra exactly.

    The polar fixture is independent of eta, so a local two-plane fixture is
    sufficient to represent each of the four equal eta shards.  The test is
    intentionally executed inside ``shard_map``: this exercises the eta
    all-reduces in ``AxisCoreReducedSpace3D`` rather than only checking array
    shapes after a device transfer.
    """
    if len(jax.devices()) < 4:
        pytest.skip("eta-sharding regression requires at least 4 JAX devices")

    global_shape = (8, 16, 8)
    serial_geometry, _serial_domain, _serial_context, _serial_exchange, _serial_scalar, serial_space = _space(
        global_shape
    )
    p = serial_space.coefficient_count
    coefficients = jnp.sin(
        jnp.arange(p * global_shape[2], dtype=jnp.float64) / 7.0
    ).reshape(p, global_shape[2])
    outer = jnp.cos(
        jnp.arange(math.prod(serial_space.outer_shape), dtype=jnp.float64) / 11.0
    ).reshape(serial_space.outer_shape)
    serial_vector = AxisCoreReducedVector3D(coefficients, outer)
    serial_full = serial_space.prolong(serial_vector)
    serial_recovered = serial_space.restrict(serial_full)
    serial_inner = serial_space.inner_product(serial_vector, serial_vector)

    local_geometry, local_domain, _local_context, *_ = polar_fixture((8, 16, 2))
    # ``polar_fixture`` derives dz from its local eta extent.  A shard's
    # physical eta spacing is the global spacing, so restore the global
    # volume weight before comparing the four local contributions.
    local_geometry = replace(
        local_geometry,
        cell_volume_geometry=replace(
            local_geometry.cell_volume_geometry,
            volume=local_geometry.cell_volume_geometry.volume / 4.0,
        ),
    )
    local_domain = replace(
        local_domain,
        shard_spec=replace(
            local_domain.shard_spec,
            global_shape=global_shape,
            shard_counts=(1, 1, 4),
        ),
        mesh_axis_names=(None, None, "z"),
    )
    reconstruction = build_axis_core_face_gradient_reconstruction(
        local_geometry.layout,
        local_domain,
        polynomial_degree=3,
        observation_ring_count=6,
        target_ring_count=3,
    )
    sharded_space = build_axis_core_reduced_space(
        reconstruction, local_geometry, local_domain
    )

    mesh = make_shard_mesh((1, 1, 4))
    partition = P(None, "z")
    coefficient_sharding = NamedSharding(mesh, partition)
    outer_sharding = NamedSharding(mesh, P(None, None, "z"))
    coefficients_sharded = jax.device_put(coefficients, coefficient_sharding)
    outer_sharded = jax.device_put(outer, outer_sharding)

    def kernel(local_coefficients, local_outer):
        vector = AxisCoreReducedVector3D(local_coefficients, local_outer)
        full = sharded_space.prolong(vector)
        recovered = sharded_space.restrict(full)
        return (
            full,
            recovered.coefficients,
            recovered.phi_outer,
            sharded_space.inner_product(vector, vector),
        )

    sharded_kernel = jax.jit(
        shard_map(
            kernel,
            mesh=mesh,
            in_specs=(partition, P(None, None, "z")),
            out_specs=(P(None, None, "z"), partition, P(None, None, "z"), P()),
            check_rep=False,
        )
    )
    sharded_full, sharded_coefficients, sharded_outer, sharded_inner = sharded_kernel(
        coefficients_sharded, outer_sharded
    )

    assert jnp.max(jnp.abs(sharded_full - serial_full)) < 2.0e-11
    assert jnp.max(jnp.abs(sharded_coefficients - serial_recovered.coefficients)) < 2.0e-11
    assert jnp.max(jnp.abs(sharded_outer - serial_recovered.phi_outer)) < 2.0e-11
    assert jnp.abs(sharded_inner - serial_inner) < 2.0e-10
