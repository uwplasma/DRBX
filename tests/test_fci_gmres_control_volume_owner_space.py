"""Owner-space masking and aggregate-weight tests for control-volume solves."""

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_regular_operator_support import polar_fixture
from drbx.native.fci_boundaries import (
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentFittedFaceRows3D,
    LocalMomentReconstruction3D,
)
from drbx.geometry import LocalControlVolumeCellGeometry3D, StencilBuilderContext
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    build_local_perp_laplacian_face_projectors,
    build_solvax_perp_laplacian_preconditioner,
)
from drbx.native.fci_gmres import (
    SolvaxGmresConfig,
    _spmd_dot,
    _spmd_norm,
    _spmd_remove_weighted_mean,
    solvax_gmres_solve,
)


def test_supplied_volume_weights_define_krylov_dot_and_norm():
    geometry, domain, *_ = polar_fixture(shape=(3, 4, 4))
    active = jnp.ones(geometry.owned_shape, dtype=bool)
    weights = jnp.ones(geometry.owned_shape, dtype=jnp.float64)
    weights = weights.at[0, 0, 0].set(4.0)
    x = jnp.ones(geometry.owned_shape, dtype=jnp.float64)
    y = jnp.zeros_like(x).at[0, 0, 0].set(3.0)

    assert jnp.allclose(
        _spmd_dot(x, y, geometry, domain, active, weights), 12.0
    )
    assert jnp.allclose(
        _spmd_norm(y, geometry, domain, active, weights), 6.0
    )


def test_inactive_zero_weight_alias_is_ignored_by_weighted_krylov_norm():
    geometry, domain, *_ = polar_fixture(shape=(3, 4, 4))
    active = jnp.ones(geometry.owned_shape, dtype=bool).at[0, 1, 0].set(False)
    weights = jnp.ones(geometry.owned_shape, dtype=jnp.float64).at[0, 1, 0].set(0.0)
    x = jnp.ones(geometry.owned_shape, dtype=jnp.float64).at[0, 1, 0].set(1.0e12)

    assert jnp.allclose(_spmd_norm(x, geometry, domain, active, weights), jnp.sqrt(47.0))
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D


def _merged_control_volume(
    geometry,
    *,
    source=(0, 1, 0),
    owner=(0, 0, 0),
    agglomeration_kind="embedded",
):
    cells = LocalControlVolumeCellGeometry3D.identity(
        geometry.layout,
        volume=jnp.ones(geometry.owned_shape, dtype=jnp.float64),
        centroid=jnp.zeros(geometry.owned_shape + (3,), dtype=jnp.float64),
    )
    source_i, source_j, source_k = source
    owner_i, owner_j, owner_k = owner
    aggregate_volume = cells.aggregate_volume.at[owner].set(2.0)
    aggregate_volume = aggregate_volume.at[source].set(0.0)
    cells = replace(
        cells,
        owner_i=cells.owner_i.at[source].set(owner_i),
        owner_j=cells.owner_j.at[source].set(owner_j),
        owner_k=cells.owner_k.at[source].set(owner_k),
        is_merged_source=cells.is_merged_source.at[source].set(True),
        is_active_owner=cells.is_active_owner.at[source].set(False),
        aggregate_volume=aggregate_volume,
    )
    return LocalEmbeddedControlVolumeGeometry3D(
        cells=cells,
        regular_faces=geometry.regular_face_geometry,
        irregular_faces=LocalControlVolumeFaceRows3D.empty(geometry.layout),
        reconstruction=LocalMomentReconstruction3D.empty(geometry.layout),
        face_functionals=LocalMomentFittedFaceRows3D.empty(geometry.layout),
        agglomeration_kind=agglomeration_kind,
    )


def test_gmres_masks_sources_and_uses_aggregate_volume_for_projection():
    geometry, domain, _context, _coordinates, _exchange, _scalar, _vector, _flux = (
        polar_fixture(shape=(3, 4, 4))
    )
    source = (0, 1, 0)
    owner = (0, 0, 0)
    active = jnp.ones(geometry.owned_shape, dtype=bool).at[source].set(False)
    weights = jnp.ones(geometry.owned_shape, dtype=jnp.float64)
    weights = weights.at[owner].set(2.0).at[source].set(0.0)
    rhs = jnp.ones(geometry.owned_shape, dtype=jnp.float64).at[source].set(91.0)
    guess = jnp.zeros_like(rhs).at[source].set(-37.0)

    solution, info = solvax_gmres_solve(
        lambda values: values,
        rhs,
        guess,
        geometry,
        domain,
        SolvaxGmresConfig(
            maxiter=8,
            restart=4,
            tol=1.0e-10,
            atol=1.0e-12,
        ),
        active_cell_mask=active,
        volume_weights=weights,
    )

    assert bool(info.converged)
    assert float(solution[source]) == 0.0
    assert jnp.allclose(solution[active], 1.0)

    weighted_field = jnp.zeros_like(rhs).at[owner].set(2.0).at[source].set(91.0)
    projected = _spmd_remove_weighted_mean(
        weighted_field,
        geometry,
        domain,
        active,
        weights,
    )
    expected_mean = 4.0 / float(jnp.sum(weights * active))
    assert jnp.allclose(projected[owner], 2.0 - expected_mean)
    assert jnp.allclose(
        jnp.sum(weights * active * projected),
        0.0,
        atol=1.0e-12,
    )
    assert float(projected[source]) == 0.0


def test_inverse_solver_cv_path_masks_source_slots_in_operator_and_solution():
    geometry, domain, _context, _coordinates, exchange, scalar, _vector, _flux = (
        polar_fixture(shape=(3, 4, 4))
    )
    cv_geometry = _merged_control_volume(
        geometry, agglomeration_kind="corner-edge"
    )
    boundary_bc = LocalControlVolumeBoundaryBC3D.empty()
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    direct_polar_context = StencilBuilderContext(
        layout=domain.layout,
        domain=domain,
    )
    solver = LocalPerpLaplacianInverseSolver(
        geometry=geometry,
        domain=domain,
        control_volume_geometry=cv_geometry,
        control_volume_boundary_bc=boundary_bc,
        halo_exchange=exchange,
        topology_filler=scalar,
        face_bc=face_bc,
        axis_regular_axes=(False, False, False),
        stencil_builder_context=direct_polar_context,
        config=SolvaxGmresConfig(
            maxiter=12,
            restart=6,
            tol=2.0e-5,
            atol=1.0e-7,
            regularization_epsilon=1.0e-3,
            preconditioner="none",
        ),
    )

    source = (0, 1, 0)
    owner = (0, 0, 0)
    field = jnp.zeros(geometry.owned_shape, dtype=jnp.float64).at[owner].set(0.25)
    field = field.at[source].set(100.0)
    applied = solver._apply_A(
        field,
        face_bc=face_bc,
        control_volume_boundary_bc=boundary_bc,
        project_mean_zero=False,
    )
    reference = solver._apply_A(
        field.at[source].set(-100.0),
        face_bc=face_bc,
        control_volume_boundary_bc=boundary_bc,
        project_mean_zero=False,
    )
    assert float(applied[source]) == 0.0
    assert jnp.allclose(applied, reference)

    rhs = jnp.zeros(geometry.owned_shape, dtype=jnp.float64).at[owner].set(1.0)
    solution = solver.solve_rlp_owner(
        rhs.at[source].set(10.0),
        guess_owned=field,
    )
    assert float(solution[source]) == 0.0


def test_corner_edge_line_u_projects_through_fine_members_and_masks_aliases():
    geometry, domain, *_ = polar_fixture(shape=(3, 4, 4))
    source = (0, 1, 0)
    cv_geometry = _merged_control_volume(
        geometry,
        source=source,
        owner=(0, 0, 0),
        agglomeration_kind="corner-edge",
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    projectors = build_local_perp_laplacian_face_projectors(
        geometry,
        domain,
        axis_regular_axes=(False, False, False),
    )
    preconditioner = build_solvax_perp_laplacian_preconditioner(
        geometry,
        domain,
        projectors,
        face_bc,
        SolvaxGmresConfig(
            regularization_epsilon=1.0e-3,
            preconditioner="line-u",
        ),
        control_volume_geometry=cv_geometry,
    )
    residual = jnp.arange(
        1,
        1 + int(jnp.prod(jnp.asarray(geometry.owned_shape))),
        dtype=jnp.float64,
    ).reshape(geometry.owned_shape)
    correction = preconditioner(residual)
    assert bool(jnp.all(jnp.isfinite(correction)))
    assert float(correction[source]) == 0.0
    assert bool(jnp.any(jnp.abs(correction) > 0.0))
