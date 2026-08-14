"""Nested radial-tree preconditioner tests for angular agglomeration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import (
    _assemble_angular_agglomeration_tree_principal_coefficients,
    _build_angular_agglomeration_line_u_preconditioner,
    _lift_cell_field_to_faces,
    _validate_concrete_angular_agglomeration_tree_assembly,
    build_local_perp_laplacian_face_projectors,
    build_solvax_perp_laplacian_preconditioner,
)
from drbx.native.fci_angular_agglomeration import (
    lower_polar_angular_agglomeration_geometry,
)


PROFILE = (8, 4, 4, 2, 1)


def _fixture(nz=3, profile=PROFILE):
    shape = (len(profile), int(profile[0]), nz)
    geometry, domain, *_ = polar_fixture(shape=shape, halo_width=1)
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    host = build_polar_angular_agglomeration_geometry(
        u, theta, eta, lambda p: np.ones(p.shape[:-1]),
        angular_group_size=profile, quadrature_order=2,
    )
    lowered = lower_polar_angular_agglomeration_geometry(host, geometry)
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(True, False, False)
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        regularization_epsilon=1.0e-3, preconditioner="line-u"
    )
    coefficients = _assemble_angular_agglomeration_tree_principal_coefficients(
        geometry, domain, projectors, face_bc, config, lowered
    )
    return geometry, domain, lowered, projectors, face_bc, config, coefficients


def _dense_tree(diagonal, child_edge, parent_i, parent_j, parent_k, active):
    shape = diagonal.shape
    active_np = np.asarray(active, dtype=bool)
    owners = np.argwhere(active_np)
    owner_number = {tuple(index): n for n, index in enumerate(owners)}
    matrix = np.zeros((len(owners), len(owners)))
    diagonal_np = np.asarray(diagonal)
    edge_np = np.asarray(child_edge)
    pi, pj, pk = map(np.asarray, (parent_i, parent_j, parent_k))
    for owner, number in owner_number.items():
        matrix[number, number] = diagonal_np[owner]
        if owner[0] > 0:
            parent = (int(pi[owner]), int(pj[owner]), int(pk[owner]))
            parent_number = owner_number[parent]
            matrix[number, parent_number] -= edge_np[owner]
            matrix[parent_number, number] -= edge_np[owner]
    return owners, matrix


def _radial_fine_conductance(geometry, projectors):
    regular = geometry.regular_face_geometry
    face_slice = geometry.layout.location_owned_slices("x_face")[0]
    centers = np.asarray(geometry.grid.x.centers_halo)
    distance = centers[face_slice.start:face_slice.stop] - centers[
        face_slice.start - 1:face_slice.stop - 1
    ]
    logical_area = np.asarray(
        _lift_cell_field_to_faces(
            geometry.spacing.dy_owned * geometry.spacing.dz_owned,
            axis=0,
            periodic=False,
        )
    )
    return (
        np.asarray(geometry.face_metric.x.J_owned)
        * np.asarray(projectors[0][..., 0, 0])
        * np.asarray(regular.x_area)
        * np.asarray(regular.x_area_fraction)
        * np.asarray(regular.x_open_mask)
        * logical_area
        / distance[:, None, None]
    )


def test_tree_line_u_matches_dense_owner_solve_for_random_rhs_and_eta():
    geometry, domain, lowered, projectors, face_bc, config, coefficients = _fixture(nz=3)
    volume, diagonal, edge, pi, pj, pk, active = coefficients
    solver = _build_angular_agglomeration_line_u_preconditioner(
        geometry, domain, projectors, face_bc, config, lowered,
        principal_coefficients=coefficients,
    )
    rng = np.random.default_rng(82)
    residual = rng.normal(size=geometry.owned_shape)
    actual = np.asarray(solver(jnp.asarray(residual)))
    owners, matrix = _dense_tree(diagonal, edge, pi, pj, pk, active)
    expected = np.zeros_like(residual)
    rhs = np.asarray(volume) * residual
    expected_values = np.linalg.solve(matrix, rhs[tuple(owners.T)])
    expected[tuple(owners.T)] = expected_values
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)
    assert np.all(actual[~np.asarray(active)] == 0.0)


def test_composite_tree_line_u_matches_dense_owner_solve():
    profile = (12, 6, 3, 1, 1)
    geometry, domain, lowered, projectors, face_bc, config, coefficients = _fixture(
        nz=4, profile=profile
    )
    volume, diagonal, edge, pi, pj, pk, active = coefficients
    solver = _build_angular_agglomeration_line_u_preconditioner(
        geometry,
        domain,
        projectors,
        face_bc,
        config,
        lowered,
        principal_coefficients=coefficients,
    )
    residual = np.random.default_rng(120631).normal(size=geometry.owned_shape)
    actual = np.asarray(solver(jnp.asarray(residual)))
    owners, matrix = _dense_tree(diagonal, edge, pi, pj, pk, active)
    expected = np.zeros_like(residual)
    rhs = np.asarray(volume) * residual
    expected[tuple(owners.T)] = np.linalg.solve(matrix, rhs[tuple(owners.T)])
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)


def test_tree_solver_jits_and_keeps_aliases_exactly_zero():
    geometry, domain, lowered, projectors, face_bc, config, coefficients = _fixture()
    solver = _build_angular_agglomeration_line_u_preconditioner(
        geometry, domain, projectors, face_bc, config, lowered,
        principal_coefficients=coefficients,
    )
    result = np.asarray(jax.jit(solver)(jnp.ones(geometry.owned_shape)))
    assert np.all(np.isfinite(result))
    assert np.all(result[~np.asarray(lowered.cells.is_active_owner)] == 0.0)


def test_production_lowering_has_no_compact_face_payload():
    geometry, _domain, lowered, *_ = _fixture()
    assert lowered.irregular_faces.max_rows == 0
    assert lowered.face_functionals is None
    assert lowered.reconstruction.max_rows == 0
    assert lowered.regular_faces is geometry.regular_face_geometry


def test_control_volume_dispatch_selects_the_angular_tree_line_u():
    geometry, domain, lowered, projectors, face_bc, config, coefficients = _fixture()
    dispatched = build_solvax_perp_laplacian_preconditioner(
        geometry, domain, projectors, face_bc, config,
        control_volume_geometry=lowered,
    )
    direct = _build_angular_agglomeration_line_u_preconditioner(
        geometry, domain, projectors, face_bc, config, lowered,
        principal_coefficients=coefficients,
    )
    residual = jnp.arange(np.prod(geometry.owned_shape), dtype=jnp.float64).reshape(geometry.owned_shape)
    np.testing.assert_allclose(
        np.asarray(dispatched(residual)), np.asarray(direct(residual)),
        rtol=2e-12, atol=2e-12,
    )


def test_duplicate_radial_subfaces_sum_into_one_child_parent_edge():
    geometry, _domain, lowered, projectors, _face_bc, _config, coefficients = _fixture()
    _volume, _diagonal, edge, _pi, _pj, _pk, _active = coefficients
    owners = np.stack(
        tuple(np.asarray(value) for value in (
            lowered.cells.owner_i, lowered.cells.owner_j, lowered.cells.owner_k
        )),
        axis=-1,
    )
    T = _radial_fine_conductance(geometry, projectors)
    expected = np.zeros_like(np.asarray(edge))
    for i in range(1, geometry.owned_shape[0]):
        for j in range(geometry.owned_shape[1]):
            for k in range(geometry.owned_shape[2]):
                left = tuple(owners[i - 1, j, k])
                right = tuple(owners[i, j, k])
                if left != right:
                    expected[right] += T[i, j, k]
    np.testing.assert_allclose(np.asarray(edge), expected, rtol=2e-12, atol=2e-12)
    assert np.max(np.asarray(lowered.cells.member_count)) > 1


def test_theta_compact_conductance_is_diagonal_only():
    _geometry, _domain, lowered, _projectors, _face_bc, _config, coefficients = _fixture()
    _volume, diagonal, edge, pi, pj, pk, active = coefficients
    owners, matrix = _dense_tree(diagonal, edge, pi, pj, pk, active)
    # The line-u graph contains radial parent/child edges only; no same-ring
    # owner pair receives an off-diagonal entry.
    for row, owner in enumerate(owners):
        for column, other in enumerate(owners):
            if owner[0] == other[0] and row != column:
                assert matrix[row, column] == 0.0


def test_assembled_tree_matrix_is_symmetric_positive_definite():
    *_prefix, coefficients = _fixture()
    _volume, diagonal, edge, pi, pj, pk, active = coefficients
    _owners, matrix = _dense_tree(diagonal, edge, pi, pj, pk, active)
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=0.0)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan])
def test_missing_or_nonpositive_radial_edge_is_rejected(bad_value):
    _geometry, _domain, lowered, _projectors, _face_bc, _config, coefficients = _fixture()
    _volume, diagonal, edge, pi, pj, pk, active = coefficients
    bad = np.array(edge)
    first_child = tuple(np.argwhere(np.asarray(active) & (np.indices(active.shape)[0] > 0))[0])
    bad[first_child] = bad_value
    with pytest.raises(ValueError, match="finite positive total radial edge"):
        _validate_concrete_angular_agglomeration_tree_assembly(
            lowered, diagonal, jnp.asarray(bad), pi, pj, pk, active
        )


def test_missing_profile_and_line_uv_are_rejected_without_fallback():
    geometry, domain, lowered, projectors, face_bc, config, coefficients = _fixture()
    object.__setattr__(lowered, "angular_group_sizes", None)
    with pytest.raises(ValueError, match="angular_group_sizes"):
        _validate_concrete_angular_agglomeration_tree_assembly(
            lowered, *coefficients[1:]
        )
    object.__setattr__(lowered, "angular_group_sizes", PROFILE)
    object.__setattr__(lowered, "angular_group_sizes", (8, 2, 4, 2, 1))
    with pytest.raises(ValueError, match="nested"):
        _validate_concrete_angular_agglomeration_tree_assembly(
            lowered, *coefficients[1:]
        )
    object.__setattr__(lowered, "angular_group_sizes", PROFILE)
    with pytest.raises(ValueError, match="supports only 'none' or 'line-u'"):
        build_solvax_perp_laplacian_preconditioner(
            geometry, domain, projectors, face_bc,
            replace(config, preconditioner="line-uv"),
            control_volume_geometry=lowered,
        )
