"""Focused tests for the production RLP cell-average prolongation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.fci_control_volumes import (
    build_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_control_volume_operators import (
    control_volume_average_basis,
    monomial_exponents,
)
from drbx.native.fci_rlp_diffusion import (
    apply_rlp_cell_average_prolongation,
    compile_rlp_cell_average_prolongation,
)


def _host(shape=(4, 8, 4)):
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=3,
        angular_group_size=(shape[1], 2, 1, 1)[: shape[0]],
    )


def _warped_intermediate_q2_host():
    """Small deterministic analogue of the ill-poised HSX q=2 rings."""

    shape = (12, 32, 32)
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)

    def jacobian(points):
        points = np.asarray(points)
        radial = np.maximum(points[..., 0], 1.0e-14)
        return radial * (
            1.0 + 0.45 * np.sin(points[..., 1] + 0.7 * points[..., 2])
        )

    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        jacobian,
        quadrature_order=3,
        angular_group_size=(32, 8, 8, 4, 4, 2, 2, 2, 2, 2, 2, 1),
    )


def _warped_refined_q2_host():
    """Refined analogue whose 24-planar-donor stencil exceeds the H bound."""

    shape = (16, 48, 16)
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)

    def jacobian(points):
        points = np.asarray(points)
        radial = np.maximum(points[..., 0], 1.0e-14)
        return radial * (
            1.0 + 0.45 * np.sin(points[..., 1] + 0.7 * points[..., 2])
        )

    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        jacobian,
        quadrature_order=3,
        angular_group_size=(
            48, 16, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 1,
        ),
    )


def test_compiler_emits_bounded_relative_eta_sparse_contract():
    host = _host()
    prolongation = compile_rlp_cell_average_prolongation(host)
    expected_rows = int(
        np.prod(host.topology.shape[1:])
        * np.count_nonzero(host.angular_group_size > 1)
    )
    assert prolongation.row_count == expected_rows
    assert prolongation.max_observations == 160
    assert prolongation.owner_i.shape == (expected_rows, 160)
    assert prolongation.owner_j.shape == (expected_rows, 160)
    assert prolongation.owner_eta_offset.shape == (expected_rows, 160)
    assert np.max(np.abs(np.asarray(prolongation.owner_eta_offset))) <= 2
    assert np.all(np.asarray(prolongation.raw_row_active))
    raw_rows = np.column_stack(
        (
            np.asarray(prolongation.raw_i),
            np.asarray(prolongation.raw_j),
            np.asarray(prolongation.raw_k),
        )
    )
    assert np.unique(raw_rows, axis=0).shape == raw_rows.shape
    np.testing.assert_array_equal(
        np.asarray(prolongation.reconstructed_raw_mask),
        np.broadcast_to(
            host.angular_group_size[:, None, None] > 1,
            host.topology.shape,
        ),
    )
    assert prolongation.diagnostics.minimum_rank == 20
    assert prolongation.diagnostics.maximum_reproduction_residual < 2.0e-12
    assert prolongation.diagnostics.maximum_constant_residual < 2.0e-14
    assert prolongation.diagnostics.maximum_conservation_residual < 2.0e-14
    assert prolongation.diagnostics.maximum_eta_line_residual < 2.0e-14
    assert prolongation.diagnostics.maximum_row_weight_l1_norm == np.max(
        np.sum(np.abs(np.asarray(prolongation.weights)), axis=1)
    )


def test_prolongation_preserves_arbitrary_eta_line_exactly():
    # Six planes exceed the cubic eta basis, so this catches leakage that a
    # four-plane polynomial interpolation can hide.
    host = _host((4, 8, 6))
    prolongation = compile_rlp_cell_average_prolongation(host)
    eta_line = np.asarray(
        (0.37, -1.25, 2.0, 0.08, -0.42, 1.31),
        dtype=np.float64,
    )
    assert prolongation.diagnostics.maximum_eta_line_residual < 2.0e-14
    owner_field = np.where(
        host.topology.is_active_owner,
        eta_line[None, None, :],
        0.0,
    )
    fine = jax.jit(apply_rlp_cell_average_prolongation)(
        jnp.asarray(owner_field),
        prolongation,
    )
    np.testing.assert_allclose(
        np.asarray(fine),
        np.broadcast_to(eta_line[None, None, :], host.topology.shape),
        rtol=0.0,
        atol=3.0e-13,
    )


def test_single_eta_plane_uses_exact_two_dimensional_cubic_basis():
    host = _host((4, 8, 1))
    prolongation = compile_rlp_cell_average_prolongation(host)
    assert prolongation.diagnostics.polynomial_degree == 3
    assert prolongation.diagnostics.eta_polynomial_degree == 0
    assert prolongation.diagnostics.basis_size == 10
    assert prolongation.diagnostics.minimum_rank == 10
    assert prolongation.diagnostics.maximum_reproduction_residual < 2.0e-12
    np.testing.assert_array_equal(
        np.asarray(prolongation.owner_eta_offset),
        np.zeros_like(np.asarray(prolongation.owner_eta_offset)),
    )


def test_prolongation_reproduces_raw_xy_cubic_averages_and_q1_identity():
    host = _host()
    prolongation = compile_rlp_cell_average_prolongation(host)
    exponents = tuple(power for power in monomial_exponents(3) if power[2] == 0)
    coefficients = np.linspace(-0.7, 0.9, len(exponents))
    owner_basis = control_volume_average_basis(
        host.aggregate_chart_centroid,
        host.aggregate_chart_second_moment,
        host.aggregate_chart_third_moment,
        exponents=exponents,
    )
    raw_basis = control_volume_average_basis(
        host.raw_chart_centroid,
        host.raw_chart_second_moment,
        host.raw_chart_third_moment,
        exponents=exponents,
    )
    expected = raw_basis @ coefficients
    owner_field = np.where(
        host.topology.is_active_owner,
        owner_basis @ coefficients,
        0.0,
    )
    got = jax.jit(apply_rlp_cell_average_prolongation)(
        jnp.asarray(owner_field), prolongation
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=3.0e-12, atol=3.0e-12)
    q1 = np.broadcast_to(
        host.angular_group_size[:, None, None] == 1,
        host.topology.shape,
    )
    np.testing.assert_array_equal(np.asarray(got)[q1], owner_field[q1])


def test_prolongation_preserves_owner_means_and_has_autodiff_transpose():
    host = _host()
    prolongation = compile_rlp_cell_average_prolongation(host)
    rng = np.random.default_rng(9281)
    owner_field = np.where(
        host.topology.is_active_owner,
        rng.normal(size=host.topology.shape),
        0.0,
    )
    fine = np.asarray(
        apply_rlp_cell_average_prolongation(jnp.asarray(owner_field), prolongation)
    )
    q = np.asarray(host.angular_group_size)
    for i in np.flatnonzero(q > 1):
        for j in range(0, host.topology.shape[1], int(q[i])):
            for k in range(host.topology.shape[2]):
                member_j = np.arange(j, j + int(q[i]))
                volume = host.raw_volume[i, member_j, k]
                restricted = np.sum(volume * fine[i, member_j, k]) / np.sum(volume)
                np.testing.assert_allclose(
                    restricted, owner_field[i, j, k], rtol=2.0e-13, atol=2.0e-13
                )

    cotangent = rng.normal(size=host.topology.shape)
    transpose = jax.linear_transpose(
        lambda values: apply_rlp_cell_average_prolongation(values, prolongation),
        jnp.asarray(owner_field),
    )(jnp.asarray(cotangent))[0]
    expected_transpose = np.where(
        np.asarray(prolongation.reconstructed_raw_mask), 0.0, cotangent
    )
    row_cotangent = cotangent[
        np.asarray(prolongation.raw_i),
        np.asarray(prolongation.raw_j),
        np.asarray(prolongation.raw_k),
    ]
    active = np.asarray(prolongation.observation_active)
    contribution = row_cotangent[:, None] * np.asarray(prolongation.weights)
    np.add.at(
        expected_transpose,
        (
            np.asarray(prolongation.owner_i)[active],
            np.asarray(prolongation.owner_j)[active],
            np.asarray(prolongation.owner_k)[active],
        ),
        contribution[active],
    )
    np.testing.assert_allclose(
        np.asarray(transpose), expected_transpose, rtol=2.0e-13, atol=2.0e-13
    )


def test_oversampled_cubic_stencil_bounds_warped_intermediate_q2_weights():
    host = _warped_intermediate_q2_host()

    # This fixture retains the failure signature found on the intermediate
    # q=2 rings of the cached HSX N32 host with the old 12-planar-donor fit.
    legacy = compile_rlp_cell_average_prolongation(
        host,
        max_observations=64,
        row_weight_l1_limit=None,
    )
    assert legacy.diagnostics.maximum_row_weight_l1_norm > 1.0e3

    prolongation = compile_rlp_cell_average_prolongation(host)
    weights = np.asarray(prolongation.weights)
    row_l1 = np.sum(np.abs(weights), axis=1)
    assert prolongation.max_observations == 160
    assert prolongation.diagnostics.maximum_row_weight_l1_norm == np.max(row_l1)
    assert prolongation.diagnostics.maximum_row_weight_l1_norm < 7.0
    assert prolongation.diagnostics.maximum_reproduction_residual < 1.0e-12
    assert prolongation.diagnostics.maximum_conservation_residual < 1.0e-14
    assert prolongation.diagnostics.maximum_eta_line_residual < 5.0e-14

    # Smooth planar cubics exercise the same target subcell reconstruction
    # that produced the large HSX center errors, including every q=2 ring.
    exponents = tuple(
        power for power in monomial_exponents(3) if power[2] == 0
    )
    coefficients = np.linspace(-0.7, 0.9, len(exponents))
    owner_basis = control_volume_average_basis(
        host.aggregate_chart_centroid,
        host.aggregate_chart_second_moment,
        host.aggregate_chart_third_moment,
        exponents=exponents,
    )
    raw_basis = control_volume_average_basis(
        host.raw_chart_centroid,
        host.raw_chart_second_moment,
        host.raw_chart_third_moment,
        exponents=exponents,
    )
    owner_field = np.where(
        host.topology.is_active_owner,
        owner_basis @ coefficients,
        0.0,
    )
    fine = np.asarray(
        apply_rlp_cell_average_prolongation(owner_field, prolongation)
    )
    np.testing.assert_allclose(
        fine, raw_basis @ coefficients, rtol=2.0e-12, atol=2.0e-12
    )

    q = np.asarray(host.angular_group_size)
    for i in np.flatnonzero(q > 1):
        for j in range(0, host.topology.shape[1], int(q[i])):
            for k in range(host.topology.shape[2]):
                member_j = np.arange(j, j + int(q[i]))
                volume = host.raw_volume[i, member_j, k]
                restricted = np.sum(volume * fine[i, member_j, k]) / np.sum(volume)
                np.testing.assert_allclose(
                    restricted, owner_field[i, j, k], rtol=2.0e-13, atol=2.0e-13
                )

    eta_line = np.sin(1.7 * np.arange(host.topology.shape[2]))
    eta_owner = np.where(
        host.topology.is_active_owner,
        eta_line[None, None, :],
        0.0,
    )
    eta_fine = np.asarray(
        apply_rlp_cell_average_prolongation(eta_owner, prolongation)
    )
    np.testing.assert_allclose(
        eta_fine,
        np.broadcast_to(eta_line[None, None, :], host.topology.shape),
        rtol=0.0,
        atol=5.0e-13,
    )


def test_adaptive_donors_expand_and_reject_an_unstable_storage_cap():
    host = _warped_intermediate_q2_host()

    with pytest.raises(ValueError, match="no stable donor stencil"):
        compile_rlp_cell_average_prolongation(host, max_observations=64)

    prolongation = compile_rlp_cell_average_prolongation(
        host,
        max_observations=120,
        minimum_planar_donors=12,
    )
    diagnostics = prolongation.diagnostics
    assert diagnostics.row_weight_l1_limit == 8.0
    assert diagnostics.expanded_planar_owner_count > 0
    assert diagnostics.minimum_planar_donor_count >= 12
    assert diagnostics.maximum_planar_donor_count > 12
    assert diagnostics.maximum_row_weight_l1_norm <= 8.0


def test_default_adaptive_contract_expands_refined_q2_stencils():
    host = _warped_refined_q2_host()
    fixed_24 = compile_rlp_cell_average_prolongation(
        host,
        max_observations=120,
        row_weight_l1_limit=None,
    )
    assert fixed_24.diagnostics.maximum_row_weight_l1_norm > 8.0

    adaptive = compile_rlp_cell_average_prolongation(host)
    diagnostics = adaptive.diagnostics
    assert diagnostics.expanded_planar_owner_count > 0
    assert diagnostics.minimum_planar_donor_count == 24
    assert diagnostics.maximum_planar_donor_count == 28
    assert diagnostics.maximum_row_weight_l1_norm <= 8.0
