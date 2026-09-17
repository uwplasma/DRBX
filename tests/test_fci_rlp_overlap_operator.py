"""Discrete invariants for pairwise owner-overlap parallel diffusion."""

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_rlp_overlap import (
    assemble_rlp_parallel_overlap_generator,
    audit_rlp_parallel_overlap_generator,
    local_parallel_diffusion_fci_rlp_overlap_op,
    lower_rlp_parallel_overlap_geometry,
)


def _ring_geometry(*, volume=(1.0, 2.0, 3.0, 4.0)):
    return lower_rlp_parallel_overlap_geometry(
        {
            "raw_shape": (4, 1, 1),
            "owner_flat_ids": np.arange(4),
            "owner_volume": np.asarray(volume),
            "link_owner_a": np.arange(4),
            "link_owner_b": np.roll(np.arange(4), -1),
            "transmissibility": np.asarray((0.7, 1.1, 0.4, 0.9)),
        }
    )


def test_pairwise_generator_has_conservation_mmatrix_and_energy_invariants():
    geometry = _ring_geometry()
    matrix = assemble_rlp_parallel_overlap_generator(geometry)
    dense = matrix.toarray()
    volume = np.asarray(geometry.owner_volume)

    np.testing.assert_allclose(dense @ np.ones(4), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(volume @ dense, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(volume[:, None] * dense, (volume[:, None] * dense).T)
    off_diagonal = dense[~np.eye(4, dtype=bool)]
    assert np.min(off_diagonal) >= 0.0
    np.testing.assert_allclose(np.diag(dense), -np.sum(dense - np.diag(np.diag(dense)), axis=1))

    rng = np.random.default_rng(42)
    for _ in range(8):
        values = rng.normal(size=4)
        assert float(values @ (volume[:, None] * dense) @ values) <= 1.0e-13
        minimum = int(np.argmin(values))
        assert float((dense @ values)[minimum]) >= -1.0e-13

    audit = audit_rlp_parallel_overlap_generator(geometry)
    assert audit["constant_residual"] <= 1.0e-14
    assert audit["weighted_conservation_residual"] <= 1.0e-14
    assert audit["weighted_symmetry_residual"] <= 1.0e-14
    assert audit["minimum_offdiagonal"] >= 0.0


def test_operator_is_jittable_and_leaves_alias_slots_exactly_zero():
    geometry = lower_rlp_parallel_overlap_geometry(
        {
            "raw_shape": (6, 1, 1),
            "owner_flat_ids": np.asarray((0, 2, 4)),
            "owner_volume": np.asarray((1.0, 1.5, 2.0)),
            "link_owner_a": np.asarray((0, 1, 2)),
            "link_owner_b": np.asarray((1, 2, 0)),
            "transmissibility": np.asarray((1.0, 0.5, 0.25)),
        }
    )
    values = jnp.asarray((2.0, 99.0, 1.0, -77.0, 3.0, 8.0)).reshape(6, 1, 1)
    eager = local_parallel_diffusion_fci_rlp_overlap_op(values, geometry, 0.3)
    compiled = jax.jit(local_parallel_diffusion_fci_rlp_overlap_op)(
        values, geometry, 0.3
    )
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager))
    assert np.all(np.asarray(eager).reshape(-1)[[1, 3, 5]] == 0.0)
    compact_rhs = np.asarray(eager).reshape(-1)[[0, 2, 4]]
    np.testing.assert_allclose(
        np.dot(np.asarray(geometry.owner_volume), compact_rhs), 0.0, atol=1.0e-14
    )


def test_uniform_one_to_one_ring_is_standard_finite_volume_laplacian():
    spacing = 0.25
    geometry = lower_rlp_parallel_overlap_geometry(
        {
            "raw_shape": (1, 1, 4),
            "owner_flat_ids": np.arange(4),
            "owner_volume": np.full(4, spacing),
            "link_owner_a": np.arange(4),
            "link_owner_b": np.roll(np.arange(4), -1),
            "transmissibility": np.full(4, 1.0 / spacing),
        }
    )
    values = jnp.asarray((1.0, 2.0, 4.0, 3.0)).reshape(1, 1, 4)
    actual = np.asarray(
        local_parallel_diffusion_fci_rlp_overlap_op(values, geometry, 1.0)
    ).reshape(-1)
    flat = np.asarray(values).reshape(-1)
    expected = (np.roll(flat, 1) - 2.0 * flat + np.roll(flat, -1)) / spacing**2
    np.testing.assert_allclose(actual, expected)
