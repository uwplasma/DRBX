"""Production RLP owner-topology and physical-volume tests."""

import numpy as np
import pytest

from drbx.geometry.fci_control_volumes import (
    build_nested_angular_group_profile,
    build_polar_angular_agglomeration_geometry,
    build_radius_dependent_angular_group_profile,
    build_radius_dependent_angular_owner_map,
)


def _grid(nu=8, ntheta=32, neta=4):
    return (
        np.linspace(0.0, 1.0, nu + 1),
        np.linspace(-np.pi, np.pi, ntheta + 1),
        np.linspace(0.0, 2.0 * np.pi, neta + 1),
    )


def test_uniform_profile_is_nested_and_divides_ntheta():
    u, theta, _ = _grid()
    profile = build_radius_dependent_angular_group_profile(u, theta)
    assert tuple(profile[:6]) == (32, 4, 4, 2, 2, 1)
    assert np.all(profile[1:] <= profile[:-1])
    assert np.all(32 % profile == 0)
    assert np.all(profile[:-1] % profile[1:] == 0)


def test_composite_theta_count_uses_nested_nonbinary_divisors():
    profile = build_nested_angular_group_profile(
        48, np.array((48.0, 10.0, 5.0, 2.5, 1.0))
    )
    assert np.array_equal(profile, np.array((48, 12, 6, 3, 1)))
    owner = build_radius_dependent_angular_owner_map(5, 48, 2, profile)
    assert np.array_equal(np.unique(owner[3, :, :, 1]), np.arange(0, 48, 3))


def test_explicit_profile_is_strictly_validated():
    u, theta, _ = _grid(nu=4)
    assert np.array_equal(
        build_radius_dependent_angular_group_profile(
            u, theta, explicit_profile=(32, 8, 4, 1)
        ),
        np.array((32, 8, 4, 1), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="non-increasing"):
        build_radius_dependent_angular_group_profile(
            u, theta, explicit_profile=(32, 2, 4, 1)
        )
    with pytest.raises(ValueError, match="divisors"):
        build_radius_dependent_angular_group_profile(
            u, theta, explicit_profile=(32, 3, 1, 1)
        )


def test_owner_map_is_idempotent_and_owner_aligned():
    q = np.array((32, 4, 4, 2, 2, 1, 1, 1), dtype=np.int32)
    owner = build_radius_dependent_angular_owner_map(8, 32, 3, q)
    assert np.array_equal(owner[tuple(np.moveaxis(owner, -1, 0))], owner)
    assert np.all(owner[1, :, :, 1] % 4 == 0)
    assert np.array_equal(np.unique(owner[0, :, :, 1]), np.array([0]))


def test_rlp_host_payload_conserves_physical_volume_and_has_no_face_fit():
    u, theta, eta = _grid()
    host = build_polar_angular_agglomeration_geometry(
        u, theta, eta, lambda points: np.ones(points.shape[:-1]),
        quadrature_order=2,
    )
    active = host.topology.is_active_owner
    assert np.isclose(host.aggregate_chart_volume[active].sum(), host.raw_volume.sum())
    owner = host.topology.owner_index
    assert np.array_equal(owner[tuple(np.moveaxis(owner, -1, 0))], owner)
    assert not hasattr(host, "face_observation_owner_index")
    assert not hasattr(host, "face_design_matrix_condition")


def test_32_cubed_production_profile_has_expected_owner_count():
    u, theta, eta = _grid(nu=32, ntheta=32, neta=32)
    host = build_polar_angular_agglomeration_geometry(
        u, theta, eta, lambda points: np.ones(points.shape[:-1]),
        quadrature_order=2,
    )
    expected = (32, 4, 4, 2, 2) + (1,) * 27
    assert tuple(host.angular_group_size) == expected
    expected_owners_per_eta = sum(32 // q for q in expected)
    assert np.count_nonzero(host.topology.is_active_owner) == 32 * expected_owners_per_eta
    assert np.all(host.raw_volume > 0.0)
