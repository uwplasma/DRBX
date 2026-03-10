from __future__ import annotations

import numpy as np

from jaxdrb.hermes_literal.communicate import (
    build_parallel_subdomain,
    slice_parallel_subdomain_2d,
    slice_parallel_subdomain_3d,
)


def test_slice_parallel_subdomain_3d_copies_internal_edge_planes_by_default() -> None:
    field = np.arange(12 * 3 * 2, dtype=np.float64).reshape(12, 3, 2)
    subdomain = build_parallel_subdomain(start=4, stop=8, open_field_line=True)
    local = np.asarray(
        slice_parallel_subdomain_3d(
            field,
            subdomain=subdomain,
            periodic_parallel=False,
            lower_boundary_open=True,
            upper_boundary_open=True,
        )
    )
    np.testing.assert_allclose(local[0], field[4])
    np.testing.assert_allclose(local[1], field[4])
    np.testing.assert_allclose(local[2:6], field[4:8])
    np.testing.assert_allclose(local[6], field[7])
    np.testing.assert_allclose(local[7], field[7])


def test_slice_parallel_subdomain_3d_can_use_neighbor_planes_when_requested() -> None:
    field = np.arange(12 * 3 * 2, dtype=np.float64).reshape(12, 3, 2)
    subdomain = build_parallel_subdomain(start=4, stop=8, open_field_line=True)
    local = np.asarray(
        slice_parallel_subdomain_3d(
            field,
            subdomain=subdomain,
            periodic_parallel=False,
            lower_boundary_open=True,
            upper_boundary_open=True,
            neighbor_planes=True,
        )
    )
    np.testing.assert_allclose(local[0], field[2])
    np.testing.assert_allclose(local[1], field[3])
    np.testing.assert_allclose(local[2:6], field[4:8])
    np.testing.assert_allclose(local[6], field[8])
    np.testing.assert_allclose(local[7], field[9])


def test_slice_parallel_subdomain_3d_applies_open_boundary_neumann_at_global_end() -> None:
    field = np.arange(6 * 2 * 2, dtype=np.float64).reshape(6, 2, 2)
    subdomain = build_parallel_subdomain(start=0, stop=3, open_field_line=True)
    local = np.asarray(
        slice_parallel_subdomain_3d(
            field,
            subdomain=subdomain,
            periodic_parallel=False,
            lower_boundary_open=True,
            upper_boundary_open=True,
        )
    )
    np.testing.assert_allclose(local[2:5], field[:3])
    np.testing.assert_allclose(local[1], local[2])
    np.testing.assert_allclose(local[0], local[1])
    np.testing.assert_allclose(local[5], field[2])
    np.testing.assert_allclose(local[6], field[2])


def test_slice_parallel_subdomain_2d_wraps_periodic_parallel_guards() -> None:
    field = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)
    subdomain = build_parallel_subdomain(start=1, stop=4, open_field_line=False)
    local = np.asarray(
        slice_parallel_subdomain_2d(
            field,
            subdomain=subdomain,
            periodic_parallel=True,
            lower_boundary_open=False,
            upper_boundary_open=False,
        )
    )
    np.testing.assert_allclose(local[0], field[4])
    np.testing.assert_allclose(local[1], field[0])
    np.testing.assert_allclose(local[2:5], field[1:4])
    np.testing.assert_allclose(local[5], field[4])
    np.testing.assert_allclose(local[6], field[0])
