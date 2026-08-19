"""Focused tests for eta-only RLP line-u coefficient assembly."""

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
from drbx.native.fci_angular_agglomeration import (
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_operators import (
    _assemble_angular_agglomeration_tree_principal_coefficients,
    _lift_cell_field_to_faces,
    build_local_perp_laplacian_face_projectors,
)
from test_fci_operators_domain_decomp import _build_domain


PROFILE = (8, 4, 4, 2, 1)


def _case(*, nz: int, shard_counts: tuple[int, int, int]):
    shape = (len(PROFILE), PROFILE[0], nz)
    geometry, *_ = polar_fixture(shape=shape, halo_width=1)
    global_shape = tuple(
        size * count for size, count in zip(shape, shard_counts)
    )
    domain = _build_domain(global_shape, 1, shard_counts)
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    host = build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        lambda p: np.ones(p.shape[:-1]),
        angular_group_size=PROFILE,
        quadrature_order=2,
    )
    # This test targets the line-u assembly.  The local topology is enough to
    # validate its owner-space coefficients; the eta slab boundary is supplied
    # by the domain shard metadata.
    lowered = lower_polar_angular_agglomeration_geometry(host, geometry)
    projectors = build_local_perp_laplacian_face_projectors(
        geometry, domain, axis_regular_axes=(False, False, False)
    )
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    config = SolvaxGmresConfig(
        regularization_epsilon=1.0e-3, preconditioner="line-u"
    )
    return geometry, domain, lowered, projectors, face_bc, config


def test_rlp_line_u_accepts_eta_sharding_and_rejects_x_theta_sharding():
    geometry, domain, lowered, projectors, face_bc, config = _case(
        nz=3, shard_counts=(1, 1, 2)
    )
    coefficients = _assemble_angular_agglomeration_tree_principal_coefficients(
        geometry, domain, projectors, face_bc, config, lowered
    )
    assert all(value.shape == geometry.owned_shape for value in coefficients[:3])

    for bad_counts in ((2, 1, 1), (1, 2, 1)):
        bad_domain = replace(
            domain,
            shard_spec=replace(domain.shard_spec, shard_counts=bad_counts),
        )
        with pytest.raises(ValueError, match="eta-only sharding"):
            _assemble_angular_agglomeration_tree_principal_coefficients(
                geometry, bad_domain, projectors, face_bc, config, lowered
            )


def test_eta_slab_endpoints_are_not_treated_as_periodic():
    geometry, domain, lowered, projectors, face_bc, config = _case(
        nz=3, shard_counts=(1, 1, 2)
    )
    base_geometry = geometry
    # Retain the ordinary perpendicular x/y projectors for a valid radial
    # tree, but make the z projector identity so the eta diagonal is visible
    # in this isolated coefficient comparison.
    projectors = (
        projectors[0],
        projectors[1],
        jnp.broadcast_to(jnp.eye(3), projectors[2].shape),
    )
    baseline = _assemble_angular_agglomeration_tree_principal_coefficients(
        base_geometry, domain, projectors, face_bc, config, lowered
    )
    # Make the owned z-face conductances visibly nonuniform.  With the polar
    # fixture, P_zz, areas, fractions, and logical area are unity, so changing
    # J isolates the eta diagonal contribution.
    z_metric = geometry.face_metric.z
    shape = z_metric.J_halo.shape
    values = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) + 1.0
    geometry = replace(
        geometry,
        face_metric=replace(
            geometry.face_metric,
            z=replace(z_metric, J_halo=values),
        ),
    )
    volume, diagonal, _edge, _pi, _pj, _pk, active = (
        _assemble_angular_agglomeration_tree_principal_coefficients(
            geometry, domain, projectors, face_bc, config, lowered
        )
    )

    # The local z-face payload has nz+1 faces.  Each local owner receives the
    # conductance of its left and right face; no modulo-nz wrap is allowed.
    modified_J = np.asarray(geometry.face_metric.z.J_owned)
    baseline_J = np.asarray(base_geometry.face_metric.z.J_owned)
    # The remaining geometric/projector factors are unchanged, so recover the
    # actual conductance delta from the baseline coefficient multiplier.
    # The polar fixture has a positive, uniform z projector; obtain the
    # non-J conductance factor from the same assembly inputs directly.
    # ``J`` is the only field changed by this test.
    regular = base_geometry.regular_face_geometry
    h = base_geometry.layout.halo_width
    nz = base_geometry.owned_shape[2]
    center_distance = (
        np.asarray(base_geometry.grid.z.centers_halo[h:h + nz + 1])
        - np.asarray(base_geometry.grid.z.centers_halo[h - 1:h + nz])
    )
    logical_area = np.asarray(_lift_cell_field_to_faces(
        base_geometry.spacing.dx_owned * base_geometry.spacing.dy_owned,
        axis=2,
        periodic=False,
    ))
    factor = (
        np.asarray(projectors[2][..., 2, 2])
        * np.asarray(regular.z_area)
        * np.asarray(regular.z_area_fraction)
        * np.asarray(regular.z_open_mask)
        * logical_area
        / np.maximum(
            center_distance,
            1.0e-30,
        )[None, None, :]
    )
    baseline_Tz = baseline_J * factor
    modified_Tz = modified_J * factor
    expected = np.zeros(geometry.owned_shape)
    delta_Tz = modified_Tz - baseline_Tz
    for i, q in enumerate(PROFILE):
        for j in range(0, geometry.owned_shape[1], q):
            for k in range(geometry.owned_shape[2]):
                expected[i, j, k] = np.sum(
                    delta_Tz[i, j:j + q, k]
                    + delta_Tz[i, j:j + q, k + 1]
                )
    np.testing.assert_allclose(
        (np.asarray(diagonal) - np.asarray(baseline[1]))[np.asarray(active)],
        expected[np.asarray(active)],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_rlp_coefficient_construction_is_jit_trace_safe():
    geometry, domain, lowered, projectors, face_bc, config = _case(
        nz=3, shard_counts=(1, 1, 1)
    )

    @jax.jit
    def construct(scale):
        scaled_projectors = tuple(value * scale for value in projectors)
        coefficients = _assemble_angular_agglomeration_tree_principal_coefficients(
            geometry, domain, scaled_projectors, face_bc, config, lowered
        )
        return coefficients[1]

    diagonal = np.asarray(construct(jnp.asarray(1.0)))
    assert diagonal.shape == geometry.owned_shape
    assert np.all(np.isfinite(diagonal))
