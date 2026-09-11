"""Regression for curvature transport across internal aggregate-owner faces."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_fci_projected_fine_grid_control_volume as projected_fixture
from drbx.geometry import (
    LocalCurvatureFaceCoefficients3D,
    build_local_conservative_stencil_from_field,
)
from drbx.geometry.fci_control_volumes import (
    build_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_halo import LocalHaloClosure3D
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import (
    aggregate_local_control_volume_average,
    expand_local_control_volume_owner_field,
    local_curvature_production_path_op,
)


def _owner_consistency_case(monkeypatch, ntheta: int, group_size: int = 2):
    """Return the smooth-mode error and fitted production-action multiplier."""

    def host(shape):
        radial_faces = np.linspace(0.0, 1.0, shape[0] + 1)
        theta_faces = np.linspace(0.0, 2.0 * np.pi, shape[1] + 1)
        eta_faces = np.linspace(0.0, 2.0 * np.pi, shape[2] + 1)
        return build_polar_angular_agglomeration_geometry(
            radial_faces,
            theta_faces,
            eta_faces,
            lambda points: np.maximum(
                np.asarray(points)[..., 0], 1.0e-14
            ),
            quadrature_order=2,
            angular_group_size=(ntheta, group_size, group_size, group_size),
        )

    monkeypatch.setattr(projected_fixture, "_host", host)
    (
        geometry,
        domain,
        context,
        exchange,
        scalar_filler,
        control_volume,
        _boundary_bc,
        face_bc,
        solver,
        _active,
        _weights,
    ) = projected_fixture._setup(shape=(4, ntheta, 4))

    theta = jnp.asarray(geometry.grid.y_centers_owned, dtype=jnp.float64)
    theta_3d = jnp.broadcast_to(
        theta[None, :, None], geometry.owned_shape
    )
    fine_fields = (
        1.0 + 1.0e-3 * jnp.sin(theta_3d),
        jnp.ones(geometry.owned_shape, dtype=jnp.float64),
        jnp.ones(geometry.owned_shape, dtype=jnp.float64),
        jnp.zeros(geometry.owned_shape, dtype=jnp.float64),
    )
    closure = LocalHaloClosure3D(
        physical_ghost_filler=solver.physical_ghost_filler,
        halo_exchange=exchange,
        topology_filler=scalar_filler,
    )

    stencils = []
    for fine in fine_fields:
        owner = aggregate_local_control_volume_average(
            fine, control_volume.cells, domain
        )
        expanded = expand_local_control_volume_owner_field(
            owner, control_volume.cells
        )
        halo = inject_owned_field_to_halo(expanded, geometry.layout)
        halo = closure(halo, domain, face_bc)
        stencils.append(
            build_local_conservative_stencil_from_field(
                halo, geometry, context
            )
        )

    zeros_x = jnp.zeros(
        geometry.layout.face_control_shape(0), dtype=jnp.float64
    )
    zeros_z = jnp.zeros(
        geometry.layout.face_control_shape(2), dtype=jnp.float64
    )
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=geometry.layout,
        x=zeros_x,
        y=jnp.asarray(geometry.face_metric.y.J_owned, dtype=jnp.float64),
        z=zeros_z,
    )
    result = local_curvature_production_path_op(
        tuple(stencils),
        geometry,
        coefficients,
        tau=1.0,
        domain=domain,
        control_volume_geometry=control_volume,
    )

    target_fine = 2.0e-3 * jnp.cos(theta_3d)
    target_owner = aggregate_local_control_volume_average(
        target_fine, control_volume.cells, domain
    )
    target = expand_local_control_volume_owner_field(
        target_owner, control_volume.cells
    )
    selected = slice(2 * group_size, ntheta - 2 * group_size)
    actual_line = np.asarray(result[2, selected, 0, 0], dtype=float)
    target_line = np.asarray(target[2, selected, 0], dtype=float)
    multiplier = float(np.dot(actual_line, target_line) / np.dot(target_line, target_line))
    relative_error = float(
        np.linalg.norm(actual_line - target_line) / np.linalg.norm(target_line)
    )
    return relative_error, multiplier


def test_third_order_same_owner_face_jumps_restore_smooth_owner_transport(monkeypatch):
    coarse_error, coarse_multiplier = _owner_consistency_case(monkeypatch, 32)
    fine_error, fine_multiplier = _owner_consistency_case(monkeypatch, 64)

    # Omitting internal same-owner jumps drives these multipliers toward 4/3
    # and makes the error grow under refinement.  The complete fluctuation
    # balance instead converges toward a unit multiplier.
    assert abs(coarse_multiplier - 1.0) < 0.04
    assert abs(fine_multiplier - 1.0) < 0.015
    assert fine_error < 0.6 * coarse_error
