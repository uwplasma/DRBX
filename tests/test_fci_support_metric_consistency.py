"""Metric contracts for the support-paired perpendicular operator.

These tests use the same coordinate-vector convention as the FCI support
operator: ``d`` is the raw coordinate derivative and the face metric enters
once as ``P = g^contra - b b``.  Keeping this small periodic reference here
also makes the expected continuum normalization unambiguous, independently
of the angular-RLP fixture and its topology machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from drbx.native.fci_boundaries import LocalBoundaryFaceBC3D


def test_unmocked_production_gradient_applies_metric_once():
    """The production contraction is ``D.T W (cP) D``, not ``D.T W P² D``.

    The real angular-RLP fixture supplies the geometry, reconstructed gradient,
    face packing, quadrature masses, and weighted adjoint.
    """

    from dataclasses import replace
    scale = 0.37
    from pathlib import Path
    import sys

    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from test_fci_projected_fine_grid_control_volume import _apply, _setup

    data = _setup(shape=(3, 8, 4))
    geometry, _domain, _context, _exchange, _scalar, _lowered, boundary_bc, face_bc, solver, active, _weights = data
    regular = geometry.regular_face_geometry
    face_shapes = (
        np.asarray(regular.x_open_mask).shape,
        np.asarray(regular.y_open_mask).shape,
        np.asarray(regular.z_open_mask).shape,
    )

    import jax.numpy as jnp
    identity = tuple(
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), shape + (3, 3))
        for shape in face_shapes
    )
    scaled = tuple(
        jnp.broadcast_to(
            scale * jnp.eye(3, dtype=jnp.float64),
            shape + (3, 3),
        )
        for shape in face_shapes
    )
    solver_identity = replace(
        solver, operator_form="support-paired", face_projectors=identity
    )
    solver_scaled = replace(
        solver, operator_form="support-paired", face_projectors=scaled
    )
    # A smooth unmocked angular mode exercises the actual FCI reconstruction.
    theta = np.arange(active.shape[1], dtype=float)[None, :, None]
    values = np.broadcast_to(np.sin(2.0 * np.pi * theta / active.shape[1]), active.shape).copy()
    values[~active] = 0.0
    reference = _apply(solver_identity, face_bc, boundary_bc, values)
    actual = _apply(solver_scaled, face_bc, boundary_bc, values)
    mass = np.asarray(_weights, dtype=float)
    reference_energy = float(np.dot(values[active] * mass[active], reference[active]))
    actual_energy = float(np.dot(values[active] * mass[active], actual[active]))
    assert reference_energy > 1.0e-14
    np.testing.assert_allclose(
        actual_energy / reference_energy, scale, rtol=2e-10, atol=2e-11
    )


@pytest.mark.parametrize("ntheta", [8, 16])
def test_cartesian_periodic_absolute_rayleigh_normalization(ntheta):
    """Measure the unmocked constant-metric Fourier quotient on a small grid.

    This intentionally records the absolute quotient against the continuum
    ``k^2``.  It is a diagnostic guard for face-family/endpoint normalization;
    unlike the scalar-projector test above it does not compare two production
    calls to one another.
    """

    from dataclasses import replace
    from pathlib import Path
    import sys

    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from test_fci_projected_fine_grid_control_volume import _apply, _setup
    import jax.numpy as jnp

    data = _setup(shape=(3, ntheta, 4))
    geometry, domain, _context, _exchange, _scalar, _lowered, boundary_bc, face_bc, solver, active, weights = data

    def identity_metric(metric):
        one = jnp.ones_like(metric.J_halo)
        zero = jnp.zeros_like(metric.J_halo)
        return replace(
            metric, J_halo=one, g11_halo=one, g22_halo=one, g33_halo=one,
            g12_halo=zero, g13_halo=zero, g23_halo=zero,
            g_11_halo=one, g_22_halo=one, g_33_halo=one,
            g_12_halo=zero, g_13_halo=zero, g_23_halo=zero,
        )

    def axial_bfield(field):
        b = jnp.zeros_like(field.B_contra_halo).at[..., 2].set(1.0)
        return replace(field, B_contra_halo=b, Bmag_halo=jnp.ones_like(field.Bmag_halo))

    face_metric = replace(
        geometry.face_metric,
        x=identity_metric(geometry.face_metric.x),
        y=identity_metric(geometry.face_metric.y),
        z=identity_metric(geometry.face_metric.z),
    )
    face_bfield = replace(
        geometry.face_bfield,
        x=axial_bfield(geometry.face_bfield.x),
        y=axial_bfield(geometry.face_bfield.y),
        z=axial_bfield(geometry.face_bfield.z),
    )
    cart_geometry = replace(
        geometry,
        cell_metric=identity_metric(geometry.cell_metric),
        face_metric=face_metric,
        cell_bfield=axial_bfield(geometry.cell_bfield),
        face_bfield=face_bfield,
        cell_volume_geometry=replace(
            geometry.cell_volume_geometry,
            volume=jnp.ones_like(geometry.cell_volume_geometry.volume),
        ),
    )
    from drbx.geometry import SIDE_SIMPLE_PERIODIC, StencilBuilderContext
    from drbx.native.fci_halo import TopologyHaloFiller3D, LocalPeriodicTopologyRule3D
    domain = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec, periodic_axes=(True, True, True),
            axis_regular_axes=(False, False, False),
            side_kind_lower=(SIDE_SIMPLE_PERIODIC,) * 3,
            side_kind_upper=(SIDE_SIMPLE_PERIODIC,) * 3,
        ),
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    face_bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    cart_solver = replace(
        solver, geometry=cart_geometry, domain=domain, stencil_builder_context=context,
        control_volume_geometry=None,
        control_volume_boundary_bc=None,
        operator_form="support-paired", axis_regular_axes=(False, False, False),
        face_projectors=None, physical_ghost_filler=None,
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
    )
    cart_active, cart_weights = cart_solver._operator_mass_weights()
    cart_active = np.asarray(cart_active, dtype=bool)
    ntheta = cart_active.shape[1]
    theta = np.arange(ntheta, dtype=float)[None, :, None]
    values = np.broadcast_to(np.sin(2.0 * np.pi * theta / ntheta), cart_active.shape).copy()
    values[~cart_active] = 0.0
    result = _apply(cart_solver, face_bc, None, values)
    mass = np.asarray(cart_weights, dtype=float)
    quotient = float(np.dot(values[cart_active] * mass[cart_active], result[cart_active]) / np.dot(values[cart_active] * mass[cart_active], values[cart_active]))
    h = 2.0 * np.pi / ntheta
    expected = (4.0 * np.sin(h / 2.0) ** 2 + 2.0 * np.sin(h) ** 2) / (3.0 * h**2)
    print(f"cartesian support Rayleigh = {quotient:.12g}; expected = {expected:.12g}")
    np.testing.assert_allclose(quotient, expected, rtol=3.0e-10, atol=3.0e-10)
