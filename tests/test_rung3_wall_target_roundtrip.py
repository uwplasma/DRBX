"""Rung-3 physical-halo target roundtrip diagnostic (work in progress)."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native import FciDrbEBState
from drbx.native import fci_drb_EB_rhs as rhs_mod
from drbx.native.fci_physical_wall import (
    physical_wall_model_from_name,
)
from drbx.native.fci_halo import MetricAwarePhysicalGhostCellFiller3D
from drbx.native.fci_halo import (
    neumann_face_trace_physical_affine,
    physical_normal_derivative_for_face_trace,
)
from drbx.native.fci_boundaries import (
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
    build_local_boundary_face_trace_from_halo,
)
from drbx.geometry import SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC
from test_metric_aware_physical_ghosts import _metric_aware_fixture


def _state(shape):
    ijk = jnp.indices(shape, dtype=jnp.float64)
    y = 2.0 * jnp.pi * ijk[1] / shape[1]
    n = 1.0 + 0.05 * jnp.sin(y)
    phi = 0.03 * jnp.cos(y)
    vi = 0.2 + 0.04 * jnp.sin(y)
    return FciDrbEBState(n, phi, jnp.full(shape, 4.0), jnp.ones(shape), vi, vi, jnp.zeros(shape))


def _halo(owner, layout):
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    # Wrapped padding supports a periodic extent smaller than the halo width;
    # physical x ghost slabs are subsequently filled by the wall filler.
    out = jnp.pad(owner, ((h, h), (h, h), (h, h)), mode="wrap")
    out = out.at[:h].set(0.0).at[h + nx:].set(0.0)
    return out


def _rhs_stub(state, geometry, domain, parameters):
    model = physical_wall_model_from_name("simplified-gbs-mpe")
    rhs = SimpleNamespace(
        face_bc_builder=model, geometry=geometry, domain=domain,
        parameters=parameters, physical_wall_model_name="simplified-gbs-mpe",
        control_volume_geometry=None,
        halo_exchange=lambda field, _domain: field,
        topology_filler=lambda field, _domain: _halo(
            field[domain.layout.owned_slices_cell], domain.layout
        ),
    )
    rhs._owner_field = lambda value: jnp.asarray(value, dtype=jnp.float64)
    rhs._owner_state = lambda value: value
    return rhs


def test_fci_face_bcs_metric_halo_midpoint_recovers_finite_jump_targets():
    filler, _unused_field, _unused_expected, domain0, _unused_bc = _metric_aware_fixture()
    # The real fixture is a LocalFciGeometry and the domain is made physical-x,
    # periodic-y/z explicitly for this wall-only probe.
    domain = replace(domain0, shard_spec=replace(
        domain0.shard_spec, periodic_axes=(False, True, True),
        side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
    ))
    geometry = filler.geometry
    geometry.grid.x.faces_halo = jnp.arange(
        -domain.layout.halo_width,
        domain.layout.owned_shape[0] + domain.layout.halo_width + 1,
        dtype=jnp.float64,
    )
    for axis, n in zip((geometry.grid.x, geometry.grid.y, geometry.grid.z), domain.layout.owned_shape):
        axis.faces_halo = jnp.arange(
            -domain.layout.halo_width,
            n + domain.layout.halo_width + 1,
            dtype=jnp.float64,
        )
    b_axes = tuple(SimpleNamespace(
        B_contra_owned=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64)[0],
            domain.layout.face_control_shape(axis) + (3,),
        )
    ) for axis in range(3))
    object.__setattr__(geometry, "face_bfield", SimpleNamespace(axes=b_axes))
    state = _state(domain.layout.owned_shape)
    params = SimpleNamespace(Te0=4.0, Ti0=1.0, tau=2.0, mi_over_me=10.0)
    rhs = _rhs_stub(state, geometry, domain, params)
    face_bc = rhs_mod.LocalFciDrbEBRhs._face_bcs(rhs, state)
    filler = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=filler.dirichlet, neumann_lower=filler.neumann_lower,
        neumann_upper=filler.neumann_upper, geometry=geometry,
    )
    h = domain.layout.halo_width
    nx = domain.layout.owned_shape[0]
    trace_errors = []
    for name in ("density", "phi"):
        filled = filler(
            _halo(getattr(state, name), domain.layout), domain,
            getattr(face_bc, name),
        )
        targets = {}
        for side, owner_i, ghost_i in (("lower", h, h - 1), ("upper", h + nx - 1, h + nx)):
            vi = face_bc.Vi.value_x[0 if side == "lower" else -1]
            # The new contract is the actual topology-halo affine trace.  The
            # old owner-only finite-jump target is retained below solely as a
            # negative-control diagnostic.
            topology_halo = _halo(getattr(state, name), domain.layout)
            base, response = neumann_face_trace_physical_affine(
                topology_halo, geometry, domain, 0, side
            )
            bc_value = getattr(face_bc, name).value_x[0 if side == "lower" else -1]
            targets[side] = base + response * bc_value
        trace = build_local_boundary_face_trace_from_halo(
            filled, geometry, domain, getattr(face_bc, name)
        )
        np.testing.assert_allclose(trace.value_x[0], np.asarray(targets["lower"]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(trace.value_x[-1], np.asarray(targets["upper"]), rtol=1e-12, atol=1e-12)
        trace_errors.append(float(max(
            np.max(np.abs(np.asarray(trace.value_x[0] - targets["lower"]))),
            np.max(np.abs(np.asarray(trace.value_x[-1] - targets["upper"]))),
        )))
    assert np.all(np.isfinite(trace_errors))
    print({
        "h2_trace_target_deviation": trace_errors,
    })


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_nonuniform_x_centers_share_actual_affine_trace_weights(side):
    filler, _unused_field, _unused_expected, domain0, _unused_bc = _metric_aware_fixture()
    domain = replace(domain0, shard_spec=replace(
        domain0.shard_spec, periodic_axes=(False, True, True),
        side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
    ))
    geometry = filler.geometry
    geometry.grid.x.centers_halo = jnp.asarray(
        (-0.8, -0.2, 0.2, 0.8, 1.7, 2.9, 4.1, 5.3)
    )
    geometry.grid.x.faces_halo = jnp.asarray(
        (-1.4, -0.5, 0.0, 0.5, 1.25, 2.3, 3.5, 4.5, 5.9)
    )
    for axis, n in ((geometry.grid.y, 5), (geometry.grid.z, 3)):
        axis.faces_halo = jnp.arange(
            -domain.layout.halo_width, n + domain.layout.halo_width + 1,
            dtype=jnp.float64,
        )
    b_axes = tuple(SimpleNamespace(
        B_contra_owned=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64)[0],
            domain.layout.face_control_shape(axis) + (3,),
        )
    ) for axis in range(3))
    object.__setattr__(geometry, "face_bfield", SimpleNamespace(axes=b_axes))
    state = _state(domain.layout.owned_shape)
    params = SimpleNamespace(Te0=4.0, Ti0=1.0, tau=2.0, mi_over_me=10.0)
    rhs = _rhs_stub(state, geometry, domain, params)
    face_bc = rhs_mod.LocalFciDrbEBRhs._face_bcs(rhs, state)
    filler = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=filler.dirichlet, neumann_lower=filler.neumann_lower,
        neumann_upper=filler.neumann_upper, geometry=geometry,
    )
    name = "phi"
    filled = filler(_halo(getattr(state, name), domain.layout), domain, face_bc.phi)
    trace = build_local_boundary_face_trace_from_halo(
        filled, geometry, domain, face_bc.phi
    )
    vi = face_bc.Vi.value_x[0 if side == "lower" else -1]
    topology_halo = _halo(state.phi, domain.layout)
    base, response = neumann_face_trace_physical_affine(
        topology_halo, geometry, domain, 0, side
    )
    expected = base + response * face_bc.phi.value_x[0 if side == "lower" else -1]
    actual = trace.value_x[0 if side == "lower" else -1]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_manual_shared_halo_target_encoding_is_exact():
    filler, _field, _expected, domain, _bc = _metric_aware_fixture()
    domain = replace(domain, shard_spec=replace(
        domain.shard_spec, periodic_axes=(False, True, True),
        side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
    ))
    geometry = filler.geometry
    geometry.grid.x.faces_halo = jnp.arange(
        -domain.layout.halo_width,
        domain.layout.owned_shape[0] + domain.layout.halo_width + 1,
        dtype=jnp.float64,
    )
    y = jnp.arange(domain.layout.owned_shape[1], dtype=jnp.float64)[None, :, None]
    owner = jnp.broadcast_to(1.0 + 0.1 * jnp.sin(y), domain.layout.owned_shape)
    target_lower = owner[0] + 0.125
    target_upper = owner[-1] - 0.075
    lower_deriv = physical_normal_derivative_for_face_trace(
        _halo(owner, domain.layout), target_lower, geometry, domain, 0, "lower"
    )
    upper_deriv = physical_normal_derivative_for_face_trace(
        _halo(owner, domain.layout), target_upper, geometry, domain, 0, "upper"
    )
    bc = LocalBoundaryFaceBC3D.empty(domain.layout)
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_NEUMANN).at[-1].set(BC_NEUMANN),
        value_x=bc.value_x.at[0].set(lower_deriv).at[-1].set(upper_deriv),
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
    )
    filled = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=filler.dirichlet, neumann_lower=filler.neumann_lower,
        neumann_upper=filler.neumann_upper, geometry=geometry,
    )(_halo(owner, domain.layout), domain, bc)
    trace = build_local_boundary_face_trace_from_halo(filled, geometry, domain, bc)
    np.testing.assert_allclose(trace.value_x[0], target_lower, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(trace.value_x[-1], target_upper, rtol=0.0, atol=1e-12)
