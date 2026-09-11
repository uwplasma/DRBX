"""Strict tests for affine Neumann face-trace reconstruction."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import (
    HaloLayout3D,
    LocalDomain3D,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
)
from drbx.native.fci_boundaries import (
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
    build_local_boundary_face_trace_from_halo,
    neumann_face_trace_coordinate_affine,
)
from drbx.native import fci_drb_EB_rhs as rhs_mod
import drbx.native.fci_physical_wall as wall_module
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    MetricAwarePhysicalGhostCellFiller3D,
    physical_normal_derivative_for_face_trace,
)


def _fixture(halo_width, shape=(4, 5, 3)):
    layout = HaloLayout3D(shape, halo_width)
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=shape, owned_start=(0, 0, 0), owned_stop=shape,
            shard_index=(0, 0, 0), shard_counts=(1, 1, 1),
            periodic_axes=(False, True, True), halo_width=halo_width,
            side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
            side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        ),
        layout=layout, mesh_axis_names=(None, None, None),
    )
    coords = tuple(jnp.arange(-halo_width, n + halo_width, dtype=jnp.float64) + 0.5
                   for n in shape)
    grid = SimpleNamespace(
        x=SimpleNamespace(centers_halo=coords[0], faces_halo=jnp.arange(-halo_width, shape[0] + halo_width + 1, dtype=jnp.float64)),
        y=SimpleNamespace(centers_halo=coords[1], faces_halo=jnp.arange(-halo_width, shape[1] + halo_width + 1, dtype=jnp.float64)),
        z=SimpleNamespace(centers_halo=coords[2], faces_halo=jnp.arange(-halo_width, shape[2] + halo_width + 1, dtype=jnp.float64)),
    )
    skew = jnp.asarray(((1.5, 0.4, 0.25), (0.4, 1.3, 0.1), (0.25, 0.1, 1.1)))
    face_metric = SimpleNamespace(axes=tuple(SimpleNamespace(
        g_contra=jnp.broadcast_to(skew, layout.cell_halo_shape + (3, 3))
    ) for _ in range(3)))
    geometry = object.__new__(__import__(
        "drbx.geometry", fromlist=["LocalFciGeometry3D"]
    ).LocalFciGeometry3D)
    object.__setattr__(geometry, "layout", layout)
    object.__setattr__(geometry, "grid", grid)
    object.__setattr__(geometry, "face_metric", face_metric)
    owned_shape = layout.owned_shape
    filler = MetricAwarePhysicalGhostCellFiller3D(
        dirichlet=tuple(GhostFillWeights1D(
            owned_weights=-jnp.ones((halo_width, 1)),
            bc_weights=2.0 * jnp.ones((halo_width,)),
        ) for _ in range(3)),
        neumann_lower=tuple(GhostFillWeights1D(
            owned_weights=jnp.ones((halo_width, 1)),
            bc_weights=jnp.zeros((halo_width,)),
        ) for _ in range(3)),
        neumann_upper=tuple(GhostFillWeights1D(
            owned_weights=jnp.ones((halo_width, 1)),
            bc_weights=jnp.zeros((halo_width,)),
        ) for _ in range(3)),
        geometry=geometry,
    )
    return layout, domain, geometry, filler


def _halo(owner, layout):
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    out = jnp.pad(owner, ((h, h), (h, h), (h, h)), mode="wrap")
    out = out.at[:h].set(0.0).at[h + nx:].set(0.0)
    return out


@pytest.mark.parametrize("halo_width", [1, 2])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_neumann_affine_coefficients_have_uniform_grid_golden_values(halo_width, side):
    layout, domain, geometry, filler = _fixture(halo_width)
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    field = jnp.broadcast_to(x**2 + 0.1 * jnp.sin(y), layout.cell_halo_shape)
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    field = field.at[:h].set(jnp.nan).at[h + nx:].set(jnp.nan)
    base, alpha = neumann_face_trace_coordinate_affine(field, geometry, domain, 0, side)
    nearest_i = h if side == "lower" else h + nx - 1
    next_i = h + 1 if side == "lower" else h + nx - 2
    nearest = field[nearest_i, h:h + ny, h:h + nz]
    nxt = field[next_i, h:h + ny, h:h + nz]
    dx = geometry.grid.x.centers_halo[1] - geometry.grid.x.centers_halo[0]
    expected_base = (9.0 * nearest - nxt) / 8.0 if halo_width == 2 else nearest
    expected_alpha = (-3.0 if side == "lower" else 3.0) * dx / 8.0 if halo_width == 2 else (-0.5 if side == "lower" else 0.5) * dx
    np.testing.assert_allclose(base, expected_base, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("halo_width", [1, 2])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_metric_filler_and_actual_trace_recover_target(halo_width, side):
    layout, domain, geometry, filler = _fixture(halo_width)
    x = geometry.grid.x.centers_halo[layout.halo_width:layout.halo_width + layout.owned_shape[0]][:, None, None]
    y = geometry.grid.y.centers_halo[layout.halo_width:layout.halo_width + layout.owned_shape[1]][None, :, None]
    owner_x = layout.halo_width if side == "lower" else layout.halo_width + layout.owned_shape[0] - 1
    face_x = layout.halo_width if side == "lower" else layout.halo_width + layout.owned_shape[0]
    owner = jnp.broadcast_to(x**2 + 0.1 * jnp.sin(y), layout.owned_shape)
    halo = _halo(owner, layout)
    target = jnp.broadcast_to(
        geometry.grid.x.faces_halo[face_x] ** 2 + 0.1 * jnp.sin(geometry.grid.y.centers_halo[layout.halo_width:layout.halo_width + layout.owned_shape[1]])[:, None],
        (layout.owned_shape[1], layout.owned_shape[2]),
    )
    derivative = physical_normal_derivative_for_face_trace(
        halo, target, geometry, domain, 0, side
    )
    bc = LocalBoundaryFaceBC3D.empty(layout)
    idx = 0 if side == "lower" else -1
    bc = replace(
        bc,
        kind_x=bc.kind_x.at[idx].set(BC_NEUMANN),
        value_x=bc.value_x.at[idx].set(derivative),
        mask_x=bc.mask_x.at[idx].set(True),
    )
    filled = filler(halo, domain, bc)
    trace = build_local_boundary_face_trace_from_halo(filled, geometry, domain, bc)
    actual = trace.value_x[idx]
    np.testing.assert_allclose(actual, target, rtol=0.0, atol=2e-12)
    eager = physical_normal_derivative_for_face_trace(halo, target, geometry, domain, 0, side)
    compiled = jax.jit(lambda value: physical_normal_derivative_for_face_trace(
        value, target, geometry, domain, 0, side
    ))(halo)
    np.testing.assert_allclose(compiled, eager, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("halo_width", [1, 2])
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_physical_affine_roundtrip_and_constant_shift(halo_width, side):
    layout, domain, geometry, filler = _fixture(halo_width)
    owner = jnp.arange(np.prod(layout.owned_shape), dtype=jnp.float64).reshape(layout.owned_shape)
    halo = _halo(owner, layout)
    base, response = wall_module.neumann_face_trace_physical_affine(halo, geometry, domain, 0, side)
    g = jnp.full_like(base, 0.37)
    bc = LocalBoundaryFaceBC3D.empty(layout)
    idx = 0 if side == "lower" else -1
    bc = replace(bc, kind_x=bc.kind_x.at[idx].set(BC_NEUMANN), value_x=bc.value_x.at[idx].set(g), mask_x=bc.mask_x.at[idx].set(True))
    trace = build_local_boundary_face_trace_from_halo(filler(halo, domain, bc), geometry, domain, bc)
    recovered = physical_normal_derivative_for_face_trace(halo, trace.value_x[idx], geometry, domain, 0, side)
    np.testing.assert_allclose(recovered, g, rtol=0.0, atol=2e-12)
    shifted_base, shifted_response = wall_module.neumann_face_trace_physical_affine(halo + 2.5, geometry, domain, 0, side)
    np.testing.assert_allclose(shifted_response, response, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(shifted_base - base, 2.5, rtol=0.0, atol=2e-12)


def test_thin_periodic_axis_skips_nonphysical_z_faces(monkeypatch):
    """A periodic extent of one cell must not be treated as a wall endpoint."""
    layout, domain, geometry, filler = _fixture(2, shape=(4, 5, 1))
    b_axes = tuple(SimpleNamespace(
        B_contra_owned=jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float64)[0],
            layout.face_control_shape(axis) + (3,),
        )
    ) for axis in range(3))
    object.__setattr__(geometry, "face_bfield", SimpleNamespace(axes=b_axes))
    ijk = jnp.indices(layout.owned_shape, dtype=jnp.float64)
    y = 2.0 * jnp.pi * ijk[1] / layout.owned_shape[1]
    state = rhs_mod.FciDrbEBState(
        1.0 + 0.05 * jnp.sin(y), 0.03 * jnp.cos(y), jnp.full(layout.owned_shape, 4.0),
        jnp.ones(layout.owned_shape), 0.2 + 0.04 * jnp.sin(y),
        0.2 + 0.04 * jnp.sin(y), jnp.zeros(layout.owned_shape)
    )
    params = SimpleNamespace(Te0=4.0, Ti0=1.0, tau=2.0, mi_over_me=10.0)
    model = wall_module.physical_wall_model_from_name("simplified-gbs-mpe")
    rhs = SimpleNamespace(face_bc_builder=model, geometry=geometry, domain=domain, parameters=params,
                          physical_wall_model_name="simplified-gbs-mpe", control_volume_geometry=None,
                          halo_exchange=lambda field, _domain: field,
                          topology_filler=lambda field, _domain: _halo(field[domain.layout.owned_slices_cell], domain.layout),
                          _owner_field=lambda value: jnp.asarray(value, dtype=jnp.float64),
                          _owner_state=lambda value: value)
    calls = []
    real_affine = wall_module.neumann_face_trace_physical_affine
    def wrapped_affine(field_halo, geometry_, domain_, axis, side, **kwargs):
        calls.append((axis, side))
        return real_affine(field_halo, geometry_, domain_, axis, side, **kwargs)
    monkeypatch.setattr(wall_module, "neumann_face_trace_physical_affine", wrapped_affine)
    face_bc = rhs_mod.LocalFciDrbEBRhs._face_bcs(rhs, state)
    assert calls and all(axis == 0 for axis, _side in calls)
    filled = filler(_halo(state.phi, layout), domain, face_bc.phi)
    trace = build_local_boundary_face_trace_from_halo(
        filled, geometry, domain, face_bc.phi
    )
    for side, index in (("lower", 0), ("upper", -1)):
        vi = face_bc.Vi.value_x[index]
        owner_i = 0 if side == "lower" else -1
        vi_owner = state.Vi[owner_i]
        vi_halo = _halo(state.Vi, layout)
        g_vi = physical_normal_derivative_for_face_trace(vi_halo, vi, geometry, domain, 0, side)
        g_phi = physical_normal_derivative_for_face_trace(
            _halo(state.phi, layout), trace.value_x[index], geometry, domain, 0, side
        )
        sigma = -1.0 if side == "lower" else 1.0
        te_trace = build_local_boundary_face_trace_from_halo(
            filler(_halo(state.Te, layout), domain, face_bc.Te), geometry, domain, face_bc.Te
        )
        ti_trace = build_local_boundary_face_trace_from_halo(
            filler(_halo(state.Ti, layout), domain, face_bc.Ti), geometry, domain, face_bc.Ti
        )
        te_face = te_trace.value_x[index]
        ti_face = ti_trace.value_x[index]
        residual = g_phi + sigma * te_face / jnp.sqrt(te_face + params.tau * ti_face) * g_vi
        np.testing.assert_allclose(np.asarray(residual), 0.0, rtol=0.0, atol=1e-11)
