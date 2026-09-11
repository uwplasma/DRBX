"""RLP regression for the raw-metric material ``div(b)`` source."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from drbx.geometry.fci_control_volumes import (  # noqa: E402
    build_polar_angular_agglomeration_geometry,
)
from drbx.native import FciDrbEBState  # noqa: E402
from drbx.native.fci_boundaries import (  # noqa: E402
    LocalControlVolumeBoundaryBC3D,
)
from drbx.native.fci_angular_agglomeration import (  # noqa: E402
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
)
from fci_drb_eb_test_helpers import _build_rhs  # noqa: E402
from shifted_torus_eb_mms_data import (  # noqa: E402
    build_shifted_torus_eb_mms_context,
)


SHEAR = 0.37


def _material_source_case(n: int = 16):
    shape = (3, n, n)
    context = build_shifted_torus_eb_mms_context(shape)
    context = replace(
        context,
        parameters=replace(
            context.parameters,
            parallel_characteristic_wall_law="energy-absorbing",
        ),
    )
    base = context.geometry
    ii, jj, kk = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        indexing="ij",
    )
    y = np.asarray(base.grid.y_centers)
    z = np.asarray(base.grid.z_centers)
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    ds = float(np.hypot(SHEAR * dy, dz))
    phase = jnp.asarray(y)[None, :, None] + jnp.asarray(z)[None, None, :]
    B = jnp.broadcast_to(jnp.exp(0.2 * jnp.cos(phase)), shape)
    phase_step = SHEAR * dy + dz
    B_forward = jnp.broadcast_to(
        jnp.exp(0.2 * jnp.cos(phase + phase_step)), shape
    )
    B_backward = jnp.broadcast_to(
        jnp.exp(0.2 * jnp.cos(phase - phase_step)), shape
    )
    zeros = jnp.zeros(shape, dtype=bool)
    maps = replace(
        base.maps,
        forward_x=jnp.asarray(ii),
        forward_y=jnp.asarray(jj + SHEAR),
        backward_x=jnp.asarray(ii),
        backward_y=jnp.asarray(jj - SHEAR),
        forward_endpoint_x=jnp.asarray(base.grid.x_centers[ii.astype(int)]),
        forward_endpoint_y=jnp.asarray(y[jj.astype(int)] + SHEAR * dy),
        forward_endpoint_z=jnp.asarray(z[kk.astype(int)] + dz),
        backward_endpoint_x=jnp.asarray(base.grid.x_centers[ii.astype(int)]),
        backward_endpoint_y=jnp.asarray(y[jj.astype(int)] - SHEAR * dy),
        backward_endpoint_z=jnp.asarray(z[kk.astype(int)] - dz),
        forward_endpoint_bmag=B_forward,
        backward_endpoint_bmag=B_backward,
        forward_length=jnp.full(shape, ds),
        backward_length=jnp.full(shape, ds),
        forward_boundary=zeros,
        backward_boundary=zeros,
    )
    global_geometry = replace(
        base,
        maps=maps,
        cell_bfield=replace(base.cell_bfield, Bmag=B),
    )
    sharded = build_local_fci_geometries(
        global_geometry,
        (1, 1, 1),
        halo_width=2,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    local = assemble_single_device_local_fci_geometry(sharded)
    host_sharded = replace(
        sharded,
        domain=replace(sharded.domain, mesh_axis_names=(None, None, None)),
    )
    rhs = _build_rhs(context, host_sharded, local)

    host_rlp = build_polar_angular_agglomeration_geometry(
        np.asarray(base.grid.x_faces),
        np.asarray(base.grid.y_faces),
        np.asarray(base.grid.z_faces),
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=3,
        angular_group_size=(n, 2, 1),
    )
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host_rlp, host_sharded.domain
    )
    local_rlp = assemble_local_polar_angular_agglomeration_geometry(
        descriptor, packed, local
    )
    rhs_rlp = replace(
        rhs,
        control_volume_geometry=local_rlp,
        control_volume_boundary_bc=LocalControlVolumeBoundaryBC3D.empty(),
        poisson_bracket_scheme="compatible-flux",
        axis_regular_axes=(True, False, False),
    )

    one = jnp.ones(shape, dtype=jnp.float64)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    state = FciDrbEBState(one, zero, one, one, zero, zero, zero)
    face_bc = rhs._face_bcs(state)
    stencil_context = rhs._stencil_builder_context()
    exact = np.broadcast_to(
        0.2 * np.sin(np.asarray(phase)) * phase_step / ds,
        shape,
    )
    return rhs, rhs_rlp, face_bc, stencil_context, exact


def test_material_geometry_source_ignores_rlp_owner_reconstruction():
    rhs, rhs_rlp, face_bc, context, exact = _material_source_case()
    fallback = jnp.full(rhs.geometry.owned_shape, -17.0, dtype=jnp.float64)
    plain = rhs._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )
    with_rlp = rhs_rlp._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )

    np.testing.assert_array_equal(
        np.asarray(with_rlp["material_div_b"]),
        np.asarray(plain["material_div_b"]),
    )
    assert bool(np.all(np.asarray(with_rlp["material_div_b_valid"])))
    assert not bool(np.any(np.asarray(with_rlp["material_div_b_fallback"])))
    relative_error = np.linalg.norm(np.asarray(with_rlp["material_div_b"]) - exact)
    relative_error /= np.linalg.norm(exact)
    assert relative_error < 0.06


def test_material_geometry_source_invalid_raw_donor_uses_fallback():
    _rhs, rhs_rlp, face_bc, context, _exact = _material_source_case(8)
    local = rhs_rlp.geometry
    owned = local.cell_bfield.owned_slices_in_halo
    bad_B = np.asarray(local.cell_bfield.Bmag_owned).copy()
    bad_B[0, 0, 0] = np.nan
    bad_bfield = replace(
        local.cell_bfield,
        Bmag_halo=local.cell_bfield.Bmag_halo.at[owned].set(jnp.asarray(bad_B)),
    )
    rhs_bad = replace(rhs_rlp, geometry=replace(local, cell_bfield=bad_bfield))
    fallback = jnp.full(local.owned_shape, 23.0, dtype=jnp.float64)
    result = rhs_bad._fci_second_order_material_div_b(
        face_bc, context, fallback_div_b=fallback
    )
    mask = np.asarray(result["material_div_b_fallback"])
    assert int(np.count_nonzero(mask)) > 1
    np.testing.assert_array_equal(
        np.asarray(result["material_div_b"])[mask],
        np.full(np.count_nonzero(mask), 23.0),
    )
