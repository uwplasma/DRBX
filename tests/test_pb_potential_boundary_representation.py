"""Cartesian MMS guard for the potential Poisson-bracket representation."""

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

from drbx.geometry import (
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
)
from drbx.native.fci_boundaries import LocalBoundaryFaceTrace3D
from drbx.native.fci_operators import local_poisson_bracket_compatible_flux_op

from test_fci_operators_domain_decomp import _build_domain, _build_local_geometry


def _identity_metric(metric):
    one = jnp.ones_like(metric.J_halo)
    zero = jnp.zeros_like(metric.J_halo)
    return replace(
        metric, J_halo=one, g11_halo=one, g22_halo=one, g33_halo=one,
        g12_halo=zero, g13_halo=zero, g23_halo=zero,
        g_11_halo=one, g_22_halo=one, g_33_halo=one,
        g_12_halo=zero, g_13_halo=zero, g_23_halo=zero,
    )


def _axial_bfield(field):
    b = jnp.zeros_like(field.B_contra_halo).at[..., 2].set(1.0)
    return replace(field, B_contra_halo=b, Bmag_halo=jnp.ones_like(field.Bmag_halo))


def _cartesian_geometry(geometry):
    face_metric = replace(
        geometry.face_metric,
        x=_identity_metric(geometry.face_metric.x),
        y=_identity_metric(geometry.face_metric.y),
        z=_identity_metric(geometry.face_metric.z),
    )
    face_bfield = replace(
        geometry.face_bfield,
        x=_axial_bfield(geometry.face_bfield.x),
        y=_axial_bfield(geometry.face_bfield.y),
        z=_axial_bfield(geometry.face_bfield.z),
    )
    return replace(
        geometry,
        cell_metric=_identity_metric(geometry.cell_metric),
        face_metric=face_metric,
        cell_bfield=_axial_bfield(geometry.cell_bfield),
        face_bfield=face_bfield,
        cell_volume_geometry=replace(
            geometry.cell_volume_geometry,
            volume=jnp.ones_like(geometry.cell_volume_geometry.volume),
        ),
    )


def _fields(geometry, homogeneous):
    hshape = geometry.halo_shape
    x = geometry.grid.x.centers_halo[:, None, None]
    y = geometry.grid.y.centers_halo[None, :, None]
    f = jnp.broadcast_to(x**2, hshape)
    g = jnp.broadcast_to(jnp.sin(y), hshape)
    if homogeneous:
        h = geometry.layout.halo_width
        nx = geometry.layout.owned_shape[0]
        f = f.at[:h].set(f[h:h + 1])
        f = f.at[h + nx:].set(f[h + nx - 1:h + nx])
    return f, g


def _potential_trace(geometry):
    trace = LocalBoundaryFaceTrace3D.empty(geometry.layout)
    xf = geometry.grid.x.faces_owned[:, None, None]
    values = jnp.broadcast_to(xf**2, trace.value_x.shape)
    masks = trace.mask_x.at[0].set(True).at[-1].set(True)
    return LocalBoundaryFaceTrace3D(
        value_x=values, value_y=trace.value_y, value_z=trace.value_z,
        mask_x=masks, mask_y=trace.mask_y, mask_z=trace.mask_z,
        layout=geometry.layout,
    )


def _case(shape):
    halo = 2
    domain = _build_domain(shape, halo, (1, 1, 1))
    geometry = _cartesian_geometry(
        _build_local_geometry(shape, halo, global_shape=shape)
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    x = geometry.grid.x.centers_owned[:, None, None]
    y = geometry.grid.y.centers_owned[None, :, None]
    expected = np.asarray(2.0 * x * jnp.cos(y))
    active = np.asarray(geometry.cell_volume_geometry.volume > 0.0)
    trace = _potential_trace(geometry)
    result = {}
    for label, homogeneous in (("correct", False), ("old_homogeneous", True)):
        f, g = _fields(geometry, homogeneous)
        fs = build_local_conservative_stencil_from_field(f, geometry, context)
        gs = build_local_conservative_stencil_from_field(g, geometry, context)
        with jax.disable_jit():
            actual = np.asarray(local_poisson_bracket_compatible_flux_op(
                fs, gs, geometry, domain=domain,
                f_boundary_trace=trace,
                characteristic_scheme="centered",
                axis_regular_axes=(False, False, False),
            ))
        error = (actual - expected)[active]
        result[label] = {
            "global_l2": float(np.linalg.norm(error) / np.sqrt(error.size)),
            "max_error": float(np.max(np.abs(error))),
        }
    return result


@pytest.mark.parametrize("shape", [(8, 32, 4), (16, 64, 4)])
def test_potential_physical_trace_beats_homogeneous_support(shape):
    result = _case(shape)
    assert result["correct"]["max_error"] < (0.03 if shape[0] == 8 else 0.02)
    assert result["old_homogeneous"]["max_error"] > 0.4
    if shape[0] == 16:
        assert result["correct"]["global_l2"] < 0.003


def test_potential_trace_control_refines():
    coarse = _case((8, 32, 4))
    fine = _case((16, 64, 4))
    assert fine["correct"]["global_l2"] < coarse["correct"]["global_l2"]
