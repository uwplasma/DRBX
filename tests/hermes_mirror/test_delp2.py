from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from jaxdrb.bc import BC2D
from jaxdrb.hermes_mirror import delp2_runtime, derive_delp2_coefficients


def _geom_for_test(**overrides):
    base = {
        "metric_dx": jnp.ones((4, 8), dtype=jnp.float64),
        "metric_dy": jnp.ones((4, 8), dtype=jnp.float64),
        "metric_dz": jnp.ones((4, 8), dtype=jnp.float64),
        "jacobian": jnp.ones((4, 8), dtype=jnp.float64),
        "gxx": jnp.ones((4, 8), dtype=jnp.float64),
        "gxy": jnp.zeros((4, 8), dtype=jnp.float64),
        "gyy": jnp.ones((4, 8), dtype=jnp.float64),
        "g23": jnp.zeros((4, 8), dtype=jnp.float64),
        "G1": jnp.zeros((4, 8), dtype=jnp.float64),
        "G3": jnp.zeros((4, 8), dtype=jnp.float64),
        "d1_dx": jnp.zeros((4, 8), dtype=jnp.float64),
        "grid": SimpleNamespace(open_field_line=False),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_derive_delp2_coefficients_uses_explicit_geometry_fields() -> None:
    geom = _geom_for_test(
        G1=2.0 * jnp.ones((4, 8), dtype=jnp.float64),
        G3=-3.0 * jnp.ones((4, 8), dtype=jnp.float64),
        d1_dx=4.0 * jnp.ones((4, 8), dtype=jnp.float64),
    )

    G1, G3, d1_dx = derive_delp2_coefficients(
        geom=geom,
        nz=4,
        nx=8,
        ny=16,
        periodic_x=True,
        periodic_parallel=True,
        lower_boundary_open=False,
        upper_boundary_open=False,
    )

    np.testing.assert_allclose(np.asarray(G1), 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(G3), -3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d1_dx), 4.0, rtol=1e-12, atol=1e-12)


def test_delp2_runtime_matches_periodic_constant_metric_laplacian() -> None:
    nz, nx, ny = 4, 8, 32
    geom = _geom_for_test(
        metric_dx=jnp.ones((nz, nx), dtype=jnp.float64),
        metric_dy=jnp.ones((nz, nx), dtype=jnp.float64),
        metric_dz=jnp.ones((nz, nx), dtype=jnp.float64),
        jacobian=jnp.ones((nz, nx), dtype=jnp.float64),
        gxx=jnp.ones((nz, nx), dtype=jnp.float64),
        gxy=jnp.zeros((nz, nx), dtype=jnp.float64),
        gyy=jnp.ones((nz, nx), dtype=jnp.float64),
        G1=jnp.zeros((nz, nx), dtype=jnp.float64),
        G3=jnp.zeros((nz, nx), dtype=jnp.float64),
        d1_dx=jnp.zeros((nz, nx), dtype=jnp.float64),
    )
    bc = BC2D(kind_x=0, kind_y=0)

    x = jnp.arange(nx, dtype=jnp.float64)
    z = jnp.arange(ny, dtype=jnp.float64)
    kx = 2.0 * jnp.pi / float(nx)
    kz = 2.0 * jnp.pi / float(ny)
    field_plane = jnp.cos(kx * x)[:, None] + jnp.sin(kz * z)[None, :]
    field = jnp.broadcast_to(field_plane[None, :, :], (nz, nx, ny))

    out = delp2_runtime(field, geom=geom, bc_field=bc)
    expected_plane = -(kx * kx) * jnp.cos(kx * x)[:, None] - (kz * kz) * jnp.sin(kz * z)[None, :]
    expected = jnp.broadcast_to(expected_plane[None, :, :], (nz, nx, ny))

    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=2e-2, atol=2e-2)
