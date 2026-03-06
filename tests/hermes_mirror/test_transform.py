from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.params import DRBSystemParams, NumericsParams
from jaxdrb.hermes_mirror import (
    build_shifted_metric_weights,
    from_field_aligned_nobndry,
    from_field_aligned_nobndry_ref,
    shifted_metric_weights_from_geometry,
    to_field_aligned_nox,
    to_field_aligned_nox_ref,
)


def _make_shift_geom(*, open_field_line: bool) -> FieldAlignedGeometryAdapter:
    params = DRBSystemParams(
        numerics=NumericsParams(
            parallel_transform="shifted",
            parallel_shift_interp="linear",
            perp_operator="fd",
            bracket="centered",
        )
    )
    grid = FieldAlignedGrid.make(
        nx=6,
        ny=8,
        nz=5,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bc_x="neumann",
        bc_y="periodic",
        dealias=False,
        open_field_line=open_field_line,
    )
    z_shift = jnp.array(
        [
            [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
            [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        ],
        dtype=jnp.float64,
    )
    return FieldAlignedGeometryAdapter.from_coefficients(
        params=params,
        grid=grid,
        curv_x=0.0,
        curv_y=0.0,
        dpar_factor=1.0,
        B=1.0,
        jacobian=1.0,
        gxx=1.0,
        gxy=0.0,
        gyy=1.0,
        g23=1.0,
        z_shift=z_shift,
    )


def test_build_shifted_metric_weights_normalizes_shift_shape() -> None:
    weights = build_shifted_metric_weights(0.5, nx=3, npar=4, nbinorm=6, open_field_line=True)

    assert weights.shift_idx.shape == (4, 3)
    assert weights.forward_index0.shape == (4, 3, 6)
    assert weights.backward_frac.shape == (4, 3, 6)
    assert weights.open_field_line is True


def test_to_field_aligned_nox_ref_matches_fused() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(10), geom.shape(), dtype=jnp.float64)

    ref = to_field_aligned_nox_ref(field, weights)
    fused = to_field_aligned_nox(field, weights)

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_from_field_aligned_nobndry_ref_matches_fused() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(11), geom.shape(), dtype=jnp.float64)

    ref = from_field_aligned_nobndry_ref(field, weights)
    fused = from_field_aligned_nobndry(field, weights)

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_to_field_aligned_nox_matches_current_geometry_adapter_linear_shift() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(12), geom.shape(), dtype=jnp.float64)

    mirror = to_field_aligned_nox(field, weights)
    current = geom.to_field_aligned_nox(field)

    np.testing.assert_allclose(np.asarray(mirror), np.asarray(current), rtol=1e-12, atol=1e-12)


def test_from_field_aligned_nobndry_preserves_parallel_and_x_boundaries() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(13), geom.shape(), dtype=jnp.float64)

    out = from_field_aligned_nobndry(field, weights)

    np.testing.assert_allclose(np.asarray(out[:, 0, :]), np.asarray(field[:, 0, :]))
    np.testing.assert_allclose(np.asarray(out[:, -1, :]), np.asarray(field[:, -1, :]))
    np.testing.assert_allclose(np.asarray(out[0, :, :]), np.asarray(field[0, :, :]))
    np.testing.assert_allclose(np.asarray(out[-1, :, :]), np.asarray(field[-1, :, :]))


def test_from_field_aligned_nobndry_interior_matches_current_geometry_transform() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(14), geom.shape(), dtype=jnp.float64)

    mirror = from_field_aligned_nobndry(field, weights)
    current = geom.from_field_aligned_nox(field)

    np.testing.assert_allclose(
        np.asarray(mirror[1:-1, 1:-1, :]),
        np.asarray(current[1:-1, 1:-1, :]),
        rtol=1e-12,
        atol=1e-12,
    )
