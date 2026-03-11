from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.core.geometry_field_aligned import FieldAlignedGeometryAdapter, FieldAlignedGrid
from jaxdrb.core.params import DRBSystemParams, NumericsParams
from jaxdrb.hermes_literal import (
    build_shifted_metric_fft_phases,
    build_shifted_metric_weights,
    from_field_aligned_all_fft,
    from_field_aligned_all_fft_ref,
    from_field_aligned_nobndry_fft,
    from_field_aligned_nobndry_fft_ref,
    shifted_metric_fft_phases_from_geometry,
    shifted_metric_weights_from_geometry,
    to_field_aligned_nox_fft,
    to_field_aligned_nox_fft_ref,
    to_field_aligned_nox,
    to_field_aligned_nox_ref,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _make_shift_geom(
    *, open_field_line: bool, interp: str = "linear"
) -> FieldAlignedGeometryAdapter:
    params = DRBSystemParams(
        numerics=NumericsParams(
            parallel_transform="shifted",
            parallel_shift_interp=interp,
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


def test_to_field_aligned_nox_matches_current_geometry_adapter_linear_shift() -> None:
    geom = _make_shift_geom(open_field_line=True)
    weights = shifted_metric_weights_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(12), geom.shape(), dtype=jnp.float64)

    literal = to_field_aligned_nox(field, weights)
    current = geom.to_field_aligned_nox(field)

    np.testing.assert_allclose(np.asarray(literal), np.asarray(current), rtol=1e-12, atol=1e-12)


def test_fft_shift_ref_matches_fused() -> None:
    geom = _make_shift_geom(open_field_line=True, interp="spectral")
    phases = shifted_metric_fft_phases_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(15), geom.shape(), dtype=jnp.float64)

    ref = to_field_aligned_nox_fft_ref(field, phases)
    fused = to_field_aligned_nox_fft(field, phases)

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_fft_shift_roundtrip_matches_dump_backed_fixture() -> None:
    fixture_path = _FIXTURE_DIR / "hermes_mirror_shiftedmetric_global_t1.npz"
    with np.load(fixture_path, allow_pickle=False) as data:
        field = jnp.asarray(data["field"], dtype=jnp.float64)
        z_shift = jnp.asarray(data["z_shift"], dtype=jnp.float64)
        zlength = float(np.asarray(data["zlength"]))

    phases = build_shifted_metric_fft_phases(
        z_shift,
        nx=field.shape[1],
        npar=field.shape[0],
        nbinorm=field.shape[2],
        zlength=zlength,
        open_field_line=True,
    )
    aligned_ref = to_field_aligned_nox_fft_ref(field, phases)
    aligned = to_field_aligned_nox_fft(field, phases)
    restored_ref = from_field_aligned_nobndry_fft_ref(aligned_ref, phases)
    restored = from_field_aligned_nobndry_fft(aligned, phases)

    np.testing.assert_allclose(np.asarray(aligned), np.asarray(aligned_ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(restored), np.asarray(restored_ref), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(aligned[:, 0, :]), np.asarray(field[:, 0, :]))
    np.testing.assert_allclose(np.asarray(aligned[:, -1, :]), np.asarray(field[:, -1, :]))
    np.testing.assert_allclose(np.asarray(restored[0, :, :]), np.asarray(aligned[0, :, :]))
    np.testing.assert_allclose(np.asarray(restored[-1, :, :]), np.asarray(aligned[-1, :, :]))
    np.testing.assert_allclose(
        np.asarray(restored[1:-1, 1:-1, :]),
        np.asarray(field[1:-1, 1:-1, :]),
        rtol=1e-8,
        atol=1e-8,
    )


def test_fft_from_field_aligned_all_ref_matches_fused() -> None:
    geom = _make_shift_geom(open_field_line=True, interp="spectral")
    phases = shifted_metric_fft_phases_from_geometry(geom)
    field = jax.random.normal(jax.random.PRNGKey(29), geom.shape(), dtype=jnp.float64)

    ref = from_field_aligned_all_fft_ref(field, phases)
    fused = from_field_aligned_all_fft(field, phases)

    np.testing.assert_allclose(np.asarray(fused), np.asarray(ref), rtol=1e-12, atol=1e-12)
