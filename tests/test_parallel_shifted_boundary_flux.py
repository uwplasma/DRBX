from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

import jaxdrb.core.terms.parallel as parallel_terms
from jaxdrb.core.terms.parallel import (
    _dpar_flux_conservative,
    _shift_boundary_flux_to_field_aligned,
)


def test_shifted_boundary_flux_passthrough_when_not_shifted_transform() -> None:
    flux = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
    params = SimpleNamespace(parallel_transform="none", parallel_shift_interp="linear")
    geom = SimpleNamespace(shift_idx=jnp.asarray([[1.0]]))
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(flux))


def test_shifted_boundary_flux_applies_integer_shift() -> None:
    flux = jnp.asarray([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
    params = SimpleNamespace(parallel_transform="shifted", parallel_shift_interp="linear")
    geom = SimpleNamespace(
        shift_idx=jnp.asarray(
            [
                [1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
    )
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    expect = np.asarray([[1.0, 2.0, 3.0, 0.0], [11.0, 12.0, 13.0, 10.0]])
    np.testing.assert_allclose(np.asarray(out), expect)


def test_shifted_boundary_flux_supports_fractional_shift() -> None:
    flux = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
    params = SimpleNamespace(parallel_transform="shifted", parallel_shift_interp="linear")
    geom = SimpleNamespace(shift_idx=jnp.asarray([[0.5]]))
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    expect = np.asarray([[0.5, 1.5, 2.5, 1.5]])
    np.testing.assert_allclose(np.asarray(out), expect)


def test_shifted_boundary_flux_uses_spectral_transform_when_requested() -> None:
    flux = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
    )
    params = SimpleNamespace(parallel_transform="shifted", parallel_shift_interp="spectral")
    geom = SimpleNamespace(
        shift_idx=jnp.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
        z_shift=jnp.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
        grid=SimpleNamespace(
            open_field_line=True,
            perp=SimpleNamespace(dy=1.0, bc=SimpleNamespace(kind_x=1)),
        ),
    )
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    expect = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [11.0, 12.0, 13.0, 10.0],
            [21.0, 22.0, 23.0, 20.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-12, atol=1e-12)


def test_shifted_parallel_flux_transforms_ghost_planes_before_mirror_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, jnp.ndarray] = {}

    def fake_div_par_mod(f, v, wave, **kwargs):
        captured["ghost_low_f"] = kwargs["ghost_low_f"]
        captured["ghost_high_f"] = kwargs["ghost_high_f"]
        captured["ghost_low_v"] = kwargs["ghost_low_v"]
        captured["ghost_high_v"] = kwargs["ghost_high_v"]
        return jnp.zeros_like(f)

    monkeypatch.setattr(parallel_terms, "hermes_div_par_mod", fake_div_par_mod)

    params = SimpleNamespace(
        parallel_transform="shifted",
        parallel_shift_interp="linear",
        parallel_limiter="mc",
        parallel_current_limiter="same",
        parallel_flux_scheme="hermes_mirror",
        parallel_fixflux=True,
        use_gpar_flux=False,
        parallel_sign=1.0,
        parallel_boundary_flux_scale=1.0,
    )
    geom = SimpleNamespace(
        grid=SimpleNamespace(open_field_line=True, dz=1.0),
        shift_idx=jnp.asarray([[1.0, 1.0], [-1.0, -1.0]]),
        to_field_aligned=object(),
        to_field_aligned_nox=lambda arr: arr,
        from_field_aligned_nox=lambda arr: arr,
        jacobian=None,
        metric_dy=None,
        dpar_factor=None,
        gpar=None,
    )
    ctx = SimpleNamespace(params=params, geom=geom)

    ghost_low_f = jnp.asarray([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
    ghost_high_f = jnp.asarray([[20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]])
    ghost_low_v = ghost_low_f + 100.0
    ghost_high_v = ghost_high_f + 100.0

    _dpar_flux_conservative(
        ctx,
        jnp.ones((2, 2, 4)),
        jnp.ones((2, 2, 4)),
        wave=jnp.ones((2, 2, 4)),
        ghost_low_f=ghost_low_f,
        ghost_high_f=ghost_high_f,
        ghost_low_v=ghost_low_v,
        ghost_high_v=ghost_high_v,
    )

    np.testing.assert_allclose(
        np.asarray(captured["ghost_low_f"]),
        np.asarray(
            _shift_boundary_flux_to_field_aligned(ghost_low_f, params=params, geom=geom, z_index=0)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(captured["ghost_high_f"]),
        np.asarray(
            _shift_boundary_flux_to_field_aligned(
                ghost_high_f, params=params, geom=geom, z_index=-1
            )
        ),
    )
    np.testing.assert_allclose(
        np.asarray(captured["ghost_low_v"]),
        np.asarray(
            _shift_boundary_flux_to_field_aligned(ghost_low_v, params=params, geom=geom, z_index=0)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(captured["ghost_high_v"]),
        np.asarray(
            _shift_boundary_flux_to_field_aligned(
                ghost_high_v, params=params, geom=geom, z_index=-1
            )
        ),
    )
