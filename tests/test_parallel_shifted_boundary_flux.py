from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.terms.parallel import _shift_boundary_flux_to_field_aligned


def test_shifted_boundary_flux_passthrough_when_not_shifted_transform() -> None:
    flux = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
    params = SimpleNamespace(parallel_transform="none")
    geom = SimpleNamespace(shift_idx=jnp.asarray([[1.0]]))
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(flux))


def test_shifted_boundary_flux_applies_integer_shift() -> None:
    flux = jnp.asarray([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])
    params = SimpleNamespace(parallel_transform="shifted")
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
    params = SimpleNamespace(parallel_transform="shifted")
    geom = SimpleNamespace(shift_idx=jnp.asarray([[0.5]]))
    out = _shift_boundary_flux_to_field_aligned(flux, params=params, geom=geom, z_index=0)
    expect = np.asarray([[0.5, 1.5, 2.5, 1.5]])
    np.testing.assert_allclose(np.asarray(out), expect)
