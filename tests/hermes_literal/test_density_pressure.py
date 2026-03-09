from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.hermes_literal import (
    DensityTransformResult,
    Field3DLayout,
    PressureTransformResult,
    density_transform_impl,
    pressure_transform_impl,
)


def _layout() -> Field3DLayout:
    return Field3DLayout(pstart=2, pend=5, xstart=2, xend=5, guard_width=2)


def test_density_transform_impl_floors_negative_density() -> None:
    arr = jnp.zeros((8, 8, 4), dtype=jnp.float64).at[3, 3, 1].set(-2.0)
    out = density_transform_impl(arr, layout=_layout())
    assert isinstance(out, DensityTransformResult)
    assert float(np.min(np.asarray(out.density))) >= 0.0


def test_pressure_transform_impl_reconstructs_temperature_consistently() -> None:
    n = jnp.full((8, 8, 4), 2.0, dtype=jnp.float64)
    p = jnp.full((8, 8, 4), 6.0, dtype=jnp.float64)
    out = pressure_transform_impl(p, n, layout=_layout(), density_floor=1e-6)
    assert isinstance(out, PressureTransformResult)
    np.testing.assert_allclose(np.asarray(out.temperature), 3.0)
    np.testing.assert_allclose(np.asarray(out.pressure), 6.0)
