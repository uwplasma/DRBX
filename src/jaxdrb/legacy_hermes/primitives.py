from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp


@dataclass(frozen=True)
class Stencil1D:
    """Mirror of Hermes `Stencil1D` in `src/div_ops.cxx`.

    The face values `L` and `R` are optional on input and populated by
    limiter helpers such as `mc_limiter`.
    """

    c: jnp.ndarray
    m: jnp.ndarray
    p: jnp.ndarray
    mm: jnp.ndarray | None = None
    pp: jnp.ndarray | None = None
    L: jnp.ndarray | None = None
    R: jnp.ndarray | None = None


def minmod(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Return the minimum-magnitude same-sign slope.

    This mirrors Hermes `minmod()` in
    `/Users/rogerio/local/hermes-3/src/div_ops.cxx`.
    """

    same_sign = (a * b > 0.0) & (a * c > 0.0)
    mag = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
    return jnp.where(same_sign, jnp.sign(a) * mag, 0.0)


def mc_limiter(stencil: Stencil1D) -> Stencil1D:
    """Apply the Hermes monotonized-central limiter.

    Source of truth:
    `/Users/rogerio/local/hermes-3/src/div_ops.cxx`, function `MC`.
    """

    slope = minmod(
        2.0 * (stencil.p - stencil.c),
        0.5 * (stencil.p - stencil.m),
        2.0 * (stencil.c - stencil.m),
    )
    return replace(stencil, L=stencil.c - 0.5 * slope, R=stencil.c + 0.5 * slope)


def limit_free(
    fm: jnp.ndarray | float,
    fc: jnp.ndarray | float,
    mode: float,
) -> jnp.ndarray:
    """Mirror Hermes `limitFree`.

    Source of truth:
    `/Users/rogerio/local/hermes-3/src/sheath_boundary_simple.cxx`, function
    `limitFree`.

    Mode mapping:
    - `0`: capped exponential free boundary
    - `1`: exponential free boundary
    - `2`: linear free boundary
    """

    mode_f = float(mode)
    if mode_f not in (0.0, 1.0, 2.0):
        raise ValueError(f"Unknown limit_free mode {mode!r}; expected 0, 1, or 2.")

    fm_arr = jnp.asarray(fm)
    fc_arr = jnp.asarray(fc)
    low_density = fm_arr < 1.0e-10
    prevent_increase = (mode_f == 0.0) & (fm_arr < fc_arr)
    safe_fm = jnp.where(low_density, 1.0, fm_arr)

    if mode_f in (0.0, 1.0):
        extrapolated = jnp.square(fc_arr) / safe_fm
    else:
        extrapolated = 2.0 * fc_arr - fm_arr

    return jnp.where(prevent_increase | low_density, fc_arr, extrapolated)
