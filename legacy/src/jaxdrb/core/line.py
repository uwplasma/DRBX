from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class LineEquilibrium(eqx.Module):
    """Background profiles along the field line used by line/flux-tube workflows."""

    n0: jnp.ndarray
    Te0: jnp.ndarray

    @classmethod
    def constant(
        cls,
        nl: int,
        *,
        n0: float = 1.0,
        Te0: float = 1.0,
        dtype=jnp.float64,
    ) -> "LineEquilibrium":
        return cls(
            n0=jnp.full((nl,), float(n0), dtype=dtype),
            Te0=jnp.full((nl,), float(Te0), dtype=dtype),
        )
