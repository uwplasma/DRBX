from __future__ import annotations

import jax.numpy as jnp


def poisson_bracket_fourier(kx: float, ky: float, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """Poisson bracket [phi, f] for a *single* Fourier mode.

    For perturbations with the same (kx, ky), the self-nonlinearity vanishes:
      [phi, f] = (∂x phi)(∂y f) - (∂y phi)(∂x f) = 0

    This is provided as a placeholder for future multi-mode (spectral) extensions.
    """

    _ = (kx, ky, phi, f)
    return jnp.zeros_like(f)
