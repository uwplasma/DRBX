"""Linear stability analysis for drift-reduced Braginskii models.

A model is linearized about an equilibrium state by taking the Jacobian of its
right-hand side, ``A = d(rhs)/d(state)``. Writing a perturbation as
``delta ~ exp(lambda * t)``, the eigenvalues ``lambda = gamma + i * Omega`` of
``A`` give the growth rate ``gamma = Re(lambda)`` and the oscillation frequency
``Omega = Im(lambda)`` of each linear eigenmode.

Two entry points are provided:

- :func:`jacobian_operator` builds the dense Jacobian of an arbitrary
  JAX-differentiable right-hand side about an equilibrium (via ``jax.jacfwd``),
  suitable for small grids or single-Fourier-mode reductions.
- :func:`eigenmodes` diagonalizes a linear operator and returns the growth
  rates, frequencies, and eigenvectors sorted by decreasing growth rate.

Both are pure JAX and therefore ``jit``/``grad``-transformable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "LinearModes",
    "jacobian_operator",
    "eigenmodes",
    "dominant_mode",
]


@dataclass(frozen=True)
class LinearModes:
    """Eigenmodes of a linear operator, sorted by decreasing growth rate.

    ``growth_rates`` and ``frequencies`` are the real and imaginary parts of the
    eigenvalues ``lambda`` (``delta ~ exp(lambda t)``); ``eigenvalues`` and
    ``eigenvectors`` are the raw complex spectrum.
    """

    growth_rates: jnp.ndarray
    frequencies: jnp.ndarray
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray

    @property
    def dominant_growth_rate(self) -> jnp.ndarray:
        return self.growth_rates[0]

    @property
    def dominant_frequency(self) -> jnp.ndarray:
        return self.frequencies[0]


def jacobian_operator(rhs, equilibrium):
    """Dense Jacobian ``A = d(rhs)/d(state)`` of a JAX rhs about ``equilibrium``.

    ``rhs`` maps a state vector (1-D array) to its time derivative. The returned
    matrix has shape ``(n, n)`` for a length-``n`` state. For a genuinely linear
    ``rhs`` this recovers the operator exactly; for a nonlinear one it is the
    tangent operator at the equilibrium.
    """

    equilibrium = jnp.asarray(equilibrium)
    return jax.jacfwd(rhs)(equilibrium)


def eigenmodes(operator) -> LinearModes:
    """Diagonalize ``operator`` and return modes sorted by decreasing growth."""

    operator = jnp.asarray(operator)
    eigenvalues, eigenvectors = jnp.linalg.eig(operator)
    order = jnp.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return LinearModes(
        growth_rates=eigenvalues.real,
        frequencies=eigenvalues.imag,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def dominant_mode(operator) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(growth_rate, frequency)`` of the fastest-growing eigenmode."""

    modes = eigenmodes(operator)
    return modes.dominant_growth_rate, modes.dominant_frequency
