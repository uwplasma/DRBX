"""Electromagnetic closure of the ``fci_braginskii`` model (``[physics] electromagnetic = true``).

The evolved ``Ve`` slot holds the canonical electron momentum ``w = Ve - (mi/me) A``; the physical
velocity ``Ve = w + (mi/me) A`` follows from Ampere's law with electron inertia,
``(-L_perp + beta (mi/me) n) A = beta n (Vi - w)``, solved once per stage (``LocalFciDrbEBRhs.em_physical``).
The inductive term then never appears: ``dw/dt`` is the electrostatic ``Ve`` row evaluated at the physical
``Ve``.  ``A`` enters the parallel gradient pairs as magnetic flutter, ``G -> G + F``,
``F f = -rho_star {A, f} / B`` (the E x B advection ``rho_star {phi, f} / B`` with ``phi -> A`` and the opposite sign,
``b1 = grad A x b / B``), with ``D -> D + D_F``, ``D_F = -M^-1 F^T M`` (see the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .fci_support_pair import build_weighted_negative_adjoint


@dataclass(frozen=True)
class FciEm:
    """Ampere coefficient ``beta = mu0 n0 Te0 / (rho_star^2 B0^2)`` of the metre-normalized model; ``A = 0`` on the walls."""

    beta: float = 1.0e-3

    def __post_init__(self) -> None:
        if not self.beta > 0.0:
            raise ValueError("electromagnetic needs beta > 0")


def flutter_pair(gradient, divergence, flutter, mass, active, dual):
    """``(G + F, D + D_F)`` with ``D_F = -M^-1 F^T M`` for a constant exact linear ``F``.

    ``D + D_F = -M^-1 (G + F)^T M`` stays the exact weighted adjoint of ``G + F`` and
    ``sum M (D + D_F) f = 0``, so every conservation statement of the electrostatic pair
    holds with the flutter added.  ``F`` acts on ``active`` cells and returns on ``dual`` rows.
    """

    masked = lambda values: jnp.where(dual, flutter(jnp.where(active, values, 0.0)), 0.0)  # noqa: E731
    adjoint = build_weighted_negative_adjoint(masked, mass, mass, primal_active=active, dual_active=dual)
    return (
        lambda values: gradient(values) + (masked(values) if values.ndim == mass.ndim else jax.vmap(masked)(values)),
        lambda values: divergence(values) + adjoint(values),
    )
