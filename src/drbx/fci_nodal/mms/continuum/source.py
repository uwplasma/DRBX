"""Pointwise continuum kernels used by the evolved manufactured solution."""
from __future__ import annotations


import jax.numpy as jnp


from ...operators.perp.curvature_principal import curvature_principal_matrix                    # noqa: E402
from ...operators.perp.sbp_curvature import PSI_VARIANTS                                 # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# pointwise continuum kernels (field axis last; logical frame)
# ---------------------------------------------------------------------------------------------------------------------


def point_bracket(h, jac, ga, gb):
    """``-h . (grad a x grad b) / |J|`` (no ``rho_star``): ``h (..., 3)``, ``jac (...)``, ``ga (..., 3)``, ``gb (..., F, 3)`` ->
    ``(..., F)``. The frozen references write ``-(h x grad a) . grad b / (|J| rho*)`` (the same triple product)."""
    cross = jnp.cross(h[..., None, :], ga[..., None, :])
    return -jnp.sum(cross * gb, axis=-1) / jac[..., None]


def laplacian_cont(A, divA, jac, G, H):
    """``(divA . grad f + A : grad grad f) / |J|``: ``G (..., F, 3)``, ``H (..., F, 3, 3)`` -> ``(..., F)``."""
    num = jnp.einsum("...j,...fj->...f", divA, G) + jnp.einsum("...ij,...fij->...f", A, H)
    return num / jac[..., None]


def continuum_curvature_rhs(values, gradients, K, B, tau=1.0, psi="phi_plus_tau_pi"):
    """Continuum curvature RHS ``(..., 4)`` at nodes from
    ``values (5, ...)``, logical ``gradients (5, ..., 3)``, logical ``K (..., 3)`` and ``B (...)`` (no ``rho_star``)."""
    if psi not in PSI_VARIANTS:
        raise ValueError(f"psi must be one of {PSI_VARIANTS}, got {psi!r}")
    values, gradients, K, B = (jnp.asarray(x, dtype=jnp.float64) for x in (values, gradients, K, B))
    C = jnp.einsum("...d,f...d->f...", K, gradients) / B
    n, te, ti = values[0], values[1], values[2]
    M = curvature_principal_matrix(n, te, ti, B, tau, psi=psi)
    material = jnp.einsum("...ij,j...->...i", M, C[:4])
    if psi == "phi_plus_tau_pi":
        cpsi = C[4] + tau * (ti * C[0] + n * C[2])
    else:
        cpsi = C[4] + tau * C[2]
    coeff = jnp.stack((-2.0 * n, -4.0 * te / 3.0, -4.0 * ti / 3.0, jnp.zeros_like(n)), axis=-1)
    return material + coeff * cpsi[..., None]


def psi_derivatives(psi: str, tau, V, G, H):
    """``psi = phi + tau p`` with its logical gradient and Hessian from the stacks of ``(n, Te, Ti, Omega, phi)``: ``V (..., 5)``,
    ``G (..., 5, 3)``, ``H (..., 5, 3, 3)`` -> ``(psi (...), grad (..., 3), hess (..., 3, 3))``."""
    n, Ti, ph = V[..., 0], V[..., 2], V[..., 4]
    gn, gT, gp = G[..., 0, :], G[..., 2, :], G[..., 4, :]
    if psi == "phi_plus_tau_pi":
        p = n * Ti
        gradp = Ti[..., None] * gn + n[..., None] * gT
        hessp = (Ti[..., None, None] * H[..., 0, :, :] + n[..., None, None] * H[..., 2, :, :]
                 + gn[..., :, None] * gT[..., None, :] + gT[..., :, None] * gn[..., None, :])
    else:
        p, gradp, hessp = Ti, gT, H[..., 2, :, :]
    return ph + tau * p, gp + tau * gradp, H[..., 4, :, :] + tau * hessp


# ---------------------------------------------------------------------------------------------------------------------
# continuum source
# ---------------------------------------------------------------------------------------------------------------------
