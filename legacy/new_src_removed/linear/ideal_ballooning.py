from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_la


def ideal_ballooning_gamma_hat(
    *,
    shat: float,
    alpha: float,
    Lh: float = float(2 * jnp.pi),
    nh: int = 513,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Ideal-ballooning growth rate from Halpern et al. (2013) Eq. (16) (Dirichlet BCs).

    This implements the Sturm–Liouville eigenproblem (Halpern et al., Phys. Plasmas 20, 052306):

        -d/dh[ w(h) dphi/dh ] - alpha * K(h) * phi + (gamma_hat^2) * w(h) * phi = 0

    where:

        w(h) = 1 + (shat * h)^2
        K(h) = cos(h) + shat * h * sin(h)

    on a finite ballooning-angle domain of length Lh with Dirichlet boundary conditions:

        phi(h = ±Lh/2) = 0.

    The returned value is gamma_hat >= 0 (square-root of minus the most negative generalized
    eigenvalue, if any).

    Notes
    -----
    - This is an *ideal-MHD-like* reduced eigenproblem, separate from the drift-reduced Braginskii
      models solved elsewhere in `jaxdrb`.
    - The discretization uses second-order conservative finite differences for the variable-
      coefficient operator and reduces the generalized eigenproblem to a symmetric standard
      eigenproblem via B^{-1/2} A B^{-1/2}.
    """

    nh = int(nh)
    if nh < 9:
        raise ValueError("nh must be >= 9.")
    if not (Lh > 0):
        raise ValueError("Lh must be > 0.")

    h = jnp.linspace(-0.5 * Lh, 0.5 * Lh, nh, endpoint=True)
    dh = h[1] - h[0]

    # Interior (Dirichlet endpoints are excluded from unknowns).
    hi = h[1:-1]

    w = 1.0 + (shat * hi) ** 2
    w = jnp.maximum(w, eps)
    K = jnp.cos(hi) + shat * hi * jnp.sin(hi)

    # Variable-coefficient diffusion operator:
    #   -(d/dh)[ w dphi/dh ]  with w evaluated at half steps by averaging.
    w_full = 1.0 + (shat * h) ** 2
    w_half = 0.5 * (w_full[1:] + w_full[:-1])  # length nh-1, at half nodes
    w_m = w_half[:-1]  # i-1/2 for interior i=1..nh-2
    w_p = w_half[1:]  # i+1/2

    # Build the symmetric tridiagonal matrix for the interior unknowns.
    # Diffusion part: A = -d/dh[w d/dh] is tridiagonal with
    #   A_ii   = (w_{i-1/2} + w_{i+1/2}) / dh^2
    #   A_i,i+1 = A_{i+1,i} = -w_{i+1/2} / dh^2
    d = (w_m + w_p) / (dh**2)
    e = -w_p[:-1] / (dh**2)  # coupling between i and i+1

    # Add potential term: -alpha*K (destabilizing).
    d = d - alpha * K

    # Generalized eigenproblem: A phi = lambda * B phi with B = diag(w).
    # Reduce to symmetric tridiagonal S = B^{-1/2} A B^{-1/2}.
    invsqrt_w = 1.0 / jnp.sqrt(w)
    dS = d * (invsqrt_w**2)  # d / w
    eS = e * (invsqrt_w[:-1] * invsqrt_w[1:])  # e / sqrt(w_i w_{i+1})

    # In this normalization, the generalized eigenvalues correspond to -gamma_hat^2.
    # Instability requires the most negative eigenvalue of S.
    lam_min = jsp_la.eigh_tridiagonal(dS, eS, eigvals_only=True, select="i", select_range=(0, 0))[0]
    gamma2 = jnp.maximum(-lam_min, 0.0)
    return jnp.sqrt(gamma2)


ideal_ballooning_gamma_hat_jit = jax.jit(ideal_ballooning_gamma_hat)
