"""The two absolute-matrix actions used by SBP curvature."""

from __future__ import annotations

import jax
import jax.numpy as jnp


#: Evaluations of the SBP curvature derivative and matrix wall SAT absolute-matrix action ``|M| jump``:
#: ``"full_4x4_eigendecomposition"`` the 4x4 ``eig`` / ``inv`` / ``cond`` of the reference calculation (kept to reproduce frozen campaigns bitwise);
#: ``"closed_form"`` (default) the same spectral action on the 3x3 ``(n, Te, Ti)`` block alone (the omega column of the
#: principal matrix is zero), from the one real root of a cubic and its Sylvester projector, with no LAPACK.
ABSOLUTE_METHODS = ("full_4x4_eigendecomposition", "closed_form")


def _absolute_primal(matrix, jump):
    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    inverse = jnp.linalg.inv(eigenvectors)
    condition = jnp.linalg.cond(eigenvectors)
    real = jnp.real(eigenvalues)
    valid = (jnp.max(jnp.abs(jnp.imag(eigenvalues)), axis=-1)
             <= 1e-10*(1+jnp.max(jnp.abs(real), axis=-1)))
    valid &= jnp.isfinite(condition) & (condition <= 1e8)
    spectral = jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors,
                                   jnp.abs(real)*jnp.einsum('...ij,...j->...i', inverse, jump)))
    fallback = jnp.linalg.norm(matrix, axis=(-2, -1))[..., None]*jump
    return jnp.where(valid[..., None], spectral, fallback), ~valid, eigenvalues, eigenvectors, inverse


@jax.custom_jvp
def _absolute_matrix_action(matrix, jump):
    return _absolute_primal(matrix, jump)[0]


@_absolute_matrix_action.defjvp
def _absolute_matrix_action_jvp(primals, tangents):
    """Fréchet derivative of V|Lambda|V^-1 away from branch crossings.

    This differentiates the qualified matrix action itself, without asking
    JAX for unsupported nonsymmetric eigenvector derivatives.  At repeated
    or sign-changing eigenvalues the action is not certified differentiable.
    """
    matrix, jump = primals
    dmatrix, djump = tangents
    action, invalid, eigenvalues, eigenvectors, inverse = _absolute_primal(matrix, jump)
    eigenvalues = jax.lax.stop_gradient(eigenvalues)
    eigenvectors = jax.lax.stop_gradient(eigenvectors)
    inverse = jax.lax.stop_gradient(inverse)
    invalid = jax.lax.stop_gradient(invalid)
    lam = jnp.real(eigenvalues)
    gap = lam[..., :, None]-lam[..., None, :]
    abs_gap = jnp.abs(lam)[..., :, None]-jnp.abs(lam)[..., None, :]
    same = jnp.abs(gap) <= 1e-12*(1+jnp.maximum(jnp.abs(lam[..., :, None]),jnp.abs(lam[..., None, :])))
    divided = jnp.where(same, jnp.sign(lam[..., :, None]), abs_gap/jnp.where(same, 1, gap))
    transformed = jnp.einsum('...ij,...jk,...kl->...il', inverse, dmatrix, eigenvectors)
    dabs = jnp.real(jnp.einsum('...ij,...jk,...kl->...il', eigenvectors,
                                   divided*transformed, inverse))
    spectral_tangent = jnp.einsum('...ij,...j->...i', dabs, jump)
    spectral_tangent += jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors,
                              jnp.abs(lam)*jnp.einsum('...ij,...j->...i',inverse,djump)))
    norm = jnp.linalg.norm(matrix, axis=(-2,-1))
    dnorm = jnp.sum(matrix*dmatrix, axis=(-2,-1))/jnp.maximum(norm,1e-300)
    fallback_tangent = dnorm[...,None]*jump+norm[...,None]*djump
    return action, jnp.where(invalid[...,None],fallback_tangent,spectral_tangent)


def _p06_absolute_action(matrix, jump):
    """The reference calculation's real-spectrum/condition test and Frobenius fallback."""
    action = _absolute_matrix_action(matrix, jump)
    # The validity mask is diagnostic and has no meaningful derivative.
    invalid = _absolute_primal(jax.lax.stop_gradient(matrix), jax.lax.stop_gradient(jump))[1]
    return action, invalid


def _validated_absolute_method(method):
    if method not in ABSOLUTE_METHODS:
        raise ValueError(f"absolute_method must be one of {ABSOLUTE_METHODS}, got {method!r}")
    return method


def _frobenius_fallback(matrix, jump):
    """``||M||_F jump`` (the fallback of every method), differentiable also at ``M = 0``."""
    square = jnp.sum(matrix*matrix, axis=(-2, -1))
    norm = jnp.where(square > 0, jnp.sqrt(jnp.where(square > 0, square, 1.)), 0.)
    return norm[..., None]*jump


# --------------------------------------------------------------------------
# "closed_form": cubic root + Sylvester projector, no LAPACK
# --------------------------------------------------------------------------
# With ``D = diag(n_safe, Te, Te)`` the (n, Te, Ti) block of ``P`` is ``A = Te D Abar D^-1`` where
#
#     Abar = [[2, 2, 2 tau], [4/3, 14/3, 4 tau/3], [4 t/3, 4 t/3, -2 tau t]],   t = Ti/Te,  r = tau t,
#
# and ``mu = lambda/Te`` are the roots of ``p(mu) = 9 mu^3 + (18 r - 60) mu^2 + (60 - 160 r) mu + 200 r``.  The
# discriminant ``6400 (r+1)(63 r^3 + 619 r^2 + 630 r + 90)/729`` is positive, so there are three distinct real roots,
# and ``p(0) = 200 r > 0``, ``p(1) > 0``, ``p(2) < 0``, ``p(10) > 0`` locate them:
# ``mu_- < 0 < 1 < mu_1 < 2 < mu_2 < 10`` (``mu_- = 0`` at ``r = 0``).  The signs are therefore known, ``sign(Abar) = I - 2 E_-`` and, as
# ``Abar E_- = mu_- E_-``, ``|Abar| = Abar - 2 mu_- E_-``.  Only ``mu_-`` and the Sylvester projector
# ``E_- = (Abar - mu_1)(Abar - mu_2)/((mu_- - mu_1)(mu_- - mu_2))`` are needed, the other roots only through
# ``mu_1 + mu_2`` and ``mu_1 mu_2`` (Vieta).  This is ``sum_i |mu_i| E_i`` with ``sum_i E_i = I``, without the
# cancellation of the two positive roots that the trigonometric formula has at ``r >~ 1e5``.  ``sign(0) = 0`` at
# ``r = 0`` (``Ti = 0`` or ``tau = 0``), where ``|lambda| = 0`` does not contribute to ``|A|`` and
# ``sign(Abar) = I - E_-`` (only the ``c^T sign(A)`` row sees the convention).


@jax.custom_jvp
def _curvature_negative_root(r):
    """The negative root (zero at ``r = 0``) of ``p``: Viète's trigonometric formula and one Newton step."""
    a2 = (18*r-60)/9; a1 = (60-160*r)/9; a0 = 200*r/9
    shift = a2/3
    p = a1-a2*a2/3
    q = 2*a2**3/27-a2*a1/3+a0
    radius = jnp.sqrt(-p/3)                                           # p <= -8.1 for r >= 0
    angle = jnp.arccos(jnp.clip(-q/(2*radius**3), -1., 1.))/3
    mu = 2*radius*jnp.cos(angle-4*jnp.pi/3)-shift                     # k = 2 of Viète: the smallest root
    mu = mu-(((9*mu+18*r-60)*mu+60-160*r)*mu+200*r)/((27*mu+2*(18*r-60))*mu+60-160*r)
    return jnp.where(r == 0, 0., mu)                                  # exactly zero root at r = 0


@_curvature_negative_root.defjvp
def _curvature_negative_root_jvp(primals, tangents):
    """Implicit derivative ``-p_r/p_mu`` (the arccos of the trigonometric form is singular as its argument -> -1)."""
    (r,), (dr,) = primals, tangents
    mu = _curvature_negative_root(r)
    dp_dmu = (27*mu+2*(18*r-60))*mu+60-160*r
    dp_dr = (18*mu-160)*mu+200
    return mu, -dp_dr/dp_dmu*dr


def _positive_pair_invariants(r, mu):
    """``(mu_1 + mu_2, mu_1 mu_2)`` of the two positive roots from Vieta, accurate over all ``r``.

    Small ``|mu_-|``: ``s = e1 - mu_-`` and ``p = e2 - mu_- s``; otherwise ``p = e3/mu_-`` and ``s = (e2 - p)/mu_-``
    (each pair is free of cancellation in its range).
    """
    e1 = (60-18*r)/9; e2 = (60-160*r)/9; e3 = -200*r/9
    small = jnp.abs(mu) < 1
    s_small = e1-mu
    p_small = e2-mu*s_small
    mu_big = jnp.where(small, -1., mu)
    p_big = e3/mu_big
    s_big = (e2-p_big)/mu_big
    return jnp.where(small, s_small, s_big), jnp.where(small, p_small, p_big)


# ``psi = phi + tau*n*Ti`` (``"phi_plus_tau_pi"``).  The curvature matrix gains ``dpsi/dn = tau Ti`` and
# ``dpsi/dTi = tau n`` (see ``curvature_principal_matrix``), so with ``r = tau t``, ``s = n tau t``
#
#     Abar = [[2 (1+s), 2, 2 tau n], [4/3 (1+s), 14/3, 4 tau n/3], [4 t (1+s)/3, 4 t/3, t tau (4 n - 10)/3]]
#
# (D unchanged) and the cubic of ``mu = lambda/Te`` is
#
#     p(mu) = 9 mu^3 + (30 (r - s) - 60) mu^2 + (60 + 100 s - 200 r - 60 r s) mu + 200 r (1 + s).
#
# Its discriminant is positive for every ``r, s >= 0`` (three distinct real roots) and ``p(0) = 200 r (1 + s) >= 0``
# with ``p -> -inf``: exactly one root is negative (zero at ``r = 0``) and two are positive, as for the ``Ti`` form.
# Hence the same decomposition ``|Abar| = Abar - 2 mu_- E_-`` with the Sylvester projector holds; only the cubic
# (negative root, Vieta invariants of the other two) and the matrix entries differ.


@jax.custom_jvp
def _cubic_negative_root(a2, a1, a0):
    """The negative root (zero if ``a0 == 0``) of the monic cubic ``mu^3 + a2 mu^2 + a1 mu + a0`` with three real roots,
    two of them positive: Viete's trigonometric formula (smallest root) and a Newton step."""
    shift = a2/3
    p = a1-a2*a2/3
    q = 2*a2**3/27-a2*a1/3+a0
    radius = jnp.sqrt(jnp.maximum(-p/3, 1e-300))
    angle = jnp.arccos(jnp.clip(-q/(2*radius**3), -1., 1.))/3
    mu = 2*radius*jnp.cos(angle-4*jnp.pi/3)-shift
    for _ in range(2):
        mu = mu-(((mu+a2)*mu+a1)*mu+a0)/((3*mu+2*a2)*mu+a1)
    return jnp.where(a0 == 0, 0., mu)


@_cubic_negative_root.defjvp
def _cubic_negative_root_jvp(primals, tangents):
    """Implicit derivative ``-(mu^2 da2 + mu da1 + da0)/p'(mu)``."""
    a2, a1, a0 = primals
    d2, d1, d0 = tangents
    mu = _cubic_negative_root(a2, a1, a0)
    dp_dmu = (3*mu+2*a2)*mu+a1
    return mu, -(mu*mu*d2+mu*d1+d0)/dp_dmu


def _positive_pair_invariants_general(e1, e2, e3, mu):
    """:func:`_positive_pair_invariants` from the elementary symmetric functions ``e1, e2, e3`` of the three roots."""
    small = jnp.abs(mu) < 1
    s_small = e1-mu
    p_small = e2-mu*s_small
    mu_big = jnp.where(small, -1., mu)
    p_big = e3/mu_big
    s_big = (e2-p_big)/mu_big
    return jnp.where(small, s_small, s_big), jnp.where(small, p_small, p_big)


def _absolute_action_closed_form(n, te, ti, b, tau, scale, matrix, jump, floor, psi="phi_plus_tau_ti"):
    """``(|M| jump, invalid)`` for ``M = scale * P(n, Te, Ti, B, tau)``, no eigensolver.

    ``invalid`` marks the non-physical nodes (``n`` or ``Te`` not above ``floor``, ``Ti < 0``, ``tau < 0``, or a
    non-finite input), which take the Frobenius fallback on ``matrix``.  They are evaluated at a safe state first, so
    neither the values nor the derivatives of the closed form see their data.
    """
    n, te, ti, b, tau, scale = jnp.broadcast_arrays(*(jnp.asarray(x) for x in (n, te, ti, b, tau, scale)))
    valid = (jnp.isfinite(n) & jnp.isfinite(te) & jnp.isfinite(ti) & jnp.isfinite(b) & jnp.isfinite(tau)
             & jnp.isfinite(scale) & (n > floor) & (te > floor) & (ti >= 0) & (tau >= 0))
    one = jnp.ones_like(te)
    n, te, ti, b, tau = (jnp.where(valid, x, one) for x in (n, te, ti, b, tau))
    scale = jnp.where(valid, scale, 0.)
    n = jnp.maximum(n, 1e-30)                         # as the principal matrix
    r = tau*ti/te
    t = ti/te
    if psi == "phi_plus_tau_pi":
        s = n*r
        a2 = (30*(r-s)-60)/9
        a1 = (60+100*s-200*r-60*r*s)/9
        a0 = 200*r*(1+s)/9
        mu = _cubic_negative_root(a2, a1, a0)
        pair_sum, pair_product = _positive_pair_invariants_general(-a2, a1, -a0, mu)
        tn = tau*n
        apply = lambda v0, v1, v2: (2*(1+s)*v0+2*v1+2*tn*v2,
                                    4*((1+s)*v0+3.5*v1+tn*v2)/3,
                                    4*t*((1+s)*v0+v1)/3+t*tau*(4*n-10)/3*v2)
    else:
        mu = _curvature_negative_root(r)
        pair_sum, pair_product = _positive_pair_invariants(r, mu)
        apply = lambda v0, v1, v2: (2*(v0+v1)+2*tau*v2, (4*v0+14*v1+4*tau*v2)/3, 4*t/3*(v0+v1)-2*tau*t*v2)
    # u in the scaled frame D^-1 u, and Abar u, Abar^2 u
    u0 = jump[..., 0]/n; u1 = jump[..., 1]/te; u2 = jump[..., 2]/te
    a0, a1, a2 = apply(u0, u1, u2)
    b0, b1, b2 = apply(a0, a1, a2)
    # E_- u = (Abar - mu_1)(Abar - mu_2) u / ((mu_- - mu_1)(mu_- - mu_2))
    denominator = mu*mu-pair_sum*mu+pair_product
    e = tuple((bb-pair_sum*aa+pair_product*uu)/denominator for bb, aa, uu in ((b0, a0, u0), (b1, a1, u1), (b2, a2, u2)))
    sign_factor = 1+(r > 0)                           # sign(0) = 0: I - E_- instead of I - 2 E_- at r = 0
    absolute = tuple(aa-2*mu*ee for aa, ee in zip((a0, a1, a2), e))                    # |Abar| u
    sgn = tuple(uu-sign_factor*ee for uu, ee in zip((u0, u1, u2), e))                  # sign(Abar) u
    magnitude = jnp.abs(scale)
    b2s = 2*b*b
    omega = b2s*((te+tau*ti)*sgn[0]+te*sgn[1]+tau*te*sgn[2])                           # c^T D sign(Abar) D^-1 u
    spectral = magnitude[..., None]*jnp.stack(
        (te*n*absolute[0], te*te*absolute[1], te*te*absolute[2], omega), axis=-1)
    fallback = _frobenius_fallback(matrix, jump)
    return jnp.where(valid[..., None], spectral, fallback), ~valid
