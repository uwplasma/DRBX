"""Live P05 scalar jumps and P06 characteristic face corrections.

Inputs are prepared common/anchored-side point samples and fixed geometry
arrays.  Point-row preparation, prescribed trace evaluation, and owner layout
adaptation are separate caller responsibilities.  No manufactured field enters
these numerical kernels.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix
from drbx.native.fci_operators import _curvature_bc_characteristic_wall_states

#: Evaluations of the P06 absolute-matrix action ``|M| jump``:
#: ``"lapack4"`` the 4x4 ``eig`` / ``inv`` / ``cond`` of the campaign; ``"block_lapack"`` the same
#: spectral action on the 3x3 ``(n, Te, Ti)`` block alone (the omega column of the principal matrix is zero);
#: ``"closed_form"`` (default) the same block from the one real root of a cubic and its Sylvester projector, with no LAPACK.
ABSOLUTE_METHODS = ("lapack4", "block_lapack", "closed_form")


def p05_scalar_face_jump(common_gradient, lower_value, upper_value, h_covariant_over_b,
                         quadrature_weight, face_axis, pairs):
    """Frozen P05 q3 jump, one oriented scalar integral per face and pair.

    ``common_gradient`` is ``(faces, q, 3, fields)``; side values are
    ``(faces, q, fields)``.  Wall exterior values must be supplied from the
    prescribed Dirichlet trace by the caller.

    The face value is ``+1/2 sum_q w |U_generator| (upper - lower)_transported``.  With the lower-plus /
    upper-minus scatter of :func:`scatter_p05_jump`, each owner relaxes toward its neighbour (the dissipative
    upwind orientation): for piecewise-constant traces ``sum_o g_o scatter_o = -1/2 sum_f sum_q w |U|
    (g_up - g_lo)^2 <= 0``.  The sign was corrected on 2026-10-03; before that the prefactor was ``-1/2`` and the
    term was anti-dissipative (records saved earlier carry the opposite jump sign).
    """
    gradient = jnp.asarray(common_gradient)
    lower = jnp.asarray(lower_value); upper = jnp.asarray(upper_value)
    h = jnp.asarray(h_covariant_over_b)
    weight = jnp.asarray(quadrature_weight)
    axis = jnp.asarray(face_axis, dtype=jnp.int32)
    pairs = jnp.asarray(pairs, dtype=jnp.int32)
    # U is the generator's coordinate advection density.  There is no 1/J
    # here: the campaign's face measure and oriented scatter already define
    # the integral.  This scalar speed is distinct from P06's eigensystem.
    velocity = -jnp.cross(h[..., None, :], jnp.moveaxis(gradient, -1, -2))
    speed = jnp.take_along_axis(velocity, axis[:, None, None, None], axis=-1)[..., 0]
    jump = upper-lower
    generator = pairs[:, 0]; transported = pairs[:, 1]
    return .5*jnp.einsum('fq,fqp->fp', weight,
                          jnp.abs(speed[..., generator])*jump[..., transported])


def scatter_p05_jump(face_jump, lower_owner, upper_owner, owner_volume):
    """P05 owner correction: lower plus, upper minus, stored-volume division.

    Paired with the ``+1/2`` face jump of :func:`p05_scalar_face_jump` this is the dissipative upwind orientation.
    """
    jump = jnp.asarray(face_jump)
    lo = jnp.asarray(lower_owner); hi = jnp.asarray(upper_owner)
    out = jnp.zeros((len(owner_volume), jump.shape[-1]), dtype=jump.dtype)
    out = out.at[jnp.maximum(lo, 0)].add(jnp.where((lo >= 0)[:, None], jump, 0))
    out = out.at[jnp.maximum(hi, 0)].add(jnp.where((hi >= 0)[:, None], -jump, 0))
    return out/jnp.asarray(owner_volume)[:, None]


def p06_midpoint_material_remainder(values, gradients, bmag, curvature_vector,
                                    evolution_weight, tau=1.0):
    """Frozen q1 material/remainder numerators for five P06 primitive fields.

    Arrays are ``(raw, q, 5)`` and ``(raw, q, 5, 3)`` with field order
    ``n, Te, Ti, omega, phi``.  ``evolution_weight`` is q1 J/B, separate
    from any stored physical volume used for diagnostic norms.
    """
    v = jnp.asarray(values); g = jnp.asarray(gradients)
    b = jnp.asarray(bmag); k = jnp.asarray(curvature_vector)
    w = jnp.asarray(evolution_weight)
    n, te, ti = v[..., 0], v[..., 1], v[..., 2]
    curvature = jnp.einsum('...d,...fd->...f', k, g)
    matrix = curvature_principal_matrix(n, te, ti, b, tau)
    material = jnp.einsum('...ij,...j->...i', matrix, curvature[..., :4])/jnp.maximum(b[..., None], 1e-30)
    cpsi = curvature[..., 4]+tau*curvature[..., 2]
    coeff = jnp.stack((-2*n/b, -4*te/(3*b), -4*ti/(3*b), jnp.zeros_like(n)), axis=-1)
    remainder = coeff*cpsi[..., None]
    return (jnp.sum(w[..., None]*material, axis=1),
            jnp.sum(w[..., None]*remainder, axis=1))


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
    """The campaign's real-spectrum/condition test and Frobenius fallback."""
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
# "block_lapack": the 3x3 (n, Te, Ti) block with LAPACK ``eig``
# --------------------------------------------------------------------------
# The principal matrix is ``P = [[A, 0], [c^T, 0]]`` (its omega column vanishes), and then exactly
# ``|P| = [[|A|, 0], [c^T sign(A), 0]]``: the right eigenvector of ``P`` for the eigenvalue ``lambda_i`` of ``A``
# is ``(v_i, c.v_i/lambda_i)``, its left eigenvector ``(w_i, 0)``, and the eigenvalue 0 of the omega row drops out of
# ``|.|``.  The same holds for ``M = s P``.  ``sign(0) = 0`` here, as in ``"closed_form"``.

def _absolute_block_primal(matrix, jump):
    block = matrix[..., :3, :3]
    row = matrix[..., 3, :3]
    eigenvalues, eigenvectors = jnp.linalg.eig(block)
    inverse = jnp.linalg.inv(eigenvectors)
    condition = jnp.linalg.cond(eigenvectors)
    real = jnp.real(eigenvalues)
    valid = (jnp.max(jnp.abs(jnp.imag(eigenvalues)), axis=-1)
             <= 1e-10*(1+jnp.max(jnp.abs(real), axis=-1)))
    valid &= jnp.isfinite(condition) & (condition <= 1e8)
    modal = jnp.einsum('...ij,...j->...i', inverse, jump[..., :3])
    absolute = jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors, jnp.abs(real)*modal))
    sign_action = jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors, jnp.sign(real)*modal))
    omega = jnp.sum(row*sign_action, axis=-1)
    spectral = jnp.concatenate((absolute, omega[..., None]), axis=-1)
    fallback = _frobenius_fallback(matrix, jump)
    return (jnp.where(valid[..., None], spectral, fallback), ~valid,
            eigenvalues, eigenvectors, inverse, modal)


@jax.custom_jvp
def _absolute_block_action(matrix, jump):
    """``(|M| jump, invalid)`` through the 3x3 block; ``invalid`` is a 0/1 float flag (no tangent)."""
    action, invalid = _absolute_block_primal(matrix, jump)[:2]
    return action, invalid.astype(action.dtype)


@_absolute_block_action.defjvp
def _absolute_block_action_jvp(primals, tangents):
    """Fréchet derivative of ``V f(Lambda) V^-1`` for ``f = |.|`` and ``f = sign`` away from branch crossings.

    As for the 4x4 action: the matrix action itself is differentiated by divided differences of ``f`` (a repeated
    eigenvalue takes ``f'``: ``sign`` and 0), not the unsupported nonsymmetric eigenvector derivatives.
    """
    matrix, jump = primals
    dmatrix, djump = tangents
    action, invalid, eigenvalues, eigenvectors, inverse, modal = (
        _absolute_block_primal(matrix, jump))
    eigenvalues, eigenvectors, inverse, modal, invalid = (
        jax.lax.stop_gradient(x) for x in (eigenvalues, eigenvectors, inverse, modal, invalid))
    lam = jnp.real(eigenvalues)
    gap = lam[..., :, None]-lam[..., None, :]
    same = jnp.abs(gap) <= 1e-12*(1+jnp.maximum(jnp.abs(lam[..., :, None]), jnp.abs(lam[..., None, :])))
    safe_gap = jnp.where(same, 1, gap)

    def divided(f, fprime):
        return jnp.where(same, fprime[..., :, None], (f[..., :, None]-f[..., None, :])/safe_gap)

    block = matrix[..., :3, :3]; row = matrix[..., 3, :3]
    dblock = dmatrix[..., :3, :3]; drow = dmatrix[..., 3, :3]; du = djump[..., :3]; u = jump[..., :3]
    transformed = jnp.einsum('...ij,...jk,...kl->...il', inverse, dblock, eigenvectors)
    absolute_divided = divided(jnp.abs(lam), jnp.sign(lam))
    sign_divided = divided(jnp.sign(lam), jnp.zeros_like(lam))
    dmodal = jnp.einsum('...ij,...j->...i', inverse, du)
    # |A| u and sign(A) u, and their tangents
    d_abs = (jnp.einsum('...ij,...jk,...kl,...l->...i', eigenvectors, absolute_divided*transformed, inverse, u)
             + jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors, jnp.abs(lam)*dmodal)))
    d_abs = jnp.real(d_abs)
    sign_action = jnp.real(jnp.einsum('...ij,...j->...i', eigenvectors, jnp.sign(lam)*modal))
    d_sign = jnp.real(jnp.einsum('...ij,...jk,...kl,...l->...i', eigenvectors, sign_divided*transformed, inverse, u)
                      + jnp.einsum('...ij,...j->...i', eigenvectors, jnp.sign(lam)*dmodal))
    d_omega = jnp.sum(drow*sign_action, axis=-1) + jnp.sum(row*d_sign, axis=-1)
    spectral_tangent = jnp.concatenate((d_abs, d_omega[..., None]), axis=-1)
    square = jnp.sum(matrix*matrix, axis=(-2, -1))
    norm = jnp.where(square > 0, jnp.sqrt(jnp.where(square > 0, square, 1.)), 0.)
    dnorm = jnp.sum(matrix*dmatrix, axis=(-2, -1))/jnp.maximum(norm, 1e-300)
    fallback_tangent = dnorm[..., None]*jump+norm[..., None]*djump
    tangent = jnp.where(invalid[..., None], fallback_tangent, spectral_tangent)
    flag = invalid.astype(action.dtype)
    return (action, flag), (tangent, jnp.zeros_like(flag))


def _p06_absolute_action_block(matrix, jump):
    """``(|M| jump, invalid)`` by the 3x3 block eigensolve and the usual validity test / Frobenius fallback."""
    action, flag = _absolute_block_action(matrix, jump)
    return action, jax.lax.stop_gradient(flag) > .5


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

def _cubic_residual(mu, r):
    return ((9*mu+18*r-60)*mu+60-160*r)*mu+200*r


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
    mu = mu-_cubic_residual(mu, r)/((27*mu+2*(18*r-60))*mu+60-160*r)
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


def _closed_form_physical(n, te, ti, b, tau, scale, floor):
    return (jnp.isfinite(n) & jnp.isfinite(te) & jnp.isfinite(ti) & jnp.isfinite(b) & jnp.isfinite(tau)
            & jnp.isfinite(scale) & (n > floor) & (te > floor) & (ti >= 0) & (tau >= 0))


def _absolute_action_closed_form(n, te, ti, b, tau, scale, matrix, jump, floor):
    """``(|M| jump, invalid)`` for ``M = scale * P(n, Te, Ti, B, tau)``, no eigensolver.

    ``invalid`` marks the non-physical nodes (``n`` or ``Te`` not above ``floor``, ``Ti < 0``, ``tau < 0``, or a
    non-finite input), which take the Frobenius fallback on ``matrix``.  They are evaluated at a safe state first, so
    neither the values nor the derivatives of the closed form see their data.
    """
    n, te, ti, b, tau, scale = jnp.broadcast_arrays(*(jnp.asarray(x) for x in (n, te, ti, b, tau, scale)))
    valid = _closed_form_physical(n, te, ti, b, tau, scale, floor)
    one = jnp.ones_like(te)
    n, te, ti, b, tau = (jnp.where(valid, x, one) for x in (n, te, ti, b, tau))
    scale = jnp.where(valid, scale, 0.)
    n = jnp.maximum(n, 1e-30)                         # as the principal matrix
    r = tau*ti/te
    mu = _curvature_negative_root(r)
    pair_sum, pair_product = _positive_pair_invariants(r, mu)
    t = ti/te
    # u in the scaled frame D^-1 u, and Abar u, Abar^2 u
    u0 = jump[..., 0]/n; u1 = jump[..., 1]/te; u2 = jump[..., 2]/te
    apply = lambda v0, v1, v2: (2*(v0+v1)+2*tau*v2, (4*v0+14*v1+4*tau*v2)/3, 4*t/3*(v0+v1)-2*tau*t*v2)
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


def _wall_identity_fallback(central, bmag, normal, tau, floor):
    """``(Fc, Qf)`` non-physical flag of the wall nodes (``n, Te, Ti`` not above ``floor``, or a non-finite input).

    The wall solve of P06 passes the same state as ``interior`` and ``boundary_trace`` (P's wall conditions enter through
    the face reconstruction), so its incoming-mode residual is exactly 0 and the exterior / working states are the
    state itself: the solve is the identity wherever it is valid, and only its fallback flag survives.
    """
    ok = (jnp.all(jnp.isfinite(central), axis=-1) & jnp.isfinite(bmag) & jnp.isfinite(normal) & jnp.isfinite(tau)
          & jnp.all(central[..., :3] > floor, axis=-1))
    return ~ok


def p06_characteristic_face_correction(common_state, lower_state, upper_state,
                                       bmag, normal, quadrature_weight, wall_mask,
                                       collapsed_mask, *, tau=1.0, positivity_floor=1e-12,
                                       wall_faces=None, absolute_method="closed_form"):
    """Return distinct lower/upper q3 P06 fluctuation numerators and counters.

    ``normal`` is ``J*K_axis/B**2``.  At physical upper radial walls the
    campaign replaces the exterior with its characteristic wall solve.  The
    returned counters distinguish its thermodynamic floor occurrence from
    the absolute-matrix spectral fallback.

    ``wall_faces`` (optional ``(Wn,)`` int array, ``flatnonzero(wall_mask)`` in
    any order, optionally padded with the out-of-range index ``len(wall_mask)``)
    restricts the ``"lapack4"`` wall characteristic solve to the wall faces: it is gathered
    there and written back where ``wall_mask`` selects it, so the result is
    bitwise that of ``wall_faces=None`` (the solve on every face, masked by
    ``wall_mask``) at a fraction of the cost.  The two must describe the same
    faces; padded entries are ignored.  The other methods do not solve at the wall (below) and ignore it.

    ``absolute_method`` (static, one of :data:`ABSOLUTE_METHODS`) selects how ``|M| jump`` is evaluated:
    ``"lapack4"`` (the campaign's 4x4 ``eig``; pin it to reproduce frozen campaigns bitwise, wall solve and counters
    included), ``"block_lapack"`` or ``"closed_form"`` (see the
    comments above ``_absolute_block_primal`` and ``_curvature_negative_root``).  The spectral-fallback counter
    counts the nodes on the Frobenius fallback: for ``"lapack4"`` / ``"block_lapack"`` those failing the real-spectrum /
    eigenvector-condition test, for ``"closed_form"`` the non-physical ones (``n`` or ``Te`` not above
    ``positivity_floor``, ``Ti < 0``, ``tau < 0``, non-finite).  ``"block_lapack"`` / ``"closed_form"`` replace the wall
    characteristic solve by its exact result, the identity (``lower = upper = central`` at wall faces, jump 0; see
    :func:`_wall_identity_fallback`), and count as wall fallback the non-physical wall nodes (``n``, ``Te`` or ``Ti`` not
    above ``positivity_floor``, non-finite inputs).  They differ from the solve only at finite non-physical nodes, which
    the solve may turn into NaN (complex spectrum, ill-conditioned eigenvectors, e.g. ``Ti -> 0``) and the identity only counts.
    """
    _validated_absolute_method(absolute_method)
    central = jnp.asarray(common_state)
    lower = jnp.asarray(lower_state)
    upper = jnp.asarray(upper_state)
    B = jnp.asarray(bmag); normal = jnp.asarray(normal)
    weight = jnp.asarray(quadrature_weight)
    wall = jnp.asarray(wall_mask, dtype=bool)
    collapsed = jnp.asarray(collapsed_mask, dtype=bool)
    if central.shape[-1] != 4:
        raise ValueError("P06 characteristic state requires four primitive fields")
    # Singular collapsed-axis geometry is bypassed.  These safe values are
    # discarded after the vectorized eigensystem, exactly as in the campaign.
    safe = jnp.array((1., 1., 1., 0.), dtype=central.dtype)
    central = jnp.where(collapsed[:, None, None], safe, central)
    lower = jnp.where(collapsed[:, None, None], safe, lower)
    upper = jnp.where(collapsed[:, None, None], safe, upper)
    if absolute_method != "lapack4":
        # The wall solve is the identity (see _wall_identity_fallback): no eig / inverse / least squares.
        lower = jnp.where(wall[:, None, None], central, lower)
        upper = jnp.where(wall[:, None, None], central, upper)
        wall_fallback_count = jnp.sum(_wall_identity_fallback(central, B, normal, tau, positivity_floor) & wall[:, None])
    elif wall_faces is None:
        exterior, working, wall_fallback = _curvature_bc_characteristic_wall_states(
            central, central, B, tau, normal, interior_on_right=False,
            positivity_floor=positivity_floor)
        lower = jnp.where(wall[:, None, None], central, lower)
        upper = jnp.where(wall[:, None, None], exterior, upper)
        central = jnp.where(wall[:, None, None], working, central)
        wall_fallback_count = jnp.sum(wall_fallback & wall[:, None])
    else:
        idx = jnp.asarray(wall_faces, dtype=jnp.int32)
        # Padded entries (idx == Fc) gather a clipped real face and are dropped by the scatters and the count.
        take = lambda a: jnp.take(a, idx, axis=0, mode="clip")
        central_wall = take(central)
        exterior, working, wall_fallback = _curvature_bc_characteristic_wall_states(
            central_wall, central_wall, take(B), tau, take(normal), interior_on_right=False,
            positivity_floor=positivity_floor)
        lower = jnp.where(wall[:, None, None], central, lower)
        upper = upper.at[idx].set(exterior, mode="drop")
        central = central.at[idx].set(working, mode="drop")
        wall_fallback_count = jnp.sum(wall_fallback & (idx < wall.shape[0])[:, None])
    matrix = -normal[..., None, None]*curvature_principal_matrix(
        central[..., 0], central[..., 1], central[..., 2], B, tau)
    jump = upper-lower
    if absolute_method == "closed_form":
        absolute, spectral_fallback = _absolute_action_closed_form(
            central[..., 0], central[..., 1], central[..., 2], B, tau, -normal, matrix, jump, positivity_floor)
    elif absolute_method == "block_lapack":
        absolute, spectral_fallback = _p06_absolute_action_block(matrix, jump)
    else:
        absolute, spectral_fallback = _p06_absolute_action(matrix, jump)
    material = jnp.einsum('...ij,...j->...i', matrix, jump)
    dplus = jnp.where(collapsed[:, None, None], 0, .5*(material+absolute))
    dminus = jnp.where(collapsed[:, None, None], 0, .5*(material-absolute))
    lower_numerator = -jnp.sum(weight[..., None]*dminus, axis=1)
    upper_numerator = -jnp.sum(weight[..., None]*dplus, axis=1)
    return (lower_numerator, upper_numerator,
            jnp.sum(spectral_fallback & ~collapsed[:, None]),
            jnp.sum((central[..., :3] <= positivity_floor) & ~collapsed[:, None, None]),
            wall_fallback_count)


def scatter_p06_characteristic(lower_numerator, upper_numerator,
                                lower_owner, upper_owner, evolution_volume):
    """Scatter separate P06 side fluctuations over q1 J/B owner mass."""
    lo = jnp.asarray(lower_owner); hi = jnp.asarray(upper_owner)
    lower = jnp.asarray(lower_numerator); upper = jnp.asarray(upper_numerator)
    out = jnp.zeros((len(evolution_volume), lower.shape[-1]), dtype=lower.dtype)
    out = out.at[jnp.maximum(lo, 0)].add(jnp.where((lo >= 0)[:, None], lower, 0))
    out = out.at[jnp.maximum(hi, 0)].add(jnp.where((hi >= 0)[:, None], upper, 0))
    return out/jnp.asarray(evolution_volume)[:, None]
