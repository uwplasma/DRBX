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


def p05_scalar_face_jump(common_gradient, lower_value, upper_value, h_covariant_over_b,
                         quadrature_weight, face_axis, pairs):
    """Frozen P05 q3 jump, one oriented scalar integral per face and pair.

    ``common_gradient`` is ``(faces, q, 3, fields)``; side values are
    ``(faces, q, fields)``.  Wall exterior values must be supplied from the
    prescribed Dirichlet trace by the caller.
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
    return -.5*jnp.einsum('fq,fqp->fp', weight,
                          jnp.abs(speed[..., generator])*jump[..., transported])


def scatter_p05_jump(face_jump, lower_owner, upper_owner, owner_volume):
    """P05 owner correction: lower plus, upper minus, stored-volume division."""
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


def p06_characteristic_face_correction(common_state, lower_state, upper_state,
                                       bmag, normal, quadrature_weight, wall_mask,
                                       collapsed_mask, *, tau=1.0, positivity_floor=1e-12,
                                       wall_faces=None):
    """Return distinct lower/upper q3 P06 fluctuation numerators and counters.

    ``normal`` is ``J*K_axis/B**2``.  At physical upper radial walls the
    campaign replaces the exterior with its characteristic wall solve.  The
    returned counters distinguish its thermodynamic floor occurrence from
    the absolute-matrix spectral fallback.

    ``wall_faces`` (optional ``(Wn,)`` int array, ``flatnonzero(wall_mask)`` in
    any order, optionally padded with the out-of-range index ``len(wall_mask)``)
    restricts the wall characteristic solve to the wall faces: it is gathered
    there and written back where ``wall_mask`` selects it, so the result is
    bitwise that of ``wall_faces=None`` (the solve on every face, masked by
    ``wall_mask``) at a fraction of the cost.  The two must describe the same
    faces; padded entries are ignored.
    """
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
    if wall_faces is None:
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
