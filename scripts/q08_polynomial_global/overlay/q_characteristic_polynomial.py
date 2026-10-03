"""Opt-in, device-native five-field characteristic polynomial splitter.

The density-scaled parallel symbol has one eigenvalue Vi and a quartic in
x=lambda-Ve. Quartic roots are bracketed by the derivative's analytic cubic
roots, then refined in fixed batched JAX loops. No eigensolver or host callback
is used. This is an implementation candidate, not a new physical flux model.
"""
import jax
import jax.numpy as jnp

from .fci_parallel_production_flux import parallel_matrix_from_state


def _poly(x, d, c, b, a):
    return (((x-d)*x+c)*x+b)*x+a


def _quartic_roots(te, ti, drift, tau, mu):
    # Scale speeds before root finding; both fast electron and slow ion modes
    # must remain resolved. Density is removed by a similarity transformation.
    speed = jnp.sqrt(jnp.maximum(1., te*mu+tau*ti*mu+te+drift*drift))
    e, r, d = te/speed**2, tau*ti/speed**2, drift/speed
    c = (-132723*e*mu+21300*e-30000*r*mu)/45000
    b = d*(169146*e*mu+30000*r*mu)/45000
    a = (-36423*d*d*e*mu+15123*e*e*mu+46505*e*r*mu)/45000
    # Stationary points: x^3 - 3d/4*x^2 + c/2*x + b/4 = 0.
    aa, bb, cc = -3*d/4, c/2, b/4
    p = bb-aa*aa/3
    q = 2*aa**3/27-aa*bb/3+cc
    radius = jnp.sqrt(jnp.maximum(-p/3, 1e-300))
    argument = -q/(2*radius**3)
    angle = jnp.arccos(jnp.clip(argument, -1., 1.))/3
    critical = jnp.sort(2*radius[..., None]*jnp.cos(
        angle[..., None]-2*jnp.pi*jnp.arange(3)/3)-aa[..., None]/3, axis=-1)
    values = _poly(critical, d[..., None], c[..., None], b[..., None], a[..., None])
    real = (p < 0) & (jnp.abs(argument) < 1)
    real &= (values[..., 0] < 0) & (values[..., 1] > 0) & (values[..., 2] < 0)
    bound = 1+jnp.maximum(jnp.maximum(jnp.abs(d), jnp.abs(c)),
                          jnp.maximum(jnp.abs(b), jnp.abs(a)))
    lo = jnp.concatenate((-bound[..., None], critical), axis=-1)
    hi = jnp.concatenate((critical, bound[..., None]), axis=-1)
    positive_left = jnp.array([True, False, True, False])

    def bisect(_, bracket):
        left, right = bracket
        mid = left+(right-left)/2
        value = _poly(mid, d[..., None], c[..., None], b[..., None], a[..., None])
        same = (value > 0) == positive_left
        return jnp.where(same, mid, left), jnp.where(same, right, mid)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (lo, hi))
    roots = lo+(hi-lo)/2
    residual = jnp.max(jnp.abs(_poly(roots, d[..., None], c[..., None],
                                    b[..., None], a[..., None])), axis=-1)
    real &= jnp.isfinite(residual) & (residual < 1e-12)
    return roots*speed[..., None], real


def polynomial_basis(state, tau, mu, *, max_condition=1e10):
    """Stopped eigenvalues/right/left factors and the candidate validity flag.

    Columns are normalized in physical primitive coordinates, matching the
    Frobenius condition test used by the legacy eigensolver. Nonphysical states
    are evaluated at safe inputs and reported invalid, never sent to a host.
    """
    state = jax.lax.stop_gradient(jnp.asarray(state, dtype=jnp.float64))
    tau, mu = (jax.lax.stop_gradient(jnp.asarray(x, dtype=state.dtype)) for x in (tau, mu))
    n, te, ti, vi, ve, tau, mu = jnp.broadcast_arrays(
        *(state[..., i] for i in range(5)), tau, mu)
    physical = jnp.all(jnp.isfinite(state), axis=-1) & jnp.isfinite(tau) & jnp.isfinite(mu)
    physical &= (n > 1e-30) & (te > 0) & (ti >= 0) & (tau >= 0) & (mu > 0)
    n, te, ti, vi, ve, tau, mu = (
        jnp.where(physical, x, safe) for x, safe in
        zip((n, te, ti, vi, ve, tau, mu), (1., 1., 1., 0., 0., 1., 1836.)))
    d = vi-ve
    x, real = _quartic_roots(te, ti, d, tau, mu)
    # For x != d, row Ti enforces delta(Ti) = (2/3) Ti delta(n)/n.
    # Use the electron row to build polynomial eigenvectors, not division by
    # x*(x-d)+71*Te/150 (which can vanish at a perfectly valid simple root).
    # The Vi component below is strictly positive on the physical domain.
    a = 71/150
    un = 1.71*(x-d[..., None])
    vte = (x*x/mu[..., None]-te[..., None]-(2/3*tau*ti)[..., None])*(x-d[..., None])
    uv = x*x/mu[..., None]+(.71*te+131/60*tau*ti)[..., None]
    waves = jnp.stack((n[..., None]*un, vte, 2/3*ti[..., None]*un, uv, x*un), axis=-2)
    # The exact contact root x=d has an independent algebraic eigenvector.
    un = .71*tau
    ut = -(.71*te+1.71*tau*ti+d*d/mu)
    ue = tau*(tau*ti+d*d/mu)
    uv = d*((2/3)*te*un-ue)/(a*te)
    contact = jnp.stack((n*un, ue, ut, uv, d*un), axis=-1)
    vectors = jnp.concatenate((contact[..., None], waves), axis=-1)
    # Max scaling before the Euclidean norm avoids unnecessary overflow.
    largest = jnp.max(jnp.abs(vectors), axis=-2, keepdims=True)
    vectors = vectors/jnp.maximum(largest, 1e-300)
    norm = jnp.linalg.norm(vectors, axis=-2, keepdims=True)
    vectors = vectors/jnp.maximum(norm, 1e-300)
    values = jnp.concatenate((vi[..., None], ve[..., None]+x), axis=-1)
    eye = jnp.broadcast_to(jnp.eye(5, dtype=state.dtype), vectors.shape)
    basic = physical & real & jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
    safe = jnp.where(basic[..., None, None], vectors, eye)
    inverse = jnp.linalg.inv(safe)
    condition = jnp.linalg.norm(safe, axis=(-2, -1))*jnp.linalg.norm(inverse, axis=(-2, -1))
    matrix = parallel_matrix_from_state(jnp.stack((n, te, ti, vi, ve), axis=-1), tau, mu)
    residual = jnp.linalg.norm(matrix@safe-safe*values[..., None, :], axis=(-2, -1))
    residual /= jnp.maximum(jnp.linalg.norm(matrix, axis=(-2, -1))*jnp.linalg.norm(safe, axis=(-2, -1)), 1e-300)
    valid = basic & jnp.isfinite(condition) & (condition <= max_condition) & (residual < 1e-10)
    return (values, jnp.where(valid[..., None, None], safe, eye),
            jnp.where(valid[..., None, None], inverse, eye), valid)


def polynomial_characteristic_split(state, tau, mu, normal=1., *,
                                    eigenvalue_tolerance=1e-10, max_condition=1e10):
    """Match the legacy split's live-matrix/frozen-projector AD convention."""
    matrix = parallel_matrix_from_state(state, tau, mu)
    values, right, left, valid = polynomial_basis(state, tau, mu, max_condition=max_condition)
    normal = jnp.asarray(normal, dtype=matrix.dtype)
    oriented = normal[..., None]*values
    plus = jnp.einsum('...ik,...k,...kj->...ij', right, oriented > eigenvalue_tolerance, left)
    minus = jnp.einsum('...ik,...k,...kj->...ij', right, oriented < -eigenvalue_tolerance, left)
    eye = jnp.broadcast_to(jnp.eye(5, dtype=matrix.dtype), matrix.shape)
    plus = jnp.where(valid[..., None, None], plus, .5*eye)
    minus = jnp.where(valid[..., None, None], minus, -.5*eye)
    tangent = jnp.abs(normal) <= eigenvalue_tolerance
    plus, minus = (jax.lax.stop_gradient(jnp.where(tangent[..., None, None], 0., x))
                   for x in (plus, minus))
    M = normal[..., None, None]*matrix
    safe = jnp.where(jnp.isfinite(M), M, 0.)
    alpha = jnp.linalg.norm(safe, axis=(-2, -1))
    ap = jnp.where(valid[..., None, None], M@plus, .5*(safe+alpha[..., None, None]*eye))
    am = jnp.where(valid[..., None, None], M@minus, .5*(safe-alpha[..., None, None]*eye))
    return ap, am, plus, minus, valid
