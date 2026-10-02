"""Explicit raw-row Q07 current/phi assembly; no production pair selection.

The caller owns current support, physical endpoint resolution and the selected
homogeneous maps. This module neither constructs an adjoint nor infers a current
BC from density. Inputs are unprojected actions on the same raw rows.
"""
from typing import NamedTuple


class CurrentPhiAction(NamedTuple):
    divergence_homogeneous: object
    divergence_lift: object
    divergence_physical: object
    vorticity_homogeneous: object
    vorticity_lift: object
    vorticity_current: object
    electron_phi: object
    electron_ti_compensation: object
    electron_generalized_force: object
    inputs_valid: object


def current_phi_from_raw(divergence_homogeneous, divergence_lift, gradient_phi,
                         gradient_ti, density, bmag, *, tau, mu):
    """Assemble one current lift and matched Ti/psi terms, before projection.

    All six raw arrays must have identical shape, including any batch axes.
    ``divergence_lift`` is already an action (physical minus homogeneous), not
    a wall value. It must come from the selected physical-current closure.
    The Ti compensation replaces the material electron Ti column; it must not
    be appended to a material action that already contains that column.
    Invalid density/geometry is flagged without clipping or changing physics.
    """
    import jax.numpy as jnp
    arrays = tuple(jnp.asarray(x) for x in (divergence_homogeneous,
        divergence_lift, gradient_phi, gradient_ti, density, bmag))
    if any(a.dtype.kind != 'f' for a in arrays):
        raise TypeError('real floating raw arrays required')
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('current/phi inputs must share the same raw shape')
    d0, lift, gp, gti, n, B = arrays
    tau, mu = jnp.asarray(tau), jnp.asarray(mu)
    if tau.ndim or mu.ndim:
        raise ValueError('scalar tau and mu required')
    valid = jnp.isfinite(tau) & (tau >= 0) & jnp.isfinite(mu) & (mu > 0)
    for a in arrays:
        valid = valid & jnp.isfinite(a)
    valid = valid & (n > 0) & (B > 0)
    coefficient = B**2/n
    physical = d0 + lift
    phi = mu*gp
    ti = mu*tau*gti
    force = phi+ti
    omega = coefficient*physical
    homogeneous_omega, lift_omega = coefficient*d0, coefficient*lift
    for value in (coefficient, physical, homogeneous_omega, lift_omega, omega, phi, ti, force):
        valid = valid & jnp.isfinite(value)
    return CurrentPhiAction(d0, lift, physical, homogeneous_omega,
        lift_omega, omega, phi, -ti, force, valid)
