"""Bounded Q07 eta-traced characteristic correction prototype.

This returns a correction only. It is not wired to a production RHS, a wall
law, or the centered density API. The DAE-reduced material matrix is reused
from the active production physics; its psi force partner must be retained
when the complete five-field block is assembled.
"""
from typing import NamedTuple
from .fci_parallel_production_flux import parallel_matrix_from_state, parallel_characteristic_split


class CharacteristicCorrection(NamedTuple):
    correction: object
    eigensystem_admissible: object
    thermodynamic_states_positive: object
    inputs_finite: object


def eta_characteristic_correction(stencil, b_eta, eta_step, *, tau, mu,
                                  characteristic_method="eig", psi="phi_plus_tau_ti"):
    """Return the upwind-minus-centered material RHS at raw traced centers.

    ``stencil`` has shape (..., 5 slots, 5 fields), with eta offsets
    (-2,-1,0,+1,+2)*eta_step and fields (n,Te,Ti,Vi,Ve). The current prototype
    uses existing h/32 caps for +/-1 and h/16 caps for +/-2: eta_step=h/64.
    The principal matrix is frozen spectrally at the reconstructed center,
    while multiplication by its live matrix remains differentiable according
    to the existing production splitter's contract.

    Split b^eta*A, not A alone: either sign of field-line parameter orientation
    is valid. Subtract centered differentiation so accepted centered material
    terms and div(b) geometry sources are not counted twice. This correction
    has no additional div(b) or electrostatic force term.

    Inadmissible eigensystems use the existing explicit Rusanov fallback and
    return a false flag. Positivity/finite flags are diagnostics, not clipping.
    Callers must audit these flags; this is not a positivity-preserving update.

    Static ``characteristic_method='polynomial'`` selects the opt-in native
    quartic-root candidate; the accepted generic ``eig`` remains the default.
    ``psi`` (default ``"phi_plus_tau_ti"``; or ``"phi_plus_tau_pi"``) selects the
    split variable of the electron force and of the principal matrix.
    """
    import jax.numpy as jnp
    q = jnp.asarray(stencil, dtype=jnp.float64)
    if q.ndim < 2 or q.shape[-2:] != (5, 5):
        raise ValueError('expected (..., 5 traced slots, 5 material fields)')
    step = jnp.asarray(eta_step, dtype=q.dtype)
    normal = jnp.asarray(b_eta, dtype=q.dtype)
    if jnp.broadcast_shapes(q.shape[:-2], step.shape, normal.shape) != q.shape[:-2]:
        raise ValueError('geometry must broadcast within the stencil batch shape')
    mm, m, center, p, pp = (q[..., i, :] for i in range(5))
    # Differences relative to center avoid cancellation for constant data.
    backward = (4*(center-m) - (center-mm))/(2*step[..., None])
    forward = (4*(p-center) - (pp-center))/(2*step[..., None])
    centered = (p-m)/(2*step[..., None])
    matrix = parallel_matrix_from_state(center, tau, mu, **({} if psi == 'phi_plus_tau_ti' else {'psi': psi}))
    if characteristic_method == "eig":
        plus, minus, _, _, valid = parallel_characteristic_split(matrix, normal=normal)
    elif characteristic_method == "polynomial":
        from .q_characteristic_polynomial import polynomial_characteristic_split
        plus, minus, _, _, valid = polynomial_characteristic_split(center, tau, mu, normal, **({} if psi == 'phi_plus_tau_ti' else {'psi': psi}))
    else:
        raise ValueError('characteristic_method must be eig or polynomial')
    correction = -jnp.einsum('...ij,...j->...i', plus, backward-centered)
    correction -= jnp.einsum('...ij,...j->...i', minus, forward-centered)
    positive = jnp.all(q[..., :3] > 0, axis=(-2, -1))
    finite = jnp.all(jnp.isfinite(q), axis=(-2, -1))
    finite &= jnp.isfinite(normal) & jnp.isfinite(step) & (step > 0)
    finite &= jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    return CharacteristicCorrection(correction, valid, positive, finite)
