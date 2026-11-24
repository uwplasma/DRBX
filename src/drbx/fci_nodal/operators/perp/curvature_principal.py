"""The local principal matrix shared by SBP curvature operators."""

from __future__ import annotations


import jax
import jax.numpy as jnp


Array = jax.Array
STATE_SIZE = 4


#: Split-variable conventions (the names of ``FciDrbEBParameters.polarization_variable``):
#: ``psi = phi + tau*Ti`` (legacy, the default of every function below) or
#: ``psi = phi + tau*n*Ti`` (``p_i = n*Ti``).
PSI_VARIABLES = ("phi_plus_tau_ti", "phi_plus_tau_pi")


def _validated_psi(psi: str) -> str:
    if psi not in PSI_VARIABLES:
        raise ValueError(f"psi must be one of {PSI_VARIABLES}, got {psi!r}")
    return psi


def curvature_principal_matrix(
    density: Array | float,
    Te: Array | float,
    Ti: Array | float,
    bmag: Array | float,
    tau: Array | float,
    *,
    k_perp_squared: Array | float | None = None,
    psi: str = "phi_plus_tau_ti",
) -> Array:
    """Return the corrected DAE-reduced curvature RHS Jacobian.

    The rows and columns are ordered ``(n, Te, Ti, omega)``.  This is the
    strict local matrix used to form the curvature RHS; in particular, the
    ``Ti`` column includes the polarization response
    ``(2 n tau, 4 tau Te/3, -2 tau Ti, 2 tau B^2)``.  If
    ``k_perp_squared`` is supplied, the optional non-local omega column is
    included as a diagnostic using the local Fourier relation
    ``delta phi = -tau delta Ti - delta omega/k_perp_squared``.

    ``psi`` selects the split variable whose curvature remainder is carried
    outside the material block (the names of
    ``FciDrbEBParameters.polarization_variable``).  The default
    ``"phi_plus_tau_ti"`` (``psi = phi + tau*Ti``) is the matrix described
    above.  ``"phi_plus_tau_pi"`` (``psi = phi + tau*n*Ti``) has
    ``d psi/d n = tau Ti`` and ``d psi/d Ti = tau n`` instead of ``0`` and
    ``tau``: on the rows ``(n, Te, Ti)`` with ``c = (2n, 4Te/3, 4Ti/3)`` the
    ``n`` column gains ``c tau Ti`` and the ``Ti`` column gains
    ``c tau (n - 1)``.  The omega row is unchanged.  The remainder in the
    RHS must use the same ``psi``; the two cancel in the continuum.
    """

    _validated_psi(psi)
    n, te, ti, b, tau_value = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (density, Te, Ti, bmag, tau)
    )
    shape = jnp.broadcast_shapes(n.shape, te.shape, ti.shape, b.shape, tau_value.shape)
    n = jnp.broadcast_to(n, shape)
    te = jnp.broadcast_to(te, shape)
    ti = jnp.broadcast_to(ti, shape)
    b = jnp.broadcast_to(b, shape)
    tau_value = jnp.broadcast_to(tau_value, shape)
    n_safe = jnp.maximum(n, 1.0e-30)
    matrix = jnp.zeros(shape + (STATE_SIZE, STATE_SIZE), dtype=jnp.float64)
    matrix = matrix.at[..., 0, 0].set(2.0 * te)
    matrix = matrix.at[..., 0, 1].set(2.0 * n)
    matrix = matrix.at[..., 0, 2].set(2.0 * n * tau_value)
    matrix = matrix.at[..., 1, 0].set(4.0 * te * te / (3.0 * n_safe))
    matrix = matrix.at[..., 1, 1].set(14.0 * te / 3.0)
    matrix = matrix.at[..., 1, 2].set(4.0 * tau_value * te / 3.0)
    matrix = matrix.at[..., 2, 0].set(4.0 * ti * te / (3.0 * n_safe))
    matrix = matrix.at[..., 2, 1].set(4.0 * ti / 3.0)
    matrix = matrix.at[..., 2, 2].set(-2.0 * tau_value * ti)
    matrix = matrix.at[..., 3, 0].set(2.0 * b * b * (te + tau_value * ti) / n_safe)
    matrix = matrix.at[..., 3, 1].set(2.0 * b * b)
    matrix = matrix.at[..., 3, 2].set(2.0 * tau_value * b * b)
    if k_perp_squared is not None:
        k2 = jnp.asarray(k_perp_squared, dtype=jnp.float64)
        k2 = jnp.broadcast_to(k2, shape)
        k2_safe = jnp.maximum(k2, 1.0e-30)
        matrix = matrix.at[..., 0, 3].set(2.0 * n / k2_safe)
        matrix = matrix.at[..., 1, 3].set(4.0 * te / (3.0 * k2_safe))
        matrix = matrix.at[..., 2, 3].set(4.0 * ti / (3.0 * k2_safe))
    if psi == "phi_plus_tau_pi":
        c = jnp.stack((2.0 * n, 4.0 * te / 3.0, 4.0 * ti / 3.0), axis=-1)
        matrix = matrix.at[..., 0:3, 0].add(c * (tau_value * ti)[..., None])
        matrix = matrix.at[..., 0:3, 2].add(c * (tau_value * (n - 1.0))[..., None])
    return matrix
