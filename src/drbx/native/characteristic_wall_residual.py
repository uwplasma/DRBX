"""Residual-based characteristic wall-state reconstruction.

The physical boundary bundle is expressed in primitive variables, whereas a
hyperbolic face may prescribe only the incoming characteristic subspace.  This
module computes the least-residual incoming correction without selecting a
particular set of primitive rows by hand.  Outgoing/stationary content is
retained exactly because every correction lies in ``range(P_in)``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def solve_incoming_characteristic_state(
    interior: Any,
    target: Any,
    incoming_projector: Any,
    *,
    incoming_basis: Any | None = None,
    incoming_active: Any | None = None,
    residual_weights: Any | None = None,
    thermodynamic_components: int = 0,
    positivity_floor: float = 1.0e-12,
    spectral_valid: Any = True,
    singular_tolerance: float = 1.0e-10,
    regularization: float = 1.0e-13,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Solve the primitive boundary residual over the incoming subspace.

    ``target`` is the primitive trace produced by the physical boundary
    operator.  ``residual_weights`` selects/scales the physical equations
    belonging to this hyperbolic subsystem.  The solve minimizes

    ``||W (U_interior + delta - U_target)||_2``

    subject to ``delta in range(P_in)``.  An SVD supplies an orthonormal basis
    for that range when the caller does not already own one.  The result is
    deliberately *not* limited or replaced: inadmissible thermodynamic values
    remain visible to the time-integrator validity checks, and an invalid
    solve returns NaNs rather than silently selecting another wall closure.
    """

    interior = jnp.asarray(interior, dtype=jnp.float64)
    target = jnp.asarray(target, dtype=jnp.float64)
    projector = jax.lax.stop_gradient(
        jnp.asarray(incoming_projector, dtype=jnp.float64)
    )
    interior, target = jnp.broadcast_arrays(interior, target)
    if interior.ndim < 1:
        raise ValueError("interior must have a trailing state dimension")
    state_size = interior.shape[-1]
    expected = interior.shape[:-1] + (state_size, state_size)
    if projector.shape != expected:
        raise ValueError(
            f"incoming_projector must have shape {expected}, got {projector.shape}"
        )
    thermodynamic_components = jnp.asarray(
        thermodynamic_components, dtype=jnp.int32
    )
    if residual_weights is None:
        weights = jnp.ones_like(interior)
    else:
        weights = jnp.asarray(residual_weights, dtype=jnp.float64)
        weights = jnp.broadcast_to(weights, interior.shape)
    weights = jnp.where(jnp.isfinite(weights), jnp.maximum(weights, 0.0), 0.0)

    if incoming_basis is None:
        # Standalone path: left singular vectors corresponding to nonzero
        # singular values span range(P_in).  Production callers already own
        # the characteristic basis and pass it below, avoiding another SVD.
        basis, singular_values, _ = jnp.linalg.svd(
            projector, full_matrices=True
        )
        scale = jnp.maximum(
            jnp.max(singular_values, axis=-1, keepdims=True), 1.0
        )
        active = singular_values > (
            jnp.asarray(singular_tolerance, dtype=jnp.float64) * scale
        )
    else:
        basis = jnp.asarray(jnp.real(incoming_basis), dtype=jnp.float64)
        if basis.shape != expected:
            raise ValueError(
                f"incoming_basis must have shape {expected}, got {basis.shape}"
            )
        if incoming_active is None:
            raise ValueError(
                "incoming_active is required when incoming_basis is supplied"
            )
        active = jnp.asarray(incoming_active, dtype=bool)
        active = jnp.broadcast_to(active, interior.shape)
    basis = jax.lax.stop_gradient(jnp.real(basis))
    active_f = active.astype(jnp.float64)
    basis = basis * active_f[..., None, :]

    weighted_basis = weights[..., :, None] * basis
    gram = jnp.einsum("...ki,...kj->...ij", weighted_basis, weighted_basis)
    inactive = 1.0 - active_f
    gram = gram + jnp.einsum("...i,ij->...ij", inactive, jnp.eye(state_size))
    gram = gram + (
        jnp.asarray(regularization, dtype=jnp.float64)
        * jnp.einsum("...i,ij->...ij", active_f, jnp.eye(state_size))
    )
    difference = target - interior
    rhs = jnp.einsum(
        "...ki,...k->...i", weighted_basis, weights * difference
    )
    coefficients = jnp.linalg.solve(gram, rhs[..., None])[..., 0] * active_f
    correction = jnp.einsum("...ij,...j->...i", basis, coefficients)

    finite_input = (
        jnp.all(jnp.isfinite(interior), axis=-1)
        & jnp.all(jnp.isfinite(target), axis=-1)
        & jnp.all(jnp.isfinite(projector), axis=(-2, -1))
    )
    finite_solve = (
        jnp.all(jnp.isfinite(coefficients), axis=-1)
        & jnp.all(jnp.isfinite(correction), axis=-1)
    )
    spectral_valid = jnp.broadcast_to(
        jnp.asarray(spectral_valid, dtype=bool), interior.shape[:-1]
    )
    solve_valid = finite_input & finite_solve & spectral_valid
    unconstrained_state = interior + correction
    state = jnp.where(
        solve_valid[..., None],
        unconstrained_state,
        jnp.full_like(unconstrained_state, jnp.nan),
    )
    thermodynamic_mask = (
        jnp.arange(state_size, dtype=jnp.int32) < thermodynamic_components
    )
    thermodynamic_admissible = jnp.all(
        (~thermodynamic_mask)
        | (
            unconstrained_state
            > jnp.asarray(positivity_floor, dtype=jnp.float64)
        ),
        axis=-1,
    )

    residual = weights * (state - target)
    target_scale = jnp.maximum(
        jnp.linalg.norm(weights * difference, axis=-1), 1.0e-30
    )
    residual_norm = jnp.linalg.norm(residual, axis=-1)
    correction_norm = jnp.linalg.norm(correction, axis=-1)
    difference_norm = jnp.maximum(jnp.linalg.norm(difference, axis=-1), 1.0e-30)
    retained_error = jnp.linalg.norm(
        jnp.einsum(
            "...ij,...j->...i",
            jnp.eye(state_size, dtype=jnp.float64) - projector,
            state - interior,
        ),
        axis=-1,
    )
    info = {
        "incoming_rank": jnp.sum(active, axis=-1),
        "solve_valid": solve_valid,
        "thermodynamic_admissible": thermodynamic_admissible,
        # Compatibility key for existing diagnostic consumers.  This solver
        # never limits the result.
        "positivity_limited": jnp.zeros_like(solve_valid),
        "fallback": ~solve_valid,
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / target_scale,
        "correction_amplification": correction_norm / difference_norm,
        "retained_error": retained_error,
        "step_scale": jnp.ones_like(residual_norm),
        "coefficients": coefficients,
    }
    return state, info


__all__ = ["solve_incoming_characteristic_state"]
