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


def apply_maximally_dissipative_characteristic_wall(
    interior: Any,
    reference: Any,
    oriented_eigenvalues: Any,
    right_eigenvectors: Any,
    left_eigenvectors: Any,
    *,
    incoming_source: Any | None = None,
    thermodynamic_components: int = 0,
    positivity_floor: float = 1.0e-12,
    spectral_valid: Any = True,
    eigenvalue_tolerance: float = 1.0e-10,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply a direct maximally dissipative characteristic wall map.

    The characteristic amplitudes are ``w = L (q - reference)``.  Modes with
    an oriented speed below ``-eigenvalue_tolerance`` are incoming and are
    replaced by ``incoming_source`` (zero by default); outgoing and stationary
    amplitudes are copied exactly.  The wall state is reconstructed directly
    as ``reference + R w_wall``.  The homogeneous default source is the
    dissipative production contract.  A nonzero ``incoming_source`` is
    prescribed physical energy input and is not guaranteed to be dissipative.
    This routine deliberately has no limiting, least-squares solve, or
    fallback closure: invalid algebra is reported by
    ``solve_valid=False`` and a NaN state, while a finite but thermodynamically
    inadmissible state is returned unchanged for downstream diagnostics.

    All array operations are batched over leading dimensions and are JIT
    compatible.  ``L^T L`` is not formed; modal energies use the requested
    unit-weight convention ``|lambda| w**2``.
    """

    interior = jnp.asarray(interior, dtype=jnp.float64)
    reference = jnp.asarray(reference, dtype=jnp.float64)
    if interior.ndim < 1:
        raise ValueError("interior must have a trailing state dimension")
    state_size = interior.shape[-1]
    interior, reference = jnp.broadcast_arrays(interior, reference)
    batch_shape = interior.shape[:-1]
    state_shape = batch_shape + (state_size,)
    matrix_shape = batch_shape + (state_size, state_size)

    eigenvalues = jnp.asarray(oriented_eigenvalues, dtype=jnp.float64)
    try:
        eigenvalues = jnp.broadcast_to(eigenvalues, state_shape)
    except ValueError as exc:
        raise ValueError(
            "oriented_eigenvalues must broadcast to interior's shape "
            f"{state_shape}, got {eigenvalues.shape}"
        ) from exc
    # Explicitly discard only representational complex parts.  The caller's
    # spectral_valid flag remains responsible for rejecting invalid complex
    # spectral data.
    right = jnp.real(jnp.asarray(right_eigenvectors)).astype(jnp.float64)
    left = jnp.real(jnp.asarray(left_eigenvectors)).astype(jnp.float64)
    if (
        right.ndim < 2
        or left.ndim < 2
        or right.shape[-2:] != (state_size, state_size)
        or left.shape[-2:] != (state_size, state_size)
    ):
        raise ValueError(
            "right_eigenvectors and left_eigenvectors must have trailing "
            f"shape {(state_size, state_size)}, got {right.shape} and "
            f"{left.shape}"
        )
    try:
        right = jnp.broadcast_to(right, matrix_shape)
        left = jnp.broadcast_to(left, matrix_shape)
    except ValueError as exc:
        raise ValueError(
            "right_eigenvectors and left_eigenvectors must broadcast to "
            f"{matrix_shape}, got {right.shape} and {left.shape}"
        ) from exc

    if incoming_source is None:
        source = jnp.zeros(state_shape, dtype=jnp.float64)
    else:
        source = jnp.asarray(incoming_source, dtype=jnp.float64)
        try:
            source = jnp.broadcast_to(source, state_shape)
        except ValueError as exc:
            raise ValueError(
                "incoming_source must broadcast to interior's shape "
                f"{state_shape}, got {source.shape}"
            ) from exc

    # Classification is based solely on the oriented speeds.  Explicitly
    # excluding non-finite speeds keeps invalid spectra from being mistaken
    # for stationary modes.
    finite_eigenvalues = jnp.isfinite(eigenvalues)
    tolerance = jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
    incoming = finite_eigenvalues & (eigenvalues < -tolerance)
    outgoing = finite_eigenvalues & (eigenvalues > tolerance)
    stationary = finite_eigenvalues & ~(incoming | outgoing)

    delta = interior - reference
    amplitudes = jnp.einsum("...ij,...j->...i", left, delta)
    wall_amplitudes = jnp.where(incoming, source, amplitudes)
    candidate = reference + jnp.einsum(
        "...ij,...j->...i", right, wall_amplitudes
    )

    identity = jnp.eye(state_size, dtype=jnp.float64)
    left_right = jnp.einsum("...ij,...jk->...ik", left, right)
    right_left = jnp.einsum("...ij,...jk->...ik", right, left)
    # A fixed algebraic tolerance avoids making LR validation depend on the
    # wave-speed threshold.  It is intentionally tight enough to catch a
    # supplied, non-dual characteristic basis without rejecting roundoff.
    lr_error = jnp.maximum(
        jnp.max(jnp.abs(left_right - identity), axis=(-2, -1)),
        jnp.max(jnp.abs(right_left - identity), axis=(-2, -1)),
    )
    lr_consistent = jnp.isfinite(lr_error) & (lr_error <= 1.0e-8)

    wall_delta = candidate - reference
    reconstructed_delta = jnp.einsum("...ij,...j->...i", right, amplitudes)
    wall_modal = jnp.einsum("...ij,...j->...i", left, wall_delta)
    outgoing_retained_error = jnp.linalg.norm(
        jnp.where(outgoing, wall_modal - amplitudes, 0.0), axis=-1
    )
    modal_reconstruction_error = jnp.linalg.norm(
        reconstructed_delta - delta, axis=-1
    )

    finite_input = (
        jnp.all(jnp.isfinite(interior), axis=-1)
        & jnp.all(jnp.isfinite(reference), axis=-1)
        & jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        & jnp.all(jnp.isfinite(right), axis=(-2, -1))
        & jnp.all(jnp.isfinite(left), axis=(-2, -1))
        & jnp.all(jnp.isfinite(source), axis=-1)
    )
    finite_solve = (
        jnp.all(jnp.isfinite(amplitudes), axis=-1)
        & jnp.all(jnp.isfinite(wall_amplitudes), axis=-1)
        & jnp.all(jnp.isfinite(candidate), axis=-1)
        & jnp.isfinite(outgoing_retained_error)
        & jnp.isfinite(modal_reconstruction_error)
    )
    spectral_valid = jnp.broadcast_to(
        jnp.asarray(spectral_valid, dtype=bool), batch_shape
    )
    valid_tolerance = jnp.isfinite(tolerance) & (tolerance >= 0.0)
    solve_valid = (
        finite_input
        & finite_solve
        & lr_consistent
        & spectral_valid
        & valid_tolerance
    )
    state = jnp.where(
        solve_valid[..., None], candidate, jnp.full_like(candidate, jnp.nan)
    )

    thermo_mask = jnp.arange(state_size, dtype=jnp.int32) < jnp.asarray(
        thermodynamic_components, dtype=jnp.int32
    )
    thermodynamic_admissible = jnp.all(
        (~thermo_mask)
        | (candidate > jnp.asarray(positivity_floor, dtype=jnp.float64)),
        axis=-1,
    )

    abs_speed = jnp.abs(eigenvalues)

    def modal_energy(mask: jnp.ndarray, modal: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.where(mask, abs_speed * modal**2, 0.0), axis=-1)

    incoming_energy_before = modal_energy(incoming, amplitudes)
    incoming_energy_after = modal_energy(incoming, wall_amplitudes)
    outgoing_energy_before = modal_energy(outgoing, amplitudes)
    outgoing_energy_after = modal_energy(outgoing, wall_amplitudes)
    stationary_energy_before = modal_energy(stationary, amplitudes)
    stationary_energy_after = modal_energy(stationary, wall_amplitudes)
    boundary_power_before = jnp.sum(
        eigenvalues * amplitudes**2, axis=-1
    )
    boundary_power_after = jnp.sum(
        eigenvalues * wall_amplitudes**2, axis=-1
    )
    # Positive boundary power is outward under the supplied normal; domain
    # energy rate is therefore the negative of outward boundary power.
    domain_energy_rate_before = -boundary_power_before
    domain_energy_rate_after = -boundary_power_after

    info = {
        "incoming_rank": jnp.sum(incoming, axis=-1),
        "outgoing_rank": jnp.sum(outgoing, axis=-1),
        "stationary_rank": jnp.sum(stationary, axis=-1),
        "incoming_energy_before": incoming_energy_before,
        "incoming_energy_after": incoming_energy_after,
        "outgoing_energy_before": outgoing_energy_before,
        "outgoing_energy_after": outgoing_energy_after,
        "stationary_energy_before": stationary_energy_before,
        "stationary_energy_after": stationary_energy_after,
        "boundary_power_before": boundary_power_before,
        "boundary_power_after": boundary_power_after,
        "domain_energy_rate_before": domain_energy_rate_before,
        "domain_energy_rate_after": domain_energy_rate_after,
        "outgoing_retained_error": outgoing_retained_error,
        "modal_reconstruction_error": modal_reconstruction_error,
        "left_right_consistency_error": lr_error,
        "solve_valid": solve_valid,
        "thermodynamic_admissible": thermodynamic_admissible,
        "fallback": ~solve_valid,
    }
    return state, info


__all__ = [
    "solve_incoming_characteristic_state",
    "apply_maximally_dissipative_characteristic_wall",
]
