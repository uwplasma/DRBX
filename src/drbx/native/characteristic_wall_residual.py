"""Characteristic wall-state reconstruction and boundary laws.

Physical wall laws are expressed in primitive variables, whereas a
hyperbolic face may prescribe only the incoming characteristic subspace.  The
strict nonlinear solver parameterizes the trace as ``q_wall = q_interior +
R_in @ a`` and solves only the incoming amplitudes ``a``.  The legacy solver
below retains its weighted least-residual behavior for compatibility.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def no_flow_boundary_residual(
    state: Any,
    *,
    ion_velocity_index: int = 3,
    electron_velocity_index: int = 4,
) -> jnp.ndarray:
    """Return the two scalar residuals for the validation no-flow wall.

    The indices are part of the boundary-law convention and are intentionally
    static Python integers, so this helper remains usable from a jitted
    incoming-characteristic solve.  The returned residual has shape ``(2,)``
    for one state and ``(..., 2)`` for a batched state.
    """

    state = jnp.asarray(state)
    return jnp.stack(
        (state[..., ion_velocity_index], state[..., electron_velocity_index]),
        axis=-1,
    )


def no_flow_boundary_jacobian(
    state: Any,
    *,
    ion_velocity_index: int = 3,
    electron_velocity_index: int = 4,
) -> jnp.ndarray:
    """Return the analytic state Jacobian of :func:`no_flow_boundary_residual`.

    The production state order is ``(n, Te, Ti, Vi, Ve)``.  Indices remain
    static Python integers, while leading batch dimensions are preserved, so
    this selector is safe inside the generic jitted Newton solve.
    """

    state = jnp.asarray(state)
    if state.ndim < 1 or state.shape[-1] <= max(
        ion_velocity_index, electron_velocity_index
    ):
        raise ValueError("state must have enough trailing components for velocity rows")
    jacobian = jnp.zeros(
        state.shape[:-1] + (2, state.shape[-1]), dtype=state.dtype
    )
    return jacobian.at[..., 0, ion_velocity_index].set(1.0).at[
        ..., 1, electron_velocity_index
    ].set(1.0)


def solve_nonlinear_incoming_characteristic_boundary(
    interior: Any,
    incoming_right_eigenvectors: Any,
    residual_fn: Any,
    *,
    jacobian_fn: Any | None = None,
    initial_coefficients: Any | None = None,
    incoming_projector: Any | None = None,
    max_iterations: int = 12,
    absolute_tolerance: float = 1.0e-11,
    relative_tolerance: float = 1.0e-10,
    singular_tolerance: float = 1.0e-12,
    maximum_condition: float = 1.0e12,
    thermodynamic_components: int = 0,
    positivity_floor: float = 1.0e-12,
    admissibility_fn: Any | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Solve a strict nonlinear law on the incoming characteristic subspace.

    The wall trace is parameterized as

    ``q_wall = q_interior + R_in @ coefficients``

    where ``R_in`` must have exactly two linearly independent columns.  The
    boundary law ``residual_fn(q_wall)`` must return exactly two scalar
    residuals.  Thus this routine solves the two incoming amplitudes and
    leaves every component outside ``range(R_in)`` unchanged by construction.

    ``jacobian_fn``, when supplied, is the analytic Jacobian of the residual
    with respect to the primitive state, with shape ``(2, n_state)``.  If it
    is omitted, JAX forward-mode autodiff supplies that Jacobian.  Both paths
    are vmapped over leading batch dimensions and are compatible with an
    outer ``jax.jit`` (the callable arguments should be static in that outer
    jit, as usual for Python callables).

    This is deliberately a strict Newton solve: there is no least-squares
    projection, limiter, fallback closure, or damped step.  Nonfinite input,
    rank-deficient ``R_in``, singular/ill-conditioned residual Jacobian,
    Newton nonconvergence, and inadmissible thermodynamics all produce
    ``solve_valid=False`` and a NaN wall state.  The diagnostic dictionary
    remains finite where possible and always includes the coefficient vector.
    """

    interior = jnp.asarray(interior, dtype=jnp.float64)
    incoming = jnp.asarray(incoming_right_eigenvectors, dtype=jnp.float64)
    if interior.ndim < 1:
        raise ValueError("interior must have a trailing state dimension")
    state_size = interior.shape[-1]
    if incoming.ndim < 2 or incoming.shape[-2] != state_size:
        raise ValueError(
            "incoming_right_eigenvectors must have trailing shape "
            f"({state_size}, 2), got {incoming.shape}"
        )
    if incoming.shape[-1] != 2:
        raise ValueError(
            "this boundary solver requires exactly two incoming modes; "
            f"got {incoming.shape[-1]}"
        )
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive Python integer")

    # Broadcast the state and incoming basis, then flatten only for the
    # pointwise vmap.  This avoids jacfwd constructing cross-batch derivatives.
    batch_shape = jnp.broadcast_shapes(interior.shape[:-1], incoming.shape[:-2])
    interior = jnp.broadcast_to(interior, batch_shape + (state_size,))
    incoming = jnp.broadcast_to(incoming, batch_shape + (state_size, 2))
    flat_interior = interior.reshape((-1, state_size))
    flat_incoming = incoming.reshape((-1, state_size, 2))

    if initial_coefficients is None:
        coefficients0 = jnp.zeros((flat_interior.shape[0], 2), dtype=jnp.float64)
    else:
        coefficients0 = jnp.asarray(initial_coefficients, dtype=jnp.float64)
        coefficients0 = jnp.broadcast_to(
            coefficients0, batch_shape + (2,)
        ).reshape((-1, 2))

    def point_residual(q: jnp.ndarray) -> jnp.ndarray:
        result = jnp.asarray(residual_fn(q), dtype=jnp.float64)
        if result.ndim != 1 or result.shape[0] != 2:
            raise ValueError(
                "residual_fn must return exactly two scalar residuals for "
                f"one state; got shape {result.shape}"
            )
        return result

    if jacobian_fn is None:
        point_jacobian = jax.jacfwd(point_residual)
    else:
        def point_jacobian(q: jnp.ndarray) -> jnp.ndarray:
            result = jnp.asarray(jacobian_fn(q), dtype=jnp.float64)
            if result.ndim != 2 or result.shape != (2, state_size):
                raise ValueError(
                    "jacobian_fn must return shape "
                    f"(2, {state_size}) for one state; got {result.shape}"
                )
            return result

    # vmap/jacfwd are kept as transformations rather than compiling an
    # inner callable.  This composes cleanly when the complete solver is
    # itself wrapped in an outer jax.jit.
    residual_batch = jax.vmap(point_residual)
    jacobian_batch = jax.vmap(point_jacobian)

    def evaluate(coefficients: jnp.ndarray):
        states = flat_interior + jnp.einsum(
            "bij,bj->bi", flat_incoming, coefficients
        )
        residual = residual_batch(states)
        primitive_jacobian = jacobian_batch(states)
        coefficient_jacobian = jnp.einsum(
            "bki,bij->bkj", primitive_jacobian, flat_incoming
        )
        return states, residual, coefficient_jacobian

    _, initial_residual, _ = evaluate(coefficients0)
    initial_norm = jnp.linalg.norm(initial_residual, axis=-1)
    scale = jnp.maximum(initial_norm, 1.0)
    finite_input = (
        jnp.all(jnp.isfinite(flat_interior), axis=-1)
        & jnp.all(jnp.isfinite(flat_incoming), axis=(-2, -1))
        & jnp.all(jnp.isfinite(coefficients0), axis=-1)
        & jnp.all(jnp.isfinite(initial_residual), axis=-1)
    )

    # Numerical rank of R_in is checked explicitly.  QR/pinv is only used for
    # a diagnostic retained-mode projector; it never changes the solve.
    incoming_singular_values = jnp.linalg.svd(
        flat_incoming, compute_uv=False
    )
    incoming_scale = jnp.maximum(
        jnp.max(incoming_singular_values, axis=-1), 1.0
    )
    incoming_active = incoming_singular_values > (
        jnp.asarray(singular_tolerance, dtype=jnp.float64) * incoming_scale[:, None]
    )
    incoming_rank = jnp.sum(incoming_active, axis=-1)
    basis_valid = (
        jnp.all(jnp.isfinite(incoming_singular_values), axis=-1)
        & (incoming_rank == 2)
    )

    def projected_jacobian_diagnostics(coefficient_jacobian: jnp.ndarray):
        """Return singular values, scaled rank, and 2-norm condition."""

        singular_values = jnp.linalg.svd(
            coefficient_jacobian, compute_uv=False
        )
        jacobian_scale = jnp.maximum(
            jnp.max(singular_values, axis=-1), 1.0
        )
        rank = jnp.sum(
            singular_values
            > jnp.asarray(singular_tolerance, dtype=jnp.float64)
            * jacobian_scale[..., None],
            axis=-1,
        )
        # SVD returns singular values in descending order.  This is the same
        # 2-norm condition number as jnp.linalg.cond, without a second SVD.
        condition = singular_values[..., 0] / singular_values[..., -1]
        return singular_values, rank, condition

    def newton_body(iteration: int, carry):
        coefficients, converged, terminated, iterations = carry
        states, residual, coefficient_jacobian = evaluate(coefficients)
        residual_norm = jnp.linalg.norm(residual, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(states), axis=-1)
            & jnp.all(jnp.isfinite(residual), axis=-1)
            & jnp.all(jnp.isfinite(coefficient_jacobian), axis=(-2, -1))
        )
        tolerance = absolute_tolerance + relative_tolerance * scale
        now_converged = finite & (residual_norm <= tolerance)
        jacobian_singular_values, jacobian_rank, jacobian_condition = (
            projected_jacobian_diagnostics(coefficient_jacobian)
        )
        jacobian_valid = (
            finite
            & jnp.all(jnp.isfinite(jacobian_singular_values), axis=-1)
            & jnp.isfinite(jacobian_condition)
            & (jacobian_condition <= maximum_condition)
            & (jacobian_rank == 2)
        )
        safe_jacobian = jnp.where(
            jacobian_valid[:, None, None],
            coefficient_jacobian,
            jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), coefficient_jacobian.shape),
        )
        step = jnp.linalg.solve(
            safe_jacobian, -residual[..., None]
        )[..., 0]
        step_valid = (
            (~converged)
            & (~terminated)
            & (~now_converged)
            & finite
            & jacobian_valid
            & jnp.all(jnp.isfinite(step), axis=-1)
            & basis_valid
            & finite_input
        )
        updated = coefficients + jnp.where(step_valid[:, None], step, 0.0)
        invalid = ~(finite & basis_valid & finite_input & jacobian_valid)
        return (
            updated,
            converged | now_converged,
            terminated | now_converged | invalid,
            iterations + step_valid.astype(jnp.int32),
        )

    def newton_condition(carry):
        iteration, coefficients, converged, terminated, iterations = carry
        return (iteration < max_iterations) & jnp.any(~terminated)

    _, coefficients, converged, terminated, iterations = jax.lax.while_loop(
        newton_condition,
        lambda carry: (
            carry[0] + 1,
            *newton_body(carry[0], carry[1:]),
        ),
        (
            jnp.asarray(0, dtype=jnp.int32),
            coefficients0,
            jnp.zeros_like(initial_norm, dtype=bool),
            jnp.zeros_like(initial_norm, dtype=bool),
            jnp.zeros_like(initial_norm, dtype=jnp.int32),
        ),
    )
    wall_state, residual, coefficient_jacobian = evaluate(coefficients)
    residual_norm = jnp.linalg.norm(residual, axis=-1)
    tolerance = absolute_tolerance + relative_tolerance * scale
    converged = converged | (
        jnp.all(jnp.isfinite(residual), axis=-1) & (residual_norm <= tolerance)
    )
    jacobian_singular_values, jacobian_rank, jacobian_condition = (
        projected_jacobian_diagnostics(coefficient_jacobian)
    )
    jacobian_valid = (
        jnp.all(jnp.isfinite(jacobian_singular_values), axis=-1)
        & jnp.isfinite(jacobian_condition)
        & (jacobian_condition <= maximum_condition)
        & (jacobian_rank == 2)
    )

    # The orthogonal projector is diagnostic only.  The correction is already
    # exactly formed from R_in, so its complementary (retained-mode) content
    # should be roundoff-level zero.
    if incoming_projector is None:
        q_basis, _ = jnp.linalg.qr(flat_incoming, mode="reduced")
        projector = jnp.einsum("bij,bkj->bik", q_basis, q_basis)
    else:
        projector = jnp.asarray(incoming_projector, dtype=jnp.float64)
        projector = jnp.broadcast_to(
            projector, batch_shape + (state_size, state_size)
        ).reshape((-1, state_size, state_size))
    correction = wall_state - flat_interior
    retained_error = jnp.linalg.norm(
        correction
        - jnp.einsum("bij,bj->bi", projector, correction),
        axis=-1,
    )
    thermo_mask = jnp.arange(state_size, dtype=jnp.int32) < jnp.asarray(
        thermodynamic_components, dtype=jnp.int32
    )
    thermodynamic_admissible = jnp.all(
        (~thermo_mask) | (wall_state > jnp.asarray(positivity_floor)), axis=-1
    )
    if admissibility_fn is None:
        admissible = thermodynamic_admissible
    else:
        admissible = jax.vmap(admissibility_fn)(wall_state).astype(bool)
    finite_final = (
        jnp.all(jnp.isfinite(wall_state), axis=-1)
        & jnp.all(jnp.isfinite(residual), axis=-1)
        & jnp.all(jnp.isfinite(coefficients), axis=-1)
        & jnp.isfinite(retained_error)
    )
    solve_valid = (
        finite_input & finite_final & basis_valid & converged & jacobian_valid & admissible
        & jnp.isfinite(jnp.asarray(absolute_tolerance))
        & jnp.isfinite(jnp.asarray(relative_tolerance))
        & (absolute_tolerance >= 0.0) & (relative_tolerance >= 0.0)
    )
    state = jnp.where(
        solve_valid[:, None], wall_state, jnp.full_like(wall_state, jnp.nan)
    ).reshape(batch_shape + (state_size,))
    info = {
        "incoming_rank": incoming_rank.reshape(batch_shape),
        "residual_norm": residual_norm.reshape(batch_shape),
        "relative_residual": (residual_norm / scale).reshape(batch_shape),
        "jacobian_condition": jacobian_condition.reshape(batch_shape),
        "jacobian_rank": jacobian_rank.reshape(batch_shape),
        "iterations": iterations.reshape(batch_shape),
        "converged": converged.reshape(batch_shape),
        "solve_valid": solve_valid.reshape(batch_shape),
        "thermodynamic_admissible": thermodynamic_admissible.reshape(batch_shape),
        "admissible": admissible.reshape(batch_shape),
        "retained_mode_error": retained_error.reshape(batch_shape),
        "retained_error": retained_error.reshape(batch_shape),
        "coefficients": coefficients.reshape(batch_shape + (2,)),
    }
    return state, info


# Short aliases make the boundary-law abstraction discoverable without
# changing the pre-existing least-residual API below.
solve_nonlinear_incoming_characteristic_state = (
    solve_nonlinear_incoming_characteristic_boundary
)


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
    "no_flow_boundary_residual",
    "no_flow_boundary_jacobian",
    "solve_nonlinear_incoming_characteristic_boundary",
    "solve_nonlinear_incoming_characteristic_state",
    "solve_incoming_characteristic_state",
    "apply_maximally_dissipative_characteristic_wall",
]
