"""Production five-field parallel characteristic flux.

This module is intentionally independent of the model RHS.  It contains the
local material principal symbol and a mapped-leg-friendly Osher fluctuation
that can be used for both ordinary FCI legs and legs whose exterior endpoint
is a wall.  The state order throughout is ``(n, Te, Ti, Vi, Ve)``.

The polarization variable ``psi = phi + tau*Ti`` has already been eliminated
from this material block.  Consequently the electron-velocity row contains
the leading ``mu*tau`` coefficient multiplying ``Ti``.  Vorticity, current
exchange, and the nonlocal polarization solve deliberately do not belong to
this five-field Riemann problem.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


STATE_SIZE = 5
_LOG_FLOOR = 1.0e-30
_DEFAULT_EIG_TOL = 1.0e-10
_DEFAULT_MAX_CONDITION = 1.0e10
_GAUSS_NODES_4 = jnp.asarray(
    (0.06943184420297371, 0.33000947820757187,
     0.6699905217924281, 0.9305681557970262), dtype=jnp.float64
)
_GAUSS_WEIGHTS_4 = jnp.asarray(
    (0.17392742256872692, 0.32607257743127307,
     0.32607257743127307, 0.17392742256872692), dtype=jnp.float64
)


def _as_state(value: Any) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape[-1:] != (STATE_SIZE,):
        raise ValueError(f"parallel state must end in ({STATE_SIZE},), got {value.shape}")
    return value


def _matvec(matrix: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", matrix, value)


def parallel_production_principal_matrix(
    density: Any,
    Te: Any,
    Ti: Any,
    Vi: Any,
    Ve: Any,
    tau: Any,
    mu: Any,
) -> jnp.ndarray:
    """Return the corrected DAE-reduced five-field parallel matrix.

    The eliminated-potential convention is ``psi=phi+tau*Ti``.  In
    particular ``A[4, 2] = mu*tau`` is intentional and must not be replaced by
    zero.  Inputs may be arbitrarily batched broadcastable arrays.
    """

    density, Te, Ti, Vi, Ve, tau, mu = [
        jnp.asarray(x, dtype=jnp.float64) for x in
        (density, Te, Ti, Vi, Ve, tau, mu)
    ]
    density, Te, Ti, Vi, Ve, tau, mu = jnp.broadcast_arrays(
        density, Te, Ti, Vi, Ve, tau, mu
    )
    n_safe = jnp.maximum(density, _LOG_FLOOR)
    dV = Vi - Ve
    matrix = jnp.zeros(density.shape + (STATE_SIZE, STATE_SIZE), dtype=jnp.float64)
    matrix = matrix.at[..., 0, 0].set(Ve)
    matrix = matrix.at[..., 0, 4].set(density)
    matrix = matrix.at[..., 1, 0].set(-1.42 * Te * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 1, 1].set(Ve)
    matrix = matrix.at[..., 1, 3].set(-1.42 * Te / 3.0)
    matrix = matrix.at[..., 1, 4].set(3.42 * Te / 3.0)
    matrix = matrix.at[..., 2, 0].set(-2.0 * Ti * dV / (3.0 * n_safe))
    matrix = matrix.at[..., 2, 2].set(Vi)
    matrix = matrix.at[..., 2, 4].set(2.0 * Ti / 3.0)
    matrix = matrix.at[..., 3, 0].set((Te + tau * Ti) / n_safe)
    matrix = matrix.at[..., 3, 1].set(1.0)
    matrix = matrix.at[..., 3, 2].set(tau)
    matrix = matrix.at[..., 3, 3].set(Vi)
    matrix = matrix.at[..., 4, 0].set(mu * Te / n_safe)
    matrix = matrix.at[..., 4, 1].set(1.71 * mu)
    matrix = matrix.at[..., 4, 2].set(mu * tau)
    matrix = matrix.at[..., 4, 4].set(Ve)
    return matrix


# Short names make this module convenient to use from flux assemblers while
# retaining the explicit production name for audit and call-site clarity.
parallel_characteristic_matrix = parallel_production_principal_matrix
parallel_principal_matrix = parallel_production_principal_matrix


def parallel_matrix_from_state(
    state: jnp.ndarray, tau: Any, mu: Any
) -> jnp.ndarray:
    """Build the principal matrix from a trailing ``(n,Te,Ti,Vi,Ve)`` state."""

    state = _as_state(state)
    return parallel_production_principal_matrix(
        state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], tau, mu
    )


def _spectral_basis(
    matrix: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one stopped-gradient live eigensystem and validity metadata.

    Nonsymmetric eigenvectors are never differentiated.  Invalid/complex or
    ill-conditioned decompositions are replaced by a finite Rusanov split.  A
    static-size eigensystem and array ``where`` operations keep this safe under
    ``jax.jit`` and arbitrary leading batch dimensions.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    frozen = jax.lax.stop_gradient(matrix)
    values, vectors = jnp.linalg.eig(frozen)
    values_real = jnp.real(values)
    finite = jnp.all(jnp.isfinite(values_real), axis=-1)
    finite = finite & jnp.all(jnp.isfinite(jnp.imag(values)), axis=-1)
    imag_ok = jnp.max(jnp.abs(jnp.imag(values)), axis=-1) <= 1.0e-8
    values_ok = finite & imag_ok
    # The basis is square and tiny.  A direct inverse avoids both the SVD in
    # ``pinv`` and the second SVD formerly used for conditioning.  Singular or
    # ill-conditioned bases are rejected below and use the Rusanov fallback.
    safe_vectors = jnp.where(values_ok[..., None, None], vectors, eye)
    inverse = jnp.linalg.inv(safe_vectors)
    condition = (
        jnp.linalg.norm(safe_vectors, axis=(-2, -1))
        * jnp.linalg.norm(inverse, axis=(-2, -1))
    )
    well_conditioned = jnp.isfinite(condition) & (condition <= max_condition)
    valid = values_ok & well_conditioned
    safe_values = jnp.where(jnp.isfinite(values_real), values_real, 0.0)
    safe_vectors = jnp.nan_to_num(safe_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    inverse = jnp.nan_to_num(inverse, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = jnp.max(jnp.abs(safe_values), axis=-1)
    alpha = jnp.where(valid, alpha, jnp.linalg.norm(matrix, axis=(-2, -1)))
    alpha = jnp.maximum(jnp.where(jnp.isfinite(alpha), alpha, 0.0), _LOG_FLOOR)
    return safe_values, safe_vectors, inverse, valid, alpha


def _projectors_from_basis(
    values: jnp.ndarray,
    vectors: jnp.ndarray,
    inverse: jnp.ndarray,
    valid: jnp.ndarray,
    normal: Any,
    *,
    eigenvalue_tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build oriented projectors without repeating an eigendecomposition."""

    normal = jnp.asarray(normal, dtype=jnp.float64)
    oriented_values = normal[..., None] * values
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), vectors.shape)
    positive = oriented_values > eigenvalue_tolerance
    negative = oriented_values < -eigenvalue_tolerance
    p_plus_char = jnp.einsum(
        "...ik,...k,...kj->...ij", vectors, positive, inverse
    )
    p_minus_char = jnp.einsum(
        "...ik,...k,...kj->...ij", vectors, negative, inverse
    )
    p_plus_char = jnp.real(p_plus_char)
    p_minus_char = jnp.real(p_minus_char)
    p_plus = jnp.where(valid[..., None, None], p_plus_char, 0.5 * eye)
    p_minus = jnp.where(valid[..., None, None], p_minus_char, -0.5 * eye)
    tangent = jnp.abs(normal) <= eigenvalue_tolerance
    return (
        jnp.where(tangent[..., None, None], 0.0, p_plus),
        jnp.where(tangent[..., None, None], 0.0, p_minus),
    )


def _spectral_data(
    matrix: jnp.ndarray,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return stopped-gradient eigenvalues, projectors, and validity flags."""

    values, vectors, inverse, valid, alpha = _spectral_basis(
        matrix,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    plus, minus = _projectors_from_basis(
        values, vectors, inverse, valid, 1.0,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    return values, plus, minus, valid, alpha


def _characteristic_split_actions(
    matrix: jnp.ndarray,
    tangent: jnp.ndarray,
    normal: Any,
    *,
    eigenvalue_tolerance: float,
    max_condition: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply the two live characteristic splits without dense projectors."""

    normal = jnp.asarray(normal, dtype=jnp.float64)
    normal_matrix = normal[..., None, None] * matrix
    values, vectors, inverse, valid, alpha = _spectral_basis(
        normal_matrix,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    coefficients = _matvec(inverse, tangent)
    positive = jnp.where(values > eigenvalue_tolerance, values, 0.0)
    negative = jnp.where(values < -eigenvalue_tolerance, values, 0.0)
    plus_characteristic = jnp.real(_matvec(vectors, positive * coefficients))
    minus_characteristic = jnp.real(_matvec(vectors, negative * coefficients))
    safe_matrix = jnp.where(jnp.isfinite(normal_matrix), normal_matrix, 0.0)
    centered = _matvec(safe_matrix, tangent)
    plus_fallback = 0.5 * (centered + alpha[..., None] * tangent)
    minus_fallback = 0.5 * (centered - alpha[..., None] * tangent)
    tangent_normal = jnp.abs(normal) <= eigenvalue_tolerance
    plus = jnp.where(valid[..., None], plus_characteristic, plus_fallback)
    minus = jnp.where(valid[..., None], minus_characteristic, minus_fallback)
    plus = jnp.where(tangent_normal[..., None], 0.0, plus)
    minus = jnp.where(tangent_normal[..., None], 0.0, minus)
    return plus, minus, valid


def parallel_characteristic_projectors(
    matrix: jnp.ndarray,
    normal: Any = 1.0,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return positive/negative projectors for ``normal * matrix``.

    The third result is a boolean admissibility mask.  On fallback points the
    projectors are the Rusanov split ``(+I/2, -I/2)``; therefore all returned
    arrays remain finite and JIT-safe.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    normal = jnp.asarray(normal, dtype=jnp.float64)
    values, vectors, inverse, valid, _alpha = _spectral_basis(
        matrix,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    plus, minus = _projectors_from_basis(
        values, vectors, inverse, valid, normal,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    return plus, minus, valid


def parallel_characteristic_decomposition(
    matrix: jnp.ndarray,
    normal: Any = 1.0,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return stopped-gradient ``(eigenvalues, right, left, admissible)``.

    The right/left factors are diagnostic outputs; production flux assembly
    should use :func:`parallel_characteristic_split` so invalid points receive
    the dissipative Rusanov split.
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    normal = jnp.asarray(normal, dtype=jnp.float64)
    normal_matrix = normal[..., None, None] * matrix
    values, right, left, valid, _alpha = _spectral_basis(
        normal_matrix,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    safe_right = jnp.where(valid[..., None, None], right, eye)
    safe_left = jnp.where(valid[..., None, None], left, eye)
    return jax.lax.stop_gradient(values), jax.lax.stop_gradient(jnp.real(safe_right)), jax.lax.stop_gradient(jnp.real(safe_left)), valid


def parallel_characteristic_split(
    matrix: jnp.ndarray,
    normal: Any = 1.0,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(A_plus, A_minus, P_plus, P_minus, admissible)``.

    The matrix is live in the multiplication, while the stopped-gradient
    projectors are frozen from the local state.  Reversing ``normal`` swaps
    the two directional pieces (up to the expected sign of the live matrix).
    """

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    p_plus, p_minus, valid = parallel_characteristic_projectors(
        matrix, normal, eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    normal = jnp.asarray(normal, dtype=jnp.float64)
    normal_matrix = normal[..., None, None] * matrix
    safe_normal_matrix = jnp.where(jnp.isfinite(normal_matrix), normal_matrix, 0.0)
    alpha = jnp.linalg.norm(safe_normal_matrix, axis=(-2, -1))
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    # The projectors returned for an inadmissible eigensystem are diagnostic
    # placeholders.  The actual split must still be dissipative, so use the
    # standard Rusanov pieces on those points.
    a_plus = jnp.where(
        valid[..., None, None],
        jnp.einsum("...ij,...jk->...ik", normal_matrix, p_plus),
        0.5 * (safe_normal_matrix + alpha[..., None, None] * eye),
    )
    a_minus = jnp.where(
        valid[..., None, None],
        jnp.einsum("...ij,...jk->...ik", normal_matrix, p_minus),
        0.5 * (safe_normal_matrix - alpha[..., None, None] * eye),
    )
    return (
        a_plus,
        a_minus,
        p_plus,
        p_minus,
        valid,
    )


def parallel_characteristic_absolute_action(
    matrix: jnp.ndarray,
    jump: jnp.ndarray,
    normal: Any = 1.0,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> jnp.ndarray:
    """Apply the frozen characteristic ``|normal*A|`` to a batched jump."""

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    jump = _as_state(jump)
    plus, minus, valid = parallel_characteristic_projectors(
        matrix, normal, eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    normal_matrix = jnp.asarray(normal, dtype=jnp.float64)[..., None, None] * matrix
    safe_normal_matrix = jnp.where(jnp.isfinite(normal_matrix), normal_matrix, 0.0)
    alpha = jnp.linalg.norm(safe_normal_matrix, axis=(-2, -1))
    # For admissible points use P+ - P-; fallback projectors produce alpha*I.
    result = _matvec(normal_matrix, _matvec(plus - minus, jump))
    fallback = alpha[..., None] * jump
    return jnp.where(valid[..., None], result, fallback)


def parallel_characteristic_absolute_matrix(
    matrix: jnp.ndarray,
    normal: Any = 1.0,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(|normal*A|, admissible)`` with a finite Rusanov fallback."""

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    plus, minus, valid = parallel_characteristic_projectors(
        matrix, normal, eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    normal_matrix = jnp.asarray(normal, dtype=jnp.float64)[..., None, None] * matrix
    safe_normal_matrix = jnp.where(jnp.isfinite(normal_matrix), normal_matrix, 0.0)
    absolute = jnp.einsum("...ij,...jk->...ik", normal_matrix, plus - minus)
    alpha = jnp.linalg.norm(safe_normal_matrix, axis=(-2, -1))
    absolute = jnp.where(valid[..., None, None], absolute, alpha[..., None, None] * jnp.eye(STATE_SIZE))
    return absolute, valid


def parallel_wall_exterior_state(
    owner: jnp.ndarray,
    candidate: jnp.ndarray,
    matrix: jnp.ndarray,
    normal: Any,
    *,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> jnp.ndarray:
    """Project a candidate wall state onto outgoing owner/incoming candidate modes."""

    owner, candidate = _as_state(owner), _as_state(candidate)
    plus, minus, _valid = parallel_characteristic_projectors(
        matrix, normal, eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    normal = jnp.asarray(normal, dtype=jnp.float64)
    incoming = jnp.where(normal[..., None, None] >= 0.0, minus, plus)
    return owner + _matvec(incoming, candidate - owner)


parallel_characteristic_wall_state = parallel_wall_exterior_state


def _positive_path(
    left: jnp.ndarray,
    right: jnp.ndarray,
    s: jnp.ndarray,
    *,
    positivity_floor: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Log-linear positive thermodynamic path and its tangent."""

    default = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    left_finite = jnp.all(jnp.isfinite(left), axis=-1)
    right_finite = jnp.all(jnp.isfinite(right), axis=-1)
    left = jnp.where(left_finite[..., None], left, default)
    right = jnp.where(right_finite[..., None], right, default)
    left_positive = jnp.maximum(left[..., :3], positivity_floor)
    right_positive = jnp.maximum(right[..., :3], positivity_floor)
    log_left = jnp.log(left_positive)
    log_right = jnp.log(right_positive)
    log_value = (1.0 - s) * log_left + s * log_right
    positive = jnp.exp(log_value)
    tangent_positive = positive * (log_right - log_left)
    linear = (1.0 - s) * left[..., 3:] + s * right[..., 3:]
    tangent_linear = right[..., 3:] - left[..., 3:]
    clipped = (~left_finite) | (~right_finite) | (left_positive != left[..., :3]).any(axis=-1) | (right_positive != right[..., :3]).any(axis=-1)
    return (
        jnp.concatenate((positive, linear), axis=-1),
        jnp.concatenate((tangent_positive, tangent_linear), axis=-1),
        clipped,
    )


def third_order_face_reconstruction(
    stencil: jnp.ndarray,
    *,
    positivity_floor: float = 1.0e-12,
    return_fallback: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct left/right face states from ``(...,4,5)`` cell values.

    The unlimited third-order stencil is used when its density and both
    temperatures are admissible.  Each side independently falls back to its
    adjacent first-order owner state, which is suitable for irregular mapped
    FCI legs where a four-cell stencil may cross a transition.
    """

    stencil = _as_state(stencil)
    if stencil.shape[-2] != 4:
        raise ValueError(f"stencil must have shape (...,4,5), got {stencil.shape}")
    qm, q0, q1, qp = [stencil[..., i, :] for i in range(4)]
    left = -qm / 6.0 + 5.0 * q0 / 6.0 + q1 / 3.0
    right = q0 / 3.0 + 5.0 * q1 / 6.0 - qp / 6.0
    left_ok = jnp.all(jnp.isfinite(left), axis=-1) & jnp.all(left[..., :3] > positivity_floor, axis=-1)
    right_ok = jnp.all(jnp.isfinite(right), axis=-1) & jnp.all(right[..., :3] > positivity_floor, axis=-1)
    q0_safe = q0.at[..., :3].set(jnp.maximum(q0[..., :3], positivity_floor))
    q1_safe = q1.at[..., :3].set(jnp.maximum(q1[..., :3], positivity_floor))
    q0_safe = jnp.where(jnp.isfinite(q0_safe), q0_safe, jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0)))
    q1_safe = jnp.where(jnp.isfinite(q1_safe), q1_safe, jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0)))
    left = jnp.where(left_ok[..., None], left, q0_safe)
    right = jnp.where(right_ok[..., None], right, q1_safe)
    fallback = jnp.stack((~left_ok, ~right_ok), axis=-1)
    if return_fallback:
        return jnp.stack((left, right), axis=-2), fallback
    return jnp.stack((left, right), axis=-2)


def parallel_path_fluctuations(
    left: jnp.ndarray,
    right: jnp.ndarray | None = None,
    tau: Any = 1.0,
    mu: Any = 1836.0,
    *,
    normal: Any = 1.0,
    exterior_state: jnp.ndarray | None = None,
    quadrature_order: int = 4,
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
    return_diagnostics: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute fixed-Gauss path-conservative Osher fluctuations.

    ``right`` is the ordinary mapped-leg endpoint.  For a wall leg callers may
    pass the wall exterior as either ``right`` or ``exterior_state``; the two
    forms are intentionally equivalent.  The positive log-linear path keeps
    ``n, Te, Ti`` admissible, while velocities use a linear path.  The
    characteristic split is canonical (one-half left/right fluctuation) and
    has no tunable penalty coefficient.
    """

    left = _as_state(left)
    if exterior_state is not None:
        if right is not None:
            raise ValueError("pass either right or exterior_state, not both")
        right = exterior_state
    if right is None:
        raise ValueError("right or exterior_state is required")
    right = _as_state(right)
    left, right = jnp.broadcast_arrays(left, right)
    if quadrature_order != 4:
        raise ValueError("production path uses fixed four-point Gauss quadrature")
    nodes, weights = _GAUSS_NODES_4, _GAUSS_WEIGHTS_4
    normal = jnp.asarray(normal, dtype=jnp.float64)
    path_fallback = jnp.any((left[..., :3] <= positivity_floor) | (right[..., :3] <= positivity_floor), axis=-1)
    path_fallback = path_fallback | ~jnp.all(jnp.isfinite(left), axis=-1) | ~jnp.all(jnp.isfinite(right), axis=-1)
    dplus = jnp.zeros_like(left)
    dminus = jnp.zeros_like(left)
    valid_all = jnp.ones(left.shape[:-1], dtype=bool)
    for node, weight in zip(nodes, weights):
        state, tangent, _clipped = _positive_path(left, right, node, positivity_floor=positivity_floor)
        matrix = parallel_production_principal_matrix(
            state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], tau, mu
        )
        plus_action, minus_action, valid = _characteristic_split_actions(
            matrix,
            tangent,
            normal,
            eigenvalue_tolerance=eigenvalue_tolerance,
            max_condition=max_condition,
        )
        dplus = dplus + weight * plus_action
        dminus = dminus + weight * minus_action
        valid_all = valid_all & valid
    # A nonpositive endpoint is replaced by the positive floor path; expose the
    # event so production diagnostics can count it without making the kernel
    # non-JIT-safe.  Spectral fallback is selected pointwise in the split.
    fallback = path_fallback | ~valid_all
    diagnostics = {
        "fallback": fallback,
        "spectral_fallback": ~valid_all,
        "positivity_clipped": path_fallback,
        "admissible": ~fallback,
    }
    if return_diagnostics:
        return dplus, dminus, diagnostics
    return dplus, dminus


parallel_osher_fluctuations = parallel_path_fluctuations
parallel_path_conservative_fluctuations = parallel_path_fluctuations


def parallel_target_row_material_residual(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: Any,
    dx_plus: Any,
    tau: Any,
    mu: Any,
    *,
    backward_wall: Any = False,
    forward_wall: Any = False,
    backward_wall_state: jnp.ndarray | None = None,
    forward_wall_state: jnp.ndarray | None = None,
    equilibrium: jnp.ndarray | None = None,
    div_b: Any = 0.0,
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply the first-order production material update to every mapped row.

    The backward fluctuation is evaluated on the oriented path
    ``minus -> center`` and the forward fluctuation on ``center -> plus``:

    ``residual = -(D_plus_backward / dx_minus + D_minus_forward / dx_plus)``.

    A wall endpoint replaces the mapped endpoint by the same characteristic
    exterior projection used by :func:`parallel_wall_exterior_state`; ordinary
    rows simply use their supplied mapped endpoints.  This makes wall and bulk
    legs one operator with different endpoint data, rather than two numerical
    fluxes.  ``div_b`` supplies the exact geometric source omitted from the
    frozen principal matrix.
    """

    center, minus, plus = _as_state(center), _as_state(minus), _as_state(plus)
    center, minus, plus = jnp.broadcast_arrays(center, minus, plus)
    dx_minus = jnp.asarray(dx_minus, dtype=jnp.float64)
    dx_plus = jnp.asarray(dx_plus, dtype=jnp.float64)
    div_b = jnp.asarray(div_b, dtype=jnp.float64)
    backward_wall = jnp.asarray(backward_wall, dtype=bool)
    forward_wall = jnp.asarray(forward_wall, dtype=bool)
    backward_wall, forward_wall, dx_minus, dx_plus, div_b = jnp.broadcast_arrays(
        backward_wall, forward_wall, dx_minus, dx_plus, div_b
    )

    if equilibrium is None:
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = _as_state(equilibrium)
    equilibrium = jnp.broadcast_to(equilibrium, center.shape)
    backward_candidate = equilibrium if backward_wall_state is None else _as_state(backward_wall_state)
    forward_candidate = equilibrium if forward_wall_state is None else _as_state(forward_wall_state)
    backward_candidate = jnp.broadcast_to(backward_candidate, center.shape)
    forward_candidate = jnp.broadcast_to(forward_candidate, center.shape)

    matrix = parallel_matrix_from_state(center, tau, mu)
    center_values, center_vectors, center_inverse, center_valid, _center_alpha = (
        _spectral_basis(
            matrix,
            eigenvalue_tolerance=eigenvalue_tolerance,
            max_condition=max_condition,
        )
    )
    backward_plus, _backward_minus = _projectors_from_basis(
        center_values,
        center_vectors,
        center_inverse,
        center_valid,
        -1.0,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    _forward_plus, forward_minus = _projectors_from_basis(
        center_values,
        center_vectors,
        center_inverse,
        center_valid,
        1.0,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    # For the backward normal the incoming projector is P+(-A); for the
    # forward normal it is P-(A).  Both are derived from the one center-state
    # live eigensystem above rather than decomposing the same matrix four times.
    wall_minus = center + _matvec(
        backward_plus, backward_candidate - center
    )
    wall_plus = center + _matvec(
        forward_minus, forward_candidate - center
    )
    backward_valid = center_valid
    forward_valid = center_valid
    minus_used = jnp.where(backward_wall[..., None], wall_minus, minus)
    plus_used = jnp.where(forward_wall[..., None], wall_plus, plus)

    # D_plus on the backward-oriented path and D_minus on the forward path
    # are the two wave-propagation contributions at this target row.
    dplus_backward, _dminus_backward, backward_info = parallel_path_fluctuations(
        minus_used, center, tau, mu, normal=1.0,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition, return_diagnostics=True,
    )
    _dplus_forward, dminus_forward, forward_info = parallel_path_fluctuations(
        center, plus_used, tau, mu, normal=1.0,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition, return_diagnostics=True,
    )
    residual = -(
        dplus_backward / jnp.maximum(jnp.abs(dx_minus), _LOG_FLOOR)[..., None]
        + dminus_forward / jnp.maximum(jnp.abs(dx_plus), _LOG_FLOOR)[..., None]
    )

    density, Te, Ti, Vi, Ve = [center[..., i] for i in range(STATE_SIZE)]
    current = density * (Vi - Ve)
    geometric = jnp.zeros_like(center)
    geometric = geometric.at[..., 0].set(-density * Ve * div_b)
    geometric = geometric.at[..., 1].set(
        (2.0 * Te / (3.0 * jnp.maximum(density, _LOG_FLOOR)))
        * (0.71 * current - density * Ve) * div_b
    )
    geometric = geometric.at[..., 2].set(
        (2.0 * Ti / (3.0 * jnp.maximum(density, _LOG_FLOOR)))
        * (current - density * Vi) * div_b
    )
    residual = residual + geometric

    diagnostics = {
        "backward_wall": backward_wall,
        "forward_wall": forward_wall,
        "wall_row": backward_wall | forward_wall,
        "ordinary_row": ~(backward_wall | forward_wall),
        "spectral_fallback": backward_info["spectral_fallback"] | forward_info["spectral_fallback"],
        "positivity_fallback": backward_info["positivity_clipped"] | forward_info["positivity_clipped"],
        "wall_spectral_fallback": (~backward_valid & backward_wall) | (~forward_valid & forward_wall),
        "fallback": backward_info["fallback"] | forward_info["fallback"] | ((~backward_valid & backward_wall) | (~forward_valid & forward_wall)),
        "admissible": backward_info["admissible"] & forward_info["admissible"] & backward_valid & forward_valid,
    }
    return residual, diagnostics


__all__ = [
    "STATE_SIZE",
    "parallel_production_principal_matrix",
    "parallel_characteristic_matrix",
    "parallel_principal_matrix",
    "parallel_matrix_from_state",
    "parallel_characteristic_decomposition",
    "parallel_characteristic_projectors",
    "parallel_characteristic_split",
    "parallel_characteristic_absolute_action",
    "parallel_characteristic_absolute_matrix",
    "parallel_wall_exterior_state",
    "parallel_characteristic_wall_state",
    "third_order_face_reconstruction",
    "parallel_path_fluctuations",
    "parallel_osher_fluctuations",
    "parallel_path_conservative_fluctuations",
    "parallel_target_row_material_residual",
]
