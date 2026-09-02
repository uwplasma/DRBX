"""Production five-field parallel characteristic flux.

This module is intentionally independent of the model RHS.  It contains the
local material principal symbol and the canonical-face characteristic update
used for both ordinary FCI legs and legs whose exterior endpoint is a wall.
The state order throughout is ``(n, Te, Ti, Vi, Ve)``.

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

from .characteristic_wall_residual import (
    apply_maximally_dissipative_characteristic_wall,
    no_flow_boundary_jacobian,
    no_flow_boundary_residual,
    solve_nonlinear_incoming_characteristic_boundary,
    solve_incoming_characteristic_state,
)


STATE_SIZE = 5
_LOG_FLOOR = 1.0e-30
_DEFAULT_EIG_TOL = 1.0e-10
_DEFAULT_MAX_CONDITION = 1.0e10


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


def _incoming_right_eigenvectors(
    values: jnp.ndarray,
    vectors: jnp.ndarray,
    *,
    orientation: str,
    eigenvalue_tolerance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Select exactly two incoming columns from the full frozen basis.

    The production five-field material block has two incoming modes at the
    subsonic wall states used by the current closure.  Boolean indexing is
    not JIT-safe for batched arrays, so stable integer ``argsort`` gathers the
    active columns while retaining a diagnostic count for rank validation.
    ``orientation='backward'`` means the wall is the minus endpoint and uses
    positive speeds; the forward endpoint uses negative speeds.
    """

    if orientation not in ("backward", "forward"):
        raise ValueError(f"unknown wall orientation {orientation!r}")
    incoming_active = (
        values > jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
        if orientation == "backward"
        else values < -jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
    )
    incoming_count = jnp.sum(incoming_active, axis=-1)
    # Active modes sort first; the original basis index breaks ties so the
    # gather is deterministic and works under jit/vmap for every batch shape.
    basis_index = jnp.arange(STATE_SIZE, dtype=jnp.int32)
    sort_key = jnp.where(incoming_active, 0, 1) * STATE_SIZE + basis_index
    selected_index = jnp.argsort(sort_key, axis=-1)[..., :2]
    selected = jnp.take_along_axis(
        jnp.real(vectors), selected_index[..., None, :], axis=-1
    )
    return selected, incoming_count


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
    _plus, minus, valid = parallel_characteristic_projectors(
        matrix, normal, eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    # The projectors are already oriented by the outward normal, so incoming
    # modes are always the negative branch.  Selecting by the sign of the
    # normal a second time reverses the backward-wall classification.
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    incoming = jnp.where(valid[..., None, None], minus, 0.5 * eye)
    return owner + _matvec(incoming, candidate - owner)


parallel_characteristic_wall_state = parallel_wall_exterior_state


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


def _canonical_leg_face_state(
    left: jnp.ndarray,
    right: jnp.ndarray,
    *,
    positivity_floor: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the mapped-leg face state and a positivity/finiteness flag.

    Thermodynamic entries use a geometric mean so the face state remains
    positive without introducing an arithmetic overshoot.  The velocity
    entries use the centered arithmetic mean.  This is deliberately a
    single face state; no quadrature samples or repeated state-dependent
    eigendecompositions are taken.
    """

    left = _as_state(left)
    right = _as_state(right)
    left, right = jnp.broadcast_arrays(left, right)
    finite = jnp.all(jnp.isfinite(left), axis=-1) & jnp.all(
        jnp.isfinite(right), axis=-1
    )
    left_thermo = jnp.maximum(left[..., :3], positivity_floor)
    right_thermo = jnp.maximum(right[..., :3], positivity_floor)
    clipped = (
        ~finite
        | jnp.any(left[..., :3] <= positivity_floor, axis=-1)
        | jnp.any(right[..., :3] <= positivity_floor, axis=-1)
    )
    face = jnp.concatenate(
        (
            jnp.sqrt(left_thermo * right_thermo),
            0.5 * (left[..., 3:] + right[..., 3:]),
        ),
        axis=-1,
    )
    default = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    face = jnp.where(jnp.all(jnp.isfinite(face), axis=-1)[..., None], face, default)
    return face, clipped


def parallel_canonical_leg_face_state(
    left: jnp.ndarray,
    right: jnp.ndarray,
    *,
    positivity_floor: float = 1.0e-12,
    return_fallback: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the single live characteristic state for a mapped leg.

    ``n, Te, Ti`` are geometric means and ``Vi, Ve`` arithmetic means.  If
    either endpoint is nonpositive/nonfinite the thermodynamic entries are
    clipped to ``positivity_floor`` and, optionally, the fallback flag is
    returned.
    """

    face, fallback = _canonical_leg_face_state(
        left, right, positivity_floor=positivity_floor
    )
    return (face, fallback) if return_fallback else face


def _live_characteristic_leg_action(
    face_state: jnp.ndarray,
    jump: jnp.ndarray,
    tau: Any,
    mu: Any,
    normal: Any,
    *,
    branch: str,
    eigenvalue_tolerance: float,
    max_condition: float,
    basis: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply one live-state characteristic branch to one mapped leg.

    ``basis`` is optionally supplied when a wall projection and its
    one-sided fluctuation share the same interior/owner matrix.  Ordinary
    legs pass no basis and therefore perform exactly one eigendecomposition
    at their canonical face state.
    """

    face_state = _as_state(face_state)
    jump = _as_state(jump)
    matrix = parallel_matrix_from_state(face_state, tau, mu)
    if basis is None:
        values, vectors, inverse, valid, alpha = _spectral_basis(
            matrix,
            eigenvalue_tolerance=eigenvalue_tolerance,
            max_condition=max_condition,
        )
    else:
        values, vectors, inverse, valid, alpha = basis
    normal = jnp.asarray(normal, dtype=jnp.float64)
    oriented_values = normal[..., None] * values
    coefficients = _matvec(inverse, jump)
    selected = (
        jnp.where(oriented_values > eigenvalue_tolerance, oriented_values, 0.0)
        if branch == "plus"
        else jnp.where(oriented_values < -eigenvalue_tolerance, oriented_values, 0.0)
    )
    action = jnp.real(_matvec(vectors, selected * coefficients))
    normal_matrix = normal[..., None, None] * matrix
    safe_normal_matrix = jnp.where(jnp.isfinite(normal_matrix), normal_matrix, 0.0)
    fallback_matrix = 0.5 * (
        safe_normal_matrix
        + (1.0 if branch == "plus" else -1.0)
        * alpha[..., None, None]
        * jnp.eye(STATE_SIZE, dtype=jnp.float64)
    )
    fallback_action = _matvec(fallback_matrix, jump)
    tangent = jnp.abs(normal) <= eigenvalue_tolerance
    action = jnp.where(valid[..., None], action, fallback_action)
    action = jnp.where(tangent[..., None], 0.0, action)
    return action, valid, values


def _prepare_material_direction_inputs(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: Any,
    dx_plus: Any,
    *,
    backward_wall: Any,
    forward_wall: Any,
    backward_wall_state: jnp.ndarray | None,
    forward_wall_state: jnp.ndarray | None,
    equilibrium: jnp.ndarray | None,
) -> tuple[jnp.ndarray, ...]:
    """Broadcast the data shared by the explicit and short-wall paths."""

    center, minus, plus = _as_state(center), _as_state(minus), _as_state(plus)
    center, minus, plus = jnp.broadcast_arrays(center, minus, plus)
    dx_minus = jnp.asarray(dx_minus, dtype=jnp.float64)
    dx_plus = jnp.asarray(dx_plus, dtype=jnp.float64)
    backward_wall = jnp.asarray(backward_wall, dtype=bool)
    forward_wall = jnp.asarray(forward_wall, dtype=bool)
    backward_wall, forward_wall, dx_minus, dx_plus = jnp.broadcast_arrays(
        backward_wall, forward_wall, dx_minus, dx_plus
    )
    if equilibrium is None:
        # Primitive material variables are normalized at the production
        # equilibrium; this is the documented reference for the energy wall
        # when callers omit an explicit equilibrium.
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = _as_state(equilibrium)
    equilibrium = jnp.broadcast_to(equilibrium, center.shape)
    backward_candidate = equilibrium if backward_wall_state is None else _as_state(backward_wall_state)
    forward_candidate = equilibrium if forward_wall_state is None else _as_state(forward_wall_state)
    backward_candidate = jnp.broadcast_to(backward_candidate, center.shape)
    forward_candidate = jnp.broadcast_to(forward_candidate, center.shape)
    backward_candidate_fallback = ~jnp.all(jnp.isfinite(backward_candidate), axis=-1)
    forward_candidate_fallback = ~jnp.all(jnp.isfinite(forward_candidate), axis=-1)
    # Do not silently substitute equilibrium for a failed physical trace.
    # Non-finite candidates propagate through the wall solve and are caught by
    # the stage validity checks; the flags remain available for diagnosis.
    return (
        center, minus, plus, dx_minus, dx_plus, backward_wall, forward_wall,
        backward_candidate, forward_candidate, backward_candidate_fallback,
        forward_candidate_fallback, equilibrium,
    )


def _material_directional_data(
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
    parallel_characteristic_wall_law: str = "primitive-least-residual",
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, ...]:
    """Return both directional material actions and frozen local Jacobians.

    This is the common face-state/wall-projection path used by the explicit
    residual and the selected short-wall implicit diagnostic.  The returned
    Jacobians are derivatives with respect to the owner ``center`` while the
    neighboring and wall states, as well as the stopped-gradient eigensystem,
    are frozen:

    ``J_backward = -A_plus / dx_minus`` and
    ``J_forward = +A_minus / dx_plus``.

    The geometric ``div_b`` source is intentionally absent here.
    """

    if parallel_characteristic_wall_law not in (
        "primitive-least-residual", "energy-absorbing", "velocity-no-flow"
    ):
        raise ValueError(
            "parallel_characteristic_wall_law must be "
            "'primitive-least-residual', 'energy-absorbing', or "
            "'velocity-no-flow', got "
            f"{parallel_characteristic_wall_law!r}"
        )

    (
        center, minus, plus, dx_minus, dx_plus, backward_wall, forward_wall,
        backward_candidate, forward_candidate, backward_candidate_fallback,
        forward_candidate_fallback, equilibrium,
    ) = _prepare_material_direction_inputs(
        center, minus, plus, dx_minus, dx_plus,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=backward_wall_state,
        forward_wall_state=forward_wall_state, equilibrium=equilibrium,
    )

    backward_face, backward_clipped = _canonical_leg_face_state(
        minus, center, positivity_floor=positivity_floor
    )
    forward_face, forward_clipped = _canonical_leg_face_state(
        center, plus, positivity_floor=positivity_floor
    )
    backward_face_used = jnp.where(backward_wall[..., None], center, backward_face)
    forward_face_used = jnp.where(forward_wall[..., None], center, forward_face)
    backward_basis = _spectral_basis(
        parallel_matrix_from_state(backward_face_used, tau, mu),
        eigenvalue_tolerance=eigenvalue_tolerance, max_condition=max_condition,
    )
    forward_basis = _spectral_basis(
        parallel_matrix_from_state(forward_face_used, tau, mu),
        eigenvalue_tolerance=eigenvalue_tolerance, max_condition=max_condition,
    )
    backward_values, backward_vectors, backward_inverse, backward_valid, backward_alpha = backward_basis
    forward_values, forward_vectors, forward_inverse, forward_valid, forward_alpha = forward_basis
    backward_wall_plus, _ = _projectors_from_basis(
        backward_values, backward_vectors, backward_inverse, backward_valid, 1.0,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    _, forward_wall_minus = _projectors_from_basis(
        forward_values, forward_vectors, forward_inverse, forward_valid, 1.0,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), backward_basis[1].shape)
    backward_wall_plus = jnp.where(
        backward_valid[..., None, None], backward_wall_plus, 0.5 * eye
    )
    forward_wall_minus = jnp.where(
        forward_valid[..., None, None], forward_wall_minus, 0.5 * eye
    )
    backward_incoming_count = jnp.sum(
        backward_values > jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
        axis=-1,
    )
    forward_incoming_count = jnp.sum(
        forward_values < -jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
        axis=-1,
    )
    # Keep the live/interior sign counts separate from any closure-specific
    # classification counts.  In particular, the no-flow wall state makes
    # the contact eigenvalue stationary even when the interior Vi is small
    # and nonzero.
    backward_interior_incoming_count = backward_incoming_count
    forward_interior_incoming_count = forward_incoming_count
    backward_classification_incoming_count = backward_incoming_count
    forward_classification_incoming_count = forward_incoming_count
    backward_classification_valid = backward_valid
    forward_classification_valid = forward_valid
    backward_classification_stationary_count = jnp.sum(
        jnp.abs(backward_values)
        <= jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
        axis=-1,
    )
    forward_classification_stationary_count = jnp.sum(
        jnp.abs(forward_values)
        <= jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
        axis=-1,
    )
    if parallel_characteristic_wall_law == "primitive-least-residual":
        wall_minus, backward_wall_solve = solve_incoming_characteristic_state(
            center,
            backward_candidate,
            backward_wall_plus,
            incoming_basis=backward_vectors,
            incoming_active=(
                backward_values
                > jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
            ),
            thermodynamic_components=3,
            positivity_floor=positivity_floor,
            spectral_valid=backward_valid,
        )
        wall_plus, forward_wall_solve = solve_incoming_characteristic_state(
            center,
            forward_candidate,
            forward_wall_minus,
            incoming_basis=forward_vectors,
            incoming_active=(
                forward_values
                < -jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
            ),
            thermodynamic_components=3,
            positivity_floor=positivity_floor,
            spectral_valid=forward_valid,
        )
    elif parallel_characteristic_wall_law == "energy-absorbing":
        # The direct map closes incoming modes against the explicit equilibrium
        # reference.  Outward speeds are -backward_values and +forward_values;
        # scalar ghost candidates are intentionally not read in this branch.
        wall_minus, backward_wall_solve = apply_maximally_dissipative_characteristic_wall(
            center,
            equilibrium,
            -backward_values,
            backward_vectors,
            backward_inverse,
            thermodynamic_components=3,
            positivity_floor=positivity_floor,
            spectral_valid=backward_valid,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        wall_plus, forward_wall_solve = apply_maximally_dissipative_characteristic_wall(
            center,
            equilibrium,
            forward_values,
            forward_vectors,
            forward_inverse,
            thermodynamic_components=3,
            positivity_floor=positivity_floor,
            spectral_valid=forward_valid,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
    else:
        # Validation-only no-flow law.  Classify the material modes at the
        # constrained wall trace, where Vi=Ve=0 and lambda_contact=0.  The
        # live/interior eigensystem above is deliberately retained for the
        # flux split; it is not the correct incoming-count state for this
        # closure when the interior Vi is small and nonzero.
        classification_state = center.at[..., 3].set(0.0).at[..., 4].set(0.0)
        backward_classification = _spectral_basis(
            parallel_matrix_from_state(classification_state, tau, mu),
            eigenvalue_tolerance=eigenvalue_tolerance,
            max_condition=max_condition,
        )
        forward_classification = backward_classification
        (
            backward_classification_values,
            backward_classification_vectors,
            backward_classification_inverse,
            backward_classification_valid,
            _,
        ) = backward_classification
        (
            forward_classification_values,
            forward_classification_vectors,
            forward_classification_inverse,
            forward_classification_valid,
            _,
        ) = forward_classification
        backward_classification_plus, _ = _projectors_from_basis(
            backward_classification_values,
            backward_classification_vectors,
            backward_classification_inverse,
            backward_classification_valid,
            1.0,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        _, forward_classification_minus = _projectors_from_basis(
            forward_classification_values,
            forward_classification_vectors,
            forward_classification_inverse,
            forward_classification_valid,
            1.0,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        backward_incoming, backward_classification_incoming_count = _incoming_right_eigenvectors(
            backward_classification_values,
            backward_classification_vectors,
            orientation="backward",
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        forward_incoming, forward_classification_incoming_count = _incoming_right_eigenvectors(
            forward_classification_values,
            forward_classification_vectors,
            orientation="forward",
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        backward_classification_stationary_count = jnp.sum(
            jnp.abs(backward_classification_values)
            <= jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
            axis=-1,
        )
        forward_classification_stationary_count = jnp.sum(
            jnp.abs(forward_classification_values)
            <= jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64),
            axis=-1,
        )
        backward_incoming_count = backward_classification_incoming_count
        forward_incoming_count = forward_classification_incoming_count

        wall_minus, backward_wall_solve = (
            solve_nonlinear_incoming_characteristic_boundary(
                center,
                backward_incoming,
                no_flow_boundary_residual,
                jacobian_fn=no_flow_boundary_jacobian,
                incoming_projector=backward_classification_plus,
                max_iterations=1,
                thermodynamic_components=3,
                positivity_floor=positivity_floor,
            )
        )
        wall_plus, forward_wall_solve = (
            solve_nonlinear_incoming_characteristic_boundary(
                center,
                forward_incoming,
                no_flow_boundary_residual,
                jacobian_fn=no_flow_boundary_jacobian,
                incoming_projector=forward_classification_minus,
                max_iterations=1,
                thermodynamic_components=3,
                positivity_floor=positivity_floor,
            )
        )
        # The generic solve validates linear independence of the packed
        # two-column R_in.  At active wall rows the physical sign mask must
        # also contain exactly two incoming modes; otherwise this fixed-size
        # two-constraint law is under/over-specified and must fail loudly.
        # Ordinary rows are allowed to have another characteristic count
        # because their mapped endpoint never consumes this wall solve.
        backward_count_valid = (~backward_wall) | (
            backward_classification_valid
            & (backward_classification_incoming_count == 2)
        )
        forward_count_valid = (~forward_wall) | (
            forward_classification_valid
            & (forward_classification_incoming_count == 2)
        )
        backward_wall_solve = dict(backward_wall_solve)
        forward_wall_solve = dict(forward_wall_solve)
        backward_wall_solve["incoming_count"] = backward_incoming_count
        forward_wall_solve["incoming_count"] = forward_incoming_count
        backward_wall_solve["solve_valid"] = (
            backward_wall_solve["solve_valid"] & backward_count_valid
        )
        forward_wall_solve["solve_valid"] = (
            forward_wall_solve["solve_valid"] & forward_count_valid
        )
        wall_minus = jnp.where(
            backward_count_valid[..., None], wall_minus, jnp.nan
        )
        wall_plus = jnp.where(
            forward_count_valid[..., None], wall_plus, jnp.nan
        )
    minus_used = jnp.where(backward_wall[..., None], wall_minus, minus)
    plus_used = jnp.where(forward_wall[..., None], wall_plus, plus)

    backward_action, backward_valid_live, _ = _live_characteristic_leg_action(
        backward_face_used, center - minus_used, tau, mu, 1.0, branch="plus",
        eigenvalue_tolerance=eigenvalue_tolerance, max_condition=max_condition,
        basis=backward_basis,
    )
    forward_action, forward_valid_live, _ = _live_characteristic_leg_action(
        forward_face_used, plus_used - center, tau, mu, 1.0, branch="minus",
        eigenvalue_tolerance=eigenvalue_tolerance, max_condition=max_condition,
        basis=forward_basis,
    )

    # Form the same split matrices as the live action, including the finite
    # Rusanov fallback.  These matrices are frozen only through their
    # eigensystem; the face matrix itself is evaluated at the current face.
    backward_matrix = parallel_matrix_from_state(backward_face_used, tau, mu)
    forward_matrix = parallel_matrix_from_state(forward_face_used, tau, mu)
    backward_safe = jnp.where(jnp.isfinite(backward_matrix), backward_matrix, 0.0)
    forward_safe = jnp.where(jnp.isfinite(forward_matrix), forward_matrix, 0.0)
    backward_plus_matrix = jnp.where(
        backward_valid[..., None, None],
        jnp.einsum("...ij,...jk->...ik", backward_matrix, backward_wall_plus),
        0.5 * (backward_safe + backward_alpha[..., None, None] * eye),
    )
    forward_minus_matrix = jnp.where(
        forward_valid[..., None, None],
        jnp.einsum("...ij,...jk->...ik", forward_matrix, forward_wall_minus),
        0.5 * (forward_safe - forward_alpha[..., None, None] * eye),
    )
    dxm_safe = jnp.maximum(jnp.abs(dx_minus), _LOG_FLOOR)
    dxp_safe = jnp.maximum(jnp.abs(dx_plus), _LOG_FLOOR)
    backward_jacobian = -backward_plus_matrix / dxm_safe[..., None, None]
    forward_jacobian = forward_minus_matrix / dxp_safe[..., None, None]

    # ``wall_minus``/``wall_plus`` are frozen, first-order characteristic
    # traces, not nonlinear primitive states.  A composite quantity exported
    # from this characteristic solve must therefore use the Jacobian of that
    # quantity at the same canonical wall state.  Evaluating
    # ``n * (Vi - Ve)`` on the projected components would add the uncontrolled
    # quadratic product ``delta_n * (delta_Vi - delta_Ve)``.  This term can be
    # enormous when a wall mismatch has large modal components even though
    # the first-order characteristic current remains moderate.
    center_current = center[..., 0] * (center[..., 3] - center[..., 4])

    def linearized_current(endpoint: jnp.ndarray) -> jnp.ndarray:
        delta = endpoint - center
        return (
            center_current
            + (center[..., 3] - center[..., 4]) * delta[..., 0]
            + center[..., 0] * (delta[..., 3] - delta[..., 4])
        )

    backward_wall_nonlinear_current = wall_minus[..., 0] * (
        wall_minus[..., 3] - wall_minus[..., 4]
    )
    forward_wall_nonlinear_current = wall_plus[..., 0] * (
        wall_plus[..., 3] - wall_plus[..., 4]
    )
    if parallel_characteristic_wall_law == "velocity-no-flow":
        # The no-flow law solves Vi=Ve exactly, so its physical wall current
        # is exactly zero.  Preserve that nonlinear invariant when exporting
        # the current to characteristic-SAT/vorticity; the linearized form
        # would spuriously retain (Vi-Ve)*delta_n.
        backward_wall_characteristic_current = backward_wall_nonlinear_current
        forward_wall_characteristic_current = forward_wall_nonlinear_current
    else:
        backward_wall_characteristic_current = linearized_current(wall_minus)
        forward_wall_characteristic_current = linearized_current(wall_plus)
    backward_ordinary_current = minus[..., 0] * (minus[..., 3] - minus[..., 4])
    forward_ordinary_current = plus[..., 0] * (plus[..., 3] - plus[..., 4])
    backward_endpoint_current = jnp.where(
        backward_wall,
        backward_wall_characteristic_current,
        backward_ordinary_current,
    )
    forward_endpoint_current = jnp.where(
        forward_wall,
        forward_wall_characteristic_current,
        forward_ordinary_current,
    )
    backward_solve_valid = backward_wall_solve["solve_valid"]
    forward_solve_valid = forward_wall_solve["solve_valid"]
    backward_solve_fallback = backward_wall_solve.get(
        "fallback", ~backward_solve_valid
    ) | ~backward_solve_valid
    forward_solve_fallback = forward_wall_solve.get(
        "fallback", ~forward_solve_valid
    ) | ~forward_solve_valid
    backward_thermo = backward_wall_solve["thermodynamic_admissible"]
    forward_thermo = forward_wall_solve["thermodynamic_admissible"]
    backward_positivity_limited = backward_wall_solve.get(
        "positivity_limited", jnp.zeros_like(backward_solve_valid)
    )
    forward_positivity_limited = forward_wall_solve.get(
        "positivity_limited", jnp.zeros_like(forward_solve_valid)
    )
    # Candidate validity is deliberately irrelevant to an active energy-law
    # or no-flow wall: those laws do not consume the mapped primitive trace.
    # Preserve raw candidate diagnostics, however, so callers can distinguish
    # an ignored trace from a finite trace rather than losing evidence that
    # supplied data were non-finite.
    backward_candidate_finite = ~backward_candidate_fallback
    forward_candidate_finite = ~forward_candidate_fallback
    backward_candidate_ignored = (
        (parallel_characteristic_wall_law in ("energy-absorbing", "velocity-no-flow"))
        & backward_wall
    )
    forward_candidate_ignored = (
        (parallel_characteristic_wall_law in ("energy-absorbing", "velocity-no-flow"))
        & forward_wall
    )
    reported_backward_candidate_fallback = jnp.where(
        backward_candidate_ignored,
        jnp.zeros_like(backward_candidate_fallback),
        backward_candidate_fallback,
    )
    reported_forward_candidate_fallback = jnp.where(
        forward_candidate_ignored,
        jnp.zeros_like(forward_candidate_fallback),
        forward_candidate_fallback,
    )
    if parallel_characteristic_wall_law == "energy-absorbing":
        reported_backward_candidate = jnp.where(
            backward_candidate_ignored[..., None], equilibrium, backward_candidate
        )
        reported_forward_candidate = jnp.where(
            forward_candidate_ignored[..., None], equilibrium, forward_candidate
        )
    else:
        reported_backward_candidate = backward_candidate
        reported_forward_candidate = forward_candidate
    # A mapped endpoint is not read on an energy-law or no-flow wall leg
    # because those direct/nonlinear laws construct their own wall state.
    # Suppress only that direction's clipping flag; ordinary NaN endpoints
    # remain invalid diagnostics.
    backward_clipped = jnp.where(
        backward_candidate_ignored, jnp.zeros_like(backward_clipped), backward_clipped
    )
    forward_clipped = jnp.where(
        forward_candidate_ignored, jnp.zeros_like(forward_clipped), forward_clipped
    )
    info = {
        "backward_wall": backward_wall,
        "forward_wall": forward_wall,
        "backward_clipped": backward_clipped,
        "forward_clipped": forward_clipped,
        "backward_candidate_fallback": reported_backward_candidate_fallback,
        "forward_candidate_fallback": reported_forward_candidate_fallback,
        "backward_candidate_finite": backward_candidate_finite,
        "forward_candidate_finite": forward_candidate_finite,
        "backward_candidate_ignored": backward_candidate_ignored,
        "forward_candidate_ignored": forward_candidate_ignored,
        "backward_wall_solve_fallback": backward_solve_fallback,
        "forward_wall_solve_fallback": forward_solve_fallback,
        "backward_wall_solve_valid": backward_solve_valid,
        "forward_wall_solve_valid": forward_solve_valid,
        "backward_wall_thermodynamic_admissible": backward_thermo,
        "forward_wall_thermodynamic_admissible": forward_thermo,
        "backward_wall_positivity_limited": backward_positivity_limited,
        "forward_wall_positivity_limited": forward_positivity_limited,
        "backward_wall_residual_norm": backward_wall_solve.get(
            "residual_norm", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_residual_norm": forward_wall_solve.get(
            "residual_norm", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_retained_error": backward_wall_solve.get(
            "retained_error", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_retained_error": forward_wall_solve.get(
            "retained_error", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_relative_residual": backward_wall_solve.get(
            "relative_residual", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_relative_residual": forward_wall_solve.get(
            "relative_residual", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_correction_amplification": backward_wall_solve.get(
            "correction_amplification", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_correction_amplification": forward_wall_solve.get(
            "correction_amplification", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_incoming_rank": backward_wall_solve["incoming_rank"],
        "forward_wall_incoming_rank": forward_wall_solve["incoming_rank"],
        "backward_wall_incoming_count": backward_incoming_count,
        "forward_wall_incoming_count": forward_incoming_count,
        "backward_wall_classification_incoming_count": (
            backward_classification_incoming_count
        ),
        "forward_wall_classification_incoming_count": (
            forward_classification_incoming_count
        ),
        "backward_wall_interior_sign_incoming_count": (
            backward_interior_incoming_count
        ),
        "forward_wall_interior_sign_incoming_count": (
            forward_interior_incoming_count
        ),
        "backward_wall_classification_stationary_count": (
            backward_classification_stationary_count
        ),
        "forward_wall_classification_stationary_count": (
            forward_classification_stationary_count
        ),
        "backward_wall_classification_valid": backward_classification_valid,
        "forward_wall_classification_valid": forward_classification_valid,
        "backward_wall_outgoing_rank": backward_wall_solve.get(
            "outgoing_rank", jnp.zeros_like(backward_solve_valid, dtype=jnp.int32)
        ),
        "forward_wall_outgoing_rank": forward_wall_solve.get(
            "outgoing_rank", jnp.zeros_like(forward_solve_valid, dtype=jnp.int32)
        ),
        "backward_wall_boundary_power_before": backward_wall_solve.get(
            "boundary_power_before", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_boundary_power_after": backward_wall_solve.get(
            "boundary_power_after", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_boundary_power_before": forward_wall_solve.get(
            "boundary_power_before", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_boundary_power_after": forward_wall_solve.get(
            "boundary_power_after", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_incoming_energy_before": backward_wall_solve.get(
            "incoming_energy_before", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "backward_wall_incoming_energy_after": backward_wall_solve.get(
            "incoming_energy_after", jnp.zeros_like(backward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_incoming_energy_before": forward_wall_solve.get(
            "incoming_energy_before", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "forward_wall_incoming_energy_after": forward_wall_solve.get(
            "incoming_energy_after", jnp.zeros_like(forward_solve_valid, dtype=jnp.float64)
        ),
        "backward_valid": backward_valid_live,
        "forward_valid": forward_valid_live,
        "backward_alpha": backward_alpha,
        "forward_alpha": forward_alpha,
        # These are the actual endpoint values consumed by the characteristic
        # update.  For ordinary mapped legs they are exactly the supplied
        # ``minus``/``plus`` states; only physical wall legs use the projected
        # candidate values.
        "backward_wall_projected_state": wall_minus,
        "forward_wall_projected_state": wall_plus,
        "backward_projected_state": minus_used,
        "forward_projected_state": plus_used,
        "backward_endpoint_state": minus_used,
        "forward_endpoint_state": plus_used,
        "backward_incoming_projector": backward_wall_plus,
        "forward_incoming_projector": forward_wall_minus,
        "backward_incoming_action": (
            wall_minus - center
            if parallel_characteristic_wall_law
            in ("energy-absorbing", "velocity-no-flow")
            else _matvec(backward_wall_plus, backward_candidate - center)
        ),
        "forward_incoming_action": (
            wall_plus - center
            if parallel_characteristic_wall_law
            in ("energy-absorbing", "velocity-no-flow")
            else _matvec(forward_wall_minus, forward_candidate - center)
        ),
        "backward_incoming_matrix": backward_plus_matrix,
        "forward_incoming_matrix": forward_minus_matrix,
        "backward_candidate_current": reported_backward_candidate[..., 0]
        * (reported_backward_candidate[..., 3] - reported_backward_candidate[..., 4]),
        "forward_candidate_current": reported_forward_candidate[..., 0]
        * (reported_forward_candidate[..., 3] - reported_forward_candidate[..., 4]),
        # These are the first-order currents carried by the incoming wall
        # characteristics.  Retain the old ``*_wall_projected_current`` names
        # as aliases for callers, but make their now-correct linearized
        # meaning explicit through the ``*_characteristic_current`` keys.
        "backward_wall_characteristic_current": backward_wall_characteristic_current,
        "forward_wall_characteristic_current": forward_wall_characteristic_current,
        "backward_wall_projected_current": backward_wall_characteristic_current,
        "forward_wall_projected_current": forward_wall_characteristic_current,
        "backward_wall_projected_nonlinear_current": backward_wall_nonlinear_current,
        "forward_wall_projected_nonlinear_current": forward_wall_nonlinear_current,
        "backward_wall_current_quadratic_remainder": (
            backward_wall_nonlinear_current - backward_wall_characteristic_current
        ),
        "forward_wall_current_quadratic_remainder": (
            forward_wall_nonlinear_current - forward_wall_characteristic_current
        ),
        "backward_projected_current": backward_endpoint_current,
        "forward_projected_current": forward_endpoint_current,
        "backward_endpoint_current": backward_endpoint_current,
        "forward_endpoint_current": forward_endpoint_current,
        "wall_row": backward_wall | forward_wall,
        "ordinary_row": ~(backward_wall | forward_wall),
    }
    return (
        center, backward_action, forward_action, backward_jacobian,
        forward_jacobian, info,
    )


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
    parallel_characteristic_wall_law: str = "primitive-least-residual",
    div_b: Any = 0.0,
    selection_dt: Any = 0.0,
    cfl_limit: float = 2.785,
    parallel_short_leg_selection: str = "cfl",
    omit_backward_wall: Any = False,
    omit_forward_wall: Any = False,
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply the live-face production material update to every mapped row.

    The backward fluctuation is evaluated on the oriented path
    ``minus -> center`` and the forward fluctuation on ``center -> plus``:

    ``residual = -(D_plus_backward / dx_minus + D_minus_forward / dx_plus)``.

    A wall endpoint replaces the mapped endpoint by a primitive-boundary
    residual solve over the complete incoming characteristic subspace;
    ordinary rows simply use their supplied mapped endpoints.  This makes wall
    and bulk legs one operator with different endpoint data, rather than two
    numerical fluxes.  ``div_b`` supplies the exact geometric source omitted from the
    frozen principal matrix.  With ``parallel_short_leg_selection='cfl'`` the
    selected physical-wall directions are those whose characteristic CFL
    exceeds ``cfl_limit``; ``'all-physical-walls'`` selects every physical
    wall direction and never ordinary mapped legs.  Selected directions are
    omitted from this explicit material contribution; the caller can add them
    with :func:`parallel_short_wall_backward_euler`.  Each leg uses one
    canonical face state and one live characteristic eigendecomposition.
    """
    div_b = jnp.asarray(div_b, dtype=jnp.float64)
    (
        center, backward_action, forward_action, backward_jacobian,
        forward_jacobian, directional,
    ) = _material_directional_data(
        center, minus, plus, dx_minus, dx_plus, tau, mu,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=backward_wall_state,
        forward_wall_state=forward_wall_state, equilibrium=equilibrium,
        parallel_characteristic_wall_law=parallel_characteristic_wall_law,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    backward_wall = directional["backward_wall"]
    forward_wall = directional["forward_wall"]
    dx_minus = jnp.asarray(dx_minus, dtype=jnp.float64)
    dx_plus = jnp.asarray(dx_plus, dtype=jnp.float64)
    dx_minus, dx_plus, div_b = jnp.broadcast_arrays(dx_minus, dx_plus, div_b)
    selection_dt = jnp.asarray(selection_dt, dtype=jnp.float64)
    selection_dt, dx_minus, dx_plus = jnp.broadcast_arrays(
        selection_dt, dx_minus, dx_plus
    )
    if parallel_short_leg_selection not in ("cfl", "all-physical-walls"):
        raise ValueError(
            "parallel_short_leg_selection must be 'cfl' or "
            "'all-physical-walls', got "
            f"{parallel_short_leg_selection!r}"
        )
    dxm_safe = jnp.maximum(jnp.abs(dx_minus), _LOG_FLOOR)
    dxp_safe = jnp.maximum(jnp.abs(dx_plus), _LOG_FLOOR)
    backward_cfl = jnp.abs(selection_dt) * directional["backward_alpha"] / dxm_safe
    forward_cfl = jnp.abs(selection_dt) * directional["forward_alpha"] / dxp_safe
    if parallel_short_leg_selection == "all-physical-walls":
        selected_backward = backward_wall
        selected_forward = forward_wall
    else:
        selected_backward = backward_wall & (backward_cfl > cfl_limit)
        selected_forward = forward_wall & (forward_cfl > cfl_limit)
    omit_backward = selected_backward | (
        backward_wall & jnp.asarray(omit_backward_wall, dtype=bool)
    )
    omit_forward = selected_forward | (
        forward_wall & jnp.asarray(omit_forward_wall, dtype=bool)
    )
    backward_action = jnp.where(omit_backward[..., None], 0.0, backward_action)
    forward_action = jnp.where(omit_forward[..., None], 0.0, forward_action)
    residual = -(
        backward_action / jnp.maximum(jnp.abs(dx_minus), _LOG_FLOOR)[..., None]
        + forward_action / jnp.maximum(jnp.abs(dx_plus), _LOG_FLOOR)[..., None]
    )
    backward_valid_live = directional["backward_valid"]
    forward_valid_live = directional["forward_valid"]
    backward_clipped = directional["backward_clipped"]
    forward_clipped = directional["forward_clipped"]
    backward_candidate_fallback = directional["backward_candidate_fallback"]
    forward_candidate_fallback = directional["forward_candidate_fallback"]

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
        "spectral_fallback": ~backward_valid_live | ~forward_valid_live,
        "backward_clipped": backward_clipped,
        "forward_clipped": forward_clipped,
        "positivity_fallback": backward_clipped | forward_clipped,
        "backward_candidate_fallback": backward_candidate_fallback,
        "forward_candidate_fallback": forward_candidate_fallback,
        "backward_candidate_finite": directional["backward_candidate_finite"],
        "forward_candidate_finite": directional["forward_candidate_finite"],
        "backward_candidate_ignored": directional["backward_candidate_ignored"],
        "forward_candidate_ignored": directional["forward_candidate_ignored"],
        "backward_wall_incoming_count": directional["backward_wall_incoming_count"],
        "forward_wall_incoming_count": directional["forward_wall_incoming_count"],
        "backward_wall_solve_valid": directional["backward_wall_solve_valid"],
        "forward_wall_solve_valid": directional["forward_wall_solve_valid"],
        "backward_wall_retained_error": directional["backward_wall_retained_error"],
        "forward_wall_retained_error": directional["forward_wall_retained_error"],
        "wall_spectral_fallback": (~backward_valid_live & backward_wall) | (~forward_valid_live & forward_wall),
        "fallback": (
            (~backward_valid_live | ~forward_valid_live)
            | backward_clipped | forward_clipped
            | backward_candidate_fallback | forward_candidate_fallback
            | (
                backward_wall
                & directional["backward_wall_solve_fallback"]
            )
            | (
                forward_wall
                & directional["forward_wall_solve_fallback"]
            )
            | (
                backward_wall
                & ~directional["backward_wall_thermodynamic_admissible"]
            )
            | (
                forward_wall
                & ~directional["forward_wall_thermodynamic_admissible"]
            )
        ),
        "admissible": (
            backward_valid_live & forward_valid_live
            & ~backward_clipped & ~forward_clipped
            & ~backward_candidate_fallback & ~forward_candidate_fallback
            & ~(
                backward_wall
                & directional["backward_wall_solve_fallback"]
            )
            & ~(
                forward_wall
                & directional["forward_wall_solve_fallback"]
            )
            & ~(
                backward_wall
                & ~directional["backward_wall_thermodynamic_admissible"]
            )
            & ~(
                forward_wall
                & ~directional["forward_wall_thermodynamic_admissible"]
            )
        ),
        "omitted_backward_wall": omit_backward,
        "omitted_forward_wall": omit_forward,
        "backward_cfl": backward_cfl,
        "forward_cfl": forward_cfl,
        "selected_backward_wall": selected_backward,
        "selected_forward_wall": selected_forward,
        "selected_wall": selected_backward | selected_forward,
    }
    return residual, diagnostics


def parallel_short_wall_material_data(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: Any,
    dx_plus: Any,
    tau: Any,
    mu: Any,
    *,
    selection_dt: Any = 0.0,
    cfl_limit: float = 2.785,
    parallel_short_leg_selection: str = "cfl",
    backward_wall: Any = False,
    forward_wall: Any = False,
    backward_wall_state: jnp.ndarray | None = None,
    forward_wall_state: jnp.ndarray | None = None,
    equilibrium: jnp.ndarray | None = None,
    parallel_characteristic_wall_law: str = "primitive-least-residual",
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return the selected short-wall material residual and frozen Jacobian.

    In ``'cfl'`` mode a direction is selected only when it is a physical wall
    leg and ``abs(selection_dt) * alpha / abs(dx) > cfl_limit``, where ``alpha``
    is the largest characteristic speed of that canonical face state.
    ``'all-physical-walls'`` selects every physical wall leg.  The returned
    ``residual`` is the sum of the selected backward and forward directional
    residuals, and ``jacobian`` is the corresponding frozen derivative with
    respect to the owner state.  No ``div_b`` source is included.  Neighbor
    and wall states, face coefficients, and the live eigensystem are all held
    fixed in this local linearization.

    The directional signs are those of the explicit material update:

    ``r_backward = (-A_plus/dx_minus) (center - wall_minus)``

    ``r_forward = (+A_minus/dx_plus) (center - wall_plus)``.
    """

    (
        center, backward_action, forward_action, backward_jacobian,
        forward_jacobian, info,
    ) = _material_directional_data(
        center, minus, plus, dx_minus, dx_plus, tau, mu,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=backward_wall_state,
        forward_wall_state=forward_wall_state, equilibrium=equilibrium,
        parallel_characteristic_wall_law=parallel_characteristic_wall_law,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    dx_minus = jnp.asarray(dx_minus, dtype=jnp.float64)
    dx_plus = jnp.asarray(dx_plus, dtype=jnp.float64)
    dx_minus, dx_plus = jnp.broadcast_arrays(dx_minus, dx_plus)
    dt = jnp.asarray(selection_dt, dtype=jnp.float64)
    dt, dx_minus, dx_plus = jnp.broadcast_arrays(dt, dx_minus, dx_plus)
    dxm_safe = jnp.maximum(jnp.abs(dx_minus), _LOG_FLOOR)
    dxp_safe = jnp.maximum(jnp.abs(dx_plus), _LOG_FLOOR)
    backward_cfl = jnp.abs(dt) * info["backward_alpha"] / dxm_safe
    forward_cfl = jnp.abs(dt) * info["forward_alpha"] / dxp_safe
    if parallel_short_leg_selection not in ("cfl", "all-physical-walls"):
        raise ValueError(
            "parallel_short_leg_selection must be 'cfl' or "
            "'all-physical-walls', got "
            f"{parallel_short_leg_selection!r}"
        )
    if parallel_short_leg_selection == "all-physical-walls":
        selected_backward = info["backward_wall"]
        selected_forward = info["forward_wall"]
    else:
        selected_backward = info["backward_wall"] & (backward_cfl > cfl_limit)
        selected_forward = info["forward_wall"] & (forward_cfl > cfl_limit)
    backward_residual = -backward_action / dxm_safe[..., None]
    forward_residual = -forward_action / dxp_safe[..., None]
    selected_residual = (
        jnp.where(selected_backward[..., None], backward_residual, 0.0)
        + jnp.where(selected_forward[..., None], forward_residual, 0.0)
    )
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), backward_jacobian.shape)
    selected_jacobian = (
        jnp.where(selected_backward[..., None, None], backward_jacobian, 0.0)
        + jnp.where(selected_forward[..., None, None], forward_jacobian, 0.0)
    )
    info = dict(info)
    info.update({
        "backward_cfl": backward_cfl,
        "forward_cfl": forward_cfl,
        "selected_backward_wall": selected_backward,
        "selected_forward_wall": selected_forward,
        "selected_wall": selected_backward | selected_forward,
        "backward_residual": backward_residual,
        "forward_residual": forward_residual,
        "backward_jacobian": backward_jacobian,
        "forward_jacobian": forward_jacobian,
        "selected_jacobian": selected_jacobian,
        "finite": jnp.all(jnp.isfinite(selected_residual), axis=-1)
        & jnp.all(jnp.isfinite(selected_jacobian), axis=(-2, -1)),
    })
    return selected_residual, selected_jacobian, info


def parallel_short_wall_backward_euler(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: Any,
    dx_plus: Any,
    tau: Any,
    mu: Any,
    *,
    selection_dt: Any,
    solve_dt: Any | None = None,
    cfl_limit: float = 2.785,
    parallel_short_leg_selection: str = "cfl",
    backward_wall: Any = False,
    forward_wall: Any = False,
    backward_wall_state: jnp.ndarray | None = None,
    forward_wall_state: jnp.ndarray | None = None,
    equilibrium: jnp.ndarray | None = None,
    parallel_characteristic_wall_law: str = "primitive-least-residual",
    coupled_residual: jnp.ndarray | None = None,
    coupled_jacobian: jnp.ndarray | None = None,
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply one local backward-Euler increment to selected wall rows.

    ``coupled_residual`` and ``coupled_jacobian`` let the caller hand off
    terms that belong to the same selected-row principal balance but are
    assembled outside the characteristic material operator.  They are
    masked by ``selected_wall`` here, so an unselected row still returns an
    exactly zero increment.  The complete update is

    ``delta = (I - solve_dt*(J_material + J_coupled))^-1``
    ``        * solve_dt*(r_material + r_coupled)``.

    If a frozen local solve is non-finite, its raw non-finite increment is
    preserved in ``updated`` and
    ``implicit_solve_fallback``/``implicit_finite`` report the failure; no
    silent zero fallback is applied.
    """

    if solve_dt is None:
        solve_dt = selection_dt
    selected_residual, selected_jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, tau, mu,
        selection_dt=selection_dt, cfl_limit=cfl_limit,
        parallel_short_leg_selection=parallel_short_leg_selection,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=backward_wall_state,
        forward_wall_state=forward_wall_state, equilibrium=equilibrium,
        parallel_characteristic_wall_law=parallel_characteristic_wall_law,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    material_residual = selected_residual
    material_jacobian = selected_jacobian
    if coupled_residual is None:
        coupled_residual = jnp.zeros_like(selected_residual)
    else:
        coupled_residual = jnp.asarray(coupled_residual, dtype=jnp.float64)
        if coupled_residual.shape != selected_residual.shape:
            raise ValueError(
                "coupled_residual must have shape "
                f"{selected_residual.shape}, got {coupled_residual.shape}"
            )
    if coupled_jacobian is None:
        coupled_jacobian = jnp.zeros_like(selected_jacobian)
    else:
        coupled_jacobian = jnp.asarray(coupled_jacobian, dtype=jnp.float64)
        if coupled_jacobian.shape != selected_jacobian.shape:
            raise ValueError(
                "coupled_jacobian must have shape "
                f"{selected_jacobian.shape}, got {coupled_jacobian.shape}"
            )
    selected_wall = info["selected_wall"]
    selected_coupled_residual = jnp.where(
        selected_wall[..., None], coupled_residual, 0.0
    )
    selected_coupled_jacobian = jnp.where(
        selected_wall[..., None, None], coupled_jacobian, 0.0
    )
    selected_residual = material_residual + selected_coupled_residual
    selected_jacobian = material_jacobian + selected_coupled_jacobian
    solve_dt = jnp.asarray(solve_dt, dtype=jnp.float64)
    solve_dt = jnp.broadcast_to(solve_dt, selected_residual.shape[:-1])
    eye = jnp.broadcast_to(
        jnp.eye(STATE_SIZE, dtype=jnp.float64), selected_jacobian.shape
    )
    system = eye - solve_dt[..., None, None] * selected_jacobian
    rhs = solve_dt[..., None] * selected_residual
    # Explicitly add a singleton RHS dimension for JAX's batched solve API;
    # newer JAX releases no longer infer a batched one-vector solve.
    delta = jnp.linalg.solve(system, rhs[..., None])[..., 0]
    solve_finite = jnp.all(jnp.isfinite(delta), axis=-1)
    updated = center + delta
    info = dict(info)
    info["selected_material_residual"] = material_residual
    info["selected_material_jacobian"] = material_jacobian
    info["selected_coupled_residual"] = selected_coupled_residual
    info["selected_coupled_jacobian"] = selected_coupled_jacobian
    info["selected_complete_residual"] = selected_residual
    info["selected_complete_jacobian"] = selected_jacobian
    info["implicit_solve_fallback"] = ~solve_finite
    info["implicit_finite"] = jnp.all(jnp.isfinite(updated), axis=-1)
    return updated, delta, info


def parallel_characteristic_wall_data(
    center: jnp.ndarray,
    minus: jnp.ndarray,
    plus: jnp.ndarray,
    dx_minus: Any,
    dx_plus: Any,
    tau: Any,
    mu: Any,
    *,
    selection_dt: Any = 0.0,
    cfl_limit: float = 2.785,
    parallel_short_leg_selection: str = "cfl",
    backward_wall: Any = False,
    forward_wall: Any = False,
    backward_wall_state: jnp.ndarray | None = None,
    forward_wall_state: jnp.ndarray | None = None,
    equilibrium: jnp.ndarray | None = None,
    parallel_characteristic_wall_law: str = "primitive-least-residual",
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = _DEFAULT_EIG_TOL,
    max_condition: float = _DEFAULT_MAX_CONDITION,
) -> dict[str, jnp.ndarray]:
    """Return the complete projected wall data consumed by the material flux.

    This boundary-facing view keeps the endpoint states and currents together
    with the incoming projectors/actions, split matrices, validity flags, and
    CFL diagnostics.  Ordinary rows expose their supplied mapped endpoints in
    ``*_endpoint_state``; no equilibrium candidate is substituted there.
    ``selected_residual`` and ``selected_jacobian`` are included for callers
    implementing the short-leg local solve.  The function is a thin wrapper
    around :func:`parallel_short_wall_material_data`, so it cannot drift from
    the residual's wall closure.
    """

    selected_residual, selected_jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, tau, mu,
        selection_dt=selection_dt, cfl_limit=cfl_limit,
        parallel_short_leg_selection=parallel_short_leg_selection,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=backward_wall_state,
        forward_wall_state=forward_wall_state, equilibrium=equilibrium,
        parallel_characteristic_wall_law=parallel_characteristic_wall_law,
        positivity_floor=positivity_floor,
        eigenvalue_tolerance=eigenvalue_tolerance,
        max_condition=max_condition,
    )
    info = dict(info)
    info["selected_residual"] = selected_residual
    info["selected_jacobian"] = selected_jacobian
    return info


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
    "parallel_canonical_leg_face_state",
    "third_order_face_reconstruction",
    "parallel_target_row_material_residual",
    "parallel_short_wall_material_data",
    "parallel_short_wall_backward_euler",
    "parallel_characteristic_wall_data",
]
