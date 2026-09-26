"""Production-ready local curvature characteristic flux primitives.

This module intentionally contains no geometry or RHS wiring.  It provides the
small, pure-JAX pieces used by the owner-face production curvature operator:

* the strict DAE-reduced four-field curvature symbol;
* a robust characteristic absolute action;
* a canonical-face live characteristic fluctuation;
* first- and third-order face reconstruction with explicit fallback metadata.

The state order throughout is ``(n, T_e, T_i, omega)``.  The strict symbol
holds the non-local polarization potential fixed.  An optional finite
``k_perp_squared`` can restore the non-local omega column for diagnostics, but
it is not used by the production local characteristic flux.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


Array = jax.Array
STATE_SIZE = 4
POSITIVE_COMPONENTS = (0, 1, 2)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ReconstructionMetadata:
    """JAX-pytree metadata emitted by face reconstruction.

    ``used_fallback`` has shape ``(..., 2)`` (left and right face states),
    while ``order_used`` has the same shape and contains the actual order used
    on each side.  Keeping this information alongside the states makes a
    positivity fallback observable in production diagnostics.
    """

    used_fallback: Array
    order_used: Array

    def tree_flatten(self):
        return (self.used_fallback, self.order_used), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def _broadcast_state(*values: Array | float) -> tuple[Array, ...]:
    arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in values)
    shape = jnp.broadcast_shapes(*(array.shape[:-1] for array in arrays))
    return tuple(jnp.broadcast_to(array, shape + (array.shape[-1],)) for array in arrays)


def _require_state(state: Array, name: str = "state") -> Array:
    value = jnp.asarray(state, dtype=jnp.float64)
    if value.ndim < 1 or value.shape[-1] != STATE_SIZE:
        raise ValueError(f"{name} must have trailing shape (4,), got {value.shape}")
    return value


def curvature_principal_matrix(
    density: Array | float,
    Te: Array | float,
    Ti: Array | float,
    bmag: Array | float,
    tau: Array | float,
    *,
    k_perp_squared: Array | float | None = None,
) -> Array:
    """Return the corrected DAE-reduced curvature principal matrix.

    The rows and columns are ordered ``(n, Te, Ti, omega)``.  This is the
    strict local matrix used for characteristic curvature fluxes; in
    particular, the ``Ti`` column includes the polarization response
    ``(2 n tau, 4 tau Te/3, -2 tau Ti, 2 tau B^2)``.  If
    ``k_perp_squared`` is supplied, the optional non-local omega column is
    included as a diagnostic using the local Fourier relation
    ``delta phi = -tau delta Ti - delta omega/k_perp_squared``.
    """

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
    return matrix


def curvature_strict_principal_matrix(
    state: Array, bmag: Array | float, tau: Array | float
) -> Array:
    """Convenience wrapper for :func:`curvature_principal_matrix`."""

    state = _require_state(state)
    return curvature_principal_matrix(state[..., 0], state[..., 1], state[..., 2], bmag, tau)


def _safe_spectral_data(
    matrix: Array,
    *,
    real_tolerance: float,
    max_condition: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return eigenvalues, projectors, validity, and fallback scale.

    Eigenvectors are stopped-gradient by design: the characteristic basis is
    frozen during Newton/JAX differentiation while the matrix action remains
    live.  Invalid/complex/ill-conditioned spectra are selected away from the
    spectral result by ``jnp.where`` and use a scalar singular-value fallback.
    """

    frozen = jax.lax.stop_gradient(matrix)
    eigenvalues, vectors = jnp.linalg.eig(frozen)
    # These are small square characteristic bases.  A direct inverse avoids
    # the SVD used by ``pinv``; defective/ill-conditioned bases are rejected
    # below and use the finite Rusanov fallback instead.
    inverse = jnp.linalg.inv(vectors)
    eigenvalues = jnp.nan_to_num(eigenvalues, nan=0.0, posinf=0.0, neginf=0.0)
    vectors = jnp.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    inverse = jnp.nan_to_num(inverse, nan=0.0, posinf=0.0, neginf=0.0)
    imaginary = jnp.abs(jnp.imag(eigenvalues))
    real_values = jnp.real(eigenvalues)
    matrix_norm = jnp.linalg.norm(frozen, axis=(-2, -1))
    vector_norm = jnp.linalg.norm(vectors, axis=(-2, -1))
    inverse_norm = jnp.linalg.norm(inverse, axis=(-2, -1))
    condition = vector_norm * inverse_norm
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    valid = finite & jnp.all(imaginary <= real_tolerance * (1.0 + jnp.abs(real_values)), axis=-1)
    valid = valid & jnp.isfinite(condition) & (condition <= max_condition)
    # The Frobenius norm is a conservative upper bound on the spectral norm
    # and avoids a second SVD at every path node.  ``sigma I`` remains a
    # symmetric positive-semidefinite absolute action on fallback points.
    sigma = jnp.where(jnp.isfinite(matrix_norm), matrix_norm, 0.0)
    return real_values, vectors, inverse, valid, sigma


def curvature_characteristic_absolute_action(
    matrix: Array,
    vector: Array,
    *,
    real_tolerance: float = 1.0e-10,
    max_condition: float = 1.0e8,
    return_fallback: bool = False,
) -> Array | tuple[Array, Array]:
    """Apply the robust characteristic absolute matrix to one/batched vectors."""

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    vector = _require_state(vector, "vector")
    if matrix.shape[-2:] != (STATE_SIZE, STATE_SIZE):
        raise ValueError(f"matrix must end in (4, 4), got {matrix.shape}")
    shape = jnp.broadcast_shapes(matrix.shape[:-2], vector.shape[:-1])
    matrix = jnp.broadcast_to(matrix, shape + (STATE_SIZE, STATE_SIZE))
    vector = jnp.broadcast_to(vector, shape + (STATE_SIZE,))
    eigenvalues, vectors, inverse, valid, sigma = _safe_spectral_data(
        matrix, real_tolerance=real_tolerance, max_condition=max_condition
    )
    projected = jnp.einsum("...ij,...j->...i", inverse, vector)
    projected = jnp.einsum("...ij,...j->...i", vectors, jnp.abs(eigenvalues) * projected)
    fallback = sigma[..., None] * vector
    action = jnp.where(valid[..., None], jnp.real(projected), fallback)
    if return_fallback:
        return action, ~valid
    return action


def curvature_characteristic_absolute_matrix(
    matrix: Array,
    **kwargs,
) -> Array:
    """Materialize the robust absolute action as a matrix (diagnostic helper)."""

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    eye = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    columns = tuple(
        curvature_characteristic_absolute_action(matrix, eye[..., :, index], **kwargs)
        for index in range(STATE_SIZE)
    )
    return jnp.stack(columns, axis=-1)


def curvature_characteristic_metric(matrix: Array, **kwargs) -> Array:
    """Return ``H=sum(P_j.T P_j)`` or identity on the robust fallback path."""

    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    frozen = jax.lax.stop_gradient(matrix)
    _eig, vectors, inverse, valid, _sigma = _safe_spectral_data(
        frozen,
        real_tolerance=kwargs.get("real_tolerance", 1.0e-10),
        max_condition=kwargs.get("max_condition", 1.0e8),
    )
    projectors = jnp.einsum("...ik,...kl,...lj->...kij", vectors, jnp.eye(STATE_SIZE), inverse)
    # The above is one projector per eigenvector.  Eigenvalue ordering is
    # irrelevant because all projectors enter the metric symmetrically.
    metric = jnp.einsum("...kmi,...kmj->...ij", projectors, projectors)
    identity = jnp.broadcast_to(jnp.eye(STATE_SIZE, dtype=jnp.float64), matrix.shape)
    return jnp.where(valid[..., None, None], jnp.real(metric), identity)


def curvature_face_linearized_fluctuations(
    left: Array,
    right: Array,
    face_state: Array,
    bmag: Array | float,
    tau: Array | float,
    *,
    real_tolerance: float = 1.0e-10,
    max_condition: float = 1.0e8,
    normal: Array | float = 1.0,
    positivity_floor: float = 1.0e-12,
    return_fallback: bool = False,
    return_diagnostics: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, Array] | tuple[Array, Array, dict[str, Array]]:
    """Return p=1 fluctuations using one frozen matrix at the face.

    ``left`` and ``right`` are the reconstructed traces that define the
    jump.  ``face_state`` is deliberately a separate argument: production
    assembly supplies the canonical conservative face value for ordinary
    faces and the adjacent interior trace at a physical wall.  The strict
    curvature matrix is evaluated once at that state, then split as
    ``D+/- = (A dq +/- |A| dq)/2``.  Thus this is the face-linearized
    production characteristic update with no quadrature points or endpoint
    averaging in the runtime path.

    The returned order is ``(right_going, left_going)`` = ``(D+, D-)``.
    Non-finite or non-positive matrix states are sanitized for a finite
    characteristic action and reported through the same fallback mask used by
    the characteristic primitive.
    """
    left = _require_state(left, "left")
    right = _require_state(right, "right")
    face_state = _require_state(face_state, "face_state")
    shape = jnp.broadcast_shapes(
        left.shape[:-1], right.shape[:-1], face_state.shape[:-1],
        jnp.shape(jnp.asarray(bmag)), jnp.shape(jnp.asarray(normal)),
    )
    left = jnp.broadcast_to(left, shape + (STATE_SIZE,))
    right = jnp.broadcast_to(right, shape + (STATE_SIZE,))
    face_state = jnp.broadcast_to(face_state, shape + (STATE_SIZE,))
    bmag = jnp.broadcast_to(jnp.asarray(bmag, dtype=jnp.float64), shape)
    normal = jnp.broadcast_to(jnp.asarray(normal, dtype=jnp.float64), shape)

    default = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=jnp.float64)
    finite = jnp.all(jnp.isfinite(face_state), axis=-1)
    clean = jnp.where(finite[..., None], face_state, default)
    positive = jnp.maximum(clean[..., POSITIVE_COMPONENTS], positivity_floor)
    matrix_state = clean.at[..., POSITIVE_COMPONENTS].set(positive)
    clipped = (
        (~finite)
        | jnp.any(positive != clean[..., POSITIVE_COMPONENTS], axis=-1)
        | (~jnp.isfinite(bmag))
        | (~jnp.isfinite(normal))
    )
    b_safe = jnp.where(jnp.isfinite(bmag), bmag, 1.0)
    normal_safe = jnp.where(jnp.isfinite(normal), normal, 0.0)
    matrix = curvature_strict_principal_matrix(matrix_state, b_safe, tau)
    normal_matrix = normal_safe[..., None, None] * matrix
    jump = right - left
    absolute, spectral_fallback = curvature_characteristic_absolute_action(
        normal_matrix,
        jump,
        real_tolerance=real_tolerance,
        max_condition=max_condition,
        return_fallback=True,
    )
    material = jnp.einsum("...ij,...j->...i", normal_matrix, jump)
    right_fluctuation = 0.5 * (material + absolute)
    left_fluctuation = 0.5 * (material - absolute)
    fallback = clipped | spectral_fallback
    if return_diagnostics:
        return right_fluctuation, left_fluctuation, {
            "fallback": fallback,
            "spectral_fallback": spectral_fallback,
            "positivity_clipped": clipped,
            "admissible": ~fallback,
        }
    if return_fallback:
        return right_fluctuation, left_fluctuation, fallback
    return right_fluctuation, left_fluctuation


def reconstruct_first_order_face_states(
    left: Array,
    right: Array,
    *,
    positivity_floor: float = 1.0e-12,
) -> tuple[Array, Array, ReconstructionMetadata]:
    """Return piecewise-constant states and metadata (never falls back)."""

    left = _require_state(left, "left")
    right = _require_state(right, "right")
    shape = jnp.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left = jnp.broadcast_to(left, shape + (STATE_SIZE,))
    right = jnp.broadcast_to(right, shape + (STATE_SIZE,))
    left = left.at[..., POSITIVE_COMPONENTS].set(jnp.maximum(left[..., POSITIVE_COMPONENTS], positivity_floor))
    right = right.at[..., POSITIVE_COMPONENTS].set(jnp.maximum(right[..., POSITIVE_COMPONENTS], positivity_floor))
    zeros = jnp.zeros(shape + (1,), dtype=bool)
    orders = jnp.ones(shape + (1,), dtype=jnp.int32)
    return left, right, ReconstructionMetadata(jnp.concatenate((zeros, zeros), axis=-1), jnp.concatenate((orders, orders), axis=-1))


def reconstruct_third_order_face_states(
    q_im1: Array,
    q_i: Array,
    q_ip1: Array,
    q_ip2: Array,
    *,
    positivity_floor: float = 1.0e-12,
) -> tuple[Array, Array, ReconstructionMetadata]:
    """Reconstruct a third-order face and fall back side-wise if nonpositive."""

    q_im1, q_i, q_ip1, q_ip2 = (_require_state(value, name) for value, name in ((q_im1, "q_im1"), (q_i, "q_i"), (q_ip1, "q_ip1"), (q_ip2, "q_ip2")))
    q_im1, q_i, q_ip1, q_ip2 = _broadcast_state(q_im1, q_i, q_ip1, q_ip2)
    left = (-q_im1 + 5.0 * q_i + 2.0 * q_ip1) / 6.0
    right = (2.0 * q_i + 5.0 * q_ip1 - q_ip2) / 6.0
    left_ok = jnp.all(jnp.isfinite(left[..., POSITIVE_COMPONENTS]), axis=-1) & jnp.all(left[..., POSITIVE_COMPONENTS] > positivity_floor, axis=-1)
    right_ok = jnp.all(jnp.isfinite(right[..., POSITIVE_COMPONENTS]), axis=-1) & jnp.all(right[..., POSITIVE_COMPONENTS] > positivity_floor, axis=-1)
    left_fallback = ~left_ok
    right_fallback = ~right_ok
    left = jnp.where(left_ok[..., None], left, q_i)
    right = jnp.where(right_ok[..., None], right, q_ip1)
    left = left.at[..., POSITIVE_COMPONENTS].set(jnp.maximum(left[..., POSITIVE_COMPONENTS], positivity_floor))
    right = right.at[..., POSITIVE_COMPONENTS].set(jnp.maximum(right[..., POSITIVE_COMPONENTS], positivity_floor))
    metadata = ReconstructionMetadata(
        jnp.stack((left_fallback, right_fallback), axis=-1),
        jnp.where(jnp.stack((left_fallback, right_fallback), axis=-1), 1, 3).astype(jnp.int32),
    )
    return left, right, metadata


# Descriptive aliases used by integration code and diagnostics.
build_curvature_principal_matrix = curvature_principal_matrix
face_linearized_curvature_fluctuations = curvature_face_linearized_fluctuations


__all__ = [
    "ReconstructionMetadata",
    "curvature_principal_matrix",
    "curvature_strict_principal_matrix",
    "curvature_characteristic_absolute_action",
    "curvature_characteristic_absolute_matrix",
    "curvature_characteristic_metric",
    "curvature_face_linearized_fluctuations",
    "reconstruct_first_order_face_states",
    "reconstruct_third_order_face_states",
    "build_curvature_principal_matrix",
    "face_linearized_curvature_fluctuations",
]
