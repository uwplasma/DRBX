from __future__ import annotations

import jax
import jax.numpy as jnp


# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ...native.fci_helpers import (  # noqa: F401
    _as_float64_array,
    _as_face_flux_array,
    _validate_axis,
    local_side_plane_shape,
    _as_local_wall_array,
    _as_local_wall_int_array,
    _as_local_wall_bool_array,
    _as_local_wall_stencil_index_array,
    _as_local_wall_stencil_weight_array,
)
from ...native.fci_helpers import _local_side_mask, local_physical_side_active  # noqa: F401


def small_batched_eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigen-decomposition of batches of real 4x4 matrices with a zero last column.

    This is the shape of every curvature principal matrix: the vorticity
    column vanishes, so ``A = [[M, 0], [r, 0]]`` has the eigenvalue 0 (vector
    ``e_3``) and the three eigenvalues of ``M``, which are the roots of a cubic
    (trigonometric form, one Newton step) with eigenvectors ``(u, r.u / lambda)``
    where ``u`` is the largest cross product of two rows of ``M - lambda I``.
    Columns are unit vectors, as LAPACK returns them.  Everything is plain
    array arithmetic, so it runs fused on any backend (a host ``geev``
    callback cost 0.7 s per CPU step and a device round trip per call on a GPU).

    The input must already be stopped-gradient.  Returns complex eigenvalues
    and real eigenvectors.  When the roots of ``M`` are not real (the cubic's
    discriminant is negative) or the residual ``|A V - V L|`` of the result
    exceeds 1e-8 of the largest entry (non-finite input, a repeated root with
    a two-dimensional eigenspace), the eigenvalues get a unit imaginary part
    and the vectors are the identity; callers reject that case through their
    spectrum-validity test exactly as for a complex LAPACK spectrum.  The
    matrix is scaled by its largest entry first, so any magnitude works; the
    zero matrix gives the identity.
    """

    a = jnp.asarray(matrix, dtype=jnp.float64)
    scale = jnp.max(jnp.abs(a), axis=(-2, -1), keepdims=True)  # eigenvectors do not depend on it
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    a = a / scale
    original = a
    m, r = a[..., :3, :3], a[..., 3, :3]
    eye = jnp.eye(3, dtype=a.dtype)
    shift = (m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]) / 3.0
    b = m - shift[..., None, None] * eye
    p_cubic = -0.5 * jnp.einsum("...ij,...ji->...", b, b)  # t^3 + P t + Q = 0, t = lambda - shift
    q_cubic = -(
        b[..., 0, 0] * (b[..., 1, 1] * b[..., 2, 2] - b[..., 1, 2] * b[..., 2, 1])
        - b[..., 0, 1] * (b[..., 1, 0] * b[..., 2, 2] - b[..., 1, 2] * b[..., 2, 0])
        + b[..., 0, 2] * (b[..., 1, 0] * b[..., 2, 1] - b[..., 1, 1] * b[..., 2, 0])
    )
    radius = jnp.sqrt(jnp.maximum(-p_cubic / 3.0, 0.0))
    cosine = q_cubic / (2.0 * jnp.where(radius > 0.0, radius, 1.0) ** 3)
    complex_roots = ~((radius > 0.0) | (q_cubic == 0.0)) | (jnp.abs(cosine) > 1.0 + 1.0e-10)
    angle = jnp.arccos(jnp.clip(-cosine, -1.0, 1.0))[..., None] / 3.0
    t = 2.0 * radius[..., None] * jnp.cos(angle - jnp.asarray((0.0, 2.0, 4.0)) * (jnp.pi / 3.0))
    # One Newton step on the characteristic polynomial removes the trigonometric rounding.
    slope = 3.0 * t**2 + p_cubic[..., None]
    t = t - jnp.where(slope != 0.0, (t**3 + p_cubic[..., None] * t + q_cubic[..., None]) / jnp.where(slope != 0.0, slope, 1.0), 0.0)
    lam = shift[..., None] + t

    rows = (m[..., None, :, :] - lam[..., :, None, None] * eye)  # (..., k, 3, 3)
    crosses = jnp.stack(
        [jnp.cross(rows[..., 0, :], rows[..., 1, :]), jnp.cross(rows[..., 1, :], rows[..., 2, :]),
         jnp.cross(rows[..., 2, :], rows[..., 0, :])]
    )
    best = jnp.argmax(jnp.linalg.norm(crosses, axis=-1), axis=0)
    pick = best[None] == jnp.arange(3).reshape((3,) + (1,) * best.ndim)
    u = jnp.sum(jnp.where(pick[..., None], crosses, 0.0), axis=0)
    norm_u = jnp.linalg.norm(u, axis=-1, keepdims=True)
    # M - lambda I of rank <= 1 (zero or scalar M): the axes are eigenvectors.
    u = jnp.where(norm_u > 1.0e-14, u / jnp.where(norm_u > 1.0e-14, norm_u, 1.0), eye)
    tail = jnp.sum(r[..., None, :] * u, axis=-1) / jnp.where(lam != 0.0, lam, 1.0)
    v = jnp.concatenate([u, tail[..., None]], axis=-1)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    e3 = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 0.0, 1.0)), v[..., :1, :].shape)
    vectors = jnp.swapaxes(jnp.concatenate([v, e3], axis=-2), -1, -2)
    values = jnp.concatenate([lam, jnp.zeros_like(lam[..., :1])], axis=-1)
    # Self-check: a degenerate spectrum the formulas cannot resolve is reported as non-real.
    residual = jnp.max(jnp.abs(original @ vectors - vectors * values[..., None, :]), axis=(-2, -1))
    invalid = complex_roots | ~(residual <= 1.0e-8)
    vectors = jnp.where(invalid[..., None, None], jnp.eye(4), vectors)
    return (values * scale[..., 0]) + 1.0j * invalid[..., None], vectors
