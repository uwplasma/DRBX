"""Matrix-free weighted negative adjoints for support-operator pairs.

The helper here deliberately knows nothing about FCI tracing, boundary data,
or a particular support layout.  It turns a supplied *homogeneous linear*
gradient ``G`` into its compatible divergence

``D = -M^-1 G^T W``.

``jax.linear_transpose`` forms the transpose without assembling ``G``.  Any
affine boundary contribution must be split from ``gradient`` before using this
module.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .._host_guards import host_asarray


Array = jnp.ndarray
Gradient = Callable[[Array], Array]
Divergence = Callable[[Array], Array]


def _validated_mass_and_mask(
    name: str,
    mass: Array,
    active: Array | None,
) -> tuple[Array, Array]:
    """Validate one diagonal measure and return JAX mass/mask arrays."""

    mass_value = jnp.asarray(mass)
    if np.dtype(mass_value.dtype).kind != "f":
        raise TypeError(f"{name}_mass must have a real floating-point dtype")
    if active is None:
        active_value = jnp.ones(mass_value.shape, dtype=bool)
    else:
        active_value = jnp.asarray(active)
        if active_value.shape != mass_value.shape:
            raise ValueError(f"{name}_active must have shape {mass_value.shape}")
        if active_value.dtype != jnp.dtype(bool):
            raise TypeError(f"{name}_active must have boolean dtype")
    # Geometry fields can be tracers when an RHS is built below ``jit`` or
    # ``shard_map``.  Shape and dtype remain available there, but attempting
    # to inspect values on the host is illegal.  Retain full eager validation
    # whenever the inputs are concrete.
    try:
        host_mass = host_asarray(mass_value)
        host_active = host_asarray(active_value)
    except jax.errors.TracerArrayConversionError:
        pass
    else:
        active_mass = host_mass[host_active]
        if not np.all(np.isfinite(active_mass)) or np.any(active_mass <= 0.0):
            raise ValueError(
                f"{name}_mass must be finite and positive on active entries"
            )
    return mass_value, active_value


def build_weighted_negative_adjoint(
    gradient: Gradient,
    primal_mass: Array,
    dual_mass: Array,
    *,
    primal_active: Array | None = None,
    dual_active: Array | None = None,
) -> Divergence:
    """Build ``D(q) = -M^-1 G^T W q`` without materializing ``G``.

    ``gradient`` must be a pure homogeneous linear map from the primal array
    shape to the dual array shape.  The mass arrays define those two shapes,
    respectively.  Inactive entries need not have positive masses; they are
    excluded from the weighted pairing and are always returned as zero.

    The returned callable accepts either one dual field with ``dual_mass``'s
    shape or any number of leading batch dimensions followed by that shape.
    It is safe to call from :func:`jax.jit`.
    """

    if not callable(gradient):
        raise TypeError("gradient must be callable")
    primal_mass_value, primal_active_value = _validated_mass_and_mask(
        "primal", primal_mass, primal_active
    )
    dual_mass_value, dual_active_value = _validated_mass_and_mask(
        "dual", dual_mass, dual_active
    )
    if primal_mass_value.dtype != dual_mass_value.dtype:
        raise TypeError("primal_mass and dual_mass must have the same dtype")

    primal_shape = primal_mass_value.shape
    dual_shape = dual_mass_value.shape
    zero_primal = jnp.zeros(primal_shape, dtype=primal_mass_value.dtype)
    dual_template = jnp.asarray(gradient(zero_primal))
    if dual_template.shape != dual_shape:
        raise ValueError(
            "gradient(zero primal) must have the same shape as dual_mass; "
            f"got {dual_template.shape}, expected {dual_shape}"
        )
    if dual_template.dtype != dual_mass_value.dtype:
        raise TypeError(
            "gradient output dtype must match primal_mass and dual_mass dtype"
        )

    # Keeping inactive denominators benign prevents undefined inactive values
    # (such as zero aggregate volume) from participating in the computation.
    safe_primal_mass = jnp.where(primal_active_value, primal_mass_value, 1.0)
    active_dual_mass = jnp.where(dual_active_value, dual_mass_value, 0.0)
    transpose = jax.linear_transpose(gradient, zero_primal)
    dual_ndim = len(dual_shape)

    def apply_one(dual_values: Array) -> Array:
        masked_values = jnp.where(dual_active_value, dual_values, 0.0)
        (adjoint,) = transpose(active_dual_mass * masked_values)
        result = -adjoint / safe_primal_mass
        return jnp.where(primal_active_value, result, 0.0)

    def divergence(dual_values: Array) -> Array:
        """Apply the weighted negative adjoint to one or more dual fields."""

        values = jnp.asarray(dual_values)
        if values.dtype != dual_mass_value.dtype:
            raise TypeError(
                f"dual_values must have dtype {dual_mass_value.dtype}, got {values.dtype}"
            )
        if dual_ndim:
            if values.ndim < dual_ndim or values.shape[-dual_ndim:] != dual_shape:
                raise ValueError(
                    "dual_values must have trailing shape "
                    f"{dual_shape}; got {values.shape}"
                )
            leading_shape = values.shape[:-dual_ndim]
        else:
            leading_shape = values.shape
        if not leading_shape:
            return apply_one(values)
        flattened = values.reshape((-1,) + dual_shape)
        return jax.vmap(apply_one)(flattened).reshape(leading_shape + primal_shape)

    return divergence


def build_support_pair(
    core_divergence: Divergence,
    mass: Array,
    *,
    active: Array,
    core: Array,
) -> tuple[Gradient, Divergence]:
    """Constant-exact weighted support pair ``(G, D)`` of a core-row divergence ``D0``.

    ``D0`` is a homogeneous linear mapped flux divergence restricted to the
    ``core`` rows.  Its weighted negative adjoint is ``H``; the diagonal
    correction ``G(f) = H(f) - H(1) f`` makes constants exact, and a second
    weighted negative adjoint closes ``D = -M^-1 G^T M``.  Then ``D`` is the
    exact adjoint of ``G`` (``sum M D(f) = 0``) and neither transpose output
    is target-masked: core rows may scatter onto excluded cells.
    """

    adjoint = build_weighted_negative_adjoint(
        core_divergence, mass, mass, primal_active=active, dual_active=core
    )
    constant = adjoint(jnp.ones(mass.shape, dtype=mass.dtype))

    def gradient(values: Array) -> Array:
        values = jnp.where(active, jnp.asarray(values, dtype=mass.dtype), 0.0)
        return adjoint(values) - constant * values

    divergence = build_weighted_negative_adjoint(
        gradient, mass, mass, primal_active=active, dual_active=active
    )
    return gradient, divergence


def assemble_sparse_rows(operator: Callable[[Array], Array], shape: tuple[int, ...]):
    """Padded sparse rows ``(cols, vals)`` of a fixed linear map on ``shape``, by coloured probing.

    Cells a multiple of ``sep`` apart (``sep`` the smallest divisor of the extent not below the
    request, requests ``(6, 6, 3)`` then wider) share a probe; two probes per colour, ``1`` and
    ``1 + flat index`` on the colour, give a row's value and, from the ratio, its column.  Probes
    run in chunks, so memory stays at a few chunks of the grid.  Returns ``None`` unless a random
    vector is then reproduced to round-off and a row has at most 64 entries, so the caller keeps
    the matrix-free operator.
    """

    n = int(np.prod(shape))
    grid = np.indices(shape).reshape(len(shape), n)
    probe = jax.jit(lambda x: jax.lax.map(lambda v: operator(v.reshape(shape)).reshape(-1), x))
    x = np.random.default_rng(0).normal(size=shape)
    want = np.asarray(jax.jit(operator)(jnp.asarray(x))).reshape(-1)
    tag = np.arange(n) + 1.0
    for request in ((6, 6, 3), (10, 10, 5), (18, 18, 9)):
        steps = [min(d for d in range(min(b, m), m + 1) if m % d == 0) for m, b in zip(shape, request[-len(shape):])]
        colour = np.ravel_multi_index(tuple(g % b for g, b in zip(grid, steps)), steps)
        found, scale = [], 0.0
        for start in range(0, int(np.prod(steps)), 128):
            onehot = (colour[None, :] == np.arange(start, min(start + 128, int(np.prod(steps))))[:, None]) * 1.0
            unit, tagged = (np.asarray(probe(jnp.asarray(v))) for v in (onehot, onehot * tag))
            scale = max(scale, np.abs(unit).max())
            row = np.nonzero(np.abs(unit) > 1.0e-14 * np.abs(unit).max())  # (colour in chunk, row)
            found.append((row[1], unit[row], tagged[row]))
        row, unit, tagged = (np.concatenate(a) for a in zip(*found))
        biggest = np.zeros(n)
        np.maximum.at(biggest, row, np.abs(unit))
        keep = (np.abs(unit) > 1.0e-13 * biggest[row]) & (np.abs(unit) > 1.0e-14 * scale)  # drops round-off
        row, unit, ratio = row[keep], unit[keep], tagged[keep] / unit[keep]
        column = np.rint(ratio) - 1.0
        if np.any((np.abs(ratio - np.rint(ratio)) > 1.0e-9 * np.maximum(ratio, 1.0)) | (ratio < 0.5)):
            continue  # two probed cells met in one row: separate the colours further
        count = np.bincount(row, minlength=n)
        width = int(count.max())
        if width > 64:  # not a sparse map
            return None
        order = np.argsort(row, kind="stable")
        row, unit, column = row[order], unit[order], column[order]
        slot = np.arange(row.size) - (np.cumsum(count) - count)[row]
        cols, vals = np.zeros((n, width), np.int32), np.zeros((n, width))
        cols[row, slot], vals[row, slot] = column, unit
        if np.max(np.abs((vals * x.reshape(-1)[cols]).sum(-1) - want)) <= 1.0e-12 * max(np.max(np.abs(want)), 1.0e-300):
            return cols, vals
    return None


def adjoint_sparse_rows(cols, vals, primal_mass, dual_mass, primal_active, dual_active):
    """Rows of ``G = -M^-1 D^T W`` from the rows of ``D`` (the weighted negative adjoint of
    :func:`build_weighted_negative_adjoint`), exact by construction."""

    n, width = cols.shape
    row_scale = -np.asarray(primal_active, float).reshape(-1) / np.where(primal_active, primal_mass, 1.0).reshape(-1)
    col_scale = (np.asarray(dual_mass) * np.asarray(dual_active)).reshape(-1)
    src, dst, val = np.repeat(np.arange(n), width), cols.reshape(-1), vals.reshape(-1)
    keep = val != 0.0
    src, dst, val = src[keep], dst[keep], val[keep]
    order = np.argsort(dst, kind="stable")
    src, dst, val = src[order], dst[order], val[order]
    count = np.bincount(dst, minlength=n)
    slot = np.arange(dst.size) - (np.cumsum(count) - count)[dst]
    out_cols = np.zeros((n, max(int(count.max()), 1)), np.int32)
    out_vals = np.zeros(out_cols.shape)
    out_cols[dst, slot], out_vals[dst, slot] = src, row_scale[dst] * val * col_scale[src]
    return out_cols, out_vals


def sparse_rows_apply(cols: Array, vals: Array, x: Array) -> Array:
    """``y = A x`` for the padded rows of :func:`assemble_sparse_rows` (one gather and a row sum)."""

    return (vals * x.reshape(-1, cols.shape[0])[:, cols]).sum(-1).reshape(x.shape)


__all__ = [
    "Gradient",
    "Divergence",
    "adjoint_sparse_rows",
    "assemble_sparse_rows",
    "build_support_pair",
    "build_weighted_negative_adjoint",
    "sparse_rows_apply",
]
