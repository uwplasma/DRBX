"""Centered direct midpoint bracket (P05) as pure JAX kernels.

This is the package, jit-able counterpart of the frozen host reference
``scripts/p05_direct_midpoint_global/direct_operator.py`` (``point_bracket``,
``direct_pair_actions``, ``project_raw_to_owners``). It carries only the
*centered* part of the qualified P05 operator: the strong bracket
``-((h x grad a) . grad b) / |J|`` evaluated at raw midpoint points from
already-reconstructed gradients, and the raw-volume projection onto owners.
The live q3 upwind jump, the gradient reconstruction and the choice of
generator/transported reconstruction per pair belong to the operator layer
built on top of these kernels; nothing here touches geometry, rows or
boundary data.

Conventions (all float64):

- ``h`` ``(P, 3)`` and ``jacobian`` ``(P,)`` are the raw midpoint ``h`` vector
  and Jacobian; the kernel uses ``|jacobian|`` like the host.
- ``gradients`` ``(P, 3, F)`` stacks the gradients of ``F`` fields (or roles).
- ``pairs`` is a *static* sequence of ``(a, b)`` field indices; the action of
  pair ``(a, b)`` is ``bracket(grad_a, grad_b)``. Pairs are evaluated in one
  gather-based vectorized pass (no Python loop over pairs inside the kernel).

The antisymmetry defect ``max |ab + ba|`` is a diagnostic; it is returned
under ``stop_gradient`` so it never contributes to derivatives.

Each public kernel validates only static shapes and then calls one jitted body,
so an eager call and a call inside an outer ``jax.jit`` run the same compiled
computation and agree bitwise (the ``apply_source_rows`` pattern) when the arrays
enter the outer ``jit`` as arguments; arrays baked into an outer jit as constants
(e.g. a closed-over ``owner_volume``) let XLA rewrite the division and are only
roundoff-equal. Data-dependent
host checks (positive finite Jacobian, finite actions, valid owner indices) are
not possible under ``jit``; use :func:`validate_bracket_inputs` on the host.
Out-of-range owner indices are silently dropped by JAX scatters, so validate
them on the host too.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["point_bracket", "pair_actions", "project_raw_to_owners", "validate_bracket_inputs"]


def _f64(x):
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the midpoint bracket requires jax_enable_x64=True (float64)")
    return jnp.asarray(x, dtype=jnp.float64)


def _static_pairs(pairs, nfields):
    out = tuple((int(a), int(b)) for a, b in pairs)
    for a, b in out:
        if not (0 <= a < nfields and 0 <= b < nfields):
            raise ValueError("pair index is outside the field axis of the gradients")
    return out


def _bracket(h, jacobian, ga, gb):
    """``-((h x ga) . gb) / |J|`` over the last (component) axis; no validation."""
    return -jnp.einsum("...i,...i->...", jnp.cross(h, ga), gb) / jnp.abs(jacobian)


@jax.jit
def _point_bracket(h, jacobian, ga, gb):
    return _bracket(h, jacobian, ga, gb)


def point_bracket(h, jacobian, grad_a, grad_b):
    """Strong bracket at points: ``-((h x grad_a) . grad_b) / |jacobian|``.

    Shapes as the host: ``grad_a``/``grad_b``/``h`` ``(..., 3)`` and
    ``jacobian`` ``(...,)``. Only static shapes are checked.
    """
    h, jacobian, grad_a, grad_b = _f64(h), _f64(jacobian), _f64(grad_a), _f64(grad_b)
    if grad_a.shape != grad_b.shape or grad_a.shape[-1:] != (3,):
        raise ValueError("both argument gradients must have matching (..., 3) shape")
    if h.shape != grad_a.shape or jacobian.shape != grad_a.shape[:-1]:
        raise ValueError("h/J shapes must match the point-gradient shape")
    return _point_bracket(h, jacobian, grad_a, grad_b)


@partial(jax.jit, static_argnames=("pairs",))
def _pair_actions(h, jacobian, gradients, *, pairs):
    if not pairs:
        return (jnp.zeros((gradients.shape[0], 0), dtype=gradients.dtype),
                jnp.zeros((), dtype=gradients.dtype))
    ia = np.array([a for a, _ in pairs], dtype=np.int32)
    ib = np.array([b for _, b in pairs], dtype=np.int32)
    ga = jnp.moveaxis(gradients[:, :, ia], 1, -1)            # (P, npairs, 3)
    gb = jnp.moveaxis(gradients[:, :, ib], 1, -1)
    hh = h[:, None, :]
    jj = jacobian[:, None]
    ab = _bracket(hh, jj, ga, gb)                            # (P, npairs)
    ba = _bracket(hh, jj, gb, ga)
    defect = jnp.max(jnp.abs(ab + ba), initial=0.0)
    return ab, jax.lax.stop_gradient(defect)


def pair_actions(h, jacobian, gradients, pairs: Sequence[tuple[int, int]]):
    """All oriented pair actions and the maximum swapped-argument defect.

    ``gradients`` is ``(P, 3, F)``, ``pairs`` a static sequence of ``(a, b)``.
    Returns ``(actions (P, len(pairs)), antisymmetry)`` where ``antisymmetry``
    is the scalar ``max |bracket(a, b) + bracket(b, a)|`` (0 for no pairs),
    detached from differentiation.
    """
    h, jacobian, gradients = _f64(h), _f64(jacobian), _f64(gradients)
    if gradients.ndim != 3 or gradients.shape[1] != 3:
        raise ValueError("gradients must have shape (points, 3, fields)")
    if h.shape != (gradients.shape[0], 3) or jacobian.shape != (gradients.shape[0],):
        raise ValueError("h/J shapes must match the point-gradient shape")
    pairs = _static_pairs(pairs, gradients.shape[2])
    return _pair_actions(h, jacobian, gradients, pairs=pairs)


@jax.jit
def _project(action, volume, owner, owner_volume):
    numerator = jnp.zeros((owner_volume.shape[0], action.shape[1]), dtype=action.dtype)
    numerator = numerator.at[owner].add(volume[:, None] * action)
    return numerator / owner_volume[:, None]


def project_raw_to_owners(raw_action, raw_volume, raw_owner, owner_volume, owner_count=None):
    """Volume-weighted projection of raw actions onto owners.

    ``owner_action[o] = sum_{raw: owner(raw)=o} volume * action / owner_volume[o]``
    for ``raw_action`` ``(R, K)``, ``raw_volume`` ``(R,)``, ``raw_owner`` ``(R,)``
    integer and ``owner_volume`` ``(O,)``. Scatter uses ``.at[].add``; owner
    indices must lie in ``[0, O)`` (checked by :func:`validate_bracket_inputs`,
    not here).
    """
    action, volume, denominator = _f64(raw_action), _f64(raw_volume), _f64(owner_volume)
    owner = jnp.asarray(raw_owner)
    if not jnp.issubdtype(owner.dtype, jnp.integer):
        raise ValueError("raw_owner must be an integer array")
    if action.ndim != 2 or volume.shape != (action.shape[0],) or owner.shape != (action.shape[0],):
        raise ValueError("raw arrays have incompatible dimensions")
    count = denominator.shape[0] if owner_count is None else int(owner_count)
    if denominator.shape != (count,):
        raise ValueError("owner volumes must match owner_count")
    return _project(action, volume, owner, denominator)


def validate_bracket_inputs(h=None, jacobian=None, gradients=None, *, actions=None,
                            raw_volume=None, raw_owner=None, owner_volume=None):
    """Host-side (NumPy) data checks mirroring the reference's ``ValueError``/
    ``FloatingPointError`` conditions; call outside ``jit``.

    Any subset of the groups may be given: ``jacobian`` must be finite and
    nonzero (the host requires ``|J| > 0``); ``actions`` (and ``gradients``,
    ``h``) must be finite; ``raw_volume`` finite; ``raw_owner`` in
    ``[0, len(owner_volume))``; ``owner_volume`` positive and finite.
    """
    if jacobian is not None:
        j = np.abs(np.asarray(jacobian, dtype=np.float64))
        if np.any(j <= 0.0) or not np.isfinite(j).all():
            raise ValueError("Jacobian must be finite and nonzero")
    for name, value in (("h", h), ("gradients", gradients), ("actions", actions), ("raw_volume", raw_volume)):
        if value is not None and not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise FloatingPointError(f"nonfinite {name}")
    if owner_volume is not None:
        ov = np.asarray(owner_volume, dtype=np.float64)
        if np.any(ov <= 0.0) or not np.isfinite(ov).all():
            raise ValueError("owner volumes must be positive and finite")
        if raw_owner is not None:
            owner = np.asarray(raw_owner, dtype=np.int64)
            if np.any(owner < 0) or np.any(owner >= len(ov)):
                raise ValueError("raw owner index is outside the owner array")
    elif raw_owner is not None and np.any(np.asarray(raw_owner) < 0):
        raise ValueError("raw owner index must be nonnegative")
