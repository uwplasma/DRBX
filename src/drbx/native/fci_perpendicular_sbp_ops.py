"""JAX building blocks of the nodal SBP perpendicular scheme: block derivatives, eta derivative, traces and lifts.

All functions are pure JAX and jit-safe; the :class:`~drbx.stencils.nodal_plan.NodalPlan` is an ordinary argument and
its ``structure`` supplies the static block/side bookkeeping. State arrays have shape ``(E, P[, ...])`` with any
trailing field axes; an ``(E, P)`` node field broadcasts against them.

Halo convention. ``plan.jac`` has the *owned* planes ``E_own`` on its leading axis. Arrays handed to the eta-stencil
functions carry ``halo`` extra planes on each side (``E_ext = E_own + 2 halo``); results are always on the owned
planes. Plane-local operators (``d1``, ``d2``, traces, lifts, SATs) act on owned planes, obtained with :func:`crop`.
On a single device the halos are a periodic wrap (:func:`extend_periodic`).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from drbx.native.owner_plane_layout import exchange_plane_halo
from drbx.stencils.nodal_plan import DenseBlockArrays, NodalPlan, RingBlockArrays

TWO_PI = 2.0 * np.pi
AXIS_NAME = "z"


# ---------------------------------------------------------------------------
# Halos
# ---------------------------------------------------------------------------
def extend_periodic(x, halo: int):
    """``(E, ...)`` -> ``(E + 2 halo, ...)`` by periodic wrap (single device)."""
    return exchange_plane_halo(x, halo, AXIS_NAME, 1)


def halo_of(plan: NodalPlan, x) -> int:
    """Halo width of an extended array relative to the owned planes of ``plan``."""
    extra = x.shape[0] - plan.jac.shape[0]
    if extra < 0 or extra % 2:
        raise ValueError(f"array has {x.shape[0]} planes, incompatible with {plan.jac.shape[0]} owned planes")
    return extra // 2


def crop(x, k: int):
    """Drop ``k`` planes from both ends of the leading axis."""
    return x if k == 0 else x[k:x.shape[0] - k]


def bc(a, like):
    """Reshape node field ``a`` for broadcasting against ``like`` (trailing field axes)."""
    return jnp.reshape(a, a.shape + (1,) * (jnp.ndim(like) - jnp.ndim(a)))


# ---------------------------------------------------------------------------
# Block-wise in-plane derivatives
# ---------------------------------------------------------------------------
def _blocks(plan: NodalPlan):
    for b, (desc, arr) in enumerate(zip(plan.structure.blocks, plan.blocks)):
        yield b, desc, arr


def _sl(desc):
    return desc[1], desc[1] + desc[2]


def d1(plan: NodalPlan, x):
    """Radial (first logical direction) derivative of ``x (E, P, ...)``."""
    out = []
    for _b, desc, arr in _blocks(plan):
        o0, o1 = _sl(desc)
        xb = x[:, o0:o1]
        if isinstance(arr, RingBlockArrays):
            _, _, _, m, N, _ = desc
            gl = xb.reshape((xb.shape[0], m, N) + xb.shape[2:])
            out.append(jnp.einsum("ab,ebj...->eaj...", arr.Du, gl).reshape(xb.shape))
        else:
            out.append(jnp.einsum("pq,eq...->ep...", arr.D1, xb))
    return jnp.concatenate(out, axis=1)


def d2(plan: NodalPlan, x):
    """Angular (second logical direction) derivative of ``x (E, P, ...)``."""
    out = []
    for _b, desc, arr in _blocks(plan):
        o0, o1 = _sl(desc)
        xb = x[:, o0:o1]
        if isinstance(arr, RingBlockArrays):
            _, _, _, m, N, _ = desc
            gl = xb.reshape((xb.shape[0], m, N) + xb.shape[2:])
            out.append(jnp.einsum("jl,eal...->eaj...", arr.Dth, gl).reshape(xb.shape))
        else:
            out.append(jnp.einsum("pq,eq...->ep...", arr.D2, xb))
    return jnp.concatenate(out, axis=1)


def d_eta(x_ext, deta, halo: int):
    """Fourth-order central eta derivative on the owned planes of a halo-``halo`` extended array (``halo >= 2``)."""
    if halo < 2:
        raise ValueError("the eta derivative needs a halo of at least 2 planes")
    e_own = x_ext.shape[0] - 2 * halo

    def at(offset):
        s = halo + offset
        return x_ext[s:s + e_own]

    return (at(-2) - 8.0 * at(-1) + 8.0 * at(1) - at(2)) / (12.0 * deta)


# ---------------------------------------------------------------------------
# Sides: trace, flux trace, scatter (transpose of the trace), lift
# ---------------------------------------------------------------------------
def _rows(plan: NodalPlan, b: int, side: str):
    for blk, name, rows in plan.structure.side_rows:
        if blk == b and name == side:
            return rows
    raise KeyError((b, side))


def _ring_t(arr: RingBlockArrays, side: str):
    return arr.tL if side == "inner" else arr.tR


def _side_N(plan: NodalPlan, b: int, side: str) -> int:
    desc = plan.structure.blocks[b]
    if desc[0] == "ring":
        return desc[4]
    arr = plan.blocks[b]
    return (arr.inner if side == "inner" else arr.outer).T.shape[0]


def trace(plan: NodalPlan, b: int, side: str, x):
    """Trace of ``x (E, P, ...)`` on side ``side`` of block ``b``: ``(E, N, ...)``."""
    desc, arr = plan.structure.blocks[b], plan.blocks[b]
    o0, o1 = _sl(desc)
    xb = x[:, o0:o1]
    if isinstance(arr, RingBlockArrays):
        _, _, _, m, N, _ = desc
        gl = xb.reshape((xb.shape[0], m, N) + xb.shape[2:])
        t = _ring_t(arr, side)
        return sum(t[r] * gl[:, r] for r in _rows(plan, b, side))
    s = arr.inner if side == "inner" else arr.outer
    return jnp.einsum("jp,ep...->ej...", s.T, xb)


def flux_trace(plan: NodalPlan, b: int, side: str, f1g, f2g):
    """Normal (+u) flux trace per radian of the flux ``(F1 g, F2 g)`` on a side."""
    arr = plan.blocks[b]
    if isinstance(arr, RingBlockArrays):
        return trace(plan, b, side, f1g)
    desc = plan.structure.blocks[b]
    o0, o1 = _sl(desc)
    s = arr.inner if side == "inner" else arr.outer
    return (jnp.einsum("jp,ep...->ej...", s.TF1, f1g[:, o0:o1]) + jnp.einsum("jp,ep...->ej...", s.TF2, f2g[:, o0:o1]))


def scatter(plan: NodalPlan, b: int, side: str, sig):
    """``T^T sig`` placed in a full ``(E, P, ...)`` array (zero outside block ``b``); ``sig`` is ``(E, N, ...)``."""
    desc, arr = plan.structure.blocks[b], plan.blocks[b]
    o0, o1 = _sl(desc)
    if isinstance(arr, RingBlockArrays):
        _, _, _, m, N, _ = desc
        t = _ring_t(arr, side)
        rows = set(_rows(plan, b, side))
        zero = jnp.zeros_like(sig)
        blk = jnp.stack([t[r] * sig if r in rows else zero for r in range(m)], axis=1)
        blk = blk.reshape((sig.shape[0], m * N) + sig.shape[2:])
    else:
        s = arr.inner if side == "inner" else arr.outer
        blk = jnp.einsum("jp,ej...->ep...", s.T, sig)
    pad = [(0, 0), (o0, plan.structure.P - o1)] + [(0, 0)] * (sig.ndim - 2)
    return jnp.pad(blk, pad)


def lift(plan: NodalPlan, b: int, side: str, sig):
    """``Hp^-1 T^T Omega sig`` on the owned planes (``Omega = 2 pi / N``)."""
    omega = TWO_PI / _side_N(plan, b, side)
    out = scatter(plan, b, side, sig)
    return out * omega / bc(plan.Hp, out)


def side_N(plan: NodalPlan, b: int, side: str) -> int:
    return _side_N(plan, b, side)
