"""JAX kernel of tensor-factored point rows (P08 step 2, task D2).

The unconditioned ``singleton``, ``ringwise`` and ``centered_radial`` point rows are, per
target, sums of products of shared 1-D factors (design section 6,
``drbx.stencils.tensor_rows``). This module applies them to owner fields **without ever
materializing the dense weights**: for each target the 4 (radial layer) x 4 (eta plane) x 7
(theta slot) donor block is gathered and contracted with the factors.

Runtime layout (all arrays, registered pytrees with static ints/strings):

- ``TensorTables``: grid-global 1-D factor tables (deduplicated by exact bit pattern across
  chunks) plus ``raw_to_owner`` (``n**3`` int32), the compact lookup singleton /
  centered_radial owners are derived from;
- ``TensorRowBatch``: the targets of one (family, has_gradient, paired) group, a few small integers
  per target (table rows, and for ringwise a 4x4 block of ring-table rows; twice for a paired target)
  and the output slots. Owner ids are never stored per target:

  - singleton / centered_radial: ``raw = (rid[l]*n + (theta[j] + shift[l]) % n)*n + eta[e]``
    (``rid = l`` or ``-l-1`` and ``shift = n//2`` for a negative layer ``l``), owner
    ``raw_to_owner[raw]``;
  - ringwise: the ring entry's 7 owners.

Contraction (sum factorization, per target and field, ``G[l, e, j]`` the donor block)::

    A_w[l, e]  = sum_j w[j] G[l, e, j]           w = tv (value, d/dr, d/deta), td (d/dtheta)
                                                 [ringwise: w depends on (l, e): vv / dd]
    B_f[l]     = sum_e f[e] A_w[l, e]            f = ev (value, d/dr, d/dtheta), ed (d/deta)
    out        = sum_l c[l] B_f[l]               c = L (value, d/dtheta, d/deta), D / dr_divisor (d/dr)

so value, d/dr, d/dtheta and d/deta share two theta contractions (tv, td) and their eta
contractions. ``dr_divisor`` is ``h = 1/n`` for ``centered_radial`` (``boundary_map`` divides its
``D`` by ``h``) and ``1`` otherwise.

A *paired* target is the row ``1/2 (A + B)`` of two such factorizations (a ``cell_stencil="symmetric"`` cell row:
the biased row A and its mirror B, ``drbx.stencils.tensor_rows``). Its batch carries, beside the index arrays of A, a
:class:`TensorMirror` with those of B; both parts are contracted exactly as above (two 4 x 4 gathers instead of one,
from the same per-apply tables) and the target's output is ``0.5 * (A + B)``. A paired batch is of one family: both
parts of a pair are singleton or both ringwise.

The first stage ``A_w`` does not depend on the target beyond a small key, so it is computed **once
per apply** (``prepare_fields``) and each target only gathers its 4 x 4 results:

- ringwise: the key is the ring entry (its 7 owners and ``vv``/``dd`` are the entry's), giving
  ``(NG, F)`` sums; a target gathers 16 entry rows;
- singleton / centered_radial: the key is (theta-table row ``a``, radial layer ``r``, half-turn shift
  ``s`` of a layer below the axis) -- the "combos" the plan's targets actually use, listed in
  ``TensorTables.combo_key`` -- and every eta plane ``k``:
  ``Tv[c, k] = sum_j tv[a, j] fields[raw_to_owner[(r, (theta_index[a, j] + s n/2) % n, k)]]`` and
  ``Td`` with ``td``. A target gathers ``Tv[combo(l), eta_index[e]]`` for its 4 x 4 (l, e): 16 rows
  instead of the 112 donors (the theta tables have a few hundred rows, so the combos are far fewer
  than the targets).

Each per-apply table is built in blocks of combos / entries (``lax.map``) and targets are processed
in blocks (``lax.map``), so the transients stay bounded: ``(targets, 16, F)`` per block, ``(combos, 7,
n, F)`` per combo block, plus the two persistent ``(NC, n, F)`` tables. A call whose whole gathered block
stays under :data:`SINGLE_BLOCK_BYTES` is applied in one vectorized piece instead (no loop; bitwise the same
values, fewer passes); ``SINGLE_BLOCK_BYTES = 0`` always blocks with the sizes below. The result agrees with the
expanded CSR rows to rounding (the summation order differs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

#: gathered elements (targets x 16 x fields) per block of targets (singleton, centered_radial)
ANGULAR_BLOCK_ELEMENTS = 1 << 16
#: gathered ring-entry elements (targets x 16 x fields) per block of targets (ringwise; measured best)
RING_BLOCK_ELEMENTS = 1 << 16
#: gathered donor elements (combos x 7 x n x fields) per block of combos of the theta tables
COMBO_BLOCK_ELEMENTS = 1 << 20
DONOR_BLOCK = 112
#: a whole call (all targets of a batch, or all combos of the theta tables) is applied without ``lax.map`` when its
#: gathered block (elements x 8 bytes: ``targets x 16 x F`` resp. ``combos x 7 x n x F``) is at most this many bytes
SINGLE_BLOCK_BYTES = 512 << 20


@dataclass(frozen=True)
class TensorTables:
    """Grid-global 1-D factor tables shared by every tensor batch of a plan."""

    n: int
    theta_index: np.ndarray         # (NT, 7) int32   theta plane ids of the 7 slots
    theta_value: np.ndarray         # (NT, 7) float64 tv
    theta_derivative: np.ndarray    # (NT, 7) float64 td
    eta_index: np.ndarray           # (NE, 4) int32   eta plane ids
    eta_value: np.ndarray           # (NE, 4) float64 ev
    eta_derivative: np.ndarray      # (NE, 4) float64 ed
    radial: np.ndarray              # (NR, 2, 4) float64  [L, D] over the layers, D undivided by h
    ring_owner: np.ndarray          # (NG, 7) int32   owner ids of a ring entry
    ring_value: np.ndarray          # (NG, 7) float64 vv
    ring_derivative: np.ndarray     # (NG, 7) float64 dd
    layers: np.ndarray              # (NL, 4) int16   signed radial layers (negative: wrapped across the axis)
    raw_to_owner: np.ndarray        # (n**3,) int32   compact owner of a raw cell (C order, eta fastest)
    combo_key: np.ndarray           # (NC,) int32     sorted keys (a*2 + s)*n + r of the (theta row, shift, radial) combos used
    combo_lookup: np.ndarray        # (NT*2*n,) int32 row of a key in ``combo_key`` (-1: unused)


jax.tree_util.register_dataclass(
    TensorTables,
    data_fields=["theta_index", "theta_value", "theta_derivative", "eta_index", "eta_value",
                 "eta_derivative", "radial", "ring_owner", "ring_value", "ring_derivative", "layers",
                 "raw_to_owner", "combo_key", "combo_lookup"],
    meta_fields=["n"])


@dataclass(frozen=True)
class TensorMirror:
    """The mirror part B of a paired batch: the index arrays of the second factorization of every target.

    Same layout (and index widths) as the batch's own index arrays, which are part A: ``t_ring`` for a ringwise
    batch, ``layer_id`` and ``t_theta`` for the others; the tables are the plan's, shared with A."""

    t_eta: np.ndarray               # (T,) eta-table row
    t_radial: np.ndarray            # (T,) radial-table row
    layer_id: np.ndarray | None     # (T,) layers-table row (not ringwise)
    t_theta: np.ndarray | None      # (T,) theta-table row (not ringwise)
    t_ring: np.ndarray | None       # (T, 4, 4) ring-table rows (ringwise)


jax.tree_util.register_dataclass(
    TensorMirror, data_fields=["t_eta", "t_radial", "layer_id", "t_theta", "t_ring"], meta_fields=[])


@dataclass(frozen=True)
class TensorRowBatch:
    """Targets of one (family, has_gradient, paired) group; per-target arrays have ``T`` rows.

    ``ringwise`` batches carry ``t_ring`` (T, 4, 4) and no ``layer_id``/``t_theta``; the other
    families carry ``layer_id`` and ``t_theta`` and no ``t_ring``. Index arrays are integers of
    any width (indices into the tables); ``value_slots``/``gradient_slots`` are int32 output rows.
    A paired batch (``mirror`` is not ``None``) holds symmetric cell rows ``1/2 (A + B)``: the index arrays above
    are part A, ``mirror`` those of part B (the module docstring).
    """

    family: str                     # static: "singleton" | "ringwise" | "centered_radial"
    dr_divisor: float               # static: divisor of D (h = 1/n for centered_radial, else 1.0)
    value_slots: np.ndarray         # (T,) int32
    gradient_slots: np.ndarray | None   # (T,) int32, gradient batches only
    t_eta: np.ndarray               # (T,) eta-table row
    t_radial: np.ndarray            # (T,) radial-table row
    layer_id: np.ndarray | None     # (T,) layers-table row (not ringwise)
    t_theta: np.ndarray | None      # (T,) theta-table row (not ringwise)
    t_ring: np.ndarray | None       # (T, 4, 4) ring-table rows (ringwise)
    mirror: TensorMirror | None = None      # part B of every target (paired batches only)

    def index_arrays(self) -> tuple:
        """The per-target index arrays that the tables are read with: A's, then B's of a paired batch (each
        ``(t_eta, t_radial, t_ring)`` for a ringwise batch, ``(t_eta, t_radial, layer_id, t_theta)`` otherwise)."""
        def part(x):
            return (x.t_eta, x.t_radial) + ((x.t_ring,) if self.t_ring is not None else (x.layer_id, x.t_theta))
        return part(self) + (() if self.mirror is None else part(self.mirror))


jax.tree_util.register_dataclass(
    TensorRowBatch,
    data_fields=["value_slots", "gradient_slots", "t_eta", "t_radial", "layer_id", "t_theta", "t_ring", "mirror"],
    meta_fields=["family", "dr_divisor"])


def tensor_nbytes(tables, batches) -> int:
    """Bytes of the array leaves of ``tables`` and ``batches``."""
    return sum(int(np.asarray(a).nbytes) for a in jax.tree_util.tree_leaves((tables, batches)))


def _weighted_sum(block, weights, axis):
    """``sum_axis block * weights`` with ``weights`` broadcast against ``block`` (field axis last)."""
    return jnp.sum(block * weights[..., None], axis=axis)


class TensorFieldContext(NamedTuple):
    """Field-dependent, target-independent intermediates shared by all batches of one apply.

    - ``theta_value`` / ``theta_derivative`` ``(NC, n, F)``: the theta contractions ``Tv`` / ``Td`` of
      every (theta row, shift, radial layer) combo and eta plane used by the singleton /
      centered_radial batches (``tv`` resp. ``td`` against the 7 raw cells of the ring); ``None``
      when no such batch needs them;
    - ``ring_value`` / ``ring_derivative`` ``(NG, F)``: the theta contractions of every ring entry,
      ``sum_j vv[j] fields[owner[j]]`` and the same with ``dd`` (a ring entry's 7 owners and factors
      do not depend on the target, so this is done once per entry instead of once per (target, layer,
      plane)). ``None`` when no batch needs them.
    """

    theta_value: object
    theta_derivative: object
    ring_value: object
    ring_derivative: object


def _blocked(function, arrays, size, row_bytes=None):
    """``function`` over ``arrays`` (leading axis ``total``) in blocks of ``size`` rows (``lax.map``), rows in order.

    ``function`` maps a block of rows to a tuple of arrays (or ``None`` entries) with one row per input row.
    One block is applied directly (also whenever ``total * row_bytes <= SINGLE_BLOCK_BYTES``); otherwise the last
    block is padded with repeated rows (dropped again).
    """
    total = arrays[0].shape[0]
    if row_bytes is not None and total * row_bytes <= SINGLE_BLOCK_BYTES:
        size = total
    size = int(max(1, min(total, size)))
    if size >= total:
        return function(arrays)
    blocks = -(-total // size)
    pad = blocks * size - total
    padded = tuple(jnp.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
                   .reshape((blocks, size) + a.shape[1:]) for a in arrays)
    out = jax.lax.map(function, padded)
    return tuple(None if o is None else o.reshape((blocks * size,) + o.shape[2:])[:total] for o in out)


def _theta_tables(tables: TensorTables, fields, gradients):
    """``(Tv, Td)`` ``(NC, n, F)`` of the combos of ``tables.combo_key`` (``Td`` is ``None`` unless ``gradients``)."""
    n = tables.n
    raw = tables.raw_to_owner.reshape(n, n, n)
    key = tables.combo_key.astype(jnp.int32)
    nf = fields.shape[1]

    def block(chunk):
        (k,) = chunk
        row, shift, radial = k // (2 * n), (k // n) % 2, k % n
        theta = (tables.theta_index[row] + (shift * (n // 2))[:, None]) % n                # (b, 7)
        data = fields[raw[radial[:, None], theta]]                                          # (b, 7, n, F)
        value = jnp.sum(data * tables.theta_value[row][:, :, None, None], axis=1)
        derivative = (jnp.sum(data * tables.theta_derivative[row][:, :, None, None], axis=1)
                      if gradients else None)
        return value, derivative

    return _blocked(block, (key,), COMBO_BLOCK_ELEMENTS // (7 * n * max(nf, 1)), 8 * 7 * n * max(nf, 1))


def prepare_fields(tables: TensorTables, batches, fields, *, values: bool = True,
                   gradients: bool = True) -> TensorFieldContext:
    """The :class:`TensorFieldContext` for ``batches`` (traceable)."""
    tables = jax.tree_util.tree_map(jnp.asarray, tables)
    angular = [b for b in batches if b.t_ring is None]
    ring = [b for b in batches if b.t_ring is not None]
    theta_value = theta_derivative = None
    if angular:
        want = gradients and any(b.gradient_slots is not None for b in angular)
        theta_value, theta_derivative = _theta_tables(tables, fields, want)
    ring_value = ring_derivative = None
    if ring:
        data = fields[tables.ring_owner]                                # (NG, 7, F)
        ring_value = _weighted_sum(data, tables.ring_value, 1)
        if gradients and any(b.gradient_slots is not None for b in ring):
            ring_derivative = _weighted_sum(data, tables.ring_derivative, 1)
    return TensorFieldContext(theta_value, theta_derivative, ring_value, ring_derivative)


def _block_apply(tables: TensorTables, static, context, arrays, *, values, gradients):
    """Value ``(m, F)`` and gradient ``(m, 3, F)`` (either may be ``None``) of one block of targets."""
    ringwise, divisor = static
    n = tables.n
    t_eta, t_radial, first = arrays[0], arrays[1], arrays[2:]
    eta = t_eta.astype(jnp.int32)
    if ringwise:
        entry = first[0].astype(jnp.int32)                              # (m, 4, 4)
        a_value = context.ring_value[entry]                             # (m, l, e, F), per-entry theta sums
        a_theta = context.ring_derivative[entry] if gradients else None
    else:
        layer = tables.layers[first[0].astype(jnp.int32)].astype(jnp.int32)     # (m, 4)
        theta_row = first[1].astype(jnp.int32)
        rid = jnp.where(layer < 0, -layer - 1, layer)
        combo = tables.combo_lookup[(theta_row[:, None] * 2 + (layer < 0)) * n + rid]   # (m, 4)
        plane = tables.eta_index[eta][:, None, :]                       # (m, 1, 4)
        a_value = context.theta_value[combo[:, :, None], plane]         # (m, l, e, F), per-combo theta sums
        a_theta = context.theta_derivative[combo[:, :, None], plane] if gradients else None
    radial = tables.radial[t_radial.astype(jnp.int32)]                  # (m, 2, 4)
    ev = tables.eta_value[eta]
    length = radial[:, 0]
    b_value = _weighted_sum(a_value, ev[:, None, :], 2)                 # (m, l, F)
    out_value = out_gradient = None
    if values:
        out_value = _weighted_sum(b_value, length, 1)
    if gradients:
        derivative = radial[:, 1] if divisor == 1.0 else radial[:, 1] / divisor
        b_theta = _weighted_sum(a_theta, ev[:, None, :], 2)
        b_eta = _weighted_sum(a_value, tables.eta_derivative[eta][:, None, :], 2)
        out_gradient = jnp.stack((_weighted_sum(b_value, derivative, 1),
                                  _weighted_sum(b_theta, length, 1),
                                  _weighted_sum(b_eta, length, 1)), axis=1)     # (m, 3, F)
    return out_value, out_gradient


def _block_apply_pair(tables: TensorTables, static, context, arrays, *, values, gradients):
    """:func:`_block_apply` of a block of paired targets: ``0.5 * (A + B)``. ``arrays`` are the index arrays of A
    followed by those of B (the same layout twice); both parts are evaluated in one pass over the stacked block."""
    half = len(arrays) // 2
    both = tuple(jnp.concatenate([a, b]) for a, b in zip(arrays[:half], arrays[half:]))
    out_value, out_gradient = _block_apply(tables, static, context, both, values=values, gradients=gradients)
    m = arrays[0].shape[0]
    mean = lambda x: None if x is None else 0.5 * (x[:m] + x[m:])
    return mean(out_value), mean(out_gradient)


def apply_tensor_batch(tables: TensorTables, batch: TensorRowBatch, fields, *,
                       values: bool = True, gradients: bool = True,
                       context: TensorFieldContext | None = None):
    """``(value (T, F) or None, gradient (T, 3, F) or None)`` of every target of ``batch``.

    Row ``t`` belongs to the target with ``batch.value_slots[t]`` (and ``gradient_slots[t]``);
    the gradient is ``None`` unless requested and the batch stores gradients. A paired batch (``batch.mirror``)
    gives ``0.5 * (A + B)`` of its two parts. ``context`` is the
    result of :func:`prepare_fields` for the batches of the call (built here for this batch if
    omitted). Traceable; the jitted entry point is ``apply_source_rows``.
    """
    tables = jax.tree_util.tree_map(jnp.asarray, tables)
    want_gradient = gradients and batch.gradient_slots is not None
    if not values and not want_gradient:
        return None, None
    if context is None:
        context = prepare_fields(tables, (batch,), fields, values=values, gradients=gradients)
    ringwise = batch.t_ring is not None
    paired = batch.mirror is not None
    total = batch.value_slots.shape[0]
    nf = fields.shape[1]
    static = (ringwise, float(batch.dr_divisor))
    arrays = batch.index_arrays()
    apply = _block_apply_pair if paired else _block_apply
    entries = (32 if paired else 16) * max(nf, 1)                       # gathered elements per target

    limit = RING_BLOCK_ELEMENTS if ringwise else ANGULAR_BLOCK_ELEMENTS
    return _blocked(lambda chunk: apply(tables, static, context, chunk, values=values, gradients=want_gradient),
                    arrays, limit // entries, 8 * entries)
