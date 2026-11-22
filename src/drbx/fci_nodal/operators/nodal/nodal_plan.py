"""Plan for the seam-independent nodal perpendicular scheme: static structure plus a pytree of arrays.

``NodalStructure`` is the hashable static description (block, face and wall descriptors) and ``NodalPlan`` the
pytree of arrays built once on the host from a :class:`~drbx.geometry.nodal_layout.NodalLayout` and a nodal metric.
The plan is passed to jitted functions as an argument (never closed over as constants); only ``structure`` is a
meta field (see ``operator_plan.py``). Arrays are host float64 NumPy arrays.

Metric and curvature are carried per node and eta plane in the block frame: ``h (E, P, 3)``, ``jac (E, P)`` (positive
``|J|``), ``B (E, P)``, ``K (E, P, 3)``; ``Hp = wxy * jac`` is the per-plane norm (the full norm is ``Hp * deta``).
``h``, ``jac`` and (when the block provides ``to_block_frame_K``, as the Cartesian core does) ``K`` are transformed to
each block's frame; ``B`` is a scalar and unchanged. ``K`` is not used by the scheme yet.

A core block (``CoreBlock``) is stored as :class:`CoreBlockArrays` and ``core_Ginv (E, d_m, d_m)`` holds, per plane, the
inverse Gram matrix of the ``nu <= p - 1`` Zernike columns under the block-frame weight ``wxy * jac`` (the data of the
shell-damping projector); ``structure.core`` describes it.

The level matched-trace velocity gradient maps ``CAA, CAB, CBA, CBB`` of a ring-ring face are built by applying the level trace matching to unit
trace vectors: with traces ``a = T_A phi`` and ``b = T_B phi``, the Fourier modes common to both sides (no Nyquist) are
set to their average, and ``[dA; dB] = [[CAA, CAB], [CBA, CBB]] [a; b]`` are the nodal corrections along each side's
trace rows. Faces touching a non-ring block carry no maps (``None``).

Nothing here imports from ``scripts/``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields as dc_fields
from typing import NamedTuple

import jax
import numpy as np

from .nodal_layout import NodalLayout, RingLevelBlock
from .sbp_core import CoreBlock
from .sbp_operators import ring_basis

SCHEMA = "drbx.nodal-plan.v2"


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodalStructure:
    """Static, hashable description of a nodal plan.

    ``blocks``: ``(kind, offset, n_nodes, m, N, i0)`` (``m = N = 0``, ``i0 = -1`` for non-ring blocks);
    ``faces``: ``(a_blk, b_blk, NA, NB, X)``; ``walls``: ``(blk, side, sign, N)``;
    ``side_rows``: ``(blk, side, rows)`` with the static ring rows carrying a nonzero trace weight;
    ``core``: ``(block_idx, p, R_c, N_c, d_m, ring_blk)`` of a Zernike core (``ring_blk`` the ring level it faces) or ``None``.
    """

    n: int
    n_eta: int
    P: int
    blocks: tuple[tuple[str, int, int, int, int, int], ...]
    faces: tuple[tuple[int, int, int, int, str], ...]
    walls: tuple[tuple[int, str, int, int], ...]
    deta: float
    du: float
    side_rows: tuple[tuple[int, str, tuple[int, ...]], ...]
    core: tuple | None = None


    @classmethod
    def from_json(cls, text: str) -> "NodalStructure":
        def tup(x):
            return tuple(tup(v) for v in x) if isinstance(x, list) else x

        raw = json.loads(text)
        if "shape_json" in raw:
            from .nodal_shape import NodalShape

            shape = NodalShape.from_dict(json.loads(raw["shape_json"]))
            if shape.version != 2 or raw["shape_json"] != json.dumps(shape.to_dict(), sort_keys=True):
                raise ValueError("noncanonical version-2 plan shape metadata")
            expected_p = (shape.p // 2 + 1) * (2 * shape.p + 2) + (shape.n_r - shape.K) * shape.n_theta
            if (raw["n"], raw["n_eta"], raw["P"]) != (shape.n_r, shape.n_eta, expected_p):
                raise ValueError("plan counts differ from version-2 shape metadata")
            cls = ReplicatedNodalStructure
        return cls(**{k: tup(v) for k, v in raw.items()})


@dataclass(frozen=True)
class ReplicatedNodalStructure(NodalStructure):
    """Immutable canonical v2 metadata, included in plan serialization and identity."""

    shape_json: str = ""


# ---------------------------------------------------------------------------
# Array containers (NamedTuples are pytrees)
# ---------------------------------------------------------------------------
class RingBlockArrays(NamedTuple):
    """Ring level: ``Du (m, m)`` (already divided by ``du``), ``Dth (N, N)``, ``tL, tR (m,)``, norm ``w (m,)``."""

    Du: np.ndarray
    Dth: np.ndarray
    tL: np.ndarray
    tR: np.ndarray
    w: np.ndarray


class SideArrays(NamedTuple):
    T: np.ndarray
    TF1: np.ndarray
    TF2: np.ndarray


class DenseBlockArrays(NamedTuple):
    D1: np.ndarray
    D2: np.ndarray
    inner: SideArrays
    outer: SideArrays


class CoreBlockArrays(NamedTuple):
    """Zernike core: Cartesian ``D1, D2``, the outer side, the matched-trace velocity gradient gradient matrices and the shell columns ``Vm``."""

    D1: np.ndarray
    D2: np.ndarray
    outer: SideArrays
    G1_phi: np.ndarray
    G1_tr: np.ndarray
    G2_phi: np.ndarray
    G2_tr: np.ndarray
    Vm: np.ndarray

    @property
    def inner(self):
        return None


class FaceArrays(NamedTuple):
    """Transfer pair ``Iab (NB, NA)``, ``Iba (NA, NB)`` and level matched-trace velocity gradient maps (``None`` unless both sides are rings)."""

    Iab: np.ndarray
    Iba: np.ndarray
    CAA: np.ndarray | None
    CAB: np.ndarray | None
    CBA: np.ndarray | None
    CBB: np.ndarray | None


@dataclass(frozen=True)
class NodalPlan:
    wxy: np.ndarray       # (P,)
    jac: np.ndarray       # (E, P)
    h: np.ndarray         # (E, P, 3)
    B: np.ndarray         # (E, P)
    K: np.ndarray         # (E, P, 3)
    Hp: np.ndarray        # (E, P)
    blocks: tuple         # RingBlockArrays | DenseBlockArrays | CoreBlockArrays per block
    faces: tuple          # FaceArrays per face
    structure: NodalStructure
    core_Ginv: np.ndarray | None = None   # (E, d_m, d_m) shell-damping Gram inverses (core only)


jax.tree_util.register_dataclass(
    NodalPlan,
    data_fields=[f.name for f in dc_fields(NodalPlan) if f.name != "structure"],
    meta_fields=["structure"],
)


class NodalMetric(NamedTuple):
    """Nodal metric data: ``h (E, P, 3)``, ``jac (E, P)`` (positive ``|J|``), ``B (E, P)``, ``K (E, P, 3)``."""

    h: np.ndarray
    jac: np.ndarray
    B: np.ndarray
    K: np.ndarray


# ---------------------------------------------------------------------------
# Level matched-trace velocity gradient maps
# ---------------------------------------------------------------------------
def _level_trace_delta(tr_a, tr_b, side_a, side_b):
    """Nodal corrections ``(dA, dB)`` equalising the common Fourier modes (no Nyquist) of two face traces.

    ``tr_*`` are ``(N_s, K)`` trace columns; ``side_*`` are ``(B, Binv, keys)`` of the ring bases.
    """
    (BA, BiA, kA), (BB, BiB, kB) = side_a, side_b
    aA, aB = BiA @ tr_a, BiB @ tr_b
    pos_b = {k: q for q, k in enumerate(kB)}
    dA, dB = np.zeros_like(aA), np.zeros_like(aB)
    for q, k in enumerate(kA):
        if k[1] == "n" or k not in pos_b:
            continue
        p = pos_b[k]
        tgt = 0.5 * (aA[q] + aB[p])
        dA[q] = tgt - aA[q]
        dB[p] = tgt - aB[p]
    return BA @ dA, BB @ dB


def level_d5c_maps(NA: int, NB: int, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(CAA, CAB, CBA, CBB)`` with ``[dA; dB] = [[CAA, CAB], [CBA, CBB]] [a; b]`` (see the module docstring)."""
    BA, BiA, kA, _ = ring_basis(NA, delta)
    BB, BiB, kB, _ = ring_basis(NB, delta)
    sa, sb = (BA, BiA, kA), (BB, BiB, kB)
    zeros_b, zeros_a = np.zeros((NB, NA)), np.zeros((NA, NB))
    CAA, CBA = _level_trace_delta(np.eye(NA), zeros_b, sa, sb)
    CAB, CBB = _level_trace_delta(zeros_a, np.eye(NB), sa, sb)
    return CAA, CAB, CBA, CBB


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def _side_arrays(side) -> SideArrays:
    return SideArrays(np.asarray(side.T, dtype=np.float64), np.asarray(side.TF1, dtype=np.float64),
                      np.asarray(side.TF2, dtype=np.float64))


def build_nodal_plan(layout: NodalLayout, metric: NodalMetric) -> NodalPlan:
    """Build the plan: apply each block's frame transform to the metric, check ``jac > 0`` and finiteness."""
    if layout.shape is not None:
        core, ring = layout.blocks
        if core.N_g != ring.N or not np.array_equal(core.theta_g, ring.theta[:ring.N]):
            raise ValueError("versioned Zernike core-ring layout core and ring traces must match")
    E, P = layout.n_eta, layout.P
    h_log = np.asarray(metric.h, dtype=np.float64)
    jac_log = np.asarray(metric.jac, dtype=np.float64)
    B = np.asarray(metric.B, dtype=np.float64)
    K = np.asarray(metric.K, dtype=np.float64)
    for name, arr, shape in (("h", h_log, (E, P, 3)), ("jac", jac_log, (E, P)), ("B", B, (E, P)), ("K", K, (E, P, 3))):
        if arr.shape != shape:
            raise ValueError(f"metric.{name} must have shape {shape}, got {arr.shape}")
    h = np.empty_like(h_log)
    jac = np.empty_like(jac_log)
    K = K.copy()
    for blk, off in zip(layout.blocks, layout.offsets):
        sl = slice(off, off + blk.n_nodes)
        hb, jb = blk.to_block_frame(h_log[:, sl], jac_log[:, sl], blk.u, blk.theta)
        h[:, sl], jac[:, sl] = hb, jb
        if hasattr(blk, "to_block_frame_K"):
            K[:, sl] = blk.to_block_frame_K(K[:, sl], blk.u, blk.theta)
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(jac))):
        raise ValueError("the nodal metric h and jac must be finite at every node (is the core metric filled?)")
    if not np.all(jac > 0.0):
        raise ValueError(f"the block-frame jacobian must be positive, min {jac.min():.3e}")

    blocks_arr, blocks_desc, side_rows = [], [], []
    core_desc, core_Ginv = None, None
    for b_idx, (blk, off) in enumerate(zip(layout.blocks, layout.offsets)):
        if isinstance(blk, CoreBlock):
            blocks_arr.append(CoreBlockArrays(blk.D1, blk.D2, _side_arrays(blk.sides["outer"]), blk.G1_phi, blk.G1_tr,
                                              blk.G2_phi, blk.G2_tr, blk.Vm))
            blocks_desc.append((blk.kind, off, blk.n_nodes, 0, 0, -1))
            Hk = layout.wxy[None, off:off + blk.n_nodes] * jac[:, off:off + blk.n_nodes]
            gram = np.einsum("qi,eq,qj->eij", blk.Vm, Hk, blk.Vm)
            core_Ginv = np.linalg.inv(gram)
            core_desc = (b_idx, blk.p, float(blk.R_c), blk.n_nodes, blk.Vm.shape[1], b_idx + 1)
        elif isinstance(blk, RingLevelBlock):
            blocks_arr.append(RingBlockArrays(blk.Du, blk.Dth, blk.radial.tL, blk.radial.tR, blk.radial.w))
            blocks_desc.append((blk.kind, off, blk.n_nodes, blk.m, blk.N, blk.i0))
        else:
            blocks_arr.append(DenseBlockArrays(np.asarray(blk.D1, dtype=np.float64), np.asarray(blk.D2, dtype=np.float64),
                                               _side_arrays(blk.sides["inner"]), _side_arrays(blk.sides["outer"])))
            blocks_desc.append((blk.kind, off, blk.n_nodes, 0, 0, -1))
        for name in blk.sides:
            side_rows.append((b_idx, name, tuple(int(r) for r in blk.sides[name].rows)))

    faces_arr, faces_desc = [], []
    for f in layout.faces:
        ring_ring = isinstance(layout.blocks[f.A[0]], RingLevelBlock) and isinstance(layout.blocks[f.B[0]], RingLevelBlock)
        maps = level_d5c_maps(f.NA, f.NB, layout.delta) if ring_ring else (None,) * 4
        faces_arr.append(FaceArrays(f.Iab, f.Iba, *maps))
        faces_desc.append((f.A[0], f.B[0], f.NA, f.NB, f.X))
    walls = tuple((w.block_idx, w.side, w.sign, layout.blocks[w.block_idx].sides[w.side].N) for w in layout.walls)

    version2 = layout.shape is not None and layout.shape.version == 2
    structure_type = ReplicatedNodalStructure if version2 else NodalStructure
    shape_kw = {"shape_json": json.dumps(layout.shape.to_dict(), sort_keys=True)} if version2 else {}
    structure = structure_type(n=layout.n, n_eta=E, P=P, blocks=tuple(blocks_desc), faces=tuple(faces_desc),
                               walls=walls, deta=layout.deta, du=layout.du, side_rows=tuple(side_rows),
                               core=core_desc, **shape_kw)
    return NodalPlan(wxy=np.asarray(layout.wxy, dtype=np.float64), jac=jac, h=h, B=B, K=K,
                     Hp=layout.wxy[None, :] * jac, blocks=tuple(blocks_arr), faces=tuple(faces_arr),
                     structure=structure, core_Ginv=core_Ginv)


# ---------------------------------------------------------------------------
# Save / load with a sha256 identity
# ---------------------------------------------------------------------------


