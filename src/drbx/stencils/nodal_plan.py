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

The level D5c maps ``CAA, CAB, CBA, CBB`` of a ring-ring face are built by applying the level trace matching to unit
trace vectors: with traces ``a = T_A phi`` and ``b = T_B phi``, the Fourier modes common to both sides (no Nyquist) are
set to their average, and ``[dA; dB] = [[CAA, CAB], [CBA, CBB]] [a; b]`` are the nodal corrections along each side's
trace rows. Faces touching a non-ring block carry no maps (``None``).

Nothing here imports from ``scripts/``.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import numpy as np

from drbx.geometry.nodal_layout import NodalLayout, RingLevelBlock, node_raw_ids
from drbx.geometry.sbp_core import CoreBlock
from drbx.geometry.sbp_operators import ring_basis

SCHEMA = "drbx.nodal-plan.v2"
METRIC_SCHEMA = "drbx.nodal-metric.v1"


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

    def to_json(self) -> str:
        return json.dumps({f.name: getattr(self, f.name) for f in dc_fields(self)})

    @classmethod
    def from_json(cls, text: str) -> "NodalStructure":
        def tup(x):
            return tuple(tup(v) for v in x) if isinstance(x, list) else x

        raw = json.loads(text)
        return cls(**{k: tup(v) for k, v in raw.items()})


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
    """Zernike core: Cartesian ``D1, D2``, the outer side, the D5c gradient matrices and the shell columns ``Vm``."""

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
    """Transfer pair ``Iab (NB, NA)``, ``Iba (NA, NB)`` and level D5c maps (``None`` unless both sides are rings)."""

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
# Level D5c maps
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
# Metric gathering and providers
# ---------------------------------------------------------------------------
def raw_id_lookup(raw_ids: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Indices ``i`` with ``raw_ids[i] == query`` for every entry of ``query`` (``searchsorted`` on sorted ids).

    Raises ``KeyError`` listing the missing ids.
    """
    raw_ids = np.asarray(raw_ids)
    query = np.asarray(query)
    order = np.argsort(raw_ids, kind="stable")
    srt = raw_ids[order]
    pos = np.clip(np.searchsorted(srt, query), 0, max(len(srt) - 1, 0))
    found = srt[pos] == query if len(srt) else np.zeros(query.shape, dtype=bool)
    if not np.all(found):
        missing = np.unique(query[~found])
        shown = ", ".join(str(int(v)) for v in missing[:10])
        raise KeyError(f"{missing.size} raw ids not in the source arrays: {shown}{' ...' if missing.size > 10 else ''}")
    return order[pos]


def gather_ring_metric(layout: NodalLayout, raw_ids, h, jac, B, K, raw_points=None) -> NodalMetric:
    """Fill the ring nodes of a :class:`NodalMetric` from raw-grid arrays addressed by ``raw_ids``; core nodes are NaN.

    Sources are ``CellPlan.raw_ids/h/jac/B/K`` (pass ``raw_points=None``) or ``GeometryArrays.raw_points`` with
    ``p05_raw_h``, ``abs(p05_raw_jacobian)``, ``p06_raw_B``, ``p06_raw_K``. With ``raw_points`` the source points
    must equal ``(node_u, node_theta, eta_k)`` to 1e-13.
    """
    E, P = layout.n_eta, layout.P
    ring = layout.node_ring >= 0
    ids = node_raw_ids(layout)[:, ring]
    idx = raw_id_lookup(raw_ids, ids.ravel())
    if raw_points is not None:
        pts = np.asarray(raw_points)[idx].reshape(E, -1, 3)
        eta = (np.arange(E) + 0.5) * layout.deta
        want = np.stack([np.broadcast_to(layout.node_u[ring], pts.shape[:2]),
                         np.broadcast_to(layout.node_theta[ring], pts.shape[:2]),
                         np.broadcast_to(eta[:, None], pts.shape[:2])], axis=-1)
        if not np.allclose(pts, want, rtol=0.0, atol=1e-13):
            raise ValueError(f"raw_points differ from the node coordinates (max {np.abs(pts - want).max():.3e})")
    out = []
    for src, tail in ((h, (3,)), (jac, ()), (B, ()), (K, (3,))):
        full = np.full((E, P) + tail, np.nan)
        full[:, ring] = np.asarray(src, dtype=np.float64)[idx].reshape((E, -1) + tail)
        out.append(full)
    return NodalMetric(*out)


@runtime_checkable
class NodalMetricProvider(Protocol):
    """Metric evaluator at logical points ``(Q, 3) = (u, theta, eta)``: ``(h (Q, 3), jac (Q,), B (Q,), K (Q, 3))``."""

    def nodal_metric(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class _GeometryProviderMetric:
    def __init__(self, provider) -> None:
        self.provider = provider

    def nodal_metric(self, points):
        points = np.asarray(points, dtype=np.float64)
        h, jac = self.provider.p05_metric(points)
        _J, B, K = self.provider.p06_curvature(points)
        return np.asarray(h), np.abs(np.asarray(jac)), np.asarray(B), np.asarray(K)


def metric_from_geometry_provider(provider) -> NodalMetricProvider:
    """Adapt a :class:`~drbx.stencils.geometry_arrays.GeometryProvider`: ``p05_metric`` (``h``, ``abs`` jacobian) and ``p06_curvature`` (``B``, ``K``)."""
    return _GeometryProviderMetric(provider)


def nodal_metric_from_callable(layout: NodalLayout, fn: Callable) -> NodalMetric:
    """Evaluate ``fn(points (E*P, 3)) -> (h, jac, B, K)`` at every node (flat index ``k * P + p``)."""
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    pts = np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                    np.broadcast_to(eta[:, None], (E, P))], axis=-1).reshape(-1, 3)
    h, jac, B, K = (np.asarray(a, dtype=np.float64) for a in fn(pts))
    return NodalMetric(h.reshape(E, P, 3), jac.reshape(E, P), B.reshape(E, P), K.reshape(E, P, 3))


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
def _side_arrays(side) -> SideArrays:
    return SideArrays(np.asarray(side.T, dtype=np.float64), np.asarray(side.TF1, dtype=np.float64),
                      np.asarray(side.TF2, dtype=np.float64))


def build_nodal_plan(layout: NodalLayout, metric: NodalMetric) -> NodalPlan:
    """Build the plan: apply each block's frame transform to the metric, check ``jac > 0`` and finiteness."""
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

    structure = NodalStructure(n=layout.n, n_eta=E, P=P, blocks=tuple(blocks_desc), faces=tuple(faces_desc),
                               walls=walls, deta=layout.deta, du=layout.du, side_rows=tuple(side_rows),
                               core=core_desc)
    return NodalPlan(wxy=np.asarray(layout.wxy, dtype=np.float64), jac=jac, h=h, B=B, K=K,
                     Hp=layout.wxy[None, :] * jac, blocks=tuple(blocks_arr), faces=tuple(faces_arr),
                     structure=structure, core_Ginv=core_Ginv)


# ---------------------------------------------------------------------------
# Save / load with a sha256 identity
# ---------------------------------------------------------------------------
_TOP_FIELDS = ("wxy", "jac", "h", "B", "K", "Hp")
_RING_FIELDS = RingBlockArrays._fields
_FACE_FIELDS = FaceArrays._fields
_CORE_FIELDS = ("D1", "D2", "G1_phi", "G1_tr", "G2_phi", "G2_tr", "Vm")


def _named_arrays(plan: NodalPlan) -> list[tuple[str, np.ndarray]]:
    named = [(name, np.asarray(getattr(plan, name))) for name in _TOP_FIELDS]
    if plan.core_Ginv is not None:
        named.append(("core_Ginv", np.asarray(plan.core_Ginv)))
    for i, blk in enumerate(plan.blocks):
        if isinstance(blk, CoreBlockArrays):
            named += [(f"block{i}.{f}", np.asarray(getattr(blk, f))) for f in _CORE_FIELDS]
            named += [(f"block{i}.outer.{f}", np.asarray(getattr(blk.outer, f))) for f in SideArrays._fields]
        elif isinstance(blk, RingBlockArrays):
            named += [(f"block{i}.{f}", np.asarray(getattr(blk, f))) for f in _RING_FIELDS]
        else:
            named += [(f"block{i}.D1", np.asarray(blk.D1)), (f"block{i}.D2", np.asarray(blk.D2))]
            for side in ("inner", "outer"):
                named += [(f"block{i}.{side}.{f}", np.asarray(getattr(getattr(blk, side), f))) for f in SideArrays._fields]
    for i, face in enumerate(plan.faces):
        named += [(f"face{i}.{f}", np.asarray(getattr(face, f))) for f in _FACE_FIELDS if getattr(face, f) is not None]
    return named


def plan_identity(plan: NodalPlan) -> str:
    """sha256 over the structure, the schema and every array (name, dtype, shape, bytes)."""
    digest = hashlib.sha256()
    for name, array in _named_arrays(plan):
        digest.update(name.encode("utf-8"))
        digest.update(repr(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(plan.structure.to_json().encode("utf-8"))
    digest.update(SCHEMA.encode("utf-8"))
    return digest.hexdigest()


def save_nodal_plan(plan: NodalPlan, path: str | Path) -> str:
    """Write the plan to one npz (arrays, structure JSON, schema, identity); returns the identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = plan_identity(plan)
    payload = dict(_named_arrays(plan))
    payload["structure"] = np.array(plan.structure.to_json())
    payload["schema"] = np.array(SCHEMA)
    payload["identity"] = np.array(identity)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)
    return identity


def load_nodal_plan(path: str | Path, *, expected_identity: str | None = None) -> NodalPlan:
    """Read a plan written by :func:`save_nodal_plan`; raises ``ValueError`` on a schema or identity mismatch."""
    with np.load(Path(path), allow_pickle=False) as data:
        schema, identity = str(data["schema"]), str(data["identity"])
        structure = NodalStructure.from_json(str(data["structure"]))
        arrays = {name: np.asarray(data[name]) for name in data.files if name not in ("schema", "identity", "structure")}
    if schema != SCHEMA:
        raise ValueError(f"unsupported nodal-plan schema {schema!r}")

    blocks = []
    for i, desc in enumerate(structure.blocks):
        if desc[0] == "ring":
            blocks.append(RingBlockArrays(*(arrays[f"block{i}.{f}"] for f in _RING_FIELDS)))
        elif desc[0] == "core":
            outer = SideArrays(*(arrays[f"block{i}.outer.{f}"] for f in SideArrays._fields))
            blocks.append(CoreBlockArrays(arrays[f"block{i}.D1"], arrays[f"block{i}.D2"], outer,
                                          *(arrays[f"block{i}.{f}"] for f in _CORE_FIELDS[2:])))
        else:
            sides = {s: SideArrays(*(arrays[f"block{i}.{s}.{f}"] for f in SideArrays._fields)) for s in ("inner", "outer")}
            blocks.append(DenseBlockArrays(arrays[f"block{i}.D1"], arrays[f"block{i}.D2"], sides["inner"], sides["outer"]))
    faces = [FaceArrays(*(arrays.get(f"face{i}.{f}") for f in _FACE_FIELDS)) for i in range(len(structure.faces))]
    plan = NodalPlan(**{name: arrays[name] for name in _TOP_FIELDS}, blocks=tuple(blocks), faces=tuple(faces),
                     structure=structure, core_Ginv=arrays.get("core_Ginv"))
    recomputed = plan_identity(plan)
    if recomputed != identity:
        raise ValueError(f"nodal-plan identity does not match its arrays (stored {identity!r}, recomputed {recomputed!r})")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"nodal-plan identity {identity!r} != expected {expected_identity!r}")
    return plan


# ---------------------------------------------------------------------------
# Logical-frame metric extraction products
# ---------------------------------------------------------------------------
def _layout_points(layout: NodalLayout) -> np.ndarray:
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def _metric_identity(arrays: dict, meta_json: str) -> str:
    digest = hashlib.sha256()
    for name in ("h", "jac", "B", "K", "points"):
        a = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(repr(a.dtype).encode("utf-8"))
        digest.update(repr(a.shape).encode("utf-8"))
        digest.update(a.tobytes())
    digest.update(meta_json.encode("utf-8"))
    digest.update(METRIC_SCHEMA.encode("utf-8"))
    return digest.hexdigest()


def save_nodal_metric(path: str | Path, metric: NodalMetric, points, meta: dict) -> str:
    """Write a logical-frame nodal metric (``h, jac, B, K`` and the ``points (E, P, 3)`` it was evaluated at) with ``meta``.

    Returns the sha256 identity over the arrays, the meta JSON and the schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"h": np.asarray(metric.h), "jac": np.asarray(metric.jac), "B": np.asarray(metric.B),
              "K": np.asarray(metric.K), "points": np.asarray(points)}
    meta_json = json.dumps(meta, sort_keys=True)
    identity = _metric_identity(arrays, meta_json)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, meta=np.array(meta_json), schema=np.array(METRIC_SCHEMA), identity=np.array(identity), **arrays)
    tmp.replace(path)
    return identity


def load_nodal_metric(path: str | Path, layout: NodalLayout, expected_identity: str | None = None) -> tuple[NodalMetric, dict]:
    """Read a file written by :func:`save_nodal_metric`; returns ``(NodalMetric, meta)``.

    Raises ``ValueError`` on a schema, identity or ``expected_identity`` mismatch, or if the stored points differ from the
    layout's node ``(u, theta, eta)`` by more than 1e-13.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        schema, identity, meta_json = str(data["schema"]), str(data["identity"]), str(data["meta"])
        arrays = {name: np.asarray(data[name]) for name in ("h", "jac", "B", "K", "points")}
    if schema != METRIC_SCHEMA:
        raise ValueError(f"unsupported nodal-metric schema {schema!r}")
    if _metric_identity(arrays, meta_json) != identity:
        raise ValueError("nodal-metric identity does not match its arrays")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"nodal-metric identity {identity!r} != expected {expected_identity!r}")
    want = _layout_points(layout)
    if arrays["points"].shape != want.shape or not np.allclose(arrays["points"], want, rtol=0.0, atol=1e-13):
        raise ValueError("the stored metric points do not match the layout's nodes")
    return NodalMetric(arrays["h"], arrays["jac"], arrays["B"], arrays["K"]), json.loads(meta_json)
