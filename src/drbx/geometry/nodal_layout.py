"""Host-side node layout for the seam-independent nodal perpendicular scheme.

The nodal state of one eta plane is a vector of ``P`` nodes: an optional opaque core
block followed by ring levels. A ring level is a set of ``m`` radial rings of ``N``
Fourier nodes each, stored ring-major (node ``offset + a * N + j``), with its own
radial SBP block (:func:`drbx.geometry.sbp_operators.radial_block`). Neighbouring
levels, and the core and the first level, meet at faces coupled by a Fourier transfer
pair. The full state is plane-major with shape ``(n_eta, P[, F])`` (flat index
``k * P + p``).

Conventions: ``n`` is the raw grid size, ``du = 1 / n``, ring nodes sit at
``u_i = (i + 1/2) / n`` and ``theta_j = delta + 2 pi j / N`` with ``delta = pi / n`` (so
every node coincides with a raw cell centre at raw theta index ``j * (n // N)``), and
eta planes sit at ``eta_k = (k + 1/2) * deta`` with ``deta = 2 pi / n_eta``.

Everything here is host NumPy built once per grid. The core block is a
:class:`Block` implemented elsewhere; :class:`DenseBlock` is its generic container.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from drbx.geometry.sbp_operators import (
    MIN_RADIAL_POINTS,
    TWO_PI,
    RadialBlockOps,
    fourier_pair,
    radial_block,
    ring_dtheta,
)

Frame = Literal["polar", "cartesian"]


@dataclass(frozen=True, eq=False)
class Side:
    """One radial side of a block: trace and normal-flux trace onto ``N`` face points.

    Face points are ``theta = delta + 2 pi j / N``. ``T`` is the block-local trace operator
    ``(N, n_b)``; the normal (+u) flux trace per radian is
    ``ftrace = TF1 @ (F1 g) + TF2 @ (F2 g)``, with ``TF1 = T`` and ``TF2 = 0`` on ring sides.
    ``rows`` lists the static ring rows with a nonzero trace weight (empty for dense sides).
    """

    N: int
    T: np.ndarray
    TF1: np.ndarray
    TF2: np.ndarray
    u_face: float
    rows: tuple[int, ...] = ()


@runtime_checkable
class Block(Protocol):
    """A block of nodes of one plane; the core implements this protocol and is opaque to the layout."""

    kind: str
    n_nodes: int
    frame: Frame
    u: np.ndarray
    theta: np.ndarray
    wxy: np.ndarray
    sides: Mapping[str, Side]
    velocity_gradient: Callable | None
    dissipation: Callable | None

    def to_block_frame(self, h_log, jac_log, u, theta):
        """Transform logical-frame ``(h (E, n_b, 3), jac (E, n_b))`` to the block frame."""


def _ring_side(t: np.ndarray, N: int, u_face: float) -> Side:
    T = np.kron(t[None, :], np.eye(N))
    return Side(N=N, T=T, TF1=T, TF2=np.zeros_like(T), u_face=float(u_face),
                rows=tuple(int(r) for r in np.nonzero(t)[0]))


class RingLevelBlock:
    """Rings ``i0 <= i < i1`` of ``N`` Fourier nodes each on the raw grid of size ``n`` (polar frame)."""

    kind = "ring"
    frame: Frame = "polar"
    velocity_gradient = None
    dissipation = None

    def __init__(self, i0: int, i1: int, N: int, n: int) -> None:
        self.i0, self.i1, self.N, self.n = int(i0), int(i1), int(N), int(n)
        m = self.i1 - self.i0
        self.m = m
        self.du = 1.0 / self.n
        self.delta = np.pi / self.n
        self.n_nodes = m * self.N
        self.radial: RadialBlockOps = radial_block(m)
        self.Du = self.radial.D_unit / self.du
        self.Dth = ring_dtheta(self.N, self.delta)
        ui = (np.arange(self.i0, self.i1) + 0.5) * self.du
        self.u = np.repeat(ui, self.N)
        self.theta = np.tile(self.delta + TWO_PI * np.arange(self.N) / self.N, m)
        self.wxy = np.repeat(self.radial.w * self.du, self.N) * (TWO_PI / self.N)
        self.sides = {
            "inner": _ring_side(self.radial.tL, self.N, self.i0 * self.du),
            "outer": _ring_side(self.radial.tR, self.N, self.i1 * self.du),
        }

    def to_block_frame(self, h_log, jac_log, u=None, theta=None):
        return h_log, jac_log


@dataclass(frozen=True, eq=False)
class DenseBlock:
    """Generic block with dense in-plane operators ``D1, D2`` (the core's container).

    ``frame_transform(h_log, jac_log, u, theta) -> (h_b, jac_b)`` is required for a ``"cartesian"`` frame.
    ``evaluate_at(values (n_eta, n_nodes[, F]), u (Q,), theta (Q,)) -> (n_eta, Q[, F])`` optionally
    evaluates a block field at arbitrary points (used by :func:`expand_to_raw`).
    """

    D1: np.ndarray
    D2: np.ndarray
    wxy: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    sides: Mapping[str, Side]
    frame: Frame = "polar"
    kind: str = "dense"
    frame_transform: Callable | None = None
    velocity_gradient: Callable | None = None
    dissipation: Callable | None = None
    evaluate_at: Callable | None = None
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_nodes", int(np.shape(self.wxy)[0]))

    def to_block_frame(self, h_log, jac_log, u=None, theta=None):
        if self.frame == "polar":
            return h_log, jac_log
        if self.frame_transform is None:
            raise NotImplementedError("a cartesian DenseBlock needs a frame_transform")
        return self.frame_transform(h_log, jac_log, u, theta)


@dataclass(frozen=True, eq=False)
class Face:
    """Interface between block ``A`` (inner, its outer side) and block ``B`` (outer, its inner side).

    ``A``, ``B`` are ``(block_idx, side_name)``; ``Iab (NB, NA)`` / ``Iba (NA, NB)`` are the transfer pair
    and ``X`` names the finer side (``"B"`` when ``NB >= NA``).
    """

    A: tuple[int, str]
    B: tuple[int, str]
    NA: int
    NB: int
    X: str
    Iab: np.ndarray
    Iba: np.ndarray


@dataclass(frozen=True)
class Wall:
    """A physical wall on side ``side`` of block ``block_idx``; ``sign`` is +1 at an outer side (normal +u), -1 at an inner wall."""

    block_idx: int
    side: str
    sign: int


@dataclass(frozen=True, eq=False)
class NodalLayout:
    """Node layout of one plane (see the module docstring); arrays are per plane node ``(P,)``."""

    n: int
    n_eta: int
    delta: float
    du: float
    deta: float
    blocks: tuple
    offsets: tuple[int, ...]
    P: int
    faces: tuple[Face, ...]
    walls: tuple[Wall, ...]
    node_u: np.ndarray
    node_theta: np.ndarray
    node_block: np.ndarray
    node_ring: np.ndarray
    node_raw_theta: np.ndarray
    wxy: np.ndarray
    levels: tuple[tuple[int, int, int], ...]
    inner: str


def build_nodal_layout(
    n: int,
    levels,
    *,
    n_eta: int | None = None,
    core: Block | None = None,
    inner: Literal["core", "wall"] = "core",
) -> NodalLayout:
    """Assemble the layout from explicit ring levels ``(i0, i1, N)``.

    Levels must be contiguous, end at ``n``, be at least 8 rings wide and have ``N | n``. With
    ``inner="core"`` a ``core`` block is required and its outer side must sit at ``u = i0 / n`` of the first
    level; with ``inner="wall"`` there is no core and the first level starts at ``i0 > 0`` with an inner wall.
    Faces join the core to level 0 and level ``l`` to ``l + 1``; walls are the last level's outer side
    (and level 0's inner side for ``inner="wall"``).
    """
    n = int(n)
    n_eta = n if n_eta is None else int(n_eta)
    levels = tuple((int(i0), int(i1), int(N)) for i0, i1, N in levels)
    if n_eta < 1:
        raise ValueError("n_eta must be positive")
    if inner not in ("core", "wall"):
        raise ValueError(f"inner must be 'core' or 'wall', got {inner!r}")
    if not levels:
        raise ValueError("at least one ring level is required")
    for i, (i0, i1, N) in enumerate(levels):
        if i1 - i0 < MIN_RADIAL_POINTS:
            raise ValueError(f"level {i} has {i1 - i0} rings, need at least {MIN_RADIAL_POINTS}")
        if N < 1 or n % N:
            raise ValueError(f"level {i}: N={N} must divide n={n}")
        if i and i0 != levels[i - 1][1]:
            raise ValueError(f"levels {i - 1} and {i} are not contiguous")
    if levels[-1][1] != n:
        raise ValueError(f"the last level must end at n={n}, got {levels[-1][1]}")
    i0_first = levels[0][0]
    if inner == "core":
        if core is None:
            raise ValueError("inner='core' requires a core block")
        if not np.isclose(core.sides["outer"].u_face, i0_first / n, rtol=0.0, atol=1e-12):
            raise ValueError(f"core outer u_face {core.sides['outer'].u_face} != first level i0/n = {i0_first / n}")
    else:
        if core is not None:
            raise ValueError("inner='wall' takes no core block")
        if i0_first <= 0:
            raise ValueError("inner='wall' requires the first level to start at i0 > 0")

    delta = np.pi / n
    blocks: list = [] if core is None else [core]
    blocks += [RingLevelBlock(i0, i1, N, n) for i0, i1, N in levels]
    sizes = [b.n_nodes for b in blocks]
    offsets = tuple(int(o) for o in np.concatenate([[0], np.cumsum(sizes)[:-1]]))
    P = int(sum(sizes))

    node_ring = np.full(P, -1, dtype=np.int64)
    node_raw_theta = np.full(P, -1, dtype=np.int64)
    node_block = np.zeros(P, dtype=np.int64)
    for b_idx, (blk, off) in enumerate(zip(blocks, offsets)):
        node_block[off:off + blk.n_nodes] = b_idx
        if isinstance(blk, RingLevelBlock):
            sl = slice(off, off + blk.n_nodes)
            node_ring[sl] = np.repeat(np.arange(blk.i0, blk.i1), blk.N)
            node_raw_theta[sl] = np.tile(np.arange(blk.N) * (n // blk.N), blk.m)

    first_ring = 0 if core is None else 1
    faces = []
    for a_idx in range(len(blocks) - 1):
        b_idx = a_idx + 1
        NA, NB = blocks[a_idx].sides["outer"].N, blocks[b_idx].sides["inner"].N
        Iab, Iba = fourier_pair(NA, NB, delta)
        faces.append(Face((a_idx, "outer"), (b_idx, "inner"), NA, NB, "B" if NB >= NA else "A", Iab, Iba))
    walls = [Wall(len(blocks) - 1, "outer", +1)]
    if inner == "wall":
        walls.append(Wall(first_ring, "inner", -1))

    return NodalLayout(
        n=n, n_eta=n_eta, delta=float(delta), du=1.0 / n, deta=TWO_PI / n_eta,
        blocks=tuple(blocks), offsets=offsets, P=P, faces=tuple(faces), walls=tuple(walls),
        node_u=np.concatenate([np.asarray(b.u, dtype=np.float64) for b in blocks]),
        node_theta=np.concatenate([np.asarray(b.theta, dtype=np.float64) for b in blocks]),
        node_block=node_block, node_ring=node_ring, node_raw_theta=node_raw_theta,
        wxy=np.concatenate([np.asarray(b.wxy, dtype=np.float64) for b in blocks]),
        levels=levels, inner=inner,
    )


def eta_planes(layout: NodalLayout) -> np.ndarray:
    """Plane centres ``eta_k = (k + 1/2) deta``."""
    return (np.arange(layout.n_eta) + 0.5) * layout.deta


def wall_points(layout: NodalLayout, wall: Wall) -> np.ndarray:
    """Face points ``(n_eta, N_w, 3)`` of ``wall`` as ``(u_face, delta + 2 pi j / N_w, eta_k)``."""
    side = layout.blocks[wall.block_idx].sides[wall.side]
    theta = layout.delta + TWO_PI * np.arange(side.N) / side.N
    eta = eta_planes(layout)
    pts = np.empty((layout.n_eta, side.N, 3))
    pts[..., 0] = side.u_face
    pts[..., 1] = theta[None, :]
    pts[..., 2] = eta[:, None]
    return pts


def node_raw_ids(layout: NodalLayout) -> np.ndarray:
    """Raw cell ids ``(n_eta, P)``: ``(i n + j (n // N)) n + k`` for ring nodes, ``-1`` for core nodes."""
    n = layout.n
    if layout.n_eta != n:
        raise ValueError("node_raw_ids needs n_eta == n (raw eta planes)")
    ring, jj = layout.node_ring, layout.node_raw_theta
    base = np.where(ring >= 0, (ring * n + jj) * n, -1)
    ids = base[None, :] + np.arange(n, dtype=np.int64)[:, None]
    return np.where(ring[None, :] >= 0, ids, -1).astype(np.int64)


def nodal_plane_layout(layout: NodalLayout):
    """Identity :class:`~drbx.native.owner_plane_layout.PlaneLayout` over the ``P`` plane-major nodes."""
    from drbx.native.owner_plane_layout import PlaneLayout

    ident = np.arange(layout.P, dtype=np.int64)
    return PlaneLayout(perm=ident, inverse=ident.copy(), m=layout.P)


def nodal_owner_layout_arrays(layout: NodalLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(owner_ring, owner_plane, owner_theta)`` over the ``P * n_eta`` flat nodes ``k * P + p``.

    Ring nodes carry their absolute ring and raw theta index; core nodes carry ring ``-1`` and their
    in-block index as theta.
    """
    E, P = layout.n_eta, layout.P
    theta = layout.node_raw_theta.copy()
    core = layout.node_ring < 0
    theta[core] = np.arange(P, dtype=np.int64)[core] - np.asarray(layout.offsets)[layout.node_block[core]]
    ring = np.tile(layout.node_ring, E)
    plane = np.repeat(np.arange(E, dtype=np.int64), P)
    return ring, plane, np.tile(theta, E)


def expand_to_raw(layout: NodalLayout, values: np.ndarray) -> np.ndarray:
    """Expand nodal ``values (n_eta, P[, F])`` to the raw grid ``(n, n, n_eta[, F])``.

    Each ring level is mapped to ``n`` points per ring by the Fourier transfer ``fourier_pair(N, n)``
    (identity when ``N == n``). Core rings are NaN unless the core block supplies ``evaluate_at``.
    """
    values = np.asarray(values, dtype=np.float64)
    scalar = values.ndim == 2
    vals = values[..., None] if scalar else values
    n, E = layout.n, layout.n_eta
    if vals.shape[:2] != (E, layout.P):
        raise ValueError(f"values must have shape ({E}, {layout.P}[, F]), got {values.shape}")
    F = vals.shape[2]
    out = np.full((n, n, E, F), np.nan)
    for blk, off in zip(layout.blocks, layout.offsets):
        if isinstance(blk, RingLevelBlock):
            Iab = fourier_pair(blk.N, n, layout.delta)[0]
            gl = vals[:, off:off + blk.n_nodes].reshape(E, blk.m, blk.N, F)
            out[blk.i0:blk.i1] = np.einsum("jl,ealf->ajef", Iab, gl)
        elif getattr(blk, "evaluate_at", None) is not None:
            i1 = layout.levels[0][0]
            ii, jj = np.meshgrid(np.arange(i1), np.arange(n), indexing="ij")
            u = ((ii + 0.5) * layout.du).ravel()
            th = (layout.delta + TWO_PI * jj / n).ravel()
            core_vals = vals[:, off:off + blk.n_nodes]
            ev = np.asarray(blk.evaluate_at(core_vals[..., 0] if scalar else core_vals, u, th))
            out[:i1] = ev.reshape(E, i1, n, F).transpose(1, 2, 0, 3)
    return out[..., 0] if scalar else out
