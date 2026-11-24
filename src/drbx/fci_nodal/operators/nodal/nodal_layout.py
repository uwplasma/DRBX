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
:class:`Block` implemented elsewhere.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .sbp_operators import (
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
        """Transform logical-frame ``(h (E, n_b, 3), jac (E, n_b))`` to the block frame.

        A block may also define ``to_block_frame_K(K (E, n_b, 3), u, theta)`` (contravariant curvature); the plan builder
        applies it when present and leaves ``K`` unchanged otherwise.
        """


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

    def __init__(self, i0: int, i1: int, N: int, n: int, *, delta: float | None = None) -> None:
        self.i0, self.i1, self.N, self.n = int(i0), int(i1), int(N), int(n)
        m = self.i1 - self.i0
        self.m = m
        self.du = 1.0 / self.n
        self.delta = np.pi / self.n if delta is None else float(delta)
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
    #: Versioned family contract; None retains the legacy cubic-lattice interpretation.
    shape: object | None = None

    @property
    def n_r(self) -> int:
        return self.n

    @property
    def n_theta(self) -> int:
        return self.n if self.shape is None else self.shape.n_theta


def build_nodal_layout(
    n: int,
    levels,
    *,
    n_eta: int | None = None,
    core: Block | None = None,
    inner: Literal["core", "wall"] = "core",
    shape=None,
) -> NodalLayout:
    """Assemble the layout from explicit ring levels ``(i0, i1, N)``.

    Levels must be contiguous, end at ``n``, be at least 8 rings wide and have ``N | n_theta`` (``n_theta=n`` on the legacy path). With
    ``inner="core"`` a ``core`` block is required and its outer side must sit at ``u = i0 / n`` of the first
    level; with ``inner="wall"`` there is no core and the first level starts at ``i0 > 0`` with an inner wall.
    Faces join the core to level 0 and level ``l`` to ``l + 1``; walls are the last level's outer side
    (and level 0's inner side for ``inner="wall"``).
    """
    n = int(n)
    n_eta = n if n_eta is None else int(n_eta)
    levels = tuple((int(i0), int(i1), int(N)) for i0, i1, N in levels)
    n_theta = n if shape is None else shape.n_theta
    if shape is not None:
        if (shape.n_r, shape.n_eta) != (n, n_eta) or levels != ((shape.K, n, n_theta),) or inner != "core":
            raise ValueError("layout does not match the versioned Zernike core-ring layout shape")
        if core is None or core.K != shape.K or core.p != shape.p or core.N_g != n_theta or not np.array_equal(core.theta_g, shape.delta + TWO_PI * np.arange(n_theta) / n_theta):
            raise ValueError("core policy or trace does not match the versioned shape")
    if n_eta < 1:
        raise ValueError("n_eta must be positive")
    if inner not in ("core", "wall"):
        raise ValueError(f"inner must be 'core' or 'wall', got {inner!r}")
    if not levels:
        raise ValueError("at least one ring level is required")
    for i, (i0, i1, N) in enumerate(levels):
        if i1 - i0 < MIN_RADIAL_POINTS:
            raise ValueError(f"level {i} has {i1 - i0} rings, need at least {MIN_RADIAL_POINTS}")
        if N < 1 or n_theta % N:
            raise ValueError(f"level {i}: N={N} must divide angular lattice={n_theta}")
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

    delta = np.pi / n if shape is None else shape.delta
    blocks: list = [] if core is None else [core]
    blocks += [RingLevelBlock(i0, i1, N, n, delta=delta) for i0, i1, N in levels]
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
            node_raw_theta[sl] = np.tile(np.arange(blk.N) * (n_theta // blk.N), blk.m)

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
        levels=levels, inner=inner, shape=shape,
    )


def eta_planes(layout: NodalLayout) -> np.ndarray:
    """Plane centres ``eta_k = (k + 1/2) deta``."""
    return (np.arange(layout.n_eta) + 0.5) * layout.deta


def node_points(layout: NodalLayout, *, planes: slice = slice(None)) -> np.ndarray:
    """Logical node coordinates ``(n_eta, P, 3) = (u, theta, eta_k)`` of the nodal state."""
    eta = eta_planes(layout)[planes]
    E, P = len(eta), layout.P
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def wall_points(layout: NodalLayout, wall: Wall, *, planes: slice = slice(None)) -> np.ndarray:
    """Face points ``(n_eta, N_w, 3)`` of ``wall`` as ``(u_face, delta + 2 pi j / N_w, eta_k)``."""
    side = layout.blocks[wall.block_idx].sides[wall.side]
    theta = layout.delta + TWO_PI * np.arange(side.N) / side.N
    eta = eta_planes(layout)[planes]
    pts = np.empty((len(eta), side.N, 3))
    pts[..., 0] = side.u_face
    pts[..., 1] = theta[None, :]
    pts[..., 2] = eta[:, None]
    return pts


