"""Norms, region masks and region errors for the nodal SBP perpendicular scheme.

``H = Hp * deta`` is the diagonal quadrature weight on ``(E, P)`` node arrays. ``h_weights``, ``h_inner`` and
``h_norm`` are pure JAX; the region masks and error table are host NumPy diagnostics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from drbx.geometry.nodal_layout import NodalLayout, RingLevelBlock
from drbx.stencils.nodal_plan import NodalPlan

#: default u-bands of the region table
DEFAULT_U_BANDS = ((0.0, 0.06), (0.06, 0.12), (0.12, 0.21), (0.21, 0.27), (0.27, 0.40), (0.40, 1.00))
#: rings in the wall band and either side of a level face
WALL_BAND_RINGS = 4


def h_weights(plan: NodalPlan):
    """Full diagonal norm ``H = Hp * deta`` with shape ``(E, P)``."""
    return plan.Hp * plan.structure.deta


def h_inner(a, b, H):
    """``sum H a b`` (over all axes of ``H``; trailing field axes of ``a``, ``b`` are summed too)."""
    w = jnp.reshape(H, H.shape + (1,) * (jnp.ndim(a) - H.ndim))
    return jnp.sum(w * a * b)


def h_norm(a, H):
    return jnp.sqrt(h_inner(a, a, H))


def ring_region_masks(layout: NodalLayout, n_eta: int,
                      bands: Sequence[tuple[float, float]] = DEFAULT_U_BANDS) -> dict[str, np.ndarray]:
    """Boolean ``(n_eta, P)`` masks by ring and u-band; empty regions are dropped.

    Regions: ``core`` (non-ring nodes), ``core_band`` (the first level's first four rings, only with a core),
    ``level_bands`` (``i - 4 <= ring <= i + 3`` around every ring-ring level face starting at ``i``), ``wall``
    (last four rings), ``adjacent_band`` (rings ``n - 6 .. n - 2``), ``wall_ring`` (ring ``n - 1``), ``inner_wall`` (first
    four rings of an ``inner="wall"`` layout), ``interior`` (none of ``core``, ``core_band``, ``level_bands``, ``wall``,
    ``inner_wall``), ``all`` and one ``uband_<lo>-<hi>`` per band.
    """
    r = np.broadcast_to(layout.node_ring[None, :], (n_eta, layout.P))
    u = np.broadcast_to(layout.node_u[None, :], (n_eta, layout.P))
    core = r < 0
    first = layout.levels[0][0]
    core_band = (r >= first) & (r <= first + 3) if layout.inner == "core" else np.zeros_like(core)
    inner_wall = (r >= first) & (r <= first + 3) if layout.inner == "wall" else np.zeros_like(core)
    level_bands = np.zeros_like(core)
    for face in layout.faces:
        blk = layout.blocks[face.B[0]]
        if isinstance(blk, RingLevelBlock) and isinstance(layout.blocks[face.A[0]], RingLevelBlock):
            level_bands |= (r >= blk.i0 - WALL_BAND_RINGS) & (r <= blk.i0 + WALL_BAND_RINGS - 1)
    wall = r >= layout.n - WALL_BAND_RINGS
    interior = ~(core | core_band | level_bands | wall | inner_wall)
    out = {"core": core, "core_band": core_band, "level_bands": level_bands, "interior": interior, "wall": wall,
           "adjacent_band": (r >= layout.n - 6) & (r <= layout.n - 2), "wall_ring": r == layout.n - 1,
           "inner_wall": inner_wall, "all": np.ones_like(core)}
    for lo, hi in bands:
        out[f"uband_{lo:.2f}-{hi:.2f}"] = (u >= lo) & (u < hi)
    return {k: np.array(v) for k, v in out.items() if v.any()}


def region_errors(err, ref, H, masks: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Per-region ``rms``, ``rel_global``, ``rel_region``, ``max``, ``volume = sum H_m`` and ``share``.

    ``share = e2_m / e2_all`` is the region's part of the global squared H-error ``sum H err^2`` (regions
    overlap, so shares need not sum to 1). ``rel_*`` are ``None`` where the reference vanishes.
    """
    err, ref, H = (np.asarray(a, dtype=np.float64) for a in (err, ref, H))
    gref = np.sqrt(np.sum(H * ref**2) / np.sum(H))
    e2_all = np.sum(H * err**2)
    out = {}
    for name, m in masks.items():
        Hm = H[m]
        e2, r2 = np.sum(Hm * err[m] ** 2), np.sum(Hm * ref[m] ** 2)
        rms = float(np.sqrt(e2 / Hm.sum()))
        out[name] = dict(rms=rms, rel_global=float(rms / gref) if gref > 0 else None,
                         rel_region=float(np.sqrt(e2 / r2)) if r2 > 0 else None, max=float(np.abs(err[m]).max()),
                         volume=float(Hm.sum()), share=float(e2 / e2_all) if e2_all > 0 else 0.0)
    return out
