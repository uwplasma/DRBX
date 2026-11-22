"""Layout families of the nodal perpendicular scheme.

Zernike core-ring layout: a Zernike core of radius ``K = n / 8`` rings and degree ``p = K + 2`` around the axis, and one ring level of
``n`` angular nodes per ring from ``K`` to ``n``.
"""
from __future__ import annotations

from .nodal_layout import NodalLayout, build_nodal_layout
from .sbp_core import build_core_block


def zernike_core_ring_spec(n: int) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    """``(K, p, levels)`` of Zernike core-ring layout for raw grid size ``n`` (``n % 8 == 0`` and ``n >= 16``)."""
    n = int(n)
    if n % 8 or n < 16:
        raise ValueError(f"Zernike core-ring layout needs n divisible by 8 and n >= 16, got n={n}")
    K = n // 8
    return K, K + 2, ((K, n, n),)


def build_zernike_core_ring_layout(n: int, n_eta: int | None = None) -> NodalLayout:
    """Zernike core-ring layout :class:`NodalLayout` with its core block (``P = 952 / 2106 / 3716`` at ``n = 32 / 48 / 64``)."""
    K, p, levels = zernike_core_ring_spec(n)
    return build_nodal_layout(n, levels, n_eta=n_eta, core=build_core_block(n, K, p), inner="core")


def build_noncubic_zernike_core_ring_layout(n_r: int, n_theta: int, n_eta: int, *, version: int = 1,
                                    field_periods: int = 4, replication_period_count: int | None = None) -> NodalLayout:
    """Independent axes and matched traces; replication metadata is opt-in with ``version=2``."""
    from .nodal_shape import NodalShape

    shape = NodalShape(n_r, n_theta, n_eta, version=version, field_periods=field_periods,
                       replication_period_count=replication_period_count)
    core = build_core_block(shape.n_r, shape.K, shape.p, n_trace=shape.n_theta, delta=shape.delta)
    return build_nodal_layout(shape.n_r, ((shape.K, shape.n_r, shape.n_theta),),
                              n_eta=shape.n_eta, core=core, inner="core", shape=shape)
