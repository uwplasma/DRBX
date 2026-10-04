"""Layout families of the nodal perpendicular scheme.

Family A: a Zernike core of radius ``K = n / 8`` rings and degree ``p = K + 2`` around the axis, and one ring level of
``n`` angular nodes per ring from ``K`` to ``n``.
"""
from __future__ import annotations

from drbx.geometry.nodal_layout import NodalLayout, build_nodal_layout
from drbx.geometry.sbp_core import build_core_block


def family_a(n: int) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    """``(K, p, levels)`` of family A for raw grid size ``n`` (``n % 8 == 0`` and ``n >= 16``)."""
    n = int(n)
    if n % 8 or n < 16:
        raise ValueError(f"family A needs n divisible by 8 and n >= 16, got n={n}")
    K = n // 8
    return K, K + 2, ((K, n, n),)


def build_family_a_layout(n: int, n_eta: int | None = None) -> NodalLayout:
    """Family-A :class:`NodalLayout` with its core block (``P = 952 / 2106 / 3716`` at ``n = 32 / 48 / 64``)."""
    K, p, levels = family_a(n)
    return build_nodal_layout(n, levels, n_eta=n_eta, core=build_core_block(n, K, p), inner="core")
