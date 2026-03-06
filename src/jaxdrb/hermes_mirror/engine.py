from __future__ import annotations

"""Temporary engine entry points for strict Hermes parity work.

The runtime engine is intentionally not wired yet. Phase 1 only lands the
mirror primitives and their tests; engine activation starts after the mirrored
operators exist.
"""


def build_system(*args, **kwargs):
    raise NotImplementedError("Mirror engine wiring starts after the primitive/operator phases.")
