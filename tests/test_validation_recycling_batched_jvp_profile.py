from __future__ import annotations

import numpy as np

from jax_drb.validation.recycling_batched_jvp_profile import _check_pmap_identity


class _ReadyArray:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float64)

    def block_until_ready(self):
        return self._value


class _FakeJax:
    def __init__(self, *, scale: float = 1.0):
        self.scale = float(scale)

    def pmap(self, function, *, devices):
        def mapped(block):
            return _ReadyArray(self.scale * function(block))

        return mapped


def test_recycling_batched_jvp_pmap_identity_helper_requires_multiple_devices() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(_FakeJax(), np, ("cpu0",))

    assert passed is False
    assert max_abs_error is None
    assert skip_reason == "fewer than two visible JAX devices"


def test_recycling_batched_jvp_pmap_identity_helper_accepts_identity_map() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(_FakeJax(), np, ("gpu0", "gpu1"))

    assert passed is True
    assert max_abs_error == 0.0
    assert skip_reason is None


def test_recycling_batched_jvp_pmap_identity_helper_rejects_corrupt_map() -> None:
    passed, max_abs_error, skip_reason = _check_pmap_identity(_FakeJax(scale=0.0), np, ("gpu0", "gpu1"))

    assert passed is False
    assert max_abs_error > 0.0
    assert "pmap identity check failed" in str(skip_reason)
