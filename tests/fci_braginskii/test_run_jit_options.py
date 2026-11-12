"""The driver compiles its kernels with device-side while loops on the GPU only."""

import jax

from drbx.fci_braginskii import run as driver


def test_jit_requests_device_side_loops_only_on_gpu(monkeypatch):
    seen = []
    monkeypatch.setattr(jax, "jit", lambda fn, **kwargs: seen.append(kwargs) or fn)
    for backend in ("cpu", "gpu"):
        monkeypatch.setattr(jax, "default_backend", lambda backend=backend: backend)
        driver._jit(len)
    assert seen[0]["compiler_options"] is None
    assert "WHILE" in seen[1]["compiler_options"]["xla_gpu_enable_command_buffer"]
