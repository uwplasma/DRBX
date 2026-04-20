from __future__ import annotations

from pathlib import Path

from jax_drb.runtime.performance import _compilation_cache_dir, _default_user_cache_root


def test_default_user_cache_root_prefers_xdg(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/jax_drb_cache_root")
    assert _default_user_cache_root() == Path("/tmp/jax_drb_cache_root")


def test_compilation_cache_dir_honors_override(monkeypatch) -> None:
    monkeypatch.setenv("JAX_DRB_CACHE_DIR", "/tmp/jax_drb_explicit_cache")
    assert _compilation_cache_dir() == Path("/tmp/jax_drb_explicit_cache")
