"""Reverse-mode gradients through ``solvax_gmres_solve`` (implicit-function VJP)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native.fci_gmres import SolvaxGmresConfig, solvax_gmres_solve
from tests.fci_braginskii.test_fci_gmres_budget import (
    _convection_dominated_operator,
    _single_device_slab,
)

jax.config.update("jax_enable_x64", True)

CONFIG = SolvaxGmresConfig(tol=1.0e-13, atol=1.0e-14, restart=50, maxiter=200)


def _loss(p, rhs, geometry, domain, weights=None, active=None, config=CONFIG, laplace=None):
    shape = geometry.owned_shape
    if laplace is None:
        apply_A = _convection_dominated_operator(shape, diffusion=0.05 * p)
    else:
        apply_A = lambda u: p * laplace(u)
    inv_diag = 1.0 / (6.0 * 0.05 + 1.0)
    phi, _ = solvax_gmres_solve(
        apply_A, rhs, jnp.zeros_like(rhs), geometry, domain, config, active_cell_mask=active,
        preconditioner=None if laplace is not None else (lambda v: inv_diag * v),
        volume_weights=weights,
    )
    return jnp.sum(phi * jnp.cos(jnp.arange(phi.size).reshape(phi.shape)))


def _check(loss, p0, step=1.0e-6, rtol=1.0e-6):
    value, grad = jax.jit(jax.value_and_grad(loss))(p0)
    fd = (loss(p0 + step) - loss(p0 - step)) / (2.0 * step)
    assert np.isfinite(float(grad)) and np.isfinite(float(value))
    assert float(grad) == pytest.approx(float(fd), rel=rtol)


def test_gradient_wrt_operator_parameter_and_rhs_matches_central_difference():
    geometry, domain = _single_device_slab()
    shape = geometry.owned_shape
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(rng.standard_normal(shape))
    weights = jnp.asarray(rng.uniform(0.5, 2.0, shape))
    active = jnp.asarray(rng.uniform(size=shape) > 0.1)
    _check(lambda p: _loss(p, rhs, geometry, domain, weights, active), 1.0)
    _check(lambda p: _loss(1.0, p * rhs, geometry, domain, weights, active), 1.0)


def test_zero_rhs_gives_zero_solution_and_finite_gradient():
    geometry, domain = _single_device_slab()
    rhs = jnp.zeros(geometry.owned_shape)
    value, grad = jax.value_and_grad(lambda p: _loss(p, rhs, geometry, domain))(1.0)
    assert float(value) == 0.0 and np.isfinite(float(grad))
    grad_rhs = jax.grad(lambda b: _loss(1.0, b, geometry, domain))(rhs)
    assert np.all(np.isfinite(np.asarray(grad_rhs)))


def test_singular_gauge_fixed_periodic_laplacian_gradient():
    geometry, domain = _single_device_slab()
    shape = geometry.owned_shape
    rhs = jnp.asarray(np.random.default_rng(2).standard_normal(shape))

    def laplace(u):
        return sum(2.0 * u - jnp.roll(u, 1, a) - jnp.roll(u, -1, a) for a in range(3))

    config = SolvaxGmresConfig(tol=1.0e-13, atol=1.0e-14, restart=60, maxiter=240, project_mean_zero=True)
    _check(lambda p: _loss(p, rhs, geometry, domain, config=config, laplace=laplace), 1.3)


def test_float32_gradient_is_finite_and_close_to_finite_difference():
    geometry, domain = _single_device_slab()  # built in float64, solved in float32
    shape = geometry.owned_shape
    enable_x64 = getattr(jax, "enable_x64", None) or __import__("jax.experimental", fromlist=["enable_x64"]).enable_x64
    with enable_x64(False):
        rhs = jnp.asarray(np.random.default_rng(1).standard_normal(shape), dtype=jnp.float32)
        config = SolvaxGmresConfig(tol=1.0e-6, atol=1.0e-7, restart=50, maxiter=200)
        loss = lambda p: _loss(p, rhs, geometry, domain, config=config)
        grad = jax.jit(jax.grad(loss))(jnp.float32(1.0))
        fd = (loss(jnp.float32(1.01)) - loss(jnp.float32(0.99))) / 0.02
        assert grad.dtype == jnp.float32 and np.isfinite(float(grad))
        assert float(grad) == pytest.approx(float(fd), rel=2.0e-2)


@pytest.mark.slow
def test_full_imex_step_gradient_matches_finite_difference_on_hsx_16_cubed(tmp_path, monkeypatch):
    """``jax.grad`` of a scalar of one compiled 16^3 IMEX step (four Krylov solves) against central differences.

    The driver's sharded advance is captured when the staged kernels are first lowered; the
    diagnostic ``pmin``/``pmax`` reductions have no JVP rule, so they are cut from the tape.
    """
    from pathlib import Path

    from drbx.cli import main

    cap = {}
    real_shard_map, real_jit = jax.shard_map, jax.jit

    class _Captured(BaseException):
        pass

    def shard_map_spy(f, *a, **k):
        out = real_shard_map(f, *a, **k)
        if getattr(f, "__name__", "") == "full_imex_advance":
            cap["advance"] = out
        return out

    class _JitSpy:
        def __init__(self, f):
            self.f = real_jit(f)

        def lower(self, *args):
            if "advance" in cap and len(args) >= 4 and hasattr(args[0], "Te"):
                cap["args"] = args[:4] + (jnp.asarray(0.0),)
                raise _Captured
            return self.f.lower(*args)

        def __call__(self, *a, **k):
            return self.f(*a, **k)

    monkeypatch.setattr(jax, "shard_map", shard_map_spy)
    monkeypatch.setattr(jax, "jit", lambda f, *a, **k: _JitSpy(f) if not (a or k) else real_jit(f, *a, **k))
    for name in ("pmin", "pmax"):
        real = getattr(jax.lax, name)
        monkeypatch.setattr(jax.lax, name, lambda x, ax, real=real: real(jax.lax.stop_gradient(x), ax))
    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n'
        "[geometry]\nresolution = [16, 16, 16]\nmetric_mesh_shape = [16, 16, 16]\n"
        "metric_poloidal_modes = 6\nmetric_toroidal_modes = 3\ntoroidal_modes = 3\n"
        f'cache_dir = "{Path(__file__).resolve().parents[2] / "output" / "hsx_geometry"}"\n'
        f'[fci_braginskii]\noutput = "{tmp_path / "hsx"}"\nfinal_time = 0.00075\nnum_steps = 1\n'
    )
    with pytest.raises(_Captured):
        main(["run", str(deck), "--quiet"])
    monkeypatch.setattr(jax, "jit", real_jit)
    monkeypatch.setattr(jax, "shard_map", real_shard_map)

    advance = cap["advance"]
    state, cell_fields, map_fields, control_volume_fields, time = cap["args"]

    def loss(a):
        out = advance(state.replace(Te=state.Te * (1.0 + a)), cell_fields, map_fields,
                      control_volume_fields, time)
        return sum(jnp.sum(v**2) for _, v in out[0].field_items())

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.0))
    forward = jax.jit(loss)
    eps = 1.0e-3
    fd = (float(forward(jnp.asarray(eps))) - float(forward(jnp.asarray(-eps)))) / (2.0 * eps)
    assert np.isfinite(float(value)) and np.isfinite(float(grad))
    assert float(grad) == pytest.approx(fd, rel=2.0e-4)
