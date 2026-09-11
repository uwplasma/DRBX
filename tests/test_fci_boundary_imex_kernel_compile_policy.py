import jax
import jax.numpy as jnp

from drbx.native import fci_boundary_imex_kernels as kernels


def test_residual_kernel_disables_only_constant_folding(monkeypatch):
    seen = {}

    def fake_jit(function, **kwargs):
        seen.update(kwargs)
        return function

    monkeypatch.setattr(kernels.jax, "jit", fake_jit)
    fn = kernels.build_coupled_residual_kernel(
        object(), residual_fn=lambda vector, base, dt, source: vector + base + dt + source
    )
    assert seen == {"compiler_options": {"xla_disable_hlo_passes": "constant_folding"}}
    expected = jnp.asarray([10.0, 11.0])
    actual = fn(jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 4.0]), 0.5, jnp.asarray([5.5, 4.5]))
    assert jnp.array_equal(actual, expected)


def test_jax_supports_the_narrow_compiler_option():
    fn = kernels.build_coupled_residual_kernel(
        object(), residual_fn=lambda vector, base, dt, source: vector + base + dt + source
    )
    with jax.disable_jit(False):
        actual = fn(jnp.asarray([1.0]), jnp.asarray([2.0]), 0.5, jnp.asarray([3.0]))
    assert jnp.allclose(actual, jnp.asarray([6.5]))
