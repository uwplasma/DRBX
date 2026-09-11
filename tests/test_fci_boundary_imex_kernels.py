import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_boundary_imex_kernels import build_coupled_residual_kernel


class _ToyModel:
    pass


def test_residual_kernel_accepts_dynamic_stage_inputs():
    calls = []

    def residual(vector, base, solve_dt, source):
        del source
        return vector - base - solve_dt * vector

    kernel = build_coupled_residual_kernel(_ToyModel(), residual_fn=residual)
    def outer_eager(*args):
        with jax.disable_jit(False):
            return kernel(*args)
    with jax.disable_jit(True):
        first = outer_eager(jnp.asarray([2.0, 3.0]), jnp.asarray([1.0, 1.0]), 0.1,
                            jnp.asarray([0.0, 0.0]))
        second = outer_eager(jnp.asarray([4.0, 5.0]), jnp.asarray([2.0, 1.0]), 0.2,
                             jnp.asarray([7.0, 8.0]))
        calls.extend((np.asarray(first), np.asarray(second)))
    np.testing.assert_allclose(calls[0], [0.8, 1.7])
    np.testing.assert_allclose(calls[1], [1.2, 3.0])


def test_residual_kernel_handles_pytree_dynamic_base_and_source():
    def residual(vector, base, solve_dt, source):
        return vector - (base["u"] + solve_dt * source["u"])

    kernel = build_coupled_residual_kernel(_ToyModel(), residual_fn=residual)
    with jax.disable_jit(False):
        value = kernel(jnp.asarray([3.0, 4.0]), {"u": jnp.asarray([1.0, 2.0])},
                       jnp.asarray(0.5), {"u": jnp.asarray([2.0, 4.0])})
    np.testing.assert_allclose(value, [1.0, 0.0])
