from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import grad, jit

from jax_drb.solver import build_fourier_helmholtz_operator, solve_fourier_helmholtz


def test_fourier_helmholtz_matches_dense_mode_solves() -> None:
    dx = jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64)
    dz = jnp.array([0.1, 0.1, 0.1], dtype=jnp.float64)
    g11 = jnp.array([1.0, 1.2, 1.1], dtype=jnp.float64)
    g33 = jnp.array([0.7, 0.8, 0.9], dtype=jnp.float64)
    rhs_scale = jnp.array([1.0, 0.9, 1.1], dtype=jnp.float64)
    rhs = jnp.array(
        [
            [1.0, 0.2, -0.5, 0.1],
            [0.3, -0.4, 0.8, 0.5],
            [-0.2, 0.6, 0.1, -0.7],
        ],
        dtype=jnp.float64,
    )
    operator = build_fourier_helmholtz_operator(dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=4)
    actual = np.asarray(solve_fourier_helmholtz(rhs, operator=operator), dtype=np.float64)

    rhs_hat = np.fft.rfft(np.asarray(rhs, dtype=np.float64) * np.asarray(rhs_scale)[:, None], axis=-1)
    expected_modes: list[np.ndarray] = []
    for mode in range(rhs_hat.shape[-1]):
        lower = np.asarray(operator.lower_diagonals[mode], dtype=np.complex128)
        diagonal = np.asarray(operator.diagonals[mode], dtype=np.complex128)
        upper = np.asarray(operator.upper_diagonals[mode], dtype=np.complex128)
        matrix = np.diag(np.asarray(diagonal))
        matrix += np.diag(np.asarray(upper[:-1]), k=1)
        matrix += np.diag(np.asarray(lower[1:]), k=-1)
        expected_modes.append(np.linalg.solve(matrix, rhs_hat[:, mode]))
    expected = np.fft.irfft(np.stack(expected_modes, axis=-1), n=4, axis=-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_fourier_helmholtz_supports_jit_and_grad() -> None:
    dx = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float64)
    dz = jnp.array([0.2, 0.2, 0.2, 0.2], dtype=jnp.float64)
    g11 = jnp.ones(4, dtype=jnp.float64)
    g33 = jnp.ones(4, dtype=jnp.float64)
    rhs_scale = jnp.ones(4, dtype=jnp.float64)
    operator = build_fourier_helmholtz_operator(dx=dx, dz=dz, g11=g11, g33=g33, rhs_scale=rhs_scale, nz=6)
    template_rhs = jnp.arange(24, dtype=jnp.float64).reshape(4, 6) / 10.0

    @jit
    def loss_fn(scale: jnp.ndarray) -> jnp.ndarray:
        solution = solve_fourier_helmholtz(scale * template_rhs, operator=operator)
        return jnp.sum(solution * solution)

    value = loss_fn(jnp.array(0.5, dtype=jnp.float64))
    derivative = grad(loss_fn)(jnp.array(0.5, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
