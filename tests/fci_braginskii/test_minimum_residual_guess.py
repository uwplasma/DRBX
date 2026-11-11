"""Minimum-residual starting guess from earlier solutions of the same step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.native.fci_gmres import (  # noqa: E402
    SolvaxGmresConfig,
    _spmd_norm,
    minimum_residual_guess,
    solvax_gmres_solve,
)
from .test_fci_gmres_budget import (  # noqa: E402
    _convection_dominated_operator,
    _single_device_slab,
)


def _setup():
    geometry, domain = _single_device_slab()
    shape = geometry.owned_shape
    apply_A = _convection_dominated_operator(shape)
    rng = np.random.default_rng(1)
    exact = jnp.asarray(rng.standard_normal(shape))
    return geometry, domain, apply_A, exact, apply_A(exact), rng


def _norm(geometry, domain, x):
    return float(_spmd_norm(x, geometry, domain))


def test_empty_and_duplicate_basis_keep_the_guess_and_zero_columns_are_ignored():
    geometry, domain, apply_A, exact, rhs, _ = _setup()
    guess = 0.3 * exact
    for basis in ([], [guess], [guess, guess], [jnp.zeros_like(exact), guess]):
        start = minimum_residual_guess(apply_A, rhs, guess, basis, geometry, domain)
        assert jnp.allclose(start, guess + (start - guess))  # finite
        assert _norm(geometry, domain, rhs - apply_A(start)) <= _norm(
            geometry, domain, rhs - apply_A(guess)
        ) * (1 + 1e-12)


def test_basis_containing_the_solution_returns_it_and_never_worsens_the_residual():
    geometry, domain, apply_A, exact, rhs, rng = _setup()
    noise = [jnp.asarray(rng.standard_normal(exact.shape)) for _ in range(2)]
    guess = jnp.zeros_like(exact)
    start = minimum_residual_guess(apply_A, rhs, guess, [noise[0], exact, noise[1]], geometry, domain)
    assert _norm(geometry, domain, start - exact) < 1e-8 * _norm(geometry, domain, exact)
    # unrelated vectors cannot make it worse than the guess
    start = minimum_residual_guess(apply_A, rhs, guess, noise, geometry, domain)
    assert _norm(geometry, domain, rhs - apply_A(start)) <= _norm(geometry, domain, rhs)


def test_extrapolates_a_linear_drift_and_cuts_the_iterations():
    geometry, domain, apply_A, exact, rhs, rng = _setup()
    drift = jnp.asarray(rng.standard_normal(exact.shape))
    # the solution lies one drift step beyond the last stored one: c = 2
    older, newer = exact - 2.0 * drift, exact - drift
    start = minimum_residual_guess(apply_A, rhs, older, [newer], geometry, domain)
    assert _norm(geometry, domain, start - exact) < 1e-8 * _norm(geometry, domain, exact)
    config = SolvaxGmresConfig(tol=1e-10, atol=1e-14, restart=60, maxiter=180)

    def iterations(x0):
        return int(solvax_gmres_solve(apply_A, rhs, x0, geometry, domain, config)[1].num_steps)

    assert iterations(start) < iterations(older)
    # exact start: the solver returns at once
    assert iterations(exact) == 0


def test_no_gradient_flows_through_the_start():
    geometry, domain, apply_A, exact, rhs, _ = _setup()
    g = jax.grad(
        lambda s: jnp.sum(minimum_residual_guess(apply_A, rhs, 0.0 * exact, [s * exact], geometry, domain))
    )(1.0)
    assert float(g) == 0.0
