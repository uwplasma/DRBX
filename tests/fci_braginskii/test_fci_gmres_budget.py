"""GMRES iteration-budget regression tests.

``solvax_gmres_solve`` must honour ``maxiter`` without collapsing the Krylov
cycle length: the old ``gcd(restart, maxiter)`` rule turned
``restart=50, maxiter=101`` into GMRES(1), which stagnates on nonsymmetric
operators.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.geometry.fci_geometry import FciGeometry3D  # noqa: E402
from drbx.fci_braginskii.native.fci_gmres import (  # noqa: E402
    SolvaxGmresConfig,
    _spmd_norm,
    gmres_cycle_budget,
    solvax_gmres_solve,
)
from drbx.fci_braginskii.native.fci_sharding import (  # noqa: E402
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
)
from drbx.geometry.open_slab import build_open_slab_geometry  # noqa: E402


@pytest.mark.parametrize(
    ("restart", "maxiter", "expected"),
    [
        (50, 101, (50, 2, 1)),
        (50, 100, (50, 2, 0)),
        (100, 30, (30, 1, 0)),
        (7, 1, (1, 1, 0)),
    ],
)
def test_gmres_cycle_budget_examples(restart, maxiter, expected):
    assert gmres_cycle_budget(restart, maxiter) == expected


@pytest.mark.parametrize(("restart", "maxiter"), [(0, 10), (10, 0)])
def test_gmres_cycle_budget_rejects_non_positive(restart, maxiter):
    with pytest.raises(ValueError):
        gmres_cycle_budget(restart, maxiter)


def test_gmres_cycle_budget_identity_grid():
    for requested in range(1, 40):
        for maxiter in range(1, 120):
            restart, cycles, tail = gmres_cycle_budget(requested, maxiter)
            assert restart == min(requested, maxiter)
            assert 0 <= tail < restart
            assert restart * cycles + tail == maxiter


def _single_device_slab(shape=(6, 6, 4)):
    legacy = build_open_slab_geometry(shape)
    global_geometry = FciGeometry3D(
        **{f.name: getattr(legacy, f.name) for f in dataclasses.fields(legacy)}
    )
    sharded = build_local_fci_geometries(
        global_geometry, (1, 1, 1), periodic_axes=(False, True, False)
    )
    geometry = assemble_single_device_local_fci_geometry(sharded)
    domain = dataclasses.replace(sharded.domain, mesh_axis_names=(None, None, None))
    return geometry, domain


def _convection_dominated_operator(shape, diffusion=0.05, velocity=(1.0, 0.7, 0.4)):
    """Dirichlet central-difference convection-diffusion (strongly nonsymmetric)."""

    def apply_A(u):
        padded = jnp.pad(u, 1)
        out = 6.0 * diffusion * u
        for axis, speed in enumerate(velocity):
            lo = [slice(1, -1)] * 3
            hi = [slice(1, -1)] * 3
            lo[axis] = slice(0, -2)
            hi[axis] = slice(2, None)
            minus, plus = padded[tuple(lo)], padded[tuple(hi)]
            out = out - diffusion * (minus + plus) + 0.5 * speed * (plus - minus)
        return out

    return apply_A


def test_coprime_budget_keeps_full_krylov_cycle_and_converges():
    geometry, domain = _single_device_slab()
    shape = geometry.owned_shape
    apply_A = _convection_dominated_operator(shape)
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(shape))
    config = SolvaxGmresConfig(tol=1.0e-8, atol=1.0e-12, restart=50, maxiter=101)

    phi, info = jax.jit(
        lambda b: solvax_gmres_solve(
            apply_A, b, jnp.zeros_like(b), geometry, domain, config
        )
    )(rhs)

    residual = float(_spmd_norm(rhs - apply_A(phi), geometry, domain))
    rhs_norm = float(_spmd_norm(rhs, geometry, domain))
    assert int(info.num_steps) <= 101
    assert bool(info.converged)
    assert residual <= config.tol * rhs_norm
    assert float(info.final_residual_l2) == pytest.approx(residual, rel=1e-6, abs=1e-14)


def test_relaxed_acceptance_is_reported_separately_from_strict_convergence():
    geometry, domain = _single_device_slab()
    shape = geometry.owned_shape
    apply_A = _convection_dominated_operator(shape)
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(shape))

    def solve(config):
        return jax.jit(
            lambda b: solvax_gmres_solve(apply_A, b, jnp.zeros_like(b), geometry, domain, config)
        )(rhs)[1]

    strict = solve(SolvaxGmresConfig(tol=1.0e-8, atol=1.0e-12, restart=50, maxiter=101))
    assert bool(strict.converged) and bool(strict.strict_converged)

    # Ten iterations cannot reach 1e-12 but do reach a 0.5 relative residual.
    relaxed = solve(
        SolvaxGmresConfig(
            tol=1.0e-12, atol=0.0, restart=10, maxiter=10, acceptance_tol=0.5,
            residual_correction_steps=0,
        )
    )
    assert bool(relaxed.converged) and not bool(relaxed.failed)
    assert not bool(relaxed.strict_converged)
    assert 1.0e-12 < float(relaxed.final_residual_rel_l2) <= 0.5
