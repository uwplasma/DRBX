"""Tests for the factorized weighted coarse polarization data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drbx.native.fci_polarization_coarse import build_coarse_data


def _fixture():
    shape = (8, 3, 6)
    rng = np.random.default_rng(20260905)
    active = rng.random(shape) > 0.15
    # Ensure every radial/eta location has at least one owner in theta.
    active[:, 0, :] = True
    weights = 0.4 + rng.random(shape)
    flat = np.flatnonzero(active.ravel())
    mass = weights.ravel()[flat]
    s = np.sqrt(mass)
    P = np.eye(flat.size) - np.outer(s, s) / np.dot(s, s)
    R = rng.standard_normal((flat.size, flat.size))
    S = R.T @ R + 0.5 * np.eye(flat.size)
    Aactive = (S @ P)
    Aactive = (P @ Aactive)
    Aactive = (Aactive / s[:, None]) * s[None, :]

    def apply(values):
        values = jnp.asarray(values, dtype=jnp.float64)
        x = values.reshape(-1)[flat]
        y = jnp.asarray(Aactive) @ x
        out = jnp.zeros(values.size, dtype=jnp.float64).at[flat].set(y)
        return out.reshape(shape)

    return shape, jnp.asarray(active), jnp.asarray(weights), apply


def test_factorized_q_and_transpose_are_weighted_and_jittable():
    shape, active, weights, apply_A = _fixture()
    data = build_coarse_data(apply_A, active, weights, radial_modes=8)
    Q = np.column_stack(
        [np.asarray(data.q_apply(jnp.eye(data.rank)[:, j], active, weights)).ravel()[active.ravel()]
         for j in range(data.rank)]
    )
    mass = np.asarray(weights).ravel()[np.asarray(active).ravel()]
    np.testing.assert_allclose(Q.T @ (mass[:, None] * Q), np.eye(data.rank), atol=2e-9)
    constant = np.ones(shape)
    np.testing.assert_allclose(np.asarray(apply_A(constant)), 0.0, atol=2e-10)
    rng = np.random.default_rng(8)
    x = rng.standard_normal(shape)
    qtx = np.asarray(data.q_transpose(jnp.asarray(x), active, weights))
    np.testing.assert_allclose(qtx, Q.T @ (mass * x.ravel()[active.ravel()]), atol=2e-8)
    eager = np.asarray(data.coarse(jnp.asarray(x), active, weights))
    compiled = np.asarray(jax.jit(data.coarse)(jnp.asarray(x), active, weights))
    np.testing.assert_allclose(compiled, eager, atol=2e-9)


def test_coarse_action_is_projection_and_constant_null():
    _shape, active, weights, apply_A = _fixture()
    data = build_coarse_data(apply_A, active, weights, radial_modes=8)
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.standard_normal(active.shape))
    c = np.asarray(data.coarse(x, active, weights))
    # Coarse() is Q H^-1 Q.T M, so it maps into the coarse span.  The
    # Galerkin exactness identity is C A Q = Q (rather than C x = x).
    qc = np.asarray(data.q_apply(jnp.ones(data.rank), active, weights))
    np.testing.assert_allclose(
        np.asarray(data.coarse(apply_A(qc), active, weights)), qc, atol=3e-8
    )
    np.testing.assert_allclose(np.asarray(data.coarse(jnp.ones(active.shape), active, weights)), 0.0, atol=2e-10)


def test_inactive_nan_values_are_ignored_and_pytree_roundtrip():
    _shape, active, weights, apply_A = _fixture()
    data = build_coarse_data(apply_A, active, weights, radial_modes=8)
    poisoned = jnp.where(active, 0.3, jnp.nan)
    result = data.coarse(poisoned, active, weights)
    assert np.isfinite(np.asarray(result)[np.asarray(active)]).all()
    leaves, treedef = jax.tree_util.tree_flatten(data)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(restored.coarse(poisoned, active, weights)), np.asarray(result), atol=2e-10)


def test_coarse_builder_rejects_nonfinite_quotient():
    shape, active, weights, _apply_A = _fixture()

    def bad_apply(values):
        return jnp.full(shape, jnp.nan, dtype=jnp.float64)

    with pytest.raises(ValueError, match="non-finite"):
        build_coarse_data(bad_apply, active, weights, radial_modes=8)


def test_coarse_builder_rejects_singular_quotient():
    shape, active, weights, _apply_A = _fixture()

    def singular_apply(values):
        return jnp.zeros_like(values)

    with pytest.raises(ValueError, match="singular|non-finite"):
        build_coarse_data(singular_apply, active, weights, radial_modes=8)


def test_nonsymmetric_coarse_lu_inverse_is_jittable_and_accurate():
    shape, active, weights, _apply = _fixture()

    def nonsymmetric(values):
        values = jnp.asarray(values, dtype=jnp.float64)
        return values + 0.2 * jnp.roll(values, 1, axis=1)

    data = build_coarse_data(nonsymmetric, active, weights, radial_modes=8)
    coeff = jnp.linspace(-0.7, 0.9, data.rank)
    q = data.q_apply(coeff, active, weights)
    expected = q
    actual = data.coarse(nonsymmetric(q), active, weights)
    compiled = jax.jit(data.coarse)(nonsymmetric(q), active, weights)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-7)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(expected), atol=2e-7)


@pytest.mark.parametrize("preconditioner", ["jacobi", "coarse-additive", "coarse-multiplicative"])
def test_production_augmented_neumann_coarse_additive_jit(preconditioner, monkeypatch):
    """Exercise the real solver's augmented solve with prebuilt coarse data."""
    from dataclasses import replace
    from tests.test_augmented_neumann_phi import (
        _all_neumann_radial_bc, _build_domain, _build_ghost_filler,
        _build_local_geometry, _mms_parallel_field,
    )
    from drbx.native.fci_halo import HaloExchange3D, LocalPeriodicTopologyRule3D, TopologyHaloFiller3D
    from drbx.native.fci_gmres import SolvaxGmresConfig
    from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver, _homogeneous_local_face_bc
    from drbx.native import fci_boundary_imex_preconditioner as cycle_module

    cycle_calls = []
    original_cycle = cycle_module.build_multiplicative_polarization_vcycle

    def observe_cycle(*args, **kwargs):
        cycle_calls.append(kwargs["applications"])
        return original_cycle(*args, **kwargs)

    monkeypatch.setattr(cycle_module, "build_multiplicative_polarization_vcycle", observe_cycle)

    shape = (8, 4, 6)
    domain = _build_domain(shape, 2, (1, 1, 1))
    domain = replace(domain, mesh_axis_names=(None, None, None))
    geometry = _build_local_geometry(shape, 2, global_shape=shape)
    face_bc = _all_neumann_radial_bc(geometry)
    config = SolvaxGmresConfig(tol=1e-8, atol=1e-10, maxiter=1000, restart=40,
                               preconditioner="jacobi", regularization_epsilon=0.0,
                               project_mean_zero=False)
    base = LocalPerpLaplacianInverseSolver(
        geometry=geometry, domain=domain, halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
        physical_ghost_filler=_build_ghost_filler(2), face_bc=face_bc,
        operator_form="support-paired", config=config,
    )
    homogeneous = _homogeneous_local_face_bc(face_bc)
    A = jax.jit(lambda x: base._apply_A(x, face_bc=homogeneous,
                                         control_volume_boundary_bc=None,
                                         project_mean_zero=False,
                                         boundary_is_homogeneous=True))
    active, weights = base._operator_mass_weights()
    coarse = build_coarse_data(A, active, weights, radial_modes=8)
    solver = replace(base, config=replace(config, preconditioner=preconditioner), coarse_data=coarse)
    rho = jnp.linspace(0.2, 1.0, shape[0])[:, None, None]
    theta = jnp.linspace(0.0, 2*jnp.pi, shape[1], endpoint=False)[None, :, None]
    eta = jnp.linspace(0.0, 2*jnp.pi, shape[2], endpoint=False)[None, None, :]
    exact = _mms_parallel_field(rho, theta, eta)
    gauge = jnp.zeros(shape).at[0].set(1.0).at[-1].set(1.0)
    target = jnp.sum(gauge * exact) / jnp.sum(gauge)
    rhs = A(exact)
    solve = jax.jit(lambda b: solver.solve_neumann_with_gauge(
        b, gauge_weights_owned=gauge, gauge_target=target,
        gauge_affine_offset=0.0, return_diagnostics=True))
    solved, info = solve(rhs)
    assert cycle_calls == ([1] if preconditioner == "coarse-multiplicative" else [])
    assert not bool(np.asarray(info.failed))
    assert bool(np.asarray(info.converged))
    residual = np.asarray(A(solved) - rhs)
    weighted_rhs = np.sqrt(np.asarray(weights) * np.asarray(active))
    rel = np.linalg.norm(weighted_rhs * residual) / max(np.linalg.norm(weighted_rhs * np.asarray(rhs)), 1e-12)
    assert rel <= 1e-8
    delta = np.asarray(solved) - np.asarray(exact)
    delta -= np.sum(np.asarray(gauge) * delta) / np.sum(np.asarray(gauge))
    np.testing.assert_allclose(delta, 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jnp.sum(gauge * solved) / jnp.sum(gauge)), float(target), atol=2e-6)
    assert np.isfinite(np.asarray(solved)).all()
    assert np.isfinite(float(info.compatibility_multiplier))
    assert abs(float(info.compatibility_multiplier)) <= 1e-7
    rhs_bad = rhs + 0.125
    solved_bad, info_bad = solve(rhs_bad)
    assert not bool(np.asarray(info_bad.failed))
    np.testing.assert_allclose(np.asarray(solved_bad), np.asarray(solved), atol=1e-5)
    np.testing.assert_allclose(float(info_bad.compatibility_multiplier), 0.125, atol=2e-6)
