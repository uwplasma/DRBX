"""Focused tests for the generic structured D2 x S1 MMPDE baseline."""

import numpy as np
import pytest

from drbx.geometry.solve_MMPDE import (
    MMPDEOptions,
    _edge_data,
    _energy_gradient,
    _infer_periodic_transform,
    solve_mmpde,
)


def translation(points, turns):
    out = np.asarray(points).copy()
    out[..., 2] += turns
    return out


def rotation_periodic(points, turns):
    angle = 0.5 * np.pi * turns
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(points) @ rotation.T + np.array([0.2, -0.1, 0.7])


def affine_mesh(nu=5, nv=5, neta=6):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    eta = np.linspace(0, 1, neta, endpoint=False)
    uu, vv, ee = np.meshgrid(u, v, eta, indexing="ij")
    return np.stack((uu, vv, ee), axis=-1), (u, v, eta)


def test_affine_periodic_translation_mesh_is_stationary():
    x, axes = affine_mesh()
    result = solve_mmpde(x, logical_axes=axes, periodic_image=translation)
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.positions, x)
    np.testing.assert_allclose(result.residual_history, [])
    assert np.all(result.minimum_jacobian_history > 0)


def test_affine_periodic_translation_mesh_is_stationary_with_default_axes():
    x, _ = affine_mesh()
    result = solve_mmpde(x, periodic_image=translation)
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.positions, x)
    np.testing.assert_allclose(result.residual_history, [])
    assert np.all(result.minimum_jacobian_history > 0)


def test_smooth_perturbation_relaxes_and_keeps_boundary_and_jacobians():
    x, axes = affine_mesh(7, 6, 8)
    rng = np.random.default_rng(1234)
    perturbation = rng.normal(scale=0.025, size=x.shape)
    interior = np.ones(x.shape[:3], dtype=bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    x[interior] += perturbation[interior]
    boundary = np.zeros(x.shape[:3], dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    result = solve_mmpde(
        x,
        logical_axes=axes,
        periodic_image=translation,
        options=MMPDEOptions(max_iterations=1000, tolerance=1e-9),
    )
    assert result.energy_history[-1] < result.energy_history[0]
    assert np.all(np.diff(result.energy_history) < 0)
    assert np.all(result.minimum_jacobian_history > 0)
    np.testing.assert_allclose(result.positions[boundary], x[boundary])


def test_full_spd_monitor_is_accepted():
    x, axes = affine_mesh()
    monitor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])
    result = solve_mmpde(x, logical_axes=axes, monitor=monitor, periodic_image=translation)
    assert result.converged


def test_projector_is_called_and_enforced():
    x, axes = affine_mesh()
    x[2, 2, :, 0] += 0.03
    calls = []
    projected = []

    def projector(candidate):
        calls.append(candidate.copy())
        candidate[1:-1, 1:-1, :, 2] += 1.0e-3
        projected.append(candidate.copy())
        return candidate

    result = solve_mmpde(
        x,
        logical_axes=axes,
        periodic_image=translation,
        projector=projector,
        options=MMPDEOptions(max_iterations=2),
    )
    assert calls
    np.testing.assert_allclose(result.positions[1:-1, 1:-1, :, 2], projected[-1][1:-1, 1:-1, :, 2])
    np.testing.assert_allclose(result.positions[0, :, :, 2], x[0, :, :, 2])


def test_candidate_validator_rejects_large_steps_and_preserves_invariants():
    x, axes = affine_mesh(7, 6, 8)
    rng = np.random.default_rng(1234)
    perturbation = rng.normal(scale=0.025, size=x.shape)
    interior = np.ones(x.shape[:3], dtype=bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    x[interior] += perturbation[interior]
    initial = x.copy()
    fixed = np.zeros(x.shape[:3], dtype=bool)
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    calls = []
    rejected = []
    accepted_states = []

    def validator(candidate):
        candidate = np.asarray(candidate)
        calls.append(candidate.copy())
        # This deliberately rejects the first, large descent step.  The
        # callback also verifies that fixed nodes have already been restored.
        np.testing.assert_allclose(candidate[fixed], initial[fixed])
        valid = bool(np.max(np.abs(candidate - initial)) <= 1.0e-3)
        if valid:
            accepted_states.append(candidate.copy())
        else:
            rejected.append(candidate.copy())
        return valid

    result = solve_mmpde(
        x,
        logical_axes=axes,
        periodic_image=translation,
        fixed_mask=fixed,
        candidate_validator=validator,
        options=MMPDEOptions(max_iterations=3, backtracking_factor=0.5),
    )
    assert len(calls) > 2
    assert rejected, "the validator should force at least one backtracking rejection"
    assert accepted_states
    # The initial mesh is validated too, so the accepted-state log contains
    # one more entry than the number of accepted relaxation steps.
    assert result.iterations + 1 == len(accepted_states)
    assert all(np.max(np.abs(state - initial)) <= 1.0e-3 for state in accepted_states)
    assert np.max(np.abs(result.positions - initial)) <= 1.0e-3
    np.testing.assert_allclose(result.positions[fixed], initial[fixed])
    assert np.all(np.diff(result.energy_history) < 0)
    assert np.all(result.minimum_jacobian_history > 0)


def test_candidate_validator_initial_mesh_and_return_contracts():
    x, axes = affine_mesh()
    with pytest.raises(ValueError, match="initial mesh rejected"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation, candidate_validator=lambda _: False)
    with pytest.raises(TypeError, match="must return a bool"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation, candidate_validator=lambda _: 1)

    def raises(_):
        raise RuntimeError("validator failure")

    with pytest.raises(RuntimeError, match="validator failure"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation, candidate_validator=raises)


def test_candidate_validator_rejects_malformed_candidate_result():
    x, axes = affine_mesh()
    x[2, 2, :, 0] += 0.03
    calls = 0

    def validator(candidate):
        nonlocal calls
        calls += 1
        return True if calls == 1 else "yes"

    with pytest.raises(TypeError, match="must return a bool"):
        solve_mmpde(
            x,
            logical_axes=axes,
            periodic_image=translation,
            candidate_validator=validator,
            options=MMPDEOptions(max_iterations=1),
        )


@pytest.mark.parametrize(
    "bad_monitor",
    [np.zeros((3, 3)), np.diag([1.0, -1.0, 1.0]), np.ones((2, 2))],
)
def test_malformed_or_non_spd_monitor_rejected(bad_monitor):
    x, axes = affine_mesh()
    with pytest.raises(ValueError):
        solve_mmpde(x, logical_axes=axes, monitor=bad_monitor, periodic_image=translation)


def test_malformed_axes_and_inverted_mesh_rejected():
    x, axes = affine_mesh()
    with pytest.raises(ValueError):
        solve_mmpde(x, logical_axes=(axes[0], axes[1][::-1], axes[2]), periodic_image=translation)
    inverted = x.copy()
    inverted[1:, :, :, 0] = inverted[:-1, :, :, 0]
    with pytest.raises(ValueError):
        solve_mmpde(inverted, logical_axes=axes, periodic_image=translation)


def test_periodic_eta_axis_must_be_uniform():
    x, axes = affine_mesh()
    nonuniform = axes[2].copy()
    nonuniform[2] += 0.03
    with pytest.raises(ValueError, match="uniformly spaced"):
        solve_mmpde(x, logical_axes=(axes[0], axes[1], nonuniform), periodic_image=translation)


def test_periodic_callback_must_be_rigid_affine():
    def nonlinear(points, turns):
        out = np.asarray(points).copy()
        out[..., 0] += turns * out[..., 1] ** 2
        return out

    with pytest.raises(ValueError, match="affine"):
        _infer_periodic_transform(nonlinear)


def test_rotational_wrap_gradient_and_nodal_monitor_pullback():
    x, axes = affine_mesh(4, 4, 5)
    monitor = np.broadcast_to(
        np.array([[4.0, 1.0, 0.2], [1.0, 2.5, 0.3], [0.2, 0.3, 1.5]]),
        x.shape[:3] + (3, 3),
    ).copy()
    transform = _infer_periodic_transform(rotation_periodic)
    edges = _edge_data(x, axes, monitor, monitor, transform, transform.rotation)
    seam = next(edge for edge in edges if edge[-1])
    expected_monitor = 0.5 * (
        monitor[3, 3, 4]
        + transform.rotation @ monitor[3, 3, 0] @ transform.rotation.T
    )
    np.testing.assert_allclose(seam[2], expected_monitor)

    fixed = np.zeros(x.shape[:3], dtype=bool)
    _, gradient = _energy_gradient(x, edges, fixed, transform, transform.rotation)
    node = (2, 2, 0)
    direction = np.array([0.31, -0.47, 0.29])
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    plus = x.copy()
    minus = x.copy()
    plus[node] += epsilon * direction
    minus[node] -= epsilon * direction
    energy_plus, _ = _energy_gradient(
        plus,
        _edge_data(plus, axes, monitor, monitor, transform, transform.rotation),
        fixed,
        transform,
        transform.rotation,
    )
    energy_minus, _ = _energy_gradient(
        minus,
        _edge_data(minus, axes, monitor, monitor, transform, transform.rotation),
        fixed,
        transform,
        transform.rotation,
    )
    finite_difference = (energy_plus - energy_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(finite_difference, gradient[node] @ direction, rtol=2e-7, atol=2e-8)
