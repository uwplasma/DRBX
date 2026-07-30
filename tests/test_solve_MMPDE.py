"""Focused tests for the generic structured D2 x S1 MMPDE baseline."""

import numpy as np
import pytest

from drbx.geometry.solve_MMPDE import (
    MMPDEOptions,
    _cell_data,
    _cell_quality_energy_gradient,
    _cell_jacobians,
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


def test_scale_aware_step_cap_limits_pre_projection_candidate():
    x, axes = affine_mesh(6, 5, 7)
    x[2, 2, :, 0] += 0.04
    captured = []

    def projector(candidate):
        captured.append(np.asarray(candidate).copy())
        return candidate

    fraction = 0.2
    result = solve_mmpde(
        x,
        logical_axes=axes,
        periodic_image=translation,
        projector=projector,
        options=MMPDEOptions(
            max_iterations=1,
            initial_step=10.0,
            maximum_step_fraction=fraction,
        ),
    )
    assert captured
    _, matrices, _ = _cell_data(
        x, axes, np.broadcast_to(np.eye(3), x.shape[:3] + (3, 3)),
        _infer_periodic_transform(translation), np.eye(3),
    )
    characteristic_length = np.median(np.linalg.norm(matrices, axis=2))
    free = np.ones(x.shape[:3], dtype=bool)
    free[[0, -1], :, :] = False
    free[:, [0, -1], :] = False
    displacement = np.linalg.norm(captured[0][free] - x[free], axis=-1)
    assert np.max(displacement) <= fraction * characteristic_length * (1.0 + 1.0e-10)
    assert result.positions.shape == x.shape


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


def test_composite_energy_option_validation():
    with pytest.raises(ValueError, match="weights"):
        solve_mmpde(affine_mesh()[0], periodic_image=translation,
                    options=MMPDEOptions(alignment_weight=-1.0))
    x, axes = affine_mesh()
    with pytest.raises(ValueError, match="weights"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation,
                    options=MMPDEOptions(volume_smoothness_weight=np.inf))
    with pytest.raises(ValueError, match="barrier_power"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation,
                    options=MMPDEOptions(barrier_power=0.0))
    with pytest.raises(ValueError, match="maximum_step_fraction"):
        solve_mmpde(x, logical_axes=axes, periodic_image=translation,
                    options=MMPDEOptions(maximum_step_fraction=np.inf))


def _rotating_torus_mesh():
    nu, nv, neta = 4, 4, 5
    u = np.linspace(0.2, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    eta = np.linspace(0.0, 1.0, neta, endpoint=False)
    uu, vv, ee = np.meshgrid(u, v, eta, indexing="ij")
    theta = 0.5 * np.pi * ee
    rho = 0.3 * uu
    x = np.stack(
        (
            (2.0 + rho * np.cos(2.0 * np.pi * vv)) * np.cos(theta),
            (2.0 + rho * np.cos(2.0 * np.pi * vv)) * np.sin(theta),
            -rho * np.sin(2.0 * np.pi * vv),
        ),
        axis=-1,
    )
    return x, (u, v, eta)


def test_composite_gradient_includes_rotational_seam_and_spd_monitor():
    x, axes = _rotating_torus_mesh()
    monitor = np.array([[4.0, 1.0, 0.2], [1.0, 2.5, 0.3], [0.2, 0.3, 1.5]])
    monitor_nodes = np.broadcast_to(monitor, x.shape[:3] + (3, 3)).copy()
    transform = _infer_periodic_transform(rotation_periodic)
    fixed = np.zeros(x.shape[:3], dtype=bool)
    edges = _edge_data(x, axes, monitor_nodes, monitor, transform, transform.rotation)
    _, matrices, cell_monitors = _cell_data(x, axes, monitor_nodes, transform, transform.rotation)
    vbar = np.mean([np.linalg.det(F) * np.sqrt(np.linalg.det(M))
                    for F, M in zip(matrices, cell_monitors)])
    initial_edge_energy, _ = _energy_gradient(x, edges, fixed, transform, transform.rotation)
    options = MMPDEOptions(
        dirichlet_weight=0.3, alignment_weight=0.4, equidistribution_weight=0.5,
        jacobian_barrier_weight=0.2, volume_smoothness_weight=0.7,
    )
    energy, gradient, _ = _cell_quality_energy_gradient(
        x, axes, fixed, monitor_nodes, transform, transform.rotation, options,
        vbar, initial_edge_energy, edges,
    )
    assert np.isfinite(energy)
    node = (1, 2, 0)
    direction = np.array([0.31, -0.47, 0.29])
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    plus, minus = x.copy(), x.copy()
    plus[node] += epsilon * direction
    minus[node] -= epsilon * direction
    energy_plus = _cell_quality_energy_gradient(
        plus, axes, fixed, monitor_nodes, transform, transform.rotation, options,
        vbar, initial_edge_energy, edges,
    )[0]
    energy_minus = _cell_quality_energy_gradient(
        minus, axes, fixed, monitor_nodes, transform, transform.rotation, options,
        vbar, initial_edge_energy, edges,
    )[0]
    finite_difference = (energy_plus - energy_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(finite_difference, gradient[node] @ direction,
                               rtol=3e-6, atol=3e-7)


def test_vectorized_cell_data_and_jacobians_match_scalar_reference():
    x, axes = _rotating_torus_mesh()
    rng = np.random.default_rng(22)
    raw = rng.normal(size=x.shape[:3] + (3, 3))
    monitor_nodes = np.einsum("...ia,...ja->...ij", raw, raw) + 0.5 * np.eye(3)
    transform = _infer_periodic_transform(rotation_periodic)
    cells, matrices, monitors = _cell_data(
        x, axes, monitor_nodes, transform, transform.rotation
    )
    reference_matrices = []
    reference_monitors = []
    reference_jacobians = []
    du, dv = np.diff(axes[0]), np.diff(axes[1])
    deta = axes[2][-1] - axes[2][-2]
    for i, j, k in cells:
        p = x[i, j, k]
        q2 = x[i, j, k + 1] if k + 1 < x.shape[2] else transform(x[i, j, 0][None, :], 1)[0]
        reference_matrices.append(np.column_stack((x[i + 1, j, k] - p, x[i, j + 1, k] - p, q2 - p)))
        values = [monitor_nodes[i, j, k], monitor_nodes[i + 1, j, k],
                  monitor_nodes[i, j + 1, k], monitor_nodes[i + 1, j + 1, k]]
        if k + 1 < x.shape[2]:
            values.extend([monitor_nodes[i, j, k + 1], monitor_nodes[i + 1, j, k + 1],
                           monitor_nodes[i, j + 1, k + 1], monitor_nodes[i + 1, j + 1, k + 1]])
        else:
            values.extend([transform.rotation @ monitor_nodes[ii, jj, 0] @ transform.rotation.T
                           for ii, jj in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))])
        reference_monitors.append(np.mean(values, axis=0))
        a = (x[i + 1, j, k] - p) / du[i]
        b = (x[i, j + 1, k] - p) / dv[j]
        c = (q2 - p) / deta
        reference_jacobians.append(np.linalg.det(np.column_stack((a, b, c))))
    np.testing.assert_allclose(matrices, reference_matrices)
    np.testing.assert_allclose(monitors, reference_monitors)
    np.testing.assert_allclose(_cell_jacobians(x, axes, transform),
                               np.asarray(reference_jacobians).reshape(x.shape[0] - 1, x.shape[1] - 1, x.shape[2]))


def test_composite_history_and_quality_improve_with_fixed_boundary():
    x, axes = affine_mesh(7, 6, 8)
    rng = np.random.default_rng(1234)
    perturbation = rng.normal(scale=0.025, size=x.shape)
    interior = np.ones(x.shape[:3], dtype=bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    x[interior] += perturbation[interior]
    fixed = ~interior

    def scaled_jacobian_min(mesh):
        _, matrices, _ = _cell_data(
            mesh, axes, np.broadcast_to(np.eye(3), mesh.shape[:3] + (3, 3)),
            _infer_periodic_transform(translation), np.eye(3),
        )
        return min(np.linalg.det(F) / np.prod(np.linalg.norm(F, axis=0)) for F in matrices)

    initial_quality = scaled_jacobian_min(x)
    result = solve_mmpde(
        x, logical_axes=axes, periodic_image=translation, fixed_mask=fixed,
        options=MMPDEOptions(max_iterations=100, tolerance=1.0e-6),
    )
    assert result.energy_history[-1] < result.energy_history[0]
    assert np.all(np.diff(result.energy_history) < 0.0)
    assert scaled_jacobian_min(result.positions) > initial_quality
    assert np.all(result.minimum_jacobian_history > 0.0)
    np.testing.assert_allclose(result.positions[fixed], x[fixed])
    expected = {"dirichlet", "alignment", "equidistribution", "jacobian_barrier",
                "volume_smoothness", "total"}
    assert expected <= set(result.component_energy_history)
    for values in result.component_energy_history.values():
        assert values.size == result.energy_history.size


def test_dirichlet_only_mode_retains_legacy_energy_path():
    x, axes = affine_mesh(6, 5, 7)
    x[2, 2, :, 0] += 0.02
    options = MMPDEOptions(
        max_iterations=4, dirichlet_weight=1.0, alignment_weight=0.0,
        equidistribution_weight=0.0, jacobian_barrier_weight=0.0,
        volume_smoothness_weight=0.0,
    )
    result = solve_mmpde(x, logical_axes=axes, periodic_image=translation, options=options)
    np.testing.assert_allclose(result.energy_history,
                               result.component_energy_history["dirichlet"])
    for name in ("alignment", "equidistribution", "jacobian_barrier", "volume_smoothness"):
        assert np.all(np.isfinite(result.component_energy_history[name]))


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
    seam = edges.seam
    expected_monitor = 0.5 * (
        monitor[3, 3, 4]
        + transform.rotation @ monitor[3, 3, 0] @ transform.rotation.T
    )
    expected_monitor = np.broadcast_to(expected_monitor, seam.monitor.shape)
    np.testing.assert_allclose(seam.monitor, expected_monitor, atol=1e-15)

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


def _reference_edge_energy_gradient(
    positions, axes, monitor, fixed, periodic_image, periodic_rotation
):
    """Literal structured-edge reference used to check the batched kernel."""

    shape = positions.shape[:3]
    du, dv, deta = (np.diff(axis) for axis in axes)
    u_node_spacing = np.concatenate((du, du[-1:]))
    v_node_spacing = np.concatenate((dv, dv[-1:]))
    eta_node_spacing = np.concatenate((deta, deta[-1:]))
    gradient = np.zeros_like(positions)
    energy = 0.0

    def monitor_at(point):
        if callable(monitor):
            return np.asarray(monitor(np.asarray(point)[None, :]), dtype=float)[0]
        values = np.asarray(monitor)
        if values.shape == (3, 3):
            return values
        return values[point_index]

    def add(pidx, qidx, coefficient, wraps=False):
        nonlocal energy
        p = positions[pidx]
        q = positions[qidx]
        if wraps:
            q = np.asarray(periodic_image(q[None, :], 1), dtype=float)[0]
        difference = q - p
        point_index = pidx
        if not callable(monitor) and np.asarray(monitor).ndim == 5:
            mm = 0.5 * (np.asarray(monitor)[pidx] + np.asarray(monitor)[qidx])
            if wraps:
                mm = 0.5 * (
                    np.asarray(monitor)[pidx]
                    + periodic_rotation @ np.asarray(monitor)[qidx] @ periodic_rotation.T
                )
        else:
            mm = monitor_at(0.5 * (p + q))
        edge_gradient = coefficient * (mm @ difference)
        energy += 0.5 * coefficient * (difference @ mm @ difference)
        gradient[pidx] -= edge_gradient
        gradient[qidx] += periodic_rotation.T @ edge_gradient if wraps else edge_gradient

    for i in range(shape[0] - 1):
        for j in range(shape[1]):
            for k in range(shape[2]):
                add((i, j, k), (i + 1, j, k), v_node_spacing[j] * eta_node_spacing[k] / du[i])
    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            for k in range(shape[2]):
                add((i, j, k), (i, j + 1, k), u_node_spacing[i] * eta_node_spacing[k] / dv[j])
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2] - 1):
                add(
                    (i, j, k), (i, j, k + 1),
                    u_node_spacing[i] * v_node_spacing[j] / deta[k],
                )
            add(
                (i, j, shape[2] - 1), (i, j, 0),
                u_node_spacing[i] * v_node_spacing[j] / deta[-1], wraps=True,
            )
    gradient[fixed] = 0.0
    return energy, gradient


def test_vectorized_edge_families_match_explicit_reference_for_static_anisotropic_monitor():
    x, _ = _rotating_torus_mesh()
    axes = (
        np.array([0.1, 0.27, 0.61, 1.0]),
        np.array([0.0, 0.18, 0.52, 1.0]),
        np.linspace(0.0, 1.0, x.shape[2], endpoint=False),
    )
    # Rebuild the mesh on the nonuniform logical axes while retaining a
    # genuinely three-dimensional, rotating periodic seam.
    uu, vv, ee = np.meshgrid(*axes, indexing="ij")
    theta = 0.5 * np.pi * ee
    rho = 0.3 * uu
    x = np.stack(
        (
            (2.0 + rho * np.cos(2.0 * np.pi * vv)) * np.cos(theta),
            (2.0 + rho * np.cos(2.0 * np.pi * vv)) * np.sin(theta),
            -rho * np.sin(2.0 * np.pi * vv),
        ), axis=-1,
    )
    rng = np.random.default_rng(42)
    raw = rng.normal(size=x.shape[:3] + (3, 3))
    monitor = np.einsum("...ia,...ja->...ij", raw, raw) + 0.7 * np.eye(3)
    fixed = np.zeros(x.shape[:3], dtype=bool)
    fixed[0, :, :] = True
    fixed[:, -1, :] = True
    transform = _infer_periodic_transform(rotation_periodic)
    edges = _edge_data(x, axes, monitor, monitor, transform, transform.rotation)
    actual = _energy_gradient(x, edges, fixed, transform, transform.rotation)
    expected = _reference_edge_energy_gradient(
        x, axes, monitor, fixed, transform, transform.rotation
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)
    assert all(family.p.ndim == 1 for family in edges)
    assert edges.seam.wraps


def test_vectorized_edge_families_batch_callable_monitor_and_preserve_fixed_nodes():
    x, axes = _rotating_torus_mesh()
    calls = []

    def callable_monitor(points):
        points = np.asarray(points)
        calls.append(points.shape[0])
        values = np.empty(points.shape[:-1] + (3, 3))
        values[...] = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.15], [0.1, 0.15, 1.7]])
        values[..., 0, 0] += 0.1 * points[..., 0] ** 2
        return values

    fixed = np.zeros(x.shape[:3], dtype=bool)
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    transform = _infer_periodic_transform(rotation_periodic)
    edges = _edge_data(x, axes, None, callable_monitor, transform, transform.rotation)
    batched_calls = calls.copy()
    actual = _energy_gradient(x, edges, fixed, transform, transform.rotation)
    calls.clear()
    expected = _reference_edge_energy_gradient(
        x, axes, callable_monitor, fixed, transform, transform.rotation
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)
    # One callback per family, rather than one callback per edge.
    assert batched_calls == [3 * 4 * 5, 4 * 3 * 5, 4 * 4 * 4, 4 * 4]
    np.testing.assert_allclose(actual[1][fixed], 0.0)
