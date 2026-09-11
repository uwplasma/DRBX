"""Second-order contracts for the production parallel material row.

These tests use the public row kernel directly.  They deliberately keep the
mapped model and the long wall/solver suites out of this file: the goal here
is to isolate spatial reconstruction, its diagnostics, and the explicit /
short-wall material split.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import drbx.native.fci_parallel_production_flux as flux
from drbx.native.fci_parallel_production_flux import (
    parallel_characteristic_matrix,
    parallel_short_wall_backward_euler,
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
)


TAU = 4.0
MU = 10.0


def _smooth_state(x):
    """A positive, genuinely five-field periodic material state."""

    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.stack(
        (
            2.0 + 0.10 * jnp.sin(x) + 0.03 * jnp.cos(2.0 * x),
            3.0 + 0.08 * jnp.cos(x) - 0.02 * jnp.sin(3.0 * x),
            5.0 + 0.12 * jnp.sin(2.0 * x + 0.2),
            0.20 + 0.05 * jnp.cos(2.0 * x) + 0.01 * jnp.sin(x),
            -0.10 + 0.04 * jnp.sin(3.0 * x) - 0.015 * jnp.cos(x),
        ),
        axis=-1,
    )


def _smooth_state_derivative(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.stack(
        (
            0.10 * jnp.cos(x) - 0.06 * jnp.sin(2.0 * x),
            -0.08 * jnp.sin(x) - 0.06 * jnp.cos(3.0 * x),
            0.24 * jnp.cos(2.0 * x + 0.2),
            -0.10 * jnp.sin(2.0 * x) + 0.01 * jnp.cos(x),
            0.12 * jnp.cos(3.0 * x) + 0.015 * jnp.sin(x),
        ),
        axis=-1,
    )


def _periodic_spacing(n: int, *, nonuniform: bool):
    computational = 2.0 * np.pi * np.arange(n, dtype=float) / n
    if nonuniform:
        # x'(xi)=1+0.14*cos(xi) stays positive, while the spacing is visibly
        # nonuniform and periodic across the wrap.
        x = computational + 0.14 * np.sin(computational)
    else:
        x = computational
    intervals = np.diff(np.concatenate((x, [x[0] + 2.0 * np.pi])))
    return (
        x,
        np.roll(intervals, 1),
        intervals,
        np.roll(intervals, 2),
        np.roll(intervals, -1),
    )


def _row_inputs(n: int, *, nonuniform: bool):
    x, dx_minus, dx_plus, dx_minus2, dx_plus2 = _periodic_spacing(
        n, nonuniform=nonuniform
    )
    states = np.asarray(_smooth_state(x))
    return (
        jnp.asarray(states),
        jnp.asarray(np.roll(states, 1, axis=0)),
        jnp.asarray(np.roll(states, -1, axis=0)),
        jnp.asarray(np.roll(states, 2, axis=0)),
        jnp.asarray(np.roll(states, -2, axis=0)),
        jnp.asarray(dx_minus),
        jnp.asarray(dx_plus),
        jnp.asarray(dx_minus2),
        jnp.asarray(dx_plus2),
        x,
    )


def _second_order_row(n: int, *, nonuniform: bool, tau: float = TAU, mu: float = MU):
    (
        center,
        minus,
        plus,
        minus2,
        plus2,
        dx_minus,
        dx_plus,
        dx_minus2,
        dx_plus2,
        x,
    ) = _row_inputs(n, nonuniform=nonuniform)
    residual, info = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        dx_minus,
        dx_plus,
        tau,
        mu,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dx_minus2,
        dx_plus2=dx_plus2,
        backward_second_valid=jnp.ones(n, dtype=bool),
        forward_second_valid=jnp.ones(n, dtype=bool),
        div_b=0.0,
    )
    matrix = jax.vmap(
        lambda state: parallel_characteristic_matrix(*state, tau=tau, mu=mu)
    )(center)
    exact = -jnp.einsum("nij,nj->ni", matrix, _smooth_state_derivative(x))
    return np.asarray(residual), np.asarray(exact), info


@pytest.mark.parametrize("nonuniform", (False, True))
def test_second_order_smooth_five_field_row_converges_on_periodic_spacing(
    nonuniform,
):
    errors = []
    for n in (32, 64, 128):
        actual, exact, info = _second_order_row(n, nonuniform=nonuniform)
        assert np.all(np.isfinite(actual))
        assert bool(np.all(np.asarray(info["ordinary_row"])))
        errors.append(np.linalg.norm(actual - exact) / np.linalg.norm(exact))
    observed = np.log2(errors[-2] / errors[-1])
    assert observed >= 1.8, (nonuniform, errors, observed)


def test_second_order_smooth_row_converges_at_physical_parameters():
    errors = []
    for n in (32, 64, 128):
        actual, exact, _ = _second_order_row(
            n, nonuniform=False, tau=1.0, mu=1836.0
        )
        assert np.all(np.isfinite(actual))
        errors.append(np.linalg.norm(actual - exact) / np.linalg.norm(exact))
    observed = np.log2(errors[-2] / errors[-1])
    assert observed >= 1.8, (errors, observed)


def _variable_b_row(n: int):
    x, dx_minus, dx_plus, dx_minus2, dx_plus2 = _periodic_spacing(
        n, nonuniform=False
    )
    h = float(dx_plus[0])
    center = jnp.asarray(_smooth_state(x))
    minus = jnp.roll(center, 1, axis=0)
    plus = jnp.roll(center, -1, axis=0)
    minus2 = jnp.roll(center, 2, axis=0)
    plus2 = jnp.roll(center, -2, axis=0)
    b = 1.0 + 0.2 * jnp.sin(jnp.asarray(x))
    b_minus = jnp.roll(b, 1)
    b_plus = jnp.roll(b, -1)
    # Raw metric source: B*d(1/B)/ds, with the same quadratic centered
    # derivative used by the material reconstruction and no owner projection.
    div_b = b * (1.0 / b_plus - 1.0 / b_minus) / (2.0 * h)
    actual, _ = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        jnp.asarray(dx_minus),
        jnp.asarray(dx_plus),
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=jnp.asarray(dx_minus2),
        dx_plus2=jnp.asarray(dx_plus2),
        backward_second_valid=jnp.ones(n, dtype=bool),
        forward_second_valid=jnp.ones(n, dtype=bool),
        div_b=div_b,
    )
    matrix = jax.vmap(
        lambda state: parallel_characteristic_matrix(*state, tau=TAU, mu=MU)
    )(center)
    transport = -jnp.einsum(
        "nij,nj->ni", matrix, _smooth_state_derivative(x)
    )
    density, te, ti, vi, ve = [center[:, i] for i in range(5)]
    current = density * (vi - ve)
    source = jnp.stack(
        (
            -density * ve * div_b,
            2.0 * te / (3.0 * density) * (0.71 * current - density * ve) * div_b,
            2.0 * ti / (3.0 * density) * (current - density * vi) * div_b,
            jnp.zeros_like(div_b),
            jnp.zeros_like(div_b),
        ),
        axis=-1,
    )
    return np.asarray(actual), np.asarray(transport + source)


def test_variable_b_raw_metric_source_converges_with_material_transport():
    errors = []
    for n in (32, 64, 128):
        actual, exact = _variable_b_row(n)
        errors.append(np.linalg.norm(actual - exact) / np.linalg.norm(exact))
    observed = np.log2(errors[-2] / errors[-1])
    assert observed >= 1.8, (errors, observed)


def test_constant_b_raw_metric_source_is_zero():
    n = 16
    center, minus, plus, minus2, plus2, dxm, dxp, dxm2, dxp2, _ = _row_inputs(
        n, nonuniform=False
    )
    div_b = jnp.ones(n) * (1.0 / 1.0 - 1.0 / 1.0) / (2.0 * dxp[0])
    actual, info = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        dxm,
        dxp,
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dxm2,
        dx_plus2=dxp2,
        backward_second_valid=jnp.ones(n, dtype=bool),
        forward_second_valid=jnp.ones(n, dtype=bool),
        div_b=div_b,
    )
    baseline, _ = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        dxm,
        dxp,
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dxm2,
        dx_plus2=dxp2,
        backward_second_valid=jnp.ones(n, dtype=bool),
        forward_second_valid=jnp.ones(n, dtype=bool),
        div_b=0.0,
    )
    np.testing.assert_array_equal(np.asarray(div_b), 0.0)
    np.testing.assert_allclose(actual, baseline, rtol=0.0, atol=0.0)
    assert bool(jnp.all(info["high_order_center_valid"]))


def test_constant_state_is_exact_null_and_source_is_order_independent():
    center = jnp.broadcast_to(
        jnp.asarray((2.0, 3.0, 5.0, 0.7, -0.2), dtype=jnp.float64), (4, 5)
    )
    kwargs = dict(
        minus2=center,
        plus2=center,
        dx_minus2=jnp.ones(4),
        dx_plus2=jnp.ones(4),
        backward_second_valid=jnp.ones(4, dtype=bool),
        forward_second_valid=jnp.ones(4, dtype=bool),
        div_b=0.0,
    )
    first, first_info = parallel_target_row_material_residual(
        center, center, center, jnp.ones(4), jnp.ones(4), TAU, MU
    )
    second, second_info = parallel_target_row_material_residual(
        center, center, center, jnp.ones(4), jnp.ones(4), TAU, MU,
        spatial_order=2, **kwargs
    )
    np.testing.assert_array_equal(first, 0.0)
    np.testing.assert_array_equal(second, 0.0)
    assert bool(jnp.all(~first_info["fallback"]))
    assert bool(jnp.all(~second_info["fallback"]))

    div_b = 0.37
    sourced, _ = parallel_target_row_material_residual(
        center, center, center, jnp.ones(4), jnp.ones(4), TAU, MU,
        spatial_order=2, div_b=div_b, **{k: v for k, v in kwargs.items() if k != "div_b"}
    )
    density, te, ti, vi, ve = (float(v) for v in center[0])
    current = density * (vi - ve)
    expected = np.asarray((
        -density * ve * div_b,
        2.0 * te / (3.0 * density) * (0.71 * current - density * ve) * div_b,
        2.0 * ti / (3.0 * density) * (current - density * vi) * div_b,
        0.0,
        0.0,
    ))
    np.testing.assert_allclose(
        np.asarray(sourced), np.broadcast_to(expected, (4, 5)), rtol=0.0, atol=2e-12
    )


def test_missing_second_points_fall_back_to_first_order_with_diagnostics():
    center, minus, plus, *_rest = _row_inputs(4, nonuniform=False)
    first, _ = parallel_target_row_material_residual(
        center, minus, plus, 1.0, 1.0, TAU, MU, div_b=0.0
    )
    fallback, info = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        1.0,
        1.0,
        TAU,
        MU,
        spatial_order=2,
        minus2=None,
        plus2=None,
        backward_second_valid=jnp.zeros(4, dtype=bool),
        forward_second_valid=jnp.zeros(4, dtype=bool),
        div_b=0.0,
    )
    np.testing.assert_allclose(fallback, first, rtol=0.0, atol=0.0)
    backward_flag = info.get("backward_high_order_fallback")
    forward_flag = info.get("forward_high_order_fallback")
    if backward_flag is None or forward_flag is None:
        aggregate = info.get("second_order_spectral_fallback")
        assert aggregate is not None, sorted(info)
        assert bool(jnp.all(aggregate))
    else:
        assert bool(jnp.all(backward_flag))
        assert bool(jnp.all(forward_flag))


def test_second_order_jit_is_batched_and_uses_validity_masks():
    data = _row_inputs(8, nonuniform=True)
    center, minus, plus, minus2, plus2, dxm, dxp, dxm2, dxp2, _x = data
    kernel = jax.jit(
        parallel_target_row_material_residual, static_argnames=("spatial_order",)
    )
    residual, info = kernel(
        center,
        minus,
        plus,
        dxm,
        dxp,
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dxm2,
        dx_plus2=dxp2,
        backward_second_valid=jnp.asarray(
            [True, True, True, True, False, True, True, True]
        ),
        forward_second_valid=jnp.ones(8, dtype=bool),
        div_b=jnp.zeros(8),
    )
    assert residual.shape == (8, 5)
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert bool(jnp.any(info["fallback"]))


def test_centered_closure_and_positivity_diagnostics_are_explicit():
    center, minus, plus, minus2, plus2, dxm, dxp, dxm2, dxp2, _x = _row_inputs(
        8, nonuniform=True
    )
    _, info = parallel_target_row_material_residual(
        center,
        minus,
        plus,
        dxm,
        dxp,
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dxm2,
        dx_plus2=dxp2,
        backward_second_valid=jnp.ones(8, dtype=bool),
        forward_second_valid=jnp.ones(8, dtype=bool),
        backward_centered_closure=jnp.ones(8, dtype=bool),
        forward_centered_closure=jnp.ones(8, dtype=bool),
        div_b=0.0,
    )
    assert bool(jnp.all(info["backward_centered_closure_used"]))
    assert bool(jnp.all(info["forward_centered_closure_used"]))
    assert bool(jnp.all(info["high_order_center_valid"]))

    invalid = center.at[2, 1].set(-1.0)
    residual, invalid_info = parallel_target_row_material_residual(
        invalid,
        minus,
        plus,
        dxm,
        dxp,
        TAU,
        MU,
        spatial_order=2,
        minus2=minus2,
        plus2=plus2,
        dx_minus2=dxm2,
        dx_plus2=dxp2,
        backward_second_valid=jnp.ones(8, dtype=bool),
        forward_second_valid=jnp.ones(8, dtype=bool),
        div_b=0.0,
    )
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert not bool(invalid_info["high_order_center_valid"][2])
    assert bool(invalid_info["backward_high_order_fallback"][2])
    assert bool(invalid_info["forward_high_order_fallback"][2])


def test_wall_center_reconstruction_mismatch_is_reported_and_falls_back():
    center = jnp.asarray([[2.0, 3.0, 5.0, 0.2, -0.1]], dtype=jnp.float64)
    wall = center + jnp.asarray([[-0.1, 0.02, 0.03, 0.01, -0.02]])
    plus = center + jnp.asarray([[0.01, -0.02, 0.01, -0.01, 0.02]])
    resolved = flux._material_directional_data(
        center,
        wall,
        plus,
        0.01,
        0.2,
        TAU,
        MU,
        backward_wall=jnp.ones(1, dtype=bool),
        forward_wall=jnp.zeros(1, dtype=bool),
        backward_wall_state=wall,
        parallel_characteristic_wall_law="physical-boundary-state",
    )[-1]
    resolved = dict(resolved)
    resolved["center"] = resolved["center"] + jnp.asarray(
        [[0.01, 0.0, 0.0, 0.0, 0.0]]
    )
    _, info = parallel_target_row_material_residual(
        center,
        wall,
        plus,
        0.01,
        0.2,
        TAU,
        MU,
        backward_wall=jnp.ones(1, dtype=bool),
        forward_wall=jnp.zeros(1, dtype=bool),
        backward_wall_state=wall,
        parallel_characteristic_wall_law="physical-boundary-state",
        spatial_order=2,
        minus2=wall,
        plus2=plus,
        dx_minus2=0.01,
        dx_plus2=0.2,
        backward_second_valid=False,
        forward_second_valid=True,
        resolved_wall_data=resolved,
        div_b=0.0,
    )
    assert bool(info["wall_center_reconstruction_mismatch"][0])
    assert bool(info["backward_high_order_fallback"][0])


def test_constant_coefficient_eigenmode_polynomial_exactness(monkeypatch):
    """Scalar/eigenmode contract with the production matrix frozen explicitly."""

    reference = jnp.asarray((2.0, 3.0, 5.0, 0.7, -0.2), dtype=jnp.float64)
    matrix = np.asarray(parallel_characteristic_matrix(*reference, tau=TAU, mu=MU))
    values, vectors = np.linalg.eig(matrix)
    mode = np.real(vectors[:, np.argmax(np.abs(np.real(values)))])
    mode /= np.linalg.norm(mode)
    x = np.asarray((-0.6, -0.1, 0.4, 0.9))
    h = 0.5
    polynomial = 0.02 * x**2 + 0.10 * x + 0.10
    derivative = 0.04 * x + 0.10
    center = reference[None, :] + jnp.asarray(polynomial[:, None] * mode)
    value = lambda coordinate: 0.02 * coordinate**2 + 0.10 * coordinate + 0.10
    minus = reference[None, :] + jnp.asarray(value(x - h)[:, None] * mode)
    plus = reference[None, :] + jnp.asarray(value(x + h)[:, None] * mode)
    minus2 = reference[None, :] + jnp.asarray(value(x - 2.0 * h)[:, None] * mode)
    plus2 = reference[None, :] + jnp.asarray(value(x + 2.0 * h)[:, None] * mode)

    original = flux.parallel_matrix_from_state
    monkeypatch.setattr(
        flux,
        "parallel_matrix_from_state",
        lambda state, tau, mu: jnp.broadcast_to(jnp.asarray(matrix), jnp.asarray(state).shape[:-1] + (5, 5)),
    )
    try:
        actual, _ = parallel_target_row_material_residual(
            center,
            minus,
            plus,
            h,
            h,
            TAU,
            MU,
            spatial_order=2,
            minus2=minus2,
            plus2=plus2,
            dx_minus2=h,
            dx_plus2=h,
            backward_second_valid=jnp.ones(4, dtype=bool),
            forward_second_valid=jnp.ones(4, dtype=bool),
            div_b=0.0,
        )
    finally:
        monkeypatch.setattr(flux, "parallel_matrix_from_state", original)
    expected = -jnp.asarray(derivative[:, None] * (matrix @ mode)[None, :])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)


def test_wall_second_order_explicit_plus_first_order_implicit_base_is_exact():
    """Physical-wall correction remains explicit while its stiff base is selected."""

    n = 2
    center = jnp.asarray(
        [[2.05, 3.02, 4.95, 0.22, -0.11], [1.96, 2.94, 5.08, 0.18, -0.07]],
        dtype=jnp.float64,
    )
    wall = center + jnp.asarray(
        [[-0.08, 0.04, -0.03, 0.12, -0.05], [0.05, -0.03, 0.02, -0.09, 0.04]],
        dtype=jnp.float64,
    )
    plus = center + jnp.asarray(
        [[0.02, -0.01, 0.03, -0.04, 0.02], [-0.03, 0.02, -0.02, 0.03, -0.01]],
        dtype=jnp.float64,
    )
    plus2 = center + 2.0 * (plus - center)
    dx_minus = jnp.asarray((0.003, 0.009))
    dx_plus = jnp.asarray((0.11, 0.17))
    common = dict(
        backward_wall=jnp.ones(n, dtype=bool),
        forward_wall=jnp.zeros(n, dtype=bool),
        backward_wall_state=wall,
        parallel_characteristic_wall_law="physical-boundary-state",
        spatial_order=2,
        minus2=wall,
        plus2=plus2,
        dx_minus2=dx_minus,
        dx_plus2=dx_plus,
        backward_second_valid=jnp.zeros(n, dtype=bool),
        forward_second_valid=jnp.ones(n, dtype=bool),
        div_b=0.0,
    )
    direct, _ = parallel_target_row_material_residual(
        center, wall, plus, dx_minus, dx_plus, TAU, MU,
        selection_dt=0.0,
        parallel_short_leg_selection="cfl",
        cfl_limit=1.0e12,
        **common,
    )
    explicit, explicit_info = parallel_target_row_material_residual(
        center, wall, plus, dx_minus, dx_plus, TAU, MU,
        selection_dt=0.0,
        parallel_short_leg_selection="all-physical-walls",
        **common,
    )
    base, base_jacobian, base_info = parallel_short_wall_material_data(
        center, wall, plus, dx_minus, dx_plus, TAU, MU,
        selection_dt=0.0,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=common["backward_wall"],
        forward_wall=common["forward_wall"],
        backward_wall_state=wall,
        parallel_characteristic_wall_law="physical-boundary-state",
    )
    np.testing.assert_allclose(direct, explicit + base, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(explicit_info["selected_wall"], True)
    np.testing.assert_array_equal(base_info["selected_wall"], True)
    assert bool(jnp.all(jnp.isfinite(base_jacobian)))

    updated, increment, be_info = parallel_short_wall_backward_euler(
        center, wall, plus, dx_minus, dx_plus, TAU, MU,
        selection_dt=0.0,
        solve_dt=jnp.asarray((1.0e-8, 2.0e-8)),
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=common["backward_wall"],
        forward_wall=common["forward_wall"],
        backward_wall_state=wall,
        parallel_characteristic_wall_law="physical-boundary-state",
    )
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert bool(jnp.all(jnp.isfinite(increment)))
    assert bool(jnp.all(be_info["selected_wall"]))
