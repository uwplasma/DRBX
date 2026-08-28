"""Contract tests for the standalone production curvature flux primitives."""

from pathlib import Path
import sys
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_MODULE_PATH = ROOT / "src" / "drbx" / "native" / "fci_curvature_production_flux.py"
_SPEC = importlib.util.spec_from_file_location("_production_curvature_flux", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
curvature_characteristic_absolute_action = _MODULE.curvature_characteristic_absolute_action
curvature_characteristic_absolute_matrix = _MODULE.curvature_characteristic_absolute_matrix
curvature_characteristic_metric = _MODULE.curvature_characteristic_metric
curvature_face_linearized_fluctuations = _MODULE.curvature_face_linearized_fluctuations
curvature_principal_matrix = _MODULE.curvature_principal_matrix
reconstruct_first_order_face_states = _MODULE.reconstruct_first_order_face_states
reconstruct_third_order_face_states = _MODULE.reconstruct_third_order_face_states


def _state():
    return jnp.asarray((1.1, 0.9, 1.2, 0.07), dtype=jnp.float64)


def test_corrected_strict_matrix_entries_and_ti_polarization_column():
    matrix = np.asarray(curvature_principal_matrix(1.1, 0.9, 1.2, 1.3, 0.8))
    expected = np.array(
        [
            [1.8, 2.2, 2.2 * 0.8, 0.0],
            [4 * 0.9**2 / (3 * 1.1), 14 * 0.9 / 3, 4 * 0.8 * 0.9 / 3, 0.0],
            [4 * 1.2 * 0.9 / (3 * 1.1), 4 * 1.2 / 3, -2 * 0.8 * 1.2, 0.0],
            [2 * 1.3**2 * (0.9 + 0.8 * 1.2) / 1.1, 2 * 1.3**2, 2 * 0.8 * 1.3**2, 0.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=2e-14)
    assert abs(matrix[0, 2]) > 0.0
    assert abs(matrix[2, 2]) > 0.0


def test_face_linearized_constant_null_and_identical_state_consistency():
    state = jnp.asarray((1.1, 0.9, 1.2, 0.07), dtype=jnp.float64)
    plus, minus, fallback = curvature_face_linearized_fluctuations(
        state, state, state, 1.3, 0.8, return_fallback=True
    )
    np.testing.assert_allclose(plus, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(minus, 0.0, atol=0.0, rtol=0.0)
    assert not bool(fallback)


def test_face_linearized_split_sums_to_face_matrix_action():
    left = jnp.asarray((1.0, 1.0, 1.0, -0.4), dtype=jnp.float64)
    right = jnp.asarray((1.2, 0.8, 1.1, 0.8), dtype=jnp.float64)
    face = jnp.asarray((1.1, 0.9, 1.05, 0.0), dtype=jnp.float64)
    plus, minus = curvature_face_linearized_fluctuations(
        left, right, face, 1.2, 0.7
    )
    matrix = curvature_principal_matrix(*face[:3], 1.2, 0.7)
    np.testing.assert_allclose(
        plus + minus, matrix @ (right - left), atol=3e-12, rtol=3e-12
    )


def test_face_linearized_uses_face_state_and_wall_interior_trace():
    left = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=jnp.float64)
    right = jnp.asarray((1.2, 0.8, 1.1, 0.3), dtype=jnp.float64)
    ordinary = jnp.asarray((1.1, 0.9, 1.05, 0.1), dtype=jnp.float64)
    changed = ordinary.at[0].set(2.0)
    ordinary_result = curvature_face_linearized_fluctuations(
        left, right, ordinary, 1.0, 0.7
    )
    changed_result = curvature_face_linearized_fluctuations(
        left, right, changed, 1.0, 0.7
    )
    assert float(jnp.linalg.norm(ordinary_result[0] - changed_result[0])) > 1.0e-8
    # A wall jump uses exterior on the left but the interior right trace for A;
    # changing the exterior state therefore must not alter the matrix result.
    interior = right
    wall_a = curvature_face_linearized_fluctuations(
        left, right, interior, 1.0, 0.7
    )
    exterior_a = curvature_face_linearized_fluctuations(
        left, right, left, 1.0, 0.7
    )
    assert float(jnp.linalg.norm(wall_a[0] - exterior_a[0])) > 1.0e-8


def test_face_linearized_batched_jit_and_finite_fallback():
    left = jnp.asarray(((1.0, 1.0, 1.0, 0.0), (1.1, 0.9, 1.2, 0.1)), dtype=jnp.float64)
    right = jnp.asarray(((1.1, 0.9, 1.2, 0.2), (1.0, 1.0, 1.0, -0.2)), dtype=jnp.float64)
    face = jnp.asarray(((1.05, 0.95, 1.1, 0.0), (jnp.nan, 1.0, 1.0, 0.0)))
    result = jax.jit(curvature_face_linearized_fluctuations)(
        left, right, face, jnp.asarray((1.0, 1.2)), 0.7
    )
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in result)
    _, _, fallback = curvature_face_linearized_fluctuations(
        left, right, face, jnp.asarray((1.0, 1.2)), 0.7, return_fallback=True
    )
    assert bool(fallback[1])


def test_characteristic_action_and_symmetrized_power_are_nonnegative():
    for n, te, ti, b, tau in ((1.0, 1.0, 1.0, 1.1, 0.7), (1.1, 0.9, 1.2, 1.3, 0.8), (0.998, 0.995, 1.001, 1.24, 0.7)):
        matrix = curvature_principal_matrix(n, te, ti, b, tau)
        jump = jnp.asarray((0.2, -0.1, 0.3, 0.15))
        absolute = curvature_characteristic_absolute_matrix(matrix)
        metric = curvature_characteristic_metric(matrix)
        ha = metric @ absolute
        np.testing.assert_allclose(ha, ha.T, atol=2e-11, rtol=2e-11)
        sym_power = jump @ (0.5 * (ha + ha.T)) @ jump
        assert float(sym_power) >= -2.0e-12
        assert float(jnp.min(jnp.linalg.eigvalsh(0.5 * (ha + ha.T)))) >= -2.0e-11
        action = curvature_characteristic_absolute_action(matrix, jump)
        np.testing.assert_allclose(action, absolute @ jump, atol=2e-11, rtol=2e-11)


def test_constant_background_q_over_b_normal_has_expected_symbol_scaling():
    """Owner measure raw/B requires Q/B, not Q, in the material symbol."""
    matrix = curvature_principal_matrix(1.05, 0.95, 1.1, 1.7, 0.8)
    jump = jnp.asarray((0.2, -0.1, 0.15, 0.0), dtype=jnp.float64)
    qface, bface = 2.4, 1.7
    unscaled = curvature_characteristic_absolute_action(matrix, jump)
    scaled = curvature_characteristic_absolute_action(
        (qface / bface) * matrix, jump
    )
    # For a positive scalar normal, the characteristic basis is unchanged and
    # every wave speed/action scales by Q/B exactly.
    np.testing.assert_allclose(
        scaled, (qface / bface) * unscaled, atol=2.0e-11, rtol=2.0e-11
    )


def test_batched_jit_and_ill_conditioned_fallback():
    states = jnp.stack((_state(), _state() * jnp.asarray((1.02, 0.98, 1.01, 1.0))))
    matrices = curvature_principal_matrix(
        states[:, 0], states[:, 1], states[:, 2], jnp.asarray((0.8, 1.3)), 0.7
    )
    vectors = jnp.asarray(((0.1, 0.2, -0.1, 0.3), (0.2, -0.1, 0.4, 0.1)))
    eager = curvature_characteristic_absolute_action(matrices, vectors)
    compiled = jax.jit(curvature_characteristic_absolute_action)(matrices, vectors)
    np.testing.assert_allclose(compiled, eager, atol=3e-11, rtol=3e-11)
    bad = jnp.asarray(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 2, 0], [0, 0, 0, -3]],
        dtype=jnp.float64,
    )
    action, used_fallback = curvature_characteristic_absolute_action(
        bad, vectors[0], return_fallback=True
    )
    assert bool(used_fallback)
    assert np.all(np.isfinite(np.asarray(action)))
    assert float(jnp.linalg.norm(action)) > 0.0


def test_reconstruction_orders_and_positivity_fallback_metadata():
    q0 = jnp.asarray((1.0, 1.0, 1.0, 0.0))
    q1 = jnp.asarray((1.1, 1.05, 0.95, 0.2))
    first_left, first_right, first_meta = reconstruct_first_order_face_states(q0, q1)
    assert np.all(np.asarray(first_meta.order_used) == 1)
    np.testing.assert_allclose(first_left, q0)
    third_left, third_right, meta = reconstruct_third_order_face_states(q0, q1, q1, q0)
    assert np.all(np.asarray(meta.order_used) == 3)
    assert not np.any(np.asarray(meta.used_fallback))
    bad = jnp.asarray((10.0, 10.0, 10.0, 0.0))
    _left, _right, bad_meta = reconstruct_third_order_face_states(bad, q0, q1, q1)
    assert np.any(np.asarray(bad_meta.used_fallback))
    assert np.all(np.asarray(_left)[..., :3] > 0.0)


def test_completed_run_state_range_is_admissible_when_available():
    path = Path(
        "/Users/yxie/Desktop/HSX drbx/prototype_runs/"
        "fci_curvature_radial_poloidal_third_order_upwind_32_t015/"
        "hsx_curvature_radial_poloidal_third_order_upwind_32_t015.npz"
    )
    if not path.exists():
        pytest.skip("completed 32^3 history is not present on this checkout")
    data = np.load(path)
    names = ("density", "Te", "Ti", "vorticity")
    if not all(name in data for name in names):
        pytest.skip("history does not expose curvature state fields")
    # A few hundred evenly spaced final-time cells keep the test cheap while
    # exercising the recorded run's actual primitive range.
    values = jnp.stack(
        tuple(jnp.asarray(data[name][-1]).reshape(-1)[::1024][:512] for name in names),
        axis=-1,
    )
    assert bool(jnp.all(values[..., :3] > 0.0))
    matrices = curvature_principal_matrix(
        values[:, 0], values[:, 1], values[:, 2], 1.0, 1.0
    )
    action, fallback = curvature_characteristic_absolute_action(
        matrices, values, return_fallback=True
    )
    assert bool(jnp.all(jnp.isfinite(action)))
    assert bool(jnp.all(jnp.isfinite(matrices)))
    assert bool(jnp.all(~fallback))
