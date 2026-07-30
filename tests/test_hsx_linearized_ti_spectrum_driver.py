"""Cheap tests for the matrix-free HSX curvature spectrum driver."""

from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "DRBX" / "src"))
sys.path.insert(0, str(WORKSPACE))
for _module_name in list(sys.modules):
    if _module_name == "drbx" or _module_name.startswith("drbx."):
        del sys.modules[_module_name]

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator

from analyze_hsx_linearized_ti_curvature_spectrum import (  # noqa: E402
    _make_linear_operators,
    _weighted_operators,
    build_parser,
)


def test_matrix_free_wrapper_and_weighted_abscissa_match_dense():
    dense = np.array([[1.0, 2.0, 0.0], [-3.0, -4.0, 1.0], [0.5, 0.0, 2.0]])
    weights = np.array([1.0, 4.0, 9.0])
    stats = type("Stats", (), {"matvec": 0, "rmatvec": 0, "elapsed_matvec": 0.0, "elapsed_rmatvec": 0.0})()
    apply = lambda x: dense @ np.asarray(x)
    adjoint = lambda x: dense.T @ np.asarray(x)
    L, _, _ = _make_linear_operators(apply, adjoint, weights, stats)
    A, H = _weighted_operators(apply, adjoint, weights, stats)
    expected_a = np.diag(np.sqrt(weights)) @ dense @ np.diag(1.0 / np.sqrt(weights))
    expected_h = 0.5 * (expected_a + expected_a.T)
    actual_a = np.column_stack([A @ np.eye(3)[:, index] for index in range(3)])
    actual_h = np.column_stack([H @ np.eye(3)[:, index] for index in range(3)])
    assert np.allclose(actual_a, expected_a)
    assert np.allclose(actual_h, expected_h)
    assert np.isclose(np.linalg.eigvalsh(expected_h)[-1], np.linalg.eigvalsh(actual_h)[-1])
    vals = np.linalg.eigvals(dense)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(L @ np.eye(3))), np.sort_complex(vals))


def test_jax_linear_transpose_dot_product_for_real_operator():
    def rhs(x):
        return jnp.array([[2.0, -1.0], [3.0, 4.0]]) @ x

    x = jnp.array([0.2, -0.7])
    y = jnp.array([1.3, 0.4])
    transpose = jax.jit(jax.linear_transpose(rhs, jnp.zeros_like(x)))
    (lt_y,) = transpose(y)
    assert np.allclose(np.vdot(rhs(x), y), np.vdot(x, lt_y), rtol=1e-12, atol=1e-12)


def test_parser_and_no_dense_matrix_construction():
    parser = build_parser()
    args = parser.parse_args(["--num-eigenvalues", "2", "--numerical-abscissa-eigenvalues", "3", "--seed", "7"])
    assert args.num_eigenvalues == 2
    assert args.numerical_abscissa_eigenvalues == 3
    source = (WORKSPACE / "analyze_hsx_linearized_ti_curvature_spectrum.py").read_text()
    assert "np.zeros((n, n))" not in source
    assert "eigs(" in source and "eigsh(" in source


def test_spectrum_parser_exposes_curvature_closure_and_metadata_wiring():
    parser = build_parser()
    args = parser.parse_args(["--curvature-flux-closure", "upwind-equilibrium"])
    assert args.curvature_flux_closure == "upwind-equilibrium"
    source = (WORKSPACE / "analyze_hsx_linearized_ti_curvature_spectrum.py").read_text()
    assert "curvature_flux_closure=args.curvature_flux_closure" in source
    assert '"curvature_flux_closure": args.curvature_flux_closure' in source
