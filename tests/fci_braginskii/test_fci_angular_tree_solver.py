"""The nested radial-tree solve equals a dense solve of the tree matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.native.fci_operators import _angular_tree_solver  # noqa: E402


@pytest.mark.parametrize(
    ("profile", "ny", "nz", "inactive_ring_owner", "all_inactive"),
    [
        ((8, 4, 4, 2, 1), 8, 3, True, False),
        ((6, 3, 3, 1), 6, 2, False, False),  # child ratios 2, 1, 3
        ((8,), 8, 2, False, False),  # a single ring (nx = 1)
        ((4, 4, 4), 4, 1, False, False),  # all ratios 1, nz = 1
        ((1, 1), 1, 2, False, False),  # ny = 1
        ((8, 4, 2, 1), 8, 2, False, True),  # nothing active: zero, no NaN
    ],
)
def test_angular_tree_solver_matches_dense_solve(profile, ny, nz, inactive_ring_owner, all_inactive):
    nx = len(profile)
    rng = np.random.default_rng(1)
    active = np.zeros((nx, ny, nz), bool)
    for i, p in enumerate(profile):
        active[i, ::p] = not all_inactive
    if inactive_ring_owner:
        active[3, 2] = False  # one inactive ring owner (children keep a zero edge to it)
    edge = np.where(active, rng.uniform(0.2, 1.0, (nx, ny, nz)), 0.0)
    edge[0] = 0.0
    for i in range(1, nx):
        edge[i, :, :] = np.where(active[i - 1, (np.arange(ny) // profile[i - 1]) * profile[i - 1]], edge[i], 0.0)
    diag = np.where(active, 4.0 + rng.uniform(size=(nx, ny, nz)), 0.0)
    rhs = np.where(active, rng.standard_normal((nx, ny, nz)), 0.0)

    solve = jax.jit(_angular_tree_solver(jnp.asarray(diag), jnp.asarray(edge), profile, jnp.asarray(active)))
    got = np.asarray(solve(jnp.asarray(rhs)))
    assert np.all(np.isfinite(got)) and np.all(got[~active] == 0.0)

    for k in range(nz):
        nodes = [(i, j) for i in range(nx) for j in range(0, ny, profile[i]) if active[i, j, k]]
        index = {n: m for m, n in enumerate(nodes)}
        matrix = np.zeros((len(nodes),) * 2)
        for (i, j), m in index.items():
            matrix[m, m] = diag[i, j, k]
            parent = (i - 1, (j // profile[i - 1]) * profile[i - 1]) if i else None
            if parent in index:
                matrix[m, index[parent]] = matrix[index[parent], m] = -edge[i, j, k]
        if nodes:
            want = np.linalg.solve(matrix, np.array([rhs[n + (k,)] for n in nodes]))
            np.testing.assert_allclose([got[n + (k,)] for n in nodes], want, rtol=1e-12, atol=1e-14)


def test_angular_tree_solver_rejects_non_nested_rings():
    ones = jnp.ones((2, 4, 1))
    for profile in ((4, 3), (2, 1)):  # not nested; innermost ring has more than one owner
        with pytest.raises(ValueError, match="nested"):
            _angular_tree_solver(ones, ones, profile, ones > 0)
