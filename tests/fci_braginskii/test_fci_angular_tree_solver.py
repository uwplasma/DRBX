"""The nested radial-tree solve equals a dense solve of the tree matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.native.fci_operators import _angular_tree_solver  # noqa: E402


def test_angular_tree_solver_matches_dense_solve():
    profile, ny, nz = (8, 4, 4, 2, 1), 8, 3
    nx = len(profile)
    rng = np.random.default_rng(1)
    active = np.zeros((nx, ny, nz), bool)
    for i, p in enumerate(profile):
        active[i, ::p] = True
    active[3, 2] = False  # one inactive ring owner (children keep a zero edge to it)
    edge = np.where(active, rng.uniform(0.2, 1.0, (nx, ny, nz)), 0.0)
    edge[0] = 0.0
    for i in range(1, nx):
        edge[i, :, :] = np.where(active[i - 1, (np.arange(ny) // profile[i - 1]) * profile[i - 1]], edge[i], 0.0)
    diag = np.where(active, 4.0 + rng.uniform(size=(nx, ny, nz)), 0.0)
    rhs = np.where(active, rng.standard_normal((nx, ny, nz)), 0.0)

    got = np.asarray(_angular_tree_solver(jnp.asarray(diag), jnp.asarray(edge), profile, jnp.asarray(active))(jnp.asarray(rhs)))

    for k in range(nz):
        nodes = [(i, j) for i in range(nx) for j in range(0, ny, profile[i]) if active[i, j, k]]
        index = {n: m for m, n in enumerate(nodes)}
        matrix = np.zeros((len(nodes),) * 2)
        for (i, j), m in index.items():
            matrix[m, m] = diag[i, j, k]
            parent = (i - 1, (j // profile[i - 1]) * profile[i - 1]) if i else None
            if parent in index:
                matrix[m, index[parent]] = matrix[index[parent], m] = -edge[i, j, k]
        want = np.linalg.solve(matrix, np.array([rhs[n + (k,)] for n in nodes]))
        np.testing.assert_allclose([got[n + (k,)] for n in nodes], want, rtol=1e-12, atol=1e-14)
    assert np.all(got[~active] == 0.0)
