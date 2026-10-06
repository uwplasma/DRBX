"""``PerpendicularParams.rho_star_convention`` reaches the sharded RHS (one in-process shard; fast).

The static field travels in the params pytree through ``shard_map``: for ``rho_star != 1`` the sharded RHS under
``"single-length"`` equals the single-device one and ``rho_star`` times the sharded ``rho_star = 1`` bracket / curvature,
and the legacy default is not the single-length result. (Multi-shard equality of the unchanged RHS is covered by
``tests/test_perpendicular_sharding.py``.)
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
from drbx.native.fci_perpendicular_rhs import FIELDS, PerpendicularParams, perpendicular_rhs
from drbx.native.fci_perpendicular_sharding import (
    make_plane_mesh, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan, sharded_perpendicular_rhs,
    to_plane_major)
from tests import perpendicular_sharding_case as case
from tests.perpendicular_synthetic import Boundary, lower_world, make_world

KINDS = ("dirichlet", "neumann", "dirichlet", "neumann", "dirichlet")
TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion")


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = np.isfinite(a)
    return float(np.max(np.abs(a[ok] - b[ok])) / max(np.max(np.abs(a[ok])), 1e-300))


def test_rho_star_convention_reaches_the_sharded_rhs():
    n = case.N
    raw_to_owner = case.plane_structured_owners(n)
    world = make_world(seed=0, n=n, raw_to_owner=raw_to_owner)
    plan = lower_world(world)
    state, phi = case._synthetic_inputs(plan, world)
    boundary = Boundary(n_fields=5)

    def dirichlet(points):
        value, gradient = boundary.dirichlet(points)
        return 1.5 + 0.4 * value, 0.4 * gradient

    bc = boundary_data_from_callables(plan, dirichlet, boundary.normal)
    perm, inverse, _m = plane_major_permutation(raw_to_owner, n)
    state_pm = {k: jnp.asarray(to_plane_major(v, inverse)) for k, v in state.items()}
    phi_pm = jnp.asarray(to_plane_major(phi, inverse))
    sp = shard_perpendicular_plan(plan, raw_to_owner, n, 1)
    mesh, sbc = make_plane_mesh(1), shard_boundary_data(bc, sp)
    base = PerpendicularParams(rho_star=1.0, tau=case.TAU, diffusion=dict(case.DIFFUSION), absolute_method="closed_form",
                               rho_star_convention="single-length")
    rho = 0.05

    def sharded(params):
        return sharded_perpendicular_rhs(sp, state_pm, phi_pm, sbc, KINDS, params, mesh)

    one = sharded(base)
    single = sharded(dataclasses.replace(base, rho_star=rho))
    legacy = sharded(dataclasses.replace(base, rho_star=rho, rho_star_convention="legacy-bracket-only"))
    ref = perpendicular_rhs(plan, state, phi, bc, KINDS, dataclasses.replace(base, rho_star=rho))
    for f in FIELDS:
        for term, factor in (("poisson_bracket", rho), ("curvature", rho), ("perpendicular_diffusion", 1.0)):
            got = np.asarray(single.terms[f][term])
            assert _rel(np.asarray(ref.terms[f][term]), got[perm]) <= 1e-12, (f, term)
            assert _rel(factor * np.asarray(one.terms[f][term]), got) <= 1e-12, (f, term)
        assert _rel(np.asarray(ref.total[f]), np.asarray(single.total[f])[perm]) <= 1e-12
        # the legacy convention divides the bracket (and leaves the curvature unscaled)
        assert _rel(np.asarray(one.terms[f]["curvature"]), np.asarray(legacy.terms[f]["curvature"])) <= 1e-12
        assert _rel(np.asarray(one.terms[f]["poisson_bracket"]) / rho, np.asarray(legacy.terms[f]["poisson_bracket"])) <= 1e-12
    assert np.max(np.abs(np.asarray(single.terms["Te"]["poisson_bracket"]))) > 0
