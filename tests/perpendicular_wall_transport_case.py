"""Synthetic wall world of the characteristic wall closure and its multi-device run (P05 bracket, ``wall_transport``).

``tests/test_perpendicular_wall_transport.py`` imports the world / input builders of this module and executes it as a
script (mode ``sharded``) with ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (set before JAX is imported), reading
the JSON on the last stdout line.

The world is the synthetic plane-structured one of ``tests.perpendicular_synthetic`` (real census, random shaped rows of
every kind, donors within two eta planes of their owners, so it can be eta-sharded) with an owner map that has *full
rings* on the four outermost radial rings: ``n / 2`` theta-aggregated owners per (ring, plane) inside, one owner per raw cell
on the rings ``n - 4 .. n - 1``, so every wall cell has the radial column stencil of the characteristic closure.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.native.fci_perpendicular_p06_operator import bc_columns                                # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables      # noqa: E402
from drbx.native.fci_perpendicular_rhs import FIELDS, PerpendicularParams, perpendicular_rhs     # noqa: E402
from drbx.native.fci_perpendicular_sharding import (                                              # noqa: E402
    make_plane_mesh, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan,
    sharded_perpendicular_rhs, to_plane_major)
from tests.perpendicular_synthetic import Boundary, lower_world, make_world                       # noqa: E402

D, NEU = "dirichlet", "neumann"
#: field kinds per column (density, Te, Ti, vorticity, phi): the mixed layout of ``test_perpendicular_rhs`` and all-Dirichlet
KINDS = {"mixed": dict(density=NEU, Te=NEU, Ti=D, vorticity=D, phi=D), "dirichlet": D}
RHO = 0.7
DIFFUSION = {"density": 1.1, "Te": 0.3, "Ti": 0.5, "vorticity": 2.0}
TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion")


def wall_structured_owners(n: int) -> np.ndarray:
    """``raw_to_owner``: theta pairs (two raw cells per owner) on the rings ``< n - 4``, one raw cell per owner on the
    four outermost rings; every owner lies in one (ring, plane) and every plane holds the same number of owners."""
    raw = np.arange(n ** 3)
    i, j, k = raw // n ** 2, (raw // n) % n, raw % n
    paired = (i * (n // 2) + j // 2) * n + k
    single = (n - 4) * (n // 2) * n + ((i - (n - 4)) * n + j) * n + k
    return np.where(i < n - 4, paired, single)


class PositiveBoundary(Boundary):
    """Trace ``1 + 0.3 sin(phase)`` (positive), physical-normal data from its gradient."""

    def dirichlet(self, points):
        v, g = super().dirichlet(points)
        return 1.0 + 0.3 * v, 0.3 * g


def admissible(world):
    """Positive convex value weights (thermodynamically admissible P06 states), as in ``test_perpendicular_rhs``."""
    for key, row in list(world.row_index.items()):
        if hasattr(row, "value") and getattr(row, "value").ndim == 2 and hasattr(row, "gradient"):
            a = np.abs(row.value)
            world.row_index[key] = dataclasses.replace(row, value=a / a.sum(axis=-1, keepdims=True))
    for key, row in list(world.neumann_index.items()):
        a = np.abs(row.value)
        world.neumann_index[key] = dataclasses.replace(row, value=a / a.sum(), boundary_value=0.02 * row.boundary_value)
    return world


def make_wall_world(n: int, seed: int = 1, owners=None):
    """``(world, raw_to_owner)`` of the synthetic wall world on ``n`` rings (``n`` even, ``>= 6``); ``owners`` restricts it
    to the bounded closure of those owners."""
    raw_to_owner = wall_structured_owners(n)
    return admissible(make_world(seed=seed, n=n, raw_to_owner=raw_to_owner, owners=owners)), raw_to_owner


def wall_inputs(world, plan):
    """``(state, phi, bc5, params)``: positive owner fields (density, Te, Ti, vorticity) and potential, the five-column
    boundary data and the parameters (default ``wall_transport``)."""
    fields = 1.0 + 0.4 * np.random.default_rng(5).uniform(size=(world.n_owners, 8))
    boundary = PositiveBoundary(8)
    bc5 = bc_columns(boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal), np.arange(5))
    params = PerpendicularParams(rho_star=RHO, tau=1.0, diffusion=dict(DIFFUSION))
    return {name: fields[:, i] for i, name in enumerate(FIELDS)}, fields[:, 4], bc5, params


def without_wall_columns(plan):
    """``plan`` with the wall column stencils removed (a plan as lowered before the characteristic closure existed)."""
    return dataclasses.replace(plan, cells=dataclasses.replace(plan.cells, wall_cells=None, wall_donors=None,
                                                               wall_weights=None))


def same_bits(a, b, terms=TERMS) -> bool:
    """Bitwise equality of two ``PerpendicularTerms`` (totals, per-term arrays, details, diagnostics; NaNs equal)."""
    pairs = [(a.total[f], b.total[f]) for f in FIELDS] + [(a.terms[f][t], b.terms[f][t]) for f in FIELDS for t in terms]
    pairs += [(a.detail[f][k], b.detail[f][k]) for f in FIELDS for k in a.detail[f]]
    pairs += [(a.diagnostics[k], b.diagnostics[k]) for k in a.diagnostics]
    return all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) for x, y in pairs)


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def run_sharded(shard_counts=(1, 2, 4), n: int = 12, seed: int = 1) -> dict:
    """The characteristic RHS sharded over ``shard_counts`` against single-device, plus the default-mode bitwise checks."""
    world, raw_to_owner = make_wall_world(n, seed)
    plan = lower_world(world)
    state, phi, bc5, params = wall_inputs(world, plan)
    char = dataclasses.replace(params, wall_transport="characteristic")
    kinds = KINDS["mixed"]
    perm, inverse, _m = plane_major_permutation(raw_to_owner, n)
    state_pm = {k: jnp.asarray(to_plane_major(v, inverse)) for k, v in state.items()}
    phi_pm = jnp.asarray(to_plane_major(phi, inverse))
    cells = plan.cells
    ring = np.asarray(cells.raw_ids) // n ** 2
    out = {"wall_cells": int(len(cells.wall_cells)), "expected_wall_cells": int((ring >= n - 2).sum()),
           "stripped_has_none": without_wall_columns(plan).cells.wall_cells is None}
    # single device: the characteristic closure changes the Dirichlet-kind brackets at the wall owners, nothing else
    ref_default = perpendicular_rhs(plan, state, phi, bc5, kinds, params)
    ref_char = perpendicular_rhs(plan, state, phi, bc5, kinds, char)
    out["changed_owners"] = {f: int((np.asarray(ref_default.terms[f]["poisson_bracket"])
                                     != np.asarray(ref_char.terms[f]["poisson_bracket"])).sum()) for f in FIELDS}

    def sharded(sp, prm, sz, terms=None, plan_kinds=kinds):
        extra = {} if terms is None else {"terms": terms}
        return sharded_perpendicular_rhs(sp, state_pm, phi_pm, shard_boundary_data(bc5, sp), plan_kinds, prm,
                                         make_plane_mesh(sz), **extra)

    for sz in shard_counts:
        sp = shard_perpendicular_plan(plan, raw_to_owner, n, sz)
        wall = np.asarray(sp.plan.cells.wall_cells)
        r = int(np.asarray(sp.plan.cells.raw_ids).shape[1])
        entry = {"wall_counts": (wall < r).sum(axis=1).tolist(), "padded_entries": int((wall >= r).sum())}
        res = sharded(sp, char, sz, ("bracket",))
        worst = {f: _rel(np.asarray(res.terms[f]["poisson_bracket"])[perm], ref_char.terms[f]["poisson_bracket"])
                 for f in FIELDS}
        entry["max_rel_bracket"] = max(worst.values())
        res_default = sharded(sp, params, sz, ("bracket",))
        entry["char_differs_from_default"] = any(
            not np.array_equal(np.asarray(res.terms[f]["poisson_bracket"]),
                               np.asarray(res_default.terms[f]["poisson_bracket"])) for f in FIELDS)
        entry["default_bitwise_without_wall_columns"] = same_bits(
            res_default, sharded(shard_perpendicular_plan(without_wall_columns(plan), raw_to_owner, n, sz), params, sz,
                                 ("bracket",)), terms=("poisson_bracket",))
        out[f"Sz{sz}"] = entry
    # all terms (bracket, curvature, diffusion) on two shards: only the Dirichlet-kind brackets change, and by the same amount
    sz = 2 if 2 in shard_counts else shard_counts[-1]
    sp = shard_perpendicular_plan(plan, raw_to_owner, n, sz)
    full_char, full_default = sharded(sp, char, sz), sharded(sp, params, sz)
    out["all_terms"] = {
        "max_rel_total": max(_rel(np.asarray(full_char.total[f])[perm], ref_char.total[f]) for f in FIELDS),
        "unchanged_terms_bitwise": all(
            np.array_equal(np.asarray(full_char.terms[f][t]), np.asarray(full_default.terms[f][t]))
            for f in FIELDS for t in ("curvature", "perpendicular_diffusion")),
        "default_bitwise_without_wall_columns": same_bits(
            full_default, sharded(shard_perpendicular_plan(without_wall_columns(plan), raw_to_owner, n, sz), params, sz))}
    # unequal wall counts: no wall entry on the first shard, so the other shards pad theirs (and the dropped cells keep
    # the plan's Dirichlet rows, as in the single-device run of the same plan)
    sz = shard_counts[-1]
    plane = np.asarray(cells.raw_ids)[np.asarray(cells.wall_cells)] % n
    keep = (plane // (n // sz)) != 0 if sz > 1 else plane % 3 != 0
    uneven = dataclasses.replace(plan, cells=dataclasses.replace(
        cells, wall_cells=np.asarray(cells.wall_cells)[keep], wall_donors=np.asarray(cells.wall_donors)[keep],
        wall_weights=np.asarray(cells.wall_weights)[keep]))
    sp = shard_perpendicular_plan(uneven, raw_to_owner, n, sz)
    wall = np.asarray(sp.plan.cells.wall_cells)
    r = int(np.asarray(sp.plan.cells.raw_ids).shape[1])
    res = sharded(sp, char, sz, ("bracket",))
    ref = perpendicular_rhs(uneven, state, phi, bc5, kinds, char, terms=("bracket",))
    out["uneven"] = {"wall_counts": (wall < r).sum(axis=1).tolist(), "padded_entries": int((wall >= r).sum()),
                     "max_rel_bracket": max(_rel(np.asarray(res.terms[f]["poisson_bracket"])[perm],
                                                 ref.terms[f]["poisson_bracket"]) for f in FIELDS),
                     "differs_from_full": any(
                         not np.array_equal(np.asarray(ref.terms[f]["poisson_bracket"]),
                                            np.asarray(ref_char.terms[f]["poisson_bracket"])) for f in FIELDS)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sharded",))
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args(argv)
    result = run_sharded(n=args.n)
    result["devices"] = len(jax.devices())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
