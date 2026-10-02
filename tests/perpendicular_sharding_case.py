"""Multi-device cases of the eta-sharded perpendicular layer (run in a subprocess with forced host devices).

``tests/test_perpendicular_sharding.py`` executes this module as a script with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (set before JAX is imported) and reads the JSON on the last
stdout line. Modes: ``halo`` (``exchange_plane_halo`` against a NumPy periodic gather) and ``synthetic`` (the combined
RHS on a plane-structured random plan, single-device against sharded).
"""
from __future__ import annotations

import argparse
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
from jax.sharding import NamedSharding, PartitionSpec as P                                       # noqa: E402

from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables       # noqa: E402
from drbx.native.fci_perpendicular_rhs import FIELDS, PerpendicularParams, perpendicular_rhs  # noqa: E402
from drbx.native.fci_perpendicular_sharding import (                                              # noqa: E402
    exchange_plane_halo, make_plane_mesh, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan,
    sharded_perpendicular_rhs, to_plane_major)

N = 12
RHO, TAU = 0.05, 1.0
DIFFUSION = {"density": 1.0e-2, "Te": 2.0e-2, "Ti": 3.0e-2, "vorticity": 4.0e-2}
TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion")


def plane_structured_owners(n: int = N) -> np.ndarray:
    """``raw_to_owner`` of ``n / 2`` theta-aggregated owners per (ring, plane), plane-interleaved numbering."""
    raw = np.arange(n ** 3)
    i, j, k = raw // n ** 2, (raw // n) % n, raw % n
    return (i * (n // 2) + j // 2) * n + k


def run_halo(shard_counts=(1, 2, 4), halos=(1, 2), n=8, m=3, features=2) -> dict:
    out = {}
    rng = np.random.default_rng(0)
    for sz in shard_counts:
        p = n // sz
        mesh = make_plane_mesh(sz)
        for h in halos:
            if sz > 1 and p < h:
                continue
            owned = rng.normal(size=(n, m, features))
            fn = jax.jit(jax.shard_map(lambda x: exchange_plane_halo(x, h, "z", sz), mesh=mesh, in_specs=P("z"),
                                       out_specs=P("z"), check_vma=False))
            got = np.asarray(fn(jax.device_put(owned, NamedSharding(mesh, P("z")))))
            expect = np.concatenate([owned[(s * p - h + np.arange(p + 2 * h)) % n] for s in range(sz)])
            out[f"Sz{sz}_h{h}"] = float(np.max(np.abs(got - expect)))
    return out


def _synthetic_inputs(plan, world):
    rng = np.random.default_rng(3)
    n_owners = world.n_owners
    state = {"density": 1.0 + 0.2 * rng.random(n_owners), "Te": 1.0 + 0.2 * rng.random(n_owners),
             "Ti": 1.0 + 0.2 * rng.random(n_owners), "vorticity": 0.3 * rng.normal(size=n_owners)}
    phi = 0.3 * rng.normal(size=n_owners)
    return state, phi


def _coverage(plan, sharded, raw_to_owner, n_shards) -> dict:
    """Entities per shard of the stacked plan against the global plan (padding has ``-1`` ids)."""
    perm, _inverse, m = plane_major_permutation(raw_to_owner, N)
    shard_of = lambda owner: (perm[owner] // m) // (N // n_shards)
    faces = sharded.plan.faces
    census = np.asarray(faces.census_row)
    kept = np.concatenate([row[row >= 0] for row in census])
    counts = np.bincount(np.searchsorted(plan.faces.census_row, kept), minlength=len(plan.faces.census_row))
    lo, hi = np.asarray(plan.faces.lower_owner), np.asarray(plan.faces.upper_owner)
    both = (lo >= 0) & (hi >= 0) & (shard_of(np.maximum(lo, 0)) != shard_of(np.maximum(hi, 0)))
    p07 = np.asarray(sharded.plan.p07.p07_id)
    kept_p07 = np.concatenate([row[row >= 0] for row in p07])
    return {"cells_kept": int((np.asarray(sharded.plan.cells.raw_ids) >= 0).sum()), "cells": len(plan.cells.raw_ids),
            "faces_min_count": int(counts.min()), "faces_max_count": int(counts.max()),
            "faces_on_two_shards": int((counts == 2).sum()), "faces_across_blocks": int(both.sum()),
            "p07_kept": int(len(kept_p07)), "p07_unique": int(len(np.unique(kept_p07))), "p07": len(plan.p07.p07_id),
            "local_rows": sharded.local_rows}


def run_synthetic(shard_counts=(1, 2, 4), seed=0, kinds=("dirichlet", "mixed")) -> dict:
    from tests.perpendicular_synthetic import Boundary, lower_world, make_world
    raw_to_owner = plane_structured_owners()
    world = make_world(seed=seed, n=N, raw_to_owner=raw_to_owner)
    plan = lower_world(world)
    state, phi = _synthetic_inputs(plan, world)
    boundary = Boundary(n_fields=5)

    def dirichlet(points):
        value, gradient = boundary.dirichlet(points)
        return 1.5 + 0.4 * value, 0.4 * gradient

    bc = boundary_data_from_callables(plan, dirichlet, boundary.normal)
    params = PerpendicularParams(rho_star=RHO, tau=TAU, diffusion=dict(DIFFUSION))
    perm, inverse, m = plane_major_permutation(raw_to_owner, N)
    state_pm = {k: jnp.asarray(to_plane_major(v, inverse)) for k, v in state.items()}
    phi_pm = jnp.asarray(to_plane_major(phi, inverse))
    sharded = {sz: shard_perpendicular_plan(plan, raw_to_owner, N, sz) for sz in shard_counts}
    out = {"coverage": {str(sz): _coverage(plan, sharded[sz], raw_to_owner, sz) for sz in shard_counts}}
    try:
        shard_perpendicular_plan(plan, raw_to_owner, N, 2, halo=2)
        out["halo2_error"] = None
    except ValueError as error:
        out["halo2_error"] = str(error)
    for kind in kinds:
        kind_list = ("dirichlet", "neumann", "dirichlet", "neumann", "dirichlet") if kind == "mixed" else kind
        ref = perpendicular_rhs(plan, state, phi, bc, kind_list, params)
        entry = {"nan_owners": int(sum(np.isnan(np.asarray(ref.terms[f][t])).sum() for f in FIELDS for t in TERMS))}
        for sz in shard_counts:
            sp = sharded[sz]
            res = sharded_perpendicular_rhs(sp, state_pm, phi_pm, shard_boundary_data(bc, sp), kind_list, params,
                                            make_plane_mesh(sz))
            worst = {}
            for f in FIELDS:
                for t in (*TERMS, "total"):
                    a = np.asarray(ref.total[f] if t == "total" else ref.terms[f][t])
                    b = np.asarray(res.total[f] if t == "total" else res.terms[f][t])[perm]
                    if not np.array_equal(np.isnan(a), np.isnan(b)):
                        worst[f"{f}/{t}"] = float("inf")
                        continue
                    ok = np.isfinite(a)
                    worst[f"{f}/{t}"] = float(np.max(np.abs(a[ok] - b[ok])) / max(np.max(np.abs(a[ok])), 1e-300))
            entry[f"Sz{sz}"] = {"max_rel": max(worst.values()), "worst": max(worst, key=worst.get),
                                "diagnostics": {k: np.asarray(v).tolist() for k, v in res.diagnostics.items()}}
        entry["single_diagnostics"] = {k: float(v) for k, v in ref.diagnostics.items()}
        out[kind] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("halo", "synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_halo() if args.mode == "halo" else run_synthetic(seed=args.seed)
    result["devices"] = len(jax.devices())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
