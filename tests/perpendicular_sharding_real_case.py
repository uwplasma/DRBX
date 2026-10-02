"""Single-device vs eta-sharded combined perpendicular RHS on the real N32 owner closure (subprocess case).

Executed by ``tests/test_perpendicular_sharding_real.py`` with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
The closure owners cover every eta plane: ``per_plane`` owners on an interior ring and on the wall ring of each plane
(pinned operator options: autodiff K, q2 faces, fixed_radius inner support, ``compact_c3`` magnetic evaluator). For the
P06N variants ``main_phi_dirichlet`` and ``main_phi_neumann`` the RHS is evaluated on the plan (single device) and on
the sharded plan for each shard count; the report holds, per shard count, variant, field and term, the maximum
difference over the closure's target owners relative to the single-device maximum. One JSON object on the last line.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = _REPO_ROOT.parent
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.native.fci_perpendicular_p06_operator import bc_columns                                 # noqa: E402
from drbx.native.fci_perpendicular_plane_preconditioner import owner_layout                       # noqa: E402
from drbx.native.fci_perpendicular_rhs import FIELDS, PHI, PerpendicularParams, perpendicular_rhs  # noqa: E402
from drbx.native.fci_perpendicular_sharding import (                                              # noqa: E402
    make_plane_mesh, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan,
    sharded_perpendicular_rhs, to_plane_major)

N = 32
RHO, TAU = 0.05, 1.0
DIFFUSION = {"density": 1.0e-2, "Te": 2.0e-2, "Ti": 3.0e-2, "vorticity": 4.0e-2}
VARIANTS = ("main_phi_dirichlet", "main_phi_neumann")
TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion", "total")
OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
           "bfield_toroidal": "compact_c3"}
INTERIOR_RING, WALL_RING = N // 2, N - 1


def closure_owners(raw_to_owner, per_plane: int = 2) -> list[int]:
    """``per_plane`` angularly spread owners on the interior ring and on the wall ring of every eta plane."""
    ring, plane, theta = owner_layout(raw_to_owner, N)
    owners = []
    for k in range(N):
        for r in (INTERIOR_RING, WALL_RING):
            ids = np.flatnonzero((ring == r) & (plane == k))
            ids = ids[np.argsort(theta[ids])]
            owners += [int(ids[(len(ids) * j) // per_plane]) for j in range(per_plane)]
    return sorted(set(owners))


def run(shard_counts=(1, 2, 4), per_plane: int = 2) -> dict:
    from perpendicular_structured.reconstruction import load_context
    from p_shared import step3_gates

    started = time.perf_counter()
    raw_to_owner = load_context(N, str(WORKSPACE)).ro
    owners = closure_owners(raw_to_owner, per_plane)
    setup = step3_gates.build_setup(N, ("p06n",), owners=owners, **OPTIONS)
    closure, plan = setup.closure, setup.closure.plan
    adapter = closure.adapters["p06n"]
    perm, inverse, m = plane_major_permutation(raw_to_owner, N)
    params = PerpendicularParams(rho_star=RHO, tau=TAU, diffusion=dict(DIFFUSION))
    layout = (*FIELDS, PHI)
    built = time.perf_counter() - started
    sharded = {}
    report = {"owners": len(owners), "setup_seconds": built, "n_owners": int(len(perm)), "m": m, "results": {}}
    for sz in shard_counts:
        t0 = time.perf_counter()
        sharded[sz] = shard_perpendicular_plan(plan, raw_to_owner, N, sz)
        report.setdefault("shard_seconds", {})[str(sz)] = time.perf_counter() - t0
    for variant in VARIANTS:
        rec = adapter.reconstructions[variant]
        columns = np.asarray(rec.columns)
        values = np.asarray(adapter.owner_values, dtype=np.float64)[:, columns]
        state = {f: values[:, i] for i, f in enumerate(FIELDS)}
        bc = bc_columns(closure.bc["p06n"], columns)
        kinds = dict(zip(layout, rec.field_kinds))
        ref = perpendicular_rhs(plan, state, values[:, 4], bc, kinds, params)
        state_pm = {f: jnp.asarray(to_plane_major(v, inverse)) for f, v in state.items()}
        phi_pm = jnp.asarray(to_plane_major(values[:, 4], inverse))
        for sz in shard_counts:
            sp = sharded[sz]
            t0 = time.perf_counter()
            res = sharded_perpendicular_rhs(sp, state_pm, phi_pm, shard_boundary_data(bc, sp), kinds, params,
                                            make_plane_mesh(sz))
            jax.block_until_ready(res.total)
            entry = {"seconds": time.perf_counter() - t0, "worst_rel": 0.0, "worst_rel_all_owners": 0.0, "fields": {}}
            for f in FIELDS:
                for t in TERMS:
                    a = np.asarray(ref.total[f] if t == "total" else ref.terms[f][t])
                    b = np.asarray(res.total[f] if t == "total" else res.terms[f][t])[perm]
                    scale = float(np.max(np.abs(a[owners])))
                    diff = float(np.max(np.abs(a[owners] - b[owners])))
                    rel = diff / scale if scale > 0 else (0.0 if diff == 0 else float("inf"))
                    all_scale = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else 0.0
                    all_diff = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0
                    rel_all = all_diff / all_scale if all_scale > 0 else 0.0
                    entry["fields"][f"{f}/{t}"] = {"scale": scale, "diff": diff, "rel": rel, "rel_all": rel_all}
                    entry["worst_rel"] = max(entry["worst_rel"], rel)
                    entry["worst_rel_all_owners"] = max(entry["worst_rel_all_owners"], rel_all)
            entry["finite_targets"] = bool(all(np.isfinite(
                np.asarray(res.total[f])[perm][owners]).all() for f in FIELDS))
            entry["diagnostics"] = {k: np.asarray(v).tolist() for k, v in res.diagnostics.items()}
            report["results"][f"{variant}/Sz{sz}"] = entry
        report.setdefault("single_diagnostics", {})[variant] = {k: float(v) for k, v in ref.diagnostics.items()}
    report["seconds"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                      # bytes on macOS, KiB on Linux
    report["peak_rss_gib"] = rss / 2 ** 30 if sys.platform == "darwin" else rss / 2 ** 20
    report["devices"] = len(jax.devices())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-plane", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(run(per_plane=args.per_plane)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
