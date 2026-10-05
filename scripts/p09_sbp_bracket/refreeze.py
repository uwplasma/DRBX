#!/usr/bin/env python3
"""P08 bracket re-freeze on the nodes: prescribed-phi arm, ``poisson_bracket`` term only.

P08 froze, per P06N field set, the owner-projected point brackets of ``phi`` against ``n, Te, Ti, omega``. On the nodal
layout the reference is the exact nodal point bracket (no owner projection, so ``N - O`` is ``N - R`` and ``O - R``
disappears): ``R_f = -(h x grad phi) . grad f / (|J| rho*)`` with the logical metric of the extraction and the exact
logical gradients of the P06N catalogue fields. The four Dirichlet-phi variants of the P08 re-freeze are covered:
``main_phi_dirichlet``, ``heldout_phi_dirichlet``, ``dirichlet_rich`` and ``control_constant_dirichlet``.

Output ``<out>/N{n}/nodal_bracket_reference.npz`` (``R (variants, 4, E, P)``, the exact field values at the nodes and the wall
face points) and ``nodal_bracket_reference_manifest.json`` (schema ``drbx.p09-nodal-bracket-reference-v1``): the identities of
the plan, the nodal metric and the P06N catalogue, the variant specification, ``rho*``, sha256 of the arrays, and the
production-minus-reference error summary of the prescribed-phi arm (``sbp_bracket`` with ``c_kappa = 1``, manufactured
wall trace).

    python refreeze.py N --out ROOT
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                                      # noqa: E402

import fields as F                                                                      # noqa: E402
from references import RHO, layout_points, load_layout_metric, log, peak_rss_gib, point_bracket    # noqa: E402

SCHEMA = "drbx.p09-nodal-bracket-reference-v1"
VARIANTS = ("main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich", "control_constant_dirichlet")
SLOTS = ("n", "Te", "Ti", "omega", "phi")
CATALOGUE = SCRIPTS / "p06n_field_derived_global" / "p06n_catalogue.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_fields(cat: dict, variant: str) -> list[str]:
    spec = cat["cases"][variant]
    return [spec[slot].split(":")[0] for slot in SLOTS]


def run(n: int, out_root: Path, c_kappa: float = 1.0) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import p06n_field_derived_global.fields as p06f
    from drbx.geometry.nodal_layout import wall_points
    from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
    from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket
    from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks
    from drbx.stencils.nodal_plan import build_nodal_plan, plan_identity

    t0 = time.perf_counter()
    cat = json.loads(CATALOGUE.read_text())
    layout, metric, meta = load_layout_metric(n, out_root)
    plan = build_nodal_plan(layout, metric)
    E, P = layout.n_eta, layout.P
    pts = layout_points(layout).reshape(-1, 3)
    wpts = wall_points(layout, layout.walls[0]).reshape(-1, 3)
    period = float(F.PERIOD)
    H = np.asarray(h_weights(plan))
    core = ring_region_masks(layout, n)["core"]
    sbp = jax.jit(lambda p, phi, g, bcd, rho, ck: sbp_bracket(p, phi, g, bcd, rho, ck))
    R = np.zeros((len(VARIANTS), 4, E, P))
    N = np.zeros_like(R)
    values = np.zeros((len(VARIANTS), 5, E, P))
    wall_vals = []
    summary = {}
    for vi, variant in enumerate(VARIANTS):
        names = variant_fields(cat, variant)
        v = np.zeros((5, E * P))
        g = np.zeros((5, E * P, 3))
        wv = []
        for s, name in enumerate(names):
            vv, gg, _ = p06f.evaluate(None, pts, name, period)
            v[s], g[s] = vv, gg
            wv.append(np.asarray(p06f.evaluate(None, wpts, name, period)[0]).reshape(E, -1))
        values[vi] = v.reshape(5, E, P)
        wall_vals.append(np.stack(wv))
        gr = g.reshape(5, E, P, 3)
        for f in range(4):
            R[vi, f] = point_bracket(metric.h, metric.jac, gr[4], gr[f])
        gstack = np.stack([values[vi, f] for f in range(4)], axis=-1)
        wstack = np.stack([wall_vals[-1][f] for f in range(4)], axis=-1)
        N[vi] = np.moveaxis(np.asarray(sbp(plan, values[vi, 4], gstack, SatBoundaryData((jnp.asarray(wstack),), None), RHO,
                                           c_kappa)), -1, 0)
        entry = {"fields": names}
        for f, slot in enumerate(SLOTS[:4]):
            e = N[vi, f] - R[vi, f]
            entry[slot] = {"E_global": float(np.sqrt(np.sum(H * e * e) / np.sum(H))),
                           "E_core": float(np.sqrt(np.sum(H[core] * e[core] ** 2) / np.sum(H[core]))),
                           "max_abs_R": float(np.abs(R[vi, f]).max()), "max_abs_N": float(np.abs(N[vi, f]).max())}
        summary[variant] = entry
        log(f"N{n} re-freeze {variant}: " + " ".join(f"{s} E={entry[s]['E_global']:.3e}" for s in SLOTS[:4]))
    path = out_root / f"N{n}"
    npz = path / "nodal_bracket_reference.npz"
    np.savez_compressed(npz, R=R, values=values, wall=np.stack(wall_vals), variants=np.array(VARIANTS),
                        slots=np.array(SLOTS))
    manifest = {
        "schema": SCHEMA, "n": n, "family": "A", "arm": "prescribed_phi", "term": "poisson_bracket", "rho_star": RHO,
        "orientation": "R_f = point_bracket(h, |J|, grad phi, grad f) / rho_star at the nodes (logical metric, logical exact gradients)",
        "owner_projection": "none (nodal reference: N - O = N - R)",
        "variants": {v: {"slots": dict(zip(SLOTS, variant_fields(cat, v)))} for v in VARIANTS},
        "R_shape": list(R.shape), "layout": {"P": P, "n_eta": E, "core_nodes": int(layout.blocks[0].n_nodes)},
        "identities": {"plan_sha256": plan_identity(plan), "metric_identity": str(np.load(path / "nodal_metric.npz")["identity"]),
                       "catalogue_file": str(CATALOGUE.relative_to(SCRIPTS.parent)), "catalogue_sha256": sha256_file(CATALOGUE),
                       "arrays_sha256": sha256_file(npz)},
        "production_arm": {"operator": "sbp_bracket", "c_kappa": c_kappa, "wall_data": "manufactured trace of each advected field",
                           "errors_N_minus_R": summary},
        "seconds": time.perf_counter() - t0, "peak_rss_gib": peak_rss_gib(),
    }
    (path / "nodal_bracket_reference_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"N{n} re-freeze written: {npz.name} ({npz.stat().st_size / 1e6:.1f} MB)")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c-kappa", type=float, default=float(F.CONFIG["c_kappa"]))
    args = ap.parse_args(argv)
    run(args.n, args.out, args.c_kappa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
