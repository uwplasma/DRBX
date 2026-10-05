#!/usr/bin/env python3
"""P08 curvature re-freeze on the nodes: prescribed arm, ``curvature`` term only.

P08 froze, per P06N field set, the owner-projected curvature RHS of the five-field state ``(n, Te, Ti, omega, phi)``. On the nodal layout the
reference is the exact nodal RHS (no owner projection, so ``N - O`` is ``N - R`` and ``O - R`` disappears):
``R = M_psi(q) C(q)[0:4] + coeff C(psi)`` with ``C(f) = (K / B) . grad f`` from the extraction's own logical ``K`` and ``B`` and the exact logical
gradients of the P06N catalogue fields (selector ``phi_plus_tau_pi``; the legacy selector gives the same ``R`` to rounding, recorded). All seven P06N
cases are frozen (the five nonconstant ones and the two constant controls), prescribed arm (fields exact, no evolved state).

Output ``<out>/<arm>/N{n}/nodal_curvature_reference.npz`` (``R (cases, 4, E, P)``, the exact field values at the nodes and the wall face points) and
``nodal_curvature_reference_manifest.json`` (schema ``drbx.p09-nodal-curvature-reference-v1``): the identities of the plan, the nodal metric and the
P06N catalogue, the case specification, ``tau``, the sha256 of the arrays and the production-minus-reference error summary of the prescribed arm
(``sbp_curvature`` centred variant, manufactured wall trace; the core-damped and dissipative variants are in ``static``).

    python curv_refreeze.py N --arm raw|filtered
"""
from __future__ import annotations

import curv_common as C

import argparse
import hashlib
import json
import time

import numpy as np

SCHEMA = "drbx.p09-nodal-curvature-reference-v1"
CATALOGUE = C.SCRIPTS / "p06n_field_derived_global" / "p06n_catalogue.json"


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(arm: str, n: int) -> dict:
    from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks
    from drbx.stencils.nodal_plan import plan_identity

    t0 = time.perf_counter()
    cat = json.loads(CATALOGUE.read_text())
    layout, _metric, _meta, plan = C.load_plan(arm, n)
    base = C.arm_dir(arm, n)
    H = np.asarray(h_weights(plan))
    core = ring_region_masks(layout, n)["core"]
    cases = list(cat["cases"])
    R, vals, wall, summary, leg = [], [], [], {}, {}
    for case in cases:
        z = np.load(base / "references" / f"{C.case_file('p06n/' + case)}.npz")
        R.append(z["R"]); vals.append(z["vals"]); wall.append(z["wall"]); leg[case] = float(z["Rdiff_legacy"])
        s = np.load(base / "static" / f"{C.case_file('p06n/' + case)}.npz")
        entry = {"slots": {k: cat["cases"][case][k] for k in C.SLOTS}}
        for f, eq in enumerate(C.EQUATIONS):
            e = s["N"][0, f] - z["R"][f]                                    # centred variant
            entry[eq] = {"E_global": float(np.sqrt(np.sum(H * e * e) / np.sum(H))),
                         "E_core": float(np.sqrt(np.sum(H[core] * e[core] ** 2) / np.sum(H[core]))),
                         "max_abs_R": float(np.abs(z["R"][f]).max()), "max_abs_N": float(np.abs(s["N"][0, f]).max())}
        summary[case] = entry
    R, vals, wall = np.stack(R), np.stack(vals), np.stack(wall)
    npz = base / "nodal_curvature_reference.npz"
    np.savez_compressed(npz, R=R, values=vals, wall=wall, cases=np.array(cases), slots=np.array(C.SLOTS))
    manifest = {
        "schema": SCHEMA, "n": n, "arm": arm, "family": "A", "mode": "prescribed", "term": "curvature", "tau": C.TAU,
        "psi": C.PSI,
        "orientation": "R = M_psi(q) C(q)[0:4] + coeff C(psi), C(f) = (K / B) . grad f; K, B of the extraction (logical frame), exact logical gradients",
        "owner_projection": "none (nodal reference: N - O = N - R)",
        "cases": {c: {"slots": summary[c]["slots"], "constant": c.startswith("control_constant")} for c in cases},
        "R_shape": list(R.shape), "R_axes": ["case", "equation (n, Te, Ti, omega)", "eta plane", "node"],
        "layout": {"P": layout.P, "n_eta": layout.n_eta, "core_nodes": int(layout.blocks[0].n_nodes)},
        "identities": {"plan_sha256": plan_identity(plan), "metric_identity": str(np.load(C.ARM_ROOT[arm] / f"N{n}" / "nodal_metric.npz")["identity"]),
                       "metric_file": str((C.ARM_ROOT[arm] / f"N{n}" / "nodal_metric.npz").relative_to(C.WORKSPACE)),
                       "catalogue_file": str(CATALOGUE.relative_to(C.WORKSPACE)), "catalogue_sha256": sha256_file(CATALOGUE),
                       "arrays_sha256": sha256_file(npz)},
        "legacy_selector_max_abs_difference_of_R": leg,
        "production_arm": {"operator": "sbp_curvature (centred, matrix wall SAT)", "absolute_method": C.METHOD,
                           "wall_data": "manufactured trace of each evolved field", "errors_N_minus_R": summary},
        "seconds": time.perf_counter() - t0, "peak_rss_gib": C.peak_rss_gib(),
    }
    (base / "nodal_curvature_reference_manifest.json").write_text(json.dumps(manifest, indent=1))
    C.log(f"{arm} N{n} curvature re-freeze written: {npz.name} ({npz.stat().st_size / 1e6:.1f} MB)")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", required=True, choices=list(C.ARM_ROOT))
    a = ap.parse_args(argv)
    run(a.arm, a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
