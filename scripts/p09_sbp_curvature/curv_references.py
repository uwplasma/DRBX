"""Nodal exact fields and curvature right-hand sides: ``<out>/<arm>/N{n}/references/<case>.npz``.

For every case (P06: four states; P06N: seven cases) the values of the five fields ``(n, Te, Ti, omega, phi)`` at all nodes and
at the outer-wall face points, and the exact nodal curvature RHS ``R = M_psi(q) C(q)[0:4] + coeff C(psi)`` with the extraction's own
logical ``K`` and ``B`` and the exact logical gradients (``C(f) = (K / B) . grad f``; chain-rule ``C(psi)``), so that ``N - R`` isolates
the derivative error. ``R`` is stored for the model default selector ``phi_plus_tau_pi``; ``Rdiff_legacy`` records the largest
difference to the same formula with ``phi_plus_tau_ti`` (the same to rounding).

    python curv_references.py N --arm raw|filtered [--cases ...]
"""
from __future__ import annotations

import curv_common as C

import argparse
import json
import time

import numpy as np


def reference_rhs(vals, grads, metric, tau=C.TAU):
    from drbx.native.fci_perpendicular_sbp_curvature import continuum_rhs
    R = continuum_rhs(vals, grads, metric.K, metric.B, tau, "phi_plus_tau_pi")
    Rl = continuum_rhs(vals, grads, metric.K, metric.B, tau, "phi_plus_tau_ti")
    return np.moveaxis(R, -1, 0), float(np.abs(R - Rl).max())


def compute_case(arm: str, n: int, case, layout, metric) -> dict:
    from drbx.geometry.nodal_layout import wall_points

    t0 = time.perf_counter()
    E, P = layout.n_eta, layout.P
    vals, grads = case.evaluate(C.layout_points(layout))
    vals, grads = vals.reshape(5, E, P), grads.reshape(5, E, P, 3)
    wv, _ = case.evaluate(wall_points(layout, layout.walls[0]))
    wv = wv.reshape(5, E, -1)
    R, dleg = reference_rhs(vals, grads, metric)
    path = C.arm_dir(arm, n) / "references"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"{C.case_file(case.label)}.npz", vals=vals, wall=wv, R=R, Rdiff_legacy=np.array(dleg),
                        label=np.array(case.label), gate_eqs=np.array(case.gate_eqs), constant=np.array(case.constant))
    rec = dict(case=case.label, seconds=time.perf_counter() - t0, max_abs_R=[float(np.abs(R[f]).max()) for f in range(4)],
               Rdiff_legacy=dleg)
    C.log(f"{arm} N{n} references {case.label}: {rec['seconds']:.1f} s max|R| " + " ".join(f"{x:.2e}" for x in rec["max_abs_R"])
          + f" legacy-pi {dleg:.1e}")
    return rec


def run(arm: str, n: int, only=None, with_mms: bool = True, frozen_provider: bool = False) -> dict:
    t0 = time.perf_counter()
    layout, metric, meta, _plan = C.load_plan(arm, n)
    ref = None
    if with_mms:
        if frozen_provider:                                  # the frozen scripts provider (~1.2 GB); same fields bit for bit
            ref = C.M3R.build_reference(C.eta_filter_option(meta))
        else:
            ref = C.lean_reference()
        C.log(f"{arm} N{n}: MMS reference ({'frozen provider' if frozen_provider else 'lean'}) ready ({time.perf_counter() - t0:.1f} s)")
    cases = C.all_cases(ref)
    if only:
        cases = [c for c in cases if c.label in only or c.name in only]
    recs = [compute_case(arm, n, c, layout, metric) for c in cases]
    path = C.arm_dir(arm, n) / "references_receipt.json"
    old = json.loads(path.read_text()) if path.exists() else {"cases": [], "invocations": []}
    keep = [r for r in old["cases"] if r["case"] not in {x["case"] for x in recs}]
    out = dict(arm=arm, n=n, cases=keep + recs,
               invocations=old.get("invocations", []) + [dict(seconds=time.perf_counter() - t0, peak_rss_gib=C.peak_rss_gib(),
                                                              cases=[r["case"] for r in recs])])
    path.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", required=True, choices=list(C.ARM_ROOT))
    ap.add_argument("--cases", default="")
    ap.add_argument("--no-mms", action="store_true")
    ap.add_argument("--frozen-provider", action="store_true", help="build the ~1.2 GB scripts provider for the MMS fields")
    a = ap.parse_args(argv)
    run(a.arm, a.n, a.cases.split(",") if a.cases else None, with_mms=not a.no_mms, frozen_provider=a.frozen_provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
