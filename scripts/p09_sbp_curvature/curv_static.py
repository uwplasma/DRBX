"""Nodal curvature operator on the exact fields: ``<out>/<arm>/N{n}/static/<case>.npz`` (``N`` for every variant).

Variants (``configuration.json``): ``centered`` (the default operator: centred split form plus the matrix wall SAT),
``centered_core`` (plus the core shell damping, ``c_kappa = 1``) and ``dissipative`` (matrix face-jump dissipation, level-face upwind
and core damping). Wall data is the exact manufactured trace of the four evolved fields (all field kinds alike). ``Nleg`` is the
centred operator with the legacy selector ``phi_plus_tau_ti`` (the same total to rounding except for the wall SAT).

    python curv_static.py N --arm raw|filtered [--cases ...]
"""
from __future__ import annotations

import curv_common as C

import argparse
import json
import time

import numpy as np


def build_variants(jax):
    from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
    from drbx.native.fci_perpendicular_sbp_curvature import sbp_curvature_ext

    def make(spec, psi=C.PSI):
        ck = float(spec["c_kappa"])
        jd = bool(spec["jump_dissipation"])
        return jax.jit(lambda plan, F_ext, q_ext, wall, B_ext: sbp_curvature_ext(plan, F_ext, q_ext, SatBoundaryData((wall,), None),
                                                                          tau=C.TAU, psi=psi, absolute_method=C.METHOD,
                                                                          jump_dissipation=jd, c_kappa=ck, B_ext=B_ext))

    fns = {name: make(spec) for name, spec in C.VARIANTS.items()}
    legacy = make(C.VARIANTS["centered"], psi="phi_plus_tau_ti")
    return fns, legacy


def run(arm: str, n: int, only=None) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    t0 = time.perf_counter()
    layout, metric, meta, plan = C.load_plan(arm, n)
    plan = jax.tree_util.tree_map(jnp.asarray, plan)
    n_chunks = 4
    fns, legacy = build_variants(jax)
    ref_dir = C.arm_dir(arm, n) / "references"
    out_dir = C.arm_dir(arm, n) / "static"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    files = sorted(ref_dir.glob("*.npz"))
    for f in files:
        z = np.load(f)
        label = str(z["label"])
        if only and label not in only and label.split("/")[1] not in only:
            continue
        ts = time.perf_counter()
        vals, wall = z["vals"], z["wall"]
        q = np.moveaxis(vals, 0, -1)                                  # (E, P, 5)
        w = np.moveaxis(wall[:4], 0, -1)                              # (E, N_w, 4)
        N = np.zeros((len(fns), 4) + vals.shape[1:])
        times = {}
        for i, (name, fn) in enumerate(fns.items()):
            tv = time.perf_counter()
            N[i] = np.moveaxis(C.chunked_apply(fn, plan, q, w, n_chunks), -1, 0)
            times[name] = time.perf_counter() - tv
        Nleg = np.moveaxis(C.chunked_apply(legacy, plan, q, w, n_chunks), -1, 0)
        np.savez_compressed(out_dir / f.name, N=N, variants=np.array(list(fns)), Nleg=Nleg)
        recs.append(dict(case=label, n_chunks=n_chunks, seconds=time.perf_counter() - ts, apply_seconds=times,
                         max_abs_N=[float(np.abs(N[0, k]).max()) for k in range(4)],
                         legacy_minus_pi_centered_max=float(np.abs(Nleg - N[0]).max())))
        C.log(f"{arm} N{n} static {label}: {recs[-1]['seconds']:.1f} s  max|N| " + " ".join(f"{x:.2e}" for x in recs[-1]["max_abs_N"]))
    path = C.arm_dir(arm, n) / "static_receipt.json"
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
    a = ap.parse_args(argv)
    run(a.arm, a.n, a.cases.split(",") if a.cases else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
