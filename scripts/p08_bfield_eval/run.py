"""One grid, one B-evaluator mode: the P-path perpendicular operators against their references on a region x knot-class sample.

    python -m p08_bfield_eval.run N --mode {spline,compact_c3} --out DIR [--per-cell 6] [--seed 0]     # from DRBX/scripts

Writes ``DIR/N{n}_{mode}/results.json`` and ``arrays.npz`` (see ``README.md``).  Fixed options: autodiff curvature, q2 face
quadrature, ``inner_support="fixed_radius"``; the only thing that differs between the two modes is ``bfield_toroidal`` (operator
rows, closure geometry and every reference of one run use the same evaluator).  The owner sample is a deterministic function of
the grid topology and the owner-centroid geometry (:mod:`p08_bfield_eval.sampling`), identical for both modes.  Single process,
threads limited to 1, a few GiB at N32; nothing large is written.
"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_bfield_eval import metrics as M                                      # noqa: E402
from p08_bfield_eval import sampling as S                                     # noqa: E402

SCHEMA = "drbx.p08-bfield-eval.v1"
OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
CAMPAIGNS = ("p06n",)
#: closure geometry arrays that carry the magnetic field (``jacobian`` / ``J`` are metric-only controls)
RAW_COEFFICIENTS = ("p05_raw_h", "p05_raw_jacobian", "p06_raw_J", "p06_raw_B", "p06_raw_K", "p07_raw_tensor",
                    "p07_raw_divergence")
FACE_COEFFICIENTS = ("p05_face_h", "p05_face_jacobian", "p06_face_J", "p06_face_B", "p06_face_K", "p07_face_tensor")
POINT_ARRAYS = ("raw_points", "face_points", "p07_face_points")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:                                              # pragma: no cover
            return f"unavailable: {exc!r}"
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([line for line in run("status", "--porcelain").splitlines() if line.strip()])}


def build_rows_safe(env, provider, owners):
    """``(owners kept, built rows, failures)``: a build error of the whole sample is traced owner by owner and the failing
    owners are dropped (recorded in ``failures``)."""
    from p_shared import owner_closure as oc
    owners = [int(o) for o in owners]
    failures: dict = {}
    try:
        return owners, oc.build_owner_rows(env, owners, provider=provider), failures
    except Exception as exc:
        failures["_whole_build"] = repr(exc)
    bad = {}
    for owner in owners:
        try:
            oc.build_owner_rows(env, [owner], provider=provider)
        except Exception as exc:
            bad[owner] = repr(exc)
    failures.update({str(k): v for k, v in bad.items()})
    keep = [o for o in owners if o not in bad]
    if not keep:
        raise RuntimeError(f"no owner of the sample could be built: {failures}")
    return keep, oc.build_owner_rows(env, keep, provider=provider), failures


def geometry_payload(env, built) -> dict:
    """The B-carrying closure coefficient arrays (and their points and owner maps) for the cross-mode comparison."""
    geometry, census, t = built["geometry"], env.census, env.t
    out = {f"geom__{name}": np.asarray(getattr(geometry, name), dtype=np.float64)
           for name in RAW_COEFFICIENTS + FACE_COEFFICIENTS + POINT_ARRAYS if getattr(geometry, name, None) is not None}
    faces = np.asarray(built["face_row_indices"], dtype=np.int64)
    out.update(raw_ids=np.asarray(built["raw_ids"], dtype=np.int64), raw_owner=np.asarray(t.ro)[built["raw_ids"]].astype(np.int64),
               face_rows=faces, face_owner_lo=np.asarray(census.owner_lo)[faces].astype(np.int64),
               face_owner_hi=np.asarray(census.owner_hi)[faces].astype(np.int64),
               p07_rows=np.asarray(built["p07_row_indices"], dtype=np.int64))
    return out


def run(n: int, mode: str, out_dir, *, per_cell: int = 6, seed: int = 0, input_root=None, sidecar=None) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    from drbx.native.fci_perpendicular_p06_operator import bc_columns
    from drbx.native.fci_perpendicular_rhs import PerpendicularParams, perpendicular_rhs
    import p06n_field_derived_global.core as p06n_core
    from p_shared import jax_replay as jr
    from p_shared import owner_closure as oc
    from p_shared import perpendicular_reference_rhs as prr
    from p_shared import replay_units as ru
    from p_shared import runner
    from p_shared.bfield import check_bfield_toroidal
    from p_shared.replay_support import DEFAULT_PATHS, DEFAULT_SIDECAR, build_environment
    from p_shared.step3_gates import G33_PARAMS, G33_VARIANTS, WORKSPACE
    from p08_bfield_eval.references import diffusion_midpoint_reference

    check_bfield_toroidal(mode)
    input_root = Path(WORKSPACE if input_root is None else input_root)
    sidecar = Path(DEFAULT_SIDECAR if sidecar is None else sidecar)
    runner.reset_peak_rss()
    clock, timings = time.perf_counter, {}
    started = clock()

    t0 = clock()
    env = build_environment(n=n, input_root=input_root, sidecar_path=sidecar, bfield_toroidal=mode, **OPTIONS)
    provider = oc.load_provider_for_env(sidecar, curvature=OPTIONS["curvature"], face_quadrature=OPTIONS["face_quadrature"],
                                        bfield_toroidal=mode)
    timings["environment"] = clock() - t0
    _log(f"N{n} {mode}: environment {timings['environment']:.1f}s")

    # -- sample (a function of env.t and the metric evaluator only; identical for both modes) ---------------------
    t0 = clock()
    t = env.t
    phi = S.owner_centroid_phi(t, env.ref.metric_evaluator.position)
    phi0, dphi = S.plane_grid(env.ref.bfield_evaluator)
    knot = S.classify_knot(phi, phi0, dphi)
    region = S.region_codes(p06n_core.regional_masks(t), S.axis_core_mask(t))
    sample, counts = S.draw_sample(region, knot, per_cell, seed)
    timings["sample"] = clock() - t0
    _log(f"sample: {len(sample)} owners")

    # -- rows, closure ---------------------------------------------------------------------------------------------
    t0 = clock()
    owners, built, failures = build_rows_safe(env, provider, sample)
    owners = np.asarray(built["owners"], dtype=np.int64)
    timings["rows"] = clock() - t0
    _log(f"rows built for {len(owners)} owners ({len(sample) - len(owners)} dropped) {timings['rows']:.1f}s")
    t0 = clock()
    oracle = ru._load_oracle_owner_values(env, DEFAULT_PATHS, CAMPAIGNS)
    closure = jr.JaxOwnerClosure(env, built, CAMPAIGNS, oracle)
    timings["closure"] = clock() - t0
    _log(f"closure {timings['closure']:.1f}s")

    # -- operator and references -------------------------------------------------------------------------------------
    params = dict(G33_PARAMS)
    layout = (*M.FIELDS, "phi")
    rhs_params = PerpendicularParams(rho_star=params["rho_star"], tau=params["tau"], diffusion=dict(params["diffusion"]))
    ref_params = prr.ReferenceParams(rho_star=params["rho_star"], tau=params["tau"], diffusion=dict(params["diffusion"]))
    support = prr.owner_support(env, built)
    adapter = closure.adapters["p06n"]
    volume = np.asarray(t.vol, dtype=np.float64)[owners]
    variants = tuple(G33_VARIANTS)
    V, F, T, m = len(variants), len(M.FIELDS), len(M.TERMS), len(owners)
    N, R, Rmid = np.zeros((V, F, T, m)), np.zeros((V, F, T, m)), np.zeros((V, F, m))
    t_op = t_ref = 0.0
    for v, variant in enumerate(variants):
        rec = adapter.reconstructions[variant]
        columns = np.asarray(rec.columns)
        values = np.asarray(adapter.owner_values, dtype=np.float64)[:, columns]
        state = {f: values[:, i] for i, f in enumerate(M.FIELDS)}
        t0 = clock()
        res = perpendicular_rhs(closure.plan, state, values[:, 4], bc_columns(closure.bc["p06n"], columns),
                                dict(zip(layout, rec.field_kinds)), rhs_params)
        for f, field in enumerate(M.FIELDS):
            for ti, term in enumerate(M.TERMS):
                N[v, f, ti] = np.asarray(res.total[field] if term == "total" else res.terms[field][term])[owners]
        t_op += clock() - t0
        t0 = clock()
        exact = prr.p06n_state(env, variant)
        ref = prr.reference_rhs(env, owners, exact, ref_params, support=support)
        mid = diffusion_midpoint_reference(env, support, exact, params["diffusion"])
        for f, field in enumerate(M.FIELDS):
            for ti, term in enumerate(M.TERMS):
                R[v, f, ti] = ref[field][term]
            Rmid[v, f] = mid[field]
        t_ref += clock() - t0
        _log(f"  {variant}: operator {t_op:.1f}s references {t_ref:.1f}s (cumulative)")
    timings["operator"], timings["references"] = t_op, t_ref

    arrays = {"owners": owners, "volume": volume, "region": region[owners], "knot": knot[owners], "phi": phi[owners],
              "N": N, "R": R, "Rmid": Rmid}
    t0 = clock()
    metrics = M.compute_metrics(arrays, variants)
    convention = M.diffusion_convention_check(arrays, variants)
    timings["metrics"] = clock() - t0
    timings["total"] = clock() - started

    kept_counts = {name: {**c, "kept": int(np.sum((region[owners] == r) & (knot[owners] == k)))}
                   for (name, c), (r, k) in zip(counts.items(), [(r, k) for r in range(len(S.REGIONS)) for k in range(len(S.KNOTS))])}
    results = {
        "schema": SCHEMA, "n": int(n), "mode": mode, "variants": list(variants), "fields": list(M.FIELDS),
        "terms": list(M.TERMS), "regions": list(S.REGIONS), "knots": list(S.KNOTS),
        "comparisons": [list(c) for c in M.COMPARISONS],
        "sample": {"per_cell": int(per_cell), "seed": int(seed), "knot_tolerance": S.KNOT_TOLERANCE, "phi0": phi0, "dphi": dphi,
                   "n_owners_drawn": int(len(sample)), "n_owners": int(len(owners)), "cells": kept_counts,
                   "owners": [int(o) for o in owners], "drawn_owners": [int(o) for o in sample]},
        "dropped_owners": {"owners": sorted(int(o) for o in set(sample.tolist()) - set(owners.tolist())),
                           "failures": failures},
        "params": params, "diffusion_convention_check": convention, "sanity": M.sanity(arrays, variants),
        "metrics": metrics,
        "receipt": {"options": {**OPTIONS, "bfield_toroidal": mode}, "campaigns": list(CAMPAIGNS),
                    "provenance_bfield_toroidal": env.ref.provenance.get("bfield_toroidal"),
                    "bfield_evaluator": {"class": type(env.ref.bfield_evaluator).__name__,
                                         "toroidal_method": getattr(env.ref.bfield_evaluator, "toroidal_method", None),
                                         "nfp": int(env.ref.bfield_evaluator.nfp), "planes_per_period": int(len(env.ref.bfield_evaluator.phi))},
                    "git": git_state(), "timings_seconds": timings, "peak_rss_gib": runner.peak_rss_gib(),
                    "rss_method": runner._RSS_METHOD, "python": sys.version.split()[0],
                    "n_raw_cells": int(len(built["raw_ids"])), "n_face_rows": int(len(built["face_row_indices"])),
                    "n_p07_rows": int(len(built["p07_row_indices"]))},
    }
    folder = Path(out_dir) / f"N{n}_{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "arrays.npz", **arrays, **geometry_payload(env, built))
    (folder / "results.json").write_text(json.dumps(results, indent=1, allow_nan=False, default=_json_default))
    _log(f"N{n} {mode}: done in {timings['total']:.1f}s, peak RSS {results['receipt']['peak_rss_gib']:.2f} GiB -> {folder}")
    return results


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("n", type=int, help="grid size (32, 48 or 64)")
    parser.add_argument("--mode", required=True, choices=("spline", "compact_c3"))
    parser.add_argument("--out", required=True, help="output folder (N{n}_{mode}/ is created inside)")
    parser.add_argument("--per-cell", type=int, default=6, help="owners per (region, knot class) cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--sidecar", default=None)
    args = parser.parse_args(argv)
    run(args.n, args.mode, args.out, per_cell=args.per_cell, seed=args.seed, input_root=args.input_root, sidecar=args.sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
