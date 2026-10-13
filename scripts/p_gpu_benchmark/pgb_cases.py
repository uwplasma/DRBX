"""Case table, ``campaign.json``, case-record schema and the ``validate`` logic of the P GPU benchmark (jax-free).

Layout of a campaign folder ``OUT``::

    OUT/campaign.json                 the campaign description + the parameters of this run (written by ``run``)
    OUT/preflight_<backend>.json      identity / backend record (stage ``preflight``)
    OUT/cases/<backend>/<case_id>.json   one JSON per case (stage ``run``; resumable: existing ok / skipped cases are kept)
    OUT/logs/                         worker logs
    OUT/validation_<backend>.json     result of stage ``validate``

``case_id`` = ``<kind>__N<n>__<method>__<dtype>`` (``build``, ``precond``, ``pcg``), ``sharding__N64`` and ``scaling``.
"""
from __future__ import annotations

import json
from pathlib import Path

CASE_SCHEMA = "drbx.p-gpu-benchmark-case-v1"
CAMPAIGN_SCHEMA = "drbx.p-gpu-benchmark-campaign-v1"
HERE = Path(__file__).resolve().parent
CAMPAIGN_FILE = HERE / "campaign.json"

ARM = "filtered"
DEFAULT_NS = (32, 48, 64)
DTYPES = ("f32", "f64")
BACKENDS = ("gpu", "cpu")
PRECOND_METHODS = ("core_schur", "core_schur_zfree", "block_jacobi", "block_thomas")
SOLVAX_METHODS = ("block_jacobi", "block_thomas")
PCG_METHODS = ("core_schur", "solvax_best")
#: kinds whose cases are skipped (status ``skipped``) when the run is restricted to the first ``max_planes`` planes
RESTRICTED_SKIP = "skipped: restricted planes"

#: filtered-arm laplacian metric files of the P10 campaign (``work/p10_campaign_inputs_20261007/inputs_manifest.json``); the
#: sha256 of the files are the identities the campaign is run against
INPUT_FILES = {
    32: {"path": "laplacian/filtered/N32/laplacian_metric.npz", "bytes": 4793452,
         "sha256": "1a9f5692a9cef9647f7648b792d0893ebdd778b2fb0572850851122269830b42"},
    48: {"path": "laplacian/filtered/N48/laplacian_metric.npz", "bytes": 15722864,
         "sha256": "3b639770513e8250207794e0bfbb245084e81449919bfdd5ec807cd64dedca4c"},
    64: {"path": "laplacian/filtered/N64/laplacian_metric.npz", "bytes": 36779120,
         "sha256": "879571919ad49fea5a1e4a70e3b03d0ed31ed7b93a1ac53fb9dd857f089805a5"},
}
MANIFEST_NAME = "inputs_manifest.json"
MANIFEST_SHA256 = "15eda8c6857c3763032e26281052d1d1dc46720a23faefc3fe15c07d4c2cc11a"

#: what the remote environment needs beyond the ``drb`` conda env (constants of the campaign, not documentation)
REMOTE_ENVIRONMENT = {
    "python": "3.12",
    "jax_local_cpu_version": "0.9.2",
    "jaxlib_local_cpu_version": "0.9.2",
    "gpu_install": "pip install -U \"jax[cuda12]==0.9.2\"  (same jax version as the drb env; installs the matching jaxlib and "
                   "nvidia-* CUDA 12 wheels; needs a driver >= 525 on the node)",
    "solvax": "0.8.7",
    "drbx": "editable install of DRBX/ at the pinned commit (pip install -e DRBX/)",
    "other": ["numpy>=2", "scipy", "equinox (dependency of solvax)", "pytest (only for tests/)"],
    "env_vars_gpu": {"JAX_PLATFORMS": "must NOT be set to cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false (set by the harness)",
                     "JAX_ENABLE_X64": "true (set by the harness)"},
    "env_vars_cpu": {"P_GPU_BENCH_CPU_THREADS": "thread count chosen by the remote setup (or --cpu-threads); sets "
                                                  "OMP/OPENBLAS/MKL/NPROC and the XLA intra-op threads"},
    "host_memory_gb": "N64: ~2 GB retained in-plane CSR + 1-4 GB build; plenty on a Perlmutter GPU node",
}

TOLERANCES = {
    "precond_rel_err": {
        "core_schur": {"f32": 1e-5, "f64": 1e-10},
        "core_schur_zfree": {"f32": 1e-5, "f64": 1e-10},
        # observed ~3e-5 at N32 (single-precision LU of the dense / bordered blocks): looser than the DRBX f32 bound
        "block_jacobi": {"f32": 1e-4, "f64": 1e-10},
        "block_thomas": {"f32": 1e-4, "f64": 1e-10},
    },
    "zfree_vs_stored": {"f32": 1e-5, "f64": 1e-10},
    "precond_rel_residual_f64": 1e-8,
    "pcg_true_residual_hinv_max": 1e-10,
    "sharded_vs_single_rel": {"f32": 1e-5, "f64": 1e-10},
}
SOLVE = {"rtol": 1e-11, "maxit": 50, "residual_norm": "h_inv", "x0": "zero",
         "rhs": "s = L_h f*, f* a smooth analytic field of the node coordinates, zero wall data (the solution is f*)",
         "source": "DRBX/scripts/p10_evolved_mms/configuration.json['solve']"}
PRECOND_RHS = {"kind": "standard_normal", "seed": 0}
DENSE_REF_MAX_N = 32
FIT_EXTRAPOLATE_N = (96, 128, 256)
USER_ESTIMATE_F32_GB = {"64": 0.47, "128": 8.4, "256": 164.0}


# ---------------------------------------------------------------------------
# case table
# ---------------------------------------------------------------------------
def case_id(kind: str, n: int | None = None, method: str | None = None, dtype: str | None = None) -> str:
    parts = [kind] + ([f"N{n}"] if n is not None else []) + ([method] if method else []) + ([dtype] if dtype else [])
    return "__".join(parts)


def case_table(ns=DEFAULT_NS) -> list[dict]:
    """Expected cases of one backend: ``{case_id, kind, n, method, dtype, required, allowed_skip}``."""
    out = []
    mem = ["skipped: memory"]
    for n in ns:
        for dt in DTYPES:
            out.append(dict(case_id=case_id("build", n, "core_schur", dt), kind="build", n=n, method="core_schur", dtype=dt,
                            required=True, allowed_skip=[]))
        for m in PRECOND_METHODS:
            for dt in DTYPES:
                req = m in ("core_schur", "core_schur_zfree")
                out.append(dict(case_id=case_id("precond", n, m, dt), kind="precond", n=n, method=m, dtype=dt, required=req,
                                allowed_skip=[] if req else mem))
        for dt in DTYPES:
            out.append(dict(case_id=case_id("pcg", n, "core_schur", dt), kind="pcg", n=n, method="core_schur", dtype=dt,
                            required=True, allowed_skip=[]))
        out.append(dict(case_id=case_id("pcg", n, "solvax_best"), kind="pcg", n=n, method="solvax_best", dtype=None, required=False,
                        allowed_skip=mem + ["skipped: no solvax variant"]))
    if 64 in ns:
        out.append(dict(case_id=case_id("sharding", 64), kind="sharding", n=64, method="core_schur", dtype="f32", required=False,
                        allowed_skip=["skipped: single device", "skipped: backend cpu"]))
    out.append(dict(case_id="scaling", kind="scaling", n=None, method=None, dtype=None, required=True, allowed_skip=[]))
    return out


def build_campaign(ns=DEFAULT_NS) -> dict:
    """The static campaign description (``campaign.json``): inputs identities, cases, tolerances, remote environment."""
    return {
        "schema": CAMPAIGN_SCHEMA,
        "name": "p_gpu_benchmark",
        "description": "P (perpendicular) path of DRBX: plane preconditioners, full psi PCG solve, core_schur build / storage "
                       "scaling, optional eta sharding; per backend gpu (JAX CUDA, one A100) and cpu (same node).",
        "arm": ARM,
        "resolutions": list(ns),
        "backends": list(BACKENDS),
        "inputs": {"manifest": MANIFEST_NAME, "manifest_sha256": MANIFEST_SHA256,
                   "files": {str(n): INPUT_FILES[n] for n in ns}},
        "remote_environment": REMOTE_ENVIRONMENT,
        "solve": SOLVE,
        "precond_rhs": PRECOND_RHS,
        "dense_reference_max_n": DENSE_REF_MAX_N,
        "tolerances": TOLERANCES,
        "scaling": {"extrapolate_n": list(FIT_EXTRAPOLATE_N), "user_estimate_f32_total_gb": USER_ESTIMATE_F32_GB},
        "cases": case_table(ns),
        "stages": ["preflight", "run", "validate"],
        "run": None,
    }


def load_campaign(out: Path | None = None) -> dict:
    """``OUT/campaign.json`` if present, else the static one next to this file."""
    if out is not None and (Path(out) / "campaign.json").is_file():
        return json.loads((Path(out) / "campaign.json").read_text())
    return json.loads(CAMPAIGN_FILE.read_text())


def case_path(out, backend: str, cid: str) -> Path:
    return Path(out) / "cases" / backend / f"{cid}.json"


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("schema", "case_id", "kind", "backend", "n", "arm", "status", "reason", "params", "results", "identity", "env",
                 "created_utc", "elapsed_s")
STATUSES = ("ok", "skipped", "failed")


def check_record_schema(rec: dict) -> list[str]:
    """Problems of a case record (empty list = well formed)."""
    errs = []
    for k in REQUIRED_KEYS:
        if k not in rec:
            errs.append(f"missing key {k!r}")
    if errs:
        return errs
    if rec["schema"] != CASE_SCHEMA:
        errs.append(f"schema {rec['schema']!r} != {CASE_SCHEMA!r}")
    if rec["status"] not in STATUSES:
        errs.append(f"status {rec['status']!r} not in {STATUSES}")
    if rec["backend"] not in BACKENDS:
        errs.append(f"backend {rec['backend']!r}")
    if rec["status"] != "ok" and not rec["reason"]:
        errs.append("a skipped / failed case needs a reason")
    if rec["status"] == "ok" and not isinstance(rec["results"], dict):
        errs.append("results must be a dict")
    for k in ("params", "identity", "env"):
        if not isinstance(rec[k], dict):
            errs.append(f"{k} must be a dict")
    return errs


def read_record(path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def _tol(campaign, method, dtype):
    return campaign["tolerances"]["precond_rel_err"][method][dtype]


def check_case_results(campaign: dict, spec: dict, rec: dict) -> list[str]:
    """Correctness problems of an ``ok`` record against the campaign tolerances."""
    tol, r, errs = campaign["tolerances"], rec["results"], []
    kind, dt, method = spec["kind"], spec["dtype"], spec["method"]
    if kind == "build":
        if not r.get("storage_bytes", 0) > 0:
            errs.append("storage_bytes missing / zero")
        if not r.get("host_build_s", 0) > 0:
            errs.append("host_build_s missing / zero")
    elif kind == "precond":
        e = r.get("rel_err_vs_ref")
        if e is None or not e <= _tol(campaign, method, dt):
            errs.append(f"rel_err_vs_ref {e} > {_tol(campaign, method, dt)} (ref: {r.get('reference')})")
        if method == "core_schur_zfree":
            d = r.get("rel_diff_vs_stored_z")
            if d is None or not d <= tol["zfree_vs_stored"][dt]:
                errs.append(f"rel_diff_vs_stored_z {d} > {tol['zfree_vs_stored'][dt]}")
        if dt == "f64":
            rr = r.get("rel_residual")
            if rr is None or not rr <= tol["precond_rel_residual_f64"]:
                errs.append(f"rel_residual {rr} > {tol['precond_rel_residual_f64']}")
        a = r.get("apply", {})
        if not a.get("median_s", 0) > 0:
            errs.append("apply.median_s missing")
    elif kind == "pcg":
        rtol, maxit = campaign["solve"]["rtol"], campaign["solve"]["maxit"]
        if not r.get("converged", False):
            errs.append("PCG not converged")
        rel = r.get("relative_residual")
        if rel is None or not rel <= rtol:
            errs.append(f"relative_residual {rel} > rtol {rtol}")
        tr = r.get("true_residual_hinv")
        if tr is None or not tr <= tol["pcg_true_residual_hinv_max"]:
            errs.append(f"true_residual_hinv {tr} > {tol['pcg_true_residual_hinv_max']}")
        if not (0 <= r.get("iterations", -1) <= maxit):
            errs.append(f"iterations {r.get('iterations')}")
    elif kind == "sharding":
        d = r.get("rel_diff_vs_single")
        if d is None or not d <= tol["sharded_vs_single_rel"]["f32"]:
            errs.append(f"rel_diff_vs_single {d} > {tol['sharded_vs_single_rel']['f32']}")
    elif kind == "scaling":
        if "shape_model" not in r:
            errs.append("scaling has no shape_model")
    return errs


def validate_campaign(out, backends=None) -> dict:
    """Check that every expected case file of ``backends`` exists, is well formed, and meets its tolerances.

    Returns ``{"ok", "backends": {backend: {"expected", "present", "problems": [...], "cases": {cid: status}}}, "restricted"}``.
    """
    out = Path(out)
    campaign = load_campaign(out)
    run = campaign.get("run") or {}
    restricted = bool(run.get("max_planes"))
    if backends is None:
        backends = [b for b in BACKENDS if (out / "cases" / b).is_dir()]
    report = {"ok": True, "backends": {}, "restricted": restricted, "campaign": campaign.get("name")}
    if not backends:
        report["ok"] = False
        report["problems"] = [f"no case directory under {out / 'cases'}"]
        return report
    for b in backends:
        problems, statuses, present = [], {}, 0
        pf = out / f"preflight_{b}.json"
        if not pf.is_file():
            problems.append(f"missing preflight_{b}.json")
        else:
            pre = json.loads(pf.read_text())
            if not pre.get("ok"):
                problems.append(f"preflight_{b}.json is not ok: {pre.get('problems')}")
        for spec in campaign["cases"]:
            cid = spec["case_id"]
            p = case_path(out, b, cid)
            if not p.is_file():
                problems.append(f"{cid}: case file missing")
                statuses[cid] = "missing"
                continue
            present += 1
            try:
                rec = read_record(p)
            except Exception as exc:                                              # noqa: BLE001
                problems.append(f"{cid}: unreadable ({exc})")
                statuses[cid] = "unreadable"
                continue
            statuses[cid] = rec.get("status")
            errs = check_record_schema(rec)
            if errs:
                problems += [f"{cid}: schema: {e}" for e in errs]
                continue
            if rec["backend"] != b or rec["case_id"] != cid:
                problems.append(f"{cid}: record is for {rec['backend']}/{rec['case_id']}")
                continue
            if rec["status"] == "failed":
                problems.append(f"{cid}: FAILED: {rec['reason']}")
            elif rec["status"] == "skipped":
                ok_skip = any(str(rec["reason"]).startswith(s) for s in spec["allowed_skip"]) or \
                    (restricted and str(rec["reason"]).startswith(RESTRICTED_SKIP))
                if not ok_skip:
                    problems.append(f"{cid}: skipped with reason {rec['reason']!r} (allowed: {spec['allowed_skip']})")
            else:
                problems += [f"{cid}: {e}" for e in check_case_results(campaign, spec, rec)]
        report["backends"][b] = {"expected": len(campaign["cases"]), "present": present, "problems": problems,
                                 "cases": statuses}
        report["ok"] = report["ok"] and not problems
    return report
