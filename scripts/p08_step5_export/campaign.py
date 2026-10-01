#!/usr/bin/env python3
"""P08 step 5: sparse export of the P07 owner action from the step-4 full-grid row artifacts (N32 / N48 / N64).

Computation only (CPU, one process). Per grid the P07 part of the step-4 artifact is lowered
(``include=("p07",)``), exported with :func:`drbx.native.fci_perpendicular_p07_sparse.export_p07_sparse` for both
kinds (all-Dirichlet, all-Neumann), and each export is gated against ``p07_action`` on the full grid (random fields
and random boundary data; the linear part separately). The small files (``p07_<kind>.npz``, ``owner_map.npz``) are
what is downloaded for local linear-solver studies.

Commands: ``verify-inputs``, ``run``, ``validate``. Run as ``python -m p08_step5_export.campaign`` from
``DRBX/scripts`` (``DRBX/scripts`` must be on ``sys.path`` for every ``p_shared`` import).
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import fcntl
import gc
import json
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent            # .../DRBX/scripts
REPO = SCRIPTS.parent            # .../DRBX
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np                                                                 # noqa: E402

from drbx.native.fci_perpendicular_p07_operator import p07_action                  # noqa: E402
from drbx.native.fci_perpendicular_p07_sparse import (                             # noqa: E402
    KINDS, apply_p07_sparse, export_p07_sparse, save_p07_sparse)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData, zero_boundary_data  # noqa: E402
from drbx.stencils.census import FaceCensus                                        # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays                           # noqa: E402
from drbx.stencils.loader import LoaderGrid                                        # noqa: E402
from drbx.stencils.operator_plan import lower_perpendicular_plan_from_artifact     # noqa: E402

from p_shared import runner                                                        # noqa: E402
from p_shared.replay_support import build_environment                              # noqa: E402
from p08_step2_global import replay                                                # noqa: E402
from p08_step4_global.smoke import check_artifact_options                          # noqa: E402

GRIDS = (32, 48, 64)
SCHEMA = "drbx.p08-step5-p07-export-v1"
#: the final operator bundle, pinned: ``configuration.json`` must say exactly this (and is hashed into the identity)
OPERATOR_OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
BLOCKS = ("matrix", "dirichlet_value", "dirichlet_tangential", "neumann_normal")

sha = runner.sha256_file
digest = runner.digest
write = runner.write_json


@contextmanager
def lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def config() -> dict:
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-5 export campaign contract")
    if cfg["operator_options"] != OPERATOR_OPTIONS:
        raise ValueError(f"configuration.json operator_options {cfg['operator_options']} differ from the pinned "
                         f"final bundle {OPERATOR_OPTIONS}")
    if not set(cfg["resolutions"]) <= set(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    if list(cfg["kinds"]) != list(KINDS):
        raise ValueError(f"configuration.json kinds {cfg['kinds']} differ from {list(KINDS)}")
    return cfg


def operator_options(cfg: dict | None = None) -> dict:
    return dict((cfg or config())["operator_options"])


SOURCE_FILES = [
    "scripts/p08_step5_export/campaign.py",
    "scripts/p08_step5_export/configuration.json",
    "scripts/p08_step4_global/smoke.py",
    "scripts/p08_step2_global/replay.py",
    "scripts/p_shared/replay_support.py",
    "src/drbx/native/fci_perpendicular_p07_sparse.py",
    "src/drbx/native/fci_perpendicular_p07_operator.py",
    "src/drbx/native/fci_perpendicular_reconstruction_state.py",
    "src/drbx/stencils/operator_plan.py",
]


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inputs (the step-4 campaign folder) and identity
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    if not Path(path).is_file():
        raise ValueError(f"missing input file: {path}")
    return json.loads(Path(path).read_text())


def read_step4(step4: Path, grids) -> dict:
    """Check the step-4 campaign folder for the requested ``grids`` and return its provenance record. Raises
    ``ValueError`` if a grid did not pass step 4's smoke + preflight gates or its artifact files are missing."""
    step4 = Path(step4)
    validation = _read_json(step4 / "validation.json")
    manifest = _read_json(step4 / "campaign_manifest.json")
    if not (step4 / "localized_sidecar.json").is_file():
        raise ValueError(f"missing input file: {step4 / 'localized_sidecar.json'}")
    identity = manifest.get("identity")
    if not identity or validation.get("identity") != identity:
        raise ValueError("step-4 validation.json and campaign_manifest.json carry different campaign identities")
    if validation.get("operator_options") != OPERATOR_OPTIONS:
        raise ValueError(f"step-4 validation.json operator options {validation.get('operator_options')} differ from "
                         f"the pinned {OPERATOR_OPTIONS}")
    out = {}
    for n in grids:
        rec = (validation.get("grids") or {}).get(str(n))
        if rec is None:
            raise ValueError(f"step-4 validation.json has no grid N{n}")
        if rec.get("smoke_pass") is not True or rec.get("preflight_pass") is not True:
            raise ValueError(f"step-4 grid N{n} did not pass (smoke_pass={rec.get('smoke_pass')}, "
                             f"preflight_pass={rec.get('preflight_pass')})")
        gdir = step4 / "artifact" / f"N{n}"
        for name in ("manifest.json", "build_identity.json"):
            if not (gdir / name).is_file():
                raise ValueError(f"missing step-4 artifact file: {gdir / name}")
        out[str(n)] = {"artifact_dir": str(gdir.resolve()),
                       "artifact_identity_sha256": digest(json.loads((gdir / "build_identity.json").read_text())),
                       "manifest_sha256": sha(gdir / "manifest.json")}
    return {"step4_identity": identity, "step4_campaign": str(step4.resolve()),
            "localized_sidecar_sha256": sha(step4 / "localized_sidecar.json"), "grids": out}


def jax_check() -> dict:
    info = replay.jax_info()
    if info["backend"] != "cpu":
        raise ValueError(f"JAX CPU backend required, got {info['backend']!r}")
    if not info["x64"]:
        raise ValueError("JAX x64 required (JAX_ENABLE_X64=true)")
    return info


def verify_inputs(*, step4: Path, input_root: Path, output: Path, grids) -> tuple[str, dict]:
    """Validate the step-4 inputs and the backend; record/check ``provenance/inputs.json``. Returns
    ``(identity, inputs)``. Refuses if the identity (configuration, sources, step-4 campaign) changed relative to a
    previous run at the same ``output``."""
    cfg = config()
    if not Path(input_root).is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")
    info = jax_check()
    inputs = read_step4(step4, grids)
    sources = source_hashes()
    commit = _commit()
    identity = digest({"configuration": cfg, "sources": sources, "step4_identity": inputs["step4_identity"],
                       "localized_sidecar_sha256": inputs["localized_sidecar_sha256"]})
    path = Path(output) / "provenance" / "inputs.json"
    previous = json.loads(path.read_text()) if path.is_file() else None
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        for n, rec in inputs["grids"].items():
            old = previous["grids"].get(n)
            if old is not None and old["artifact_identity_sha256"] != rec["artifact_identity_sha256"]:
                raise ValueError(f"artifact identity of N{n} changed; use a new output folder")
    merged = {**(previous["grids"] if previous else {}), **inputs["grids"]}
    record = {"identity": identity, "campaign": "p08_step5_export", "configuration": cfg,
              "operator_options": operator_options(cfg), "source_hashes": sources, "commit": commit,
              "step4_identity": inputs["step4_identity"], "step4_campaign": inputs["step4_campaign"],
              "localized_sidecar_sha256": inputs["localized_sidecar_sha256"], "grids": merged,
              "input_root": str(Path(input_root).resolve()), "jax": info,
              "python": sys.version, "platform": platform.platform()}
    write(path, record)
    return identity, record


# ---------------------------------------------------------------------------
# Per-grid pieces
# ---------------------------------------------------------------------------
def lower_p07_plan(env, artifact: dict, artifact_root: Path, n: int):
    """Only the P07 part of the artifact, on the host (``replay.lower_plan`` with ``include=("p07",)``, no
    ``device_put``)."""
    t = env.t
    grid_dir = Path(artifact["grid_dir"])
    geometry = GeometryArrays.load(grid_dir / "geometry.npz")
    census_path = grid_dir / "census.npz"
    if census_path.is_file():
        stored = FaceCensus.load(census_path)
        if not (np.array_equal(stored.keys(), env.census.keys()) and np.array_equal(stored.owner_lo, env.census.owner_lo)
                and np.array_equal(stored.owner_hi, env.census.owner_hi)):
            raise ValueError("the artifact's census.npz differs from the environment's census")
    grid = LoaderGrid.from_arrays(n=int(n), raw_to_owner=t.ro, eta_centers=t.centers[2])
    return lower_perpendicular_plan_from_artifact(
        artifact_root, n, grid=grid, census=env.census, geometry=geometry, raw_volume=t.rv, owner_volume=t.vol,
        identity=artifact["identity"], include=("p07",))


def gate_export(plan, op, kind: str, *, columns: int, seed: int, tolerance: float) -> dict:
    """``apply_p07_sparse`` vs ``p07_action`` on random fields / boundary data (full action and linear part)."""
    rng = np.random.default_rng(seed)
    n_owners = len(plan.p07.owner_volume)
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    u = rng.normal(size=(n_owners, columns))
    bc = BoundaryData(rng.normal(size=(qd, columns)), rng.normal(size=(qd, 2, columns)), rng.normal(size=(qn, columns)))

    def compare(name, mine, ref):
        scale = float(np.max(np.abs(ref))) if ref.size else 0.0
        diff = float(np.max(np.abs(mine - ref))) if ref.size else 0.0
        return {"max_abs_diff": diff, "max_abs_action": scale, "relative": diff / scale if scale > 0 else None,
                "pass": bool(scale > 0 and diff <= tolerance * scale and np.isfinite(diff))}

    full = compare("full", apply_p07_sparse(op, u, bc), np.asarray(p07_action(plan, u, bc, kind)))
    linear = compare("linear", np.asarray(op.matrix @ u),
                     np.asarray(p07_action(plan, u, zero_boundary_data(plan, columns), kind)))
    return {"kind": kind, "tolerance": tolerance, "columns": columns, "seed": seed, "full": full, "linear": linear,
            "pass": bool(full["pass"] and linear["pass"])}


def _file_record(path: Path) -> dict:
    return {"path": path.name, "bytes": int(path.stat().st_size), "sha256": sha(path)}


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def _peak_rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1 << 30) if sys.platform == "darwin" else float(peak) / (1 << 20)


def receipt_path(output: Path, n: int) -> Path:
    return Path(output) / f"N{n}" / "export_receipt.json"


def _done(output: Path, n: int, identity: str) -> bool:
    path = receipt_path(output, n)
    if not path.is_file():
        return False
    rec = json.loads(path.read_text())
    return rec.get("identity") == identity and rec.get("pass") is True


def export_grid(*, n: int, step4: Path, input_root: Path, output: Path, identity: str, inputs: dict) -> dict:
    cfg = config()
    options = operator_options(cfg)
    step4 = Path(step4)
    gdir = Path(output) / f"N{n}"
    gdir.mkdir(parents=True, exist_ok=True)
    seconds, started = {}, time.perf_counter()

    def lap(name):
        nonlocal started
        now = time.perf_counter()
        seconds[name] = now - started
        started = now

    env = build_environment(n=n, input_root=input_root, sidecar_path=step4 / "localized_sidecar.json", **options)
    lap("environment")
    artifact = replay.load_artifact(step4 / "artifact", n)
    check_artifact_options(artifact["identity"], options)
    art_sha = digest(artifact["identity"])
    recorded = inputs["grids"].get(str(n), {}).get("artifact_identity_sha256")
    if recorded is not None and recorded != art_sha:
        raise ValueError(f"artifact identity of N{n} differs from provenance/inputs.json")
    lap("load_artifact")
    plan = lower_p07_plan(env, artifact, step4 / "artifact", n)
    n_owners = len(plan.p07.owner_volume)
    if n_owners != len(env.t.vol):
        raise ValueError(f"P07 owner count {n_owners} != environment owner count {len(env.t.vol)}")
    lap("lower_p07")

    files, gates, nnz, shapes = {}, {}, {}, {}
    for kind in cfg["kinds"]:
        op = export_p07_sparse(plan, kind)
        lap(f"export_{kind}")
        gates[kind] = gate_export(plan, op, kind, columns=int(cfg["gate_columns"]), seed=int(cfg["gate_seed"]),
                                  tolerance=float(cfg["gate_relative_tolerance"]))
        lap(f"gate_{kind}")
        path = gdir / f"p07_{kind}.npz"
        save_p07_sparse(path, op, {"campaign": identity, "n": int(n), "kind": kind,
                                   "artifact_identity_sha256": art_sha, "operator_options": options})
        files[kind] = _file_record(path)
        nnz[kind] = {b: int(getattr(op, b).nnz) for b in BLOCKS}
        shapes[kind] = {b: [int(s) for s in getattr(op, b).shape] for b in BLOCKS}
        lap(f"save_{kind}")
        del op
    t = env.t
    omap = gdir / "owner_map.npz"
    _save_npz(omap, owner_volume=np.asarray(t.vol), raw_to_owner=np.asarray(t.ro),
              centers_u=np.asarray(t.centers[0]), centers_theta=np.asarray(t.centers[1]),
              centers_eta=np.asarray(t.centers[2]))
    files["owner_map"] = _file_record(omap)
    lap("owner_map")
    passed = bool(all(g["pass"] for g in gates.values()))
    receipt = {"identity": identity, "n": int(n), "pass": passed, "n_owners": int(n_owners),
               "dirichlet_points": int(len(plan.dirichlet_points)), "neumann_points": int(len(plan.neumann_points)),
               "operator_options": options, "artifact_identity_sha256": art_sha, "gates": gates, "nnz": nnz,
               "shapes": shapes, "files": files, "total_bytes": int(sum(f["bytes"] for f in files.values())),
               "seconds": seconds, "wall_seconds": float(sum(seconds.values())),
               "peak_rss_gib_process_lifetime": _peak_rss_gib()}
    write(receipt_path(output, n), receipt)
    del plan, env
    gc.collect()
    return receipt


def run(*, step4, input_root, output, identity, inputs, grids) -> dict:
    results = {}
    for n in grids:
        if _done(output, n, identity):
            results[str(n)] = {"status": "skipped", "pass": True}
            continue
        rec = export_grid(n=n, step4=step4, input_root=input_root, output=output, identity=identity, inputs=inputs)
        results[str(n)] = {"status": "complete", "pass": rec["pass"], "wall_seconds": rec["wall_seconds"]}
        if not rec["pass"]:
            break                                   # receipt is written; stop after the failing grid
    return results


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------
def validate(*, output: Path, identity: str, grids) -> dict:
    """Every requested grid needs a passing receipt of this identity whose files exist with matching sha256."""
    out_grids, summary, problems = {}, {}, {}
    for n in grids:
        errs = []
        path = receipt_path(output, n)
        rec = json.loads(path.read_text()) if path.is_file() else None
        if rec is None:
            errs.append("no export_receipt.json")
        else:
            if rec.get("identity") != identity:
                errs.append("receipt belongs to another campaign identity")
            if rec.get("pass") is not True:
                errs.append("export gate failed")
            for name, f in rec.get("files", {}).items():
                fp = Path(output) / f"N{n}" / f["path"]
                if not fp.is_file():
                    errs.append(f"missing file {f['path']}")
                elif fp.stat().st_size != f["bytes"] or sha(fp) != f["sha256"]:
                    errs.append(f"sha256/size mismatch for {f['path']}")
            if not all(k in rec.get("files", {}) for k in (*config()["kinds"], "owner_map")):
                errs.append("receipt lacks some export files")
        out_grids[str(n)] = {"pass": not errs, "errors": errs}
        if rec is not None:
            summary[str(n)] = {"n_owners": rec.get("n_owners"), "nnz": rec.get("nnz"), "shapes": rec.get("shapes"),
                               "bytes": {k: f["bytes"] for k, f in rec.get("files", {}).items()},
                               "total_bytes": rec.get("total_bytes"), "gates": rec.get("gates"),
                               "wall_seconds": rec.get("wall_seconds"),
                               "peak_rss_gib_process_lifetime": rec.get("peak_rss_gib_process_lifetime"),
                               "pass": rec.get("pass")}
    payload = {"identity": identity, "resolutions": list(grids), "operator_options": operator_options(),
               "all_pass": bool(all(g["pass"] for g in out_grids.values())), "grids": out_grids}
    write(Path(output) / "validation.json", payload)
    write(Path(output) / "summary" / "step5_export_summary.json",
          {"identity": identity, "operator_options": operator_options(), "all_pass": payload["all_pass"],
           "grids": summary})
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "run", "validate"))
    p.add_argument("--step4-campaign", type=Path, required=True,
                   help="the step-4 output's campaign/ folder (artifact/, localized_sidecar.json, validation.json, "
                        "campaign_manifest.json)")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grids", type=int, nargs="+", choices=GRIDS, default=list(GRIDS))
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    args.output = args.output.resolve()
    args.input_root = args.input_root.resolve()
    args.step4_campaign = args.step4_campaign.resolve()
    if args.grids != sorted(set(args.grids)):
        raise ValueError("grids must be unique and ascending")
    with lock(args.output):
        for name in ("logs", "invocations"):
            (args.output / name).mkdir(exist_ok=True, parents=True)
        identity, inputs = verify_inputs(step4=args.step4_campaign, input_root=args.input_root, output=args.output,
                                         grids=args.grids)
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
              {"argv": list(sys.argv if argv is None else argv), "identity": identity, "time": time.time()})
        summary: dict = {"command": args.command, "identity": identity, "status": "complete"}
        ok = True
        if args.command == "run":
            results = run(step4=args.step4_campaign, input_root=args.input_root, output=args.output,
                          identity=identity, inputs=inputs, grids=args.grids)
            summary["grids"] = results
            ok = all(r["pass"] for r in results.values()) and len(results) == len(args.grids)
            if ok:
                ok = validate(output=args.output, identity=identity, grids=args.grids)["all_pass"]
        elif args.command == "validate":
            ok = validate(output=args.output, identity=identity, grids=args.grids)["all_pass"]
        summary["all_pass"] = bool(ok)
        if not ok:
            summary["status"] = "failed_gates"
        write(args.output / "last_exit.json", {"command": args.command, "status": summary["status"],
                                               "time": time.time(), "summary": summary})
        print(json.dumps(summary, default=str), flush=True)
        if not ok:
            raise SystemExit(1)
        return summary


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        try:
            a = parse()
            write(a.output / "last_exit.json", {"command": a.command, "status": "failed", "error": repr(exc),
                                                "time": time.time()})
        finally:
            raise
