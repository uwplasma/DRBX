"""The G3 replay stage: one grid's row artifact -> plan -> JAX operators -> frozen-oracle comparison.

``run_replay_stage`` is the code path the remote campaign runs per grid (``campaign.run`` /
``run-stage --stage replay``); the bounded local test (``tests/test_p08_step2_campaign_real.py``) runs the
*same function* on a real tensor artifact of the N32 owner closure with the owner set restricted through
``selection``.

1. ``build_environment`` (campaign geometry, census, frozen continuum reference) and the oracles' saved
   ``owner_values`` (the fields the artifact rows are applied to);
2. the plan: ``lower_perpendicular_plan_from_artifact`` -- streaming, chunk file by chunk file, point
   chunks decoded *factored* (tensor sources are never expanded), sha256/schema/identity checked -- and one
   ``jax.device_put`` of the whole plan;
3. campaigns **sequentially** (``p05, p05n_frozen, p05n_upwind, p06n, p06_legacy, p07, p07n``): a fresh
   ``p_shared.jax_replay.JaxOwnerClosure`` on the shared plan evaluates the campaign's boundary data once at
   the plan's point tables (batched by ``boundary_batch`` points) and runs its operators in column blocks
   (``column_block`` fields, ``p06n_variant_block`` P06N variants per call); the result (host-format sparse
   owner numerators, exactly the layout of the step-1 replay unit outputs) is checkpointed to
   ``<output>/_chunks/N{n}/jax_<campaign>/`` (``runner.write_unit``: identity + sha256 receipt) and the
   per-campaign intermediates are freed before the next campaign;
4. ``comparison.compare_operator_terms`` applies the step-1 reduction/comparison to the merged terms;
   ``replay.json`` + ``report.md`` (step-1 format, ``write_report``), ``receipts.json`` and
   ``stage_receipt.json`` are written.

Resume: a campaign with a valid checkpoint is loaded, not recomputed (the plan is not even lowered if every
campaign is done); a finished stage (valid ``stage_receipt.json``) is skipped. A checkpoint or receipt of
another identity, or a corrupt one, raises (never silently recomputed).
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import jax                                                                     # noqa: E402
jax.config.update("jax_enable_x64", True)

from drbx.stencils import artifact as art                                     # noqa: E402
from drbx.stencils.census import FaceCensus                                    # noqa: E402
from drbx.stencils.geometry_arrays import GeometryArrays                       # noqa: E402
from drbx.stencils.loader import LoaderGrid                                    # noqa: E402
from drbx.stencils.operator_plan import lower_perpendicular_plan_from_artifact, plan_nbytes  # noqa: E402

from p_shared import jax_replay as jr                                          # noqa: E402
from p_shared import replay_units as ru                                        # noqa: E402
from p_shared import runner                                                    # noqa: E402
from p_shared.replay_support import _json_default, build_environment, write_report  # noqa: E402
from p08_step2_global import comparison                                        # noqa: E402

__all__ = ["SCHEMA", "load_artifact", "artifact_summary", "lower_plan", "evaluate_campaign", "evaluate_campaigns",
           "checkpoint_unit", "checkpoint_identity", "run_replay_stage", "jax_info", "plan_summary"]

SCHEMA = "drbx.p08-step2b-jax-replay.v1"
STAGE_RECEIPT = "stage_receipt.json"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def jax_info() -> dict:
    """Backend / precision / threading facts recorded in every receipt."""
    import jaxlib
    return {"backend": jax.default_backend(), "x64": bool(jax.config.jax_enable_x64),
            "jax": jax.__version__, "jaxlib": jaxlib.__version__, "devices": [str(d) for d in jax.devices()],
            "cpu_count": os.cpu_count(),
            "env": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS",
                                                   "OPENBLAS_NUM_THREADS", "XLA_FLAGS")}}


@contextlib.contextmanager
def _phase(phases: dict, name: str):
    """Time a phase and record the peak RSS of this process inside it (``runner.reset_peak_rss`` window)."""
    method = runner.reset_peak_rss()
    started = time.perf_counter()
    yield
    phases[name] = {"seconds": time.perf_counter() - started, "peak_rss_gib": runner.peak_rss_gib(),
                    "rss_method": method}


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------
def artifact_summary(manifest: dict, receipt: dict | None = None) -> dict:
    """Bytes / files / sources per group of a v3 manifest plus the build receipt's tensor-encoding counts."""
    groups = {}
    for group, entries in manifest.get("chunks", {}).items():
        groups[group] = {"files": len(entries), "bytes": int(sum(e.get("bytes", 0) for e in entries)),
                         "sources": int(sum(e.get("sources", 0) for e in entries)),
                         "targets": int(sum(e.get("targets", 0) for e in entries))}
    summary = {"schema": manifest.get("schema"), "groups": groups,
               "total_bytes": int(sum(g["bytes"] for g in groups.values()))}
    if receipt is not None:
        diagnostics = receipt.get("diagnostics", {})
        summary["tensor_encoding"] = diagnostics.get("tensor_encoding", {})
        summary["build"] = {k: receipt.get(k) for k in ("wall_seconds", "cpu_seconds", "peak_rss_gib",
                                                       "effective_workers", "bytes_per_row_kind", "row_counts",
                                                       "reentry")}
        summary["point_row_families"] = diagnostics.get("point_row_families", {})
    return summary


def load_artifact(artifact_root, n: int, *, identity: dict | None = None,
                  require_tensor: bool = False) -> dict:
    """Open ``<artifact_root>/N{n}``: schema (v3 only), manifest identity and the (optional) build receipt.

    ``identity`` defaults to the grid's ``build_identity.json``. ``require_tensor`` refuses an artifact whose
    build receipt shows no tensor-encoded sources (a CSR-only build, ``P_SHARED_CSR_ONLY`` set)."""
    grid_dir = Path(artifact_root) / f"N{int(n)}"
    manifest = json.loads((grid_dir / "manifest.json").read_text())
    if manifest.get("schema") != art.SCHEMA:
        raise ValueError(f"the G3 replay needs a {art.SCHEMA} artifact, got {manifest.get('schema')!r}")
    if identity is None:
        identity = json.loads((grid_dir / "build_identity.json").read_text())
    if manifest.get("identity") != art._json_safe(identity):
        raise ValueError("row artifact identity mismatch")
    receipt_path = grid_dir / "build_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    summary = artifact_summary(manifest, receipt)
    if require_tensor:
        counts = summary.get("tensor_encoding") or {}
        if not counts.get("tensor_sources", 0):
            raise ValueError("the artifact holds no tensor-encoded sources (build_receipt.json diagnostics."
                             "tensor_encoding.tensor_sources == 0): rebuild with P_SHARED_CSR_ONLY unset")
    return {"grid_dir": grid_dir, "identity": identity, "manifest": manifest, "receipt": receipt,
            "summary": summary}


def plan_summary(plan, *, nbytes: int | None = None) -> dict:
    return {"cells": int(len(plan.cells.raw_ids)), "faces": int(len(plan.faces.census_row)),
            "p07_faces": int(len(plan.p07.p07_id)), "dirichlet_points": int(len(plan.dirichlet_points)),
            "neumann_points": int(len(plan.neumann_points)),
            "plan_bytes": int(plan_nbytes(plan) if nbytes is None else nbytes)}


def lower_plan(env, artifact: dict, artifact_root, n: int, *, selection: dict | None = None,
               device_put: bool = True):
    """The full-grid plan streamed from the artifact (or restricted to ``selection``: a dict with
    ``raw_ids`` / ``face_rows`` / ``p07_rows``, as the bounded test uses on an owner-closure artifact).
    Returns ``(plan, summary)``."""
    t = env.t
    grid_dir = artifact["grid_dir"]
    geometry = GeometryArrays.load(grid_dir / "geometry.npz")
    census_path = grid_dir / "census.npz"
    if census_path.is_file():
        stored = FaceCensus.load(census_path)
        if not (np.array_equal(stored.keys(), env.census.keys()) and np.array_equal(stored.owner_lo, env.census.owner_lo)
                and np.array_equal(stored.owner_hi, env.census.owner_hi)):
            raise ValueError("the artifact's census.npz differs from the environment's census")
    grid = LoaderGrid.from_arrays(n=int(n), raw_to_owner=t.ro, eta_centers=t.centers[2])
    kwargs = {} if selection is None else dict(raw_ids=selection["raw_ids"], face_rows=selection["face_rows"],
                                               p07_rows=selection["p07_rows"])
    plan = lower_perpendicular_plan_from_artifact(
        artifact_root, n, grid=grid, census=env.census, geometry=geometry, raw_volume=t.rv, owner_volume=t.vol,
        identity=artifact["identity"], **kwargs)
    summary = plan_summary(plan)
    if device_put:
        plan = jax.device_put(plan)
    return plan, summary


# ---------------------------------------------------------------------------
# Campaign evaluation and checkpoints
# ---------------------------------------------------------------------------
def checkpoint_unit(n: int, campaign: str) -> dict:
    """The ``runner`` unit under which a campaign's terms are checkpointed."""
    return {"stage": f"jax_{campaign}", "n": int(n), "start": 0, "stop": 1}


def checkpoint_identity(*, campaign_identity: str, artifact_identity: dict, n: int, cfg: dict) -> dict:
    """A checkpoint is valid only for this campaign identity, artifact identity and blocking/boundary policy."""
    return {"campaign": campaign_identity, "artifact": runner.digest(art._json_safe(artifact_identity)), "n": int(n),
            "blocking": {k: cfg.get(k) for k in ("column_block", "p06n_variant_block", "boundary_batch",
                                                 "wall_cache")}}


def evaluate_campaign(env, plan, oracle: dict, campaign: str, cfg: dict, log=None) -> tuple[dict, dict]:
    """One campaign on the shared plan: ``(out, record)`` (host-format terms; timings and peak RSS).
    ``log`` receives one line per boundary-data slice (the host callbacks can take many minutes)."""
    runner.reset_peak_rss()
    started = time.perf_counter()
    progress = None if log is None else (
        lambda label, done, total: log(f"  {label}: {done}/{total} boundary points "
                                       f"({time.perf_counter() - started:.0f} s)"))
    closure = jr.JaxOwnerClosure(env, None, (campaign,), oracle, wall_cache=bool(cfg.get("wall_cache", False)),
                                 plan=plan, column_block=cfg.get("column_block"),
                                 variant_block=cfg.get("p06n_variant_block"), boundary_batch=cfg.get("boundary_batch"),
                                 boundary_progress=progress)
    setup = time.perf_counter() - started
    out = closure.evaluate(include_host_only=False)
    record = {"campaign": campaign, "seconds": time.perf_counter() - started, "setup_seconds": setup,
              "boundary_data_seconds": closure.timings.get("boundary_data"),
              "operator_seconds": closure.timings.get("operators_nominal"),
              "peak_rss_gib": runner.peak_rss_gib()}
    del closure
    gc.collect()
    return out, record


def _merge(target: dict, data: dict) -> None:
    for section in ("cells", "faces", "p07"):
        target[section].update(data.get(section, {}))


def evaluate_campaigns(env, plan_factory, oracle: dict, campaigns, cfg: dict, *, output=None,
                       identity: dict | None = None, n: int | None = None, log=_log) -> tuple[dict, dict]:
    """Merged host-format ``out`` and ``{campaign: record}`` of ``campaigns`` (sequential, checkpointed).

    ``plan_factory()`` returns the plan (called at most once, only when some campaign needs computing).
    Without ``output`` nothing is checkpointed."""
    merged = {"cells": {}, "faces": {}, "p07": {}}
    records: dict = {}
    plan = None
    for campaign in campaigns:
        unit = checkpoint_unit(n if n is not None else env.n, campaign)
        if output is not None and runner.valid_unit(output, unit, identity):
            data = ru._read_unit_output(Path(output), unit)
            records[campaign] = {**json.loads(runner.receipt_path(output, unit).read_text()).get("record", {}),
                                 "resumed": True}
            log(f"{campaign}: valid checkpoint, skipped")
        else:
            if plan is None:
                plan = plan_factory()
            log(f"{campaign}: evaluating")
            started = time.time()
            data, record = evaluate_campaign(env, plan, oracle, campaign, cfg, log=log)
            records[campaign] = {**record, "resumed": False}
            if output is not None:
                runner.write_unit(output, unit, identity, chunks={"chunk": ru._pack_unit_arrays(data)},
                                  started=started, extra={"record": record})
            log(f"{campaign}: done in {record['seconds']:.1f} s (boundary data {record['boundary_data_seconds']:.1f} s), "
                f"peak RSS {record['peak_rss_gib']:.2f} GiB")
        _merge(merged, data)
    return merged, records


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def _rewrite_report(output: Path, replay: dict) -> Path:
    path = write_report(replay, output)
    lines = path.read_text().splitlines()
    lines[0] = f"# P08 step 2b G3 JAX full-grid replay gate -- N{replay['n']}"
    omitted = "; ".join(f"{name}: {', '.join(terms)}" for name, terms in comparison.OMITTED_TERMS.items())
    note = ("Operator terms only, computed by the JAX operators of the plan streamed from the schema-v3 artifact "
            "(same oracles, normalizations, region masks, Tier-B ratio and corrected pointwise cap as step 1). "
            f"Omitted host-only MMS reference terms (gated in step 1): {omitted}.")
    lines[1:1] = ["", note]
    path.write_text("\n".join(lines) + "\n")
    return path


def _stage_receipt_valid(output: Path, identity: dict) -> dict | None:
    path = output / STAGE_RECEIPT
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved["identity"] != identity:
        raise ValueError(f"stale replay stage receipt {path}: identity changed; use a new output folder")
    replay_path = output / "replay.json"
    if not replay_path.is_file() or runner.sha256_file(replay_path) != saved["replay_sha256"]:
        raise ValueError(f"replay.json does not match {path}")
    return saved


def run_replay_stage(*, artifact_root, output, n: int, input_root, sidecar_path, paths: dict, campaigns,
                     campaign_identity: str, cfg: dict, selection: dict | None = None, compare: bool = True,
                     return_out: bool = False, artifact_identity: dict | None = None, log=_log) -> dict:
    """Run (or resume) the JAX replay of grid ``n``; see the module docstring.

    Returns the stage receipt (``receipts.json``); with ``return_out`` also ``"out"``, the merged host-format
    terms. ``selection``/``compare=False`` is the bounded-test mode (no full-grid comparison)."""
    output = Path(output)
    campaigns = tuple(campaigns)
    if compare and selection is not None:
        raise ValueError("the oracle comparison needs the full grid (selection must be None)")
    started = time.time()
    phases: dict = {}
    artifact = load_artifact(artifact_root, n, identity=artifact_identity,
                             require_tensor=bool(cfg.get("tensor_encoding_required", False)) and selection is None)
    ident = checkpoint_identity(campaign_identity=campaign_identity, artifact_identity=artifact["identity"], n=n,
                                cfg=cfg)
    if compare:
        done = _stage_receipt_valid(output, ident)
        if done is not None:
            log(f"N{n}: replay stage already complete, skipped")
            return {**done, "skipped": True}
    runner.require_cpu_backend()

    with _phase(phases, "environment"):
        env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), curvature="fd")
        oracle = ru._load_oracle_owner_values(env, dict(paths), campaigns)
    log(f"N{n}: environment ready in {phases['environment']['seconds']:.1f} s")

    plan_info: dict = {}

    def plan_factory():
        with _phase(phases, "lower_plan"):
            plan, info = lower_plan(env, artifact, artifact_root, n, selection=selection,
                                    device_put=bool(cfg.get("device_put_plan", True)))
        plan_info.update(info)
        log(f"N{n}: plan lowered in {phases['lower_plan']['seconds']:.1f} s {info}")
        return plan

    with _phase(phases, "campaigns"):
        out, records = evaluate_campaigns(env, plan_factory, oracle, campaigns, cfg, output=output,
                                          identity=ident, n=n, log=log)
    gc.collect()
    # every campaign (and the plan lowering) opens its own peak-RSS window: the enclosing phase is their maximum
    phases["campaigns"]["peak_rss_gib"] = max([phases["campaigns"]["peak_rss_gib"]]
                                              + [r.get("peak_rss_gib", 0.0) for r in records.values()]
                                              + [phases.get("lower_plan", {}).get("peak_rss_gib", 0.0)])

    receipts = {"schema": SCHEMA, "n": int(n), "identity": ident, "artifact": artifact["summary"],
                "process_lifetime_peak_rss_gib": runner._ru_maxrss_gib(),
                "plan": plan_info or None, "campaigns": records, "phases": phases, "jax": jax_info(),
                "config": {k: cfg.get(k) for k in ("column_block", "p06n_variant_block", "boundary_batch",
                                                   "wall_cache", "device_put_plan")},
                "boundary_data_route": "live" if not cfg.get("wall_cache", False) else "wall_cache"}
    if return_out:
        receipts["out"] = out
    if not compare:
        receipts["wall_seconds"] = time.time() - started
        return receipts

    with _phase(phases, "compare"):
        results = comparison.compare_operator_terms(env=env, out=out, paths=dict(paths), campaigns=campaigns, n=n)
    receipts["wall_seconds"] = time.time() - started
    all_pass = all(term.get("pass", False) for result in results.values()
                   for term in result["terms"].values())
    replay = {"schema": SCHEMA, "n": int(n), "artifact_root": str(artifact_root),
              "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "wall_seconds": receipts["wall_seconds"], "campaigns": results, "campaign_identity": campaign_identity,
              "scope": {"operator_terms_only": True, "omitted_host_only_terms": comparison.OMITTED_TERMS},
              "all_terms_pass": bool(all_pass), "g3": {k: v for k, v in receipts.items() if k != "out"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True, default=_json_default))
    _rewrite_report(output, replay)
    runner.write_json(output / "receipts.json", {k: v for k, v in receipts.items() if k != "out"})
    runner.write_json(output / STAGE_RECEIPT, {"identity": ident, "replay_sha256": runner.sha256_file(output / "replay.json"),
                                               "all_terms_pass": bool(all_pass), "wall_seconds": receipts["wall_seconds"]})
    log(f"N{n}: replay stage complete in {receipts['wall_seconds']:.1f} s; all terms pass: {all_pass}")
    receipts["all_terms_pass"] = bool(all_pass)
    return receipts
