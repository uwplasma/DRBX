"""The step-4 smoke stage: one grid's final-operator row artifact -> plan -> JAX operators -> smoke check.

``run_smoke_stage`` is what ``campaign run`` / ``run-stage --stage replay`` runs per grid. It reuses the step-2 JAX
replay (``p08_step2_global.replay.run_replay_stage`` with ``operator_options`` = the pinned final options,
``compare=False``, ``return_out=True``): plan streamed from the artifact, the seven campaign keys evaluated
sequentially, per-campaign checkpoints under ``<output>/_chunks``. Then:

1. **Smoke (the gate).** Every floating-point term array of the merged host-format ``out`` -- the sparse owner
   numerators of every campaign key, the per-face P05 jump values -- is finite (no NaN / Inf) on every owner and
   face; counts, non-finite counts and ``max |value|`` are reported per campaign and term. A campaign key without
   any floating-point output also fails.
2. **Change versus the frozen oracles (informational, never a gate).** ``comparison.compare_operator_terms`` with
   the final-options environment against the frozen step-1 oracles, wrapped so that an exception is recorded
   instead of raised. The frozen oracles describe the old fd / q3 / profile7 operator, so a Tier-B ratio there is
   the operator change relative to the archived spatial error, not an error; ``pass`` fields are not verdicts.
3. **Statistics**: artifact bytes (per row kind and total), row counts, point-row family counts (including
   ``coupled_quartic``), tensor-encoding counts and fallbacks, build wall / CPU seconds and peak RSS; replay
   timings and peak RSS per phase and campaign; plan bytes.

Outputs under ``<output>`` (= ``replay/N{n}``): ``smoke.json``, ``report.md``, ``change_vs_frozen_oracles.json``
(the full comparison, when it succeeded), ``plan_info.json``, ``smoke_receipt.json`` (identity + sha256 of
``smoke.json``: a finished stage is skipped on resume) and the checkpoints.
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared import runner                                                    # noqa: E402
from p_shared.replay_support import build_environment                          # noqa: E402
from p08_step2_global import comparison, replay                                # noqa: E402

__all__ = ["SCHEMA", "SMOKE_RECEIPT", "campaign_of", "finite_report", "check_artifact_options", "artifact_stats",
           "change_vs_frozen_oracles", "sanitize", "run_smoke_stage", "write_report"]

SCHEMA = "drbx.p08-step4-smoke.v1"
SMOKE_RECEIPT = "smoke_receipt.json"
CHANGE_NOTE = ("INFORMATIONAL, NOT A GATE. The frozen step-1 oracles describe the old fd / q3 / profile7 operator; "
               "with the final operator (autodiff / q2 / fixed_radius) the Tier-B ratio below is the operator change "
               "relative to the archived spatial error of that old operator, and a failing 'nominal_pass' is expected.")

#: campaign -> the prefix of its term keys in the host-format ``out`` (``p06_legacy`` keys start with ``p06legacy``)
_KEY_PREFIX = {"p05": "p05", "p05n_frozen": "p05n_frozen", "p05n_upwind": "p05n_upwind", "p06n": "p06n",
               "p06_legacy": "p06legacy", "p07": "p07", "p07n": "p07n"}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextlib.contextmanager
def _phase(phases: dict, name: str):
    method = runner.reset_peak_rss()
    started = time.perf_counter()
    yield
    phases[name] = {"seconds": time.perf_counter() - started, "peak_rss_gib": runner.peak_rss_gib(),
                    "rss_method": method}


def sanitize(obj):
    """JSON-safe copy: numpy scalars/arrays to Python, non-finite floats to the strings ``"nan"`` / ``"inf"`` /
    ``"-inf"`` (``runner.write_json`` refuses NaN; a smoke report must be able to *say* NaN)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return "nan" if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


# ---------------------------------------------------------------------------
# The smoke check
# ---------------------------------------------------------------------------
def campaign_of(key: str) -> str | None:
    """The campaign a host-format term key belongs to (longest matching prefix), else ``None`` (shared keys such as
    ``q1_evolution_volume``)."""
    best, best_len = None, -1
    for name, prefix in _KEY_PREFIX.items():
        if (key == prefix or key.startswith(prefix + "_")) and len(prefix) > best_len:
            best, best_len = name, len(prefix)
    return best


def _leaves(obj):
    """Every numeric leaf of a (nested) host-format value: arrays and scalars, through dicts, lists and the
    ``(uniq, numerator)`` tuples."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _leaves(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _leaves(value)
    elif obj is None or isinstance(obj, (str, bytes, bool, np.bool_)):
        return
    elif isinstance(obj, (np.ndarray, np.generic, int, float, complex)):
        yield np.asarray(obj)


def _array_stats(arrays) -> dict:
    stats = {"arrays": 0, "elements": 0, "nan": 0, "inf": 0, "nonfinite": 0, "max_abs": 0.0}
    for a in arrays:
        if not np.issubdtype(a.dtype, np.inexact):        # integer ids / ranks cannot be non-finite
            continue
        finite = np.isfinite(a)
        n_nan = int(np.count_nonzero(np.isnan(a)))
        n_bad = int(a.size - np.count_nonzero(finite))
        stats["arrays"] += 1
        stats["elements"] += int(a.size)
        stats["nan"] += n_nan
        stats["inf"] += n_bad - n_nan
        stats["nonfinite"] += n_bad
        if n_bad == a.size:
            continue
        mag = np.abs(a[finite]) if n_bad else np.abs(a)
        if mag.size:
            stats["max_abs"] = max(stats["max_abs"], float(np.max(mag)))
    return stats


def finite_report(out: dict, campaigns) -> dict:
    """The smoke check of the merged host-format ``out`` (``{"cells": ..., "faces": ..., "p07": ...}``).

    ``all_finite`` is True iff no floating-point leaf holds NaN / Inf *and* every campaign in ``campaigns`` has at
    least one floating-point output (``missing_campaigns`` otherwise). ``campaigns[<name>]["terms"]`` is keyed by
    ``"<section>.<key>"``."""
    by_campaign: dict = {name: {"terms": {}} for name in campaigns}
    shared: dict = {}
    for section in ("cells", "faces", "p07"):
        for key, value in out.get(section, {}).items():
            stats = _array_stats(_leaves(value))
            owner = campaign_of(key)
            if owner is None or owner not in by_campaign:
                shared[f"{section}.{key}"] = stats
            else:
                by_campaign[owner]["terms"][f"{section}.{key}"] = stats
    total_nonfinite = total_elements = 0
    missing = []
    for name, entry in by_campaign.items():
        terms = entry["terms"]
        entry.update(elements=sum(t["elements"] for t in terms.values()),
                     nonfinite=sum(t["nonfinite"] for t in terms.values()),
                     max_abs=max([t["max_abs"] for t in terms.values()], default=0.0))
        entry["all_finite"] = entry["nonfinite"] == 0 and entry["elements"] > 0
        if entry["elements"] == 0:
            missing.append(name)
        total_nonfinite += entry["nonfinite"]
        total_elements += entry["elements"]
    for stats in shared.values():
        total_nonfinite += stats["nonfinite"]
        total_elements += stats["elements"]
    return {"all_finite": bool(total_nonfinite == 0 and not missing), "nonfinite_total": int(total_nonfinite),
            "elements_total": int(total_elements), "missing_campaigns": missing, "campaigns": by_campaign,
            "shared": shared}


# ---------------------------------------------------------------------------
# Informational: change versus the frozen oracles
# ---------------------------------------------------------------------------
def _max_abs_diff(term: dict):
    if "variants" in term:
        values = [v["pointwise"]["max_abs_diff"] for v in term["variants"] if "pointwise" in v]
    elif "pointwise" in term:
        values = [term["pointwise"]["max_abs_diff"]]
    else:
        values = []
    return max(values) if values else None


def summarize_change(results: dict) -> dict:
    """Per campaign and term: the Tier-B ratio (operator change relative to the archived spatial error), the region
    where it is largest, the largest absolute difference, the pointwise-cap violations (informational) and the
    nominal pass flag of the step-2 comparison (NOT a verdict here)."""
    campaigns: dict = {}
    worst, worst_term = None, None
    for name, result in results.items():
        terms = {}
        for term, value in result.get("terms", {}).items():
            ratio = value.get("worst_ratio")
            violations = (value["pointwise"]["violations"] if "pointwise" in value
                          else value.get("total_pointwise_violations"))
            terms[term] = {"tier_b_ratio": ratio, "worst_region": value.get("worst_region"),
                           "max_abs_diff": _max_abs_diff(value), "pointwise_violations": violations,
                           "nominal_pass": value.get("pass")}
            if ratio is not None and (worst is None or not ratio <= worst):      # NaN counts as worst
                worst, worst_term = ratio, f"{name}.{term}"
        campaigns[name] = terms
    return {"campaigns": campaigns, "max_tier_b_ratio": worst, "max_tier_b_ratio_term": worst_term}


def change_vs_frozen_oracles(*, env, out: dict, paths: dict, campaigns, n: int, full_path: Path | None = None) -> dict:
    """The informational comparison with the frozen oracles; an exception is recorded, never raised."""
    record = {"informational": True, "gate": False, "note": CHANGE_NOTE}
    started = time.perf_counter()
    try:
        results = comparison.compare_operator_terms(env=env, out=out, paths=dict(paths), campaigns=tuple(campaigns),
                                                    n=n)
        record.update(status="ok", **summarize_change(results))
        if full_path is not None:
            Path(full_path).write_text(json.dumps(sanitize(results), indent=1, sort_keys=True))
            record["full_results"] = Path(full_path).name
    except Exception as exc:                                    # noqa: BLE001 -- recorded, not fatal, by design
        record.update(status="error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
    record["seconds"] = time.perf_counter() - started
    return record


# ---------------------------------------------------------------------------
# Artifact statistics
# ---------------------------------------------------------------------------
def check_artifact_options(identity: dict, options: dict) -> None:
    """The artifact's recorded build policy must carry exactly the pinned options (a non-default option enters
    ``build_artifact.build_policy``; the receipts of a default / other-option build never match)."""
    policy = identity.get("policy", {})
    seen = {"curvature": policy.get("curvature", "fd"),
            "face_quadrature": policy.get("quadrature", {}).get("face", "q3"),
            "inner_support": policy.get("inner_support", "profile7")}
    if seen != dict(options):
        raise ValueError(f"the artifact was built with operator options {seen}, not the pinned {dict(options)}")


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return int(total)


def artifact_stats(artifact_root, n: int, *, loaded: dict | None = None) -> dict:
    """Bytes, row counts, family counts, tensor-encoding counts and build timings of ``<artifact_root>/N{n}``."""
    loaded = loaded or replay.load_artifact(artifact_root, n)
    grid_dir = loaded["grid_dir"]
    receipt = loaded["receipt"] or {}
    diagnostics = receipt.get("diagnostics", {})
    families = dict(diagnostics.get("point_row_families", {}))
    per_kind = dict(receipt.get("bytes_per_row_kind") or {g: v["bytes"] for g, v in loaded["summary"]["groups"].items()})
    policy = loaded["identity"].get("policy", {})
    return {
        "path": str(grid_dir.resolve()),
        "policy": {k: policy.get(k) for k in ("curvature", "quadrature", "inner_support")},
        "bytes_per_row_kind": per_kind,
        "row_bytes_total": int(sum(per_kind.values())),
        "geometry_bytes": int((grid_dir / "geometry.npz").stat().st_size) if (grid_dir / "geometry.npz").is_file() else None,
        "total_bytes": _tree_bytes(grid_dir),
        "manifest_groups": loaded["summary"]["groups"],
        "row_counts": receipt.get("row_counts"),
        "total_point_rows": diagnostics.get("total_point_rows"),
        "total_neumann_rows": diagnostics.get("total_neumann_rows"),
        "total_integrated_rows": diagnostics.get("total_integrated_rows"),
        "point_row_families": families,
        "coupled_quartic": int(families.get("coupled_quartic", 0)),
        "p07_integrated_row_families": diagnostics.get("p07_integrated_row_families", {}),
        "tensor_encoding": diagnostics.get("tensor_encoding", {}),
        "max_point_row_residual": diagnostics.get("max_point_row_residual"),
        "max_neumann_constraint_residual": diagnostics.get("max_neumann_constraint_residual"),
        "max_neumann_condition": diagnostics.get("max_neumann_condition"),
        "build": {k: receipt.get(k) for k in ("wall_seconds", "cpu_seconds", "peak_rss_gib", "effective_workers",
                                              "chunk_files", "reentry", "disk_free_gib_before",
                                              "disk_free_gib_after")},
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    return "n/a" if x is None else (x if isinstance(x, str) else f"{x:.3e}")


def write_report(smoke: dict, output: Path) -> Path:
    n = smoke["n"]
    art, fin, change = smoke["artifact"], smoke["finite"], smoke["change_vs_frozen_oracles"]
    lines = [f"# P08 step 4 smoke check of the final bundle operator -- N{n}", "",
             f"Operator options: `{json.dumps(smoke['operator_options'], sort_keys=True)}`. "
             "Smoke only: no MMS acceptance gate (the frozen oracles describe the old operator).", "",
             f"**Smoke: {'PASS' if smoke['smoke_pass'] else 'FAIL'}** -- {fin['nonfinite_total']} non-finite of "
             f"{fin['elements_total']} term elements"
             + (f"; campaigns without output: {fin['missing_campaigns']}" if fin["missing_campaigns"] else "") + ".", "",
             "| Campaign | Elements | Non-finite | max abs |", "|---|---|---|---|"]
    for name, entry in fin["campaigns"].items():
        lines.append(f"| {name} | {entry['elements']} | {entry['nonfinite']} | {_fmt(entry['max_abs'])} |")
    lines += ["", "## Artifact", "",
              f"`{art['path']}`: total {art['total_bytes'] / 2 ** 30:.2f} GiB (rows {art['row_bytes_total'] / 2 ** 30:.2f} GiB; "
              + ", ".join(f"{k} {v / 2 ** 30:.2f}" for k, v in art["bytes_per_row_kind"].items()) + " GiB).", "",
              f"Point-row families: `{json.dumps(art['point_row_families'], sort_keys=True)}` "
              f"(coupled_quartic: {art['coupled_quartic']}). Tensor encoding: `{json.dumps(art['tensor_encoding'], sort_keys=True)}`.", "",
              f"Build: {_fmt(art['build'].get('wall_seconds'))} s wall, peak worker RSS "
              f"{_fmt(art['build'].get('peak_rss_gib'))} GiB.", "",
              "## Replay", "",
              f"{smoke['replay']['wall_seconds']:.0f} s wall; plan {json.dumps(smoke['replay'].get('plan'))}.", "",
              "| Phase | Seconds | Peak RSS (GiB) |", "|---|---|---|"]
    for name, phase in smoke["replay"]["phases"].items():
        lines.append(f"| {name} | {phase['seconds']:.1f} | {phase['peak_rss_gib']:.2f} |")
    lines += ["", "| Campaign | Seconds | Peak RSS (GiB) | Resumed |", "|---|---|---|---|"]
    for name, rec in smoke["replay"]["campaigns"].items():
        lines.append(f"| {name} | {_fmt(rec.get('seconds'))} | {_fmt(rec.get('peak_rss_gib'))} | {rec.get('resumed')} |")
    lines += ["", "## Change versus the frozen oracles (informational, not a gate)", "", change["note"], ""]
    if change["status"] != "ok":
        lines.append(f"Comparison did not run: `{change.get('error')}`")
    else:
        lines += [f"Largest Tier-B ratio: {_fmt(change['max_tier_b_ratio'])} ({change['max_tier_b_ratio_term']}).", "",
                  "| Term | Tier-B ratio | Region | max abs diff |", "|---|---|---|---|"]
        for name, terms in change["campaigns"].items():
            for term, rec in terms.items():
                lines.append(f"| {name}.{term} | {_fmt(rec['tier_b_ratio'])} | {rec['worst_region'] or ''} | "
                             f"{_fmt(rec['max_abs_diff'])} |")
    path = Path(output) / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _receipt_valid(output: Path, identity: dict) -> dict | None:
    path = output / SMOKE_RECEIPT
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved["identity"] != identity:
        raise ValueError(f"stale smoke receipt {path}: identity changed; use a new output folder")
    smoke_path = output / "smoke.json"
    if not smoke_path.is_file() or runner.sha256_file(smoke_path) != saved["smoke_sha256"]:
        raise ValueError(f"smoke.json does not match {path}")
    return saved


def run_smoke_stage(*, artifact_root, output, n: int, input_root, sidecar_path, paths: dict, campaigns,
                    campaign_identity: str, cfg: dict, operator_options: dict, log=_log) -> dict:
    """Run (or resume) the smoke stage of grid ``n``; returns the stage summary (``smoke_pass``, wall seconds,
    ``skipped``). See the module docstring."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = dict(operator_options)
    campaigns = tuple(campaigns)
    artifact = replay.load_artifact(artifact_root, n, require_tensor=bool(cfg.get("tensor_encoding_required", False)))
    check_artifact_options(artifact["identity"], options)
    ident = replay.checkpoint_identity(campaign_identity=campaign_identity, artifact_identity=artifact["identity"],
                                      n=n, cfg=cfg)
    done = _receipt_valid(output, ident)
    if done is not None:
        log(f"N{n}: smoke stage already complete, skipped")
        return {**done, "skipped": True}

    started = time.time()
    result = replay.run_replay_stage(
        artifact_root=artifact_root, output=output, n=n, input_root=input_root, sidecar_path=sidecar_path,
        paths=paths, campaigns=campaigns, campaign_identity=campaign_identity, cfg=cfg, compare=False,
        return_out=True, artifact_identity=artifact["identity"], operator_options=options, log=log)
    out = result.pop("out")
    plan_info_path = output / "plan_info.json"
    if result.get("plan"):
        runner.write_json(plan_info_path, result["plan"])
    plan_info = result.get("plan") or (json.loads(plan_info_path.read_text()) if plan_info_path.is_file() else None)
    phases = dict(result["phases"])

    with _phase(phases, "smoke_finite_check"):
        finite = finite_report(out, campaigns)
    log(f"N{n}: smoke finite check: {finite['nonfinite_total']} non-finite of {finite['elements_total']} elements "
        f"({'PASS' if finite['all_finite'] else 'FAIL'})")

    with _phase(phases, "change_vs_frozen_oracles"):
        env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), **options)
        change = change_vs_frozen_oracles(env=env, out=out, paths=paths, campaigns=campaigns, n=n,
                                          full_path=output / "change_vs_frozen_oracles.json")
    del env, out
    gc.collect()
    log(f"N{n}: change versus the frozen oracles ({change['status']}): max Tier-B ratio "
        f"{change.get('max_tier_b_ratio')} (informational)")

    wall = time.time() - started
    stats = artifact_stats(artifact_root, n, loaded=artifact)
    smoke = sanitize({
        "schema": SCHEMA, "n": int(n), "campaign_identity": campaign_identity, "operator_options": options,
        "smoke_pass": bool(finite["all_finite"]), "finite": finite, "artifact": stats,
        "replay": {"wall_seconds": wall, "plan": plan_info, "phases": phases, "campaigns": result["campaigns"],
                   "process_lifetime_peak_rss_gib": result.get("process_lifetime_peak_rss_gib"),
                   "jax": result["jax"], "config": result["config"],
                   "peak_rss_gib": max([p["peak_rss_gib"] for p in phases.values()])},
        "change_vs_frozen_oracles": change,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    runner.write_json(output / "smoke.json", smoke)
    write_report(smoke, output)
    summary = {"identity": ident, "smoke_sha256": runner.sha256_file(output / "smoke.json"),
               "smoke_pass": smoke["smoke_pass"], "wall_seconds": wall}
    runner.write_json(output / SMOKE_RECEIPT, summary)
    log(f"N{n}: smoke stage complete in {wall:.1f} s; smoke pass: {smoke['smoke_pass']}")
    return summary
