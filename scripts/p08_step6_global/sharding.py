"""The ``sharding`` stage of the P08 step-6 campaign: eta-sharded RHS and potential solve against the single device.

``campaign.run_grid`` calls :func:`run` after ``reduce`` of every grid and stores the returned record verbatim in
``N{n}/sharding.json`` (with the campaign identity); ``campaign.validate`` reads it. The record has a ``"status"``:
``"pass"`` (every gate of every requested shard count passed), ``"fail"`` (a gate failed; makes ``validate`` report
``grids_pass = False``) or ``"pending"`` (no gate failed but not every configured shard count was run, because
``--sharding-max-shards`` capped the list; ``validate`` records the stage as pending and the stage is redone without the
cap). ``"not_implemented"`` is kept for old records. Also reachable as ``run-stage --stage sharding --n N``; not run by
``preflight``.

What is checked per grid ``n`` (the plan, the potential solver and the states are those of the JAX stage: this package's
:func:`jaxstage.prepare`, boundary data of 5.3's ``variant_boundary_data`` / ``psi_boundary_data``):

* **RHS**: the prescribed-phi arm (``phi = phi_bar``), all four fields and the terms ``poisson_bracket``, ``curvature``,
  ``perpendicular_diffusion``, ``total``, by the unchanged ``perpendicular_rhs`` on one device and by
  ``sharded_perpendicular_rhs`` over ``Sz`` eta shards, for the variants ``transverse_dirichlet`` and the catalogue
  ``main_phi_dirichlet`` (P06N owner values from the oracle root, exactly as 5.3). Metric per (variant, Sz, field, term):
  ``max|sharded - single| / max|single|`` over all owners (back in the global owner order); gate <= ``rhs_gate``
  (1e-12); whether the arrays are bitwise equal is recorded.
* **phi**: the Dirichlet psi solve of each variant, cold start, the measurement rtol of the campaign: single-device
  ``solve_phi`` against ``sharded_solve_phi``. Gate: both converged, iterations differ by <= 1 and ``||psi_sharded -
  psi_single||_M <= 10 rtol ||psi_single||_M``. The right-hand side is the reference stage's ``O(psi)`` for the
  transverse variant (as the JAX stage's solved arm); the catalogue variant has no reference in this campaign
  (``p08_step5_combined`` references are not part of it), so its right-hand side is the discrete ``A psi_bar + B g``
  (``rhs_source`` is recorded; single and sharded solve the same system, which is what the gate compares).

It runs in a SUBPROCESS (this file as a script, ``--worker``) with ``XLA_FLAGS=--xla_force_host_platform_device_count=8``
(the flag must precede the JAX import), so the campaign process stays single-device. The worker's log is
``logs/sharding_N{n}.log``; its spec / result are ``work/sharding/N{n}/{spec,result}.json``. Per shard count the sharded
plan and solver are freed before the next one. The worker records raw metrics; :func:`assess` applies the gates in the
parent. This module imports no JAX at import time.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCHEMA = "drbx.p08-step6-sharding.v1"
PENDING_STATUSES = ("not_implemented", "pending")
PASS_STATUSES = ("pass",)
FAIL_STATUSES = ("fail",)
KNOWN_STATUSES = PENDING_STATUSES + PASS_STATUSES + FAIL_STATUSES
RHS_TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion", "total")
FORCE_FLAG = "--xla_force_host_platform_device_count"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Parent side: shard counts, the subprocess, the gates
# ---------------------------------------------------------------------------
def shard_counts(n: int, scfg: dict, max_shards: int | None = None) -> tuple[list, list]:
    """``(run, skipped)``: the configured shard counts that divide ``n`` with a block of at least ``min_block_planes``
    planes, and of those the ones above ``max_shards`` (``None``: no cap) are moved to ``skipped``."""
    eligible = [int(s) for s in scfg["shard_counts"] if n % int(s) == 0 and n // int(s) >= int(scfg["min_block_planes"])]
    if max_shards is None:
        return eligible, []
    return [s for s in eligible if s <= int(max_shards)], [s for s in eligible if s > int(max_shards)]


def worker_environment(devices: int, base: dict | None = None) -> dict:
    """The worker's environment: the campaign's CPU / x64 / thread settings and ``devices`` forced host devices (any
    earlier forced-device flag in ``XLA_FLAGS`` is replaced)."""
    env = dict(os.environ if base is None else base)
    flags = [f for f in env.get("XLA_FLAGS", "").split() if not f.startswith(FORCE_FLAG)]
    env["XLA_FLAGS"] = " ".join([*flags, f"{FORCE_FLAG}={int(devices)}"])
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "true"
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
    return env


def worker_command(spec_path: Path, result_path: Path) -> list:
    """The subprocess command (this file as a script); tests substitute another worker through this function."""
    return [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path), "--result", str(result_path)]


def stage_paths(output, n: int) -> SimpleNamespace:
    output = Path(output)
    work = output / "work" / "sharding" / f"N{int(n)}"
    return SimpleNamespace(work=work, spec=work / "spec.json", result=work / "result.json",
                           log=output / "logs" / f"sharding_N{int(n)}.log", record=output / f"N{int(n)}" / "sharding.json")


def stored_record(output, n: int, identity: str) -> dict | None:
    """The stored record of this identity with a final status (``pass`` / ``fail``), else ``None`` (resume rule)."""
    path = stage_paths(output, n).record
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get("identity") != identity or saved.get("status") not in PASS_STATUSES + FAIL_STATUSES:
        return None
    return saved


def _entry_pass_rhs(entry: dict, gate: float) -> bool:
    worst = entry.get("worst_rel")
    return bool(entry.get("finite") and worst is not None and worst == worst and worst <= gate)


def _entry_pass_phi(entry: dict, single: dict, rtol: float, iteration_tolerance: int, factor: float) -> dict:
    iterations = abs(int(entry["iterations"]) - int(single["iterations"]))
    tolerance = float(factor) * float(rtol) * float(entry["psi_m"])
    checks = {"converged": bool(single["converged"] and entry["converged"]),
              "iterations": iterations <= int(iteration_tolerance),
              "solution": bool(entry["dpsi_m"] == entry["dpsi_m"] and entry["dpsi_m"] <= tolerance)}
    return {"iteration_difference": iterations, "solution_tolerance_m": tolerance, "checks": checks,
            "pass": bool(all(checks.values()))}


def assess(result: dict, scfg: dict, *, requested: list | None = None) -> dict:
    """Apply the gates to the worker's raw ``result`` and return the stage record (a new dict).

    ``rhs[variant]["Sz{s}"]`` needs ``finite`` and ``worst_rel``; ``phi[variant]`` needs ``single`` and ``Sz{s}`` with
    ``converged``, ``iterations``, ``dpsi_m``, ``psi_m``; ``phi_rtol`` is the solve rtol. A missing entry for a shard count
    that was run fails the stage. ``requested`` (default: the shard counts run) is the configured list: when some of it
    was not run and nothing failed, the status is ``pending``."""
    record = json.loads(json.dumps(result))
    run_counts = [int(s) for s in record.get("shard_counts", [])]
    requested = run_counts if requested is None else [int(s) for s in requested]
    gate = float(scfg["rhs_gate"])
    rtol = float(record["phi_rtol"])
    failures: list = []
    for variant, entries in (record.get("rhs") or {}).items():
        for sz in run_counts:
            entry = entries.get(f"Sz{sz}")
            if entry is None:
                failures.append(f"rhs/{variant}/Sz{sz}: missing")
                continue
            entry["gate"] = gate
            entry["pass"] = _entry_pass_rhs(entry, gate)
            if not entry["pass"]:
                failures.append(f"rhs/{variant}/Sz{sz}: worst_rel {entry.get('worst_rel')} (gate {gate})")
    for variant, entries in (record.get("phi") or {}).items():
        single = entries["single"]
        for sz in run_counts:
            entry = entries.get(f"Sz{sz}")
            if entry is None:
                failures.append(f"phi/{variant}/Sz{sz}: missing")
                continue
            entry.update(_entry_pass_phi(entry, single, rtol, int(scfg["phi_iteration_tolerance"]),
                                         float(scfg["phi_solution_rtol_factor"])))
            if not entry["pass"]:
                bad = [k for k, v in entry["checks"].items() if not v]
                failures.append(f"phi/{variant}/Sz{sz}: {bad} (iterations {entry['iterations']} vs "
                                f"{single['iterations']}, dpsi_m {entry['dpsi_m']}, psi_m {entry['psi_m']})")
    if not record.get("rhs") or not record.get("phi") or not run_counts:
        failures.append("no RHS / phi / shard-count results")
    complete = sorted(run_counts) == sorted(requested)
    rhs_ok = bool(record.get("rhs")) and all(e["pass"] for v in record["rhs"].values()
                                              for k, e in v.items() if k.startswith("Sz"))
    phi_ok = bool(record.get("phi")) and all(e["pass"] for v in record["phi"].values()
                                              for k, e in v.items() if k.startswith("Sz"))
    bitwise = bool(record.get("rhs")) and all(e.get("bitwise") for v in record["rhs"].values()
                                               for k, e in v.items() if k.startswith("Sz"))
    status = "fail" if failures else ("pass" if complete else "pending")
    record.update(status=status, gates={"rhs": rhs_ok, "phi": phi_ok, "complete": complete, "rhs_bitwise": bitwise,
                                        "rhs_gate": gate, "phi_iteration_tolerance": int(scfg["phi_iteration_tolerance"]),
                                        "phi_solution_rtol_factor": float(scfg["phi_solution_rtol_factor"])},
                  failures=failures, shard_counts_requested=requested)
    return record


def run(*, n: int, args, identity: str, inputs: dict, cfg: dict, log=_log) -> dict:
    """Run (or skip, when ``N{n}/sharding.json`` of this identity is final) the sharding stage of grid ``n``."""
    scfg = cfg["sharding"]
    output = Path(args.output)
    done = stored_record(output, n, identity)
    if done is not None:
        log(f"N{n}: sharding record of this identity is {done['status']}, skipped")
        return {k: v for k, v in done.items() if k not in ("identity", "n")}
    max_shards = getattr(args, "sharding_max_shards", None)
    if max_shards is None:
        max_shards = scfg.get("max_shards")
    counts, skipped = shard_counts(n, scfg, max_shards)
    if not counts:
        raise ValueError(f"N{n}: no shard count of {scfg['shard_counts']} is admissible (cap {max_shards})")
    paths = stage_paths(output, n)
    for folder in (paths.work, paths.log.parent):
        folder.mkdir(parents=True, exist_ok=True)
    paths.result.unlink(missing_ok=True)
    spec = {"schema": SCHEMA, "n": int(n), "identity": identity, "shard_counts": counts, "cfg": cfg, "inputs": inputs,
            "output": str(output), "input_root": str(args.input_root),
            "oracle_root": None if getattr(args, "oracle_root", None) is None else str(args.oracle_root)}
    paths.spec.write_text(json.dumps(spec, default=str))
    command = worker_command(paths.spec, paths.result)
    log(f"N{n}: sharding worker, shard counts {counts}, {scfg['devices']} forced host devices, log {paths.log}")
    started = time.time()
    with open(paths.log, "ab") as handle:
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} sharding N{n} {command}\n".encode())
        handle.flush()
        done_proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=worker_environment(scfg["devices"]), check=False)
    if done_proc.returncode != 0 or not paths.result.is_file():
        tail = "\n".join(paths.log.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"the sharding worker of N{n} failed (exit {done_proc.returncode}); log {paths.log}:\n{tail}")
    result = json.loads(paths.result.read_text())
    if result.get("n") != int(n) or result.get("identity") != identity:
        raise ValueError("the sharding worker result belongs to another grid / identity")
    record = assess(result, scfg, requested=[*counts, *skipped])
    record.update(shard_counts_skipped=skipped, max_shards=max_shards, wall_seconds_parent=time.time() - started,
                  schema=SCHEMA, log=str(paths.log))
    log(f"N{n}: sharding {record['status']} (rhs {record['gates']['rhs']}, phi {record['gates']['phi']}, "
        f"bitwise {record['gates']['rhs_bitwise']}, peak RSS {record.get('peak_rss_gib', {}).get('final')} GiB)")
    return record


# ---------------------------------------------------------------------------
# Worker side (JAX): the checks themselves
# ---------------------------------------------------------------------------
def peak_rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                    # bytes on macOS, KiB on Linux
    return peak / 2 ** 30 if sys.platform == "darwin" else peak / 2 ** 20


def _collect(out) -> dict:
    """``{(field, term): float64 owner array}`` of a ``PerpendicularTerms`` for the four gate terms."""
    import numpy as np

    from drbx.native.fci_perpendicular_rhs import FIELDS

    arrays = {}
    for field in FIELDS:
        for term in RHS_TERMS:
            source = out.total[field] if term == "total" else out.terms[field][term]
            arrays[(field, term)] = np.asarray(source, dtype=np.float64)
    return arrays


def _compare(single: dict, sharded: dict, perm, mask) -> dict:
    """Per (field, term) ``max|sharded - single| / max|single|`` over ``mask`` (``None``: all owners)."""
    import numpy as np

    fields, worst, bitwise, finite = {}, 0.0, True, True
    for key, a in single.items():
        b = sharded[key][perm]                                                   # plane-major -> global owner order
        if mask is not None:
            a, b = a[mask], b[mask]
        ok = bool(np.isfinite(a).all() and np.isfinite(b).all())
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        exact = bool(ok and diff == 0.0)
        if not ok:
            rel = float("inf")
        elif scale > 0.0:
            rel = diff / scale
        else:
            rel = 0.0 if diff == 0.0 else float("inf")
        fields[f"{key[0]}/{key[1]}"] = {"scale": scale, "diff": diff, "rel": rel, "bitwise": exact}
        worst, bitwise, finite = max(worst, rel), bitwise and exact, finite and ok
    return {"worst_rel": worst, "bitwise": bitwise, "finite": finite, "fields": fields}


def _block(tree):
    import jax

    return jax.block_until_ready(tree)


def single_rhs(setup, params, log) -> dict:
    """The single-device RHS of every case (two calls: compile + run); returns ``{case: {...arrays...}}``."""
    import time as _time

    from drbx.native.fci_perpendicular_rhs import perpendicular_rhs

    out = {}
    for case in setup.cases:
        t0 = _time.perf_counter()
        first = _block(perpendicular_rhs(setup.plan, case.state, case.phi, case.bc, case.kinds, params).total)
        t1 = _time.perf_counter()
        res = perpendicular_rhs(setup.plan, case.state, case.phi, case.bc, case.kinds, params)
        _block(res.total)
        t2 = _time.perf_counter()
        del first
        arrays = _collect(res)
        del res
        out[case.name] = {"arrays": arrays, "first_seconds": t1 - t0, "seconds": t2 - t1}
        log(f"single-device RHS {case.name}: first {t1 - t0:.1f} s, second {t2 - t1:.1f} s")
    return out


def sharded_rhs(setup, sz: int, singles: dict, params, log) -> dict:
    """Lower the plan over ``sz`` shards, run every case twice and compare with the single-device arrays; the sharded plan is
    freed on return."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from drbx.native.fci_perpendicular_rhs import FIELDS
    from drbx.native.fci_perpendicular_sharding import (
        make_plane_mesh, place_sharded, plane_major_permutation, shard_boundary_data, shard_perpendicular_plan,
        sharded_perpendicular_rhs, to_plane_major)

    perm, inverse, _m = plane_major_permutation(setup.raw_to_owner, setup.n)
    mesh = make_plane_mesh(sz)
    t0 = time.perf_counter()
    host = jax.tree_util.tree_map(np.asarray, setup.plan)                 # zero-copy views of the CPU arrays
    sharded = shard_perpendicular_plan(host, setup.raw_to_owner, setup.n, sz, setup.halo)
    lowering = time.perf_counter() - t0
    bcs = {case.name: place_sharded(shard_boundary_data(case.bc, sharded), mesh) for case in setup.cases}
    placed = place_sharded(sharded, mesh)
    del sharded, host
    gc.collect()
    out = {}
    for case in setup.cases:
        state_pm = {f: jnp.asarray(to_plane_major(case.state[f], inverse)) for f in FIELDS}
        phi_pm = jnp.asarray(to_plane_major(case.phi, inverse))
        t0 = time.perf_counter()
        first = _block(sharded_perpendicular_rhs(placed, state_pm, phi_pm, bcs[case.name], case.kinds, params,
                                                 mesh).total)
        t1 = time.perf_counter()
        res = sharded_perpendicular_rhs(placed, state_pm, phi_pm, bcs[case.name], case.kinds, params, mesh)
        _block(res.total)
        t2 = time.perf_counter()
        del first
        entry = _compare(singles[case.name]["arrays"], _collect(res), perm, setup.mask)
        del res
        entry.update(lowering_seconds=lowering, first_seconds=t1 - t0, seconds=t2 - t1,
                     single_first_seconds=singles[case.name]["first_seconds"],
                     single_seconds=singles[case.name]["seconds"])
        out[case.name] = entry
        log(f"Sz{sz} RHS {case.name}: worst rel {entry['worst_rel']:.2e} bitwise {entry['bitwise']}, lowering "
            f"{lowering:.1f} s, first {t1 - t0:.1f} s, second {t2 - t1:.1f} s (single {entry['single_seconds']:.1f} s)")
    del placed, bcs
    gc.collect()
    return out


def _m_norm(volume, x) -> float:
    import numpy as np

    return float(np.sqrt(np.sum(np.asarray(volume) * np.asarray(x, dtype=np.float64) ** 2)))


def single_phi(setup, log) -> dict:
    """The single-device cold-start solve of every case (two calls)."""
    from drbx.native.fci_perpendicular_phi_solver import solve_phi

    out = {}
    for case in setup.cases:
        psi, first = solve_phi(setup.solver, case.rhs, case.bc)
        _psi2, second = solve_phi(setup.solver, case.rhs, case.bc)
        out[case.name] = {"psi": psi, "iterations": int(first["iterations"]), "converged": bool(first["converged"]),
                          "relative_residual": float(first["relative_residual"]), "first_seconds": first["seconds"],
                          "seconds": second["seconds"], "psi_m": _m_norm(setup.volume, psi),
                          "rhs_source": case.rhs_source}
        log(f"single-device solve {case.name}: {first['iterations']} it, first {first['seconds']:.1f} s, second "
            f"{second['seconds']:.1f} s ({case.rhs_source})")
    return out


def sharded_phi(setup, sz: int, singles: dict, log) -> dict:
    """Lower the potential solver over ``sz`` shards, solve every case twice and compare; freed on return."""
    import numpy as np

    from drbx.native.fci_perpendicular_phi_sharding import shard_phi_solver, sharded_solve_phi
    from drbx.native.fci_perpendicular_sharding import from_plane_major, make_plane_mesh, to_plane_major

    mesh = make_plane_mesh(sz)
    t0 = time.perf_counter()
    sharded = shard_phi_solver(setup.solver, setup.raw_to_owner, setup.n, sz, mesh=mesh)
    lowering = time.perf_counter() - t0
    out = {}
    for case in setup.cases:
        rhs_pm = to_plane_major(np.asarray(case.rhs, dtype=np.float64), sharded.inverse)
        psi_pm, one = sharded_solve_phi(sharded, rhs_pm, case.bc, mesh=mesh)
        psi_pm2, two = sharded_solve_phi(sharded, rhs_pm, case.bc, mesh=mesh)
        psi = from_plane_major(np.asarray(psi_pm), sharded.perm)
        del psi_pm, psi_pm2
        single = singles[case.name]
        entry = {"iterations": int(one["iterations"]), "converged": bool(one["converged"]),
                 "relative_residual": float(one["relative_residual"]),
                 "dpsi_m": _m_norm(setup.volume, psi - single["psi"]), "psi_m": single["psi_m"],
                 "lowering_seconds": lowering, "lowering_breakdown": dict(sharded.setup_seconds),
                 "first_seconds": one["seconds"], "seconds": two["seconds"],
                 "single_first_seconds": single["first_seconds"], "single_seconds": single["seconds"]}
        entry["rel_diff"] = entry["dpsi_m"] / entry["psi_m"] if entry["psi_m"] else float("inf")
        out[case.name] = entry
        log(f"Sz{sz} solve {case.name}: {one['iterations']} it (single {single['iterations']}), rel diff "
            f"{entry['rel_diff']:.2e}, lowering {lowering:.1f} s, first {one['seconds']:.1f} s, second "
            f"{two['seconds']:.1f} s (single {single['seconds']:.1f} s)")
    del sharded
    gc.collect()
    return out


def run_checks(*, n: int, shard_counts, rhs, phi, params, phi_rtol: float, log=_log) -> dict:
    """The raw metrics (no gates) of the RHS part (``rhs``: ``plan, raw_to_owner, n, halo, mask, cases``) and the potential
    part (``phi``: ``solver, raw_to_owner, n, volume, cases``), per shard count, with the single-device references first.
    The sharded plan / solver of one shard count is freed before the next. Peak RSS is recorded after every stage."""
    import jax

    counts = [int(s) for s in shard_counts]
    if len(jax.devices()) < max(counts):
        raise RuntimeError(f"{max(counts)} shards need {max(counts)} devices, {len(jax.devices())} are available "
                           f"(XLA_FLAGS {os.environ.get('XLA_FLAGS')!r})")
    result = {"n": int(n), "shard_counts": counts, "devices": len(jax.devices()), "phi_rtol": float(phi_rtol),
              "rhs": {c.name: {} for c in rhs.cases}, "phi": {c.name: {} for c in phi.cases}, "peak_rss_gib": {}}
    t_single = time.perf_counter()
    rss = result["peak_rss_gib"]
    rhs_single = single_rhs(rhs, params, log)
    phi_single = single_phi(phi, log)
    for name, entry in rhs_single.items():
        result["rhs"][name]["single"] = {"first_seconds": entry["first_seconds"], "seconds": entry["seconds"]}
    for name, entry in phi_single.items():
        result["phi"][name]["single"] = {k: v for k, v in entry.items() if k != "psi"}
    result["single_seconds"] = time.perf_counter() - t_single
    rss["single"] = peak_rss_gib()
    for sz in counts:
        for name, entry in sharded_rhs(rhs, sz, rhs_single, params, log).items():
            result["rhs"][name][f"Sz{sz}"] = entry
        rss[f"Sz{sz}_rhs"] = peak_rss_gib()
        for name, entry in sharded_phi(phi, sz, phi_single, log).items():
            result["phi"][name][f"Sz{sz}"] = entry
        rss[f"Sz{sz}_phi"] = peak_rss_gib()
        gc.collect()
    rss["final"] = peak_rss_gib()
    return result


def _case_arrays(prep, cfg: dict, variant: str, omega_rhs):
    """``(rhs case, phi case, kinds)`` of ``variant`` on ``prep`` (a prepared grid whose adapter / exact_state know the
    variant); ``omega_rhs`` ``None``: the psi right-hand side is the discrete ``A psi_bar + B g``."""
    import numpy as np

    from drbx.native.fci_perpendicular_phi_solver import phi_boundary_term
    from drbx.native.fci_perpendicular_rhs import FIELDS, PHI
    from p08_step5_combined.jaxstage import psi_boundary_data, variant_boundary_data

    tau = float(cfg["params"]["tau"])
    rec = prep.adapter.reconstructions[variant]
    columns = np.asarray(rec.columns)
    kinds = tuple(rec.field_kinds)
    values = np.asarray(prep.adapter.owner_values)
    state = {name: np.asarray(values[:, c], dtype=np.float64) for name, c in zip(FIELDS, columns[:4])}
    phi_bar = np.asarray(values[:, columns[4]], dtype=np.float64)
    bc = variant_boundary_data(prep, variant)
    bc_psi = psi_boundary_data(bc, tau)
    if omega_rhs is None:
        psi_bar = phi_bar + tau * state["Ti"]
        rhs = np.asarray(prep.solver.op.matrix @ psi_bar, dtype=np.float64) + np.asarray(
            phi_boundary_term(prep.solver, bc_psi), dtype=np.float64)
        source = "discrete_operator"
    else:
        rhs, source = np.asarray(omega_rhs, dtype=np.float64), "reference_O"
    return (SimpleNamespace(name=variant, state=state, phi=phi_bar, bc=bc, kinds=dict(zip((*FIELDS, PHI), kinds))),
            SimpleNamespace(name=variant, rhs=rhs, bc=bc_psi, rhs_source=source), kinds)


def worker(spec: dict) -> dict:
    """The campaign worker: prepare the grid exactly as the JAX stage, build the transverse and catalogue cases, run the
    checks. Returns the raw result."""
    import jax

    from drbx.native.fci_perpendicular_rhs import PerpendicularParams, FIELDS
    from p_shared import campaign_fields as cf
    from p_shared import perpendicular_reference_rhs as prr
    from p_shared import replay_units as ru
    from p08_step4_global import campaign as step4
    from p08_step5_combined import references as ref5
    from p08_step6_global import jaxstage, references as refs6

    if not jax.config.jax_enable_x64:
        raise RuntimeError("the sharding worker needs x64")
    n, identity, cfg, inputs = int(spec["n"]), spec["identity"], spec["cfg"], spec["inputs"]
    scfg = cfg["sharding"]
    output = Path(spec["output"])
    started = time.time()
    values, column_names, _manifest = refs6.load_owner_values(output, n, identity)
    refs = ref5.load_references(output, n, identity)
    prep = jaxstage.prepare(n=n, cfg=cfg, options=cfg["operator_options"],
                            artifact_root=Path(inputs["refreeze"]["path"]) / "artifact",
                            sidecar_path=output / "localized_sidecar.json", input_root=Path(spec["input_root"]),
                            inputs=inputs, owner_values=values, column_names=column_names)
    tvar, cvar = scfg["transverse_variant"], scfg["catalogue_variant"]
    fs = cfg["variants"][tvar]["field_set"]
    rhs_t, phi_t, kinds_t = _case_arrays(prep, cfg, tvar, refs[ref5.psi_key(fs, "O")])
    if list(kinds_t) != list(cfg["variants"][tvar]["kinds"]):
        raise ValueError(f"variant {tvar!r}: kinds {kinds_t} differ from the configuration")
    paths = step4.oracle_paths(spec.get("oracle_root"), spec["input_root"])
    oracle = ru._load_oracle_owner_values(prep.env, dict(paths), ("p06n",))
    adapter = cf.P06NAdapter(prep.env.ref, prep.env.t.g.eta_period, oracle["p06n"]["owner_values"])
    cat = SimpleNamespace(**{**vars(prep), "adapter": adapter,
                             "exact_state": lambda variant: prr.P06NState(adapter, variant)})
    rhs_c, phi_c, kinds_c = _case_arrays(cat, cfg, cvar, None)
    if list(kinds_c) != list(scfg["catalogue_kinds"]):
        raise ValueError(f"catalogue variant {cvar!r}: kinds {kinds_c} differ from the pinned {scfg['catalogue_kinds']}")
    tau = float(cfg["params"]["tau"])
    params = PerpendicularParams(rho_star=float(cfg["params"]["rho_star"]), tau=tau,
                                 diffusion={f: float(cfg["params"]["diffusion"][f]) for f in FIELDS})
    raw = prep.env.t.ro
    rhs = SimpleNamespace(plan=prep.plan, raw_to_owner=raw, n=n, halo=int(scfg["plan_halo"]), mask=None,
                          cases=[rhs_t, rhs_c])
    phi = SimpleNamespace(solver=prep.solver, raw_to_owner=raw, n=n, volume=prep.solver.op.owner_volume,
                          cases=[phi_t, phi_c])
    _log(f"N{n}: prepared in {time.time() - started:.1f} s, peak RSS {peak_rss_gib():.2f} GiB; shard counts "
         f"{spec['shard_counts']}")
    result = run_checks(n=n, shard_counts=spec["shard_counts"], rhs=rhs, phi=phi, params=params,
                        phi_rtol=float(cfg["phi_solve_rtol"]))
    result.update(identity=identity, source="campaign", setup_seconds=prep.setup_seconds,
                  owners=int(len(prep.owner_volume)), variants=[tvar, cvar], wall_seconds=time.time() - started,
                  rhs_arm="prescribed_phi", plan_halo=int(scfg["plan_halo"]), phi_halo=int(scfg["phi_halo"]))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P08 step-6 sharding worker (run by sharding.run, forced host devices)")
    parser.add_argument("--worker", type=Path, required=True, help="the spec JSON written by sharding.run")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.worker.read_text())
    result = worker(spec)
    result["peak_rss_gib"]["final"] = peak_rss_gib()
    tmp = args.result.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(args.result)
    _log(f"worker done, peak RSS {peak_rss_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
