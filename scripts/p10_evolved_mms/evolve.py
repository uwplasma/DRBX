"""RK4 evolution driver of the P10 evolved MMS (chunk C4): one run per (arm, N, mode, pattern, source), written to a run
directory that the reducer (C6) reads.

    python -m p10_evolved_mms.evolve --bundle DIR --mode {diffusion,hyperbolic,coupled} --source {continuum,discrete} \\
        --pattern {NNN-D,DDDD} --dt DT [--T T] --out RUN_DIR [--chunk 25] [--rho-star ... --a-phi ... --time-scale ...]

(``scripts/`` on ``sys.path``). The run advances ``q(0) = q_exact(0)`` with :class:`~drbx.native.fci_time_integrator.Rk4Stepper` of
:func:`model.make_stage_rhs`'s stage (bundle and parameters are jit *arguments*; ``dt`` and the step index are traced; the time of
step ``k`` is ``k * dt``, computed as a product, never accumulated).

Segments ("chunks"): the step axis is cut at the multiples of ``chunk``, at the snapshot steps ``round(f * nsteps)`` and at
``nsteps``. A segment of exactly ``chunk`` steps is one ``lax.scan`` program (static length ``chunk``); every other segment runs the
1-step jitted program in a host loop, so a run compiles at most two step programs (plus the snapshot diagnostics program). After
each segment (in this order, so that the checkpoint is the commit point): snapshot (if the segment ends on a snapshot step),
``run.json``, ``checkpoint.npz`` (atomic: write to a temporary file, then ``os.replace``). The segmentation depends only on
``(chunk, nsteps, snapshot steps)``, so a resumed run executes exactly the programs of the uninterrupted run on exactly the same
inputs (bitwise reproducible).

Step numbering: steps are ``1 .. nsteps``; step ``k`` advances ``(k-1) dt -> k dt`` and ``checkpoint["step"]`` is the number of
completed steps. ``stop_step`` is the first step ``k`` for which one of its four stage states / stage RHS or its result violates the
stop rule: a non-converged potential solve, a non-finite value, or ``min n``, ``min Te``, ``min Ti <= 0``. A violating segment is
recorded (its last-step state is checkpointed), the run stops after it with ``status = "stopped"``; no retuning, no snapshot of a
violating segment. Re-running a ``"complete"`` or ``"stopped"`` run directory with ``resume=True`` returns the stored record without
computing; ``resume=False`` on an existing run directory raises (nothing is deleted).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import collections
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks                    # noqa: E402
from drbx.native.fci_time_integrator import Rk4Stepper                                              # noqa: E402
from p10_evolved_mms import bundle as bundle_mod                                                    # noqa: E402
from p10_evolved_mms import model as M                                                              # noqa: E402
from p10_evolved_mms.fields import CONFIG, FIELDS, MmsParams, default_params                        # noqa: E402

SCHEMA = "drbx.p10-run-v1"
SNAPSHOT_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)
#: trace counts of the step programs (incremented when a program is traced = compiled); for the compile-budget test
TRACES = collections.Counter()
_IDENTITY_KEYS = ("bundle_sha256", "arm", "n", "n_eta", "P", "N_wall", "arm_identity", "layout_sha256", "nodal_plan_sha256",
                  "laplacian_plan_sha256", "nodal_metric_identity", "laplacian_metric_identity", "synthetic")


# ---------------------------------------------------------------------------------------------------------------------
# the step programs
# ---------------------------------------------------------------------------------------------------------------------
def _reduce_step(aux, qn) -> dict:
    """Per-step reductions of the four :class:`~model.StageInfo` and the new state ``qn (E, P, 4)`` (scalars)."""
    def stack(name):
        return jnp.stack([getattr(a, name) for a in aux])

    return dict(max_cg_iterations=stack("cg_iterations").max(),
                max_cg_relative_residual=stack("cg_relative_residual").max(),
                all_converged=jnp.all(stack("cg_converged")),
                min_n=jnp.minimum(stack("min_n").min(), qn[..., 0].min()),
                min_Te=jnp.minimum(stack("min_Te").min(), qn[..., 1].min()),
                min_Ti=jnp.minimum(stack("min_Ti").min(), qn[..., 2].min()),
                all_finite=jnp.all(stack("finite")) & jnp.all(jnp.isfinite(qn)))


def _step(rhs_u, bundle, p, q, carry, k, dt):
    """One RK4 step from ``t = k dt`` (step number ``k + 1``): ``(q, carry)`` at ``k dt`` -> ``(q, carry)`` at ``(k+1) dt`` and the
    step's reductions."""
    t = k.astype(jnp.float64) * dt
    stepper = Rk4Stepper(lambda s, tt, c: rhs_u(bundle, p, s, tt, c))
    res = stepper(M.NodalState.from_array(q), time=t, timestep=dt, carry=carry)
    qn = res.state.array()
    return qn, res.carry, _reduce_step(res.stage_aux, qn)


@lru_cache(maxsize=32)
def _programs(rhs_u, chunk: int):
    """``(scan_program, one_program)`` for the unbound stage ``rhs_u(bundle, p, state, t, carry)`` (cached by object identity):
    ``f(bundle, p, q, carry, k0, dt) -> (q, carry, reductions)`` over ``chunk`` steps (reductions ``(chunk,)``) or one step
    (reductions scalar); ``k0`` is the index of the first step's start."""
    def scan_fn(bundle, p, q, carry, k0, dt):
        TRACES["scan"] += 1

        def body(c, i):
            qc, cc = c
            qn, cn, ys = _step(rhs_u, bundle, p, qc, cc, k0 + i, dt)
            return (qn, cn), ys

        (q, carry), ys = jax.lax.scan(body, (q, carry), jnp.arange(chunk, dtype=k0.dtype))
        return q, carry, ys

    def one_fn(bundle, p, q, carry, k0, dt):
        TRACES["one"] += 1
        return _step(rhs_u, bundle, p, q, carry, k0, dt)

    return jax.jit(scan_fn), jax.jit(one_fn)


# ---------------------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------------------
def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        writer(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    _atomic_write(path, lambda fh: fh.write(json.dumps(obj, indent=1, sort_keys=False).encode()))


def _write_npz(path: Path, **arrays) -> None:
    _atomic_write(path, lambda fh: np.savez(fh, **arrays))


def _digest(state: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def write_geometry(bundle, path: Path) -> None:
    """``geometry.npz``: ``H (E, P)``, ``region_names``, ``region_masks (R, E, P)``, ``n``, ``n_eta``."""
    masks = ring_region_masks(bundle.layout, bundle.E)
    names = list(masks)
    _write_npz(path, H=np.asarray(h_weights(bundle.ctx.plan), dtype=np.float64), region_names=np.array(names),
               region_masks=np.stack([masks[k] for k in names]).astype(bool), n=np.asarray(bundle.n), n_eta=np.asarray(bundle.E))


def _write_checkpoint(path: Path, q, carry, step: int, dt: float) -> None:
    state = np.asarray(q, dtype=np.float64)
    _write_npz(path, state=state, carry=np.asarray(carry, dtype=np.float64), t=np.asarray(step * dt), step=np.asarray(step),
               digest=np.asarray(_digest(state)))


def _read_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as z:
        state, carry, step, digest = np.asarray(z["state"]), np.asarray(z["carry"]), int(z["step"]), str(z["digest"])
    if _digest(state) != digest:
        raise ValueError(f"{path}: the stored state does not match its digest (corrupt checkpoint)")
    return state, carry, step


def _jsonable_params(p: MmsParams) -> dict:
    out = {}
    for name, value in p._asdict().items():
        a = np.asarray(value, dtype=np.float64)
        out[name] = a.tolist() if a.ndim else float(a)
    return out


def _identity(bundle) -> dict:
    ident = bundle_mod._jsonable(bundle.identity)
    return {k: ident[k] for k in _IDENTITY_KEYS if k in ident}


# ---------------------------------------------------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------------------------------------------------
def _write_snapshot(out: Path, index: int, step: int, dt: float, q, *, bundle, p, p_j, cfg, diagnostics: bool) -> float:
    t0 = time.perf_counter()
    t = step * dt
    arrays = dict(t=np.asarray(t), step=np.asarray(step), q=np.asarray(q, dtype=np.float64))
    if diagnostics:
        diag = M.snapshot_diagnostics_jit(cfg)(bundle, p_j, jnp.asarray(t, dtype=jnp.float64), jnp.asarray(q))
        arrays.update({k: np.asarray(v) for k, v in diag.items()})
    else:
        arrays["q_exact"] = np.asarray(M.exact_state(bundle, p, t, cfg.fields))
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    _write_npz(out / "snapshots" / f"snap_{index:02d}.npz", **arrays)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------------
def _stop_reason(ys: dict, idx: int) -> str:
    reasons = []
    if not bool(ys["all_converged"][idx]):
        reasons.append("potential solve (CG) did not converge")
    if not bool(ys["all_finite"][idx]):
        reasons.append("non-finite value")
    for name in ("n", "Te", "Ti"):
        if not float(ys[f"min_{name}"][idx]) > 0.0:
            reasons.append(f"min {name} <= 0")
    return "; ".join(reasons)


def run(bundle, mode: str, source: str, p: MmsParams, pattern: str, *, dt: float, T: float, out_dir, chunk: int = 25,
        resume: bool = True, snapshot_fracs=SNAPSHOT_FRACS, rhs_fn=None, fields=FIELDS, opts_override: dict | None = None,
        diagnostics: bool = True, on_chunk=None) -> dict:
    """Run (or resume) one evolution and return its ``run.json`` record (the contract of ``p10_c3_c6_contract.md``).

    ``dt`` must satisfy ``|round(T / dt) dt - T| <= 1e-12 T`` (chosen as ``T / integer`` by the preflight rule). ``rhs_fn``
    (default: the stage of ``mode`` / ``source``) is a ``rhs_fn(state, t, carry) -> (NodalState, carry, StageInfo)`` closure (a
    test hook; recorded as ``"rhs": "custom"``). ``diagnostics=False`` writes snapshots with ``q, q_exact`` only (no ``tau`` /
    potential solves; for toy right-hand sides). ``on_chunk(record)`` is called after each segment's checkpoint. ``opts_override``
    and ``fields`` are forwarded to :func:`model.stage_config` (test hooks)."""
    out = Path(out_dir)
    cfg = M.stage_config(mode, source, pattern, opts_override=opts_override, fields=fields)
    dt, T, chunk = float(dt), float(T), int(chunk)
    nsteps = int(round(T / dt))
    if nsteps < 1 or abs(nsteps * dt - T) > 1e-12 * T:
        raise ValueError(f"dt = {dt!r} does not divide T = {T!r}: round(T/dt) = {nsteps}, |nsteps dt - T| = {abs(nsteps * dt - T):.3e}")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    snap_steps = [int(round(f * nsteps)) for f in snapshot_fracs]
    if any(b <= a for a, b in zip([0] + snap_steps, snap_steps)) or (snap_steps and snap_steps[-1] > nsteps):
        raise ValueError(f"the snapshot steps {snap_steps} must be strictly increasing, in 1..{nsteps} (nsteps too small?)")
    bounds = sorted(set(range(chunk, nsteps, chunk)) | set(snap_steps) | {nsteps})

    header = dict(schema=SCHEMA, identity=_identity(bundle), mode=mode, source=source, pattern=pattern,
                  params=_jsonable_params(p), dt=dt, T=T, nsteps=nsteps, chunk=chunk, snapshot_steps=snap_steps,
                  rhs="model" if rhs_fn is None else "custom")
    run_json, ckpt_path = out / "run.json", out / "checkpoint.npz"
    p_j = M.as_jax_params(p)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)

    if rhs_fn is None:
        rhs_u = M.stage_rhs_unbound(cfg)
    else:
        def rhs_u(_bundle, _p, state, t, carry):
            return rhs_fn(state, t, carry)
    scan_prog, one_prog = _programs(rhs_u, chunk)

    record = None
    if run_json.exists():
        if not resume:
            raise FileExistsError(f"{run_json} exists; pass resume=True to continue it (nothing is deleted)")
        if ckpt_path.exists():
            record = json.loads(run_json.read_text())
            bad = {k: (record.get(k), v) for k, v in header.items() if record.get(k) != v}
            if bad:
                raise ValueError(f"cannot resume {out}: the stored run differs in (stored, requested) {bad}")
    step = 0
    if record is not None:
        state, carry, step = _read_checkpoint(ckpt_path)
        q, carry = jnp.asarray(state), jnp.asarray(carry)
        # the checkpoint is the commit point: drop the records of segments it does not cover (recomputed deterministically)
        chunks = [c for c in record["chunks"] if c["step_end"] <= step]
        done = ((record["status"] == "complete" and step == nsteps) or
                (record["status"] == "stopped" and chunks and chunks[-1]["step_end"] == step and record["stop_step"] is not None
                 and record["stop_step"] <= step))
        if done:
            return record
        record["chunks"], record["status"], record["stop_reason"], record["stop_step"] = chunks, "running", None, None
        record.setdefault("resumes", []).append({"step": step, "git_commit": bundle_mod.git_commit()})
    else:
        out.mkdir(parents=True, exist_ok=True)
        record = dict(header, git_commit=bundle_mod.git_commit(), status="running", stop_reason=None, stop_step=None, chunks=[],
                      timings={"total_wall_s": 0.0, "snapshot_wall_s": 0.0})
        write_geometry(bundle, out / "geometry.npz")
        q = M.exact_state(bundle, p, 0.0, fields)
        carry = M.initial_carry(bundle, mode, p, 0.0, fields)
        snap_s = _write_snapshot(out, 0, 0, dt, q, bundle=bundle, p=p, p_j=p_j, cfg=cfg, diagnostics=diagnostics)
        record["timings"]["snapshot_wall_s"] += snap_s
        _write_json(run_json, record)
        _write_checkpoint(ckpt_path, q, carry, 0, dt)
    q = jnp.asarray(q, dtype=jnp.float64)
    carry = jnp.asarray(carry, dtype=jnp.float64)

    while step < nsteps:
        end = next(b for b in bounds if b > step)
        t_seg = time.perf_counter()
        pieces, k = [], step
        if end - step == chunk:
            q, carry, ys = scan_prog(bundle, p_j, q, carry, jnp.asarray(k, dtype=jnp.int64), dt_j)
            pieces.append(ys)
        else:
            for k in range(step, end):
                q, carry, ys = one_prog(bundle, p_j, q, carry, jnp.asarray(k, dtype=jnp.int64), dt_j)
                pieces.append(ys)
        jax.block_until_ready((q, carry))
        ys = {name: np.concatenate([np.atleast_1d(np.asarray(piece[name])) for piece in pieces]) for name in pieces[0]}
        wall = time.perf_counter() - t_seg
        fail = ~ys["all_converged"] | ~ys["all_finite"] | ~(ys["min_n"] > 0.0) | ~(ys["min_Te"] > 0.0) | ~(ys["min_Ti"] > 0.0)
        step = end
        chunk_record = dict(step_end=end, t_end=end * dt, max_cg_iterations=int(ys["max_cg_iterations"].max()),
                            max_cg_relative_residual=float(ys["max_cg_relative_residual"].max()),
                            all_converged=bool(ys["all_converged"].all()), min_n=float(ys["min_n"].min()),
                            min_Te=float(ys["min_Te"].min()), min_Ti=float(ys["min_Ti"].min()),
                            all_finite=bool(ys["all_finite"].all()), wall_s=wall)
        record["chunks"].append(chunk_record)
        record["timings"]["total_wall_s"] += wall
        if fail.any():
            idx = int(np.argmax(fail))
            record["status"] = "stopped"
            record["stop_step"] = int(end - len(fail) + idx + 1)               # the segment holds steps end-len+1 .. end
            record["stop_reason"] = _stop_reason(ys, idx)
        else:
            if end in snap_steps:
                record["timings"]["snapshot_wall_s"] += _write_snapshot(out, snap_steps.index(end) + 1, end, dt, q, bundle=bundle,
                                                                        p=p, p_j=p_j, cfg=cfg, diagnostics=diagnostics)
            if end == nsteps:
                record["status"] = "complete"
        _write_json(run_json, record)
        _write_checkpoint(ckpt_path, q, carry, end, dt)
        if on_chunk is not None:
            on_chunk(chunk_record)
        if record["status"] == "stopped":
            break
    return record


# ---------------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, required=True, help="a bundle directory (bundle.save_bundle)")
    ap.add_argument("--mode", choices=tuple(M.MODES), required=True)
    ap.add_argument("--source", choices=M.SOURCES, required=True)
    ap.add_argument("--pattern", choices=tuple(CONFIG["patterns"]), required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=float(CONFIG["T"]))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--no-resume", action="store_true", help="fail if the run directory already holds a run")
    for key in ("rho_star", "tau", "a_phi", "a_omega", "time_scale", "w1"):
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=None, help="override of configuration.json")
    args = ap.parse_args(argv)
    override = {k: getattr(args, k) for k in ("rho_star", "tau", "a_phi", "a_omega", "time_scale", "w1") if getattr(args, k) is not None}
    p = default_params(**override)
    bundle = bundle_mod.load_bundle(args.bundle)
    record = run(bundle, args.mode, args.source, p, args.pattern, dt=args.dt, T=args.T, out_dir=args.out, chunk=args.chunk,
                 resume=not args.no_resume)
    print(json.dumps({k: record[k] for k in ("status", "stop_reason", "stop_step", "nsteps", "dt", "T")}, indent=1))
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
