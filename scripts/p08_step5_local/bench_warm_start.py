"""Warm-start proxy for time stepping: preconditioned FGMRES on a smooth sequence of exact discrete solutions (P08 step 5).

For a grid the benchmark data is loaded as in ``bench_preconditioners`` (Dirichlet export, frozen ``phi_bar`` columns,
cached boundary data).  The trajectory is ``phi_s = phi_bar[:, 0] cos(w s) + phi_bar[:, 1] sin(w s)``, ``s = 0..S-1``, with the
boundary term ``B g_s = bsrc[:, 0] cos(w s) + bsrc[:, 1] sin(w s)`` (the same combination of the cached boundary-source
columns) and ``rhs_s = A phi_s + B g_s``, so ``phi_s`` is the exact solution of step ``s``.  ``w`` is chosen so that the mean
M-weighted relative change per step ``||phi_{s+1} - phi_s|| / ||phi_s||`` equals ``delta`` (``--deltas``).

For every preconditioner, ``rtol`` and ``delta`` the sequence is solved *cold* (``x0 = 0``) and *warm* (``x0`` = the previous
step's solution; step 0 is cold in both).  The solver's convergence test is ``||rhs_eff - A x||_M <= rtol ||rhs_eff||_M``,
i.e. relative to the norm of the right-hand side, *not* to the initial residual; a warm start therefore reduces the
iteration count only through the smaller initial residual (about ``delta`` for the previous solution), and a looser
``rtol`` or a smaller ``delta`` saves more.  Averages in the table are over steps ``1..S-1`` (same steps for cold and warm).
Per-step iterations, wall times and the true M-weighted relative error ``||x - phi_s|| / ||phi_s||`` are in the json.

The jitted solve of ``preconditioners.make_solver`` takes the preconditioner data as an argument; one untimed solve per
(preconditioner, rtol) compiles the program before the timed sequences.

CLI (from ``DRBX/scripts``)::

    python -m p08_step5_local.bench_warm_start --export EXPORT --study-out STUDY_OUT --grids 32 \\
        --preconds jacobi plane_jax plane_jax32 --steps 20 --deltas 1e-2 1e-3 1e-4 --rtols 1e-10 1e-8 --out OUT
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np                                                                   # noqa: E402

from p08_step5_local import bench_preconditioners as bp                              # noqa: E402  (sets the CPU/x64/thread env)
from p08_step5_local import preconditioners as pc                                    # noqa: E402
from p08_step5_local.solver_study import DEFAULT_FROZEN, _f, _json_safe, _table     # noqa: E402

import jax                                                                           # noqa: E402
from drbx.native.fci_perpendicular_p07_solve import P07SolveConfig, p07_linear_system  # noqa: E402

__all__ = ["omega_for_delta", "trajectory", "solve_sequence", "warm_start_grid", "summarize_warm", "main"]


def _wnorm(x: np.ndarray, vol: np.ndarray) -> float:
    return float(np.sqrt(np.sum(vol * x * x)))


def _mean_step_change(p0: np.ndarray, p1: np.ndarray, vol: np.ndarray, omega: float, steps: int) -> float:
    s = np.arange(steps)
    phis = [p0 * np.cos(omega * k) + p1 * np.sin(omega * k) for k in s]
    return float(np.mean([_wnorm(phis[k + 1] - phis[k], vol) / _wnorm(phis[k], vol) for k in range(steps - 1)]))


def omega_for_delta(p0: np.ndarray, p1: np.ndarray, vol: np.ndarray, delta: float, steps: int) -> float:
    """``omega`` with mean over ``s`` of ``||phi_{s+1} - phi_s||_M / ||phi_s||_M`` equal to ``delta`` (fixed-point iteration)."""
    if steps < 2:
        raise ValueError("need at least two steps")
    omega = float(delta)
    for _ in range(12):
        ratio = _mean_step_change(p0, p1, vol, omega, steps)
        omega *= delta / ratio
        if abs(ratio / delta - 1.0) < 1e-12:
            break
    return omega


def trajectory(grid: dict, omega: float, steps: int):
    """Yield ``(s, phi_s, rhs_s, boundary_term_s)`` with ``rhs_s = A phi_s + B g_s`` (``phi_s`` exact for step ``s``)."""
    op = grid["op"]
    p0, p1 = grid["phi_bar"][:, 0], grid["phi_bar"][:, 1]
    b0, b1 = grid["bsrc"][:, 0], grid["bsrc"][:, 1]
    for s in range(steps):
        c, sn = np.cos(omega * s), np.sin(omega * s)
        phi = c * p0 + sn * p1
        bs = c * b0 + sn * b1
        yield s, phi, op.matrix @ phi + bs, bs


def solve_sequence(run: Callable, grid: dict, omega: float, steps: int, warm: bool) -> dict:
    """Solve the trajectory with ``run`` (``make_solver``); per-step iterations, wall time, error, residual."""
    vol = np.asarray(grid["op"].owner_volume, dtype=np.float64)
    its, secs, errs, convs, ress = [], [], [], [], []
    x_prev = None
    for s, phi, rhs, bs in trajectory(grid, omega, steps):
        t0 = time.perf_counter()
        x, info = run(rhs, bs, x_prev if warm else None)
        jax.block_until_ready((x, info))
        secs.append(time.perf_counter() - t0)
        xn = np.asarray(x)
        its.append(int(info["iterations"]))
        convs.append(bool(info["converged"]))
        ress.append(float(info["relative_residual"]))
        errs.append(_wnorm(xn - phi, vol) / _wnorm(phi, vol))
        x_prev = xn
    return {"iterations": its, "wall_s": secs, "rel_error": errs, "converged": convs, "rel_residual": ress}


def _mean(x: Sequence[float]) -> float:
    return float(np.mean(x[1:])) if len(x) > 1 else float(np.mean(x))


def warm_start_grid(n: int, export: Path, study_out: Path, frozen: Path, out: Path, preconds: Sequence[str],
                    steps: int, deltas: Sequence[float], rtols: Sequence[float], restart: int, max_restarts: int,
                    log: Callable[[str], None] | None = None) -> dict:
    say = log or (lambda m: None)
    grid = bp.load_grid(n, export, study_out, frozen)
    op = grid["op"]
    vol = np.asarray(op.owner_volume, dtype=np.float64)
    system = p07_linear_system(op)
    p0, p1 = grid["phi_bar"][:, 0], grid["phi_bar"][:, 1]
    omegas = {float(d): omega_for_delta(p0, p1, vol, float(d), steps) for d in deltas}
    say(f"N{n}: n_owners={op.n_owners}, omega per delta: " + ", ".join(f"{d:g}: {w:.4g}" for d, w in omegas.items()))
    res: dict = {"grid": n, "n_owners": int(op.n_owners),
                 "settings": {"steps": steps, "deltas": [float(d) for d in deltas], "rtols": [float(r) for r in rtols],
                              "restart": restart, "max_restarts": max_restarts, "preconds": list(preconds),
                              "omega": {str(k): v for k, v in omegas.items()}},
                 "preconditioners": {}}
    for name in preconds:
        entry: dict = {"error": None, "runs": []}
        res["preconditioners"][name] = entry
        try:
            say(f"  [{name}] {bp.REGISTRY[name]}")
            pre = bp.build_candidate(name, grid)
            entry.update(setup_s=pre.setup_seconds, nbytes=pre.nbytes, jax_native=pre.jax_native)
            for rtol in rtols:
                config = P07SolveConfig(rtol=float(rtol), restart=restart, max_restarts=max_restarts)
                run = pc.make_solver(system, pre, config)
                t0 = time.perf_counter()
                _, _, rhs0, bs0 = next(trajectory(grid, 0.0, 1))
                jax.block_until_ready(run(rhs0, bs0, None))
                entry.setdefault("warmup_s", {})[f"{rtol:g}"] = time.perf_counter() - t0
                for delta in deltas:
                    om = omegas[float(delta)]
                    cold = solve_sequence(run, grid, om, steps, warm=False)
                    warm = solve_sequence(run, grid, om, steps, warm=True)
                    rec = {"rtol": float(rtol), "delta": float(delta), "omega": om, "cold": cold, "warm": warm,
                           "cold_its": _mean(cold["iterations"]), "warm_its": _mean(warm["iterations"]),
                           "cold_s": _mean(cold["wall_s"]), "warm_s": _mean(warm["wall_s"]),
                           "max_error": max(max(cold["rel_error"]), max(warm["rel_error"])),
                           "all_converged": all(cold["converged"]) and all(warm["converged"])}
                    rec["warm_speedup"] = rec["cold_s"] / rec["warm_s"]
                    entry["runs"].append(rec)
                    say(f"    rtol {rtol:g} delta {delta:g}: its cold {rec['cold_its']:.2f} warm {rec['warm_its']:.2f}, "
                        f"s/step cold {rec['cold_s']:.3f} warm {rec['warm_s']:.3f}, max err {rec['max_error']:.2e}")
            del pre, run
        except Exception as exc:                                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            entry["error"] = f"{type(exc).__name__}: {exc}"
            say(f"    FAILED: {entry['error']}")
        gc.collect()
        jax.clear_caches()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(res)
    (out / f"warm_N{n}.json").write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / f"warm_N{n}.md").write_text(summarize_warm(safe))
    say(f"N{n}: wrote {out / f'warm_N{n}.json'} and .md")
    return safe


def summarize_warm(res: dict) -> str:
    st = res["settings"]
    lines = [f"# P07 Dirichlet warm-start proxy, N{res['grid']}", "",
             f"n_owners = {res['n_owners']}; {st['steps']} steps of a smooth trajectory of exact solutions "
             f"phi_s = phi_bar_0 cos(w s) + phi_bar_1 sin(w s); delta = mean relative change per step; FGMRES restart "
             f"{st['restart']}, max_restarts {st['max_restarts']}.", "",
             "Cold: x0 = 0; warm: x0 = previous solution (step 0 cold in both). Averages are over steps 1..S-1. The "
             "convergence test is relative to ||rhs||_M (not to the initial residual), so the warm start saves "
             "iterations only through the smaller initial residual. speedup = cold s/step / warm s/step. max error is the "
             "largest true M-weighted relative error ||x - phi_s|| / ||phi_s|| over all steps, cold and warm.", ""]
    rows = []
    for name, e in res["preconditioners"].items():
        if e.get("error") is not None:
            lines.append(f"- {name}: FAILED: {e['error']}")
            continue
        for r in e["runs"]:
            rows.append([name, f"{r['rtol']:g}", f"{r['delta']:g}", _f(r["cold_its"], ".2f"), _f(r["warm_its"], ".2f"),
                         _f(r["cold_s"], ".3f"), _f(r["warm_s"], ".3f"), _f(r["warm_speedup"], ".2f"),
                         _f(r["max_error"]) + ("" if r["all_converged"] else " (NOT CONV.)")])
    lines += _table(["prec", "rtol", "delta", "cold its/step", "warm its/step", "cold s/step", "warm s/step",
                     "warm speedup", "max error"], rows)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, type=Path, help="EXPORT directory holding N{n}/ subfolders")
    ap.add_argument("--study-out", required=True, type=Path, help="solver_study output directory (cached boundary data)")
    ap.add_argument("--grids", nargs="+", type=int, default=[32])
    ap.add_argument("--preconds", nargs="+", default=["jacobi", "plane_jax", "plane_jax32"], choices=sorted(bp.REGISTRY))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--deltas", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--rtols", nargs="+", type=float, default=[1e-10, 1e-8])
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN, help="frozen P07N campaign directory")
    ap.add_argument("--restart", type=int, default=50)
    ap.add_argument("--max-restarts", type=int, default=40)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:8.1f}s] {msg}", flush=True)

    for n in args.grids:
        log(f"=== grid N{n} ===")
        warm_start_grid(n, args.export, args.study_out, args.frozen, args.out, args.preconds, args.steps, args.deltas,
                        args.rtols, args.restart, args.max_restarts, log)
    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
