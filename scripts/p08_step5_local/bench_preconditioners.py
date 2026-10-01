"""End-to-end benchmark of preconditioner candidates for the P07 Dirichlet phi solve (P08 step 5).

For every grid the Dirichlet export is loaded, the frozen campaign arrays (``D``, ``O_q3``, ``phi_bar``) and the cached
boundary data of ``solver_study`` are attached, the JAX system is built once, and every candidate (see
:data:`REGISTRY`) is timed with the protocol of ``PRECONDITIONERS.md``: host *setup*, JIT *warm-up* (first solve, rhs
``D`` field 0), then ``--repeats`` timed solves of rhs ``D`` field 0 (known solution ``phi_bar``) and rhs ``O_q3`` field 1.
Results: ``OUT/bench_N{n}.json`` and ``OUT/bench_N{n}.md``.

CLI (from ``DRBX/scripts``)::

    python -m p08_step5_local.bench_preconditioners --export EXPORT --study-out STUDY_OUT --grids 32 48 \\
        --candidates jacobi cheb4 cheb8 plane sa_jac sa_cheb ilu --rtol 1e-10 --restart 50 --max-restarts 40 \\
        --repeats 2 --out OUT
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import gc
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                              # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from drbx.native.fci_perpendicular_p07_solve import (                                # noqa: E402
    P07SolveConfig, p07_linear_system, solve_p07_dirichlet_jit)
from drbx.native.fci_perpendicular_p07_sparse import boundary_source, load_p07_sparse  # noqa: E402
from p08_step5_local import preconditioners as pc                                    # noqa: E402
from p08_step5_local.solver_study import (                                           # noqa: E402
    DEFAULT_FROZEN, _f, _json_safe, _table, load_or_build_boundary_data)

__all__ = ["REGISTRY", "owner_plane_index", "owner_geometry_index", "build_candidate", "load_grid", "bench_candidate", "bench_grid", "summarize_grid", "main"]

BUILDERS: dict[str, Callable] = {
    "jacobi": pc.jacobi, "chebyshev": pc.chebyshev, "plane_block": pc.plane_block,
    "smoothed_aggregation": pc.smoothed_aggregation, "ilu": pc.ilu, "plane_ring_block": pc.plane_ring_block,
    "plane_bgs_forward": pc.plane_bgs_forward, "plane_bgs_multicolor": pc.plane_bgs_multicolor}
# builders that also take owner_ring / owner_theta (see ``preconditioners.ring_layout``)
RING_BUILDERS = frozenset({"plane_ring_block", "plane_bgs_forward", "plane_bgs_multicolor"})

# candidate name -> (builder, params).  Add entries here to try more variants.
REGISTRY: dict[str, tuple[str, dict]] = {
    "jacobi": ("jacobi", {}),
    "cheb4": ("chebyshev", {"degree": 4}),
    "cheb8": ("chebyshev", {"degree": 8}),
    "plane": ("plane_block", {}),
    "sa_jac": ("smoothed_aggregation", {"smoother": "jacobi"}),
    "sa_cheb": ("smoothed_aggregation", {"smoother": "chebyshev"}),
    "sa_jac_vs": ("smoothed_aggregation", {"smoother": "jacobi", "volume_scaling": True}),
    "sa_cheb_vs": ("smoothed_aggregation", {"smoother": "chebyshev", "volume_scaling": True}),
    "sa_jac_f": ("smoothed_aggregation", {"smoother": "jacobi", "filter_smoothing": True}),
    "sa_cheb_f": ("smoothed_aggregation", {"smoother": "chebyshev", "filter_smoothing": True}),
    "ilu": ("ilu", {}),
    "plane_jax": ("plane_ring_block", {}),
    "plane_jax32": ("plane_ring_block", {"factor_dtype": "float32"}),
    "bgs_fwd": ("plane_bgs_forward", {}),
    "bgs_mc4": ("plane_bgs_multicolor", {"colors": 4, "symmetric": False}),
    "bgs_mc4_sym": ("plane_bgs_multicolor", {"colors": 4, "symmetric": True}),
    "bgs_mc4_host": ("plane_bgs_multicolor", {"colors": 4, "symmetric": False, "plane_solver": "host"}),
}
DEFAULT_CANDIDATES = ("jacobi", "cheb4", "cheb8", "plane", "sa_jac", "sa_cheb", "ilu")


def _rss_mb() -> float:
    """Peak resident set size of this process in MB (high-water mark)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1024.0 ** 2 if platform.system() == "Darwin" else 1024.0)


def owner_plane_index(raw_to_owner: np.ndarray, n: int, n_owners: int) -> np.ndarray:
    """eta-plane index of every owner (all raw cells of an owner share ``k = raw % n``)."""
    raw = np.asarray(raw_to_owner).reshape(-1)
    if raw.shape != (n ** 3,):
        raise ValueError(f"raw_to_owner must have shape ({n ** 3},), got {raw.shape}")
    kk = np.arange(n ** 3) % n
    valid = raw >= 0
    k_of = np.full(n_owners, -1, dtype=np.int64)
    k_of[raw[valid]] = kk[valid]
    if np.any(k_of < 0):
        raise ValueError("some owners have no raw cell in raw_to_owner")
    if not np.array_equal(k_of[raw[valid]], kk[valid]):
        raise ValueError("raw cells of one owner do not share the same eta plane")
    return k_of


def owner_geometry_index(raw_to_owner: np.ndarray, n: int, n_owners: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ring, plane, theta)`` of every owner from ``raw_to_owner`` (raw cell ``raw = (i n + j) n + k``).

    ``ring = i`` and ``plane = k`` (shared by all raw cells of an owner, checked); ``theta`` is the smallest raw theta
    index ``j`` of the owner (the angular order of the owners of one ring and plane; ``preconditioners.ring_layout``
    turns it into a position).
    """
    plane = owner_plane_index(raw_to_owner, n, n_owners)
    raw = np.asarray(raw_to_owner).reshape(-1)
    cell = np.arange(n ** 3)
    valid = raw >= 0
    ii, jj = cell // n ** 2, (cell // n) % n
    ring = np.full(n_owners, -1, dtype=np.int64)
    ring[raw[valid]] = ii[valid]
    if not np.array_equal(ring[raw[valid]], ii[valid]):
        raise ValueError("raw cells of one owner do not share the same ring")
    theta = np.full(n_owners, n, dtype=np.int64)
    np.minimum.at(theta, raw[valid], jj[valid])
    return ring, plane, theta


def _no_adapter():
    raise FileNotFoundError("cached boundary data missing or stale: run solver_study first (same --study-out)")


def load_grid(n: int, export: Path, study_out: Path, frozen: Path) -> dict:
    gdir = Path(export) / f"N{n}"
    op = load_p07_sparse(gdir / "p07_dirichlet.npz")
    if op.kind != "dirichlet":
        raise ValueError(f"N{n}: expected a dirichlet export, got {op.kind!r}")
    with np.load(gdir / "owner_map.npz") as z:
        raw_to_owner = np.asarray(z["raw_to_owner"])
    ring, plane, theta = owner_geometry_index(raw_to_owner, n, op.n_owners)
    with np.load(Path(frozen) / f"N{n}.global.npz") as z:
        d_act, o_q3 = np.asarray(z["D"], dtype=np.float64), np.asarray(z["O_q3"], dtype=np.float64)
    with np.load(Path(frozen) / f"N{n}.owner_values.npz") as z:
        phi_bar = np.asarray(z["values"], dtype=np.float64)
    for what, arr in (("D", d_act), ("O_q3", o_q3), ("phi_bar", phi_bar)):
        if arr.shape[0] != op.n_owners or arr.ndim != 2:
            raise ValueError(f"N{n}: {what} has shape {arr.shape} for {op.n_owners} owners")
    if phi_bar.shape[1] < 2 or o_q3.shape[1] < 2:
        raise ValueError(f"N{n}: need at least two fields (D field 0, O_q3 field 1)")
    bc = load_or_build_boundary_data(Path(study_out) / f"N{n}" / "boundary_data.npz", op.dirichlet_points,
                                     op.neumann_points, _no_adapter, phi_bar.shape[1])
    bsrc = np.asarray(boundary_source(op, bc), dtype=np.float64)
    return {"n": n, "op": op, "plane": plane, "ring": ring, "theta": theta, "D": d_act, "O_q3": o_q3, "phi_bar": phi_bar, "bsrc": bsrc}


def _wnorm(x: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * x * x)))


def _one_solve(run: Callable, op, vol: np.ndarray, rhs: np.ndarray, bsrc: np.ndarray,
               phi_ref: np.ndarray | None, x0: np.ndarray | None = None) -> dict:
    """One timed solve with ``run = preconditioners.make_solver(system, pre, config)`` (preconditioner data is a jit argument)."""
    t0 = time.perf_counter()
    x, info = run(rhs, bsrc, x0)
    jax.block_until_ready((x, info))
    dt = time.perf_counter() - t0
    phi = np.asarray(x)
    eff = rhs - bsrc
    rec = {"wall_s": dt, "iterations": int(info["iterations"]), "converged": bool(info["converged"]),
           "rel_residual": _wnorm(eff - op.matrix @ phi, vol) / _wnorm(eff, vol),
           "rel_residual_solver": float(info["relative_residual"])}
    if phi_ref is not None:
        rec["rel_error"] = _wnorm(phi - phi_ref, vol) / _wnorm(phi_ref, vol)
    return rec


def build_candidate_via_builders(grid: dict, builder: str, params: dict) -> pc.Preconditioner:
    extra = {"owner_ring": grid["ring"], "owner_theta": grid["theta"]} if builder in RING_BUILDERS else {}
    return BUILDERS[builder](grid["op"], owner_plane=grid["plane"], **extra, **params)


def build_candidate(name: str, grid: dict) -> pc.Preconditioner:
    """Build registry candidate ``name`` for a loaded ``grid`` (adds ring/theta for the ring/plane builders)."""
    builder, params = REGISTRY[name]
    return build_candidate_via_builders(grid, builder, params)


def bench_candidate(name: str, grid: dict, system, config: P07SolveConfig, repeats: int,
                    log: Callable[[str], None] | None = None) -> dict:
    """Setup, warm-up and timed solves of one candidate (never raises: failures go to ``error``)."""
    say = log or (lambda m: None)
    builder, params = REGISTRY[name]
    op, vol = grid["op"], np.asarray(grid["op"].owner_volume, dtype=np.float64)
    rec: dict = {"name": name, "builder": builder, "params": params, "error": None}
    rss0 = _rss_mb()
    pre = None
    try:
        pre = build_candidate_via_builders(grid, builder, params)
        rec.update(setup_s=pre.setup_seconds, nbytes=pre.nbytes, jax_native=pre.jax_native, info=pre.info)
        say(f"    setup {pre.setup_seconds:.2f} s, {pre.nbytes / 1e6:.1f} MB")
        d_rhs, d_b, d_ref = grid["D"][:, 0], grid["bsrc"][:, 0], grid["phi_bar"][:, 0]
        o_rhs, o_b = grid["O_q3"][:, 1], grid["bsrc"][:, 1]
        run = pc.make_solver(system, pre, config)
        rec["data_as_argument"] = bool(run.uses_data_argument)
        warm = _one_solve(run, op, vol, d_rhs, d_b, d_ref)
        rec["warmup_s"] = warm["wall_s"]
        rec["warmup"] = warm
        say(f"    warm-up {warm['wall_s']:.2f} s ({warm['iterations']} it, converged={warm['converged']})")
        d_runs, o_runs = [], []
        for _ in range(int(repeats)):
            d_runs.append(_one_solve(run, op, vol, d_rhs, d_b, d_ref))
            o_runs.append(_one_solve(run, op, vol, o_rhs, o_b, None))
        rec["runs_D"], rec["runs_O"] = d_runs, o_runs
        times = [r["wall_s"] for r in d_runs + o_runs]
        rec["solve_s"] = float(np.mean(times))
        rec["solve_s_D"] = float(np.mean([r["wall_s"] for r in d_runs]))
        rec["solve_s_O"] = float(np.mean([r["wall_s"] for r in o_runs]))
        rec["iterations_D"], rec["iterations_O"] = d_runs[-1]["iterations"], o_runs[-1]["iterations"]
        rec["converged"] = bool(all(r["converged"] for r in d_runs + o_runs))
        rec["rel_error"] = d_runs[-1]["rel_error"]
        rec["rel_residual"] = float(max(r["rel_residual"] for r in d_runs + o_runs))
        rec["time_1"] = rec["setup_s"] + rec["warmup_s"] + rec["solve_s"]
        rec["time_100"] = rec["setup_s"] + rec["warmup_s"] + 100.0 * rec["solve_s"]
        say(f"    solve {rec['solve_s']:.3f} s ({rec['iterations_D']}/{rec['iterations_O']} it), "
            f"rel err {rec['rel_error']:.2e}, rel res {rec['rel_residual']:.2e}")
    except Exception as exc:                                                    # noqa: BLE001
        traceback.print_exc()
        rec["error"] = f"{type(exc).__name__}: {exc}"
        say(f"    FAILED: {rec['error']}")
    finally:
        del pre
        gc.collect()
        jax.clear_caches()
    rec["peak_rss_mb"] = _rss_mb()
    rec["rss_growth_mb"] = rec["peak_rss_mb"] - rss0
    return rec


def _add_speedups(cands: dict) -> None:
    ref = cands.get("jacobi")
    ok = ref is not None and ref.get("error") is None and "time_1" in ref
    for rec in cands.values():
        if rec.get("error") is None and "time_1" in rec and ok:
            rec["speedup_1"] = ref["time_1"] / rec["time_1"]
            rec["speedup_100"] = ref["time_100"] / rec["time_100"]
        else:
            rec["speedup_1"] = rec["speedup_100"] = None


def summarize_grid(res: dict) -> str:
    cands = res["candidates"]
    st = res["settings"]
    lines = [f"# P07 Dirichlet preconditioner benchmark, N{res['grid']}", "",
             f"n_owners = {res['n_owners']}, nnz = {res['nnz']}; FGMRES rtol {st['rtol']:g}, restart {st['restart']}, "
             f"max_restarts {st['max_restarts']}, repeats {st['repeats']}.", "",
             "setup = host build; warm-up = first solve (JIT compile + one solve); solve = mean of the timed solves "
             "(rhs D field 0 and rhs O_q3 field 1). time_1 = setup + warm-up + solve, time_100 = setup + warm-up + "
             "100 solve (speedups vs jacobi). Iterations are D / O_q3; true rel error is ||phi - phi_bar||_M / "
             "||phi_bar||_M (rhs D); rel residual is the largest independently recomputed one.", ""]
    rows = []
    for name, r in cands.items():
        if r.get("error") is not None:
            rows.append([name, *["-"] * 11, "-", f"FAILED: {r['error']}"])
            continue
        rows.append([name, _f(r["setup_s"], ".2f"), _f(r["warmup_s"], ".2f"), _f(r["solve_s"], ".3f"),
                     f"{r['iterations_D']} / {r['iterations_O']}" + ("" if r["converged"] else " (NOT CONV.)"),
                     _f(r["rel_error"]), _f(r["rel_residual"]), _f(r["time_1"], ".1f"), _f(r["time_100"], ".1f"),
                     _f(r["speedup_1"], ".2f"), _f(r["speedup_100"], ".2f"), _f(r["nbytes"] / 1e6, ".1f"),
                     str(r["jax_native"])])
    lines += _table(["candidate", "setup s", "warm-up s", "solve s", "iterations", "true rel error", "rel residual",
                     "time_1 s", "time_100 s", "speedup_1", "speedup_100", "nbytes MB", "jax_native"],
                    [row[:13] for row in rows])
    for row in rows:
        if len(row) > 13:
            lines.append(f"- {row[0]}: {row[13]}")
    rb = [(n, r["info"]) for n, r in cands.items() if r.get("error") is None and "S" in r.get("info", {})]
    if rb:
        lines += ["", "## Ring/plane block solvers", ""]
        for name, info in rb:
            bd = ", ".join(f"{k} {v:.1f}" for k, v in info["setup_breakdown_s"].items())
            cond = f", max block cond1 {info['max_block_cond1']:.2e}" if "max_block_cond1" in info else ""
            lines.append(f"- {name}: S = {info['S']} super-rings, B = {info['B']}, w = {info['w']}, padded fraction "
                         f"{info['padded_fraction']:.3f}, {info['n_planes']} planes{cond}; setup breakdown (s) {bd}")
    sa = [(n, r["info"]) for n, r in cands.items() if r.get("error") is None and "operator_complexity" in r.get("info", {})]
    if sa:
        lines += ["", "## Multigrid hierarchies", ""]
        for name, info in sa:
            lv = "; ".join(f"{x['n']} ({x['nnz_per_row']:.0f}/row)" for x in info["levels"])
            lines.append(f"- {name}: levels {lv}; operator complexity {info['operator_complexity']:.2f}, grid complexity "
                         f"{info['grid_complexity']:.3f}; setup breakdown (s) "
                         + ", ".join(f"{k} {v:.1f}" for k, v in info["setup_breakdown_s"].items()))
    return "\n".join(lines) + "\n"


def bench_grid(n: int, export: Path, study_out: Path, frozen: Path, out: Path, candidates: Sequence[str],
               config: P07SolveConfig, repeats: int, log: Callable[[str], None] | None = None) -> dict:
    say = log or (lambda m: None)
    t0 = time.perf_counter()
    grid = load_grid(n, export, study_out, frozen)
    op = grid["op"]
    system = p07_linear_system(op)
    say(f"N{n}: loaded n_owners={op.n_owners}, nnz={op.matrix.nnz}, {len(np.unique(grid['plane']))} planes "
        f"({time.perf_counter() - t0:.1f} s)")
    res: dict = {"grid": n, "n_owners": int(op.n_owners), "nnz": int(op.matrix.nnz),
                 "settings": {"rtol": config.rtol, "restart": config.restart, "max_restarts": config.max_restarts,
                              "repeats": repeats, "candidates": list(candidates), "jax": jax.__version__},
                 "candidates": {}}
    # jacobi first: it is the speedup reference
    for name in sorted(set(candidates), key=lambda c: (c != "jacobi", list(candidates).index(c))):
        say(f"  [{name}] {REGISTRY[name]}")
        res["candidates"][name] = bench_candidate(name, grid, system, config, repeats, say)
    _add_speedups(res["candidates"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(res)
    (out / f"bench_N{n}.json").write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / f"bench_N{n}.md").write_text(summarize_grid(safe))
    say(f"N{n}: wrote {out / f'bench_N{n}.json'} and .md")
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, type=Path, help="EXPORT directory holding N{n}/ subfolders")
    ap.add_argument("--study-out", required=True, type=Path, help="solver_study output directory (cached boundary data)")
    ap.add_argument("--grids", nargs="+", type=int, default=[32])
    ap.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES), choices=sorted(REGISTRY))
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN, help="frozen P07N campaign directory")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--restart", type=int, default=50)
    ap.add_argument("--max-restarts", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:8.1f}s] {msg}", flush=True)

    config = P07SolveConfig(rtol=args.rtol, restart=args.restart, max_restarts=args.max_restarts)
    for n in args.grids:
        log(f"=== grid N{n} ===")
        bench_grid(n, args.export, args.study_out, args.frozen, args.out, args.candidates, config, args.repeats, log)
    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
