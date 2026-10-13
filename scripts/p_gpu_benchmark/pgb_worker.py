"""Per-resolution worker of the P GPU benchmark: runs every pending case of one ``(backend, n)`` and writes one JSON per case.

Called as a fresh process by ``run_benchmark.py run`` (``worker`` subcommand), after :mod:`pgb_env` configured the environment.
Order: host builds of ``core_schur`` (production streaming assembly, before anything else is retained, so the RSS numbers are
the build's), then the in-plane CSR groups are assembled once and kept for the other variants and the independent residual check,
then the precond cases, the PCG cases and the sharding case.
"""
from __future__ import annotations

import fnmatch
import functools
import gc
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import solvax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", os.environ.get("P_GPU_BENCH_MATMUL_PRECISION", "highest"))

from drbx.geometry.sbp_laplacian import build_laplacian_plan                                        # noqa: E402
from drbx.geometry.sbp_laplacian_assembly import iter_inplane_blocks, plane_group_size              # noqa: E402
from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol                                # noqa: E402
from drbx.native.fci_perpendicular_plane_preconditioner import (                                    # noqa: E402
    apply_core_schur_preconditioner, build_core_schur_preconditioner)
from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form                              # noqa: E402

import pgb_cases as K                                                                               # noqa: E402
import pgb_inputs as I                                                                              # noqa: E402
import pgb_precond as PC                                                                            # noqa: E402
import pgb_shard as SH                                                                              # noqa: E402
import pgb_timing as T                                                                              # noqa: E402

FDT = {"f32": "float32", "f64": "float64"}


def _clean(o):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to ``None``."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1, sort_keys=False))
    os.replace(tmp, path)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={T.current_rss_gib():.2f}GiB {msg}", flush=True)


class Worker:
    def __init__(self, *, inputs, out, backend: str, n: int, reps: int = 20, solve_reps: int = 5, max_planes: int | None = None,
                 force: bool = False, patterns=None, mem_budget_gb: float | None = None):
        self.inputs, self.out, self.backend, self.n = Path(inputs), Path(out), backend, int(n)
        self.reps, self.solve_reps, self.max_planes, self.force = int(reps), int(solve_reps), max_planes, force
        self.patterns, self.mem_budget_gb = patterns, mem_budget_gb
        self.campaign = K.load_campaign(self.out)
        self.cache: dict = {}
        self.outs: dict = {}
        self.env = T.environment_record()
        self.env.update(solvax=solvax.__version__, backend=backend, threads=os.environ.get("P_GPU_BENCH_CPU_THREADS"),
                        affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
        if backend == "gpu" and self.env["default_backend"] not in ("gpu", "cuda"):
            raise SystemExit(f"--backend gpu but the JAX default backend is {self.env['default_backend']!r}")
        if backend == "cpu" and self.env["default_backend"] != "cpu":
            raise SystemExit(f"--backend cpu but the JAX default backend is {self.env['default_backend']!r}")

    # ---- records ---------------------------------------------------------------------------------------------------------
    def path(self, cid):
        return K.case_path(self.out, self.backend, cid)

    def pending(self) -> list[dict]:
        specs = [s for s in K.case_table([self.n]) if s["kind"] not in ("scaling",)]
        if self.n != 64:
            specs = [s for s in specs if s["kind"] != "sharding"]
        if self.patterns:
            specs = [s for s in specs if any(fnmatch.fnmatch(s["case_id"], p) for p in self.patterns)]
        out = []
        for s in specs:
            p = self.path(s["case_id"])
            if not self.force and p.is_file():
                try:
                    if json.loads(p.read_text()).get("status") in ("ok", "skipped"):
                        continue
                except Exception:                                                  # noqa: BLE001
                    pass
            out.append(s)
        return out

    def record(self, spec, status, reason, params, results, t_start):
        rec = dict(schema=K.CASE_SCHEMA, case_id=spec["case_id"], kind=spec["kind"], backend=self.backend, n=spec["n"], arm=K.ARM,
                   method=spec["method"], dtype=spec["dtype"], status=status, reason=reason, params=params, results=results,
                   identity=self.identity, env=self.env, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   elapsed_s=time.perf_counter() - t_start)
        write_json_atomic(self.path(spec["case_id"]), rec)
        log(f"{spec['case_id']}: {status}" + (f" ({reason})" if reason else ""))
        return rec

    def guarded(self, spec, fn):
        """Run ``fn() -> (params, results)``; a ``SkipCase`` / OOM becomes a skipped record, any other error a failed one."""
        t0 = time.perf_counter()
        try:
            params, results = fn()
            return self.record(spec, "ok", None, params, results, t0)
        except SkipCase as s:
            return self.record(spec, "skipped", s.reason, {}, {}, t0)
        except BaseException as exc:                                               # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            gc.collect()
            if T.is_oom(exc):
                return self.record(spec, "skipped", f"skipped: memory (runtime: {type(exc).__name__})", {}, {}, t0)
            tb = traceback.format_exc()
            return self.record(spec, "failed", f"{type(exc).__name__}: {exc}", {"traceback": tb[-4000:]}, {}, t0)

    # ---- shared state ----------------------------------------------------------------------------------------------------
    def setup(self):
        t0 = time.perf_counter()
        path = I.metric_path(self.inputs, self.n)
        self.file_sha = I.sha256_file(path)
        exp = K.INPUT_FILES[self.n]["sha256"]
        if self.file_sha != exp:
            raise SystemExit(f"N{self.n}: input sha256 {self.file_sha} != campaign identity {exp}")
        self.metric = I.Metric(path, self.n)
        self.identity = dict(metric_file=K.INPUT_FILES[self.n]["path"], metric_sha256=self.file_sha,
                             metric_identity=self.metric.identity, n=self.n, arm=K.ARM, max_planes=self.max_planes)
        self.plan = build_laplacian_plan(self.metric.layout, self.metric.metric)
        st = self.plan.structure
        self.st = st
        self.E_all, self.P, self.Nc, self.m, self.N = st.n_eta, st.P, st.Nc, st.m, st.N
        self.E, self.group = PC.plane_group_choice(self.E_all, plane_group_size(st), self.max_planes)
        self.plan_seconds = time.perf_counter() - t0
        log(f"N{self.n}: plan built in {self.plan_seconds:.1f}s; E={self.E_all} (using {self.E}) P={self.P} Nc={self.Nc} m={self.m} "
            f"N={self.N}; group_planes={self.group}")
        self.rhs = np.random.default_rng(self.campaign["precond_rhs"]["seed"]).standard_normal(self.E * self.P)

    def budget(self):
        return T.free_budget_bytes(self.backend, self.mem_budget_gb)

    def check_memory(self, need_bytes: int, what: str, margin: float = 0.85):
        b = self.budget()
        if b is not None and need_bytes > margin * b:
            raise SkipCase(f"skipped: memory ({what}: needs ~{need_bytes / 2**30:.1f} GiB, budget {b / 2**30:.1f} GiB)")

    @property
    def blocks(self) -> PC.PlaneBlocks:
        if "blocks" not in self.cache:
            t0 = time.perf_counter()
            groups = iter_inplane_blocks(self.plan, group_planes=self.group)
            if self.max_planes:
                groups = PC.restrict_groups(groups, self.E)
            else:
                groups = list(groups)
            self.cache["blocks"] = PC.PlaneBlocks(groups, n_planes=self.E, nc=self.Nc, m=self.m, N=self.N)
            b = self.cache["blocks"]
            self.assembly = dict(assembly_s=time.perf_counter() - t0, nnz=b.nnz, csr_bytes=b.csr_bytes, group_planes=self.group,
                                 ring_reach=b.ring_reach())
            log(f"in-plane CSR groups assembled in {self.assembly['assembly_s']:.1f}s: nnz={b.nnz} "
                f"({b.csr_bytes / 2**30:.2f} GiB), ring reach {self.assembly['ring_reach']}")
        return self.cache["blocks"]

    # ---- core_schur build ------------------------------------------------------------------------------------------------
    def core(self, dt: str):
        key = ("core", dt)
        if key in self.cache:
            return self.cache[key]
        if ("core_err", dt) in self.cache:
            raise self.cache[("core_err", dt)]
        spec = next(s for s in K.case_table([self.n]) if s["case_id"] == K.case_id("build", self.n, "core_schur", dt))
        t0 = time.perf_counter()
        rss0, peak0 = T.current_rss_gib(), T.peak_rss_gib()
        dev0 = T.device_memory()
        tb = time.perf_counter()
        try:
            if self.max_planes:                                                   # restricted planes: from the retained groups
                prec = build_core_schur_preconditioner(self.blocks.groups, n_planes=self.E, n_core=self.Nc, n_rings=self.m,
                                                       ring_size=self.N, factor_dtype=FDT[dt])
            else:                                                                 # production path: streaming assembly
                prec = sol.build_dirichlet_preconditioner(self.plan, factor_dtype=FDT[dt], method="core_schur")
            T.block(prec)
        except BaseException as exc:                                              # noqa: BLE001
            self.cache[("core_err", dt)] = exc
            raise
        build_s = time.perf_counter() - tb
        self.cache[key] = prec
        self.cache["build_s_" + dt] = build_s
        meta = prec.meta
        comps = {k: int(getattr(prec, k).nbytes) for k in ("lo", "dinv", "z", "sinv")}
        info = prec.info
        res = dict(host_build_s=build_s, setup_seconds=info.get("setup_seconds"), setup_breakdown_s=info.get("setup_breakdown_s"),
                   rss_before_gib=rss0, rss_after_gib=T.current_rss_gib(), peak_rss_before_gib=peak0,
                   peak_rss_after_gib=T.peak_rss_gib(), storage_bytes=int(prec.nbytes), components_bytes=comps,
                   storage_bytes_per_plane=int(prec.nbytes) // self.E, max_block_cond1=info.get("max_block_cond1"),
                   nnz=info.get("nnz"), padded_fraction=info.get("padded_fraction"),
                   structure=dict(E=self.E, P=self.P, Nc=self.Nc, m=self.m, N=self.N, S=meta.S, B=meta.B, w=meta.w,
                                  rpb=meta.rings_per_block),
                   peak_rss_note="peak_rss_*_gib is the process high-water mark (getrusage): the f32 build runs first, after the plan "
                                 "and before any CSR is retained; the f64 build's peak includes the resident f32 factors on the CPU "
                                 "backend",
                   device_memory_before=dev0, device_memory_after=T.device_memory())
        if spec["case_id"] in {s["case_id"] for s in self.todo}:
            self.record(spec, "ok", None, dict(factor_dtype=FDT[dt], rings_per_block="auto", method="core_schur",
                                              assembly="streaming" if not self.max_planes else "retained groups",
                                              max_planes=self.max_planes), res, t0)
        log(f"core_schur {dt} built in {build_s:.1f}s, {prec.nbytes / 2**20:.1f} MiB (S={meta.S} B={meta.B} w={meta.w})")
        return prec

    # ---- references ------------------------------------------------------------------------------------------------------
    def reference(self):
        if "ref" not in self.cache:
            if self.n <= self.campaign["dense_reference_max_n"]:
                t0 = time.perf_counter()
                self.cache["ref"] = (PC.dense_reference(self.blocks, self.rhs), "dense_lu_f64_host")
                log(f"dense reference in {time.perf_counter() - t0:.1f}s")
            else:
                prec = self.core("f64")
                y = jax.jit(apply_core_schur_preconditioner)(prec, jnp.asarray(self.rhs))
                self.cache["ref"] = (np.asarray(y), "core_schur_f64_same_backend")
        return self.cache["ref"]

    def stored_out(self, dt):
        if ("core_schur", dt) not in self.outs:
            y = jax.jit(apply_core_schur_preconditioner)(self.core(dt), jnp.asarray(self.rhs))
            self.outs[("core_schur", dt)] = np.asarray(y)
        return self.outs[("core_schur", dt)]

    # ---- precond cases ---------------------------------------------------------------------------------------------------
    def make_precond(self, method: str, dt: str):
        """``(fn, args, info)`` of a variant; ``fn(*args)`` is the jittable apply on the flat vector."""
        P, Nc, m, N, E = self.P, self.Nc, self.m, self.N, self.E
        if method == "core_schur":
            prec = self.core(dt)
            return apply_core_schur_preconditioner, (prec,), dict(
                storage_bytes=int(prec.nbytes), build_s=self.cache.get("build_s_" + dt), S=prec.meta.S, B=prec.meta.B, w=prec.meta.w,
                rings_per_block=prec.meta.rings_per_block)
        if method == "core_schur_zfree":
            core = self.core(dt)
            t0 = time.perf_counter()
            zp = PC.build_zfree_preconditioner(core, self.blocks)
            T.block(zp)
            return PC.apply_zfree_preconditioner, (zp,), dict(
                storage_bytes=int(zp.nbytes), build_s=time.perf_counter() - t0, S=zp.meta.S, B=zp.meta.B, w=zp.meta.w, nr=zp.meta.nr,
                coupled_rings=zp.meta.nr / N, mrc_density=zp.info["mrc_density"], mrc_bytes=int(zp.mrc.nbytes),
                z_bytes_not_stored=int(core.z.nbytes), structure=dict(E=E, Nc=Nc, m=m, N=N))
        if method == "block_jacobi":
            est = PC.est_block_jacobi_bytes(E, P, dt)
            self.check_memory(est, "dense block_jacobi LU")
            t0 = time.perf_counter()
            mods, sizes = PC.build_block_jacobi(self.blocks, dt)
            fn = functools.partial(PC.apply_block_jacobi, sizes=sizes, P=P, dt=dt)
            return fn, (mods,), dict(storage_bytes=T.tree_nbytes(mods), build_s=time.perf_counter() - t0, estimate_bytes=est,
                                     chunk_planes=max(sizes))
        if method == "block_thomas":
            reach = self.blocks.ring_reach()
            est = PC.est_block_thomas_bytes(E, Nc, m, N, reach, dt)
            self.check_memory(est, "block_thomas + bordered core")
            t0 = time.perf_counter()
            state, lay = PC.build_block_thomas(self.blocks, dt)
            fn = functools.partial(PC.apply_block_thomas, nc=Nc, mN=m * N, n_s=lay["n_s"], Bk=lay["Bk"], dt=dt)
            return fn, (state,), dict(storage_bytes=T.tree_nbytes(state), build_s=time.perf_counter() - t0, estimate_bytes=est,
                                      S=lay["n_s"], B=lay["Bk"], w=1, rings_per_block=lay["rpb"])
        raise ValueError(method)

    def run_precond(self, spec):
        method, dt = spec["method"], spec["dtype"]
        P = self.P

        def go():
            ref, ref_kind = self.reference()
            dev0, rss0 = T.device_memory(), T.current_rss_gib()
            fn, args, info = self.make_precond(method, dt)
            r = jnp.asarray(self.rhs)
            dev1 = T.device_memory()
            stats, y = T.time_jit(fn, *args, r, reps=self.reps)
            x = np.asarray(y)
            self.outs[(method, dt)] = x
            e, e_pl = PC.rel_err(x, ref, P)
            ax = self.blocks.matvec(x)
            resid = float(np.linalg.norm(ax - self.rhs) / np.linalg.norm(self.rhs))
            res = dict(apply=stats, per_plane_ms=1e3 * stats["median_s"] / self.E, rel_err_vs_ref=e, rel_err_vs_ref_max_plane=e_pl,
                       reference=ref_kind, rel_residual=resid, residual_check="||blockdiag(M) x - r|| / ||r|| on the host CSR groups",
                       **info, storage_bytes_per_plane=info["storage_bytes"] // self.E,
                       device_bytes_in_use_before=dev0.get("bytes_in_use"), device_bytes_in_use_after_build=dev1.get("bytes_in_use"),
                       device_state_delta_bytes=(dev1["bytes_in_use"] - dev0["bytes_in_use"]) if dev1 and dev0 else None,
                       device_peak_bytes_in_use=T.device_memory().get("peak_bytes_in_use"),
                       host_rss_gib=T.current_rss_gib(), host_rss_delta_gib=T.current_rss_gib() - rss0)
            if method == "core_schur_zfree":
                stored = self.stored_out(dt)
                res["rel_diff_vs_stored_z"] = float(np.linalg.norm(x - stored) / np.linalg.norm(stored))
                res["storage_ratio_vs_stored_z"] = info["storage_bytes"] / self.core(dt).nbytes
                res["apply_time_ratio_vs_stored_z"] = stats["median_s"] / self.time_stored(dt)["median_s"]
            if method == "core_schur":
                self.cache[("time_stored", dt)] = stats
            params = dict(method=method, dtype=dt, planes=self.E, unknowns=self.E * P, reps=self.reps, rhs=self.campaign["precond_rhs"],
                          max_planes=self.max_planes, assembly=getattr(self, "assembly", None))
            del fn, args
            gc.collect()
            return params, res

        return self.guarded(spec, go)

    def time_stored(self, dt):
        if ("time_stored", dt) not in self.cache:
            stats, _ = T.time_jit(apply_core_schur_preconditioner, self.core(dt), jnp.asarray(self.rhs), reps=self.reps)
            self.cache[("time_stored", dt)] = stats
        return self.cache[("time_stored", dt)]

    # ---- PCG -------------------------------------------------------------------------------------------------------------
    def pcg_setup(self):
        if "pcg" in self.cache:
            return self.cache["pcg"]
        E, P, deta = self.E, self.P, self.st.deta
        plan_dev = jax.tree_util.tree_map(jnp.asarray, self.plan)
        pts = self.metric.points
        u, th, eta = pts[..., 0], pts[..., 1], pts[..., 2]
        eta_c = 2 * np.pi * eta / (E * self.st.deta)
        fstar = (1 + 0.3 * np.cos(eta_c)) * np.cos(2 * th) * np.sin(2.5 * u) + 0.2 * np.cos(3 * th + eta_c) * u ** 2
        fstar = fstar / np.abs(fstar).max()

        def mv(lp, v):
            return laplacian_form(lp, v, None, "dirichlet", None, 1.0)

        mv_jit = jax.jit(mv)
        fstar_d = jnp.asarray(fstar)
        rhs = mv_jit(plan_dev, fstar_d)                                           # s = L f*,  rhs = -form(0) - H s = M f*
        T.block(rhs)
        norm_h = lambda lp, r: jnp.sqrt(jnp.sum(r * r / (lp.Hp * deta)))          # noqa: E731
        ctx = dict(plan_dev=plan_dev, fstar=fstar, rhs=rhs, mv=mv, mv_jit=mv_jit, norm_h=norm_h,
                   H=np.asarray(self.plan.Hp) * deta, x0=jnp.zeros((E, P)))

        def make_solve(apply_fn):
            def solve(lp, b, prec, x0, rtol, maxit):
                matvec = lambda v: mv(lp, v)                                      # noqa: E731
                precond = lambda r: apply_fn(prec, r.reshape(-1)).reshape(E, P)   # noqa: E731
                return sol._pcg(matvec, b, precond, lambda r: norm_h(lp, r), x0, rtol, maxit)
            return solve

        ctx["make_solve"] = make_solve
        self.cache["pcg"] = ctx
        return ctx

    def pcg_case(self, spec, apply_fn, prec, label: str, package_prec=None, extra_params=None):
        c = self.pcg_setup()
        E, P = self.E, self.P
        rtol, maxit = self.campaign["solve"]["rtol"], self.campaign["solve"]["maxit"]
        rt, mi = jnp.asarray(rtol, jnp.float64), jnp.asarray(maxit, jnp.int32)
        dev0 = T.device_memory()
        solve = c["make_solve"](apply_fn)
        stats, out = T.time_jit(solve, c["plan_dev"], c["rhs"], prec, c["x0"], rt, mi, reps=self.solve_reps)
        x, info = out
        its, rel, conv = int(info["iterations"]), float(info["relative_residual"]), bool(info["converged"])
        # independent true residual (H^-1 norm) and solution error against f*
        r_true = c["rhs"] - c["mv_jit"](c["plan_dev"], x)
        true_res = float(c["norm_h"](c["plan_dev"], r_true) / c["norm_h"](c["plan_dev"], c["rhs"]))
        xn = np.asarray(x)
        H = c["H"]
        err = float(np.sqrt(np.sum(H * (xn - c["fstar"]) ** 2) / np.sum(H * c["fstar"] ** 2)))
        mv_stats, _ = T.time_jit(c["mv"], c["plan_dev"], x, reps=self.reps)
        r_flat = c["rhs"].reshape(-1)
        pc_stats, _ = T.time_jit(apply_fn, prec, r_flat, reps=self.reps)
        res = dict(iterations=its, converged=conv, relative_residual=rel, true_residual_hinv=true_res, solution_error_rel_hweighted=err,
                   solve=stats, solve_warm_median_s=stats["median_s"], solve_compile_s=stats["compile_s"],
                   per_iteration_ms=1e3 * stats["median_s"] / max(its, 1), matvec=mv_stats, precond=pc_stats,
                   matvec_ms=1e3 * mv_stats["median_s"], precond_ms=1e3 * pc_stats["median_s"],
                   matvec_applications=its + 1, precond_applications=its + 1,
                   predicted_solve_s=(its + 1) * (mv_stats["median_s"] + pc_stats["median_s"]),
                   precond_storage_bytes=T.tree_nbytes(prec), preconditioner=label,
                   device_bytes_in_use_before=dev0.get("bytes_in_use"), device_peak_bytes_in_use=T.device_memory().get("peak_bytes_in_use"),
                   host_rss_gib=T.current_rss_gib())
        if package_prec is not None:                                              # cross-check against the package's own solve
            s_rhs = -c["rhs"] / (c["plan_dev"].Hp * self.st.deta)                 # s with rhs = -H s  =>  s = -rhs / H
            _x, pinfo = sol.solve_dirichlet_jit(c["plan_dev"], s_rhs, None, package_prec, rtol=rtol, maxit=maxit)
            res["package_solve_dirichlet_iterations"] = int(pinfo["iterations"])
            res["package_solve_dirichlet_converged"] = bool(pinfo["converged"])
        params = dict(rtol=rtol, maxit=maxit, residual_norm="h_inv", x0="zero", rhs=self.campaign["solve"]["rhs"], planes=E,
                      unknowns=E * P, solve_reps=self.solve_reps, reps=self.reps, **(extra_params or {}))
        return params, res

    def run_pcg_core(self, spec):
        dt = spec["dtype"]

        def go():
            if self.max_planes:
                raise SkipCase(K.RESTRICTED_SKIP)
            prec = self.core(dt)
            return self.pcg_case(spec, apply_core_schur_preconditioner, prec, f"core_schur_{dt}", package_prec=prec,
                                 extra_params=dict(preconditioner=f"core_schur_{dt}"))

        return self.guarded(spec, go)

    def best_solvax(self):
        best = None
        for m in K.SOLVAX_METHODS:
            for dt in K.DTYPES:
                p = self.path(K.case_id("precond", self.n, m, dt))
                if not p.is_file():
                    continue
                rec = json.loads(p.read_text())
                if rec["status"] != "ok":
                    continue
                if not rec["results"]["rel_err_vs_ref"] <= K.TOLERANCES["precond_rel_err"][m][dt]:
                    continue
                t = rec["results"]["apply"]["median_s"]
                if best is None or t < best[0]:
                    best = (t, m, dt)
        return best

    def run_pcg_solvax(self, spec):
        def go():
            if self.max_planes:
                raise SkipCase(K.RESTRICTED_SKIP)
            best = self.best_solvax()
            if best is None:
                raise SkipCase("skipped: no solvax variant (none ran ok)")
            t_apply, method, dt = best
            fn, args, info = self.make_precond(method, dt)
            state = args[0]
            apply_fn = lambda prec, r: fn(prec, r)                                 # noqa: E731
            return self.pcg_case(spec, apply_fn, state, f"solvax_{method}_{dt}",
                                 extra_params=dict(preconditioner=f"solvax_{method}_{dt}", chosen_by="min median apply time among ok "
                                                   "solvax precond cases of this backend and n", chosen_apply_median_s=t_apply))

        return self.guarded(spec, go)

    # ---- sharding --------------------------------------------------------------------------------------------------------
    def run_sharding(self, spec):
        def go():
            if self.max_planes:
                raise SkipCase(K.RESTRICTED_SKIP)
            if self.backend != "gpu" and os.environ.get("P_GPU_BENCH_SHARD_CPU") != "1":
                raise SkipCase("skipped: backend cpu (sharding is a GPU-node measurement)")
            devs = jax.devices()
            if len(devs) < 2:
                raise SkipCase(f"skipped: single device ({len(devs)} visible)")
            res = SH.run_sharding(self.core("f32"), self.rhs, reps=self.reps, devices=devs)
            return dict(method="core_schur", dtype="f32", planes=self.E, reps=self.reps), res

        return self.guarded(spec, go)

    # ---- driver ----------------------------------------------------------------------------------------------------------
    def run(self):
        self.todo = self.pending()
        if not self.todo:
            log(f"N{self.n}/{self.backend}: nothing pending")
            return []
        self.identity = dict(n=self.n, arm=K.ARM)
        self.setup()
        need = lambda dts, kinds=None, methods=None: any(                                   # noqa: E731
            s["dtype"] in dts and (kinds is None or s["kind"] in kinds) and (methods is None or s["method"] in methods)
            for s in self.todo)
        for dt in K.DTYPES:                                                                 # builds first (clean RSS)
            if need([dt]) or (dt == "f32" and any(s["kind"] == "sharding" for s in self.todo)) or \
                    (dt == "f64" and self.n > self.campaign["dense_reference_max_n"] and any(s["kind"] == "precond" for s in self.todo)):
                self.guarded_build(dt)
        for s in self.todo:
            if s["kind"] == "precond":
                self.run_precond(s)
        for s in self.todo:
            if s["kind"] == "pcg" and s["method"] == "core_schur":
                self.run_pcg_core(s)
        for s in self.todo:
            if s["kind"] == "pcg" and s["method"] == "solvax_best":
                self.run_pcg_solvax(s)
        for s in self.todo:
            if s["kind"] == "sharding":
                self.run_sharding(s)
        return self.todo

    def guarded_build(self, dt):
        spec = next(s for s in K.case_table([self.n]) if s["case_id"] == K.case_id("build", self.n, "core_schur", dt))
        t0 = time.perf_counter()
        try:
            self.core(dt)
        except BaseException as exc:                                                        # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            self.record(spec, "failed", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()[-4000:]}, {}, t0)


class SkipCase(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
