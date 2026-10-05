#!/usr/bin/env python3
"""Spectral and time-step stage of the nodal bracket (M3d): energy identity, numerical abscissa, |lambda|max and RK4 dt,
core-only/ring-only |lambda|, the ``c_kappa`` sweep, and (N32 only) the Cayley band and the RK4 power iteration.

One process per ``N``; results are written part by part to ``<out>/N{n}/spectral/<part>.json`` (a finished part is skipped
unless ``--force``). The operator is the production nodal bracket acting on a scalar field ``g`` with the velocity flux
``F`` of a prescribed potential (the pinned ``phi_switch`` or the control ``phi_wave``), zero wall data:

- ``L``     = ``linear_operator`` (no dissipation), whose energy bound is ``omega <= max(c / 2)``;
- ``A_ck``  = ``sbp_bracket`` = ``L`` plus the face-jump dissipation and the core shell damping with ``c_kappa = ck``.

Parts: ``energy`` (energy identity), ``omega`` (matrix-free numerical abscissa of ``L`` and ``A_1`` against ``max c/2``),
``lm`` (``eigs`` 'LM' from two start vectors for the full, core-only and ring-only production operator, RK4 dt),
``sweep`` (static errors, abscissa, |lambda|max and dt for ``c_kappa`` in the sweep), ``cayley`` and ``rk4``
(N32 only: they do not run at larger N). ``--summarize`` writes ``<out>/spectral_summary.json`` with the tables, the
core/ring scaling fit and its projection to N128/N256.

    python spectral.py N --out ROOT [--parts energy,omega,lm,sweep]
    python spectral.py 32 --out ROOT --parts cayley,rk4
    python spectral.py --summarize --out ROOT
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                                      # noqa: E402
import scipy.sparse.linalg as spla                                                      # noqa: E402

import fields as F                                                                      # noqa: E402
from references import RHO, load_layout_metric, log, peak_rss_gib                       # noqa: E402

CONFIG = F.CONFIG
GRIDS = tuple(CONFIG["resolutions"])
SWEEP = tuple(CONFIG["c_kappa_sweep"])
POTENTIALS = {"switch": ("switch", "phi_switch"), "wave": ("wave", "phi_wave")}
THRESHOLDS = {"error_change": 0.01, "order_change": 0.05, "dt_change": 0.05}


def sh(x):
    return float(x)


# ---------------------------------------------------------------------------------------------- setup
class Ctx:
    """Plan, device copy, jitted operator kernels and the potential-dependent flux of one resolution."""

    def __init__(self, n: int, out_root: Path):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from drbx.native import fci_perpendicular_sbp_bracket as br
        from drbx.native import fci_perpendicular_sbp_ops as ops
        from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks
        from drbx.stencils.nodal_plan import build_nodal_plan

        self.jax, self.jnp, self.br, self.ops = jax, jnp, br, ops
        self.n, self.out_root = n, out_root
        self.layout, self.metric, _ = load_layout_metric(n, out_root)
        self.plan = build_nodal_plan(self.layout, self.metric)
        self.plan_dev = jax.tree_util.tree_map(jnp.asarray, self.plan)
        self.E, self.P = self.layout.n_eta, self.layout.P
        self.Nc = self.layout.blocks[0].n_nodes
        self.H = np.asarray(h_weights(self.plan))
        self.masks = ring_region_masks(self.layout, n)
        halo = br.HALO

        def prod(plan, Fe, x, ck):
            return br.sbp_bracket_ext(plan, Fe, ops.extend_periodic(x, halo), None, ck)

        def lin(plan, Fe, x):
            return br.linear_operator_ext(plan, Fe, ops.extend_periodic(x, halo), None)

        def embed(sel, z):                               # z (E, Nsel) -> (E, P)
            full = jnp.zeros((self.E, self.P))
            return full.at[:, sel].set(z.reshape(self.E, -1))

        self._prod = jax.jit(lambda plan, Fe, x, ck: prod(plan, Fe, x, ck))
        self._lin = jax.jit(lin)
        self._vjp_prod = jax.jit(lambda plan, Fe, x, y, ck: jax.vjp(lambda v: prod(plan, Fe, v, ck), x)[1](y)[0])
        self._vjp_lin = jax.jit(lambda plan, Fe, x, y: jax.vjp(lambda v: lin(plan, Fe, v), x)[1](y)[0])
        core_sel, ring_sel = slice(0, self.Nc), slice(self.Nc, self.P)
        self._prod_core = jax.jit(lambda plan, Fe, z, ck: prod(plan, Fe, embed(core_sel, z), ck)[:, core_sel])
        self._prod_ring = jax.jit(lambda plan, Fe, z, ck: prod(plan, Fe, embed(ring_sel, z), ck)[:, ring_sel])
        self._flux = jax.jit(lambda plan, phi, rho: br.velocity_flux(plan, phi, rho))
        self._comp = jax.jit(lambda plan, Fe: br.compressibility_ext(plan, Fe))
        self._transport = jax.jit(lambda plan, F_, g: br.transport(plan, F_, g))
        self._advect = jax.jit(lambda plan, Fe, ge: br.advect(plan, Fe, ge))
        self.matvecs = 0

    def potential(self, which: str) -> np.ndarray:
        z = np.load(self.out_root / f"N{self.n}" / "references" / f"{POTENTIALS[which][0]}.npz")
        names = [str(x) for x in z["names"]]
        return z["vals"][names.index(POTENTIALS[which][1])]

    def flux(self, which: str):
        phi = self.jnp.asarray(self.potential(which))
        Fl = self._flux(self.plan_dev, phi, RHO)
        return Fl, self.ops.extend_periodic(Fl, self.br.HALO)

    # -- operator callables (flat vectors) ---------------------------------------------------
    def op_prod(self, Fe, ck: float):
        shape = (self.E, self.P)

        def mv(x):
            self.matvecs += 1
            return np.asarray(self._prod(self.plan_dev, Fe, self.jnp.asarray(np.asarray(x, dtype=np.float64).reshape(shape)), ck)).reshape(-1)

        def rmv(y):
            self.matvecs += 1
            x0 = self.jnp.zeros(shape)
            return np.asarray(self._vjp_prod(self.plan_dev, Fe, x0, self.jnp.asarray(np.asarray(y, dtype=np.float64).reshape(shape)), ck)).reshape(-1)

        return spla.LinearOperator((self.E * self.P,) * 2, matvec=mv, rmatvec=rmv, dtype=np.float64)

    def op_lin(self, Fe):
        shape = (self.E, self.P)

        def mv(x):
            self.matvecs += 1
            return np.asarray(self._lin(self.plan_dev, Fe, self.jnp.asarray(np.asarray(x, dtype=np.float64).reshape(shape)))).reshape(-1)

        def rmv(y):
            self.matvecs += 1
            return np.asarray(self._vjp_lin(self.plan_dev, Fe, self.jnp.zeros(shape),
                                            self.jnp.asarray(np.asarray(y, dtype=np.float64).reshape(shape)))).reshape(-1)

        return spla.LinearOperator((self.E * self.P,) * 2, matvec=mv, rmatvec=rmv, dtype=np.float64)

    def op_block(self, Fe, ck: float, which: str):
        fn = self._prod_core if which == "core" else self._prod_ring
        n_sel = self.Nc if which == "core" else self.P - self.Nc

        def mv(x):
            self.matvecs += 1
            z = self.jnp.asarray(np.asarray(x, dtype=np.float64).reshape(self.E, n_sel))
            return np.asarray(fn(self.plan_dev, Fe, z, ck)).reshape(-1)

        return spla.LinearOperator((self.E * n_sel,) * 2, matvec=mv, dtype=np.float64)


# ---------------------------------------------------------------------------------------------- eigen helpers
def rk4_amp(z):
    return np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)


def rk4_dt(eigs):
    from drbx.validation.sbp_audit import rk4_dt as _rk4_dt
    return float(_rk4_dt(eigs))


def lm_eigs(op, k: int = 6, seeds=(0, 1), ncv: int = 40, tol: float = 1e-6, maxiter: int = 4000) -> dict:
    """``eigs`` 'LM' from two start vectors: eigenvalues, |lambda|max of each, their relative difference and the RK4 dt."""
    out, allv = {}, []
    for s in seeds:
        t0 = time.perf_counter()
        v0 = np.random.default_rng(s).standard_normal(op.shape[0])
        try:
            ev = spla.eigs(op, k=k, which="LM", ncv=min(ncv, op.shape[0] - 2), tol=tol, maxiter=maxiter, v0=v0,
                           return_eigenvectors=False)
            conv = True
        except spla.ArpackNoConvergence as exc:
            ev, conv = exc.eigenvalues, False
        ev = ev[np.argsort(-np.abs(ev))]
        out[f"start{s}"] = {"eig": [[sh(e.real), sh(e.imag)] for e in ev], "converged": conv,
                            "absmax": sh(np.abs(ev).max()) if len(ev) else None, "seconds": time.perf_counter() - t0}
        allv.extend(ev.tolist())
    mags = [out[f"start{s}"]["absmax"] for s in seeds if out[f"start{s}"]["absmax"] is not None]
    out["absmax"] = max(mags) if mags else None
    out["rel_diff_starts"] = (abs(mags[0] - mags[1]) / out["absmax"]) if len(mags) == 2 else None
    out["dt_rk4"] = rk4_dt(allv) if allv else None
    return out


def abscissa(op, H, ncv: int = 60, tol: float = 1e-6, maxiter: int = 20000) -> dict:
    """Largest eigenvalue of ``sym(H^(1/2) L H^(-1/2))`` by ``eigsh`` 'LA' (matrix free; needs ``rmatvec``)."""
    h = np.asarray(H, dtype=np.float64).reshape(-1)
    d, di = np.sqrt(h), 1.0 / np.sqrt(h)

    def sym(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return 0.5 * (d * op.matvec(di * x) + di * op.rmatvec(d * x))

    sop = spla.LinearOperator(op.shape, matvec=sym, dtype=np.float64)
    t0 = time.perf_counter()
    try:
        ev = spla.eigsh(sop, k=1, which="LA", ncv=min(ncv, op.shape[0] - 1), tol=tol, maxiter=maxiter,
                        v0=np.random.default_rng(0).standard_normal(op.shape[0]), return_eigenvectors=False)
        conv = True
    except spla.ArpackNoConvergence as exc:
        ev, conv = exc.eigenvalues, False
    return {"omega": sh(ev.max()) if len(ev) else None, "converged": conv, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------------------------- parts
def part_energy(ctx: Ctx) -> dict:
    from drbx.validation.sbp_audit import energy_identity

    out = {}
    plan = ctx.plan_dev
    for which in POTENTIALS:
        Fl, Fe = ctx.flux(which)
        Fn = np.asarray(Fl)

        def T_apply(g):
            return np.asarray(ctx._transport(plan, Fl, ctx.jnp.asarray(g)))

        def wall_term(g):
            total = 0.0
            for blk, side, sign, N in ctx.plan.structure.walls:
                a = np.asarray(ctx.ops.trace(plan, blk, side, ctx.jnp.asarray(g)))
                v = np.asarray(ctx.ops.flux_trace(plan, blk, side, Fn[..., 0], Fn[..., 1]))
                total += -0.5 * (2 * np.pi / N) * sign * np.sum(v * a * a)
            return total

        def scale_apply(g):
            return np.asarray(ctx._advect(plan, Fe, ctx.ops.extend_periodic(ctx.jnp.asarray(g), ctx.br.HALO)))

        res = [energy_identity(T_apply, ctx.plan.Hp, wall_term, np.random.default_rng(s), scale_apply=scale_apply)
               for s in (0, 1)]
        out[which] = {"residuals": res, "max": max(res), "gate": 1e-13, "pass": bool(max(res) <= 1e-13)}
        log(f"N{ctx.n} energy {which}: {max(res):.2e}")
    return out


def part_omega(ctx: Ctx) -> dict:
    out = {}
    for which in POTENTIALS:
        Fl, Fe = ctx.flux(which)
        c = np.asarray(ctx._comp(ctx.plan_dev, Fe))
        half_c_max = float(0.5 * c.max())
        row = {"max_half_c": half_c_max, "min_half_c": float(0.5 * c.min())}
        ctx.matvecs = 0
        om = abscissa(ctx.op_lin(Fe), ctx.H)
        row["L_no_dissipation"] = {**om, "ratio_to_max_half_c": (om["omega"] / half_c_max) if half_c_max > 0 else None,
                                   "bound_holds": bool(om["omega"] <= half_c_max * (1 + 1e-8)) if half_c_max > 0 else
                                   bool(om["omega"] <= half_c_max + 1e-8 * abs(half_c_max)), "matvecs": ctx.matvecs}
        ctx.matvecs = 0
        omd = abscissa(ctx.op_prod(Fe, float(CONFIG["c_kappa"])), ctx.H)
        row["production_ck1"] = {**omd, "matvecs": ctx.matvecs}
        out[which] = row
        log(f"N{ctx.n} omega {which}: L {om['omega']:.4g} (max c/2 {half_c_max:.4g}, ratio {row['L_no_dissipation']['ratio_to_max_half_c']}), "
            f"production {omd['omega']:.4g}")
    return out


def lm_block(ctx: Ctx, Fe, ck: float, full: bool = True, blocks: bool = True) -> dict:
    row = {}
    if full:
        ctx.matvecs = 0
        row["full"] = lm_eigs(ctx.op_prod(Fe, ck))
        row["full"]["matvecs"] = ctx.matvecs
    if blocks:
        for which in ("core", "ring"):
            ctx.matvecs = 0
            row[which] = lm_eigs(ctx.op_block(Fe, ck, which), ncv=30)
            row[which]["matvecs"] = ctx.matvecs
    return row


def part_lm(ctx: Ctx) -> dict:
    out = {}
    for which in POTENTIALS:
        Fl, Fe = ctx.flux(which)
        row = lm_block(ctx, Fe, float(CONFIG["c_kappa"]))
        V = np.asarray(Fl)[:, :ctx.Nc, :2] / np.asarray(ctx.plan.jac)[:, :ctx.Nc, None]
        core = ctx.plan.structure.core
        row["core_model"] = {"p": core[1], "R_c": core[2], "vxy_max": float(np.sqrt((V**2).sum(-1)).max()),
                             "p2_vxy_over_Rc": float(core[1] ** 2 * np.sqrt((V**2).sum(-1)).max() / core[2])}
        out[which] = row
        log(f"N{ctx.n} lm {which}: full {row['full']['absmax']:.4g} (starts diff {row['full']['rel_diff_starts']}), "
            f"core {row['core']['absmax']:.4g}, ring {row['ring']['absmax']:.4g}, dt {row['full']['dt_rk4']:.3e}")
    return out


def sweep_static(ctx: Ctx, ck: float) -> dict:
    """Production errors (global and core, H-rms over the plane subset of each catalogue) for every pair with ``ck``."""
    from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData

    plan, jnp = ctx.plan_dev, ctx.jnp
    sbp = ctx.jax.jit(lambda p, phi, g, bcd, rho, c: ctx.br.sbp_bracket(p, phi, g, bcd, rho, c))
    core_mask = ctx.masks["core"]
    out = {}
    for name in F.SET_NAMES:
        rpath = ctx.out_root / f"N{ctx.n}" / "references" / f"{name}.npz"
        if not rpath.exists():
            continue
        z = np.load(rpath)
        names = [str(x) for x in z["names"]]
        pairs = [F.Pair(**d) for d in json.loads(str(z["pairs"]))]
        planes = z["planes"] if "planes" in z.files else np.arange(ctx.E)
        idx = {nm: i for i, nm in enumerate(names)}
        vals, wall, R = z["vals"], z["wall"], z["R"]
        rows = {}
        for a in dict.fromkeys(p.a for p in pairs):
            members = [k for k, p in enumerate(pairs) if p.a == a]
            g = np.stack([vals[idx[pairs[k].b]] for k in members], axis=-1)
            wv = np.stack([wall[idx[pairs[k].b]] for k in members], axis=-1)
            res = np.asarray(sbp(plan, vals[idx[a]], g, SatBoundaryData((jnp.asarray(wv),), None), RHO, ck))
            for j, k in enumerate(members):
                if pairs[k].constant:
                    continue
                e = res[..., j][planes] - R[k]
                H = ctx.H[planes]
                cm = core_mask[planes]
                rows[pairs[k].label] = {"E": float(np.sqrt(np.sum(H * e * e) / np.sum(H))),
                                        "E_core": float(np.sqrt(np.sum(H[cm] * e[cm] ** 2) / np.sum(H[cm]))),
                                        "gate": pairs[k].gate}
        out[name] = rows
    return out


def part_sweep(ctx: Ctx) -> dict:
    out = {}
    for ck in SWEEP:
        t0 = time.perf_counter()
        row = {"static": sweep_static(ctx, ck)}
        for which in POTENTIALS:
            Fl, Fe = ctx.flux(which)
            ctx.matvecs = 0
            om = abscissa(ctx.op_prod(Fe, ck), ctx.H)
            lm = lm_block(ctx, Fe, ck, full=True, blocks=False)["full"]
            row[which] = {"omega": om["omega"], "omega_converged": om["converged"], "absmax": lm["absmax"],
                          "rel_diff_starts": lm["rel_diff_starts"], "dt_rk4": lm["dt_rk4"], "converged": lm["start0"]["converged"]}
        row["seconds"] = time.perf_counter() - t0
        out[f"ck{ck}"] = row
        log(f"N{ctx.n} sweep ck={ck}: switch |lam| {row['switch']['absmax']:.4g} dt {row['switch']['dt_rk4']:.3e} "
            f"omega {row['switch']['omega']:.3g} ({row['seconds']:.0f} s)")
    return out


# ---------------------------------------------------------------------------------------------- N32 only
def assemble_matrix(ctx: Ctx, Fe, ck: float):
    """Sparse production operator by probing (index ``k * P + p``)."""
    from drbx.validation.sbp_audit import assemble_by_probing

    def apply(x):
        return ctx.br.sbp_bracket_ext(ctx.plan_dev, Fe, ctx.ops.extend_periodic(x, ctx.br.HALO), None, ck)

    return assemble_by_probing(apply, ctx.P, ctx.E, batch=32)


def cayley(A, sigma: float, k: int = 12, ncv: int = 80, tol: float = 1e-10, maxiter: int = 500, gate: float = 1e-8):
    import scipy.sparse as sp

    n = A.shape[0]
    lu = spla.splu((A - sigma * sp.identity(n, format="csc")).tocsc())
    Ap = (A + sigma * sp.identity(n, format="csr")).tocsr()
    op = spla.LinearOperator((n, n), matvec=lambda x: lu.solve(Ap @ np.asarray(x, dtype=np.float64)), dtype=np.float64)
    try:
        mu, vec = spla.eigs(op, k=k, which="LM", ncv=ncv, tol=tol, maxiter=maxiter,
                            v0=np.random.default_rng(0).standard_normal(n))
        conv = True
    except spla.ArpackNoConvergence as exc:
        mu, vec, conv = exc.eigenvalues, exc.eigenvectors, False
    lam = sigma * (mu + 1.0) / (mu - 1.0)
    keep = []
    for i in range(len(lam)):
        x = vec[:, i]
        r = np.linalg.norm(A @ x - lam[i] * x) / max(np.linalg.norm(A @ x), np.linalg.norm(lam[i] * x), 1e-300)
        keep.append((lam[i], r, r <= gate))
    return conv, keep


def part_cayley(ctx: Ctx, potentials=("switch",), maxiter: int = 80, tol: float = 1e-9) -> dict:
    """Cayley band at N32 (sigma = 2 |lambda|max of the production operator): ``k = 12``, ``ncv = 80``, residual gate 1e-8.

    The assembled matrix is cached next to the results (``A_<potential>.npz``) so that a rerun only repeats the solves.
    """
    import scipy.sparse as sp

    if ctx.n != 32:
        raise ValueError("the Cayley band runs at N32 only (N48/N64: not run, remote option)")
    out = {}
    path = ctx.out_root / f"N{ctx.n}" / "spectral"
    lm = json.loads((path / "lm.json").read_text())
    for which in potentials:
        Fl, Fe = ctx.flux(which)
        rho_spec = lm[which]["full"]["absmax"]
        sigma = 2.0 * rho_spec
        t0 = time.perf_counter()
        cache = path / f"A_{which}.npz"
        if cache.exists():
            A = sp.load_npz(cache).tocsr()
        else:
            A = assemble_matrix(ctx, Fe, float(CONFIG["c_kappa"]))
            sp.save_npz(cache, A.tocsr())
        t_asm = time.perf_counter() - t0
        t1 = time.perf_counter()
        conv, band = cayley(A, sigma, maxiter=maxiter, tol=tol)
        accepted = sorted([(l, r) for l, r, ok in band if ok], key=lambda t: -t[0].real)
        out[which] = {"rho_lm": rho_spec, "sigma": sigma, "nnz": int(A.nnz), "assemble_seconds": t_asm,
                      "cayley_seconds": time.perf_counter() - t1, "converged": conv, "k": 12, "ncv": 80, "maxiter": maxiter,
                      "tol": tol, "accepted": [[sh(l.real), sh(l.imag), sh(r)] for l, r in accepted],
                      "n_ritz": len(band), "n_rejected": len(band) - len(accepted),
                      "rightmost_real": sh(accepted[0][0].real) if accepted else None}
        log(f"N32 cayley {which}: sigma {sigma:.4g}, converged {conv}, accepted {len(accepted)}/{len(band)}, rightmost Re "
            f"{out[which]['rightmost_real']}  ({out[which]['cayley_seconds']:.0f} s)")
        del A
    return out


def part_rk4(ctx: Ctx, cfl: float = 0.1, window: float = 0.004, t_max: float = 0.2, max_wall: float = 4800.0) -> dict:
    """RK4 power iteration on the production operator: window growth rates until three consecutive agree to 1%."""
    if ctx.n != 32:
        raise ValueError("the RK4 power iteration runs at N32 only (N48/N64: not run, remote option)")
    jax, jnp = ctx.jax, ctx.jnp
    lm = json.loads((ctx.out_root / f"N{ctx.n}" / "spectral" / "lm.json").read_text())
    out = {}
    gradients = F.transverse_set("switch").evaluate(
        np.stack([np.broadcast_to(ctx.layout.node_u, (ctx.E, ctx.P)), np.broadcast_to(ctx.layout.node_theta, (ctx.E, ctx.P)),
                  np.broadcast_to(((np.arange(ctx.E) + 0.5) * ctx.layout.deta)[:, None], (ctx.E, ctx.P))], -1))[1]
    for which in ("switch",):
        Fl, Fe = ctx.flux(which)
        c = np.asarray(ctx._comp(ctx.plan_dev, Fe))
        if which == "switch":
            grad_phi = gradients[4].reshape(ctx.E, ctx.P, 3)
            S = 0.5 * 2 * np.sum(ctx.metric.K * grad_phi, axis=-1) / (ctx.metric.B * RHO)
        dt = cfl * lm[which]["full"]["dt_rk4"]
        n_win = max(1, int(round(window / dt)))
        dt = window / n_win
        c_dev = jnp.asarray(c)
        step = jax.jit(lambda plan, Fl_, Fe_, cc, v, ck: _rk4_step(ctx, plan, Fl_, Fe_, cc, v, ck, dt))
        Hd = jnp.asarray(ctx.H)
        v = jnp.asarray(np.random.default_rng(11).standard_normal((ctx.E, ctx.P)))
        v = v / jnp.sqrt(jnp.sum(Hd * v * v))
        rates, t, t0 = [], 0.0, time.perf_counter()
        while t < t_max - 1e-12:
            logg = 0.0
            for _ in range(n_win):
                v = step(ctx.plan_dev, Fl, Fe, c_dev, v, float(CONFIG["c_kappa"]))
                nrm = jnp.sqrt(jnp.sum(Hd * v * v))
                logg += float(jnp.log(nrm))
                v = v / nrm
            t += window
            rates.append(logg / window)
            if len(rates) >= 4 and all(abs(rates[-i] - rates[-i - 1]) < 0.01 * abs(rates[-1]) for i in (1, 2, 3)):
                break
            log(f"N32 rk4 window {len(rates)}: rate {rates[-1]:.5g} (t = {t:.3f}, {time.perf_counter() - t0:.0f} s)")
            if time.perf_counter() - t0 > max_wall:
                break
        vv = np.asarray(v)
        Av = np.asarray(ctx._prod(ctx.plan_dev, Fe, v, float(CONFIG["c_kappa"])))
        rq = float(np.sum(ctx.H * vv * Av) / np.sum(ctx.H * vv * vv))
        w = ctx.H * vv**2
        w = w / w.sum()
        out[which] = {"dt": dt, "cfl": cfl, "steps_per_window": n_win, "window": window, "windows": len(rates),
                      "window_rates": rates, "t_end": t, "settled": bool(len(rates) >= 4 and all(
                          abs(rates[-i] - rates[-i - 1]) < 0.01 * abs(rates[-1]) for i in (1, 2, 3))),
                      "stopped_by_wall_clock_cap": bool(time.perf_counter() - t0 > max_wall), "max_wall_seconds": max_wall,
                      "rayleigh_quotient": rq, "mode_weighted_half_c": float(np.sum(w * 0.5 * c)),
                      "mode_weighted_S": float(np.sum(w * S)),
                      "localization": {k: float(w[m].sum()) for k, m in ctx.masks.items() if not k.startswith("uband")},
                      "seconds": time.perf_counter() - t0}
        log(f"N32 rk4 {which}: rates {[round(r, 3) for r in rates]}, RQ {rq:.4g}, settled {out[which]['settled']} "
            f"({out[which]['seconds']:.0f} s)")
    return out


def _rk4_step(ctx: Ctx, plan, Fl, Fe, c, v, ck, dt):
    """One RK4 step of ``dv/dt = A_ck v`` with the compressibility ``c`` fixed (no per-stage recomputation)."""
    from drbx.native import fci_perpendicular_sbp_dissipation as dis
    from drbx.native import fci_perpendicular_sbp_ops as ops

    br = ctx.br

    def f(x):
        xe = ops.extend_periodic(x, br.HALO)
        return (br.transport_ext(plan, Fe, xe) + 0.5 * c * x + br.wall_inflow(plan, Fl, x, None)
                + br.interface_upwind(plan, Fl, x) + dis.dissipation_ext(plan, Fe, xe, ck))

    k1 = f(v)
    k2 = f(v + 0.5 * dt * k1)
    k3 = f(v + 0.5 * dt * k2)
    k4 = f(v + dt * k3)
    return v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


PARTS = {"energy": part_energy, "omega": part_omega, "lm": part_lm, "sweep": part_sweep, "cayley": part_cayley,
         "rk4": part_rk4}
N32_ONLY = ("cayley", "rk4")


def run(n: int, out_root: Path, parts, force: bool = False) -> dict:
    t0 = time.perf_counter()
    path = out_root / f"N{n}" / "spectral"
    path.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(n, out_root)
    log(f"N{n} spectral: P={ctx.P}, E={ctx.E}, unknowns {ctx.E * ctx.P}")
    receipts = {}
    for part in parts:
        if part in N32_ONLY and n != 32:
            (path / f"{part}.json").write_text(json.dumps({"status": "not run (remote option)", "n": n}))
            continue
        f = path / f"{part}.json"
        if f.exists() and not force:
            log(f"N{n} {part}: exists, skipped")
            continue
        ts = time.perf_counter()
        res = PARTS[part](ctx)
        res["_seconds"] = time.perf_counter() - ts
        res["_peak_rss_gib"] = peak_rss_gib()
        f.write_text(json.dumps(res, indent=1))
        receipts[part] = {"seconds": res["_seconds"], "peak_rss_gib": res["_peak_rss_gib"]}
    out = {"n": n, "seconds": time.perf_counter() - t0, "peak_rss_gib": peak_rss_gib(), "parts": receipts}
    (out_root / f"N{n}" / f"spectral_receipt_{'_'.join(parts)}.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------- summary
def powerlaw(ns, ys):
    a, b = np.polyfit(np.log(ns), np.log(ys), 1)
    return float(a), float(np.exp(b))


def summarize(out_root: Path) -> dict:
    ns = [n for n in GRIDS if (out_root / f"N{n}" / "spectral" / "lm.json").exists()]
    load = lambda n, part: json.loads((out_root / f"N{n}" / "spectral" / f"{part}.json").read_text())      # noqa: E731
    S = {"grids": ns, "thresholds": THRESHOLDS, "tables": {}}
    for which in POTENTIALS:
        rows = {}
        for n in ns:
            en = load(n, "energy")[which]["max"] if (out_root / f"N{n}" / "spectral" / "energy.json").exists() else None
            om = load(n, "omega")[which] if (out_root / f"N{n}" / "spectral" / "omega.json").exists() else None
            lm = load(n, "lm")[which]
            rows[n] = {"energy_identity": en,
                       "omega_L": om["L_no_dissipation"]["omega"] if om else None,
                       "max_half_c": om["max_half_c"] if om else None,
                       "omega_over_max_half_c": om["L_no_dissipation"]["ratio_to_max_half_c"] if om else None,
                       "bound_holds": om["L_no_dissipation"]["bound_holds"] if om else None,
                       "omega_production": om["production_ck1"]["omega"] if om else None,
                       "lambda_max": lm["full"]["absmax"], "lambda_rel_diff_starts": lm["full"]["rel_diff_starts"],
                       "lambda_core": lm["core"]["absmax"], "lambda_ring": lm["ring"]["absmax"],
                       "dt_rk4": lm["full"]["dt_rk4"], "dt_core_only": lm["core"]["dt_rk4"], "dt_ring_only": lm["ring"]["dt_rk4"],
                       "p2_vxy_over_Rc": lm["core_model"]["p2_vxy_over_Rc"], "vxy_max": lm["core_model"]["vxy_max"]}
        tab = {"rows": rows}
        if len(ns) >= 2:
            ac, bc = powerlaw(ns, [rows[n]["lambda_core"] for n in ns])
            ar, br_ = powerlaw(ns, [rows[n]["lambda_ring"] for n in ns])
            c_model = np.mean([rows[n]["lambda_core"] / rows[n]["p2_vxy_over_Rc"] for n in ns])
            v_ref = rows[ns[-1]]["vxy_max"]                               # |V_xy| of the core is resolution independent
            proj = {}
            for n in (128, 256):
                p = n // 8 + 2
                lc_fit, lr_fit = bc * n**ac, br_ * n**ar
                lc_model = c_model * p**2 * v_ref * 8.0                   # R_c = 1/8 for family A
                proj[n] = {"lambda_core_fit": lc_fit, "lambda_ring_fit": lr_fit, "lambda_core_model": float(lc_model),
                           "dt_fit": rk4_dt([max(lc_fit, lr_fit) * 1j]), "dt_model": rk4_dt([max(lc_model, lr_fit) * 1j]),
                           "core_sets_dt_fit": bool(lc_fit > lr_fit), "core_sets_dt_model": bool(lc_model > lr_fit)}
            n_cross = (br_ / bc) ** (1.0 / (ac - ar)) if ac != ar else None
            tab.update({"fit_core": {"exponent": ac, "prefactor": bc}, "fit_ring": {"exponent": ar, "prefactor": br_},
                        "core_over_p2V_Rc": float(c_model), "projection": proj,
                        "n_where_core_equals_ring_fit": float(n_cross) if n_cross else None,
                        "core_sets_dt_measured": {n: bool(rows[n]["lambda_core"] > rows[n]["lambda_ring"]) for n in ns}})
        S["tables"][which] = tab
    cay = out_root / "N32" / "spectral" / "cayley.json"
    rk = out_root / "N32" / "spectral" / "rk4.json"
    S["cayley_N32"] = json.loads(cay.read_text()) if cay.exists() else None
    S["rk4_N32"] = json.loads(rk.read_text()) if rk.exists() else None
    S["cayley_rk4_N48_N64"] = "not run (remote option)"
    S["sweep"] = sweep_summary(out_root, ns, load)
    (out_root / "spectral_summary.json").write_text(json.dumps(S, indent=1))
    return S


def sweep_summary(out_root: Path, ns, load) -> dict:
    ns = [n for n in ns if (out_root / f"N{n}" / "spectral" / "sweep.json").exists()]
    if not ns:
        return {}
    data = {n: load(n, "sweep") for n in ns}
    ref_key = f"ck{float(CONFIG['c_kappa'])}"
    out = {"c_kappa": list(SWEEP), "grids": ns, "per_ck": {}, "thresholds": THRESHOLDS}
    def orders(E):
        return [math.log(E[i] / E[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(ns) - 1)]
    for ck in SWEEP:
        key = f"ck{ck}"
        res = {"switch": {}, "wave": {}}
        for which in ("switch", "wave"):
            res[which] = {"omega": [data[n][key][which]["omega"] for n in ns], "lambda_max": [data[n][key][which]["absmax"] for n in ns],
                          "dt": [data[n][key][which]["dt_rk4"] for n in ns]}
        stat = {}
        for name in data[ns[0]][key]["static"]:
            for label, row in data[ns[0]][key]["static"][name].items():
                if not all(label in data[n][key]["static"].get(name, {}) for n in ns):
                    continue
                E = [data[n][key]["static"][name][label]["E"] for n in ns]
                Ec = [data[n][key]["static"][name][label]["E_core"] for n in ns]
                stat[f"{name}:{label}"] = {"gate": row["gate"], "E": E, "E_core": Ec, "orders": orders(E) if len(ns) > 1 else []}
        res["static"] = stat
        out["per_ck"][key] = res
    ref = out["per_ck"][ref_key]
    cmp_ = {}
    for ck in SWEEP:
        if ck == float(CONFIG["c_kappa"]):
            continue
        cur = out["per_ck"][f"ck{ck}"]
        dE, dEc, dOrd = [], [], []
        for label, row in cur["static"].items():
            r = ref["static"][label]
            dE += [abs(a - b) / b for a, b in zip(row["E"], r["E"]) if b > 0]
            dEc += [abs(a - b) / b for a, b in zip(row["E_core"], r["E_core"]) if b > 0]
            dOrd += [abs(a - b) for a, b in zip(row["orders"], r["orders"])]
        ddt = [abs(a - b) / b for w in ("switch", "wave") for a, b in zip(cur[w]["dt"], ref[w]["dt"])]
        cmp_[f"ck{ck}_vs_ck1"] = {"max_rel_change_E_global": max(dE), "max_rel_change_E_core": max(dEc),
                                  "max_order_change": max(dOrd), "max_rel_change_dt": max(ddt),
                                  "within_thresholds": bool(max(dE) < THRESHOLDS["error_change"] and max(dOrd) < THRESHOLDS["order_change"]
                                                            and max(ddt) < THRESHOLDS["dt_change"])}
    out["comparison"] = cmp_
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", default="energy,omega,lm,sweep")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.out)
        return 0
    run(args.n, args.out, args.parts.split(","), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
