#!/usr/bin/env python3
"""Face coefficients and definiteness audit of the nodal Laplacian plan at one arm and resolution.

1. **Face coefficients.** min / max of ``auu_f (E, m + 1, N)``, ``att_h (E, m, N)`` and ``aee_h (E, P)`` (the radial Lagrange,
   angular and eta Fourier half-node interpolants of the diagonal of ``A``; the Fourier ones can ring negative where ``A``
   varies sharply), the nodal diagonals for reference, and every negative entry with its location.
2. **Definiteness audit.** ``audit_laplacian_plan`` (H-symmetry; lowest eigenvalues of the H-normalised symmetric part of the
   Dirichlet and Neumann energy matrices by bounded LOBPCG), with the eigenvectors of any direction below the noise floor
   located by region (core, core band, wall rings, interior), ring, theta seam and eta seam participation.

    python audit.py N --arm raw|filtered [--out ROOT] [--mode host|mf|both] [--k 4] [--maxiter 600] [--tol 1e-15]
        [--variant interp|ee|ee_tt|all] [--faces-root ROOT]

``--mode host``: the package ``audit_laplacian_plan`` (host sparse assembly; N32 only within the 4 GB budget), once with the
package defaults (80 LOBPCG iterations, tolerance 1e-7 rho) and once refined (the near-null Neumann cluster sits at ~1e-12 rho,
below the default tolerance). ``--mode mf``: the matrix-free LOBPCG of ``audit_mf.py`` (any N). ``--estimate-only`` prints the
host-assembly memory estimate. Output ``<out>/<arm>/N<N>/audit<tag>.json``. ``--variant`` takes the listed face families from the
tensor evaluated at the faces (``extract_metric.py --faces``) and writes to ``<faces-root>/<arm>/N<N>/<variant>/`` instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG, log                                                          # noqa: E402

import numpy as np                                                                      # noqa: E402

EPS = float(np.finfo(np.float64).eps)


def _neg_list(a: np.ndarray, decode, top: int = 12):
    """Count and the ``top`` most negative entries of ``a`` with ``decode(flat_index) -> dict``."""
    flat = a.ravel()
    idx = np.flatnonzero(flat < 0)
    out = {"count": int(idx.size), "fraction": float(idx.size / flat.size), "entries": []}
    if idx.size:
        order = idx[np.argsort(flat[idx])][:top]
        out["entries"] = [dict(value=float(flat[i]), **decode(int(i))) for i in order]
    return out


def coefficient_report(lp) -> dict:
    st = lp.structure
    E, P, Nc, m, N, i0 = st.n_eta, st.P, st.Nc, st.m, st.N, st.i0
    auu, att, aee = np.asarray(lp.auu_f), np.asarray(lp.att_h), np.asarray(lp.aee_h)
    A = np.asarray(lp.A)
    ring = A[:, Nc:]

    def dec_auu(i):
        k, f, j = np.unravel_index(i, auu.shape)
        return dict(plane=int(k), face=int(f), where="core|ring interface" if f == 0 else ("wall" if f == m else "ring face"),
                    face_ring_below=int(i0 + f - 1) if f > 0 else None, theta_index=int(j))

    def dec_att(i):
        k, r, j = np.unravel_index(i, att.shape)
        return dict(plane=int(k), ring=int(i0 + r), theta_half_index=int(j))

    def dec_aee(i):
        k, p = np.unravel_index(i, aee.shape)
        return dict(plane=int(k), node=int(p), region="core" if p < Nc else "ring",
                    ring=None if p < Nc else int(i0 + (p - Nc) // N), theta_index=int(p) if p < Nc else int((p - Nc) % N))

    def ext(a):
        return dict(min=float(a.min()), max=float(a.max()), ratio_min_over_max=float(a.min() / a.max()))

    return dict(
        auu_f=dict(**ext(auu), negative=_neg_list(auu, dec_auu), shape=list(auu.shape)),
        att_h=dict(**ext(att), negative=_neg_list(att, dec_att), shape=list(att.shape)),
        aee_h=dict(**ext(aee), negative=_neg_list(aee, dec_aee), shape=list(aee.shape)),
        nodal_diagonal=dict(Auu_ring=ext(ring[..., 0, 0]), Att_ring=ext(ring[..., 1, 1]), Aee_all=ext(A[..., 2, 2]),
                            Axx_core=ext(A[:, :Nc, 0, 0]), Ayy_core=ext(A[:, :Nc, 1, 1]), Aee_core=ext(A[:, :Nc, 2, 2])),
        penalties=dict(tau=float(lp.tau), tau_w=float(lp.tau_w), kappa_max=float(np.max(lp.kappa))),
    )


def locate(vec, layout, H, top: int = 6) -> dict:
    """Participation of the nodal vector ``vec (E, P)`` (energy weights ``H vec^2``) by region, ring, seam and plane."""
    from drbx.native.fci_perpendicular_sbp_norms import ring_region_masks

    E, P = layout.n_eta, layout.P
    n = layout.n
    w = H * vec * vec
    w = w / w.sum()
    masks = ring_region_masks(layout, n)
    vol = H / H.sum()
    regions = {k: dict(participation=float(w[m].sum()), volume=float(vol[m].sum()),
                       enrichment=float(w[m].sum() / vol[m].sum())) for k, m in masks.items()
               if not k.startswith("uband")}
    ring = np.broadcast_to(np.asarray(layout.node_ring)[None, :], (E, P))
    per_ring = {int(r): float(w[ring == r].sum()) for r in np.unique(ring)}
    top_rings = sorted(per_ring.items(), key=lambda kv: -kv[1])[:top]
    plane = np.broadcast_to(np.arange(E)[:, None], (E, P))
    theta_idx = np.broadcast_to(np.concatenate([np.full(layout.blocks[0].n_nodes, -1),
                                                np.tile(np.arange(layout.blocks[1].N), layout.blocks[1].m)])[None, :], (E, P))
    seam_eta = (plane <= 1) | (plane >= E - 2)
    seam_theta = (theta_idx >= 0) & ((theta_idx <= 1) | (theta_idx >= layout.blocks[1].N - 2))
    per_plane = w.sum(axis=1)
    k0 = np.unravel_index(np.argmax(np.abs(vec)), vec.shape)
    pts = [dict(plane=int(i), node=int(p), ring=int(ring[i, p]), u=float(layout.node_u[p]), theta=float(layout.node_theta[p]),
                eta=float((i + 0.5) * layout.deta), weight=float(w[i, p])) for i, p in
           (np.unravel_index(j, w.shape) for j in np.argsort(w.ravel())[::-1][:top])]
    # toroidal-only content: the part of v that is constant on every eta plane (radially/angularly uniform), and its harmonics
    Hp_ = np.asarray(H)
    hp = Hp_.sum(axis=1)
    a_k = (Hp_ * vec).sum(axis=1) / hp
    plane_mean_frac = float((hp * a_k ** 2).sum() / (Hp_ * vec ** 2).sum())
    ak_hat = np.fft.rfft(a_k)
    power = np.abs(ak_hat[1:]) ** 2
    dom = int(np.argmax(power) + 1) if power.size else 0
    return dict(regions=regions, top_rings=top_rings, eta_seam_planes_0_1_and_last2=dict(
        participation=float(w[seam_eta].sum()), volume=float(vol[seam_eta].sum())),
        theta_seam_j_0_1_and_last2=dict(participation=float(w[seam_theta].sum()), volume=float(vol[seam_theta].sum())),
        plane_participation_max=float(per_plane.max()), plane_participation_argmax=int(per_plane.argmax()),
        plane_participation_uniform=float(1.0 / E), eta_plane_mean_energy_fraction=plane_mean_frac,
        dominant_eta_harmonic_of_plane_means=dom, peak_nodes=pts, max_abs_node=[int(k0[0]), int(k0[1])])


def _finish(r: dict, vecs, plan, md, H) -> dict:
    """Common post-processing: noise floor, below-floor flags and the located negative directions."""
    st = plan.structure
    low = np.asarray(r["lowest"])
    rho = r["rho"]
    r["lowest_over_rho"] = (low / rho).tolist()
    r["noise_floor_eps_rho"] = EPS * rho
    r["below_noise_floor"] = [bool(x < -10 * EPS * rho) for x in low]
    if r.get("residuals") is not None:
        r["converged_residual_below_1e-9_rho"] = [bool(x <= 1e-9) for x in r["residuals"]]
    locs = []
    if vecs is not None:
        sq = np.sqrt(H.ravel())
        for c in range(vecs.shape[1]):
            if low[c] < 0:
                v = (vecs[:, c] / sq).reshape(st.n_eta, st.P)
                locs.append(dict(eigenvalue=float(low[c]), **locate(v, md.layout, H)))
    r["negative_directions"] = locs
    return r


def audit(n: int, arm: str, out_root: Path, k: int, maxiter: int, tol: float, mode: str, groups: int, factor_dtype: str,
          tag: str = "", estimate_only: bool = False, variant: str | None = None, faces_root: Path = C.FACES_ROOT) -> dict:
    t0 = time.perf_counter()
    md = C.load_metric(out_root, arm, n)
    plan = C.variant_plan(md, variant, faces_root)
    rdir = C.arm_dir(out_root, arm, n) if variant is None else C.variant_dir(faces_root, arm, n, variant)
    rdir.mkdir(parents=True, exist_ok=True)
    st = plan.structure
    t_plan = time.perf_counter() - t0
    log(f"{arm} N{n}: plan built {t_plan:.1f}s (E={st.n_eta}, P={st.P}, unknowns {st.n_eta * st.P}, tau={float(plan.tau):.1f}, "
        f"tau_w={float(plan.tau_w):.1f})")
    res = dict(n=n, arm=arm, unknowns=st.n_eta * st.P, plan_seconds=t_plan, coefficients=coefficient_report(plan),
               metric_identity=md.identity, mode=mode, variant=variant, evaluated=list(st.evaluated))
    log("coefficients: " + "; ".join(f"{key} [{res['coefficients'][key]['min']:.4g}, {res['coefficients'][key]['max']:.4g}] "
                                     f"neg {res['coefficients'][key]['negative']['count']}" for key in ("auu_f", "att_h", "aee_h")))
    H = np.asarray(plan.Hp) * st.deta
    if estimate_only:
        N_ = st.N
        nnz = st.n_eta * st.P * 12 * N_
        res["host_assembly_estimate"] = dict(nnz_per_row_model="12 N (387 measured at N32)", nnz=nnz, transient_gib=nnz * 71 / 2 ** 30)
        C.write_json(rdir / "audit_estimate.json", res)
        log(f"host full-matrix estimate: nnz {nnz:.2e}, transient {nnz * 71 / 2 ** 30:.1f} GiB")
        return res
    out = {}
    vec_store = {}
    sq_ = np.sqrt(H.ravel())
    if mode in ("host", "both"):
        from drbx.validation.sbp_laplacian_audit import LaplacianAssembly, audit_laplacian_plan

        asm = LaplacianAssembly(plan)
        Md = asm.matrix("dirichlet")
        res["matrix"] = dict(nnz=int(Md.nnz), nnz_per_row=float(Md.nnz / Md.shape[0]),
                             bytes_csr=int(Md.data.nbytes + Md.indices.nbytes + Md.indptr.nbytes), rss_gib=C.peak_rss_gib())
        log(f"assembled Dirichlet matrix: nnz {Md.nnz:.3e} ({Md.nnz / Md.shape[0]:.0f} per row), rss {C.peak_rss_gib():.2f} GiB")
        del Md, asm
        for label, it, tl in (("package_default", 80, 1e-7), ("refined", maxiter, tol)):
            ta = time.perf_counter()
            a = audit_laplacian_plan(plan, k=k, lobpcg_maxiter=it, lobpcg_tol=tl, return_vectors=True)
            log(f"audit_laplacian_plan[{label}] {time.perf_counter() - ta:.1f}s flags={a['flags']}")
            o = {"flags": a["flags"], "seconds": a["seconds"], "parameters": dict(k=k, lobpcg_maxiter=it, lobpcg_tol=tl)}
            for kind in ("dirichlet", "neumann"):
                r = dict(a[kind])
                vecs = r.pop("vectors")
                if label == "refined":
                    vec_store[f"host_{kind}"] = (vecs / sq_[:, None]).astype(np.float64)
                o[kind] = _finish(r, vecs, plan, md, H)
                log(f"  [{label}] {kind}: h_sym {r['h_symmetry']:.2e} rho {r['rho']:.4e} lowest {np.array2string(np.asarray(r['lowest']), precision=5)} "
                    f"(/rho {np.array2string(np.asarray(r['lowest']) / r['rho'], precision=2)}) negatives {r['n_negative']} "
                    f"null_defect {r['null_defect']} resid/rho {np.array2string(np.asarray(r['residuals']), precision=1)} it {r['iterations']} {r['seconds']:.1f}s")
            out[f"host_{label}"] = o
    if mode in ("mf", "both"):
        import tools as T
        from audit_mf import matrix_free_audit

        o = {"parameters": dict(k=k, maxiter=maxiter, tol_rel_rho=tol, block_size_dirichlet=k + 2, block_size_neumann=k)}
        res["preconditioner"] = {}
        rho_d = None
        for kind in ("dirichlet", "neumann"):
            tb = time.perf_counter()
            shift = 0.0 if kind == "dirichlet" else 1e-9 * rho_d * float(H.min())     # as the package's lobpcg preconditioner
            prec = T.build_merged_preconditioner(plan, groups, factor_dtype, kind=kind, shift=shift, log=log,
                                                 cond_max=1e12 if kind == "dirichlet" else 1e30)
            res["preconditioner"][kind] = dict(groups=groups, factor_dtype=factor_dtype, build_seconds=time.perf_counter() - tb,
                                               storage_gib=prec.nbytes / 2 ** 30, peak_rss_gib=C.peak_rss_gib(), shift=shift,
                                               max_block_cond1=prec.info["max_block_cond1"])
            log(f"{kind} preconditioner {time.perf_counter() - tb:.1f}s storage {prec.nbytes / 2 ** 30:.2f} GiB rss {C.peak_rss_gib():.2f}")
            # a block larger than the 4-fold (nfp = 4) Dirichlet cluster: LOBPCG with k = 4 stalls on a fourfold cluster
            r = matrix_free_audit(plan, kind, prec, k=k + 2 if kind == "dirichlet" else k, maxiter=maxiter, tol_rel_rho=tol)
            rho_d = r["rho"] if kind == "dirichlet" else rho_d
            vecs = r.pop("vectors")
            vec_store[f"mf_{kind}"] = (vecs / sq_[:, None]).astype(np.float64)
            o[kind] = _finish(r, vecs, plan, md, H)
            log(f"  [matrix-free] {kind}: h_sym_probe {r.get('h_symmetry_probe'):.2e} rho {r['rho']:.4e} lowest {np.array2string(np.asarray(r['lowest']), precision=5)} "
                f"(/rho {np.array2string(np.asarray(r['lowest']) / r['rho'], precision=2)}) negatives {r['n_negative']} null_defect {r['null_defect']} "
                f"resid/rho {np.array2string(np.asarray(r['residuals']), precision=1)} it {r['iterations']} {r['seconds']:.1f}s rss {C.peak_rss_gib():.2f}")
            del prec
        out["matrix_free"] = o
    res["audit"] = out
    if vec_store:
        np.savez_compressed(rdir / f"audit_vectors{tag}.npz", **vec_store)   # nodal values (n, k) of the lowest vectors
    res["seconds"] = time.perf_counter() - t0
    res["peak_rss_gib"] = C.peak_rss_gib()
    C.write_json(rdir / f"audit{tag}.json", res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--k", type=int, default=int(CONFIG["audit"]["k"]))
    ap.add_argument("--maxiter", type=int, default=600, help="refined LOBPCG iterations (the package default run uses 80)")
    ap.add_argument("--tol", type=float, default=1e-15, help="refined LOBPCG residual tolerance relative to rho")
    ap.add_argument("--mode", choices=("host", "mf", "both"), default="host")
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--factor-dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--variant", choices=tuple(C.VARIANTS), default=None)
    ap.add_argument("--faces-root", type=Path, default=C.FACES_ROOT)
    args = ap.parse_args(argv)
    groups = args.groups or {32: 1, 48: 8, 64: 16}[args.n]
    audit(args.n, args.arm, args.out, args.k, args.maxiter, args.tol, args.mode, groups, args.factor_dtype, args.tag,
          args.estimate_only, args.variant, args.faces_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
