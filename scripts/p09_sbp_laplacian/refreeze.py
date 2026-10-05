#!/usr/bin/env python3
"""P08 re-freeze on the nodes, solved-phi arm: the exact nodal Laplacian references per P06N field set.

P08 froze, per P06N field set, owner-projected references of the perpendicular diffusion of ``n, Te, Ti, omega`` and of the
polarization of ``psi = phi + tau n Ti``. On the nodal layout the reference is the exact nodal Laplacian
``R_f = div(P_perp grad f)`` per unit volume (``fields.exact_laplacian``; no owner projection, so ``N - O`` is ``N - R``) with the
extracted logical metric and the exact logical derivatives of the P06N catalogue fields, for all seven P06N cases and the slots
``n, Te, Ti, omega, phi, q = n Ti, psi = phi + tau q``. Production: ``laplacian_action`` with the manufactured wall data of each
slot (Dirichlet value or Neumann conormal flux, per the catalogue kind of the slot).

Solved-phi arm (the four Dirichlet-phi cases): the potential is *solved* from ``L phi = tau L_h q - omega_c`` by the package CG,
with ``omega_c = tau R_q - R_phi`` so the catalogue phi is the exact solution; the error of the solved phi against it is stored.

Output ``<out>/<arm>/N<n>/nodal_laplacian_reference.npz`` and ``nodal_laplacian_reference_manifest.json`` (schema
``drbx.p09-nodal-laplacian-reference-v1``).

    python refreeze.py N --arm raw|filtered [--out ROOT] [--groups G] [--factor-dtype float32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG, log                                                          # noqa: E402
import fields as Fl                                                                     # noqa: E402
import tools as T                                                                       # noqa: E402

import numpy as np                                                                      # noqa: E402

SCHEMA = CONFIG["refreeze"]["schema"]
SOLVED_VARIANTS = tuple(CONFIG["refreeze"]["variants"])
DEFAULT_GROUPS = {32: 1, 48: 8, 64: 16}
SLOT_NAMES = Fl.SLOTS + Fl.DERIVED


def run(n: int, arm: str, out_root: Path, groups: int, factor_dtype: str) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol
    from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action_jit
    from drbx.native.fci_perpendicular_sbp_norms import ring_region_masks

    t0 = time.perf_counter()
    cat_json = json.loads(Fl.CATALOGUE_PATH.read_text())
    variants = list(cat_json["cases"])
    md = C.load_metric(out_root, arm, n)
    plan = C.build_plan(md)
    st = plan.structure
    E, P, Nw = st.n_eta, st.P, st.N
    H = np.asarray(plan.Hp) * st.deta
    core = ring_region_masks(md.layout, n)["core"]
    nv, ns = len(variants), len(SLOT_NAMES)
    R = np.zeros((nv, ns, E, P)); Vv = np.zeros_like(R)
    wall_val = np.zeros((nv, ns, E, Nw)); wall_con = np.zeros_like(wall_val)
    kinds_arr = np.zeros((nv, ns), dtype="U9")
    summary, spec_out = {}, {}
    for vi, variant in enumerate(variants):
        cat, spec = Fl.p06n_variant(variant)
        data = Fl.NodalData(md, cat)
        R[vi], Vv[vi] = data.R, data.V
        wall_val[vi] = np.moveaxis(data.wall_value, -1, 0)
        wall_con[vi] = np.moveaxis(data.wall_conormal, -1, 0)
        kinds_arr[vi] = cat.kinds
        spec_out[variant] = spec
        bcd = LaplacianBoundaryData(value=(jnp.asarray(data.wall_value),), conormal=(jnp.asarray(data.wall_conormal),))
        Nv = np.asarray(laplacian_action_jit(plan, jnp.asarray(np.moveaxis(data.V, 0, -1)), bcd, tuple(cat.kinds), None, 1.0))
        Nn_v = np.moveaxis(Nv, -1, 0)
        entry = {"spec": spec, "kinds": dict(zip(SLOT_NAMES, cat.kinds))}
        for f, name in enumerate(SLOT_NAMES):
            e = Nn_v[f] - R[vi, f]
            gref = float(np.sqrt(np.sum(H * R[vi, f] ** 2)))
            entry[name] = {"E_rel_global": float(np.sqrt(np.sum(H * e * e)) / gref) if gref > 0 else None,
                           "E_rms": float(np.sqrt(np.sum(H * e * e) / H.sum())),
                           "E_rms_core": float(np.sqrt(np.sum(H[core] * e[core] ** 2) / H[core].sum())),
                           "max_abs_R": float(np.abs(R[vi, f]).max()), "max_abs_N": float(np.abs(Nn_v[f]).max())}
        summary[variant] = entry
        log(f"{arm} N{n} re-freeze {variant}: " + " ".join(
            f"{s}={entry[s]['E_rel_global']:.2e}" if entry[s]["E_rel_global"] is not None else f"{s}=|N|{entry[s]['max_abs_N']:.0e}"
            for s in SLOT_NAMES))
    # --- solved-phi arm
    tb = time.perf_counter()
    prec = T.build_merged_preconditioner(plan, groups, factor_dtype, log=log)
    t_prec = time.perf_counter() - tb
    solved = {}
    tau = Fl.TAU
    zero = jnp.zeros((E, P))
    om_c = {}
    for variant in SOLVED_VARIANTS:
        vi = variants.index(variant)
        iq, ip = SLOT_NAMES.index("q"), SLOT_NAMES.index("phi")
        kinds = summary[variant]["kinds"]
        if kinds["phi"] != "dirichlet" or kinds["q"] != "dirichlet":
            continue
        omega_c = tau * R[vi, iq] - R[vi, ip]                      # exact omega consistent with the catalogue phi
        om_c[variant] = omega_c
        gq = jnp.asarray(wall_val[vi, iq]); gp = jnp.asarray(wall_val[vi, ip])
        Lq = laplacian_action_jit(plan, jnp.asarray(Vv[vi, iq]), LaplacianBoundaryData(value=(gq,)), "dirichlet", None, 1.0)
        s = tau * Lq - jnp.asarray(omega_c)
        ts = time.perf_counter()
        x, info = sol.solve_dirichlet_jit(plan, s, LaplacianBoundaryData(value=(gp,)), prec, rtol=1e-10, maxit=200)
        x = np.asarray(x)
        err = x - Vv[vi, ip]
        solved[variant] = dict(iterations=int(info["iterations"]), relative_residual=float(info["relative_residual"]),
                               converged=bool(info["converged"]), seconds=time.perf_counter() - ts,
                               error_rel_global=float(np.sqrt(np.sum(H * err * err) / np.sum(H * Vv[vi, ip] ** 2))),
                               error_max=float(np.abs(err).max()),
                               source_error_rel=(float(np.sqrt(np.sum(H * (np.asarray(s) - (tau * R[vi, iq] - omega_c)) ** 2)
                                                               / np.sum(H * R[vi, ip] ** 2)))
                                                 if np.sum(H * R[vi, ip] ** 2) > 0 else None))
        log(f"  solved phi {variant}: it={solved[variant]['iterations']} err={solved[variant]['error_rel_global']:.3e} "
            f"source_err={solved[variant]['source_error_rel']}")
    path = C.arm_dir(out_root, arm, n)
    npz = path / "nodal_laplacian_reference.npz"
    omega_arr = np.stack([om_c[v] for v in SOLVED_VARIANTS if v in om_c]) if om_c else np.zeros((0, E, P))
    np.savez_compressed(npz, R=R, values=Vv, wall_value=wall_val, wall_conormal=wall_con, kinds=kinds_arr,
                        variants=np.array(variants), slots=np.array(SLOT_NAMES), omega_consistent=omega_arr,
                        omega_consistent_variants=np.array([v for v in SOLVED_VARIANTS if v in om_c]))
    manifest = {
        "schema": SCHEMA, "n": n, "arm": arm, "family": "A", "arm_role": "solved_phi (all slots, production vs exact nodal Laplacian)",
        "term": "perpendicular Laplacian div(P_perp grad f) per unit volume (diffusion of n, Te, Ti, omega; polarization of psi)",
        "tau": tau, "orientation": "R_f = (divA_j d_j f + A_ij d_i d_j f) / |J| at the nodes, A = |J| P_perp (logical frame)",
        "owner_projection": "none (nodal reference: N - O = N - R)",
        "slots": list(SLOT_NAMES), "variants": spec_out, "R_shape": list(R.shape),
        "wall_data": "Dirichlet: manufactured value; Neumann: exact conormal flux (A grad f)^u at the wall points (slot kind from the catalogue)",
        "layout": {"P": P, "n_eta": E, "core_nodes": int(md.layout.blocks[0].n_nodes)},
        "identities": {"layout_sha256": C.layout_identity(md.layout), "plan_sha256": C.plan_identity(plan), "metric_identity": md.identity,
                       "metric_file_sha256": C.sha256_file(path / C.METRIC_FILE), "eta_filter_arm_sha256":
                       (md.meta.get("eta_filter") or {}).get("arm_sha256"), "sidecar_sha256": md.meta.get("sidecar_sha256"),
                       "catalogue_file": str(Fl.CATALOGUE_PATH.relative_to(Fl.SCRIPTS.parent)),
                       "catalogue_sha256": C.sha256_file(Fl.CATALOGUE_PATH), "arrays_sha256": C.sha256_file(npz)},
        "production": {"operator": "laplacian_action", "tau_mult": 1.0, "errors_N_minus_R": summary},
        "solved_phi": {"preconditioner": dict(groups=groups, factor_dtype=factor_dtype, build_seconds=t_prec),
                       "rhs": "tau L_h q - omega_c, omega_c = tau R_q - R_phi (catalogue phi is the exact solution)", "cases": solved},
        "seconds": time.perf_counter() - t0, "peak_rss_gib": C.peak_rss_gib(),
    }
    C.write_json(path / "nodal_laplacian_reference_manifest.json", manifest)
    log(f"{arm} N{n} re-freeze written: {npz.name} ({npz.stat().st_size / 1e6:.1f} MB)")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--factor-dtype", default="float64", choices=("float64", "float32"))
    args = ap.parse_args(argv)
    run(args.n, args.arm, args.out, args.groups or DEFAULT_GROUPS[args.n], args.factor_dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
