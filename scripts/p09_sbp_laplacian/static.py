#!/usr/bin/env python3
"""Static accuracy of the nodal Laplacian: ``N - R`` per field, set and region at one arm and resolution.

``N = laplacian_action(plan, f_nodes, wall data)`` with the manufactured wall data of the exact field; ``R`` the exact
continuum ``div(P_perp grad f)`` per unit volume at the nodes with the same extracted metric (``fields.exact_laplacian``).

Sets (all with the unit coefficient):

* ``transverse`` (gate), ``transverse_wave`` (control): ``n, Te, Ti, omega, phi, q = n Ti, psi = phi + tau q``, Dirichlet data;
  also Neumann with the exact conormal flux (report);
* ``p07n``: ``field_b1, field_e3, field_e12, heldout_field_b2, constant`` with Dirichlet data, Neumann conormal data
  ``(A grad f)^u`` and Neumann physical-normal data ``n . grad f`` (converted in the operator).

Output ``<out>/<arm>/N<N>/static.json`` (and ``static/<set>_<mode>.npz`` with ``N`` and ``R``).

    python static.py N --arm raw|filtered [--out ROOT] [--variant interp|ee|ee_tt|all] [--faces-root ROOT]

``--variant`` builds the plan with the listed face families taken from the tensor evaluated at the faces (``extract_metric.py
--faces``) and writes ``<faces-root>/<arm>/N<N>/<variant>/static.json`` (and ``static/``) instead of the M5b location.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import log                                                                  # noqa: E402
import fields as Fl                                                                     # noqa: E402

import numpy as np                                                                      # noqa: E402

MODES = {"dirichlet": ("dirichlet", "value", "conormal"), "neumann_conormal": ("neumann", "conormal", "conormal"),
         "neumann_normal": ("neumann", "normal_derivative", "physical")}
SETS = {"transverse": ("dirichlet", "neumann_conormal"), "transverse_wave": ("dirichlet",),
        "p07n": ("dirichlet", "neumann_conormal", "neumann_normal")}


def table(err, ref, H, masks):
    from drbx.native.fci_perpendicular_sbp_norms import region_errors

    gref = float(np.sqrt(np.sum(H * ref ** 2)))
    e2 = float(np.sum(H * err ** 2))
    row = dict(rel_global=float(np.sqrt(e2) / gref) if gref > 0 else None, rms=float(np.sqrt(e2 / np.sum(H))),
               max=float(np.abs(err).max()), ref_rms=float(gref / np.sqrt(np.sum(H))), ref_max=float(np.abs(ref).max()))
    row["regions"] = region_errors(err, ref, H, masks)
    return row


def run(n: int, arm: str, out_root: Path, variant: str | None = None, faces_root: Path = C.FACES_ROOT) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native.fci_perpendicular_sbp_laplacian import LaplacianBoundaryData, laplacian_action_jit
    from drbx.native.fci_perpendicular_sbp_norms import ring_region_masks

    t0 = time.perf_counter()
    md = C.load_metric(out_root, arm, n)
    plan = C.variant_plan(md, variant, faces_root)
    st = plan.structure
    H = np.asarray(plan.Hp) * st.deta
    masks = ring_region_masks(md.layout, n)
    res = dict(n=n, arm=arm, metric_identity=md.identity, tau=float(plan.tau), tau_w=float(plan.tau_w), sets={},
               variant=variant, evaluated=list(st.evaluated))
    rdir = C.arm_dir(out_root, arm, n) if variant is None else C.variant_dir(faces_root, arm, n, variant)
    outdir = rdir / "static"
    outdir.mkdir(parents=True, exist_ok=True)
    for sname, modes in SETS.items():
        ts = time.perf_counter()
        cat = Fl.make_catalogue(sname)
        data = Fl.NodalData(md, cat)
        log(f"{arm} N{n} {sname}: exact data {time.perf_counter() - ts:.1f}s (fields {len(cat.names)})")
        V = jnp.asarray(np.moveaxis(data.V, 0, -1))                    # (E, P, F)
        R = data.R
        sres = {}
        for mode in modes:
            kind, key, nmode = MODES[mode]
            wall = {"dirichlet": data.wall_value, "neumann_conormal": data.wall_conormal, "neumann_normal": data.wall_normal}[mode]
            bcd = LaplacianBoundaryData(**{key: (jnp.asarray(wall),)})
            tm = time.perf_counter()
            Nn = np.moveaxis(np.asarray(laplacian_action_jit(plan, V, bcd, kind, None, 1.0, neumann_mode=nmode)), -1, 0)
            sec = time.perf_counter() - tm
            mres = {}
            for f, name in enumerate(cat.names):
                e = Nn[f] - R[f]
                row = table(e, R[f], H, masks)
                row["max_abs_N"] = float(np.abs(Nn[f]).max())
                mres[name] = row
            sres[mode] = dict(seconds=sec, fields=mres)
            np.savez_compressed(outdir / f"{sname}_{mode}.npz", N=Nn, R=R, names=np.array(cat.names))
            log(f"  {sname}/{mode} ({sec:.1f}s): " + " ".join(f"{k}={v['rel_global']:.3e}" if v["rel_global"] is not None
                                                           else f"{k}=|N|{v['max_abs_N']:.1e}" for k, v in mres.items()))
        res["sets"][sname] = sres
    res["seconds"] = time.perf_counter() - t0
    res["peak_rss_gib"] = C.peak_rss_gib()
    C.write_json(rdir / "static.json", res)
    log(f"{arm} N{n}: static done {res['seconds']:.1f}s")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--variant", choices=tuple(C.VARIANTS), default=None)
    ap.add_argument("--faces-root", type=Path, default=C.FACES_ROOT)
    args = ap.parse_args(argv)
    run(args.n, args.arm, args.out, args.variant, args.faces_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
