"""Production and control brackets on the nodes: ``<out>/N{n}/static/<set>.npz`` (arrays ``N``, ``N0``).

For each catalogue and each potential-like field ``a`` the velocity flux (with level and core D5c) is built once and the
bracket is applied to the stack of every advected field ``b`` of the pairs with that ``a``, with the manufactured wall
trace of ``b`` as inflow data (all field kinds, Dirichlet- and Neumann-structured alike):

- ``N``  : the production operator ``sbp_bracket`` (``L g`` plus dissipation, ``c_kappa`` as given);
- ``N0`` : the control ``linear_operator`` (no dissipation).

    python static.py N --out ROOT [--c-kappa 1.0] [--sets ...]
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
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

import fields as F                                                                      # noqa: E402
from references import RHO, available_sets, load_layout_metric, log, peak_rss_gib                       # noqa: E402


def run(n: int, out_root: Path, c_kappa: float = 1.0, sets=None) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
    from drbx.native.fci_perpendicular_sbp_bracket import linear_operator, sbp_bracket, velocity_flux
    from drbx.stencils.nodal_plan import build_nodal_plan

    t0 = time.perf_counter()
    sets = tuple(sets or available_sets(out_root, n))
    layout, metric, meta = load_layout_metric(n, out_root)
    plan = build_nodal_plan(layout, metric)
    sbp = jax.jit(lambda p, phi, g, bcd, rho, ck: sbp_bracket(p, phi, g, bcd, rho, ck))
    lin = jax.jit(lambda p, Fl, g, bcd: linear_operator(p, Fl, g, bcd))
    flux = jax.jit(lambda p, phi, rho: velocity_flux(p, phi, rho))
    receipts = []
    for name in sets:
        ts = time.perf_counter()
        z = np.load(out_root / f"N{n}" / "references" / f"{name}.npz")
        names = [str(x) for x in z["names"]]
        pairs = [F.Pair(**d) for d in json.loads(str(z["pairs"]))]
        vals, wall = z["vals"], z["wall"]
        idx = {nm: i for i, nm in enumerate(names)}
        N = np.zeros((len(pairs),) + vals.shape[1:])                 # all planes (the eta stencils need them)
        N0 = np.zeros_like(N)
        for a in dict.fromkeys(p.a for p in pairs):
            members = [k for k, p in enumerate(pairs) if p.a == a]
            g = np.stack([vals[idx[pairs[k].b]] for k in members], axis=-1)                  # (E, P, Fb)
            wv = np.stack([wall[idx[pairs[k].b]] for k in members], axis=-1)                 # (E, N_w, Fb)
            bcd = SatBoundaryData((jnp.asarray(wv),), None)
            phi = vals[idx[a]]
            Fl = flux(plan, phi, RHO)
            out = np.asarray(sbp(plan, phi, g, bcd, RHO, c_kappa))
            out0 = np.asarray(lin(plan, Fl, g, bcd))
            for j, k in enumerate(members):
                N[k], N0[k] = out[..., j], out0[..., j]
        path = out_root / f"N{n}" / "static"
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / f"{name}.npz", N=N, N0=N0, c_kappa=np.array(c_kappa))
        receipts.append(dict(set=name, seconds=time.perf_counter() - ts, pairs=len(pairs)))
        log(f"N{n} static {name}: {receipts[-1]['seconds']:.1f} s")
    out = dict(n=n, c_kappa=c_kappa, seconds=time.perf_counter() - t0, peak_rss_gib=peak_rss_gib(), sets=receipts)
    (out_root / f"N{n}" / "static_receipt.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c-kappa", type=float, default=float(F.CONFIG["c_kappa"]))
    ap.add_argument("--sets", default="")
    args = ap.parse_args(argv)
    run(args.n, args.out, args.c_kappa, args.sets.split(",") if args.sets else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
