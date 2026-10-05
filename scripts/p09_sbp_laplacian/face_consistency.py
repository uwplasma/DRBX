#!/usr/bin/env python3
"""Consistency of the evaluated ``A^{eta eta}`` faces with the nodal cross terms, and the toroidal-only Rayleigh quotients.

The energy form pairs the collocated cross terms ``2 D_u f A_{u eta} D_eta f`` (nodal ``A``) with the staggered diagonal
``(G_eta f)^2 a_ee,h`` (face values). The pointwise tensor ``A = |J| P_perp g^-1`` is positive semidefinite and, where the
field line is nearly in the ``(u, eta)`` or ``(theta, eta)`` plane, nearly singular in that plane: ``A_{u eta}^2 ~ A_uu A_eta eta``. The
form stays semidefinite only while the diagonal paired with the cross terms is not smaller than what they imply, so replacing the
interpolated ``a_ee,h`` by values that are smaller than the nodal ``A_eta eta`` at neighbouring planes can create negative directions
that the interpolation (which only smooths the nodal values) does not.

Reported (``<faces-root>/<arm>/N<N>/face_consistency.json``):

* quantiles of ``a_ee,h(min of the two half planes around node plane k) / A_eta eta(node)`` and of the discriminants
  ``A_{u eta}^2 / (A_uu a_ee,face_min)``, ``A_{theta eta}^2 / (A_theta theta a_ee,face_min)`` on the ring nodes (``> 1``: the
  nodal cross term exceeds what the face diagonal supports);
* the Rayleigh quotients ``f^T M f / f^T H f`` of the toroidal-only modes ``cos(m eta), sin(m eta)`` (Neumann, zero data) for each variant.

    python face_consistency.py N --arm raw [--out ROOT] [--faces-root ROOT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import log                                                                  # noqa: E402

import numpy as np                                                                      # noqa: E402

QS = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]


def run(n: int, arm: str, out_root: Path, faces_root: Path, mmax: int = 8) -> dict:
    import jax
    import jax.numpy as jnp
    from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form

    md = C.load_metric(out_root, arm, n)
    fd = C.load_faces(faces_root, arm, n, md)
    E, P = md.layout.n_eta, md.layout.P
    Nc = md.layout.blocks[0].n_nodes
    A = md.A
    aee_n = A[..., 2, 2]
    h = fd.A_ee_h
    fmin = np.minimum(h, np.roll(h, 1, axis=0))                    # half planes k - 1/2 and k + 1/2 of node plane k
    ratio = fmin / np.maximum(aee_n, 1e-300)
    du = A[..., 0, 2] ** 2 / np.maximum(A[..., 0, 0] * fmin, 1e-300)
    dt = A[..., 1, 2] ** 2 / np.maximum(A[..., 1, 1] * fmin, 1e-300)
    res = dict(n=n, arm=arm, metric_identity=md.identity, faces_identity=fd.identity, quantiles=QS,
               face_min_over_node_aee=np.quantile(ratio, QS).tolist(),
               disc_u_eta_ring=np.quantile(du[:, Nc:], QS).tolist(), disc_theta_eta_ring=np.quantile(dt[:, Nc:], QS).tolist(),
               fraction_disc_u_eta_above_1=float((du[:, Nc:] > 1).mean()), fraction_disc_theta_eta_above_1=float((dt[:, Nc:] > 1).mean()),
               nodal_disc_u_eta_max=float((A[..., 0, 2] ** 2 / np.maximum(A[..., 0, 0] * aee_n, 1e-300)).max()), toroidal_rayleigh={})
    eta = (np.arange(E) + 0.5) * md.layout.deta
    form = jax.jit(lambda lp, v: laplacian_form(lp, v, None, "neumann", None, 1.0, neumann_mode="conormal"))
    for v in C.VARIANTS:
        plan = C.build_plan(md, faces=fd.face_metric(v))
        H = np.asarray(plan.Hp) * plan.structure.deta
        rq = {}
        for m in range(1, mmax + 1):
            for nm, fn in (("cos", np.cos), ("sin", np.sin)):
                f = np.broadcast_to(fn(m * eta)[:, None], (E, P)).copy()
                Mf = np.asarray(form(plan, jnp.asarray(f)))
                rq[f"{nm}{m}"] = float((f * Mf).sum() / (H * f * f).sum())
        res["toroidal_rayleigh"][v] = rq
        log(f"{arm} N{n} {v}: toroidal Rayleigh " + " ".join(f"{k}={x:.2e}" for k, x in list(rq.items())[:6]))
    log(f"face_min/node a_ee quantiles {np.array2string(np.asarray(res['face_min_over_node_aee']), precision=3)}; "
        f"disc_u fraction>1 {res['fraction_disc_u_eta_above_1']:.2f}, disc_theta {res['fraction_disc_theta_eta_above_1']:.2f}")
    C.write_json(C.faces_dir(faces_root, arm, n) / "face_consistency.json", res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--faces-root", type=Path, default=C.FACES_ROOT)
    args = ap.parse_args(argv)
    run(args.n, args.arm, args.out, args.faces_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
