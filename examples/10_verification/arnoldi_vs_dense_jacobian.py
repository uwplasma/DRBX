"""Verify Arnoldi eigenvalues against a dense Jacobian on a tiny linear problem.

This script builds a small cold-ion DRB linear operator in a slab geometry, forms the dense Jacobian
by applying the matrix-free J·v matvec to basis vectors, then compares the leading eigenvalues to
the results returned by `jaxdrb.linear.arnoldi`.

This is meant as a solver-workflow verification for reviewers:

- J·v via `jax.linearize`
- matrix-free Arnoldi (Ritz values)
- comparison to dense eigensolve for a small size
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import Equilibrium, State
from jaxdrb.models.params import DRBParams


def flatten_state(y: State) -> np.ndarray:
    parts = [
        np.asarray(y.n),
        np.asarray(y.omega),
        np.asarray(y.vpar_e),
        np.asarray(y.vpar_i),
        np.asarray(y.Te),
    ]
    return np.concatenate([p.reshape(-1) for p in parts]).astype(np.complex128)


def unflatten_state(v: np.ndarray, nl: int) -> State:
    n = v[0:nl]
    w = v[nl : 2 * nl]
    ve = v[2 * nl : 3 * nl]
    vi = v[3 * nl : 4 * nl]
    Te = v[4 * nl : 5 * nl]
    return State(n=n, omega=w, vpar_e=ve, vpar_i=vi, Te=Te)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_arnoldi_dense_verify")
    p.add_argument("--nl", type=int, default=9)
    p.add_argument("--kx", type=float, default=0.3)
    p.add_argument("--ky", type=float, default=0.7)
    p.add_argument("--m", type=int, default=60, help="Arnoldi dimension.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nl = int(args.nl)
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)
    params = DRBParams(
        curvature_on=False,
        boussinesq=True,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        omega_n=0.8,
        omega_Te=0.0,
        eta=0.3,
        me_hat=0.0,
    )
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y0 = State.zeros(nl)

    mv = linear_matvec(y0, params, geom, kx=float(args.kx), ky=float(args.ky), eq=eq)
    n = 5 * nl
    J = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        e = np.zeros((n,), dtype=np.complex128)
        e[i] = 1.0 + 0.0j
        col = mv(unflatten_state(e, nl))
        J[:, i] = flatten_state(col)

    evals_dense = np.linalg.eigvals(J)
    idx = np.argsort(evals_dense.real)[::-1]
    evals_dense = evals_dense[idx]

    res = arnoldi_eigs(mv, y0, m=min(int(args.m), n), nev=12, seed=0)
    evals_arn = res.eigenvalues

    print(f"[arnoldi-dense] leading dense eigenvalue:  {evals_dense[0]}")
    print(
        f"[arnoldi-dense] leading arnoldi eigenvalue: {evals_arn[0]} (resid {res.residual_norms[0]:.2e})"
    )
    print(
        f"[arnoldi-dense] |Δ|/(|λ|+1) = {abs(evals_arn[0] - evals_dense[0]) / (abs(evals_dense[0]) + 1):.2e}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.0))
    ax.scatter(evals_dense.real, evals_dense.imag, s=18, alpha=0.6, label="dense")
    ax.scatter(evals_arn.real, evals_arn.imag, s=60, marker="x", label="arnoldi ritz")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Arnoldi vs dense Jacobian eigenvalues (tiny problem)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "eigs.npz",
        evals_dense=evals_dense,
        evals_arn=evals_arn,
        residual_norms=res.residual_norms,
    )
    print(f"[arnoldi-dense] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
