from __future__ import annotations

import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import Equilibrium, State
from jaxdrb.models.params import DRBParams


def _flatten_state(y: State) -> np.ndarray:
    parts = [
        np.asarray(y.n),
        np.asarray(y.omega),
        np.asarray(y.vpar_e),
        np.asarray(y.vpar_i),
        np.asarray(y.Te),
    ]
    return np.concatenate([p.reshape(-1) for p in parts]).astype(np.complex128)


def _unflatten_state(v: np.ndarray, nl: int, dtype=np.complex128) -> State:
    n = v[0:nl].astype(dtype)
    w = v[nl : 2 * nl].astype(dtype)
    ve = v[2 * nl : 3 * nl].astype(dtype)
    vi = v[3 * nl : 4 * nl].astype(dtype)
    Te = v[4 * nl : 5 * nl].astype(dtype)
    return State(n=n, omega=w, vpar_e=ve, vpar_i=vi, Te=Te)


def test_arnoldi_matches_dense_eigs_on_tiny_problem() -> None:
    """Arnoldi leading eigenvalues should match dense eigenvalues on a tiny system.

    This is a solver-workflow verification: matrix-free J·v (jax.linearize) + Arnoldi vs
    an explicitly formed Jacobian. Keep sizes small for CI.
    """

    nl = 9
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
    kx = 0.3
    ky = 0.7

    mv = linear_matvec(y0, params, geom, kx=kx, ky=ky, eq=eq)

    # Build dense Jacobian by applying mv to basis vectors.
    n = 5 * nl
    J = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        e = np.zeros((n,), dtype=np.complex128)
        e[i] = 1.0 + 0.0j
        col = mv(_unflatten_state(e, nl))
        J[:, i] = _flatten_state(col)

    evals_dense = np.linalg.eigvals(J)
    evals_dense = evals_dense[np.argsort(evals_dense.real)[::-1]]

    res = arnoldi_eigs(mv, y0, m=min(60, n), nev=6, seed=0)
    evals_arn = res.eigenvalues

    # Compare the leading eigenvalues by real part.
    # For a tiny problem Arnoldi residuals should be small.
    assert float(np.max(res.residual_norms)) < 1e-6
    assert abs(evals_arn[0] - evals_dense[0]) / (abs(evals_dense[0]) + 1.0) < 1e-6
