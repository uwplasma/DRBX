"""Synthetic family-A bundle for the P10 harness tests (no HSX files): the analytic shaped-disk testbed
(``tests/sbp_laplacian_testbed.py``: non-orthogonal, anisotropic, eta dependent) for ``A = |J| P_perp``, ``|J|``, ``g^{u j}``
and the smooth ``(h, B, K)`` of ``tests/test_fci_nodal_perpendicular_rhs.py`` for the bracket / curvature, with ``d_i A^{ij}``
by autodiff of the analytic tensor. The nodal and the Laplacian plans share one jacobian, so ``Hp`` agrees bitwise.

    synthetic_inputs(n, n_eta, arm)  -> the arguments of ``bundle.assemble_bundle``
    synthetic_bundle(n, n_eta, arm)  -> the cached :class:`~bundle.Bundle` (default ``n = 16, n_eta = 8``)
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
for _p in (str(SCRIPTS), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from drbx.geometry.nodal_families import build_family_a_layout                                  # noqa: E402
from drbx.geometry.nodal_layout import node_points, wall_points                                 # noqa: E402
from drbx.stencils.nodal_plan import NodalMetric                                                # noqa: E402
from p10_evolved_mms import bundle as bundle_mod                                                # noqa: E402
from tests import sbp_laplacian_testbed as tb                                                   # noqa: E402

#: the ``eta_filter`` option recorded for the synthetic "filtered" arm (identity tests only; the geometry is the same)
SYNTHETIC_ARM_FILTER = {"quantity": "synthetic", "max_harmonic_per_period": 3, "arm_sha256": "0" * 64}


def _div_tensor(q):
    """``d_i A^{ij}`` of the analytic tensor at the logical point ``q``."""
    dA = jax.jacfwd(lambda x: tb.tensor(x)[0])(q)                       # dA[i, j, k] = d_k A_ij
    return jnp.einsum("iji->j", dA)


def _full_geometry(q):
    A, J, G = tb.tensor(q)
    return A, _div_tensor(q), J, G[0]


def nodal_h_B_K(pts):
    """Smooth ``(h, |J|, B, K)`` of the nodal metric at logical points ``(Q, 3)`` (the jacobian is the testbed's)."""
    _A, J, _G = tb._batched(tb._geometry_fn, pts)
    u, th, et = np.asarray(pts).T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    K = np.stack([0.3 * np.cos(th), 0.1 * u + 0.05, 0.2 * np.sin(et) + 0.1], -1)
    return h, np.abs(J), 1.0 + 0.2 * u, K


def _sha(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=4)
def synthetic_inputs(n: int = 16, n_eta: int = 8, arm: str = "raw"):
    """``(arm, layout, nodal, nodal_meta, lap, lap_meta, provenance)`` for :func:`bundle.assemble_bundle` (cached; do not mutate)."""
    layout = build_family_a_layout(n, n_eta=n_eta)
    pts = node_points(layout)
    wpts = wall_points(layout, layout.walls[0])
    E, P, Nw = layout.n_eta, layout.P, wpts.shape[1]
    A, divA, J, gu = (np.asarray(x) for x in tb._batched(_full_geometry, pts.reshape(-1, 3)))
    Aw, _dw, Jw, gw = (np.asarray(x) for x in tb._batched(_full_geometry, wpts.reshape(-1, 3)))
    if not (J.min() > 0 and Jw.min() > 0):
        raise ValueError("the synthetic jacobian must be positive")
    h, jac, B, K = nodal_h_B_K(pts.reshape(-1, 3))
    nodal = NodalMetric(h.reshape(E, P, 3), jac.reshape(E, P), B.reshape(E, P), K.reshape(E, P, 3))
    lap = dict(points=pts, A=A.reshape(E, P, 3, 3), divA=divA.reshape(E, P, 3), J=J.reshape(E, P), ginv_u=gu.reshape(E, P, 3),
               wall_points=wpts, wall_A=Aw.reshape(E, Nw, 3, 3), wall_ginv_u=gw.reshape(E, Nw, 3), wall_J=Jw.reshape(E, Nw),
               sign=1.0)
    eta_filter = None if arm == "raw" else dict(SYNTHETIC_ARM_FILTER)
    nodal_meta = dict(N=n, P=P, n_eta=E, family="A", synthetic=True)
    lap_meta = dict(n=n, arm=arm, P=P, n_eta=E, family="A", synthetic=True)
    if eta_filter is not None:
        nodal_meta["eta_filter"] = eta_filter
    lap_meta["eta_filter"] = eta_filter
    prov = dict(n=n, synthetic=True,
                nodal_metric_identity=_sha(dict(h=nodal.h, jac=nodal.jac, B=nodal.B, K=nodal.K)),
                laplacian_metric_identity=_sha(dict(A=lap["A"], divA=lap["divA"], J=lap["J"], ginv_u=lap["ginv_u"])))
    return arm, layout, nodal, nodal_meta, lap, lap_meta, prov


@lru_cache(maxsize=4)
def synthetic_bundle(n: int = 16, n_eta: int = 8, arm: str = "raw", build_preconditioner: bool = True):
    arm, layout, nodal, nodal_meta, lap, lap_meta, prov = synthetic_inputs(n, n_eta, arm)
    return bundle_mod.assemble_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, prov,
                                      build_preconditioner=build_preconditioner)
