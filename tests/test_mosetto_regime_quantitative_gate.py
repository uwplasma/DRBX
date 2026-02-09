from __future__ import annotations

import numpy as np

from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.models.params import DRBParams


def _gamma_max(scan) -> float:
    return float(np.max(scan.gamma_eigs))


def _regime_label(*, eta: float, omega_n: float, geom, ky: np.ndarray) -> str:
    base = DRBParams(
        omega_n=float(omega_n),
        omega_Te=0.0,
        eta=float(eta),
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    kwargs = dict(
        ky=ky,
        kx=0.0,
        arnoldi_m=10,
        arnoldi_tol=8e-3,
        arnoldi_max_m=120,
        nev=2,
        do_initial_value=False,
        verbose=False,
        seed=0,
    )
    g_full = _gamma_max(scan_ky(base, geom, **kwargs))
    g_noc = _gamma_max(
        scan_ky(DRBParams(**{**base.__dict__, "curvature_on": False}), geom, **kwargs)
    )
    g_noi = _gamma_max(scan_ky(DRBParams(**{**base.__dict__, "me_hat": 0.0}), geom, **kwargs))

    if g_full <= 1e-12:
        return "stable"
    dw_like = g_noc >= 0.5 * g_full
    resistive_like = g_noi >= 0.5 * g_full
    if dw_like and (not resistive_like):
        return "InDW"
    if dw_like and resistive_like:
        return "RDW"
    if (not dw_like) and (not resistive_like):
        return "InBM"
    return "RBM"


def test_mosetto_like_regime_transition_low_to_high_collisionality() -> None:
    """Quantitative gate for the dominant low-eta/high-eta branch transition.

    For the workflow settings used in `mosetto2012_regime_map.py`, low collisionality points are
    typically classified as RBM-like while high collisionality points are RDW-like.
    """
    geom = CircularTokamakGeometry.make(
        nl=24, shat=0.8, q=3.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )
    ky = np.linspace(0.08, 0.9, 6)

    # Low-collisionality branch is ballooning/resistive-like in this reduced model setup.
    assert _regime_label(eta=0.02, omega_n=0.2, geom=geom, ky=ky) == "RBM"
    assert _regime_label(eta=0.02, omega_n=2.0, geom=geom, ky=ky) == "RBM"

    # High-collisionality branch transitions to drift-wave/resistive-like.
    assert _regime_label(eta=2.0, omega_n=0.2, geom=geom, ky=ky) == "RDW"
    assert _regime_label(eta=2.0, omega_n=2.0, geom=geom, ky=ky) == "RDW"
