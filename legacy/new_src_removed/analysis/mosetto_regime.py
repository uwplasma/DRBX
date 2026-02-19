from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MosettoCalibration:
    """Calibration constants for the 4-regime Mosetto-style map.

    The thresholds follow the workflow described around Sec. V of Mosetto et al. (2012),
    using:
      - RDW/RBM threshold at ŝ=0 from their analytic estimate,
      - InDW/InBM threshold at ŝ=0 from their analytic estimate,
      - RDW/InDW transition d-threshold fit anchored to reported points:
          d_crit(ŝ=0)≈3.55, d_crit(ŝ=5)≈1.12.

    Since `jaxdrb` uses reduced normalized parameters, we map code knobs to the paper proxies
    through the simple scaling factors below.
    """

    gamma_ratio: float = 1.0
    rln_per_omega_n: float = 40.0
    d_scale: float = 1.0
    dcrit0: float = 3.55
    dcrit5: float = 1.12


def threshold_rln_rdw_rbm(gamma_ratio: float = 1.0) -> float:
    """Threshold R/Ln where RDW and RBM have equal peak growth at ŝ≈0.

    Mosetto et al. (2012) provide:
      R/Ln = 2(1+g) / [0.085 (1+1.71 g)]^2
    """
    g = float(gamma_ratio)
    return 2.0 * (1.0 + g) / (0.085 * (1.0 + 1.71 * g)) ** 2


def threshold_rln_indw_inbm(gamma_ratio: float = 1.0) -> float:
    """Threshold R/Ln where InDW and InBM have equal peak growth at ŝ≈0.

    Mosetto et al. (2012) provide:
      R/Ln = 2(1+g) / [0.17 (1+1.71 g)]^2
    """
    g = float(gamma_ratio)
    return 2.0 * (1.0 + g) / (0.17 * (1.0 + 1.71 * g)) ** 2


def threshold_d_resistive_inertial(shat: float, cal: MosettoCalibration) -> float:
    """Fit for d-threshold separating RDW and InDW as a function of |ŝ|.

    We use a simple rational fit constrained by:
      d_crit(0)=dcrit0, d_crit(5)=dcrit5.
    """
    s = abs(float(shat))
    if cal.dcrit5 <= 0.0:
        return cal.dcrit0
    slope = max((cal.dcrit0 / cal.dcrit5 - 1.0) / 5.0, 0.0)
    return cal.dcrit0 / (1.0 + slope * s)


def classify_regime(
    *,
    eta: float,
    omega_n: float,
    me_hat: float,
    shat: float,
    calibration: MosettoCalibration | None = None,
) -> tuple[str, dict[str, float]]:
    """Classify (InDW, RDW, InBM, RBM) with calibrated Mosetto-style boundaries."""
    cal = calibration or MosettoCalibration()
    me_eff = max(float(me_hat), 1e-12)
    d = cal.d_scale * float(eta) / me_eff
    dcrit = threshold_d_resistive_inertial(shat, cal)
    is_resistive = d >= dcrit

    rln = cal.rln_per_omega_n * float(omega_n)
    thr_r = threshold_rln_rdw_rbm(cal.gamma_ratio)
    thr_i = threshold_rln_indw_inbm(cal.gamma_ratio)
    rln_thr = thr_r if is_resistive else thr_i
    is_dw = rln >= rln_thr

    if is_dw and is_resistive:
        label = "RDW"
    elif is_dw and (not is_resistive):
        label = "InDW"
    elif (not is_dw) and is_resistive:
        label = "RBM"
    else:
        label = "InBM"

    meta = {
        "d": d,
        "d_crit": dcrit,
        "R_over_Ln_proxy": rln,
        "R_over_Ln_threshold": rln_thr,
        "R_over_Ln_threshold_resistive": thr_r,
        "R_over_Ln_threshold_inertial": thr_i,
    }
    return label, meta


def classify_grid(
    *,
    eta: np.ndarray,
    omega_n: np.ndarray,
    me_hat: float,
    shat: float,
    calibration: MosettoCalibration | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorized regime classification on an (eta, omega_n) grid."""
    cal = calibration or MosettoCalibration()
    labels = np.empty((eta.size, omega_n.size), dtype=int)
    d_arr = np.empty_like(labels, dtype=float)
    rln_arr = np.empty_like(labels, dtype=float)
    rln_thr_arr = np.empty_like(labels, dtype=float)

    name_to_idx = {"InDW": 0, "RDW": 1, "InBM": 2, "RBM": 3}
    for i, eta_i in enumerate(eta):
        for j, wn_j in enumerate(omega_n):
            label, meta = classify_regime(
                eta=float(eta_i),
                omega_n=float(wn_j),
                me_hat=float(me_hat),
                shat=float(shat),
                calibration=cal,
            )
            labels[i, j] = name_to_idx[label]
            d_arr[i, j] = meta["d"]
            rln_arr[i, j] = meta["R_over_Ln_proxy"]
            rln_thr_arr[i, j] = meta["R_over_Ln_threshold"]

    extras = {
        "d_proxy": d_arr,
        "rln_proxy": rln_arr,
        "rln_threshold": rln_thr_arr,
        "d_threshold": np.full_like(d_arr, threshold_d_resistive_inertial(shat, cal), dtype=float),
    }
    return labels, extras
