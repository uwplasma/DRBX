from __future__ import annotations

import numpy as np

from jaxdrb.analysis.mosetto_regime import (
    MosettoCalibration,
    classify_regime,
    threshold_d_resistive_inertial,
    threshold_rln_indw_inbm,
    threshold_rln_rdw_rbm,
)


def test_mosetto_reference_thresholds_gamma1_match_reported_values() -> None:
    # Mosetto 2012 transition discussion (g=1, ŝ=0): ~75.2 and ~18.8.
    thr_r = threshold_rln_rdw_rbm(1.0)
    thr_i = threshold_rln_indw_inbm(1.0)
    assert np.isclose(thr_r, 75.2, rtol=4e-3, atol=0.25)
    assert np.isclose(thr_i, 18.8, rtol=4e-3, atol=0.25)


def test_mosetto_d_threshold_fit_hits_anchor_points() -> None:
    cal = MosettoCalibration(dcrit0=3.55, dcrit5=1.12)
    assert np.isclose(threshold_d_resistive_inertial(0.0, cal), 3.55, rtol=0.0, atol=1e-12)
    assert np.isclose(threshold_d_resistive_inertial(5.0, cal), 1.12, rtol=0.0, atol=1e-12)


def test_calibrated_classifier_produces_all_four_regimes() -> None:
    cal = MosettoCalibration(gamma_ratio=1.0, rln_per_omega_n=40.0, d_scale=1.0)
    shat = 0.8
    me_hat = 0.05

    # Inertial + DW => InDW (d=0.6, R/Ln=80)
    label, _ = classify_regime(
        eta=0.03, omega_n=2.0, me_hat=me_hat, shat=shat, calibration=cal
    )
    assert label == "InDW"

    # Resistive + DW => RDW (d=10, R/Ln=80)
    label, _ = classify_regime(
        eta=0.5, omega_n=2.0, me_hat=me_hat, shat=shat, calibration=cal
    )
    assert label == "RDW"

    # Inertial + BM => InBM (d=0.6, R/Ln=8)
    label, _ = classify_regime(
        eta=0.03, omega_n=0.2, me_hat=me_hat, shat=shat, calibration=cal
    )
    assert label == "InBM"

    # Resistive + BM => RBM (d=10, R/Ln=8)
    label, _ = classify_regime(
        eta=0.5, omega_n=0.2, me_hat=me_hat, shat=shat, calibration=cal
    )
    assert label == "RBM"
