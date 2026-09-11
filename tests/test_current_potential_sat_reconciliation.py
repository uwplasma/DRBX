"""Recount the persisted SAT reconciliation; these are captured-map audits.

Live current-closure regressions are in the boundary payload isolation tests.
The matrices here intentionally document a remaining production mismatch.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


WORK = Path(__file__).resolve().parents[1] / "work/boundary_load_audit"


def _capture():
    report = json.loads((WORK / "current_potential_sat_reconciliation.json").read_text())
    path = WORK / "current_potential_sat_reconciliation.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["array_sha256"]
    assert report["status"] == "completed"
    assert report["work_only"] and not report["production_operator_replaced"]
    with np.load(path, allow_pickle=False) as saved:
        return report, {name: saved[name] for name in saved.files}


def test_existing_trace_lifts_cannot_explain_complete_live_gradient_difference():
    report, raw = _capture()
    trace = np.vstack([raw[name] for name in ("T_coordinate", "T_endpoint", "T_dual")])
    delta = raw["G_live"] - raw["G0"]
    probe = raw["complete_zero_trace_probe"]
    np.testing.assert_allclose(np.linalg.norm(probe), 1.0, atol=2e-14)
    np.testing.assert_allclose(trace @ probe, 0.0, atol=2e-14)
    # Any lift using only these traces vanishes on this probe, while the
    # actual gradient difference remains finite.
    mismatch = np.linalg.norm(delta @ probe)
    assert mismatch > 0.2
    np.testing.assert_allclose(
        mismatch, report["complete_zero_trace_probe"]["gradient_mismatch_l2"], atol=2e-14
    )
    remainder = delta - delta @ np.linalg.pinv(trace, rcond=1e-11) @ trace
    np.testing.assert_allclose(
        np.linalg.norm(remainder) / np.linalg.norm(delta),
        report["trace_factorizations"]["combined"]["relative_best_lift_remainder"],
        atol=2e-14,
    )
    # The old coordinate-zero control did not zero the SAT-dual trace.
    previous = raw["previous_phi_probe"]
    np.testing.assert_allclose(raw["T_coordinate"] @ previous, 0.0, atol=2e-14)
    np.testing.assert_allclose(raw["T_endpoint"] @ previous, 0.0, atol=2e-14)
    assert np.max(np.abs(raw["T_dual"] @ previous)) > 1.0


def test_candidate_pairs_volume_work_but_retained_lift_has_a_different_trace():
    report, raw = _capture()
    mass, gradient, divergence = raw["mass"], raw["G_live"], raw["D_candidate"]
    np.testing.assert_allclose(
        mass[:, None] * divergence + gradient.T * mass[None, :], 0.0, atol=2e-14
    )
    np.testing.assert_allclose(gradient @ np.ones(len(mass)), 0.0, atol=2e-14)
    np.testing.assert_allclose(mass @ divergence, 0.0, atol=2e-14)
    # Retaining the canonical endpoint injection determines a particular
    # dual trace. It is not automatically the plasma trace used by the sheath.
    lift, trace, weight = raw["current_lift"], raw["T_dual"], raw["endpoint_weights"]
    np.testing.assert_allclose(
        mass[:, None] * lift, trace.T * weight[None, :], atol=2e-14
    )
    assert np.linalg.norm(trace - raw["T_endpoint"]) > 1.0
    rng = np.random.default_rng(119)
    phi, current = rng.normal(size=(2, len(mass)))
    endpoint_current = rng.normal(size=lift.shape[1])
    volume_work = phi @ (mass * (divergence @ current + lift @ endpoint_current))
    volume_work += current @ (mass * (gradient @ phi))
    trace_work = (trace @ phi) @ (weight * endpoint_current)
    np.testing.assert_allclose(volume_work, trace_work, atol=2e-13)
    assert not report["candidate"]["independent_physical_wall_power_verified"]
    assert report["current_lift"]["endpoint_weights_are_induced_not_verified_physical_area"]
