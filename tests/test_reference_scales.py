"""Unit tests for the single-length rho-star reference-scale helper."""

from __future__ import annotations

import math

import pytest

from drbx.runtime.reference_scales import (
    DEUTERIUM_ION_MASS_AMU,
    HYDROGEN_ION_MASS_AMU,
    reference_rho_star,
)

# CODATA 2018, typed independently of the module under test.
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def _hand_formula(te_ev, b0, amu, length):
    mass = amu * AMU
    rho_s = math.sqrt(mass * te_ev * E_CHARGE) / (E_CHARGE * b0)
    c_s = math.sqrt(te_ev * E_CHARGE / mass)
    return rho_s / length, c_s, length / c_s, rho_s


def test_hydrogen_20ev_hsx_values_match_hand_formula():
    scales = reference_rho_star(20.0, 1.02)
    rho_star, c_s, t_ref, rho_s = _hand_formula(20.0, 1.02, 1.00728, 1.0)
    assert scales.rho_star == pytest.approx(rho_star, rel=1e-12)
    assert scales.c_s0 == pytest.approx(c_s, rel=1e-12)
    assert scales.t_ref == pytest.approx(t_ref, rel=1e-12)
    assert scales.rho_s0_m == pytest.approx(rho_s, rel=1e-12)
    # Physical sanity (report values): rho_s ~ 0.45 mm, rho* ~ 4.5e-4,
    # c_s ~ 4.4e4 m/s, t_ref ~ 22.9 us.
    assert scales.rho_s0_m == pytest.approx(0.45e-3, rel=0.02)
    assert scales.rho_star == pytest.approx(4.5e-4, rel=0.02)
    assert scales.c_s0 == pytest.approx(4.4e4, rel=0.01)
    assert scales.t_ref == pytest.approx(22.9e-6, rel=0.01)
    assert scales.ion_mass_amu == HYDROGEN_ION_MASS_AMU == 1.00728
    assert scales.l_ref_m == 1.0


@pytest.mark.parametrize("te_ev", [10.0, 50.0])
def test_other_temperatures_match_hand_formula(te_ev):
    scales = reference_rho_star(te_ev, 1.02)
    rho_star, c_s, t_ref, _ = _hand_formula(te_ev, 1.02, 1.00728, 1.0)
    assert scales.rho_star == pytest.approx(rho_star, rel=1e-12)
    assert scales.c_s0 == pytest.approx(c_s, rel=1e-12)
    assert scales.t_ref == pytest.approx(t_ref, rel=1e-12)


def test_scaling_laws():
    base = reference_rho_star(20.0, 1.0)
    assert reference_rho_star(80.0, 1.0).rho_star == pytest.approx(2.0 * base.rho_star)
    assert reference_rho_star(20.0, 2.0).rho_star == pytest.approx(0.5 * base.rho_star)
    longer = reference_rho_star(20.0, 1.0, l_ref_m=4.0)
    assert longer.rho_star == pytest.approx(0.25 * base.rho_star)
    assert longer.t_ref == pytest.approx(4.0 * base.t_ref)
    assert longer.c_s0 == base.c_s0
    assert longer.rho_s0_m == base.rho_s0_m
    deuterium = reference_rho_star(20.0, 1.0, ion_mass_amu=DEUTERIUM_ION_MASS_AMU)
    ratio = math.sqrt(DEUTERIUM_ION_MASS_AMU / HYDROGEN_ION_MASS_AMU)
    assert deuterium.rho_star == pytest.approx(ratio * base.rho_star)
    assert deuterium.c_s0 == pytest.approx(base.c_s0 / ratio)


def test_rho_star_is_rho_s_over_l_and_t_ref_is_l_over_cs():
    scales = reference_rho_star(33.0, 0.9, ion_mass_amu=2.0, l_ref_m=1.7)
    assert scales.rho_star == pytest.approx(scales.rho_s0_m / scales.l_ref_m)
    assert scales.t_ref == pytest.approx(scales.l_ref_m / scales.c_s0)
    # rho_s0 = c_s0 / Omega_ci
    omega_ci = E_CHARGE * scales.b0_tesla / scales.ion_mass_kg
    assert scales.rho_s0_m == pytest.approx(scales.c_s0 / omega_ci, rel=1e-12)


def test_as_dict_is_json_serializable():
    import json

    payload = reference_rho_star(20.0, 1.02).as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert {"rho_star", "c_s0", "t_ref", "rho_s0_m"} <= set(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"te_ev": 0.0, "b0_tesla": 1.0},
        {"te_ev": -1.0, "b0_tesla": 1.0},
        {"te_ev": 20.0, "b0_tesla": 0.0},
        {"te_ev": 20.0, "b0_tesla": float("nan")},
        {"te_ev": float("inf"), "b0_tesla": 1.0},
        {"te_ev": 20.0, "b0_tesla": 1.0, "ion_mass_amu": 0.0},
        {"te_ev": 20.0, "b0_tesla": 1.0, "l_ref_m": -1.0},
        {"te_ev": "hot", "b0_tesla": 1.0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        reference_rho_star(**kwargs)
