from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config


def _load_cfg(path: str) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _omega_l2(arr) -> float:
    a = np.asarray(arr)
    return float(np.sqrt(np.mean(a * a)))


def test_phi_par_dissipation_switch_controls_extra_dissipation_term() -> None:
    cfg = _load_cfg("examples/field_aligned_3d/input.toml")
    cfg["geometry"].update({"nx": 8, "ny": 8, "nz": 16})
    cfg["physics"].update(
        {"omega_n": 0.0, "omega_Te": 0.0, "curvature_on": False, "source_on": False}
    )
    cfg["transport"].update(
        {
            "Dn": 0.0,
            "DOmega": 0.0,
            "DTe": 0.0,
            "Dn4": 0.0,
            "DOmega4": 0.0,
            "DTe4": 0.0,
            "mu_lin_n": 0.0,
            "mu_lin_omega": 0.0,
            "mu_lin_Te": 0.0,
            "phi_par_dissipation": 0.2,
            "vort_par_dissipation": 0.0,
        }
    )
    cfg["initial"].update({"amplitude": 1e-3, "noise_mode": "state", "noise_fields": ["omega"]})
    cfg["terms"] = {"term_schedule": ["extra_dissipation"]}

    cfg_on = copy.deepcopy(cfg)
    cfg_on["transport"]["phi_dissipation_on"] = True
    built_on = build_system_from_config(cfg_on)
    rhs_on = built_on.system.rhs(0.0, built_on.state)

    cfg_off = copy.deepcopy(cfg)
    cfg_off["transport"]["phi_dissipation_on"] = False
    built_off = build_system_from_config(cfg_off)
    rhs_off = built_off.system.rhs(0.0, built_off.state)

    assert _omega_l2(rhs_on.omega) > 0.0
    assert _omega_l2(rhs_off.omega) < 1e-14


def test_phi_sheath_dissipation_switch_controls_sol_phi_term() -> None:
    cfg = _load_cfg(
        "examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml"
    )
    cfg["physics"].update(
        {
            "source_on": False,
            "omega_n": 0.0,
            "omega_Te": 0.0,
            "sol_on": True,
            "sol_xs": 0.4,
            "sol_width": 0.05,
            "sol_source_n0": 0.0,
            "sol_source_Te0": 0.0,
            "sol_relax_core": 0.0,
            "sol_relax_open": 0.0,
            "sol_sheath_phi_on": True,
            "sol_sheath_phi_implicit": False,
            "sol_parallel_loss_q": 4.0,
            "sol_sheath_phi_model": "linear",
            "sol_sheath_phi_coeff": 1.0,
            "sol_sheath_phi_lambda": 3.0,
        }
    )
    cfg["transport"].update(
        {
            "Dn": 0.0,
            "DOmega": 0.0,
            "DTe": 0.0,
            "Dn4": 0.0,
            "DOmega4": 0.0,
            "DTe4": 0.0,
            "mu_lin_n": 0.0,
            "mu_lin_omega": 0.0,
            "mu_lin_Te": 0.0,
        }
    )
    cfg["initial"].update({"amplitude": 1e-3, "noise_mode": "state", "noise_fields": ["omega"]})
    cfg["terms"] = {"term_schedule": ["sol_sheath_phi"]}

    cfg_on = copy.deepcopy(cfg)
    cfg_on.setdefault("closures", {})["sol_sheath_phi_dissipation_on"] = True
    built_on = build_system_from_config(cfg_on)
    rhs_on = built_on.system.rhs(0.0, built_on.state)

    cfg_off = copy.deepcopy(cfg)
    cfg_off.setdefault("closures", {})["sol_sheath_phi_dissipation_on"] = False
    built_off = build_system_from_config(cfg_off)
    rhs_off = built_off.system.rhs(0.0, built_off.state)

    assert _omega_l2(rhs_on.omega) > 0.0
    assert _omega_l2(rhs_off.omega) < 1e-14


def test_core_vorticity_damping_switch_controls_mu_lin_omega() -> None:
    cfg = _load_cfg("examples/field_aligned_3d/input.toml")
    cfg["geometry"].update({"nx": 8, "ny": 8, "nz": 16})
    cfg["physics"].update(
        {"omega_n": 0.0, "omega_Te": 0.0, "curvature_on": False, "source_on": False}
    )
    cfg["transport"].update(
        {
            "Dn": 0.0,
            "DOmega": 0.0,
            "DTe": 0.0,
            "Dn4": 0.0,
            "DOmega4": 0.0,
            "DTe4": 0.0,
            "mu_lin_n": 0.0,
            "mu_lin_omega": 0.25,
            "mu_lin_Te": 0.0,
        }
    )
    cfg["initial"].update({"amplitude": 1e-3, "noise_mode": "state", "noise_fields": ["omega"]})
    cfg["terms"] = {"term_schedule": ["diffusion"]}

    cfg_on = copy.deepcopy(cfg)
    cfg_on["transport"]["core_vorticity_damping_on"] = True
    built_on = build_system_from_config(cfg_on)
    rhs_on = built_on.system.rhs(0.0, built_on.state)

    cfg_off = copy.deepcopy(cfg)
    cfg_off["transport"]["core_vorticity_damping_on"] = False
    built_off = build_system_from_config(cfg_off)
    rhs_off = built_off.system.rhs(0.0, built_off.state)

    assert _omega_l2(rhs_on.omega) > 0.0
    assert _omega_l2(rhs_off.omega) < 1e-14
