from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config


def _load_cfg(path: str) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def test_exb_poloidal_ddy_scheme_switch_changes_rhs_and_stays_finite() -> None:
    cfg = _load_cfg("examples/open_field_line/input_tokamak_bxcv_parity_strict_early.toml")
    cfg["physics"].update({"source_on": False, "omega_n": 0.0, "omega_Te": 0.0})
    cfg["terms"] = {"term_schedule": ["advection"]}
    cfg["time"].update({"jit": False})

    cfg_face = copy.deepcopy(cfg)
    cfg_face["numerics"]["exb_poloidal_ddy_scheme"] = "face"
    built_face = build_system_from_config(cfg_face)
    rhs_face = built_face.system.rhs(0.0, built_face.state)

    cfg_c2 = copy.deepcopy(cfg)
    cfg_c2["numerics"]["exb_poloidal_ddy_scheme"] = "c2"
    built_c2 = build_system_from_config(cfg_c2)
    rhs_c2 = built_c2.system.rhs(0.0, built_c2.state)

    n_face = np.asarray(rhs_face.n)
    n_c2 = np.asarray(rhs_c2.n)

    assert np.isfinite(n_face).all()
    assert np.isfinite(n_c2).all()
    assert float(np.sqrt(np.mean(n_face**2))) > 0.0
    assert float(np.sqrt(np.mean(n_c2**2))) > 0.0
    assert not np.allclose(n_face, n_c2)
