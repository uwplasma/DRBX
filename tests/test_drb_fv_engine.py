from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from jaxdrb.driver import build_system_from_config, run_simulation
from jaxdrb.io.config import load_config


def _cfg() -> dict:
    return {
        "engine": "drb_fv",
        "geometry": {"kind": "slab", "nx": 16, "ny": 12, "nz": 6, "Lx": 1.0, "Ly": 1.0, "Lz": 2.0},
        "initial": {"n0": 1.0, "Te0": 1.0, "omega0": 0.1},
        "time": {"method": "rk4", "dt": 1e-3, "nsteps": 4, "save_every": 2, "return_numpy": True},
    }


def _literal_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (
        repo_root
        / "examples"
        / "open_field_line"
        / "input_tokamak_bxcv_alignment_strict_early.toml"
    )
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    time_cfg = dict(cfg["time"])
    time_cfg["nsteps"] = 1
    time_cfg["save_every"] = 1
    time_cfg["return_numpy"] = True
    cfg["time"] = time_cfg
    return cfg


def test_build_system_drb_fv_engine() -> None:
    built = build_system_from_config(_cfg())
    assert str(getattr(built.system, "engine", "")) == "drb_fv"
    dy = built.system.rhs(0.0, built.state)
    assert dy.n.shape == built.state.n.shape
    assert dy.Te.shape == built.state.Te.shape


def test_run_simulation_drb_fv_smoke() -> None:
    run = run_simulation(_cfg(), as_numpy=True)
    assert np.asarray(run.times).size >= 2
    assert np.isfinite(np.asarray(run.diagnostics["rms_n"])).all()
    assert np.isfinite(np.asarray(run.diagnostics["rms_Te"])).all()


def test_build_system_hermes_literal_engine() -> None:
    built = build_system_from_config(_literal_cfg())
    assert str(getattr(built.system, "engine", "")) == "hermes_literal"
    dy = built.system.rhs(0.0, built.state)
    assert dy.n.shape == built.state.n.shape
    assert dy.Te.shape == built.state.Te.shape


def test_hermes_literal_rhs_with_phi_smoke() -> None:
    built = build_system_from_config(_literal_cfg())
    dy, phi = built.system.rhs_with_phi(0.0, built.state)
    assert dy.n.shape == built.state.n.shape
    assert dy.Te.shape == built.state.Te.shape
    assert phi.shape == built.state.omega.shape


def test_drb_fv_scheduler_ctx_override() -> None:
    built = build_system_from_config(_cfg())
    phi_override = np.ones_like(np.asarray(built.state.omega))

    class _Ctx:
        phi = phi_override

    split, term_map = built.system.scheduler.run_with_terms(_Ctx(), built.state)
    assert "parallel" in term_map
    total = split.total()
    assert total.n.shape == built.state.n.shape


def test_load_config_engine_alias(tmp_path: Path) -> None:
    cfg_path = tmp_path / "input.toml"
    cfg_path.write_text('engine = "fv_drb"\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.data["engine"] == "drb_fv"


def test_load_config_hermes_literal_alias(tmp_path: Path) -> None:
    cfg_path = tmp_path / "input.toml"
    cfg_path.write_text('engine = "literal_hermes"\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.data["engine"] == "hermes_literal"


def test_load_config_invalid_engine(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text('engine = "invalid_engine"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        _ = load_config(cfg_path)


def test_build_system_drb_fv_from_coeff_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    coeff_path = (
        repo_root / "examples" / "open_field_line" / "axisym_tokamak_bxcv_hermes_norm_parcurv.npz"
    )
    cfg = {
        "engine": "drb_fv",
        "geometry": {
            "kind": "axisymmetric",
            "coeff_path": str(coeff_path),
            "open_field_line": True,
        },
        "initial": {"n0": 1.0, "Te0": 1.0, "omega0": 0.0},
    }
    built = build_system_from_config(cfg)
    assert built.state.n.shape == (48, 32, 81)
    assert built.system.geom.bxcv is not None
    assert built.system.geom.gxx is not None
    assert built.system.geom.gyy is not None
    assert built.system.geom.dpar_factor is not None
    assert np.isfinite(np.asarray(built.system.geom.bxcv)).all()
    assert np.isfinite(np.asarray(built.system.geom.jacobian)).all()
    assert np.isclose(float(built.system.params.dx), 5043.46089151533 / 32.0)
    assert np.isclose(float(built.system.params.dy), 1.2566370614359172 / 81.0)
