from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config, run_simulation


def _load_open_field_cfg() -> dict:
    path = Path("examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_calibrated.toml")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_initial_profile_shapes_when_only_n_is_perturbed() -> None:
    cfg = _load_open_field_cfg()
    cfg["initial"]["noise_fields"] = ["n"]
    built = build_system_from_config(cfg)
    shape = built.state.n.shape
    assert built.state.Te.shape == shape
    assert built.state.omega.shape == shape
    assert built.state.vpar_e.shape == shape
    assert built.state.vpar_i.shape == shape


def test_short_run_with_n_only_perturbation_does_not_crash() -> None:
    cfg = _load_open_field_cfg()
    cfg["initial"]["noise_fields"] = ["n"]
    cfg["time"]["nsteps"] = 2
    cfg["time"]["save_every"] = 1
    cfg["time"]["save_fields"] = False
    cfg["time"]["return_numpy"] = True
    run_simulation(cfg, as_numpy=True)


def test_mixmode_x_minus_parallel_is_supported() -> None:
    cfg = _load_open_field_cfg()
    cfg["initial"]["n_profile"] = "gaussian_mixmode"
    cfg["initial"]["n_profile_amp"] = 0.0
    cfg["initial"]["mixmode_amp"] = 1.0e-3
    cfg["initial"]["mixmode_terms"] = ["x-z"]
    cfg["initial"]["mixmode_mode"] = "bout"
    cfg["initial"]["amplitude"] = 0.0
    cfg["time"]["nsteps"] = 1
    cfg["time"]["save_every"] = 1
    cfg["time"]["save_fields"] = False
    cfg["time"]["return_numpy"] = True
    run_simulation(cfg, as_numpy=True)


def test_global_mixmode_overlay_on_linear_profile_runs() -> None:
    cfg = _load_open_field_cfg()
    cfg["initial"]["n_profile"] = "linear_x"
    cfg["initial"]["n_mixmode_amp"] = 1.0e-3
    cfg["initial"]["n_mixmode_terms"] = ["x-z"]
    cfg["initial"]["n_mixmode_mode"] = "bout"
    cfg["initial"]["amplitude"] = 0.0
    cfg["time"]["nsteps"] = 1
    cfg["time"]["save_every"] = 1
    cfg["time"]["save_fields"] = False
    cfg["time"]["return_numpy"] = True
    run_simulation(cfg, as_numpy=True)


def test_hermes_exact_ic_uses_pressure_consistent_temperature() -> None:
    path = Path(
        "examples/open_field_line/input_tokamak_bxcv_benchmark_alignment_hermes_exact_ic.toml"
    )
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg["time"]["nsteps"] = 1
    cfg["time"]["save_every"] = 1
    cfg["time"]["save_fields"] = False
    cfg["time"]["return_numpy"] = True
    built = build_system_from_config(cfg)

    n = np.asarray(built.state.n)
    Te = np.asarray(built.state.Te)
    assert n.shape == Te.shape

    nx = n.shape[1]
    x = np.linspace(0.0, 1.0, nx, endpoint=True)[None, :, None]
    p_expected = 3.0 * (1.0 - 0.9 * x) ** 2
    p_actual = n * Te
    rel = np.max(np.abs(p_actual - p_expected)) / max(1e-12, np.max(np.abs(p_expected)))
    assert rel < 1e-10
