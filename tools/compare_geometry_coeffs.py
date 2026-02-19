#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxdrb.driver import build_system_from_config
from jaxdrb.io import load_config
from jaxdrb.core.geometry_axisymmetric import load_axisymmetric_coefficients


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2))) if a.size else 0.0


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(_rms(b), 1e-12)
    return _rms(a - b) / denom


def main() -> None:
    p = argparse.ArgumentParser(description="Compare geometry coefficients against a reference file")
    p.add_argument("--config", required=True, help="jax_drb TOML config")
    p.add_argument("--coeff", required=True, help="Reference coeffs (.npz or .nc)")
    args = p.parse_args()

    cfg = load_config(args.config)
    built = build_system_from_config(cfg.data)
    geom = built.system.geom

    ref = load_axisymmetric_coefficients(Path(args.coeff))

    def grab(name: str) -> np.ndarray:
        val = getattr(geom, name, None)
        if val is None:
            raise ValueError(f"Geometry has no attribute '{name}'.")
        return np.asarray(val)

    report = {}
    for key, ref_key in (
        ("curv_x", "curv_x"),
        ("curv_y", "curv_y"),
        ("dpar_factor", "dpar_factor"),
        ("B", "B"),
    ):
        ref_arr = np.asarray(ref[ref_key]).reshape(-1)
        geom_arr = grab(key).reshape(-1)
        n = min(ref_arr.size, geom_arr.size)
        if n == 0:
            report[key] = {"error": None}
            continue
        report[key] = {
            "rms_ref": _rms(ref_arr[:n]),
            "rms_geom": _rms(geom_arr[:n]),
            "rel_error": _rel_error(geom_arr[:n], ref_arr[:n]),
        }

    for key, stats in report.items():
        print(f"{key}: {stats}")


if __name__ == "__main__":
    main()
