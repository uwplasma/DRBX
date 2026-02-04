"""
essos_biotsavart_r14.py

Purpose
-------
Run `jaxdrb` on a Biot-Savart magnetic field (coils) via ESSOS by:

  1) loading a coils JSON (ESSOS format),
  2) tracing a field line near R0=1.4 (as in essos/examples/trace_fieldlines_coils.py),
  3) converting it into a `TabulatedGeometry` file,
  4) running a ky scan and producing standard diagnostics.

This is a template for "coil-defined stellarator/tokamak" studies where you want to see how linear
stability changes as you modify the coil set or starting location.

Run
---
  python examples/scripts/07_essos_geometries/essos_biotsavart_r14.py

Outputs
-------
Written to `out/examples/07_essos_geometries/essos_biotsavart_r14/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (
    save_geometry_overview,
    save_json,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.essos import biotsavart_fieldline_to_tabulated
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def _default_coils_json() -> Path | None:
    env = os.environ.get("ESSOS_COILS_JSON")
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = (
        ROOT.parent
        / "essos"
        / "examples"
        / "input_files"
        / "ESSOS_biot_savart_LandremanPaulQA.json"
    )
    if p.exists():
        return p
    try:
        import essos  # type: ignore

        pkg = Path(essos.__file__).resolve().parents[1]
        cand = pkg.parent / "examples" / "input_files" / "ESSOS_biot_savart_LandremanPaulQA.json"
        return cand if cand.exists() else None
    except Exception:
        return None


def main() -> None:
    out_dir = Path("out/examples/07_essos_geometries/essos_biotsavart_r14")
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    coils_json = _default_coils_json()
    if coils_json is None:
        raise SystemExit(
            "Could not find an ESSOS coils JSON.\n"
            "Set ESSOS_COILS_JSON=/path/to/coils.json or place ESSOS next to this repo."
        )
    print(f"Using coils JSON: {coils_json}", flush=True)

    R0 = 1.4
    geom_file = out_dir / "geom_biotsavart_r14.npz"
    print("Tracing field line and building tabulated geometry via ESSOS…", flush=True)
    res = biotsavart_fieldline_to_tabulated(
        coils_json=coils_json,
        R0=R0,
        Z0=0.0,
        phi0=0.0,
        nsteps=600 if fast else 1800,
        nout=192 if fast else 512,
        maxtime=900.0,
        out_path=geom_file,
    )
    print(f"Wrote geometry: {res.path}", flush=True)

    geom = TabulatedGeometry.from_npz(geom_file)
    save_geometry_overview(out_dir, geom=geom, kx=0.0, ky=0.3)

    model = get_model("cold-ion-es")
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    ky = np.linspace(0.06, 1.2, 22)
    kx = 0.0
    print("Running ky scan…", flush=True)
    scan = scan_ky(
        params,
        geom,
        ky=ky,
        kx=kx,
        nl=int(geom.l.size),
        model=model,
        arnoldi_m=28,
        arnoldi_tol=3e-3,
        nev=6,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=0,
    )

    save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        title="ESSOS Biot-Savart: ky scan (cold-ion ES)",
        filename="scan_panel.png",
    )

    np.savez(
        out_dir / "results.npz",
        ky=scan.ky,
        gamma_eigs=scan.gamma_eigs,
        omega_eigs=scan.omega_eigs,
        eigs=scan.eigs,
    )
    save_json(
        out_dir / "params.json",
        {
            "model": model.name,
            "coils_json": str(coils_json),
            "R0": R0,
            "geom_file": str(geom_file),
            "meta": res.meta,
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params": params.__dict__,
            "notes": ["Perp metric uses a local orthonormal frame around the traced field line."],
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
