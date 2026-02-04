"""
essos_vmec_edge_s09.py

Purpose
-------
Run `jaxdrb` on a VMEC equilibrium via ESSOS by:

  1) loading a VMEC `wout_*.nc` file with `essos.fields.Vmec`,
  2) constructing a *field-line-following* tabulated geometry at fixed flux label `s=0.9`,
  3) running a ky scan and producing the standard diagnostic figures.

This example is meant to be:

- pedagogic (lots of printed progress),
- hackable (easy to change s, field period length, and scan parameters),
- a template for “geometry provider” pipelines based on VMEC.

Where does the wout file come from?
----------------------------------
If you have the ESSOS source checkout (recommended), we use its bundled example file:

  essos/examples/input_files/wout_QH_simple_scaled.nc

Otherwise, set:

  ESSOS_VMEC_WOUT=/path/to/wout.nc

Run
---
  python examples/07_essos_geometries/essos_vmec_edge_s09.py

Outputs
-------
Written to `out/examples/07_essos_geometries/essos_vmec_edge_s09/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (
    save_eigenfunction_panel,
    save_eigenvalue_spectrum,
    save_geometry_overview,
    save_json,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.essos import vmec_fieldline_to_tabulated
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def _default_wout_path() -> Path | None:
    # 1) explicit env var
    env = os.environ.get("ESSOS_VMEC_WOUT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # 2) local sibling checkout (common in this repo layout)
    p = ROOT.parent / "essos" / "examples" / "input_files" / "wout_QH_simple_scaled.nc"
    if p.exists():
        return p
    # 3) best effort: locate essos package and look for examples/
    try:
        import essos  # type: ignore

        pkg = Path(essos.__file__).resolve().parents[1]
        cand = pkg.parent / "examples" / "input_files" / "wout_QH_simple_scaled.nc"
        return cand if cand.exists() else None
    except Exception:
        return None


def main() -> None:
    out_dir = Path("out/examples/07_essos_geometries/essos_vmec_edge_s09")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    wout = _default_wout_path()
    if wout is None:
        raise SystemExit(
            "Could not find a VMEC wout file.\n"
            "Set ESSOS_VMEC_WOUT=/path/to/wout.nc or place ESSOS next to this repo."
        )

    print(f"Using VMEC file: {wout}", flush=True)

    # --- Build tabulated geometry along a field line ---
    s = 0.9
    alpha = 0.0
    nphi = 192
    nfield_periods = 1
    geom_file = out_dir / "geom_vmec_s09.npz"
    print("Building tabulated field-line geometry via ESSOS…", flush=True)
    res = vmec_fieldline_to_tabulated(
        wout_file=wout,
        s=s,
        alpha=alpha,
        nphi=nphi,
        nfield_periods=nfield_periods,
        out_path=geom_file,
    )
    print(f"Wrote geometry: {res.path}", flush=True)
    print(f"VMEC iota(s)≈{res.meta['iota']:.6f}", flush=True)

    geom = TabulatedGeometry.from_npz(geom_file)

    # --- Model / scan parameters ---
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
    kx = 0.0
    ky = np.linspace(0.06, 1.0, 20)

    print("Running ky scan…", flush=True)
    scan = scan_ky(
        params,
        geom,
        ky=ky,
        kx=kx,
        nl=int(geom.l.size),
        model=model,
        arnoldi_m=30,
        arnoldi_tol=2e-3,
        nev=6,
        do_initial_value=True,
        tmax=25.0,
        dt0=0.02,
        nsave=160,
        verbose=True,
        print_every=1,
        seed=0,
    )

    ratio = np.maximum(scan.gamma_eigs, 0.0) / scan.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(scan.ky[i_star])
    lam_star = complex(scan.gamma_eigs[i_star] + 1j * scan.omega_eigs[i_star])
    print(f"ky*={ky_star:.4f}  lambda*={lam_star.real:+.4e}{lam_star.imag:+.4e}i", flush=True)

    save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star)
    save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        title="ESSOS VMEC s=0.9: ky scan (cold-ion ES)",
        filename="scan_panel.png",
    )

    # Eigenfunction + spectrum at ky*
    y_eq = model.equilibrium(int(geom.l.size))
    rhs_kwargs = {"eq": model.default_eq(int(geom.l.size))}
    import jax

    v0 = model.random_state(jax.random.PRNGKey(0), int(geom.l.size), amplitude=1e-3)
    matvec = linear_matvec_from_rhs(
        model.rhs, y_eq, params, geom, kx=kx, ky=ky_star, rhs_kwargs=rhs_kwargs
    )
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=80, seed=0)
    arn = arnoldi_eigs(matvec, v0, m=80, nev=80, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star,
        filename="eigenfunctions.png",
    )
    save_eigenvalue_spectrum(out_dir, eigenvalues=arn.eigenvalues, highlight=ritz.eigenvalue)

    np.savez(
        out_dir / "results.npz",
        ky=scan.ky,
        gamma_eigs=scan.gamma_eigs,
        omega_eigs=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        omega_iv=scan.omega_iv,
        eigs=scan.eigs,
        ky_star=ky_star,
        lam_star=np.asarray(lam_star),
    )
    save_json(
        out_dir / "params.json",
        {
            "model": model.name,
            "wout_file": str(wout),
            "s": s,
            "alpha": alpha,
            "geom_file": str(geom_file),
            "meta": res.meta,
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params": params.__dict__,
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
